import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils import get_dataset, denormalize, save_images, clip
from hook import ConvFeatureHook
import copy
import random
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from rded_models import ConvNet


def validate_args(args):
    supported = {
        'CIFAR-10': {'ResNet18', 'ConvNetW128D3'},
        'CIFAR-100': {'ResNet18', 'ConvNetW128D3'},
    }

    if args.dataset not in supported:
        print(
            f"[Error] Unsupported dataset '{args.dataset}' for synthesis_small.py. "
            f"Supported datasets: {', '.join(sorted(supported))}."
        )
        return False

    if args.model not in supported[args.dataset]:
        print(
            f"[Error] Unsupported model '{args.model}' for dataset '{args.dataset}' in synthesis_small.py. "
            f"Supported models for this dataset: {', '.join(sorted(supported[args.dataset]))}."
        )
        return False

    return True


def main(args):
    if not validate_args(args):
        return

    args.device = 'cuda:{}'.format(args.cuda)

    channel, im_size, num_classes, dst_train, dst_test = get_dataset(args.dataset, args.data_path)

    if args.dataset == 'CIFAR-10':
        iteration = 1000
        lr_img = 0.25

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465],
                                [0.2023, 0.1994, 0.2010])
        ])

    elif args.dataset == 'CIFAR-100':
        iteration = 1000
        lr_img = 0.25

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                [0.2675, 0.2565, 0.2761])
        ])

    dic = {}
    for i, c in enumerate(dst_train.targets):
        if c in dic.keys():
            dic[c].append(i)
        else:
            dic[c] = [i]

    def get_one_ipc(ipc_id):
        images = []
        for c in range(num_classes):
            image = dst_train.data[dic[c][ipc_id]]
            images.append(val_transform(image))
        return np.stack(images, axis=0)

    model_dir = os.path.join(args.model_path, args.dataset)
    syn_dir = os.path.join(args.syn_path, args.dataset)
    syn_data_path = os.path.join(syn_dir, 'conv_{}_{}_{}_{}_{}'.format(args.model, args.ipc, args.filter, args.cos, args.beta))

    if args.cos:
        beta_scheduler = [args.beta * (1. + np.cos(np.pi * ipc / args.ipc)) / 2 for ipc in range(args.ipc)]
    else:
        beta_scheduler = [args.beta for ipc in range(args.ipc)]

    if not os.path.exists(syn_data_path):
        os.makedirs(syn_data_path, exist_ok=True)

    model_name = args.model + '.pth'
    print(model_name)
    ckpt_path = os.path.join(model_dir, model_name)

    if args.model == 'ResNet18':
        model_name = 'ResNet18.pth'
        ckpt_path = os.path.join(model_dir, model_name)

        model = torchvision.models.get_model('resnet18', weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()

    elif args.model == 'ConvNetW128D3':
        model = ConvNet(channel=3, num_classes=num_classes, net_width=128, net_depth=3, net_norm="batch", net_act="relu", net_pooling="avgpooling", im_size=(32, 32))
        ckpt_path = os.path.join(model_dir, model_name)

    else:
        return

    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location='cpu'))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    label_model = copy.deepcopy(model)

    model = model.to(args.device)
    label_model = label_model.to(args.device)

    load_tag = True
    rescale = []
    loss_r_feature_layers = []
    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d) and 'downsample' not in name:

            _hook_module = ConvFeatureHook(module, name=model_name + "=" + name,
                                           save_path=args.statistic_path,
                                           dataset=args.dataset,
                                           filter=args.filter,
                                           signal=args.signal,
                                           ema=args.ema,
                                           feat='input',
                                           device=args.device)
            _hook_module.set_hook(pre=True)
            load_tag = load_tag & _hook_module.load_tag
            loss_r_feature_layers.append(_hook_module)
            rescale.append(1.)

    if not load_tag:
        if args.dataset == 'CIFAR-10':
            train_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=val_transform)
        elif args.dataset == 'CIFAR-100':
            train_dataset = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=val_transform)
        else:
            return

        print('Pre Hook')
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=16, pin_memory=True)

        with torch.no_grad():
            for i, (data, labels) in tqdm(enumerate(train_loader)):
                data, labels = data.to(args.device), labels.to(args.device)

                onehot_labels = F.one_hot(labels, num_classes=num_classes).float()
                pseudo_labels = torch.softmax(label_model(data), dim=-1).detach()

                for _loss_t_feature_layer in loss_r_feature_layers:
                    _loss_t_feature_layer.onehot_labels = onehot_labels
                    _loss_t_feature_layer.pseudo_labels = pseudo_labels

                _ = model(data)

            for _loss_t_feature_layer in loss_r_feature_layers:
                _loss_t_feature_layer.save()

        print("Training Statistic Information Is Successfully Saved")
    else:
        print("Training Statistic Information Is Successfully Load")

    print('Post Hook')
    for _loss_t_feature_layer in loss_r_feature_layers:
        _loss_t_feature_layer.set_hook(pre=False)


    '''Image Synthesis'''

    batch_id = 0.
    for ipc_id in range(args.ipc):

        batch_id += 1.
        for _loss_t_feature_layer in loss_r_feature_layers:
            _loss_t_feature_layer.beta = beta_scheduler[ipc_id]
            _loss_t_feature_layer.batch_id = batch_id

        # syn_img = torch.randn((len(idx), 3, im_size[0], im_size[0]), requires_grad=True, device=args.device, dtype=torch.float)
        syn_img = torch.tensor(get_one_ipc(ipc_id), requires_grad=True, device=args.device).float()
        syn_lab = torch.tensor(np.arange(num_classes), requires_grad=False, device=args.device).long()

        img_optimizer = torch.optim.Adam([syn_img], lr=lr_img, betas=(0.5, 0.9), eps=1e-8)
        img_scheduler = get_cosine_schedule_with_warmup(img_optimizer, num_warmup_steps=0, num_training_steps=iteration)
        criterion = nn.CrossEntropyLoss().to(args.device)

        loss_avg = 0.
        for it in range(iteration):

            loss = 0.

            aug_function = transforms.Compose(
                [
                    transforms.RandomResizedCrop(im_size[0], antialias=True),
                    transforms.RandomHorizontalFlip(),
                ]
            )
            jit_img = aug_function(syn_img)

            off1 = random.randint(0, args.jitter)
            off2 = random.randint(0, args.jitter)
            jit_img = torch.roll(jit_img, shifts=(off1, off2), dims=(2, 3))

            if args.signal == 'class':
                onehot_labels = F.one_hot(syn_lab, num_classes=num_classes).float().detach()
                for _loss_t_feature_layer in loss_r_feature_layers:
                    _loss_t_feature_layer.onehot_labels = onehot_labels

            if args.signal == 'mix':
                pseudo_labels = torch.softmax(label_model(jit_img), dim=-1).detach()
                for _loss_t_feature_layer in loss_r_feature_layers:
                    _loss_t_feature_layer.pseudo_labels = pseudo_labels

            pred = model(jit_img)
            loss_ce = criterion(pred, syn_lab)

            loss_r_feature = sum(
                [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

            loss = loss_ce + 0.1 * loss_r_feature

            img_optimizer.zero_grad()
            loss.backward()
            img_optimizer.step()
            img_scheduler.step()

            loss_avg += loss.item()

            syn_img.data = clip(syn_img.data, args.dataset)

        loss_avg /= iteration
        print(ipc_id, batch_id, loss_avg, beta_scheduler[ipc_id])

        #### Save Synthetic Images #####
        syn_img = syn_img.detach().cpu()
        syn_img = denormalize(syn_img, args.dataset)
        save_images(syn_data_path, syn_img, syn_lab, ipc_id)

        if args.ema:
            for _loss_t_feature_layer in loss_r_feature_layers:
                _loss_t_feature_layer.ema_update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR-10', help='dataset')
    parser.add_argument('--model', type=str, default='ResNet18', help='model')
    parser.add_argument('--cuda', type=int, default=0, help='GPU id')

    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--num_eval', type=int, default=1, help='how many networks to evaluate on')
    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')
    parser.add_argument('--batch', type=int, default=200, help='how often to evaluate')
    parser.add_argument('--jitter', type=int, default=4, help='random shift on the synthetic data')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--model_path', type=str, default='ckpt', help='buffer path')
    parser.add_argument('--statistic_path', type=str, default='statistic', help='buffer path')
    parser.add_argument('--syn_path', type=str, default='syn_data', help='buffer path')

    parser.add_argument('--feat', type=str, default='input', help='input or output')
    parser.add_argument('--filter', type=str, default='HFM', help='HFM or LFM')
    parser.add_argument('--signal', type=str, default='mean', help='mean or class')
    parser.add_argument('--ema', action='store_true', help='input or output')
    parser.add_argument('--cos', action='store_true', help='scheduler')
    parser.add_argument('--beta', type=float, default=0.1, help='inverse matrix')

    args = parser.parse_args()

    main(args)


# python synthesis_small.py --dataset CIFAR-10 --model ResNet18 --ipc 10 --filter HFM --ema --cos --beta 0.1
