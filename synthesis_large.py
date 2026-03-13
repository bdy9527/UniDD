import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from utils import get_dataset, denormalize, save_images, clip
from hook import ConvFeatureHook
import copy
import random
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader
from tiny_imagenet_dataset import TinyImageNet
from torchvision.datasets import ImageFolder
from datasets import load_from_disk
from transformers import get_cosine_schedule_with_warmup
# from baseline import get_network as ti_get_network
from rded_models import ConvNet


def validate_args(args):
    supported = {
        'Tiny': {'ResNet18', 'ConvNetW128D4'},
        'ImageNet': {'ResNet18'},
    }

    if args.dataset not in supported:
        print(
            f"[Error] Unsupported dataset '{args.dataset}' for synthesis_large.py. "
            f"Supported datasets: {', '.join(sorted(supported))}."
        )
        return False

    if args.model not in supported[args.dataset]:
        print(
            f"[Error] Unsupported model '{args.model}' for dataset '{args.dataset}' in synthesis_large.py. "
            f"Supported models for this dataset: {', '.join(sorted(supported[args.dataset]))}."
        )
        return False

    return True


def main(args):
    if not validate_args(args):
        return

    args.device = 'cuda:{}'.format(args.cuda)

    channel, im_size, num_classes, dst_train, dst_test = get_dataset(args.dataset, args.data_path)

    if args.dataset == 'Tiny':
        iteration = 1000
        lr_img = 0.1

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.4802, 0.4481, 0.3975],
                                std = [0.2302, 0.2265, 0.2262])
        ])

    elif args.dataset == 'ImageNet':
        iteration = 1000
        lr_img = 0.1

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        rded_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    dic = {}
    for i, c in enumerate(dst_train.targets):
        if c in dic.keys():
            dic[c].append(i)
        else:
            dic[c] = [i]

    if args.ipc == 1:
        rded_ds = load_from_disk('hf_dataset/rded-ipc-10')

        rded_dic = {}
        for i, c in enumerate(rded_ds['train']['class_id']):
            if c in rded_dic.keys():
                rded_dic[c].append(i)
            else:
                rded_dic[c] = [i]

        def get_rded_ipc(ipc_id, class_idx):
            images = []
            for c in class_idx:
                image_PIL = rded_ds['train'][rded_dic[c][ipc_id]]['image']
                images.append(rded_transforms(image_PIL))
            return np.stack(images, axis=0)

    else:
    
        def get_one_ipc(ipc_id, class_idx):
            images = []
            for c in class_idx:
                #image_path = dst_train.data[random.choice(dic[c])][0]
                if args.dataset == 'Tiny':
                    image_path = dst_train.data[dic[c][ipc_id]][0]
                else:
                    image_path = dst_train.imgs[dic[c][ipc_id]][0]
                image = default_loader(image_path)
                images.append(val_transform(image))
            return np.stack(images, axis=0)


    model_dir = os.path.join(args.model_path, args.dataset)
    syn_dir = os.path.join(args.syn_path, args.dataset)
    syn_data_path = os.path.join(syn_dir, 'conv_{}_{}_{}_{}_{}'.format(args.model, args.ipc, args.filter, args.scheduler, args.beta))

    if args.scheduler == 'cos':
        beta_scheduler = [args.beta * (1. + np.cos(np.pi * ipc / args.ipc)) / 2 for ipc in range(args.ipc)]
    elif args.scheduler == 'linear':
        beta_scheduler = [args.beta * (1. - ipc / args.ipc) for ipc in range(args.ipc)]
    else:
        beta_scheduler = [args.beta for ipc in range(args.ipc)]

    if not os.path.exists(syn_data_path):
        os.makedirs(syn_data_path, exist_ok=True)
    model_name = args.model + '.pth'
    print(model_name)
    ckpt_path = os.path.join(model_dir, model_name)

    if args.dataset == 'Tiny':

        if args.model == 'ResNet18':
            model = torchvision.models.get_model('resnet18', weights=None, num_classes=num_classes)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()
        elif args.model == 'ConvNetW128D4':
            model = ConvNet(channel=3, num_classes=200, net_width=128, net_depth=4, net_norm="batch", net_act="relu", net_pooling="avgpooling", im_size=(64, 64))
            # model = ti_get_network('ConvNetW128', channel=3, num_classes=200, im_size=(64, 64), dist=False)

    elif args.dataset == 'ImageNet':
        model = torchvision.models.get_model('resnet18', weights=None, num_classes=num_classes)
    else:
        return

    model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
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

        if isinstance(module, nn.Conv2d) and 'shortcut' not in name and 'downsample' not in name:

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
        if args.dataset == 'Tiny':
            train_dataset = TinyImageNet(args.data_path, split='train', download=False, transform=val_transform)
        elif args.dataset == 'ImageNet':
            train_dataset = ImageFolder(root=os.path.join(args.data_path, 'ImageNet', 'train'), transform=val_transform)
        else:
            return

        print('Pre Hook')
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

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

        targets_all = torch.LongTensor(np.arange(num_classes))

        for _loss_t_feature_layer in loss_r_feature_layers:
            _loss_t_feature_layer.beta = beta_scheduler[ipc_id]

        for kk_index, kk in enumerate(range(0, num_classes, args.batch)):
            batch_id += 1.

            for _loss_t_feature_layer in loss_r_feature_layers:
                _loss_t_feature_layer.batch_id = batch_id

            idx = [i for i in range(kk, min(kk+args.batch, num_classes))]

            # syn_img = torch.randn((len(idx), 3, im_size[0], im_size[0]), requires_grad=True, device=args.device, dtype=torch.float)
            if args.ipc > 1:
                syn_img = torch.tensor(get_one_ipc(ipc_id, idx), requires_grad=True, device=args.device, dtype=torch.float)
            else:
                syn_img = torch.tensor(get_rded_ipc(ipc_id, idx), requires_grad=True, device=args.device, dtype=torch.float)
            syn_lab = targets_all[idx].to(args.device)

            img_optimizer = torch.optim.Adam([syn_img], lr=lr_img, betas=(0.5, 0.9), eps=1e-8)
            img_scheduler = get_cosine_schedule_with_warmup(img_optimizer, num_warmup_steps=0, num_training_steps=iteration)
            criterion = nn.CrossEntropyLoss().to(args.device)

            # time_record = []
            loss_avg = 0.
            for it in range(iteration):

                t1 = time.time()
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

                t2 = time.time()

                # time_record.append(t2 - t1)
                # print(np.mean(time_record))

            if iteration != 0:
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

    parser.add_argument('--dataset', type=str, default='Tiny', help='dataset')
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
    parser.add_argument('--scheduler', default='none', help='scheduler')
    # parser.add_argument('--cos', action='store_true', help='scheduler')
    # parser.add_argument('--linear', action='store_true', help='scheduler')
    parser.add_argument('--beta', type=float, default=0.1, help='inverse matrix')

    args = parser.parse_args()

    # python synthesis_large.py --dataset Tiny --model ConvNetW128D4 --ipc 50 --filter HFM --ema --cos --beta 1.0
    # python synthesis_large.py --dataset ImageNet --model ResNet18 --ipc 10 --filter HFM --ema --scheduler cos --beta 0.2

    main(args)
