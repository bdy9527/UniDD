import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import get_dataset, get_network, get_daparam, TensorDataset, epoch, ParamDiffAug
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, SequentialLR, LinearLR
import torchvision
from transformers import get_cosine_schedule_with_warmup
# import models as ti_models
from baseline import get_network as ti_get_network
from rded_models import ConvNet


from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def evaluate(args):

    args.device = 'cuda:{}'.format(args.cuda)

    channel, im_size, num_classes, dst_train, dst_test = get_dataset(args.dataset, args.data_path)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=16)

    save_dir = os.path.join(args.buffer_path, args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

    ckpt_path = os.path.join(save_dir, args.model + '.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    for p in model.parameters():
        p.requires_grad = False

    model = model.to(args.device)
    model.eval()

    pred = []
    label = []
    for i_batch, datum in enumerate(testloader):
        img = datum[0].float().to(args.device)
        lab = datum[1].to(args.device)

        output = model(img)
        pred += list(np.argmax(output.cpu().data.numpy(), axis=-1))
        label += list(lab.cpu().data.numpy())

    print(ckpt_path, np.sum(np.equal(pred, label)) / len(dst_test))



def main(args):

    args.device = 'cuda:{}'.format(args.cuda)

    channel, im_size, num_classes, dst_train, dst_test = get_dataset(args.dataset, args.data_path)

    save_dir = os.path.join(args.buffer_path, args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    criterion = nn.CrossEntropyLoss().to(args.device)

    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=128, shuffle=True,
                                            num_workers=16, pin_memory=True)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False)


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
    
    model = model.to(args.device)
    model.train()

    if args.dataset == 'Tiny':
        args.train_epochs = 100
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=1e-4)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=args.train_epochs)

    elif args.dataset == 'ImageNet':
        args.train_epochs = 90
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    else:
        return


    for e in range(args.train_epochs):
        loss_avg, acc_avg, num_exp = 0, 0, 0

        t1 = time.time()

        model.train()
        for i_batch, datum in enumerate(trainloader):

            img = datum[0].float().to(args.device)
            lab = datum[1].to(args.device)

            n_b = lab.shape[0]

            output = model(img)
            loss = criterion(output, lab)

            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

            loss_avg += loss.item()*n_b
            acc_avg += acc
            num_exp += n_b

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        loss_avg /= num_exp
        acc_avg /= num_exp

        t2 = time.time()

        print(e, loss_avg, acc_avg, t2 - t1)

        # if (e + 1) % 30 == 0:
        #     model.eval()

        #     pred = []
        #     label = []
        #     for i_batch, datum in enumerate(testloader):
        #         img = datum[0].float().to(args.device)
        #         lab = datum[1].to(args.device)

        #         output = model(img)
        #         pred += list(np.argmax(output.cpu().data.numpy(), axis=-1))
        #         label += list(lab.cpu().data.numpy())

        #     print(np.sum(np.equal(pred, label)) / len(dst_test))

        torch.save(model.state_dict(), os.path.join(save_dir, '{}.pth'.format(args.model)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='Tiny', help='dataset')
    parser.add_argument('--model', type=str, default='ResNet18', help='model')
    parser.add_argument('--cuda', type=int, default=0, help='GPU id')

    parser.add_argument('--expert', type=int, default=1, help='number of experts')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='ckpt', help='buffer path')

    args = parser.parse_args()
    main(args)
    # evaluate(args)
