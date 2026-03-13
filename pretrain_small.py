import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from utils import get_dataset, get_network
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
from transformers import get_cosine_schedule_with_warmup
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

    if args.model == 'ResNet18':
        model = torchvision.models.get_model('resnet18', weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        ckpt = os.path.join('ckpt', args.dataset, 'ResNet18.pth')

    elif args.model == 'ConvNetW128':
        model = ConvNet(channel=3, num_classes=num_classes, net_width=128, net_depth=3, net_norm="batch", net_act="relu", net_pooling="avgpooling", im_size=(32, 32))
        ckpt = os.path.join('ckpt', args.dataset, '{}.pth'.format(args.model))

    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
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

    print(ckpt, np.sum(np.equal(pred, label)) / len(dst_test))



def main(args):

    args.device = 'cuda:{}'.format(args.cuda)

    channel, im_size, num_classes, dst_train, dst_test = get_dataset(args.dataset, args.data_path)

    print(channel, im_size, num_classes)
    print(len(dst_train), len(dst_test))

    save_dir = os.path.join(args.buffer_path, args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    criterion = nn.CrossEntropyLoss().to(args.device)

    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=128, shuffle=True,
                                            num_workers=4, pin_memory=True)

    if args.model == 'ResNet18':
        model = torchvision.models.get_model('resnet18', weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
    elif args.model == 'ConvNetW128':
        model = get_network('ConvNetW128', channel=channel, num_classes=num_classes, im_size=im_size, dist=False)

    model = model.to(args.device)
    model.train()

    if args.dataset == 'CIFAR-10':
        args.train_epochs = 200
    elif args.dataset == 'CIFAR-100':
        args.train_epochs = 100
    else:
        return

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.train_epochs)
    
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
    parser.add_argument('--dataset', type=str, default='CIFAR-10', help='dataset')
    parser.add_argument('--model', type=str, default='ResNet18', help='model')
    parser.add_argument('--cuda', type=int, default=0, help='GPU id')

    parser.add_argument('--expert', type=int, default=1, help='number of experts')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='ckpt', help='buffer path')

    args = parser.parse_args()
    main(args)
    # evaluate(args)
