import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

import argparse
import os
import copy
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from utils import ImageFolderIPC, get_dataset
from torch.optim.lr_scheduler import LambdaLR
# import models as ti_models
# from baseline import get_network as ti_get_network
from rded_models import ConvNet


parser = argparse.ArgumentParser(description="PyTorch Post-Training")

parser.add_argument('--dataset', type=str, default='Tiny', help='dataset')
parser.add_argument('--model', type=str, default='ResNet18', help='model')
parser.add_argument('--val_model', type=str, default='ResNet18', help='model')
parser.add_argument('--cuda', type=int, default=0, help='GPU id')
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--filter', type=str, default='low', help='low, high or poly')
parser.add_argument('--feat', type=str, default='avg', help='hw or avg')
parser.add_argument('--mix_type', type=str, default='cutmix', help='mixup or cutmix')
parser.add_argument("--mixup", type=float, default=0.8, help="mixup alpha, mixup enabled if > 0. (default: 0.8)")
parser.add_argument("--cutmix", type=float, default=1.0, help="cutmix alpha, cutmix enabled if > 0. (default: 1.0)")

parser.add_argument("--output_dir", default="./save", type=str)
parser.add_argument("--syn_data_path", default="syn_data", type=str)
parser.add_argument("--syn_folder", type=str, default="")
parser.add_argument("--teacher_path", default="./ckpt", type=str)
parser.add_argument("--ipc", default=50, type=int)
parser.add_argument("--epochs", default=100, type=int)
args = parser.parse_args()

# python generate_soft_label_online.py --dataset Tiny --model ConvNetW128D4 --val_model ConvNetW128D4 --ipc 50 --syn_folder conv_ConvNetW128D4_50_HFM_True_0.5
# python generate_soft_label_online.py --dataset Tiny --model ResNet18 --val_model ResNet18 --ipc 50 --syn_folder conv_ResNet18_50_HFM_True_0.1
# python generate_soft_label_online.py --dataset ImageNet --model ResNet18 --val_model ResNet18 --ipc 10 --syn_folder conv_ResNet18_10_HFM_True_0.1

args.device = 'cuda:{}'.format(args.cuda)
args.output_dir = os.path.join(args.output_dir, args.dataset)
#args.syn_data_path = os.path.join(args.syn_data_path, args.dataset, 'syn_img_{}_{}'.format(args.filter, args.feat))
args.syn_data_path = os.path.join(args.syn_data_path, args.dataset)
if args.syn_folder:
    args.syn_data_path = os.path.join(args.syn_data_path, args.syn_folder)
args.teacher_path = os.path.join(args.teacher_path, args.dataset)

print(args.syn_data_path)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

channel, im_size, num_classes, dst_train, dst_test = get_dataset(args.dataset)

if args.dataset == 'Tiny':
    args.epochs = 300
    args.batch_size = 100

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(im_size[0], scale=(0.5, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                std=[0.2302, 0.2265, 0.2262])
        ]
    )

elif args.dataset == 'ImageNet':
    args.epochs = 300

    if args.ipc == 1:
        args.batch_size = 20
    else:
        args.batch_size = 100

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(im_size[0], scale=(0.5, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ]
    )

elif args.dataset == 'CIFAR-100':
    args.epochs = 1000
    args.batch_size = 64

    if args.ipc == 1:
        args.batch_size = 20
    elif args.ipc == 10:
        args.batch_size = 100
    elif args.ipc == 50:
        args.batch_size = 200
    else:
        args.batch_size = 100


    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(im_size[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                std=[0.2675, 0.2565, 0.2761])
        ]
    )
elif args.dataset == 'CIFAR-10':
    args.epochs = 1000
    args.batch_size = 64

    if args.ipc == 1:
        args.batch_size = 10
    elif args.ipc == 10:
        args.batch_size = 50
    elif args.ipc == 50:
        args.batch_size = 100
    else:
        args.batch_size = 100

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(im_size[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                std=[0.2023, 0.1994, 0.2010])
        ]
    )


print(len(dst_test))
print("=> Using IPC setting of ", args.ipc)
trainset = ImageFolderIPC(root=args.syn_data_path, transform=transform_train, ipc=args.ipc)
print(len(trainset))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=16, pin_memory=True)

# Model
print("==> Building model..")

if args.dataset == 'ImageNet':

    model_teacher = torchvision.models.__dict__['resnet18'](pretrained=True)
    for p in model_teacher.parameters():
        p.requires_grad = False

    print(args.val_model)
    net = torchvision.models.get_model(args.val_model, weights=None, num_classes=num_classes)

    model_teacher = model_teacher.to(args.device)
    model_teacher.eval()

    net = net.to(args.device)
    net.train()

elif args.dataset == 'Tiny':

    if args.model == 'ResNet18':
        model_teacher = torchvision.models.get_model('resnet18', weights=None, num_classes=num_classes)
        model_teacher.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model_teacher.maxpool = nn.Identity()
    elif args.model == 'ConvNetW128D4':
        model_teacher = ConvNet(channel=3, num_classes=200, net_width=128, net_depth=4, net_norm="batch", net_act="relu", net_pooling="avgpooling", im_size=(64, 64))

    ckpt_path = os.path.join(args.teacher_path, args.model + '.pth')
    model_teacher.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    for p in model_teacher.parameters():
        p.requires_grad = False

    print(args.val_model)
    if args.val_model == 'ConvNetW128D4':
        net = ConvNet(channel=3, num_classes=200, net_width=128, net_depth=4, net_norm="batch", net_act="relu", net_pooling="avgpooling", im_size=(64, 64))
    else:
        net = torchvision.models.get_model(args.val_model, weights=None, num_classes=num_classes)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        net.maxpool = nn.Identity()

    model_teacher = model_teacher.to(args.device)
    model_teacher.eval()

    net = net.to(args.device)
    net.train()


else: # CIFAR
    
    if args.model == 'ConvNetW128D3':
        model_teacher = ConvNet(channel=3, num_classes=num_classes, net_width=128, net_depth=3, net_norm="batch", net_act="relu", net_pooling="avgpooling", im_size=(32, 32))
    
        ckpt_path = os.path.join(args.teacher_path, 'ConvNetW128D3.pth')
        model_teacher.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location='cpu'))
        model_teacher.eval()
        for p in model_teacher.parameters():
            p.requires_grad = False

        net = ConvNet(channel=3, num_classes=num_classes, net_width=128, net_depth=3, net_norm="batch", net_act="relu", net_pooling="avgpooling", im_size=(32, 32))
    
    elif args.model == 'ResNet18':
        model_teacher = torchvision.models.get_model('resnet18', weights=None, num_classes=num_classes)
        model_teacher.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model_teacher.maxpool = nn.Identity()

        ckpt_path = os.path.join(args.teacher_path, 'ResNet18.pth')
        model_teacher.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        for p in model_teacher.parameters():
            p.requires_grad = False

        net = torchvision.models.get_model('resnet18', weights=None, num_classes=num_classes)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        net.maxpool = nn.Identity()

    model_teacher = model_teacher.to(args.device)
    model_teacher.eval()

    net = net.to(args.device)
    net.train()


'''
if args.dataset == 'Tiny':
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, betas=[0.9, 0.999], weight_decay=1e-2)
elif args.dataset == 'ImageNet':
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    # 5e-4 / 2e-4 for swin_t and deit_t
elif args.dataset == 'CIFAR-100':
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
elif args.dataset == 'CIFAR-10':
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=5e-4)
'''

optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, betas=[0.9, 0.999], weight_decay=1e-2)
scheduler = LambdaLR(
    optimizer,
    lambda step: 0.5 * (1.0 + math.cos(math.pi * step / args.epochs / 2))
    if step <= args.epochs
    else 0,
    last_epoch=-1,
)


criterion = nn.CrossEntropyLoss()
loss_function_kl = nn.KLDivLoss(reduction="batchmean")

args.temperature = 20

# if 'CIFAR' in args.dataset:
#     args.temperature = 30
# else:
#     args.temperature = 20


# def mixup_data(x, y, alpha=0.8):
#     """
#     Returns mixed inputs, mixed targets, and mixing coefficients.
#     For normal learning
#     """
#     lam = np.random.beta(alpha, alpha)
#     batch_size = x.size()[0]
#     index = torch.randperm(batch_size).to(x.device)
#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]
#     return mixed_x, y_a, y_b, lam

def cutmix(images, args, rand_index=None, lam=None, bbox=None):
    rand_index = torch.randperm(images.size()[0]).to(args.device)
    lam = np.random.beta(args.cutmix, args.cutmix)
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)

    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    return images, rand_index.cpu(), lam, [bbx1, bby1, bbx2, bby2]


def mixup(images, args, rand_index=None, lam=None):
    rand_index = torch.randperm(images.size()[0]).to(args.device)
    lam = np.random.beta(args.mixup, args.mixup)

    mixed_images = lam * images + (1 - lam) * images[rand_index]
    return mixed_images, rand_index.cpu(), lam, None


def mix_aug(images, args, rand_index=None, lam=None, bbox=None):
    if args.mix_type == "mixup":
        return mixup(images, args, rand_index, lam)
    elif args.mix_type == "cutmix":
        return cutmix(images, args, rand_index, lam, bbox)
    else:
        return images, None, None, None


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# Train
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    t1 = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        # inputs, _, _, _ = mixup_data(inputs, targets)
        inputs, _, _, _ = mix_aug(inputs, args)

        optimizer.zero_grad()
        outputs = net(inputs)
        soft_label = model_teacher(inputs).detach()
        
        outputs_ = F.log_softmax(outputs / args.temperature, dim=1)
        soft_label = F.softmax(soft_label / args.temperature, dim=1)

        # loss = args.temperature * args.temperature * loss_function_kl(outputs_, soft_label)
        loss = loss_function_kl(outputs_, soft_label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    t2 = time.time()

    print(f"Epoch: [{epoch}], Acc@1 {100.*correct/total:.3f}, Loss {train_loss/(batch_idx+1):.4f}, Time {t2-t1:.4f}")

# Test
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # print(f"Test: Acc@1 {100.*correct/total:.3f}, Loss {test_loss/(batch_idx+1):.4f}")

    # Save checkpoint.
    acc = 100.0 * correct / total
    # if acc > best_acc:
    # save last checkpoint
    # if True:
    #     state = {
    #         "state_dict": net.state_dict(),
    #         "acc": acc,
    #         "epoch": epoch,
    #     }
    #     # if not os.path.isdir('checkpoint'):
    #     #     os.mkdir('checkpoint')

    #     path = os.path.join(args.output_dir, "./ckpt.pth")
    #     torch.save(state, path)
    #     best_acc = acc

    return acc


best_acc1 = 0
start_time = time.time()
for epoch in range(args.epochs):

    train(epoch)
    
    # fast test
    # if (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
    #     test(epoch)
    # scheduler.step()

    if epoch % 10 == 9 or epoch == args.epochs - 1:
        if epoch > args.epochs * 0.8:
            top1 = test(epoch)
        else:
            top1 = 0
    else:
        top1 = 0

    scheduler.step()
    if top1 > best_acc1:
        best_acc1 = max(top1, best_acc1)
        best_epoch = epoch

end_time = time.time()

print(f"Best accuracy is {best_acc1}@{best_epoch}")
print(f"total time: {end_time - start_time} s")
print(args.syn_data_path)
