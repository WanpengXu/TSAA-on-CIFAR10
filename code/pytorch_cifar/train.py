'''Train CIFAR10 with PyTorch.'''
import time
from shutil import copyfile

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--cifar_dir', type=str,
#                     default='/root/autodl-tmp/TSAA_Capstone_Project_Big_Files/dataset/CIFAR_10/cifar-10-batches-py/train',
#                     help='Path of cifar dataset')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--model', type=str, default='res50', help='Classifier model is trained: incv3, res50, ...')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# train_set = datasets.ImageFolder(root=args.train_dir, transform=transform_train)
train_set = datasets.CIFAR10(
    root='/root/autodl-tmp/TSAA_Capstone_Project_Big_Files/dataset', train=True,
    download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=12)

# test_set = datasets.ImageFolder(root=args.test_dir, transform=transform_test)
test_set = datasets.CIFAR10(
    root='./root/autodl-tmp/TSAA_Capstone_Project_Big_Files/dataset', train=False,
    download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=12)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = None
if args.model == 'res18':
    net = ResNet18()
elif args.model == 'res50':
    net = ResNet50()
elif args.model == 'res34':
    net = ResNet34()
elif args.model == 'res101':
    net = ResNet101()
elif args.model == 'res152':
    net = ResNet152()
elif args.model == 'vgg11':
    net = VGG('VGG11')
elif args.model == 'vgg13':
    net = VGG('VGG13')
elif args.model == 'vgg16':
    net = VGG('VGG16')
elif args.model == 'vgg19':
    net = VGG('VGG19')
elif args.model == 'google':
    net = GoogLeNet()
elif args.model == 'mobile':
    net = MobileNet()
elif args.model == 'mobilev2':
    net = MobileNetV2()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint_path = os.path.join(os.getcwd(), 'checkpoint')
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

checkpoint_path = os.path.join(checkpoint_path, args.model)
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(checkpoint_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(checkpoint_path, 'ckpt.pth'))
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['acc']
    net.load_state_dict(checkpoint['net_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


# Training
def train(epoch):
    print('\nEpoch: {}'.format(epoch))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # len(train_loader)=ceil(50000/128)=391 batches
        progress_bar(batch_idx, len(train_loader), 'Loss: {:.3f} | Acc: {:.3f}% ({}/{})'.format(
            train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: {:.3f} | Acc: {:.3f}% ({}/{})'.format(
                test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('==> Saving..')
        state = {
            'epoch': epoch,
            'acc': acc,
            'net_state_dict': net.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict()
        }
        now = time.strftime("%Y%m%d", time.localtime())
        torch.save(state, os.path.join(checkpoint_path, f'{now}_ckpt.pth'))
        best_acc = acc


for epoch in range(start_epoch, 200):
    train(epoch)
    test(epoch)
    scheduler.step()
