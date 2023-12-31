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

# from torchvision.models import ResNet50_Weights, Inception_V3_Weights

from generators import GeneratorResnet

import os
import argparse

from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--generator_weight_path', type=str,
                    default='/root/autodl-tmp/TSAA_Capstone_Project/code/temp/adv_train_res50_eps255.pth',
                    help='Path of generator weight')
parser.add_argument('--eps', type=int,
                    default=255, help='Perturbation Budget')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Number of training samples/batch')
parser.add_argument('--model', type=str, default='res50', help='Classifier model is trained: incv3, res50, ...')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

import logging

checkpoint_path = os.path.join(os.getcwd(), 'checkpoint')
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

checkpoint_path = os.path.join(checkpoint_path, args.model)
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

logging.basicConfig(level=logging.DEBUG, format='',
                    handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(checkpoint_path, 'log0527.log'))])

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
logging.info('==> Preparing data..')
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

train_set = datasets.CIFAR10(
    root='/root/autodl-tmp/TSAA_Capstone_Project_Big_Files/dataset', train=True,
    download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=12)

test_set = datasets.CIFAR10(
    root='/root/autodl-tmp/TSAA_Capstone_Project_Big_Files/dataset', train=False,
    download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=12)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Classifier Model
logging.info('==> Building model..')

model = torchvision.models.resnet50(pretrained=True)    # will be deprecated
# model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)
model.to(device)
if device != 'cpu':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

# Generator Model
netG = GeneratorResnet(evaluate=True, data_dim='high')
netG.load_state_dict(torch.load(args.generator_weight_path))
netG = netG.to(device)
if device != 'cpu':
    netG = torch.nn.DataParallel(netG)
netG.eval()

if args.resume:
    # Load checkpoint.
    logging.info('==> Resuming from checkpoint..')
    assert os.path.isdir(checkpoint_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(checkpoint_path, 'ckpt.pth'))
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['acc']
    model.load_state_dict(checkpoint['net_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


def original_train(epoch):
    logging.info(f'\nEpoch: {epoch}')
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
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
    logging.info('Loss: {:.3f} | Acc: {:.3f}% ({}/{})'.format(
        train_loss / len(train_loader), 100. * correct / total, correct, total))


def original_test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: {:.3f} | Acc: {:.3f}% ({}/{})'.format(
                test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        logging.info('Loss: {:.3f} | Acc: {:.3f}% ({}/{})'.format(
            test_loss / len(test_loader), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        logging.info('==> Saving..')
        state = {
            'epoch': epoch,
            'acc': acc,
            'net_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict()
        }
        now = time.strftime("%Y%m%d", time.localtime())
        torch.save(state, os.path.join(checkpoint_path, f'{now}_ckpt.pth'))
        best_acc = acc


# Training
def train(epoch):
    logging.info('\nEpoch: {}'.format(epoch))
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        advs, _, _, _ = netG(inputs)
        combined_inputs = torch.cat([inputs, advs], dim=0)
        combined_labels = torch.cat([labels, labels], dim=0)

        optimizer.zero_grad()
        outputs = model(combined_inputs)
        loss = criterion(outputs, combined_labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += combined_labels.size(0)
        correct += predicted.eq(combined_labels).sum().item()

        # len(train_loader)=ceil(50000/128)=391 batches
        progress_bar(batch_idx, len(train_loader), 'Loss: {:.3f} | Acc: {:.3f}% ({}/{})'.format(
            train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    logging.info('Loss: {:.3f} | Acc: {:.3f}% ({}/{})'.format(
        train_loss / len(train_loader), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            advs, _, _, _ = netG(inputs)
            combined_inputs = torch.cat([inputs, advs], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)

            outputs = model(combined_inputs)
            loss = criterion(outputs, combined_labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += combined_labels.size(0)
            correct += predicted.eq(combined_labels).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: {:.3f} | Acc: {:.3f}% ({}/{})'.format(
                test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        logging.info('Loss: {:.3f} | Acc: {:.3f}% ({}/{})'.format(
            test_loss / len(test_loader), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        logging.info('==> Saving..')
        state = {
            'epoch': epoch,
            'acc': acc,
            'net_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict()
        }
        now = time.strftime("%Y%m%d", time.localtime())
        torch.save(state, os.path.join(checkpoint_path, f'{now}_ckpt.pth'))
        best_acc = acc


criterion = nn.CrossEntropyLoss()

# 权重冻结，仅训练全连接层
for name, para in model.named_parameters():
    if "fc" not in name:
        # para.requires_grad_(False)
        para.requires_grad = False
paras = [para for para in model.parameters() if para.requires_grad]
optimizer = optim.Adam(paras, lr=args.lr)

for epoch in range(10):
    original_train(epoch)
    original_test(epoch)

# 权重解冻，整体训练
for name, para in model.named_parameters():
    if "fc" not in name:
        para.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
for epoch in range(start_epoch, 200):   # 不管怎么都训练到200轮位置
    train(epoch)
    test(epoch)
    # scheduler.step()
    # original_train(epoch)
    # original_test(epoch)
