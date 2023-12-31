import argparse

from torch.backends import cudnn

from utils import get_device, _normalize, get_run_time

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F

from generators import GeneratorResnet


def get_args():
    parser = argparse.ArgumentParser(description='Test sparse')
    parser.add_argument('--test_dir',
                        default='/root/autodl-tmp/TSAA_Capstone_Project_Big_Files/dataset/CIFAR_10/cifar-10-batches-py/test',
                        help='Path of test set')
    parser.add_argument('--num_classes', type=int,
                        default=10, help='Number of dataset classes')
    # original model
    # parser.add_argument('--model', type=str,
    #                     default='res34', help='Model against GAN is trained: incv3, res50')
    parser.add_argument('--generator_weight_path', type=str,
                        default='/root/autodl-tmp/TSAA_Capstone_Project/code/temp/netG_-1_vgg11_eps255_epoch49_lam_spa0.002_loss10.3.pth',
                        help='Path of generator weight')
    parser.add_argument('--eps', type=int,
                        default=255, help='Perturbation Budget')
    # target model
    parser.add_argument('--target_model', type=str,
                        default='res50', help='Model under attack : vgg16, incv3, res50, dense161')
    parser.add_argument('--target_model_weight', type=str,
                        default='/root/autodl-tmp/TSAA_Capstone_Project/code/temp/20230528_ckpt.pth',
                        help='Weight of model')
    parser.add_argument('--target_class', type=int,
                        default=-1, help='-1 if untargeted')

    parser.add_argument('--batch_size', type=int,
                        default=10, help='Batch Size')
    return parser.parse_args()


args = get_args()
print(args)
print(f'target_model=\033[1m{args.target_model}\033[0m')

# GPU
device = get_device()

# if args.model == 'incv3':
#     netG = GeneratorResnet(eps=args.eps / 255., evaluate=True, data_dim='high')
# else:
#     netG = GeneratorResnet(eps=args.eps / 255., evaluate=True, data_dim='high')

print('==> Building generator..')

netG = GeneratorResnet(eps=args.eps / 255., evaluate=True, data_dim='high')
netG.load_state_dict(torch.load(args.generator_weight_path))
netG = netG.to(device)
if device != 'cpu':
    netG = torch.nn.DataParallel(netG)
netG.eval()

print('==> Building classifier..')
target_model = torchvision.models.resnet50(pretrained=False)
target_model.fc = nn.Linear(target_model.fc.in_features, 10)
target_model.to(device)
if device != 'cpu':
    target_model = torch.nn.DataParallel(target_model)
    cudnn.benchmark = True

checkpoint = torch.load(args.target_model_weight)
target_model.load_state_dict(checkpoint['net_state_dict'])
target_model.eval()
print(f"target_model_acc={checkpoint['acc']}")


def trans_incep(x):
    x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
    return x


# Setup-Data
data_transform = transforms.Compose([
    # transforms.Resize(scale_size),
    # transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    # transforms.Normalize(mean=mean, std=std)
])

def normalize(t):
    # CIFAR-10 test dataset
    mean = (0.4940, 0.4850, 0.4504)
    std = (0.2467, 0.2429, 0.2616)
    return _normalize(t, mean, std)


test_dir = args.test_dir
test_set = datasets.ImageFolder(test_dir, data_transform)
test_size = len(test_set)
print('Test data size:', test_size)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=12,
                                          pin_memory=True)

# Evaluation
adv_acc = 0
clean_acc = 0
fool_rate = 0
target_rate = 0
norm = 0
time_count = 0

for i, (img, label) in enumerate(test_loader):
    img, label = img.to(device), label.to(device)

    # Clean Image Classify
    if 'inc' in args.target_model or 'xcep' in args.target_model:
        clean_out = target_model(normalize(trans_incep(img.clone().detach())))
    else:
        clean_out = target_model(normalize(img.clone().detach()))
    clean_acc += (clean_out.argmax(dim=-1) == label).sum().item()
    # vutils.save_image(vutils.make_grid(img.detach(), normalize=True, scale_each=True), 'clean image.png')
    # print(clean_out.argmax(dim=-1))

    # Adversarial Image Classify
    # adv, _, adv_0, adv_00 = netG(img)
    time, (adv, _, adv_0, adv_00) = get_run_time(netG, img)
    time_count += time

    if 'inc' in args.target_model or 'xcep' in args.target_model:
        adv_out = target_model(normalize(trans_incep(adv.clone().detach())))
    else:
        adv_out = target_model(normalize(adv.clone().detach()))
    adv_acc += (adv_out.argmax(dim=-1) == label).sum().item()

    # 只是干扰分类成功的概率，不是把原本分类对干扰成分类错的概率
    fool_rate += (adv_out.argmax(dim=-1) != clean_out.argmax(dim=-1)).sum().item()
    # vutils.save_image(vutils.make_grid(adv, normalize=True, scale_each=True), 'adversarial image.png')
    # quit()
    # print(adv_out.argmax(dim=-1))
    # print('-'*30)
    if args.target_class != -1:
        target_class = torch.LongTensor(img.size(0))
        target_class.fill_(args.target_class)
        target_class = target_class.to(device)
        target_rate += (adv_out.argmax(dim=-1) == target_class).sum().item()

    norm += torch.norm(adv_0.clone().detach(), 0)

print('L0 norm:', norm / test_size)
print('time:', time_count / test_size)  # What's the meaning
if args.target_class != -1:
    print(
        'Acc in Clean Image: {0:.3%}\t Acc in Adversarial Image: {1:.3%}\t Fooling Rate:{2:.3%}\t Target Class Success Rate:{3:.3%}'.format(
            clean_acc / test_size, adv_acc / test_size, fool_rate / test_size, target_rate / test_size))
else:
    print('Acc in Clean Image: {0:.3%}\t Acc in Adversarial Image: {1:.3%}\t Fooling Rate:{2:.3%}'.format(
        clean_acc / test_size, adv_acc / test_size, fool_rate / test_size))
