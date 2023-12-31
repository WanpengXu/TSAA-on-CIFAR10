#!/bin/bash

cd /root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/

#python train.py --model res50
#python train.py --model vgg16
#python train.py --model res18
#python train.py --model res34
#python train.py --model vgg11
#python train.py --model vgg13
#python train.py --model res50
#python train.py --model res101
python train.py --model res152
python train.py --model vgg19
python train.py --model google
python train.py --model mobile
python train.py --model mobilev2

/usr/bin/shutdown