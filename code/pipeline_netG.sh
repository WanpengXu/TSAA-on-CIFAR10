##!/bin/bash
#

# External logging

#cd /root/autodl-tmp/TSAA_Capstone_Project/code/
#
#timestamp=$(date +%Y-%m-%d_%H-%M-%S)
#filename=log/my_train_${timestamp}.log

#python -u ./my_train.py --lam_spa 0.0047 2>&1 | tee $filename
#python -u ./my_train.py --lam_spa 0.0049 2>&1 | tee $filename
#python -u ./my_train.py --lam_spa 0.005 2>&1 | tee $filename
#python -u ./my_train.py --lam_spa 0.0053 2>&1 | tee $filename
#python -u ./my_train.py --lam_spa 0.0055 2>&1 | tee $filename
#python -u ./my_train.py --lam_spa 0.0057 2>&1 | tee $filename
#python -u ./my_train.py --lam_spa 0.0059 2>&1 | tee $filename
#python -u ./my_train.py --lam_spa 0.006 2>&1 | tee $filename

#python -u ./my_train.py --lam_spa 0.0001 2>&1 | tee $filename
#python -u ./my_train.py --lam_spa 0.0005 2>&1 | tee $filename
#python -u ./my_train.py --lam_spa 0.001 2>&1 | tee $filename


#python -u ./my_train.py --lam_spa 0.0001 2>&1 | tee $filename
#python -u ./my_train.py --lam_spa 0.0002 2>&1 | tee $filename
#python -u ./my_train.py --lam_spa 0.0003 2>&1 | tee $filename
#python -u ./my_train.py --lam_spa 0.0004 2>&1 | tee $filename
#python -u ./my_train.py --lam_spa 0.0005 2>&1 | tee $filename
#python -u ./my_train.py --lam_spa 0.0006 2>&1 | tee $filename
#python -u ./my_train.py --lam_spa 0.0007 2>&1 | tee $filename
#python -u ./my_train.py --lam_spa 0.0008 2>&1 | tee $filename
#python -u ./my_train.py --lam_spa 0.0009 2>&1 | tee $filename
#python -u ./my_train.py --lam_spa 0.001 2>&1 | tee $filename

# Internal logging

python ./my_train.py  --lam_spa=0.001 --model="mobilev2" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/mobilev2/20230409_ckpt.pth'
python ./my_train.py  --lam_spa=0.003 --model="mobilev2" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/mobilev2/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.003 --model="mobilev2" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/mobilev2/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.004 --model="mobilev2" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/mobilev2/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.005 --model="mobilev2" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/mobilev2/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.006 --model="mobilev2" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/mobilev2/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.007 --model="mobilev2" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/mobilev2/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.008 --model="mobilev2" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/mobilev2/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.009 --model="mobilev2" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/mobilev2/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.01 --model="mobilev2" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/mobilev2/20230409_ckpt.pth'

python ./my_train.py  --lam_spa=0.001 --model="vgg16" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/temp/VGG16.pth'
python ./my_train.py  --lam_spa=0.005 --model="vgg16" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/temp/VGG16.pth'


#
#python ./my_train.py  --lam_spa=0.001 --model="res18" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res18/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.002 --model="res18" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res18/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.003 --model="res18" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res18/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.004 --model="res18" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res18/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.005 --model="res18" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res18/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.006 --model="res18" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res18/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.007 --model="res18" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res18/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.008 --model="res18" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res18/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.009 --model="res18" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res18/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.01 --model="res18" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res18/20230403_ckpt.pth'
#
#
#python ./my_train.py  --lam_spa=0.001 --model="res34" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res34/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.002 --model="res34" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res34/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.003 --model="res34" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res34/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.004 --model="res34" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res34/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.005 --model="res34" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res34/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.006 --model="res34" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res34/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.007 --model="res34" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res34/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.008 --model="res34" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res34/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.009 --model="res34" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res34/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.01 --model="res34" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res34/20230403_ckpt.pth'
#
#
#python ./my_train.py  --lam_spa=0.001 --model="vgg11" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/vgg11/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.002 --model="vgg11" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/vgg11/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.003 --model="vgg11" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/vgg11/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.004 --model="vgg11" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/vgg11/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.005 --model="vgg11" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/vgg11/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.006 --model="vgg11" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/vgg11/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.007 --model="vgg11" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/vgg11/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.008 --model="vgg11" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/vgg11/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.009 --model="vgg11" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/vgg11/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.01 --model="vgg11" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/vgg11/20230403_ckpt.pth'
#
#python ./my_train.py  --lam_spa=0.001 --model="vgg13" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/vgg13/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.002 --model="vgg13" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/vgg13/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.003 --model="vgg13" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/vgg13/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.004 --model="vgg13" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/vgg13/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.005 --model="vgg13" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/vgg13/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.006 --model="vgg13" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/vgg13/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.007 --model="vgg13" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/vgg13/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.008 --model="vgg13" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/vgg13/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.009 --model="vgg13" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/vgg13/20230403_ckpt.pth'
#python ./my_train.py  --lam_spa=0.01 --model="vgg13" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/vgg13/20230403_ckpt.pth'

#/usr/bin/shutdown

#python ./my_train.py --lam_spa 0.004
#python ./my_train.py --lam_spa 0.005
#python ./my_train.py --lam_spa 0.006
#python ./my_train.py --lam_spa 0.007
#python ./my_train.py --lam_spa 0.008
#python ./my_train.py --lam_spa 0.009
#python ./my_train.py --lam_spa 0.01



#python ./my_train.py  --lam_spa=0.0001 --model="res50" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res50/20230406_ckpt.pth'
#python ./my_train.py  --lam_spa=0.0003 --model="res50" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res50/20230406_ckpt.pth'
#python ./my_train.py  --lam_spa=0.0005 --model="res50" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res50/20230406_ckpt.pth'
#python ./my_train.py  --lam_spa=0.0007 --model="res50" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res50/20230406_ckpt.pth'
#python ./my_train.py  --lam_spa=0.0009 --model="res50" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res50/20230406_ckpt.pth'
#python ./my_train.py  --lam_spa=0.001 --model="res50" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res50/20230406_ckpt.pth'
#python ./my_train.py  --lam_spa=0.002 --model="res50" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res50/20230406_ckpt.pth'
#python ./my_train.py  --lam_spa=0.003 --model="res50" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res50/20230406_ckpt.pth'
#python ./my_train.py  --lam_spa=0.004 --model="res50" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res50/20230406_ckpt.pth'
#python ./my_train.py  --lam_spa=0.005 --model="res50" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res50/20230406_ckpt.pth'
#
python ./my_train.py  --lam_spa=0.001 --model="google" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/google/20230409_ckpt.pth'
python ./my_train.py  --lam_spa=0.003 --model="google" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/google/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.0005 --model="google" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/google/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.0007 --model="google" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/google/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.0009 --model="google" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/google/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.001 --model="google" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/google/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.002 --model="google" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/google/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.003 --model="google" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/google/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.004 --model="google" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/google/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.005 --model="google" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/google/20230409_ckpt.pth'
#
python ./my_train.py  --lam_spa=0.001 --model="mobile" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/mobile/20230409_ckpt.pth'
python ./my_train.py  --lam_spa=0.003 --model="mobile" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/mobile/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.0005 --model="mobile" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/mobile/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.0007 --model="mobile" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/mobile/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.0009 --model="mobile" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/mobile/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.001 --model="mobile" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/mobile/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.002 --model="mobile" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/mobile/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.003 --model="mobile" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/mobile/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.004 --model="mobile" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/mobile/20230409_ckpt.pth'
#python ./my_train.py  --lam_spa=0.005 --model="mobile" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/mobile/20230409_ckpt.pth'

#python ./my_train.py  --lam_spa=0.0002 --model="res50" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res50/20230406_ckpt.pth'
#python ./my_train.py  --lam_spa=0.0003 --model="res50" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res50/20230406_ckpt.pth'
#python ./my_train.py  --lam_spa=0.0004 --model="res50" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res50/20230406_ckpt.pth'
#python ./my_train.py  --lam_spa=0.0005 --model="res50" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res50/20230406_ckpt.pth'
#python ./my_train.py  --lam_spa=0.0006 --model="res50" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res50/20230406_ckpt.pth'
#python ./my_train.py  --lam_spa=0.0007 --model="res50" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res50/20230406_ckpt.pth'
#python ./my_train.py  --lam_spa=0.0008 --model="res50" --model_weight='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint/res50/20230406_ckpt.pth'

#/usr/bin/shutdown