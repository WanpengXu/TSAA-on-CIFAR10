#!/bin/bash

cd /root/autodl-tmp/TSAA_Capstone_Project/code/

filename=log/one_shot_my_eval_$(date +%Y-%m-%d_%H-%M-%S).log
touch "$filename"

weight_folder='/root/autodl-tmp/TSAA_Capstone_Project/code/pytorch_cifar/checkpoint'

python -u ./my_eval.py --target_model='res18' --target_model_weight="${weight_folder}/res18/20230403_ckpt.pth" 2>&1 | tee -a $filename
python -u ./my_eval.py --target_model='res34' --target_model_weight="${weight_folder}/res34/20230403_ckpt.pth" 2>&1 | tee -a $filename
python -u ./my_eval.py --target_model='res50' --target_model_weight="${weight_folder}/res50/20230406_ckpt.pth" 2>&1 | tee -a $filename
python -u ./my_eval.py --target_model='res101' --target_model_weight="${weight_folder}/res101/20230407_ckpt.pth" 2>&1 | tee -a $filename
python -u ./my_eval.py --target_model='res152' --target_model_weight="${weight_folder}/res152/20230409_ckpt.pth" 2>&1 | tee -a $filename
python -u ./my_eval.py --target_model='vgg11' --target_model_weight="${weight_folder}/vgg11/20230403_ckpt.pth" 2>&1 | tee -a $filename
python -u ./my_eval.py --target_model='vgg13' --target_model_weight="${weight_folder}/vgg13/20230403_ckpt.pth" 2>&1 | tee -a $filename
python -u ./my_eval.py --target_model='vgg16' --target_model_weight="${weight_folder}/vgg16/ckpt.pth" 2>&1 | tee -a $filename
python -u ./my_eval.py --target_model='vgg19' --target_model_weight="${weight_folder}/vgg19/20230409_ckpt.pth" 2>&1 | tee -a $filename
python -u ./my_eval.py --target_model='google' --target_model_weight="${weight_folder}/google/20230409_ckpt.pth" 2>&1 | tee -a $filename
python -u ./my_eval.py --target_model='mobile' --target_model_weight="${weight_folder}/mobile/20230409_ckpt.pth" 2>&1 | tee -a $filename
python -u ./my_eval.py --target_model='mobilev2' --target_model_weight="${weight_folder}/mobilev2/20230409_ckpt.pth" 2>&1 | tee -a $filename

#/usr/bin/shutdown
