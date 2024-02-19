# TSAA-on-CIFAR10

## Description

本仓库是使用 TSAA 算法在 CIFAR10 数据集上实现的一个可迁移稀疏对抗样本生成系统，仅用于《人工智能与机器学习》课程设计的评分。

This repository is a transferable sparse adversarial example generation system implementing the TSAA algorithm on the CIFAR10 dataset, exclusively intended for grading the 《人工智能与机器学习》 course project. 

如有侵权请联系本人删除。

Please contact me for removal if there is any infringement.

## Installation

```bash
conda create -n aiml python=3.10
conda activate aiml

git clone https://github.com/WanpengXu/TSAA-on-CIFAR10.git
cd TSAA-on-CIFAR10
chmod +x download_weights.sh
download_weights.sh
pip install -r requirements.txt
```

## Quick Start

```bash
cd code
gradio app.py
```

---

![image-20231231200522161](https://testingcf.jsdelivr.net/gh/WanpengXu/myPicGo/img/ms20231231200522654.png)
