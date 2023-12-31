import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 控制生成器的特征图
ngf = 64

class GeneratorResnet(nn.Module):
    # data_dim='low'测试MNIST
    def __init__(self, data_dim='high', eps=1.0, evaluate=False):
        super(GeneratorResnet, self).__init__()
        self.data_dim = data_dim
        # Input_size = 3, n, n
        # H = floor((H + 2 * padding - kernel_size) / stride) + 1
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),                                      # H = 32 + 2 * 3 = 38, 这步是为了kernel_size=7准备的6个新格
            # nn.Conv2d(1, ngf, kernel_size=7, padding=0, bias=False),    # H = 38 - 7 + 1 = 32
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),    # H = 38 - 7 + 1 = 32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),        # H = floor(32 + 2 - 3 / 2) + 1 = 16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),    # H = (16 + 2 - 3 / 2) + 1 = 8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        if self.data_dim == 'high':
            self.resblock3 = ResidualBlock(ngf * 4)
            self.resblock4 = ResidualBlock(ngf * 4)
            self.resblock5 = ResidualBlock(ngf * 4)
            self.resblock6 = ResidualBlock(ngf * 4)
            

        # Input size = 3, n/4, n/4
        # H = (H - 1) * stride + output_padding - 2 * padding + kernel_size
        self.upsampl_inf1 = nn.Sequential(
            # 反卷积
            # H = 7 * 2 + 1 - 2 + 3 = 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl_inf2 = nn.Sequential(
            # H = 15 * 2 + 1 - 2 + 3 = 32
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf_inf = nn.Sequential(
            # H = 32 + 2 * 3 = 38
            nn.ReflectionPad2d(3),
            # H= 38 - 7 + 1 = 32
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )

        # self.decoder_inf = nn.Sequential(
        #     self.upsampl_inf1,
        #     self.upsampl_inf2,
        #     self.blockf_inf
        # )
        
        # Input size = 3, n/4, n/4
        self.upsampl_01 = nn.Sequential(
            # H = 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl_02 = nn.Sequential(
            # H = 32
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf_0 = nn.Sequential(
            # H = 38
            nn.ReflectionPad2d(3),
            # H = 32
            nn.Conv2d(ngf, 1, kernel_size=7, padding=0)
        )

        # self.decoder_0 = nn.Sequential(
        #     self.upsampl_01,
        #     self.upsampl_02,
        #     self.blockf_0
        # )

        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)
        
        self.eps = eps
        self.evaluate = evaluate

    def forward(self, input):
        # 编码器
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        if self.data_dim == 'high':
            x = self.resblock3(x)
            x = self.resblock4(x)
            x = self.resblock5(x)
            x = self.resblock6(x)
        code = x
        # 解码器D_1
        x = self.upsampl_inf1(code)
        x = self.upsampl_inf2(x)
        x = self.blockf_inf(x)
        # x = self.decoder_inf(code)
        x_inf = self.eps * torch.tanh(x) # Output range [-eps, eps]
        # 解码器D_2
        x = self.upsampl_01(code)
        x = self.upsampl_02(x)
        x = self.blockf_0(x)
        # x = self.decoder_0(code)
        x = (torch.tanh(x) + 1) / 2
        # 推理阶段
        if self.evaluate:
            # 小于0.5的元素替换为0，大于等于0.5的元素替换为1，形成二值图像，同时.detach()剥离出计算图阻止反向传播
            x_0 = torch.where(x<0.5, torch.zeros_like(x).detach(), torch.ones_like(x).detach())
        else:
            # 小于0.5的元素替换为0，大于等于0.5的元素替换为1，形成二值图像，同时.detach()剥离出计算图阻止反向传播
            # 然后随机取约一半的像素保留原值，另一半量化为上面生成的二值图像并.detach()禁止反向传播
            x_0 = torch.where(torch.rand(x.shape).cuda()<0.5, x, torch.where(x<0.5, torch.zeros_like(x), torch.ones_like(x)).detach())
        # 合成
        x_out = torch.clamp(x_inf * x_0 + input, min=0, max=1)
        return x_out, x_inf, x_0, x


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        # H = n
        self.block = nn.Sequential(
            # H = n + 2
            nn.ReflectionPad2d(1),
            # H = n + 2 - 3 + 1 = n
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),

            nn.Dropout(0.5),

            # H = H = n + 2
            nn.ReflectionPad2d(1),
            # # H = n + 2 - 3 + 1 = n
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual



if __name__ == '__main__':
    netG = GeneratorResnet(data_dim='low', evaluate=True)
    # netG.to('cuda')
    # from torchsummary import summary
    # summary(netG, input_size=(3, 32, 32))
    # netG.to('cpu')
    test_sample = torch.rand(1, 3, 32, 32)
    print('Generator output:', netG(test_sample)[0].size())
    print('Generator parameters:', sum(p.numel() for p in netG.parameters() if p.requires_grad))