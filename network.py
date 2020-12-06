import torch
from torch import nn
from typing import Type, Union, List

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self,ngf,nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d(3, ngf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.Conv2d( ngf * 4, ngf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (nc) x 64 x 64
            nn.Conv2d( ngf * 2, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.Conv2d( ngf, nc, 3, 1, 1, bias=False),
            nn.ReLU()
        )
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self,ndf, nc):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class BasicBlock(nn.Module):
    def __init__(self,
        in_channel: int,
        out_channel: int,
        ):
        super(BasicBlock,self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channel,out_channel, 3,padding = 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel,out_channel, 3,padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        self.resize = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size = 1, bias = False),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        identity = self.resize(x)
        out = self.block(x)
        out += identity
        out = nn.ReLU()(out)

        return out


class GeneratorSkip(nn.Module):
    def __init__(self,ngf,nc,block):
        super(GeneratorSkip, self).__init__()
        self.layer1 = block(3,ngf*8)
        self.layer2 = block(ngf*8,ngf*4)
        self.layer3 = block(ngf*4,ngf*2)
        self.layer4 = block(ngf*2,ngf)
        self.tail = nn.Sequential(
            nn.Conv2d( ngf, nc, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.tail(x)
        return out

class GeneratorSkipMultitask(nn.Module):
    def __init__(self,ngf,block):
        super(GeneratorSkipMultitask, self).__init__()
        self.layer1 = block(3,ngf*8)
        self.layer2 = block(ngf*8,ngf*4)
        self.layer3 = block(ngf*4,ngf*2)
        self.layer4 = block(ngf*2,ngf)
        self.nor = nn.Sequential(
            block(ngf,ngf//2),
            nn.Conv2d( ngf//2, 3, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.disp = nn.Sequential(
            block(ngf,ngf//2),
            nn.Conv2d( ngf//2, 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.rough = nn.Sequential(
            block(ngf,ngf//2),
            nn.Conv2d( ngf//2, 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        nor = self.nor(x)
        disp = self.disp(x)
        rough = self.rough(x)
        return nor, disp, rough