from torch import nn
from typing import Type, Union, List
class BasicBlock(nn.Module):
    def __init__(self,
        in_channel: int,
        out_channel: int,
        ):
        super(BasicBlock,self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channel,out_channel, 3,padding = 2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel,out_channel, 3),
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




class ResNet(nn.Module):
    def __init__(self,
        in_channel: int = 3,
        out_channel: int = 1,
        block: Type[Union[BasicBlock]] = BasicBlock,
        layers: List[int] = [2,2,2,2],
        num_classes: int = 10):
        super(ResNet,self).__init__()

        self.model_head = nn.Sequential(
            nn.Conv2d(3, in_channel, 7, 2, 3, bias= False),
            nn.BatchNorm2d(in_channel),
            nn.MaxPool2d(3,2,1)
        )
        self.layer1 = self._make_layer(block,3, 64, layers[0])
        self.layer2 = self._make_layer(block,64, 128,layers[1])
        self.layer3 = self._make_layer(block,128, 256,layers[2])
        self.layer4 = self._make_layer(block,256, 512,layers[3])
        self.model_tail_disp = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size = 2, stride =2),
            nn.ConvTranspose2d(256, 128, kernel_size = 2, stride =2),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )
    def forward(self, x):
        x = self.model_head(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        disp = self.model_tail_disp(x)
        return disp

    def _make_layer(self, block,in_channel, out_channel, layer_num):
        layers = []
        for i in range(layer_num):
            layers.append(block(in_channel, out_channel))
            in_channel = out_channel
        return nn.Sequential(*layers)