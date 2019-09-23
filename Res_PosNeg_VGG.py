import torch
import torch.nn as nn
import torch.nn.functional as F
from PosNegBlocks import PosNegConv2d, PosNegLinear, PosNegMaxPool2d, PosNegTop
from ResBlock import ResBlock

class Res_PosNeg_VGG(nn.Module):
    def __init__(self):
        super(Res_PosNeg_VGG, self).__init__()
        self.conv1 = nn.Sequential(
            ResBlock(1, 8),
            ResBlock(8, 8),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.BatchNorm2d(8),
        )
        self.conv2 = nn.Sequential(
            ResBlock(8, 16),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.BatchNorm2d(16),
        )
        self.conv3 = nn.Sequential(
            ResBlock(16, 32),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.BatchNorm2d(32),
        )
        self.linear1 = PosNegLinear(256, 32)
        self.linear2 = PosNegTop(32, 10)

    def forward(self, x):
        '''
            size changing: 
            [1,32,32] -> [8,16,16] -> [16,8,8] -> [32,4,4]
          ~ [256:256][pos,neg] -> [32,32][pos,neg]
          ~ [32,32][pos,neg] -> 10[pos]
        '''
        bs = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        xpos = x[:,:16].view(bs, -1)
        xneg = x[:,16:].view(bs, -1)
        xs = self.linear1((xpos, xneg))
        x = self.linear2(xs)
        return x
