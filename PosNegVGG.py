import torch
import torch.nn as nn
import torch.nn.functional as F
from PosNegBlocks import PosNegConv2d, PosNegLinear, PosNegMaxPool2d, PosNegTop

class PosNegVgg(nn.Module):
    def __init__(self):
        super(PosNegVgg, self).__init__()
        self.conv1 = nn.Sequential(
            PosNegConv2d(1, 8, (3, 3), 1),
            PosNegConv2d(8, 8, (3, 3), 1),
            PosNegMaxPool2d(8),
        )
        self.conv2 = nn.Sequential(
            PosNegConv2d(8, 8, (3, 3), 1),
            PosNegMaxPool2d(8),
        )
        self.conv3 = nn.Sequential(
            PosNegConv2d(8, 16,(3, 3), 1),
            PosNegMaxPool2d(16),
        )
        self.linear1 = PosNegLinear(256, 32)
        self.linear2 = PosNegTop(32, 10)

    def forward(self, x):
        '''
            size changing: 
            [1:1,32,32] -> [4:4,16,16] -> [8:8,8,8] -> [16:16,4,4]
          ~ [256:256][pos,neg] -> [32,32][pos,neg]
          ~ [32,32][pos,neg] -> 10[pos]
        '''
        bs = x.shape[0]
        xs = self.conv1((x, x))
        xs = self.conv2(xs)
        xs = self.conv3(xs)
        xpos = xs[0].reshape(bs, 256)
        xneg = xs[1].reshape(bs, 256)
        xs = self.linear1((xpos, xneg))
        x = self.linear2(xs)
        return x