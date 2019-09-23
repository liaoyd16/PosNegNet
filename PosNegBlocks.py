import torch
import torch.nn as nn
import torch.nn.functional as F
from meta_const import *

class PosNegConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super(PosNegConv2d, self).__init__()
        self.conv2d_pos2pos = torch.nn.Parameter(torch.rand(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.conv2d_pos2neg = torch.nn.Parameter(torch.rand(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.conv2d_neg2pos = torch.nn.Parameter(torch.rand(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.conv2d_neg2neg = torch.nn.Parameter(torch.rand(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.bias_pos = torch.nn.Parameter(torch.rand(out_channels, 1, 1))
        self.bias_neg = torch.nn.Parameter(torch.rand(out_channels, 1, 1))

        self.padding = padding
        self.stride = stride

    def forward(self, xs):
        '''
            # x_pos: [in_channels, h, w]    x_neg: [in_channels, h, w]
            #   v                      \\     /          v
            #   v pos2pos[k_0] pos2neg[k_1] neg2pos[k_0] v neg2neg[k_1]
            #   v                          X             v
            # y_pos: [out_channels, h', w'] y_neg: [out_channels, h', w']
        '''
        x_pos = xs[0]
        x_neg = xs[1]
        y_pos =   F.conv2d(x_pos, (self.conv2d_pos2pos)**POW, padding=self.padding, stride=self.stride)\
                - F.conv2d(x_neg, (self.conv2d_neg2pos)**POW, padding=self.padding, stride=self.stride)\
                + self.bias_pos
        y_pos[:,:x_pos.shape[1]] = y_pos[:,:x_pos.shape[1]] + x_pos
        y_pos = F.relu(y_pos)

        y_neg =   F.conv2d(x_pos, (self.conv2d_pos2neg)**POW, padding=self.padding, stride=self.stride)\
                - F.conv2d(x_neg, (self.conv2d_neg2neg)**POW, padding=self.padding, stride=self.stride)\
                + self.bias_neg
        y_neg[:,:x_neg.shape[1]] = y_neg[:,:x_neg.shape[1]] + x_neg
        y_neg = F.relu(y_neg)

        return (y_pos, y_neg)


class PosNegMaxPool2d(nn.Module):
    def __init__(self, channels):
        super(PosNegMaxPool2d, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2))
        self.batchnorm = nn.BatchNorm2d(channels)

    def forward(self, xs):
        return (self.batchnorm(self.maxpool(xs[0])), self.batchnorm(self.maxpool(xs[1])))


class PosNegLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PosNegLinear, self).__init__()
        self.weights_pos2pos = torch.nn.Parameter(torch.rand(out_dim, in_dim))
        self.weights_pos2neg = torch.nn.Parameter(torch.rand(out_dim, in_dim))
        self.weights_neg2pos = torch.nn.Parameter(torch.rand(out_dim, in_dim))
        self.weights_neg2neg = torch.nn.Parameter(torch.rand(out_dim, in_dim))
        self.bias_pos = torch.nn.Parameter(torch.rand(out_dim))
        self.bias_neg = torch.nn.Parameter(torch.rand(out_dim))

        self.weights_pos2pos.requires_grad = True
        self.weights_pos2neg.requires_grad = True
        self.weights_neg2pos.requires_grad = True
        self.weights_neg2neg.requires_grad = True
        self.bias_pos.requires_grad = True
        self.bias_neg.requires_grad = True

    def forward(self, xs):
        x_pos = xs[0]
        x_neg = xs[1]
        y_pos = F.relu(F.linear(x_pos, (self.weights_pos2pos)**POW)\
                     - F.linear(x_neg, (self.weights_neg2pos)**POW)\
                     + self.bias_pos)
        y_neg = F.relu(F.linear(x_pos, (self.weights_pos2neg)**POW)\
                     - F.linear(x_neg, (self.weights_neg2neg)**POW)\
                     + self.bias_neg)
        return (y_pos, y_neg)


class PosNegTop(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PosNegTop, self).__init__()
        self.weight_pos2top = torch.nn.Parameter(torch.rand(out_dim, in_dim))
        self.weight_neg2top = torch.nn.Parameter(torch.rand(out_dim, in_dim))
        self.bias_top = torch.nn.Parameter(torch.rand(out_dim))

        self.weight_pos2top.requires_grad = True
        self.weight_neg2top.requires_grad = True
        self.bias_top.requires_grad = True

    def forward(self, xs):
        x_pos = xs[0]
        x_neg = xs[1]
        y_top = F.linear(x_pos, (self.weight_pos2top)**POW)\
                     - F.linear(x_neg, (self.weight_neg2top)**POW)\
                     + self.bias_top
        return F.softmax(y_top, dim=1)
