# -*-coding:utf-8-*-
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable

class GaussianNoise(nn.Module):

    def __init__(self, input_shape, std=0.05):
        super(GaussianNoise, self).__init__()
        self.input_shape = input_shape
        self.std = std

    def forward(self, x):
        batch_size = x.size(0)
        shape = (batch_size,) + self.input_shape
        self.noise = Variable(torch.zeros(shape).cuda())
        self.noise.data.normal_(0, std=self.std)
        return x.cuda() + self.noise