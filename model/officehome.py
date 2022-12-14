import torch.nn as nn
import torch.nn.functional as F
from .resnet import get_resnet
import torch
from .utils import GaussianNoise

feature_dict = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048}


class OfficehomeNet(nn.Module):
    def __init__(self, backbone, bn_momentum, pretrained=True, data_parallel=True):
        super(OfficehomeNet, self).__init__()
        encoder = get_resnet(backbone, momentumn=bn_momentum, pretrained=pretrained)
        if data_parallel:
            self.encoder = nn.DataParallel(encoder)
        else:
            self.encoder = encoder

    def forward(self, x):
        feature = self.encoder(x)
        return feature


class OfficehomeClassifier(nn.Module):
    def __init__(self, backbone, classes=65, data_parallel=True, drop_out=0.0):
        super(OfficehomeClassifier, self).__init__()
        linear = nn.Sequential()
        linear.add_module("fc", nn.Linear(feature_dict[backbone], classes))

        if data_parallel:
            self.linear = nn.DataParallel(linear)
        else:
            self.linear = linear

    def forward(self, feature):
        feature = torch.flatten(feature, 1)
        feature = self.linear(feature)
        return feature
