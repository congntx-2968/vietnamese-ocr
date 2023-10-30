import torch
from torch import nn
from torchvision import models
from einops import rearrange
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.efficientnet import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
)


model_dict = {
    'efficientnet_b0': [efficientnet_b0, 1280],
    'efficientnet_b1': [efficientnet_b1, 1280],
    'efficientnet_b2': [efficientnet_b2, 1408],
    'efficientnet_b3': [efficientnet_b3, 1536],
    'efficientnet_b4': [efficientnet_b4, 1792],
    'efficientnet_b5': [efficientnet_b5, 2048],
    'efficientnet_b6': [efficientnet_b6, 2304],
    'efficientnet_b7': [efficientnet_b7, 2560]
}


class EfficientNet(nn.Module):
    def __init__(self, name, hidden, pretrained=True, dropout=0.5, freeze=False, **kwargs):
        super(EfficientNet, self).__init__()
        model, dims = model_dict[name]
        cnn = model(pretrained=pretrained, **kwargs)

        self.features = cnn.features
        self.dropout = nn.Dropout(dropout)
        self.last_conv_1x1 = nn.Conv2d(dims, hidden, 1)

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Shape:
            - x: (N, C, H, W)
            - output: (W, N, C)
        """

        conv = self.features(x)
        conv = self.dropout(conv)
        conv = self.last_conv_1x1(conv)

        conv = conv.transpose(-1, -2)
        conv = conv.mean(-1)
        conv = conv.permute(-1, 0, 1)
        return conv


def efficientnet_b0(name, hidden, pretrained=True, dropout=0.5):
    return EfficientNet(name, hidden, pretrained, dropout)
