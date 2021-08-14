# ! -*- coding: utf-8 -*-
from typing import Callable, Optional
import torch
import torch.nn as nn
import math
from torch.nn.modules import dropout
from functools import partial
from collections import OrderedDict
from torch.nn.modules.conv import Conv2d

# efficientnet_V2 vs efficientnet
"""
1. Architecture update by NAS search(mobilenetV3 first used NAS): 
          - MBConv and Fused-MBConv
          - Only using 3x3 kernel size 
          - expand-ratio [1, 4, 6] 
          - removing last stride-1 stage(stage8 in efficientNet_V1)
2. Progressive Learning:
          - increasing image size during training 
          - increasing stronger regularization(dropout, augmentation, mixup)
"""

# Refs:
"""
1. https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
2. https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/Test11_efficientnetV2
3. original paper
"""
def _make_divisible(v, divisor, min_value=None):
    """
    It ensures that all layers have a channel number that is divisible by 8
    In practice, the channels are a little different with the paper. For efficientnetv2-s: 272 --> 256, 1792 ---> 1280
    refs: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class StochasticDepth(nn.Module):
    def __init__(self,
                drop_prob):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        survival_prob = 1 - self.drop_prob
        if not self.training:
            return x
        
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < survival_prob
        return torch.div(x, survival_prob) * binary_tensor


class CNNBlock(nn.Module):
    def __init__(self, 
                in_channels: int, 
                out_channels: int, 
                kernel_size: int, 
                stride: int, 
                padding: int, 
                groups:int = 1, 
                norm_layer: Optional[Callable[..., nn.Module]] = None, 
                activation_layer: Optional[Callable[..., nn.Module]] = None
                ):
        super(CNNBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU
        self.cnn = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups
        )
        self.bn = norm_layer(out_channels)
        self.activation = activation_layer()
    
    def forward(self, x):
        return self.activation(self.bn(self.cnn(x)))


class SqueezeExciation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        expanded_channels: int,
        se_ratio: float = 0.25
        ):
        super(SqueezeExciation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, _make_divisible(in_channels*se_ratio, 8), 1),
            nn.SiLU(),
            nn.Conv2d(_make_divisible(in_channels*se_ratio, 8), expanded_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.se(x) * x

        

class MBConvBlock(nn.Module):
    def __init__(self, 
                in_channels: int,
                out_channels: int,
                drop_connect_rate: float,
                expand_ratio: int,
                kernel_size: int,
                stride: int,
                norm_layer: Optional[Callable[..., nn.Module]] = None,
                padding: int =1,
                se_ratio: int =0.25):
        super(MBConvBlock, self).__init__()
        if stride not in [1, 2]:
            raise ValueError("illegal stride value, must be 1 or 2.")
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = (in_channels == out_channels and stride == 1)
        # assert expand_ratio != 1
        self.expanded_channels = in_channels * expand_ratio
        
        # increase dim(channels), point-wise conv
        self.expand_cnn = CNNBlock(
            in_channels=in_channels,
            out_channels=self.expanded_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=norm_layer
        )
        # depth-wise conv
        self.dw_cnn = CNNBlock(
            self.expanded_channels,
            self.expanded_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            groups=self.expanded_channels,
            norm_layer=norm_layer
        )
        # se
        self.se = SqueezeExciation(
            in_channels,
            self.expanded_channels,
            se_ratio
        )
        # project-conv
        self.project_cnn = CNNBlock(
            self.expanded_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=1,
            norm_layer=norm_layer,
            activation_layer=nn.Identity
        )
    
    # stochastic_depth
    def stochastic_depth(self, x):
        if not self.training:
            return x
        
        survival_prob = 1 - self.drop_connect_rate
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < survival_prob
        return torch.div(x, survival_prob) * binary_tensor

    def forward(self, x):
        x = self.expand_cnn(x)
        x = self.dw_cnn(x)
        x = self.se(x)
        x = self.project_cnn(x)
        if self.use_residual:
            return (self.stochastic_depth(x) + x)
        else:
            return x
            

class FuseMBConvBlock(nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                drop_connect_rate: float,
                expand_ratio: int,
                kernel_size: int,
                stride: int,
                norm_layer: Optional[Callable[..., nn.Module]] = None,
                padding: int =1):
        super(FuseMBConvBlock, self).__init__()
        if stride not in [1, 2]:
            raise ValueError("illegal stride value, must be 1 or 2.")
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = (in_channels == out_channels and stride == 1)
        self.has_expansion = (expand_ratio!=1)
        expanded_channels = in_channels * expand_ratio

        # when expansion != 1: expand_conv --> project_conv
        if self.has_expansion:
            self.expansion_cnn = nn.Sequential(
                # Expand_Conv
                CNNBlock(
                    in_channels=in_channels,
                    out_channels=expanded_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1,
                    norm_layer=norm_layer
                ),
                # project_conv
                CNNBlock(
                    in_channels=expanded_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_layer=norm_layer,
                    activation_layer=nn.Identity
                )
            )
        else:
            self.without_expansion_cnn = CNNBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
                norm_layer=norm_layer
            )
    
    def stochastic_depth(self, x):
        if not self.training:
            return x
    
        survival_prob = 1 - self.drop_connect_rate
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < survival_prob
        return torch.div(x, survival_prob) * binary_tensor
    
    
    def forward(self, x):
        if self.has_expansion:
            x = self.expansion_cnn(x)
        else:
            x = self.without_expansion_cnn(x)
        
        if self.use_residual:
            return self.stochastic_depth(x) + x
        else:
            return x

        
class EfficientNetV2(nn.Module):
    def __init__(self, 
                cfgs, 
                num_classes,
                num_features=1280, # according the source code
                drop_connect_rate=0.2,
                dropout_rate=0.2
                ):
        super(EfficientNetV2, self).__init__()
        self.cfgs = cfgs
        self.dropout_rate = dropout_rate
        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)
        self.pool =  nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
                        nn.Dropout(self.dropout_rate),
                        nn.Linear(num_features, num_classes)
              )

        # build first layer
        in_channels = self.cfgs[0][4]
        self.stem = CNNBlock(
            in_channels=3,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_layer=norm_layer
        )

        # build backbones
        blocks = []
        sum_blocks = sum(i[0] for i in self.cfgs)
        block_id = 0
        for cfg in self.cfgs:
            assert len(cfg) == 7
            repeats = cfg[0]
            operator = FuseMBConvBlock if cfg[-2]==0 else MBConvBlock
            for i in range(repeats):
                blocks.append(operator(
                    in_channels=cfg[4] if i==0 else cfg[5], # for repeated layers, first inputs received from the upper stage
                    out_channels=cfg[5],
                    expand_ratio=cfg[3],
                    kernel_size=cfg[1],
                    stride=cfg[2] if i==0 else 1, # for repeated stages, only use stride in the first layer
                    drop_connect_rate= drop_connect_rate * block_id /sum_blocks,
                    norm_layer=norm_layer
                ))
                block_id += 1
                in_channels = cfg[5]
        
        self.blocks = nn.Sequential(*blocks)
        
        # build the last several layers
        in_channels = self.cfgs[-1][5]
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(num_features),
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x)
        return self.classifier(x.view(x.shape[0], -1))

        
def efficientnet_v2_s(num_classes):

    # repeat, kernel, stride, expansion, in_c, out_c, operator(0:fuse, 1:mbconv)
    model_config = [[2, 3, 1, 1, 24, 24, 0],
                    [4, 3, 2, 4, 24, 48, 0],
                    [4, 3, 2, 4, 48, 64, 0],
                    [6, 3, 2, 4, 64, 128, 1],
                    [9, 3, 1, 6, 128, 160, 1],
                    [15, 3, 2, 6, 160, 256, 1]]
    model = EfficientNetV2(cfgs=model_config, num_classes=num_classes, dropout_rate=0.2)
    return model


def efficientnet_v2_m(num_classes):
    # repeat, kernel, stride, expansion, in_c, out_c, operator(0:fuse, 1:mbconv)
    model_config = [
        [3, 3, 1, 1, 24, 24, 0],
        [5, 3, 2, 4, 24, 48, 0],
        [5, 3, 2, 4, 48, 80, 0],
        [7, 3, 2, 4, 80, 160, 1],
        [14, 3, 1, 6, 160, 176, 1],
        [18, 3, 2, 6, 176, 304, 1],
        [5, 3, 1, 6, 304, 512, 1]
    ]
    model = EfficientNetV2(cfgs=model_config, num_classes=num_classes, dropout_rate=0.2)
    return model


def efficientnet_v2_l(num_classes):
    # repeat, kernel, stride, expansion, in_c, out_c, operator(0:fuse, 1:mbconv)
    model_config = [
        [4, 3, 1, 1, 32, 32, 0],
        [7, 3, 2, 4, 32, 64, 0],
        [7, 3, 2, 4, 64, 96, 0],
        [10, 3, 2, 4, 96, 192, 1],
        [19, 3, 1, 6, 192, 224, 1],
        [25, 3, 2, 6, 224, 384, 1],
        [7, 3, 1, 6, 384, 640, 1]
    ]
    model = EfficientNetV2(cfgs=model_config, num_classes=num_classes, dropout_rate=0.4)
    return model


def test():
    num_classes = 10
    num_examples = 5
    img_size = 300

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Your device is {device}")
    x = torch.randn((num_examples, 3, img_size, img_size)).to(device)
    model = efficientnet_v2_m(num_classes).to(device)
    print(model(x).shape)


test()

