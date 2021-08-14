# ! -*- coding: utf-8 -*-
import os
import sys
from functools import partial, total_ordering
from typing import Callable, List, Optional
import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.nn.init import normal_
from torch.nn.modules import activation
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.flatten import Flatten
from torch.nn.modules.module import Module
from torch.nn.modules.pooling import AdaptiveAvgPool2d
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)
# from EfficientNet.efficientnet_V2 import _make_divisible


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# Bottleneck block definition
class BottleNeckBlock(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                groups=1,
                norm_layer: Optional[Callable[..., nn.Module]] = None,
                activation_layer: Optional[Callable[..., nn.Module]] =None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(BottleNeckBlock, self).__init__()
        self.bneck = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False
            ),
            norm_layer(out_channels),
            activation_layer()
        )

    def forward(self, x):
        return self.bneck(x)

# Squeeze-and-Excite block definition
class SqueezeExciation(nn.Module):
    def __init__(self,
                in_channels,
                se_ratio=0.25
                ):
        super(SqueezeExciation,self).__init__()
        expand_channels = _make_divisible(in_channels*se_ratio, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, expand_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_channels, in_channels, 1),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x):
        return x * self.se(x)


# Customize the config list of inverted residual block
class InvertedResidualConfig:
    def __init__(self,
                in_planes,
                kernel,
                expand_c,
                out_planes,
                use_se,
                activation,
                stride,
                width_multi):
        self.in_channels = self.adjust_channels(in_planes, width_multi)
        self.out_channels = self.adjust_channels(out_planes, width_multi)
        self.kernel = kernel
        self.stride = stride
        self.expand_c = self.adjust_channels(expand_c,width_multi)
        self.use_se = use_se
        self.use_hs = (activation == 'HS') # HS h-swish activation

    @staticmethod
    def adjust_channels(channels, width_multi):
        return _make_divisible(channels * width_multi, 8)


# expand_conv --> dw_conv --> projection_conv
class InvertedResidualBlock(nn.Module):
    def __init__(self,
                cfg: InvertedResidualConfig,
                norm_layer:Callable[..., nn.Module]):
        super(InvertedResidualBlock, self).__init__()
        if cfg.stride not in [1, 2]:
            raise ValueError("Illegal stride value, must be 1 or 2!")
        self.use_residual = (cfg.in_channels == cfg.out_channels and cfg.stride == 1) # apply short-cut when applying se-block
        
        features :List[nn.Module]= []
        activation_layer = nn.Hardswish if cfg.use_hs else nn.ReLU 

        # expand_conv
        if cfg.expand_c != cfg.in_channels:
            features.append(
                BottleNeckBlock(
                    in_channels=cfg.in_channels,
                    out_channels=cfg.expand_c,
                    kernel_size=1,
                    activation_layer=activation_layer
                )
            )
        # dw_cov
        features.append(
            BottleNeckBlock(
                cfg.expand_c,
                cfg.expand_c,
                cfg.kernel,
                cfg.stride,
                groups=cfg.expand_c,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )
        )
        # squeeze-and-excitation
        if cfg.use_se:
            features.append(SqueezeExciation(cfg.expand_c))
        
        # project_cov
        features.append(
            BottleNeckBlock(
                cfg.expand_c,
                cfg.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Identity
            )
        )

        self.block = nn.Sequential(*features)
        self.out_channels = cfg.out_channels
        self. is_strided = (cfg.stride>1)

    def forward(self, x):
        results = self.block(x)
        if self.use_residual:
            return results + x
        else:
            return results


class MobileNet_V3(nn.Module):
    def __init__(self,
                inverted_block_setting: List[InvertedResidualConfig],
                last_channels,
                num_classes,
                drop_out_rate=0.2,
                block:Optional[Callable[..., nn.Module]] =None,
                norm_layer:Optional[Callable[..., nn.Module]] =None,
                ):
        super(MobileNet_V3, self).__init__()
        
        if not inverted_block_setting:
            raise ValueError("The inverted block setting must not be empty")
        elif not(isinstance(inverted_block_setting, List)) and \
            all([isinstance(i, InvertedResidualConfig) for i in inverted_block_setting]):
            raise ValueError("The tpye of inverted block must be List(InvertedResidualConfig)")

        if block is None:
            block = InvertedResidualBlock

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.9)

        features = []

        # build the first layer
        firstconv_out_c = inverted_block_setting[0].out_channels
        features.append(
            BottleNeckBlock(
                3, 
                firstconv_out_c,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish
            )
        )
        # build the block layers
        for cfg in inverted_block_setting:
            features.append(InvertedResidualBlock(cfg, norm_layer))

        # build the last several layers
        lastconv_in_c = inverted_block_setting[-1].out_channels
        lastconv_out_c = 6 * lastconv_in_c # for small and large mobilenet, the out channels are 6 times for in channels
        features.append(
            BottleNeckBlock(
                lastconv_in_c,
                lastconv_out_c,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish)
            )
        self.blocks = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_out_c, last_channels),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=drop_out_rate, inplace=True),
            nn.Linear(last_channels, num_classes)
        )

        # initial weights
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d,nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        
    
    def _foward_impl(self, x):
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._foward_impl(x)
    

def mobilenet_v3_large(num_classes, width_multi=1.0, reduced_tail=False):
    """
    Args:
    num_classes(int): the classes that you want to classifier.
    width_multi(float): the hyper-parameter that modify the channels number. The default is 1.0
    reduced_tail(bool): A boolean value that reduces the channels counts of all feature layers \
        which would reduce the channel redundancy and be used in the backbone of detectation and segmentation if True.
    """
    width_multi = width_multi
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    # from pytorch official
    reduce_div = 2 if reduced_tail else 1

    inverted_block_setting = [
        # input_c, kernel, expand_c, output_c, use_se, NL, stride --(from paper)
        bneck_conf(16, 3, 16, 16, False, 'RE', 1),
        bneck_conf(16, 3, 64, 24, False, 'RE', 2),
        bneck_conf(24, 3, 72, 24, False, 'RE', 1),
        bneck_conf(24, 5, 72, 40, True, 'RE', 2),
        bneck_conf(40, 5, 120, 40, True, 'RE', 1),
        bneck_conf(40, 5, 120, 40, True, 'RE', 1),
        bneck_conf(40, 3, 240, 80, False, 'HS', 2),
        bneck_conf(80, 3, 200, 80, False, 'HS', 1),
        bneck_conf(80, 3, 184, 80, False, 'HS', 1),
        bneck_conf(80, 3, 184, 80, False, 'HS', 1),
        bneck_conf(80, 3, 480, 112, True, 'HS', 1),
        bneck_conf(112, 3, 672, 112, True, 'HS', 1),
        bneck_conf(112, 5, 672, 160, True, 'HS', 2),
        bneck_conf(160 // reduce_div, 5, 960 // reduce_div, 160 // reduce_div, True, 'HS', 1),
        bneck_conf(160 // reduce_div, 5, 960 // reduce_div, 160 // reduce_div, True, 'HS', 1), 
    ]

    last_channels = adjust_channels(1280 // reduce_div)
    
    model = MobileNet_V3(inverted_block_setting, last_channels, num_classes=num_classes, drop_out_rate=0.4)
    return model


def mobilenet_v3_small(num_classes, width_multi=1.0, reduced_tail=False):
    
    width_multi = width_multi
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_div = 2 if reduced_tail else 1

    inverted_block_setting = [
        # input_c, kernel, expand_c, output_c, use_se, NL, stride --(from paper)
        bneck_conf(16, 3, 16, 16, True, 'RE', 2),
        bneck_conf(16, 3, 72, 24, False, 'RE', 2),
        bneck_conf(24, 3, 88, 24, False, 'RE', 2),
        bneck_conf(24, 5, 96, 40, True, 'HS', 2),
        bneck_conf(40, 5, 240, 40, True, 'HS', 1),
        bneck_conf(40, 5, 240, 40, True, 'HS', 1),
        bneck_conf(40, 5, 120, 48, True, 'HS', 1),
        bneck_conf(48, 5, 144, 48, True, 'HS', 1),
        bneck_conf(48, 5, 288, 96, True, 'HS', 2),
        bneck_conf(96 // reduce_div, 5, 576 // reduce_div, 96 // reduce_div, True, 'HS', 1),
        bneck_conf(96 // reduce_div, 5, 576 // reduce_div, 96 // reduce_div, True, 'HS', 1), 
    ]

    last_channels = adjust_channels(1024 // reduce_div)
    
    model = MobileNet_V3(inverted_block_setting, last_channels, num_classes=num_classes, drop_out_rate=0.2)
    return model


def get_model_parameters(model:nn.Module):
    total_params = 0
    for layer in list(model.parameters()):
        layer_parameters = 1
        for l in list(layer.size()):
            layer_parameters *= l
        total_params += layer_parameters
    return total_params



def test():
    small_net = mobilenet_v3_small(num_classes=100,width_multi=1.5)
    num_examples = 5
    img_size = 300
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Your device is {device}")
    total_small_params = get_model_parameters(small_net)
    total_large_params = get_model_parameters(mobilenet_v3_large(num_classes=100))
    print(f"Parameters of small model are {total_small_params}.")
    print(f"Parameters of large model are {total_large_params}.")

    x = torch.randn((num_examples, 3, img_size, img_size)).to(device)
    model = small_net.to(device)
    print(model(x).shape)

test()