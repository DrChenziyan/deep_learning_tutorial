# ! -*- coding: utf-8 -*-
from itertools import repeat
import torch
from torch import nn
import math
from torch._C import INSERT_FOLD_PREPACK_OPS
from torch.nn.modules import dropout, padding

from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.linear import Linear

# ref:https://www.youtube.com/watch?v=fR_0o25kigM&t=418s


base_model = [
          # list: [expend_ratio, channels, repeats, stride, kernel_size]
          [1, 16, 1, 1, 3],
          [6, 24, 2, 2, 3],
          [6, 40, 2, 2, 5],
          [6, 80, 3, 2, 3],
          [6, 112, 3, 1, 5],
          [6, 192, 4, 2, 5],
          [6, 320, 1, 1, 3]
]

phi_values = {
          # tuple: (phi, resolution, drop_rate)
          # search by google
          "b0": (0, 224, 0.2),
          "b1": (0.5, 240, 0.2),
          "b2": (1, 260, 0.3),
          "b3": (2, 300, 0.3),
          "b4": (3, 380, 0.4),
          "b5": (4, 456, 0.4),
          "b6": (5, 528, 0.5),
          "b7": (6, 600, 0.5)
}


class CNNBlock(nn.Module):
          def __init__(self, in_channels, out_channels, kernel_size, strides, paddings, groups=1):
              super(CNNBlock, self).__init__()
              self.cnn = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=strides,
                    padding=paddings,
                    # depthwise conv
                    groups=groups
              )
              self.bn = nn.BatchNorm2d(out_channels)
              self.SiLu = nn.SiLU()
          
          
          def forward(self, x):
                    x = self.cnn(x)
                    x = self.bn(x)
                    return self.SiLu(x)  


class SqueezeExcitation(nn.Module):
          def __init__(self, in_channels, reduced_dim):
              super(SqueezeExcitation, self).__init__()
              self.se = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), # C x H x W --> C x 1 x 1
                    nn.Conv2d(in_channels, reduced_dim, 1),
                    nn.SiLU(), # torch version >= 1.7.0
                    nn.Conv2d(reduced_dim, in_channels, 1),
                    nn.Sigmoid()

              )
          
          def forward(self, x):
                    return x * self.se(x)



class InvertedResidualBlock(nn.Module):
          def __init__(self, 
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    strides, 
                    paddings, 
                    expand_ratio, 
                    reduction=4, # squeeze excitation
                    survival_prob = 0.8 # for stochastic depth
          ):
              super(InvertedResidualBlock, self).__init__()
              self.survival_prob = survival_prob
              self.use_residual = in_channels == out_channels and strides == 1
              hidden_dim = in_channels * expand_ratio
              self.expand = in_channels != hidden_dim
              reduced_dim = int(in_channels / reduction)
              
              if self.expand:
                    # expansion_layer
                    self.expand_conv = CNNBlock(
                              in_channels,
                              hidden_dim,
                              kernel_size=3,
                              strides=1,
                              paddings=1
                    )
              # depthwise conv
              self.conv = nn.Sequential(
                        CNNBlock(hidden_dim, hidden_dim, kernel_size, strides, paddings, groups=hidden_dim),
                        SqueezeExcitation(hidden_dim, hidden_dim),
                        nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                        nn.BatchNorm2d(out_channels),

              )

          def stochastic_depth(self, x):
                    if not self.training:
                              return x

                    binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
                    return torch.div(x, self.survival_prob) * binary_tensor

          
          def forward(self, inputs):
                    x = self.expand_conv(inputs) if self.expand else inputs

                    if self.use_residual:
                              return self.stochastic_depth(self.conv(x)) + inputs
                    else:
                              return self.conv(x)

     

class EfficientNet(nn.Module):
          def __init__(self, version, num_classes):
              super(EfficientNet, self).__init__()
              width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
              last_channels = math.ceil(1280 * width_factor)
              self.pool = nn.AdaptiveAvgPool2d(1)
              self.features = self.create_features(width_factor, depth_factor, last_channels)
              self.classifier = nn.Sequential(
                        nn.Dropout(dropout_rate),
                        nn.Linear(last_channels, num_classes)
              )

          def calculate_factors(self, version, alpha=1.2, beta=1.1):
                    phi, res, drop_rate = phi_values[version]
                    depth_factor = alpha ** phi
                    width_factor = beta ** phi
                    return width_factor, depth_factor, drop_rate

          def create_features(self, width_factor, depth_factor, last_channels):
                    channels = int(32 * width_factor)
                    features = [CNNBlock(3, channels, 3, strides=2, paddings=1)]
                    in_channels = channels


                    for expand_ratio, channels, repeats, stride, kernel_size in base_model:
                              out_channels = 4 * math.ceil(int(channels * width_factor) / 4)
                              layer_repeats = math.ceil(repeats * depth_factor)

                              for layer in range(layer_repeats):
                                        features.append(
                                                  InvertedResidualBlock(
                                                            in_channels,
                                                            out_channels,
                                                            expand_ratio=expand_ratio,
                                                            strides = stride if layer == 0 else 1,
                                                            kernel_size = kernel_size,
                                                            paddings= kernel_size // 2  # if k==1, pad=0; if k==3, pad=1; if k==5, pad=2
                                                  )
                                        )
                                        in_channels = out_channels
                    features.append(
                              CNNBlock(
                                        in_channels,
                                        last_channels,
                                        kernel_size=1,
                                        strides=1,
                                        paddings=0
                              )

                    )
                    return nn.Sequential(*features)

          def forward(self, x):
                    x = self.pool(self.features(x))
                    return self.classifier(x.view(x.shape[0], -1))



def test():
          device = "cuda" if torch.cuda.is_available() else 'cpu'
          version = 'b0'
          phi, res, drop_rate = phi_values[version]
          num_examples, num_classes = 4, 5
          x = torch.randn((num_examples, 3, res, res)).to(device)
          model = EfficientNet(
                    version,
                    num_classes
          ).to(device)

          print(model(x).shape)

test()