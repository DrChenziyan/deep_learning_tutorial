#! -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.functional import pad
import torch.nn.functional as F
from torch.nn.modules import padding
from torch.nn.modules.distance import PairwiseDistance
from ViT import *

"""
the code is referred from the official code https://github.com/pengzhiliang/Conformer

In ViT, I have already defined the Attention and MLP class.
In this script, I should define the ConvBottleNeck and FCU upsampling and downsampling class, \
    then concatenating them into Conformer class.

date: 2021-8-19
"""

class ConvBlock(nn.Module):
    """
    CNN bottleneck block
    """
    def __init__(self,
                in_planes, out_planes, stride=1, groups=1,
                down_sampling=False, # do the conv and batchnorm for the residual part
                drop_path_ratio=.0,
                norm_layer: Optional[Callable[..., nn.Module]]=None,
                act_layer: Optional[Callable[..., nn.Module]]=None,
                drop_block:Optional[Callable[..., nn.Module]]=None,
                ):
        super(ConvBlock, self).__init__()
        
        expansion=4, # out_planes dims are 4 times of med_planes

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU
        # Herein, if the out_planes is 256, med_planes is 64 which is equal to in_planes in this paper.
        med_planes = out_planes // expansion    
        self.drop_path_ratio = drop_path_ratio
        
        self.conv1 = nn.Conv2d(in_planes, med_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=True)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(out_planes)
        self.act3 = act_layer(inplace=True)
        
        self.drop_path = DropPath(drop_prob=drop_path_ratio)
        self.drop_block = drop_block
        self.down_sampling = down_sampling
        
        # down-sampling
        if self.down_sampling:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False),
                norm_layer(out_planes)
            )

    def forward(self, x):
        shortcut = x
        
        x = self.conv1(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        

        if self.drop_path_ratio>0:
            x = self.drop_path(x)
        
        if self.down_sampling:
            shortcut = self.shortcut(x)

        x += shortcut
        x = self.act3(x)
        return x

class FcuDown(nn.Module):
    """
    CNN feature maps -> Transformer patch embeddings
    Down_Sampling module: 1 x 1 Conv --> Average Pooling --> Layer Norm --> activation
    """
    def __init__(self, in_planes, out_planes, 
                dw_stride,  # down_sampling the spatial size through average pooling
                norm_layer: Optional[Callable[..., nn.Module]]=None,
                act_layer: Optional[Callable[..., nn.Module]]=None
                ):
        super(FcuDown, self).__init__()
        self.dw_stride = dw_stride

        if norm_layer is None:
            norm_layer = nn.LayerNorm
        if act_layer is None:
            act_layer = nn.GELU

        self.conv_proj = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)
        
        self.norm = norm_layer(out_planes)
        self.act = act_layer()

    def forward(self, x, cls_token):
        # input: [B, Cin, H, W]
        # conv_proj: [B, embed_dim, H, W]
        # pooling: [B, embed_dim, H', W']
        x = self.conv_proj(x)
        # flatten: [B, embed_dim, H'W']
        # transpose: [B, H'W'(num_patches), embed_dim]
        x = self.pooling(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.act(x)

        # concat with cls token
        x = torch.cat([cls_token[:,0][:, None, :], x], dim=1)
        return x

class FcuUp(nn.Module):
    """
    Transformer patch embedding --> CNN feature maps
    Up_Sampling module: 1 x 1 Conv --> F.interpolate (up samples the spatial size)
    """ 
    def __init__(self, in_planes, out_planes,
                up_stride, 
                norm_layer: Optional[Callable[..., nn.Module]]=None,
                act_layer: Optional[Callable[..., nn.Module]]=None
                ):
        super(FcuUp, self).__init__()
        self.up_stride = up_stride

        if norm_layer is None:
            norm_layer = nn.LayerNorm
        if act_layer is None:
            act_layer = nn.GELU

        self.conv_proj = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = norm_layer(out_planes)
        self.act = act_layer()

    def forward(self, x, W, H):
        # input: [B, num_patches+1, embeded_dim]
        B, N, C = x.shape
        assert N == W * H, f'Wrong {W} or/and {H}, please check!'
        x_r = x[:,1:, :].transpose(1, 2).reshape(B, C, W, H)
        # conv_proj: [B, embed_dim, W', H'] --> [B, C(conv), W', H']
        x_r = self.conv_proj(x_r)
        x_r = self.norm(x_r)
        x_r = self.act(x_r)
        # interpolate: [B, C, H, W]
        x_r = F.interpolate(x_r, size=(self.up_stride * H, self.up_stride * W))
        return x_r

    
class Med_ConvBlock(nn.Module):
    """ special case for down sampling
    """
    pass


class ConvTransBlock(nn.Module):
    def __init__(self, in_planes,
                out_planes, # CNN out_planes
                fcu_stride,
                stride, down_sampling,
                embed_dim, num_heads,
                mlp_ratio=4, qkv_bias=False, qk_scale=.0,
                drop_out_ratio=.0,
                attn_drop_ratio=.0,
                drop_path_ratio=.0,
                groups=1,
                last_fusion=False,
                norm_layer: Optional[Callable[..., nn.Module]]=None,
                act_layer: Optional[Callable[..., nn.Module]]=None
                ):
        super(ConvTransBlock, self).__init__()
        expansion:int = 4
        self.embed_dim = embed_dim
        self.fcu_stride = fcu_stride
        self.stride = stride
        self.last_fusion = last_fusion
        
        self.cnn_block = ConvBlock(in_planes, out_planes, down_sampling=down_sampling, 
                                    stride=stride, groups=groups, drop_path_ratio=drop_path_ratio)
        # todo: know last_fusion
        if last_fusion:
            self.fusion_block = ConvBlock(out_planes, out_planes, stride=2, down_sampling=True, groups=groups)
        else:
            self.fusion_block = ConvBlock(out_planes, out_planes, groups=groups)
        
        # Down sampling, the input should be the out_planes//expansion,  
        self.squeeze_block = FcuDown(in_planes=out_planes//expansion, out_planes=embed_dim, dw_stride=fcu_stride)
        # Up sampling
        self.expand_block = FcuUp(in_planes=embed_dim, out_planes=out_planes//expansion, up_stride=fcu_stride)

        # attention
        self.attn_bolck = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop_path_prob=drop_path_ratio, attn_drop_ratio=attn_drop_ratio,
            dropout_ratio=drop_out_ratio
        )
    
    def forward(self, x, x_t):
        pass
    pass


class Conformer(nn.Module):
    pass



        

