# ! -*- coding: utf-8 -*-
import torch
from torch._C import R
import torch.nn as nn
import math
from ViT import *

# ref: https://github.com/microsoft/Swin-Transformer

"""
1. Def window_partation and window_reverse func:
    - window_partation: partation an image into several windows
    - reverse op of window_partation.
2. Def WindowAttention class
3. Def MLP class(also import from ViT.py)
4. Def SwinTransformerBlock class
5. Def SwinTransformer class 
6. Pass the cfg and inistance the class
"""

def window_partation(x, window_size):
    """
    Split an image into several windows.
    Args:
        x:(B, H, W, C)
        window_size(int): window size
    Return:
        windows: (B, num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H//window_size, window_size, W//window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse function of window_partation
    """
    x = windows.view(0, H//window_size, W//window_size, window_size, window_size, -1)


class WindowAttention(nn.Module):
    """
    Implement W-MSA with relative position bias which supports shifted and non-shifted window.
    """
    def __init__(self,
                dim, 
                num_heads,
                window_size,
                qkv_bias:bool =False,
                qk_scale=None,
                attn_drop_ratio=0.,
                proj_drop_ratio=0.):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)

        # Generate the relative position index of each token in a window
        # 1. define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*self.window_size[0]-1) * (2*self.window_size[1]-1)), num_heads
        ) # (2, Wh-1, Ww-1, num_heads)

        # 2. get pair-wise relative position
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w])) # [2, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1) # [2, Wh * Ww]
        # expand dimension and minus
        relative_coords = coords_flatten[:,:,None] - coords_flatten[:,None,:] # [2, Wh * Ww, Wh * Ww]
        relative_coords = relative_coords.permute(1,2,0).contiguous()   # [Wh * Ww, Wh * Ww, 2]
        # start from 0
        relative_coords[:,:,0] += self.window_size[0] - 1
        relative_coords[:,:,1] += self.window_size[1] - 1
        # differentiate `x` and `y`
        relative_coords[:,:,1] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)   # [Wh * Ww, Wh * Ww]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop_ratio)

    def forward(self, x):
        # [B, num_patches+1(N), total_embeded_dim]
        B, N, C = x.shape
        # to_qkv
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # shape:[B, num_heads, N, embeded_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # msa
        attn = (q @ k.transpose(-2, -1)) * self.scale #[B, num_heads, N, N]
        # w-mas: qk+bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0]*self.window_size[1], self.window_size[0], self.window_size[1], -1
        ) # [Wh*Ww(N), Wh*Ww, num_heads]
        relative_position_bias = relative_position_bias.permute(2, 0, 1) # [num_heads, N, N]
        attn = attn + relative_position_bias.unsqueeze(0) # [B, num_heads, N, N]

        # shift window by applying a mask



class Block(nn.Module):
    pass


class SwinTransformer(nn.Module):
    pass

