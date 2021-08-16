# ! -*- coding: utf-8 -*-
from math import e
import torch
from torch._C import R
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn.modules.module import T
import torch.utils.checkpoint as checkpoint
from ViT import *

# ref: https://github.com/microsoft/Swin-Transformer

"""
1. Def window_partition and window_reverse func:
    - window_partition: partation an image into several windows
    - reverse op of window_partation.
2. Def WindowAttention class
3. Def MLP class(also import from ViT.py)
4. Def SwinTransformerBlock class
5. Def SwinTransformer class 
6. Pass the cfg and inistance the class
"""


def window_partition(x, window_size):
    """
    Split an image into several windows.
    Args:
        x:(B, H, W, C)
        window_size(int): window size
    Return:
        windows: (B, num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse function of window_partition
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    Implement W-MSA with relative position bias which supports shifted and non-shifted window.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size:tuple,
                 qkv_bias: bool = False,
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
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)), num_heads
        )  # (2, Wh-1, Ww-1, num_heads)

        # 2. get pair-wise relative position
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # [2, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1)  # [2, Wh * Ww]
        # expand dimension and minus
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Wh * Ww, Wh * Ww]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Wh * Ww, Wh * Ww, 2]
        # start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        # differentiate `x` and `y`
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Wh * Ww, Wh * Ww]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop_ratio)

    def forward(self, x, mask=None):
        # [B, num_patches+1(N), total_embedded_dim]
        B, N, C = x.shape
        # to_qkv
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # shape:[B, num_heads, N, embedded_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # msa
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        # w-mas: qk+bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0], self.window_size[1], -1
        )  # [Wh*Ww(N), Wh*Ww, num_heads]
        relative_position_bias = relative_position_bias.permute(2, 0, 1)  # [num_heads, N, N]
        attn = attn + relative_position_bias.unsqueeze(0)  # [B, num_heads, N, N]

        # shift window by applying a mask
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N) # [B, num_heads, N, N]
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        
        attn = self.attn_drop(attn)
        res = (attn @ v).transpose(1, 2).reshape(B, N, C)
        res = self.proj(res)
        return self.proj_drop(res)

    def flops(self, N):
        """Calculate flops for 1 window with token length of N
        """
        flops = 0
        # to_qkv
        flops += N * self.dim * 3 * self.dim
        # attn -- qk
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        # qk @ v
        flops += self.num_heads * N * N (self.dim // self.num_heads)
        # proj
        flops += N * self.dim * self.dim
        return flops


class Block(nn.Module):
    def __init__(self,
                dim,
                input_resolution,
                num_heads,
                window_size=7,
                shift_size=0,
                mlp_ratio=4,
                qkv_bias: bool = False,
                qk_scale=None,
                attn_drop_ratio=0.,
                droppath_ratio=0.,
                dropout_ratio=0.,
                activation_layer:Optional[Callable[..., nn.Module]]=None,
                norm_layer:Optional[Callable[..., nn.Module]]=None
                ):
        super(Block, self).__init__()

        if activation_layer is None:
            activation_layer = nn.GELU()
        if norm_layer is None:
            norm_layer = nn.LayerNorm(dim)
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        # if input resolution smaller than window size, windows partition will be passed
        if min(self.input_resolution) <= self.shift_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0<= self.shift_size < self.window_size, 'shift window size must be small than window size'

        # Layer_norm1 --> w-msa 
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, num_heads, (window_size, window_size), qkv_bias, qk_scale, attn_drop_ratio, dropout_ratio)
        
        # drop_path
        if droppath_ratio > 0:
            self.drop_path = DropPath(drop_prob=droppath_ratio)
        else:
            self.drop_path = nn.Identity()

        # layer_norm --> mlp 
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(mlp_ratio*dim), dim, activation_layer, dropout_ratio=dropout_ratio)

        # layer_norm --> sw-msa
        # copy from the source code, need to understand more concretly!!!
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))    # [1, H, W, 1]
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x): # x.shape --> [B, HW, embeded_dim]
        H, W = self.input_resolution
        B, M, C = x.shape
        assert M == H*W, f'input shape must be {self.input_resolution}.'

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))
        else:
            shifted_x = x
        
        # partition window
        window_x = window_partition(shifted_x, self.window_size)    #[nW, window_size, window_size, C]
        window_x = window_x.view(-1, self.window_size*self.window_size, C)  #[nW, window_size*window_size, C]
        # w-msa or sw-msa
        attn_windows = self.attn(window_x, mask=self.attn_mask) #[nW, window_size*window_size, C]

        # merge window
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic window
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1,2))
        else:
            shifted_x = x
        x = x.view(B, M, C)

        # residual
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    """
    Patch Merging Layer. 
    """
    def __init__(self,
                input_resolution,
                dim,
                norm_layer: Optional[Callable[..., nn.Module]]=None
                ):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm = norm_layer()

    def forward(self, x):
        H, W = self.input_resolution
        B, M, C = x.shape
        assert M == H*W, f'input shape must be {self.input_resolution}.'
        assert H % 2 == 0 and W % 2 == 0, f'x size {H}x{W} are not even.'

        x = x.view(B, H, W, C)
        
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # concatenate the last dimension
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
    
    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    def __init__(self,
                dim,
                input_resolution,
                depth,
                num_heads,
                window_size,
                mlp_ratio=4,
                qkv_bias=False,
                qk_scale=0,
                attn_drop_ratio=0.,
                droppath_ratio=0.,
                dropout_ratio=0.,
                norm_layer:Optional[Callable[..., nn.Module]]=None,
                downsample=None,
                use_checkpoint=False,
                ):
        super(BasicLayer, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint


        # build blocks
        self.blocks = nn.ModuleList([
            Block(dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if(i%2==0)else window_size//2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio, 
                droppath_ratio=droppath_ratio,
                dropout_ratio=dropout_ratio, norm_layer=norm_layer)
            for i in range(depth)
        ])
    
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim, norm_layer)
        else:
            self.downsample is None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops
   

class SwinTransformer(nn.Module):
    def __init__(self,
                img_size=224,
                patch_size=4,
                in_planes=3,
                num_classes=1000,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 4],
                embed_layer=PatchEmbedding_2D,
                window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                dropout_ratio=0., attn_drop_ratio=0., droppath_ratio=0.1,
                norm_layer:Optional[Callable[..., nn.Module]]=None, 
                ape=False, patch_norm=True,
                use_checkpoint=False, **kwargs):
        super(SwinTransformer, self).__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        # patch_embeding
        self.patch_embeding = embed_layer(img_size, patch_size, in_planes, embed_dim)
        num_patches = self.patch_embedding.num_patches
        grid_size = self.patch_embeding.grid_size
        self.grid_size = grid_size

        # absolute position encoding
        if self.ape is True:
            self.absolute_position_encoding = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
            trunc_normal_(self.absolute_position_encoding, std=.02)
        
        # drop out
        self.pos_drop = nn.Dropout(p=dropout_ratio)

        dpr = [x.item() for x in torch.linspace(0, droppath_ratio, sum(depths))]

        # bcakbone blocks
        self.layers = nn.ModuleList()
        for layer_i in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** layer_i),
                input_resolution=(
                    self.grid_size[0] // (2 ** layer_i), self.grid_size[1] // (2 ** layer_i)
                ),
                depths = depths[layer_i], num_heads=num_heads[layer_i],
                window_size=window_size, mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop_ratio=attn_drop_ratio,
                droppath_ratio=dpr[sum(depths[:layer_i]): sum(depths[:layer_i+1])],
                dropout_ratio=dropout_ratio,norm_layer=norm_layer,
                downsample=PatchMerging if (layer_i < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)

        # last several layers
        self.norm_layer = norm_layer(self.num_features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # classification
        self.head = nn.Linear(self.num_features, self.num_classes) if self.num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    

    def _forward_impl(self, x):
        # [B, C, H, W] -> [B,N,embed_dim]
        x = self.patch_embeding(x)
        if self.ape:
            x = x + self.absolute_position_encoding(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_layer(x) # [B, N, embed_dim]
        x = self.pool(x.transpose(1, 2))
        x = torch.flatten(x,1)
        
        return x

    def forward(self, x):
        x = self._forward_impl(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embeding.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.grid_size[0] * self.grid_size[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops





