# ! -*- coding: utf-8 -*-
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
from torch.functional import einsum
from einops import rearrange
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.module import T
# from ViT.Transformer.ViT import *
# import sys 
# sys.path.append(r'/Users/li_lab/Desktop/PythonApplication/deep_learning_tutorial/ViT/Transformer/ViT.py')
# Refs:
# https://github.com/slwang9353/MobileFormer
# https://github.com/ACheun9/Pytorch-implementation-of-Mobile-Former


# bottleneck_lite
class BottleNeckLite(nn.Module):
    def __init__(self, 
                in_planes, 
                expand_size, 
                out_planes, 
                kernel_size=3, 
                stride=1,
                groups=None,
                activation_layer:Optional[Callable[..., nn.Module]]=None,
                norm_layer:Optional[Callable[..., nn.Module]]=None
                ):
        super(BottleNeckLite, self).__init__()
        self.in_planes = in_planes
        self.expand_size = expand_size
        self.out_planes = out_planes
        padding = (kernel_size - 1) // 2
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
    
        self.bneck_lite = nn.Sequential(
            nn.Conv2d(in_channels=self.in_planes, out_channels=self.expand_size, 
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=self.in_planes), # groups=in_channels
            activation_layer(inplace=True),
            nn.Conv2d(in_channels=self.expand_size, out_channels=out_planes, kernel_size=1, stride=1),
            norm_layer(self.out_planes)
        )
    
    def forward(self, x):
        return self.bneck_lite(x)
    


# dynamic_Relu
class DyReLU(nn.Module):
    """
    the dynamic activation is defined as a function fθ(x)(x) with learnable parameters θ(x), which adapt to the input x.
    Relu: y = max{x, 0}
    general Relu: y = max{a_c^kx_c + b_c^k}
    dynamic_Relu: y = max{a_c^k(x)x_c + b_c^k(x)}

    ref: https://github.com/Islanna/DynamicReLU
    """
    def __init__(self, dim, expand_size, reduction=4, k=2):
        super(DyReLU, self).__init__()
        self.dim = dim
        self.k = k
        self.expand_size = expand_size

        self.fc1 = nn.Linear(dim, dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim // reduction, 2 * k * expand_size)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float())

    def get_relu_coefs(self, z):
        theta = z[:, 0, :]
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x, z):
        # theta shape [B, 2*dim]
        theta = self.get_relu_coefs(z)
        # relu_coefs shape [B, expand_size, 2*k]
        relu_coefs = theta.view(-1, self.expand_size, 2*self.k) * self.lambdas + self.init_v
        # BxCxHxW -> HxWxBxCx1
        x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
        # HxWxBxCx2 -> BxCxHxW
        result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

        return result
    

class Mobile(nn.Module):
    """
    if stride=2, do the downsampling: dw -> pw -> dw -> pw
    else: pw -> dw -> pw (invtertedResidualBlock)
    """
    def __init__(self, in_planes, expand_size, out_planes, 
                embeded_dim, kernel_size=3, stride=1, reduction=4, k=2):
        super(Mobile, self).__init__()
        self.in_planes = in_planes
        self.expand_size = expand_size
        self.out_planes = out_planes
        self.dim = embeded_dim
        self.stride = stride
        
        if self.stride == 2:
            # do the down-sampling
            padding = (kernel_size - 1) // 2
            self.down_smaple = nn.Sequential(
                nn.Conv2d(in_planes, expand_size, kernel_size=kernel_size, stride=self.stride, padding=padding, groups=in_planes),
                nn.BatchNorm2d(self.expand_size),
                nn.ReLU6(inplace=True)
            )
            self.conv_1 = nn.Conv2d(expand_size, in_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn_1 = nn.BatchNorm2d(self.in_planes)
            # self.act_1 = nn.ReLU6(inplace=True)
            self.act_1 = DyReLU(self.dim, in_planes)

            self.conv_2 = nn.Conv2d(in_planes, expand_size, kernel_size=kernel_size, stride=1, padding=padding, groups=in_planes, bias=False)
            self.bn_2 = nn.BatchNorm2d(expand_size)
            # self.act_2 = nn.ReLU6(inplace=True)
            self.act_2 = DyReLU(self.dim, expand_size)

            self.conv_3 = nn.Conv2d(expand_size, out_planes, kernel_size=1, stride=1, bias=False)
            self.bn_3 = BatchNorm2d(out_planes)

        
        else:
            padding = (kernel_size - 1) // 2
            self.conv_1 = nn.Conv2d(in_planes, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn_1 = nn.BatchNorm2d(expand_size)
            # self.act_1 = nn.ReLU6(inplace=True)
            self.act_1 = DyReLU(self.dim, expand_size)

            self.conv_2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=1, padding=padding, groups=expand_size, bias=False)
            self.bn_2 = nn.BatchNorm2d(expand_size)
            # self.act_2 = nn.ReLU6(inplace=True)
            self.act_2 = DyReLU(self.dim, expand_size)

            self.conv_3 = nn.Conv2d(expand_size, out_planes, kernel_size=1, stride=1, bias=False)
            self.bn_3 = nn.BatchNorm2d(out_planes)
    
    def forward(self, x, z):
        if self.stride == 2:
            x = self.down_smaple(x)
        x = self.bn_1(self.conv_1(x))
        x = self.act_1(x, z)
        x = self.bn_2(self.conv_2(x))
        x = self.act_2(x, z)
        return self.bn_3(self.conv_3(x))


class Attention(nn.Module):
    def __init__(self,
                dim,  # the token dim(embeded_dim)
                num_heads,
                qkv_bias=False,
                qk_scale=None,
                attn_drop_ratio=0.,
                proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads # Multi-head attention, mean divison
        self.scale = qk_scale or head_dim ** (-0.5)
        self.to_qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [B, num_patches+1, total_embeded_dim]
        B, N, C = x.shape
        # to_qkv: [B, N, C] --> [B, N, 3*C]
        # reshape: [B, N, 3*C] --> [B, N, 3, num_heads, embeded_dim_per_head]
        # permute(transpose): [B, N, 3, num_heads, embeded_dim_per_head] --> [3, B, num_heads, N, embeded_dim_per_head]
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q.shape == k.shape == v.shape
        # shape:[B, num_heads, N, embeded_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # q c-dot k.transpose()
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1) # do softmax per row
        attn = self.attn_drop(attn)
        # shape:[B, num_heads, N, N]

        # attn multiply v
        # @:[B, num_heads, N, N] @ [B, num_heads, N, embeded_dim_per_head] == [B, num_heads, N, embeded_dim_per_head]
        # transpose: [B, num_heads, N, embeded_dim_per_head] --> [B, N, num_heads, embeded_dim_per_head]
        # shape: [B, N, num_heads, embeded_dim_per_head] --> [B, N, C]
        res = (attn @ v).transpose(1, 2).reshape(B, N, C)
        res = self.proj(res)
        return self.proj_drop(res)


def stochastic_depth(x, drop_prob, training=False):
    """
    Apply stochastic depth function by StochasticDepth(DropPath) Class
    """ 
    if drop_prob == 0 or not training:
        return x
    survival_prob = 1 - drop_prob
    binary_tensor = torch.rand(x.shape[0], 1, 1, 1, dtype=x.dtype, device=x.device) < survival_prob
    return torch.div(x, survival_prob) * binary_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return stochastic_depth(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self,
                in_features, # input_features
                hidden_features, # 4 * input_features, which we could a use 'mlp-ratio' parameter in the "Block class" to achive 
                out_features=None,
                activation_layer:Optional[Callable[..., nn.Module]] =None,
                dropout_ratio=0.
                ):
        super(MLP, self).__init__()
        out_features = in_features if out_features is None else in_features
        hidden_features = hidden_features if hidden_features else in_features
        
        if activation_layer == None:
            activation_layer = nn.GELU
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            activation_layer(),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(p=dropout_ratio)
        )

    def forward(self, x):
        return self.mlp(x)

class Former(nn.Module):
    def __init__(self, 
                embeded_dim, num_heads, 
                mlp_ratio=2,    # Note that the expand_ratio is 2
                qkv_bias=False, qk_scale=None,
                drop_path_prob=0., attn_drop_ratio=0., dropout_ratio=0.,
                activation_layer:Optional[Callable[..., nn.Module]]=None,
                ):
        super(Former, self).__init__()

        if activation_layer is None:
            activation_layer = nn.GELU

        self.norm = nn.LayerNorm(embeded_dim)
        self.msa = Attention(embeded_dim, num_heads, qkv_bias, qk_scale, attn_drop_ratio, dropout_ratio)
        
        if drop_path_prob > 0:
            self.drop_path = DropPath(drop_prob=drop_path_prob)
        else:
            self.drop_path = nn.Identity()

        self.mlp = MLP(embeded_dim, int(mlp_ratio*embeded_dim), embeded_dim, activation_layer, dropout_ratio=dropout_ratio)

    def forward(self, x):
        x += self.drop_path(self.msa(self.norm(x)))
        x += self.drop_path(self.mlp(self.norm(x)))
        return x


class Mobile2Former(nn.Module):
    """ 
    local features --> global features
    input: x[B, C, H, W], z[B, M, d]
    output: z[B, M, d]

    """
    def __init__(self, num_heads, dim, channels,
                dropout_ratio=0.):
        super(Mobile2Former, self).__init__()
        inner_channels = num_heads * channels
        self.num_heads = num_heads
        self.dim = dim
        self.project_Q = nn.Linear(self.dim, inner_channels)
        self.project_out = nn.Sequential(
            nn.Linear(inner_channels, self.dim),
            nn.Dropout(p=dropout_ratio)
        )
        self.scale = channels ** (-0.5)
        self.shortcut = nn.Identity()
    
    def forward(self, x, z):
        B, M, d= z.shape
        B, C, H, W = x.shape
        shortcut = self.shortcut(z)
        # x shape change [B, C, H, W] --> [B, C, HW] --> [B, C, HW]
        # unsqueeze: expand the second dimension of x: [B, C, HW] --> [B, 1, C, HW]
        x = x.contiguous().view(B, C, H*W).unsqueeze(1)
        # z shape [B, M, d] --> [B, M, inner_channels] -> [B, num_heads, M, C]
        z = self.project_Q(z).view(B, self.num_heads, M, C)
        # [B, num_heads, M, C] @ [B, num_heads, C, HW] = [B, num_heads, M, HW]
        attn = z @ x * self.scale
        attn = attn.softmax(dim=-1)
        # attn @ x
        # transpose: [B, 1, HW, C]
        # [B, num_heads, M, HW] @ [B, 1, HW, C] = [B, num_heads, M, C]
        # premute and view:[B, num_heads, M, C] --> [B, M, num_heads, C] --> [B, M, inner_channels]
        res = attn @ x.transpose(-2, -1)
        res = res.permute(0, 2, 1, 3).reshape(B, M, -1)
        # [B, M, inner_channels] --> [B, M, d]
        res = self.project_out(res)
        return res + shortcut



class Former2Mobile(nn.Module):
    """
    global features to local features
    input: x[B, C, H, W], z[B, M, d]
    output: x[B, C, H, W]
    """
    def __init__(self, num_heads, dim, channels,
                dropout_ratio=0.):
        super(Former2Mobile, self).__init__()
        self.channels = channels
        inner_channels = num_heads * channels
        self.num_heads = num_heads
        self.dim = dim
        self.project_K = nn.Linear(self.dim, inner_channels)
        self.project_V = nn.Linear(self.dim, inner_channels)
        self.project_out = nn.Sequential(
            nn.Linear(inner_channels, channels),
            nn.Dropout(p=dropout_ratio)
        )
        self.scale = channels ** (-0.05)
        self.shortcut = nn.Identity()

    def forward(self, x, z):
        B, M, d= z.shape
        B, C, H, W = x.shape
        shortcut = self.shortcut(x)

        # x shape change [B, C, H, W] --> [B, C, HW] --> [B, HW, C]
        # unsqueeze: expand the second dimension of x: [B, HW, C] --> [B, 1, HW, C]
        x = x.contiguous().view(B, C, H*W).transpose(-2, -1).unsqueeze(1)
        # [B, M, d] --> [B, M, inner_channels] --> [B, num_heads, M, C]
        k = self.project_K(z).view(B, self.num_heads, M, C)
        v = self.project_V(z).view(B, self.num_heads, M, C)
        # x @ k
        # transpose k [B, num_heads, M, C] --> [B, num_heads, C, M]
        # [B, 1, HW, C] @ [B, num_heads, C, M] = [B, num_heads, HW, M]
        attn = x @ k.transpose(2, 3) * self.scale
        attn = attn.softmax(dim=-1)
        # attn @ v
        # [B, num_heads, HW, M] @ [B, num_heads, M, C] = [B, num_heads, HW, C]
        # premute and view [B, num_heads, HW, C] --> [B, HW, num_heads, C] --> [B, HW, inner_channels]
        res = attn @ v
        res = res.permute(0, 2, 1, 3).reshape(B, H*W, self.num_heads*C)
        # [B, HW, inner_channels] --> [B, HW, C]
        res = self.project_out(res).view(B, C, H, W)
        
        return res + shortcut


class MobileFormerBlock(nn.Module):
    def __init__(self, inp, exp, out, se, stride,heads,
                embeded_dim, expand_ratio=2):
        super(MobileFormerBlock, self).__init__()
        self.in_planes = inp
        self.expand_size = exp
        self.out_planes = out
        self.embeded_dim = embeded_dim
        self.num_heads = heads
        self.stride = stride
        self.expand_ratio = expand_ratio

        self.mobile = Mobile(self.in_planes, self.expand_size, self.out_planes, self.embeded_dim, stride=self.stride) # Note the stride must pass to the instance
        self.former = Former(self.embeded_dim, self.num_heads, mlp_ratio=self.expand_ratio)
        self.mobile2former = Mobile2Former(self.num_heads, self.embeded_dim, self.in_planes)
        self.former2mobile = Former2Mobile(self.num_heads, self.embeded_dim, self.out_planes)

    def forward(self, x, z):
        """
        mobile2former --> former --> mobile --> former2mobile
        """
        z_hidden = self.mobile2former(x, z)
        z_out = self.former(z_hidden)
        x_hidden = self.mobile(x, z_out)
        x_out = self.former2mobile(x_hidden, z_out)
        return x_out, z_out


class Mobile_Former(nn.Module):
    """
    framework:
    stem --> bneck-lite --> stages and repeat(1, 2, 3, 4, 5) --> up_project 1x1 --> avg_pool --> fc1 --> fc2
    """
    def __init__(self, cfg, initial_channel=3, dropout_ratio=0., num_classes=1000):
        super(Mobile_Former, self).__init__()
        self.tokens = nn.Parameter(torch.randn(cfg['token'], cfg['embed_dim']), requires_grad=True) # initialize z[M, d]
        self.stem = nn.Sequential(
            nn.Conv2d(initial_channel, cfg['stem_out_dim'], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cfg['stem_out_dim']),
            nn.ReLU6(inplace=True)
        )
        self.bneck = BottleNeckLite(in_planes=cfg['stem_out_dim'], expand_size=cfg['bneck']['bneck_exp'],
                                    out_planes=cfg['bneck']['bneck_out'], 
                                    kernel_size=3, stride=cfg['bneck']['stride'])
        
        self.blocks = nn.ModuleList()
        for kwargs in cfg['block']:
            self.blocks.append(
                MobileFormerBlock(**kwargs, embeded_dim=cfg['embed_dim'])
            )
        
        # last several layers
        last_in_channels = cfg['block'][-1]['out']
        last_expand_size = cfg['block'][-1]['exp']

        self.project_out = nn.Sequential(
            nn.Conv2d(last_in_channels, last_expand_size, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(last_expand_size),
            )
        
        self.avgpool =  nn.AvgPool2d((7,7))

        self.head = nn.Sequential(
            nn.Linear(last_expand_size+cfg['embed_dim'], cfg["fc_dimension"]),
            nn.Hardswish(),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(cfg["fc_dimension"], num_classes)
        )
        
        # init weights
        self.apply(self._init_weights)

    
    def _forward_impl(self, x):
        B, _, _, _ = x.shape
        tokens = self.tokens.repeat(B, 1, 1)
        x = self.bneck(self.stem(x))
        for block in self.blocks:
            x, tokens = block(x, tokens)
        x = self.project_out(x)
        x = self.avgpool(x).view(B, -1)
        x = torch.cat([x, tokens[:, 0, :]], dim=-1)
        return self.head(x)

    def forward(self, x):
        return self._forward_impl(x)

    def _init_weights(self, m):
        if isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def test():
    from config import config_294
    model = Mobile_Former(config_294)
    inputs = torch.randn((10, 3, 224, 224))
    print(inputs.shape)
    print("Total number of parameters in networks is {} M".format(sum(x.numel() for x in model.parameters()) / 1e6))
    output = model(inputs)
    print(output.shape)

test()