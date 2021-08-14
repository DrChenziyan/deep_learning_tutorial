from functools import partial
from torch.nn import parameter
from typing import Callable, Optional, OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import flatten
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.module import Module
from thop import profile
from fvcore.nn import FlopCountAnalysis
# Apply ViT from stracth
"""
ViT: https://arxiv.org/pdf/2010.11929.pdf

refs:   1. https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
        2.https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/vision_transformer
        3.https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

Notations:
    1. MLP heads -- similiar to last several layers of Linear classification in CNN, but could be more complex in larger datasets like ImageNet21k.
    2. Before Attetion, add a stochastic depth function following with batch norm, which was not planted in the figs.
    3. Positional embedding is necessary but shows little difference between different ways of encoding.
"""

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


class PatchEmbedding_2D(nn.Module):
    """
    2D-dim patch embedding
    """
    def __init__(self, 
                img_size:int,
                patch_size:int,
                in_planes:int,
                embed_dim: int,
                norm_layer=None,
                flatten=True):
        super(PatchEmbedding_2D, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # Apply Conv2D to realize the patch embedding.
        self.conv = nn.Conv2d(
            in_channels=in_planes,
            out_channels=embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        ) 
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1],\
            f'Input size must be ({self.img_size[0]} * {self.img_size[1]}). \
                You have to check the input size that must to be matched.'
        x = self.conv(x)
        # flatten: [B, C, H, W] --> [B, C, HW]
        # transpose: [B, C, HW] --> [B, HW, C]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


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

# Encoder Architecture
"""
Layer norm 
    |
Multi-head Attention
    |
DropPath(stochastic depth) 
    | + (Residual)
Layer Norm
    |
MLP block
    |
DropPath
    | + (Residual)
"""
class Block(nn.Module):
    def __init__(self,
                dim, # total input dim (num_patches + 1, total_embeded_dims) (3, 224, 224) -> (PatchEmbedding_2D) -> (16*16+1, 768)
                num_heads,
                mlp_ratio=4,
                qkv_bias=False,
                qk_scale=None,
                drop_path_prob=0.,
                attn_drop_ratio=0.,
                dropout_ratio=0.,
                activation_layer:Optional[Callable[..., nn.Module]]=None,
                norm_layer:Optional[Callable[..., nn.Module]]=None
                ):
        super(Block, self).__init__()

        if activation_layer is None:
            activation_layer = nn.GELU()
        if norm_layer is None:
            norm_layer = nn.LayerNorm(dim)
        
        # Layer Norm_1 -> Multi-head -> Drop_path
        self.norm = norm_layer(dim)
        self.msa = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop_ratio, dropout_ratio)
        if drop_path_prob > 0:
            self.drop_path = DropPath(drop_prob=drop_path_prob)
        else:
            self.drop_path = nn.Identity()

        # Layer Norm_2 -> MLP block -> Drop_path
        # use the mlp_ratio to increase the dim
        self.mlp = MLP(dim, int(mlp_ratio*dim), dim, activation_layer, dropout_ratio=dropout_ratio)
    
    def forward(self, x):
        x += self.drop_path(self.msa(self.norm(x)))
        x += self.drop_path(self.mlp(self.norm(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                img_size=224,
                patch_size=16,
                in_planes=3,
                num_classes=1000,
                embed_dim=768, 
                num_layers=12, # Numbers of encoding block
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                representation_size=None, # if none last several stages are Linear, else, Linear->tanh->Linear
                drop_path_prob=0.,
                attn_drop_ratio=0.,
                dropout_ratio=0.,
                embed_layer=PatchEmbedding_2D,
                activation_layer:Optional[Callable[..., nn.Module]]=None,
                norm_layer:Optional[Callable[..., nn.Module]]=None
                ):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        activation_layer = nn.GELU if activation_layer is None else activation_layer

        # embedding
        self.patch_embedding = embed_layer(
            img_size, patch_size, in_planes, embed_dim, 
        )

        num_patches = self.patch_embedding.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) # absoluate positional embedding: one value one position.
        self.pos_drop = nn.Dropout(p=dropout_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_prob, num_layers)]

        # encoding blocks
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop_path_prob=dpr[i], 
                    attn_drop_ratio=attn_drop_ratio, dropout_ratio=dropout_ratio,
                    activation_layer=activation_layer, norm_layer=norm_layer
                )
                for i in range(num_layers)
            ]
        )
        self.norm_layer = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ])
            )
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # last several layers for classification
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
        # weight init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.apply(self._init_vit_weights)

    def _forward_impl(self, x):
        # [B, C, H, W] -> [B,N,embed_dim]
        x = self.patch_embedding(x)
        # create cls_token and epand the dim same as the x 
        # ([1, 1, embed_dim] -> [B, 1, embed_dim])
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embedding)
        x = self.blocks(x)
        x = self.norm_layer(x)
        return self.pre_logits(x[:, 0])

    def forward(self, x):
        x = self._forward_impl(x)
        x = self.head(x)
        return x

    
    def _init_vit_weights(self,m):
        # weight initialization
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)


# Base
def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              num_layers=12,
                              num_heads=12,
                              dropout_ratio=0.2,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model

def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              num_layers=12,
                              num_heads=12,
                              dropout_ratio=0.2,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_384_in21k(num_classes: int = 21843, has_logits: bool = True):
    model = VisionTransformer(img_size=384,
                              patch_size=32,
                              embed_dim=768,
                              num_layers=12,
                              num_heads=12,
                              dropout_ratio=0.2,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


# Large
def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              num_layers=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_384_in21k(num_classes: int = 21843, has_logits: bool = True):
    model = VisionTransformer(img_size=384,
                              patch_size=16,
                              embed_dim=1024,
                              num_layers=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              num_layers=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_384_in21k(num_classes: int = 21843, has_logits: bool = True):
    model = VisionTransformer(img_size=384,
                              patch_size=32,
                              embed_dim=1024,
                              num_layers=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


# Huge
def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              num_layers=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model

def get_model_parameters(model:nn.Module):
    total_params = 0
    for layer in list(model.parameters()):
        layer_parameters = 1
        for l in list(layer.size()):
            layer_parameters *= l
        total_params += layer_parameters
    return total_params

    


# ------------------------------------------------------------------------------------------
def test():
    num_examples = 10
    img_size = 224
    model = vit_base_patch16_224_in21k(num_classes=100)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # parameters = get_model_parameters(model)
    print(f"Your device is {device}")

    x = torch.randn((num_examples, 3, img_size, img_size)).to(device)
    model = model.to(device)
    # macs, params = profile(model, inputs=(x,))
    # print(f"Parameters of your model are {params}. And FLOPs are {macs}")
    flops = FlopCountAnalysis(model, x)
    print(f"FLOPs are {flops.total()}")
    print(model(x).shape)


def print_all_parameters():
    vit_base_patch16_224_params = get_model_parameters(vit_base_patch16_224_in21k(num_classes=100))
    vit_base_patch32_224_params = get_model_parameters(vit_base_patch32_224_in21k(num_classes=100))
    vit_base_patch32_384_params = get_model_parameters(vit_base_patch32_384_in21k(num_classes=100))

    vit_large_patch16_224_params = get_model_parameters(vit_large_patch16_224_in21k(num_classes=100))
    vit_large_patch16_384_params = get_model_parameters(vit_large_patch16_384_in21k(num_classes=100))
    vit_large_patch32_224_params = get_model_parameters(vit_large_patch32_224_in21k(num_classes=100))
    vit_large_patch32_384_params = get_model_parameters(vit_large_patch32_384_in21k(num_classes=100))

    vit_huge_patch14_224_params = get_model_parameters(vit_huge_patch14_224_in21k(num_classes=100))

    print("vit_base_model")
    print(f"vit_base_patch16_224_params: {int(vit_base_patch16_224_params)// 1000000}")
    print(f"vit_base_patch32_224_params: {int(vit_base_patch32_224_params)// 1000000}")
    print(f"vit_base_patch32_384_params: {int(vit_base_patch32_384_params)// 1000000}")
    print('-'* 20)
    
    print("vit_large_model")
    print(f"vit_large_patch16_224_params: {int(vit_large_patch16_224_params) // 1000000}")
    print(f"vit_large_patch16_384_params: {int(vit_large_patch16_384_params) // 1000000}")
    print(f"vit_large_patch32_224_params: {int(vit_large_patch32_224_params) // 1000000}")
    print(f"vit_large_patch32_384_params: {int(vit_large_patch32_384_params) // 1000000}")
    print('-'* 20)

    print("vit_huge_model")
    print(f"vit_huge_patch16_224_params: {int(vit_huge_patch14_224_params) // 1000000}")



def main():
    # test()
    print_all_parameters()


if __name__ == '__main__':
    main()




    




    
