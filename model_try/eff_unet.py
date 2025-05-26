import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict
import itertools

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers import to_2tuple

# 定义扩展率字典（保持原样）
expansion_ratios_L = {
    '0': [4, 4, 4, 4, 4],
    '1': [4, 4, 4, 4, 4],
    '2': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 4, 3, 3, 3, 3, 4, 4, 4],
}

expansion_ratios_L_reverse = {
    '3': [4, 4, 4, 4, 4],
    '2': [4, 4, 4, 4, 4],
    '1': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '0': [4, 4, 4, 3, 3, 3, 3, 4, 4, 4],
}

#############################################
# Attention4D 等模块保持不变
#############################################
class Attention4D(torch.nn.Module):
    def __init__(self, dim=384, key_dim=32, num_heads=8,
                 attn_ratio=4,
                 resolution=7,
                 act_layer=nn.ReLU,
                 stride=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads

        if stride is not None:
            self.resolution = math.ceil(resolution / stride)
            self.stride_conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim),
                nn.BatchNorm2d(dim),
            )
            self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
        else:
            self.resolution = resolution
            self.stride_conv = None
            self.upsample = None

        self.N = self.resolution ** 2
        self.d = int(attn_ratio * key_dim)
        self.dh = self.d * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + self.nh_kd * 2
        self.q = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
            nn.BatchNorm2d(self.num_heads * self.key_dim),
        )
        self.k = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
            nn.BatchNorm2d(self.num_heads * self.key_dim),
        )
        self.v = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.d, 1),
            nn.BatchNorm2d(self.num_heads * self.d),
        )
        self.v_local = nn.Sequential(
            nn.Conv2d(self.num_heads * self.d, self.num_heads * self.d,
                      kernel_size=3, stride=1, padding=1, groups=self.num_heads * self.d),
            nn.BatchNorm2d(self.num_heads * self.d),
        )
        self.talking_head1 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1)
        self.talking_head2 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1)

        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2d(self.dh, dim, 1),
            nn.BatchNorm2d(dim),
        )

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(self.N, self.N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.stride_conv is not None:
            x = self.stride_conv(x)
        q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)
        attn = (q @ k) * self.scale + (self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab)
        attn = self.talking_head1(attn)
        attn = attn.softmax(dim=-1)
        attn = self.talking_head2(attn)
        x = (attn @ v)
        out = x.transpose(2, 3).reshape(B, self.dh, self.resolution, self.resolution) + v_local
        if self.upsample is not None:
            out = self.upsample(out)
        out = self.proj(out)
        return out

#############################################
# stem, merge, Embedding, Expanding 定义保持不变
#############################################
def stem(in_chs, out_chs, act_layer=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs // 2),
        act_layer(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs),
        act_layer(),
    )

def merge(in_chs, out_chs, act_layer=nn.ReLU):
    return nn.Sequential(
        nn.ConvTranspose2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_chs // 2),
        act_layer(),
        nn.ConvTranspose2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_chs),
        act_layer(),
    )

# Embedding, Expanding 等模块保持原样
# （此处省略 Embedding, Expanding, Mlp, AttnFFN, FFN, eformer_block, eformer_block_up, Flatten, CCA 等模块的代码，
# 请保留你现有代码不变）

#############################################
# 修改后的 Eff_Unet 模型定义
#############################################


class CCA(nn.Module):
    """
    Channel-wise Cross Attention (CCA) Block.
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x)
        )
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 将动态的 kernel 尺寸转换为 Python 常量
        H = int(x.size(2))
        W = int(x.size(3))
        avg_pool_x = F.avg_pool2d(x, (H, W), stride=(H, W))
        
        channel_att_x = self.mlp_x(avg_pool_x)
        
        H_g = int(g.size(2))
        W_g = int(g.size(3))
        avg_pool_g = F.avg_pool2d(g, (H_g, W_g), stride=(H_g, W_g))
        channel_att_g = self.mlp_g(avg_pool_g)
        
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum)
        # 确保扩展时使用静态尺寸
        scale = scale.unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class Eff_Unet(nn.Module):
    def __init__(self, layers, embed_dims,
                 mlp_ratios=4, downsamples=None,
                 pool_size=3,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.GELU,
                 num_classes=9,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 init_cfg=None,
                 pretrained=None,
                 vit_num=0,
                 distillation=True,
                 resolution=None,
                 in_chans=3,
                 input_size=None,  # 新增参数，格式为 (C, H, W)
                 e_ratios=expansion_ratios_L,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        # 如果提供了 input_size 且未显式给出 resolution，则用 input_size 的高度作为 resolution
        if resolution is None and input_size is not None:
            resolution = input_size[1]
        elif resolution is None:
            resolution = 224
        self.resolution = resolution

        self.patch_embed = stem(in_chans, embed_dims[0], act_layer=act_layer)
        self.up = merge(embed_dims[0], embed_dims[0], act_layer=act_layer)
        self.output = nn.Conv2d(embed_dims[0], self.num_classes, kernel_size=1, bias=False)
    
        network_down = []
        res_arr = []
        if self.num_classes == 1:
            self.sig = nn.Sigmoid()    

        for i in range(len(layers)):
            res = math.ceil(self.resolution / (2 ** (i + 2)))
            res_arr.append(res)
            stage = eformer_block(embed_dims[i], i, layers,
                                  pool_size=pool_size, mlp_ratio=mlp_ratios,
                                  act_layer=act_layer, norm_layer=norm_layer,
                                  drop_rate=drop_rate, drop_path_rate=drop_path_rate,
                                  use_layer_scale=use_layer_scale,
                                  layer_scale_init_value=layer_scale_init_value,
                                  resolution=res,
                                  vit_num=vit_num,
                                  e_ratios=e_ratios)
            network_down.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                asub = True if i >= 2 else False
                network_down.append(
                    Embedding(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1],
                        resolution=res,
                        asub=asub,
                        act_layer=act_layer, norm_layer=norm_layer,
                    )
                )
        self.network_down_layers = nn.ModuleList(network_down)
        
        # build decoder layers
        self.network_up_layers = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        self.skip_activation = nn.ReLU()
        self.coatt = nn.ModuleList()
        layers_reverse = layers[::-1]
        embed_dim_reverse = embed_dims[::-1]
        res_arr = res_arr[::-1]
        self.embed_dim_reverse = embed_dim_reverse
        for i in range(len(layers)):
            layers_reverse = layers[::-1]
            embed_dim_reverse = embed_dims[::-1]
            skip_down_conv = nn.Conv2d(2 * embed_dim_reverse[i], embed_dim_reverse[i], kernel_size=3, padding=1)
            self.coatt.append(CCA(F_g=embed_dim_reverse[i], F_x=embed_dim_reverse[i]))
            if i > 0:
                asub = True if i <= 2 else False
                self.network_up_layers.append(
                    Expanding(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dim_reverse[i-1], embed_dim=embed_dim_reverse[i],
                        resolution=res_arr[i-1], resolution2=res_arr[i],
                        asub=asub,
                        act_layer=act_layer, norm_layer=norm_layer,
                    )
                )
            network_up = eformer_block_up(embed_dim_reverse[i], i, layers_reverse,
                                          pool_size=pool_size, mlp_ratio=mlp_ratios,
                                          act_layer=act_layer, norm_layer=norm_layer,
                                          drop_rate=drop_rate, drop_path_rate=drop_path_rate,
                                          use_layer_scale=use_layer_scale,
                                          layer_scale_init_value=layer_scale_init_value,
                                          resolution=res_arr[i],
                                          vit_num=vit_num,
                                          e_ratios=expansion_ratios_L_reverse)
            self.network_up_layers.append(network_up)
            self.concat_back_dim.append(skip_down_conv)

        self.out_indices = [0, 2, 4, 6]
        for i_emb, i_layer in enumerate(self.out_indices):
            if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                layer = nn.Identity()
            else:
                layer = norm_layer(embed_dims[i_emb])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network_down_layers):
            x = block(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return x, outs
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        x, x_downsample = self.forward_tokens(x)
        return x, x_downsample
    
    def forward_up_features(self, x, x_downsample):
        skip_conn = [0, 2, 4, 6]
        x_downsample_reverse = x_downsample[::-1]
        for idx, block in enumerate(self.network_up_layers):
            if idx not in skip_conn:
                x = block(x)
            else:
                if self.num_classes != 1:
                    skip_x_att = self.coatt[skip_conn.index(idx)](x, x_downsample_reverse[skip_conn.index(idx)])
                    x = torch.cat([x, skip_x_att], 1)
                    x = self.concat_back_dim[skip_conn.index(idx)](x)
                    x = block(x)
                else:
                    x = torch.cat([x, x_downsample_reverse[skip_conn.index(idx)]], 1)
                    x = self.concat_back_dim[skip_conn.index(idx)](x)
                    x = block(x)
        return x
    
    def up_sample(self, x):
        return self.up(x)
    
    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_sample(x)
        x = self.output(x)
        if self.num_classes == 1:
            x = self.sig(x)
        return x
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
def eformer_block(dim, index, layers,
                  pool_size=3, mlp_ratio=4.,
                  act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                  drop_rate=0., drop_path_rate=0.,
                  use_layer_scale=True, layer_scale_init_value=1e-5,
                  vit_num=1, resolution=7, e_ratios=None):
    """
    构建一个 eFormer 块，包含若干个 AttnFFN 或 FFN 模块。
    参数 e_ratios 应该是一个字典，其 key 为字符串形式的层索引，
    value 为一个列表，对应每个 block 的 mlp_ratio。
    """
    blocks = []
    total_blocks = sum(layers)
    for block_idx in range(layers[index]):
        # 计算当前 block 在整个网络中的相对位置，决定 drop_path_rate
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (total_blocks - 1)
        # 获取当前 block 的 mlp_ratio，从 e_ratios 字典中获取
        if e_ratios is not None:
            mlp_ratio_current = e_ratios[str(index)][block_idx]
        else:
            mlp_ratio_current = mlp_ratio
        # 当 index>=2 且处于最后 vit_num 个 block 时，使用 AttnFFN
        if index >= 2 and block_idx > layers[index] - 1 - vit_num:
            stride = 2 if index == 2 else None
            blocks.append(AttnFFN(
                dim, mlp_ratio=mlp_ratio_current,
                act_layer=act_layer, norm_layer=norm_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                resolution=resolution,
                stride=stride,
            ))
        else:
            blocks.append(FFN(
                dim, pool_size=pool_size, mlp_ratio=mlp_ratio_current,
                act_layer=act_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))
    return nn.Sequential(*blocks)

class AttnFFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 resolution=7, stride=None):
        super().__init__()
        self.token_mixer = Attention4D(dim, resolution=resolution, act_layer=act_layer, stride=stride)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop, mid_conv=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1))
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1))

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(x))
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x

from timm.models.layers import to_2tuple

class Expanding(nn.Module):
    """
    Expanding block for the decoder part.
    This module upsamples the input feature map.
    """
    def __init__(self, patch_size=2, stride=2, padding=1,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d,
                 light=False, asub=False, resolution=None, resolution2=None, act_layer=nn.ReLU):
        super().__init__()
        self.light = light
        self.asub = asub

        if self.light:
            self.new_proj = nn.Sequential(
                nn.ConvTranspose2d(in_chans, in_chans, kernel_size=3, stride=2, padding=1, output_padding=1, groups=in_chans),
                norm_layer(in_chans),
                act_layer(),
                nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1, padding=0),
                norm_layer(embed_dim),
            )
            self.skip = nn.Sequential(
                nn.ConvTranspose2d(in_chans, embed_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(embed_dim)
            )
        elif self.asub:
            # 如果 asub 为 True，则使用注意力模块上采样（需定义 Attention4DUpsample）
            self.attn = Attention4DUpsample(dim=in_chans, out_dim=embed_dim,
                                            resolution=resolution, resolution2=resolution2, act_layer=act_layer)
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.conv = nn.ConvTranspose2d(in_chans, embed_dim, kernel_size=patch_size,
                                           stride=stride, padding=padding, output_padding=1)
            self.bn = norm_layer(embed_dim)
        else:
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.proj = nn.ConvTranspose2d(in_chans, embed_dim, kernel_size=patch_size,
                                           stride=stride, padding=padding, output_padding=1)
            self.norm = norm_layer(embed_dim)
    
    def forward(self, x):
        if self.light:
            out = self.new_proj(x) + self.skip(x)
        elif self.asub:
            out_conv = self.conv(x)
            out_conv = self.bn(out_conv)
            out = self.attn(x) + out_conv
        else:
            x = self.proj(x)
            out = self.norm(x)
        return out


class FFN(nn.Module):
    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, mid_conv=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1))

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.mlp(x))
        return x

def eformer_block_up(dim, index, layers,
                     pool_size=3, mlp_ratio=4.,
                     act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                     drop_rate=0., drop_path_rate=0.,
                     use_layer_scale=True, layer_scale_init_value=1e-5,
                     vit_num=1, resolution=7, e_ratios=None):
    """
    类似于 eformer_block，但用于上采样部分。这里可以采用与下采样部分类似的结构，
    或者根据需要进行调整。
    """
    blocks = []
    total_blocks = sum(layers)
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[index+1:])) / (total_blocks - 1)
        if e_ratios is not None:
            mlp_ratio_current = e_ratios[str(index)][block_idx]
        else:
            mlp_ratio_current = mlp_ratio
        if index <= 1 and block_idx > layers[index] - 1 - vit_num:
            stride = 2 if index == 1 else None
            blocks.append(AttnFFN(
                dim, mlp_ratio=mlp_ratio_current,
                act_layer=act_layer, norm_layer=norm_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                resolution=resolution,
                stride=stride,
            ))
        else:
            blocks.append(FFN(
                dim, pool_size=pool_size, mlp_ratio=mlp_ratio_current,
                act_layer=act_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))
    return nn.Sequential(*blocks)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 9,
        'input_size': (3, 224, 224),  # 修改为3通道
        'pool_size': None,
        'crop_pct': .95,
        'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }
from timm.models.layers import to_2tuple

class Embedding(nn.Module):
    def __init__(self, patch_size=3, stride=2, padding=1,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d,
                 light=False, asub=False, resolution=None, act_layer=nn.ReLU):
        """
        Embedding 模块，用于将输入图像转换为嵌入表示。
        参数：
            patch_size: 卷积核大小
            stride: 卷积步幅
            padding: 填充
            in_chans: 输入通道数
            embed_dim: 输出通道数（嵌入维度）
            norm_layer: 归一化层
            light: 是否使用轻量版结构
            asub: 是否使用附加注意力模块（需配合下采样 Attention）
            resolution: 分辨率（可选，用于注意力模块）
            act_layer: 激活函数
        """
        super().__init__()
        self.light = light
        self.asub = asub

        if self.light:
            self.new_proj = nn.Sequential(
                nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=2, padding=1, groups=in_chans),
                norm_layer(in_chans),
                nn.Hardswish(),
                nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1, padding=0),
                norm_layer(embed_dim),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=2, padding=0),
                norm_layer(embed_dim)
            )
        elif self.asub:
            # 如果使用 asub 模式，需使用下采样注意力模块 Attention4DDownsample
            self.attn = Attention4DDownsample(dim=in_chans, out_dim=embed_dim,
                                              resolution=resolution, act_layer=act_layer)
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                                  stride=stride, padding=padding)
            self.bn = norm_layer(embed_dim) if norm_layer else nn.Identity()
        else:
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                                  stride=stride, padding=padding)
            self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if self.light:
            out = self.new_proj(x) + self.skip(x)
        elif self.asub:
            out_conv = self.conv(x)
            out_conv = self.bn(out_conv)
            out = self.attn(x) + out_conv
        else:
            x = self.proj(x)
            out = self.norm(x)
        return out
class Mlp(nn.Module):
    """
    多层感知机（MLP），使用 1x1 卷积实现。
    输入形状为 [B, C, H, W]，适用于图像特征处理。
    如果 mid_conv=True，会在两层之间使用 3x3 深度卷积进行局部特征融合。
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, mid_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv

        # 第一层 1x1 卷积
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act = act_layer()
        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.drop1 = nn.Dropout(drop)

        # 可选中间 3x3 深度卷积
        if self.mid_conv:
            self.mid = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)
            self.mid_norm = nn.BatchNorm2d(hidden_features)
            self.act_mid = act_layer()
        
        # 第二层 1x1 卷积
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        if self.mid_conv:
            x = self.mid(x)
            x = self.mid_norm(x)
            x = self.act_mid(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.drop2(x)
        return x
class UpLGQuery(nn.Module):
    """
    UpLGQuery: 用于上采样阶段的局部查询模块，
    通过转置卷积和双线性上采样结合实现上采样特征提取。
    """
    def __init__(self, in_dim, out_dim, resolution1, resolution2):
        super().__init__()
        self.resolution1 = resolution1
        self.resolution2 = resolution2
        self.pool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.local = nn.Sequential(
            nn.ConvTranspose2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, output_padding=1, groups=in_dim),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        local_q = self.local(x)
        pool_q = self.pool(x)
        q = local_q + pool_q
        q = self.proj(q)
        return q

class Attention4DUpsample(nn.Module):
    """
    Attention4DUpsample: 上采样阶段的注意力模块，
    它与 Attention4DDownsample 类似，但适用于上采样特征。
    """
    def __init__(self, dim=384, key_dim=16, num_heads=8,
                 attn_ratio=4,
                 resolution=7,
                 resolution2=7,
                 out_dim=None,
                 act_layer=nn.ReLU):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads

        self.resolution = resolution  # 原始分辨率
        self.d = int(attn_ratio * key_dim)
        self.dh = self.d * num_heads
        self.attn_ratio = attn_ratio
        self.out_dim = out_dim if out_dim is not None else dim
        self.resolution2 = resolution2  # 上采样后的分辨率
        # 使用 UpLGQuery 进行上采样查询
        self.q = UpLGQuery(dim, self.num_heads * self.key_dim, self.resolution, self.resolution2)

        self.N = self.resolution ** 2
        self.N2 = self.resolution2 ** 2

        self.k = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.key_dim, kernel_size=1),
            nn.BatchNorm2d(self.num_heads * self.key_dim),
        )
        self.v = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.d, kernel_size=1),
            nn.BatchNorm2d(self.num_heads * self.d),
        )
        self.v_local = nn.Sequential(
            nn.ConvTranspose2d(self.num_heads * self.d, self.num_heads * self.d,
                               kernel_size=3, stride=2, padding=1, output_padding=1, groups=self.num_heads * self.d),
            nn.BatchNorm2d(self.num_heads * self.d),
        )
        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2d(self.dh, self.out_dim, kernel_size=1),
            nn.BatchNorm2d(self.out_dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, self.N2).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(2, 3)
        out = x.reshape(B, self.dh, self.resolution2, self.resolution2) + v_local
        out = self.proj(out)
        return out
class Attention4DDownsample(nn.Module):
    def __init__(self, dim=384, key_dim=16, num_heads=8,
                 attn_ratio=4,
                 resolution=7,
                 act_layer=nn.ReLU):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads

        self.resolution = resolution
        self.d = int(attn_ratio * key_dim)
        self.dh = self.d * num_heads
        self.attn_ratio = attn_ratio

        # token数量
        self.N = self.resolution ** 2

        # 定义 q, k, v 模块
        self.q = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
            nn.BatchNorm2d(self.num_heads * self.key_dim)
        )
        self.k = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
            nn.BatchNorm2d(self.num_heads * self.key_dim)
        )
        self.v = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.d, 1),
            nn.BatchNorm2d(self.num_heads * self.d)
        )
        # v_local 用于局部信息提取，采用卷积下采样
        self.v_local = nn.Sequential(
            nn.Conv2d(self.num_heads * self.d, self.num_heads * self.d,
                      kernel_size=3, stride=2, padding=1, groups=self.num_heads * self.d),
            nn.BatchNorm2d(self.num_heads * self.d)
        )
        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2d(self.dh, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # q: reshape为 [B, num_heads, N, key_dim]
        q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)
        # k: reshape为 [B, num_heads, key_dim, N]
        k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 2, 3)
        # v: reshape为 [B, num_heads, N, d]
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)
        
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x_attn = attn @ v
        # 将结果重新排列为 [B, dh, resolution, resolution]
        x_attn = x_attn.transpose(2, 3).reshape(B, self.dh, self.resolution, self.resolution)
        out = x_attn + v_local
        out = self.proj(out)
        return out

def load_from(model, ckpt_path):
    pretrained_path = ckpt_path
    if pretrained_path is not None:
        print("pretrained_path:{}".format(pretrained_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        pretrained_dict = pretrained_dict['model']
        model_dict = model.state_dict()                
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "network." in k:
                up_layer_num = 6 - int(k.split('.')[1])
                current_k_up = "network_up_layers." + str(up_layer_num) + '.' + '.'.join(k.split('.')[2:])
                full_dict.update({current_k_up: v})
                full_dict["network_down_layers." + '.'.join(k.split('.')[1:])] = full_dict.pop(k)
        for k in list(full_dict.keys()):
            if k in model_dict:
                if full_dict[k].shape != model_dict[k].shape:
                    print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                    del full_dict[k]
        msg = model.load_state_dict(full_dict, strict=False)
        print("pretrained weights are loaded")
        return model
    else:
        print("none pretrain")
