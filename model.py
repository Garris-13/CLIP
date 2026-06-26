from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Bottleneck residual block，在保持或改变通道数/空间分辨率的同时，通过残差连接（identity + residual）稳定训练、加快收敛并保持梯度流动
class Bottleneck(nn.Module):
    expansion = 4 # 最终输出通道数是内部中间通道数（planes）的4倍

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        # 利用1×1的卷积降维减少计算量

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        # 当stride > 1时，使用平均池化层来降低空间分辨率，否则使用恒等映射。这里是平均池化，专门负责空间降采样
        # 原始ResNet是在3×3卷积时直接设stride = 2来降采样，CLIP改为“先3×3卷（步长1），再平均池化降采样”。这个设计被认为对某些下游任务更友好

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        # 形状不匹配（分辨率变了/通道数变了）则下采样
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            #这么做使得原图像每一个像素不会被浪费，而Resnet直接用stride=2 的 1 × 1 卷积，一下字把“缩小尺寸”和“调整通道”这两件事同时办了，这样会导致大部分特征丢失
            #顺序执行有序字典里的内容
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),#空间降采样，“-1”指前置处理，与主分支的avgpool对应
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))
    # 通过conv1先压缩到64，在64通道上做3×3卷积，再用conv3恢复，计算量远小于在原来的256通道上做3×3卷积(减少计算成本)

    #
    def forward(self, x: torch.Tensor):
        identity = x #恒等映射，保留原始输入

        out = self.relu1(self.bn1(self.conv1(x))) #降维，通道压缩
        out = self.relu2(self.bn2(self.conv2(out))) #空间特征提取
        out = self.avgpool(out) #分辨率下采样（不丢弃空间像素）
        out = self.bn3(self.conv3(out)) #恢复通道，升维

        if self.downsample is not None:
            identity = self.downsample(x) #调整形状不匹配

        out += identity
        out = self.relu3(out)
        return out

#注意力池化层，采用多头自注意力机制（Multi-Head Attention） 来做池化
#Resnet在提取完特征后通常使用全局平均池化
#普通池化（GAP）是死板的数学平均；而 AttentionPool2d 则是带偏向性的语义聚拢。
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        #代表特征图上所有的像素点总数，+1是专门给 Global Query（全局查询/类似于 Transformer 的 CLS Token） 预留的位置
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        #把后两维（宽高）压缩成一个，并调整维度顺序为[序列长度(HW), Batch大小(N), 通道数(C)]，（这是Transformer期待的输出格式）

        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        #在第一个维度（像素）上取平均，算出Global Token；然后把全局平均特征和原本的像素特征拼接在一起

        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x, #query只传了x的第0项，key与value传了完整的特征
            #这是一次Targeted Attention。我们只关心“全局特征（Query）”去和“图上所有的像素点（Key）”做匹配。
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        #把在 __init__ 中定义好的各线性层的weight和bias喂给 PyTorch 底层的 C++ 高性能加速函数
        return x.squeeze(0)
    #由于输出结果 x 的形状是 [1, 32, output_dim]，第一维的 1 已经没有意义了。使用 squeeze(0) 将其消除，最终返回 [32, output_dim]（即 [Batch_Size, 最终视觉向量维度]）
    #这就是代表整张图片的特征向量，可以直接拿去和文本向量算相似度了。

#组装成完整网络 多层 Stem、残差主干构建、前向传播
class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim # 最终输出的 CLIP 视觉向量维度（如 512）
        self.input_resolution = input_resolution # 输入图像的分辨率（默认 224x224）

        # the 3-layer stem
        # 第一层卷积：把 3 通道输入变成 width // 2 (即 32) 通道，高宽减半 (stride=2)
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        # 第二层卷积：保持 32 通道，高宽不变
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        # 第三层卷积：把通道提升到 width (即 64)，高宽不变
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        # 平均池化：再次将高宽减半。
        self.avgpool = nn.AvgPool2d(2)


        # residual layers
        # 内部状态变量 _inplanes，记录当前特征图的实际通道数，会随着层数加深动态改变
        self._inplanes = width  # this is a *mutable* variable used during construction
        # 初始值为 64

        # 构建四大层，逐渐加深通道，缩小高宽
        self.layer1 = self._make_layer(width, layers[0]) # 出来通道变成 width * 4 = 256
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2) # 出来通道变成 512, 尺寸减半
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        # 2048，与 layer4 输出通道一致，告诉池化层每个像素点由一个 2048 维的向量组成
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)
        # input_resolution // 32 ：经历前面 5 次下采样（Stem 里 2 次，layer2,3,4各1次），
        # 224 变为了 224 // 32 = 7。所以最终特征图的分辨率是 7x7

    # 构建重复残差块（Bottleneck）的自动化工具
    def _make_layer(self, planes, blocks, stride=1):
        # 1. 每一层的第一个块负责调整通道数或缩小高宽（传递了 stride），其需要改变特征图分辨率，同时对接上一层的通道数
        layers = [Bottleneck(self._inplanes, planes, stride)]
        # 2. 算一下第一个块输出后的实际通道数。
        self._inplanes = planes * Bottleneck.expansion
        # 3. 后续的块（从 1 到 blocks-1）：它们不需要改变尺寸和通道数，只是纯粹加深网络，在相同分辨率下安稳地提取更深层的特征
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        # 4. 用 nn.Sequential 把这些块打包起来，* 把 layers 列表解包，将里面所有的 Bottleneck 实例按顺序喂给 nn.Sequential 打包并返回
        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        #确保输入的图像数据类型（比如 FP16、FP32）与卷积层权重的数据类型完全一致
        x = x.type(self.conv1.weight.dtype)

        #让图像通过浅层特征提取层 (Stem)
        # 输入: [Batch, 3, 224, 224] -> 输出: [Batch, 64, 56, 56]
        x = stem(x)

        # 纵穿四个残差层，通道越来越粗，图像分辨率越来越小
        x = self.layer1(x) # 输出: [Batch, 256, 56, 56]
        x = self.layer2(x) # 输出: [Batch, 512, 28, 28]
        x = self.layer3(x) # 输出: [Batch, 1024, 14, 14]
        x = self.layer4(x) # 输出: [Batch, 2048, 7, 7]
        # 4. 扔进注意力池化层，提炼出全图的最终语义特征向量
        # 输出: [Batch, output_dim]（如 [Batch, 512]）
        x = self.attnpool(x)

        return x

#抗溢出层归一化，继承自 PyTorch 官方的 nn.LayerNorm
#背景痛点：CLIP 为了极大地提升训练速度并减少显存占用，大量使用了 FP16（半精度浮点数，16位） 进行计算。但是 FP16 的数值表示范围非常窄（最大只能到 65504）。
#层归一化在内部计算时，需要计算特征的方差，算方差时要进行平方和运算，如果输入 x 的维度很大，很多数平方再相加，其结果极易超过 65504，从而导致 NaN
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    #用极小的计算代价换取完美的数值稳定性
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype  #输入张量原始的数据类型（通常是 torch.float16）

        ret = super().forward(x.type(torch.float32))
        #  x.type(torch.float32)：在计算前，强制把数据转换为 FP32（单精度浮点数，32位）
        #    super().forward(...)：调用官方底层的 C++ LayerNorm 函数进行归一化计算

        return ret.type(orig_type)
        # type(orig_type)：计算完毕后，再把结果强制安全地转换回原始的类型（FP16）并返回

#极速高斯误差线性单元：GELU（Gaussian Error Linear Unit）是现代 Transformer（如 BERT, GPT）中最标准、最常用的激活函数。但标准的 GELU 计算公式包含误差函数（erf），在硬件底层计算起来非常慢。
#对标准 GELU 的一种高速数学近似，用 Sigmoid(1.702 * x) 完美地拟合了高斯累积分布函数
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x) #实验证明，这种微小的近似带来的精度损失几乎为零，但却换来了底层计算速度的明显提升

#CLIP 模型中 Transformer 块的底层核心实现
#文本编码器（Text Encoder），还是其ViT版视觉编码器，其内部堆叠的无数个 Standard Transformer Layer，全都是由这个类实例化出来的。
# CLIP 里的通用计算砖块，在文本编码器里，它一砖一瓦地堆叠，通过传递特殊的 attn_mask（下三角矩阵），让文字只能看到左边，学到句子的上下文语义；
# 在 Vision Transformer 里，它同样一砖一瓦地堆叠，但不需要 attn_mask，让打碎的图像 Patch 之间通过自注意力互相通信，拼凑出整张图的全局逻辑。
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        #d_model：特征的隐藏层维度（Hidden Dimension）；n_head：多头注意力机制的头数；attn_mask：注意力掩码（Mask），在文本端极其重要（用于实现 Causal Mask，防止未来的文字泄露给当前位置）
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head) #直接调用 PyTorch 官方的高性能多头自注意力（Multi-head Self-Attention）模块
        self.ln_1 = LayerNorm(d_model) #实例化在上一步经过 OpenAI 抗 FP16 溢出魔改的 LayerNorm。这是 Attention 之前的前置层归一化（Pre-LN）。
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)), #c_fc（Fully Connected）：将特征维度从 d_model 放大 4 倍，变为 d_model * 4。
            ("gelu", QuickGELU()), #gelu：通过极速激活函数 QuickGELU() 进行非线性变换。
            ("c_proj", nn.Linear(d_model * 4, d_model)) #c_proj（Projection）：再把特征维度从 d_model * 4 压缩回原来的 d_model，以便进行残差相加。
        ])) #构建前馈网络（Feed-Forward Network / MLP）。同样是典型的“放大再缩小”。

        self.ln_2 = LayerNorm(d_model) #ln_2 是进入 MLP 之前的第二道层归一化；同时将掩码保存为类属性。
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        # 1. 动态对齐掩码（Mask）的数据类型和所在设备（CPU/GPU）。
        #    因为输入 x 可能会因为半精度训练发生类型变化，掩码必须随时跟 x 保持一致，否则报错。
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None

        # 2. 执行自注意力计算。
        #    由于是自注意力（Self-Attention），所以 Query, Key, Value 传入的全都是同一个张量 x。
        #    need_weights=False：告诉 PyTorch 只要计算结果，不需要传回注意力权重矩阵（节省显存）。
        #    [0]：PyTorch 的 MultiheadAttention 会返回两个东西：(output, weights)，我们只需要第 0 项的 output。
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    # Pre-LN（前置归一化）残差结构
    def forward(self, x: torch.Tensor):
        # 第一阶段：多头自注意力 + 残差连接
        x = x + self.attention(self.ln_1(x))
        #传统的 Transformer 采用 Post-LN（先 Attention 再 Norm）。
        #而 GPT 和 CLIP 等现代模型全部采用 Pre-LN（先对 x 做 ln_1 归一化，再送给 attention）。算完之后，直接加回没有被归一化污染的原始 x 上。这样可以确保有一条纯净的“梯度直通高速公路”，极其利于深层网络的稳定训练。

        # 第二阶段：前馈网络（MLP）+ 残差连接
        x = x + self.mlp(self.ln_2(x)) #同样的逻辑，对经历过第一阶段融合的 x 先做 ln_2 归一化，扔进 mlp 提取高维语义特征，最后再次通过残差连接与未归一化的 x 相加。
        return x

#通用的 Transformer 包装盒，OpenAI 只需要给文本端实例化一个 Transformer(width=512, layers=12, ...)，再给视觉端实例化一个 Transformer(width=768, layers=12, ...)
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width # 特征的隐藏层维度（d_model，如 512 或 768）
        self.layers = layers # 需要堆叠的 Transformer 核心块的总层数（如 12 层）
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        #[ResidualAttentionBlock(...) for _ in range(layers)]：这是一个 Python 列表推导式。如果 layers=12，它就会在内存里连续创建 12 个独立的、拥有各自权重的 ResidualAttentionBlock 实例，并存入一个 Python 列表中。
        #星号 * 解包：由于 nn.Sequential 不接受列表作为输入，它只接受一个一个并列的子模块参数。所以通过 * 号把这个拥有 12 个块的列表“拆开”、“解包”成 12 个独立的参数。
        #nn.Sequential(...) 打包：将解包后的 12 个块按顺序塞进 PyTorch 的顺序容器中。在前向传播时，数据会雷打不动地按照 第0块 ──> 第1块 ──> ... ──> 第11块 的顺序依次纵穿过去。

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    #这里的 x 是输入的特征张量（或者是文本端刚加完位置编码的文本 Token，或者是视觉端刚打碎的图像 Patch 向量）。
    #把 x 扔进刚才打包好的 self.resblocks 顺序流水线里。nn.Sequential 会自发地调用内部每一个 ResidualAttentionBlock 的 forward 函数。
    # 当 x 从最后一步跳出来时，它已经经历了多次的深层自注意力和MLP的洗礼，已经从表面的离散特征升华为了蕴含深刻上下文语义的高维向量。

# 把一张静态的图片打碎成一系列“拼图Token”，然后用纯 Transformer 结构来提取图像语义。
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution # 输入图像分辨率
        self.output_dim = output_dim # 最终对齐的视觉向量维度（如 512）
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        #这是 ViT 实现“图片变单词”的经典操作。假设图片是 224x224，patch_size 是 32。这一层卷积核大小是 32×32，步长也是 32。它在图上滑动时，刚好把图切成了 224×224/32/32 = 49$ 块。同时，它直接把这 49 个方块投影到了 width维的通道空间。

        scale = width ** -0.5 # 缩放因子，用于稳定权重初始化
        self.class_embedding = nn.Parameter(scale * torch.randn(width)) # 初始化 [CLS] Token，这是一个可学习的全局图像特征占位符
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)) # 初始化位置编码。由于有 49 个方块，加上 1 个 [CLS] Token，一共有 50 个位置
        self.ln_pre = LayerNorm(width) # 进入 Transformer 之前的层归一化

        self.transformer = Transformer(width, layers, heads) # 实例化多层 ResidualAttentionBlock 组成的 Transformer 引擎

        self.ln_post = LayerNorm(width) # 走出 Transformer 之后的层归一化
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))# 最后的线性投影矩阵（用矩阵乘法 @ 来做维度映射，将 width 变成 output_dim）
        #生成一个形状为 (width, output_dim) 的矩阵，其中每个元素都独立地从标准正态分布（均值 0，方差 1）中采样。

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        #嵌入 [CLS] Token 与位置编码
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        #创建一个形状为 (batch_size, 1, width) 的全零张量，batch 大小与 x 相同。然会通过广播，将一个一维的分类 token 扩展成 (batch_size, 1, width) 的形状，然后通过 torch.cat 强行“粘”在 49 个特征的最前面。
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND  (NLD (Batch, Length, Dim) -> LND (Length, Batch, Dim))
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])
        #把第 0 号 Token（也就是 [CLS] Token）给切了出来（这里已经“吸干”了后面 49 个图像碎片的全部全局核心语义），然后做一次最后的归一化：ln_post

        if self.proj is not None:
            x = x @ self.proj # 矩阵乘法：将 768 维特征投射到指定的 output_dim

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length # 文本固定上下文长度

        #如果 vision_layers 传入的是一个元组，说明要构建 ResNet；如果传入的是一个整数，说明要构建由 12 层组成的 Vision Transformer。
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64 #（vision_width // 64 是大模型常用的 Heads 设定规律）
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )


        # 实例化文本端的 Transformer 引擎，并传入下面要生成的 attn_mask
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width) # 词嵌入层
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width) # 文本走出 Transformer 后的最终归一化

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim)) # 文本投影矩阵
        #文本特征出来的维度是 transformer_width（如 512），而视觉出来的特征维度是 embed_dim（如 512）。这两个多模态向量空间必须大小完全一致才能算相似度，这个投影矩阵就是负责把文本维度映射到与视觉对齐的公共空间的桥梁。
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) # 可学习的温度系数 exp(τ)
        # 初值设为 ln(1 / 0.07) ~= 2.659。后面用 exp() 展开后就是标准对比学习里的常数 1/0.07 ~= 14.28。这是一个可学习的温度调节系数，能让相似度矩阵的数值差异更明显，防止 Softmax 梯度饱和。

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    #在 ResNet 块的最后一层 BatchNorm 上将权重初始化为 0。这是大模型训练的绝招：让网络在刚开始训练时，每个残差块的初始输出就是原始输入（因为主干道被零初始化直接干掉了），这能保证深层网络在刚开局时像浅层网络一样稳定好训。
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)


    #建立因果掩码（Causal Mask）
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf")) # 全填上负无穷
        mask.triu_(1)  # zero out the lower diagonal
        # 保持上三角为负无穷，其余下三角和对角线刷成 0
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    # 输入图片，吐出视觉核心向量（形状：[Batch, embed_dim]）
    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) # 归一化

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        # 精准摘取 EOT (End of Text) 标志位的特征，一句话在被 Tokenize 后，最后一个有效单词后面紧跟的都是结束标记 [EOT]。由于 [EOT] 的 Token ID 在句子里通常数值最大（或者是通过特定规则填充），通过 argmax 就能精准抓到每句话真正结束处的那个索引位置。
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image) # 形状: [Batch, embed_dim]
        text_features = self.encode_text(text) # 形状: [Batch, embed_dim]

        # normalized features （L2 Norm）
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        #将向量的模长全部缩放到 1。这样后续做矩阵乘法点积时，算出来的结果就是纯粹的余弦相似度（Cosine Similarity），取值严格限制在 [-1, 1] 之间。

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t() #最终产生一个 [Batch, Batch] 的巨型网格矩阵
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

# 将模型的所有权重脱胎换骨地转换为半精度（FP16）以极大地压榨显卡算力
def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        # 如果是常规的卷积层或全连接线性层，直接全转为半精度
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        # 如果是 PyTorch 自带的多头注意力层，把里面所有的投影矩阵权重和偏置全转为半精度
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        # 如果是 CLIP 自定义的一些特定投影矩阵（如文本和视觉的公共空间投影面），也转为半精度
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    # model.apply 是 PyTorch 的内置方法，它会递归地遍历模型里的每一个子模块（Layer），
    # 并对它们逐一执行 _convert_weights_to_fp16 函数。
    model.apply(_convert_weights_to_fp16)

# 扔给它一个预训练好的权重字典 state_dict，它能自己查算出来这个模型当年是用什么参数训练的。
def build_model(state_dict: dict):
    # 检查字典里有没有 "visual.proj" 这个键，如果有，说明这是 Vision Transforme派系；如果没有，说明是 ResNet 派系。
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0] # 卷积核的输出通道数，就是视觉特征的宽度 width
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")]) # 多少个注意力权重键，（ ViT 一共有多少层 layers）
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1] # 卷积核的单边大小，就是拼图碎片的尺寸 patch_size
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5) # 拿位置编码总数减去1（CLS Token占的位置），再开平方，反推出网格单边格子数
        image_resolution = vision_patch_size * grid_size # 格子数 * 拼图大小 = 原始图像分辨率
    else:
        # 分别计算 layer1, 2, 3, 4 内部包含了多少个重复的残差块（Bottleneck）
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        # 从注意力池化（AttentionPool2d）的位置编码里反推网格大小
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    # 反推文本端和公共空间的参数
    embed_dim = state_dict["text_projection"].shape[1] # 投影矩阵的输出列数，就是公共对齐空间的维度
    context_length = state_dict["positional_embedding"].shape[0] # 文本位置编码的行数，就是最大文本长度
    vocab_size = state_dict["token_embedding.weight"].shape[0] # 词嵌入矩阵的行数，就是词表大小
    transformer_width = state_dict["ln_final.weight"].shape[0] # 文本最终归一化层的通道数，就是文本 Transformer 的宽度
    transformer_heads = transformer_width // 64 # 每个 Head 固定分配 64 维，由此反推有多少个 Attention Heads
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks"))) # 有多少个文本残差块，得到文本 Transformer 的总层数

    # 把所有反推出来的超参数喂给 CLIP 构造函数，建一个“空壳”模型
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    ## 删掉权重字典里一些用于记录信息、但不是真正网络权重的元数据键，防止报错
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model) #全面换装 FP16 半精度
    model.load_state_dict(state_dict) # 把解密好的预训练权重灌注到模型里去
    return model.eval() #切换到“推理/评估模式”
