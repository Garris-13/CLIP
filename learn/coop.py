import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # Transformer 前向（原生 CLIP 需要 permute 为 [seq_len, batch, dim]）
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    # 目的：定义出那个需要被优化的参数 self.ctx，并准备好不参与训练的句子前后缀

    # 接收三个参数：cfg（配置字典/对象，包含超参数）、classnames（当前数据集的类别名称列表，如 ["dog", "cat"]）、clip_model（加载好的预训练 CLIP 模型）
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames) # 统计类别总数
        n_ctx = cfg.TRAINER.COOP.N_CTX # 从配置中读取 CoOp 设定的可学习上下文 Token 数量 N_CTX（通常设为 4 或 16），即论文中的 M
        ctx_init = cfg.TRAINER.COOP.CTX_INIT # 从配置中读取初始化上下文的文本字符串（例如 "a photo of a"）。如果为空字符串，则采用随机初始化
        dtype = clip_model.dtype # 获取 CLIP 模型使用的数据类型
        ctx_dim = clip_model.ln_final.weight.shape[0] # 获取 CLIP 文本编码器最终层（ln_final LayerNorm）的权重维度。实际上取的是词嵌入的特征维度（CLIP-ViT-B/16 中为 512，CLIP-ViT-L/14 中为 768）
        clip_imsize = clip_model.visual.input_resolution # 读取 CLIP 视觉编码器要求的输入图像尺寸（如 224）
        cfg_imsize = cfg.INPUT.SIZE[0] # 读取用户配置文件中设定的图像尺寸
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        # 断言检查，强制要求用户配置的图像尺寸必须与 CLIP 模型原生尺寸一致。因为 CoOp 不重训视觉编码器，尺寸变了会导致位置编码报错。

        if ctx_init:  # 是否指定了初始化文本字符
            # use given words to initialize context vectors
            # 方式 A：指定单词初始化（例如给定了 "a photo of a"）
            ctx_init = ctx_init.replace("_", " ") # 将初始化文本中的下划线替换为空格（例如 "a_photo_of_a" 变为 "a photo of a"），方便 BPE 分词器处理。
            n_ctx = len(ctx_init.split(" ")) # 重新计算 n_ctx 的值为实际初始化文本包含的单词个数（覆盖配置文件中的设定）。例如 "a photo of a" 算出来是 4。
            prompt = clip.tokenize(ctx_init) # 调用 CLIP 内置的分词器（BPE），将文本转换为 Token ID 张量，形状为 [1, 总token数]，包含 [SOS] 开始符和 [EOS] 结束符。
            with torch.no_grad(): # 禁用梯度计算上下文管理器，确保接下来的操作不会构建计算图（节省显存且加速）
                embedding = clip_model.token_embedding(prompt).type(dtype) # 查 CLIP 的词嵌入表（token_embedding），将 Token ID 映射为稠密向量，形状为 [1, 总token数, ctx_dim]，并转换精度。
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :] # 核心切片操作。提取第 0 个样本，跳过第一个 Token（即 [SOS]），然后连续取出 n_ctx 个 Token 对应的向量（即 "a" "photo" "of" "a" 的词嵌入）。这组向量将作为可学习参数的初始值。
            prompt_prefix = ctx_init # 记录当前初始化的文本内容，用于后续打印日志。

        else: # 随机初始化分支
            # random initialization

            #  CSC（Class-Specific Context，类别特定上下文） 模式
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype) # 创建空的张量，形状为 [类别数, M, 特征维度]。意味着每一个类别都拥有自己专属的 M 个上下文向量，总参数量为 n_cls * M * dim。
            # 统一上下文模式
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype) # 创建空的张量，形状仅为 [M, 特征维度]。所有类别复用这一组向量，总参数量为 M * dim
            nn.init.normal_(ctx_vectors, std=0.02) # 使用均值为 0、标准差为 0.02 的正态分布随机填充 ctx_vectors。这是论文中推荐的随机初始化方式。
            prompt_prefix = " ".join(["X"] * n_ctx) # 将提示前缀设为 "X X X ..."（共 n_ctx 个 X），仅用于日志打印显示占位符，实际计算使用的是随机向量。

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # 将 ctx 注册为可训练参数
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        # 形状为 (M, D)， 上下文 Token 长度设为 M（一般会认为设定为4或16），CLIP 的词嵌入维度是 D（512） 时
        # 这就是论文里最核心的 [V]_1 [V]_2 ... [V]_M。它是代码里唯一会被加入优化器并回传梯度更新的变量。
        # 将上一步得到的 ctx_vectors 封装成 nn.Parameter，这意味着它会被自动添加到模型的 .parameters() 迭代器中，并在后续的优化器（如 SGD/Adam）中被梯度更新。这是 CoOp 能够优化的根基。


        classnames = [name.replace("_", " ") for name in classnames] # 格式化类别名，将下划线替换为空格

        # 使用 CLIP 的 BPE 分词器对每个类别名进行编码，计算每个类别名占用了几个子词（Subword）。例如 "husky" 可能占 1 个 Token，但 "golden retriever" 可能占 3 个 Token。
        # 这个长度后续用于从后缀中精准切分类别词。
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        # 构建初始的完整提示模板列表（仅用于提取嵌入，不参与训练）。例如 "X X X dog." 或 "a photo of a dog."（末尾加句号符合 CLIP 预训练格式）。
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        # 对每个类别的完整提示文本进行分词，并将所有结果的 Token ID 张量在行（第 0 维）上拼接，得到形状为 [n_cls, 最大token数] 的二维张量。
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad(): # 再次禁用梯度，因为接下来提取的固定前后缀向量不需要更新
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # 将上述 Token ID 送入 CLIP 词嵌入表，提取对应的稠密向量，形状为 [n_cls, 总序列长度, ctx_dim]。

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        # 从嵌入中提取每个类别序列的第一个 Token 向量（即 [SOS] 开始符），形状 [n_cls, 1, dim]。register_buffer 将其注册为模型的持久化状态（保存在 state_dict 中），但不参与梯度更新。

        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        # 从嵌入中提取跳过 [SOS] 和所有可学习上下文 ctx 之后的所有剩余 Token，即 [类别词, ..., EOS]，形状 [n_cls, 剩余长度, dim]。同样固定不变。

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        # 将计算得到的类别数、上下文长度、分词结果、名称长度、以及类别词摆放位置（end / middle / front）保存为成员变量，供 forward 方法调用。

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            #  0 维插入一个维度变为 [1, M, dim]，然后调用 expand 广播为 [n_cls, M, dim]。注意 expand 不复制内存，只是视图（View），使得后续可以按类别维度统一拼接。

        # 从缓冲区取出固定的前缀（[SOS]）和后缀（类别词 + [EOS]）。
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end": # 判断类别词放在最后（这是论文的默认方案）
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            # 在序列维度（dim=1）上拼接，得到序列顺序为 [SOS] + [上下文向量] + [类别词 + EOS]。最终输出形状 [n_cls, 1 + M + 后缀长度, dim]。

        # “类别词放在中间”
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2 # 将上下文向量平分成前后两半（整数除法，如果 M 为奇数，前半部分少一个）。
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i] # 取出当前类别的名称长度（占几个 BPE Token）
                prefix_i = prefix[i : i + 1, :, :] # 取出当前类别的 [SOS] 向量，保留维度 [1, 1, dim]。
                class_i = suffix[i : i + 1, :name_len, :] # 从当前类别的后缀中，切出开头的 name_len 个 Token，即类别词本身的嵌入，形状 [1, name_len, dim]。
                suffix_i = suffix[i : i + 1, name_len:, :] # 从当前类别的后缀中，切出剩余的部分（即 [EOS] 等），形状 [1, 剩余长度, dim]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :] # 取出当前类别的上下文向量的前半部分，形状 [1, half_n_ctx, dim]。
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :] # 取出当前类别的上下文向量的后半部分，形状 [1, half_n_ctx, dim]。
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                ) # 按 [SOS] + [上下文前半] + [类别词] + [上下文后半] + [EOS] 的顺序拼接，构成该类别最终输入的 Token 嵌入序列。
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0) # 所有类别在 0 维（批量/类别维）拼接起来，恢复形状为 [n_cls, 总长度, dim]。

        # 进入“类别词放在最前面”的分支（紧跟在 [SOS] 后）。
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
             # 逻辑类似，只是拼接顺序变为 [SOS] + [类别词] + [全部上下文] + [EOS]。

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model) # CoOp 的灵魂，它负责维护可学习的上下文向量 ctx，并在前向时生成拼接好的完整 Prompt 词嵌入。
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        # 将 PromptLearner 中预处理好、包含所有类别完整提示的 Token ID 张量（形状为 [n_cls, max_seq_len]）引用过来。
        # 注意：这里存储的是 Token ID（整数），而不是嵌入向量。它主要用于后续在 TextEncoder 中定位 [EOS]（结束符）的位置。
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model) # 实例化一个自定义的文本编码器（TextEncoder），传入 clip_model 中的文本相关组件
        self.logit_scale = clip_model.logit_scale
        # 引用 CLIP 模型中的可学习温度系数 logit_scale（对数标量）。CLIP 在对比学习时会乘以这个系数来缩放相似度（logits = scale * img_feat @ text_feat.T）。该参数通常也会在微调时参与训练。
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype)) # 定义前向传播函数，接收一个批次（Batch）的图像张量 image，形状通常为 [batch_size, 3, H, W]。

        prompts = self.prompt_learner() # 🌟
        # 调用 prompt_learner 的 forward()，获取当前（经过训练迭代后）完整的提示词嵌入张量。
        # 输出形状为 [n_cls, 序列长度, ctx_dim]（例如 [1000, 77, 512]）。这个张量已经是稠密的浮点数，而非整数 ID。

        tokenized_prompts = self.tokenized_prompts # Token ID 张量。它不会随着训练更新，仅作为一个固定参考传入 text_encoder，用于找出每个提示序列中 [EOS] 标记的具体位置
        text_features = self.text_encoder(prompts, tokenized_prompts) # 上面生成的提示嵌入 prompts 和 Token IDs tokenized_prompts 送入自定义文本编码器。
        # text_encoder 内部会执行以下操作
        # 加上可学习的位置编码（positional_embedding）。
        # 送入 Transformer 进行自注意力计算。
        # 经过 ln_final。
        # 根据 tokenized_prompts 取出每个序列 [EOS] 位置的向量。
        # 乘以 text_projection 矩阵（投影到与图像特征相同的多模态空间）。
        # 输出 text_features 形状为 [n_cls, 特征维度]（如 [1000, 512]）。

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # 对文本，图像特征进行 L2 归一化
        # 这使得所有文本特征向量都被映射到单位超球面上。目的是后续计算余弦相似度时，消除向量模长带来的影响，只关注方向。

        logit_scale = self.logit_scale.exp() # 将可学习的对数温度系数 logit_scale 取指数（exp），强制其为正值
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]
        # 断言检查，强制要求配置文件中 TRAINER.COOP.PREC 参数的值必须是 "fp16"、"fp32" 或 "amp" 三者之一。

    def build_model(self):
        cfg = self.cfg # 获取当前实例的配置对象，赋值给局部变量 cfg，方便后续频繁调用（缩短代码长度）
        classnames = self.dm.dataset.classnames # 通过数据管理器（self.dm）获取当前数据集的类别名称列表（例如 ["dog", "cat", ...]）。这里的 dm 是父类 TrainerX 在初始化时创建的数据模块。

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        # 调用辅助函数 load_clip_to_cpu，根据配置加载预训练的 CLIP 模型（默认先加载到 CPU 以节省显存，后续再移动到设备）。这个函数通常会处理 openai 或 custom 的权重路径。
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        # 核心实例化。将配置、类别列表和（可能转换过精度的）CLIP 模型传入，创建CustomCLIP 对象。这个对象包含了 PromptLearner和固定的文本/图像编码器。

        print("Turning off gradients in both the image and the text encoder")
        # 锁死除 prompt_learner 以外的一切参数（包括文本和视觉分支）
        # 参数冻结的关键循环。遍历 CustomCLIP 中的所有参数（named_parameters），如果参数的名字中不包含子字符串 "prompt_learner"，则将其 requires_grad 属性设置为 False。
        # CLIP 的视觉编码器（image_encoder）、文本编码器（text_encoder）的内部参数、以及 logit_scale 都被冻结，不参与梯度更新。
        # 只有 CustomCLIP 中 prompt_learner 模块下的 self.ctx（上下文向量）可以接收梯度。
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
            # 加载预训练权重到 prompt_learner 中。这通常用于在多个数据集间迁移训练好的上下文向量，实现“提示迁移”。

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        # 只将 prompt_learner 的参数传给优化器构建函数 build_optimizer。这意味着优化器的参数列表里只包含可学习的上下文向量（ctx），数量极少（通常只有几 KB）。这也解释了为什么 CoOp 显存占用小且训练快。
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM) # 使用同样的优化器构建学习率调度器（如 CosineAnnealingLR、StepLR 等），用于在训练过程中动态调整学习率。
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        # 调用父类的方法，将 prompt_learner、优化器和调度器注册到训练器的管理系统中。这样父类提供的保存检查点（Checkpoint）、恢复训练等功能就能自动包含这些组件。

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    # 单步训练的核心逻辑。父类 TrainerX 的训练循环会在每个迭代中调用这个方法，传入一个批次的数据字典。
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch) # 从批次数据中解析出图像张量和标签张量。
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp": # 进入 AMP 提供的 autocast() 上下文管理器。
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            # 在该上下文中，PyTorch 会自动为合适的算子（如 Conv、MatMul）选择 float16 执行以加速，而对不安全的算子保持 float32。随后执行模型前向得到 logits，并计算交叉熵损失。
            self.optim.zero_grad() # 清空上一轮迭代的梯度，防止梯度累加。
            self.scaler.scale(loss).backward() # 对损失值进行缩放（乘以一个大系数），然后调用反向传播。缩放是为了防止在 float16 下梯度下溢（变成 0）。
            self.scaler.step(self.optim) # 更新优化器。
            self.scaler.update()
        else: # 常规训练逻辑，执行常规前向、计算损失，然后调用父类的 model_backward_and_update 方法（该方法封装了 loss.backward()、梯度裁剪和 optimizer.step()）。
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        # 判断当前是否是一个 Epoch 中的最后一个批次（batch_idx 从 0 开始计数）。如果是，则调用 update_lr 方法更新学习率（例如执行学习率调度器的 step()）。
        # 部分调度器（如 StepLR）按 Epoch 更新，放在此处正合适。
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
