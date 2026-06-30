import torch
import torch.nn as nn
import torch.optim as optim
import clip

device = "mps" if torch.backends.mps.is_available() else "cpu"
# 加载时强行指定使用 float32，防止 Mac mps 设备上出现精度转换冲突
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.float()

# 1. 彻底冻结 CLIP 官方模型的所有参数
for param in model.parameters():
    param.requires_grad = False


# -------------------------------------------------------------
# 🧩 核心模块：官方标准的 CoOp Prompt Learner 实现
# -------------------------------------------------------------
class CoOpPromptLearner(nn.Module):
    def __init__(self, class_names, clip_model, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_classes = len(class_names)

        token_embedding = clip_model.token_embedding
        embedding_dim = token_embedding.embedding_dim  # 512

        # 核心：定义连续、可学习的 Context 向量 (CoOp 的灵魂)
        self.ctx = nn.Parameter(torch.empty(num_tokens, embedding_dim))
        nn.init.normal_(self.ctx, std=0.02)

        # 记录类别对应的 Token ID，用于后续定位 EOS
        self.tokenized_prompts = torch.cat([clip.tokenize(f"X {c}") for c in class_names]).to(device)

        with torch.no_grad():
            embedding_classes = token_embedding(self.tokenized_prompts).float()

        # 提取官方自带的 [SOS] 和 [EOS] 特征以及类别自身的 Embedding
        self.register_buffer("token_prefix", embedding_classes[:, :1, :])  # [SOS]
        self.register_buffer("token_suffix", embedding_classes[:, 1 + num_tokens:, :])  # [CLS] + [EOS] + padding

    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.num_classes, -1, -1)  # [Num_Classes, 4, 512]
        # 拼接成完整的输入：[SOS] + [可学习的 Context] + [类别与结束符]
        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
        return prompts


# -------------------------------------------------------------
# 🧪 任务二：跑起来一个完整的训练/微调实验
# -------------------------------------------------------------
class_names = ["cat", "dog", "sports car"]
prompt_learner = CoOpPromptLearner(class_names, model, num_tokens=4).to(device)

# 🚨 只把可学习的 prompt_learner.ctx 送进优化器
optimizer = optim.AdamW([prompt_learner.ctx], lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("🚀 开始模拟一个 Epoch 的微调实验...")

# 模拟下游任务的数据：输入一张随机图片（假设它是 cat，即标签为 0）
mock_image = torch.randn(1, 3, 224, 224).to(device)
mock_label = torch.tensor([0]).to(device)  # 真实标签是 cat

# ---- 前向传播 (Forward) ----
# 1. 组装最新的可学习文本特征
prompts_embeddings = prompt_learner()  # [3, 77, 512]

# 2. 经过 CLIP 完整的文本 Transformer 骨干网络
x = prompts_embeddings + model.positional_embedding.type(model.dtype)
x = x.permute(1, 0, 2)  # NLD -> LND
x = model.transformer(x)
x = x.permute(1, 0, 2)  # LND -> NLD
x = model.ln_final(x).type(model.dtype)

# 3. 提取 EOS 特征并进行投影矩阵乘法 (@)
text_features = x[torch.arange(x.shape[0]), prompt_learner.tokenized_prompts.argmax(dim=-1)] @ model.text_projection

# 4. 提取图像特征
image_features = model.encode_image(mock_image)

# 5. 计算图像与 3 个类别文本的相似度 logits
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)
logits = (image_features @ text_features.T) * model.logit_scale.exp()

# ---- 反向传播与更新 (Backward & Optimize) ----
loss = criterion(logits, mock_label)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("✅ 实验成功跑通！")
print(f"📊 当前仿真 Loss: {loss.item():.4f}")
print("🔍 检查 ctx 梯度的范数 (确保参数真的被训练更新了):", prompt_learner.ctx.grad.norm().item())
print("--------------------------------------------------\n")