import torch
import torch.nn as nn
import torch.optim as optim
import clip

# 1. 基础环境配置 (针对你的 Mac M5 加速)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🔄 1. 正在初始化设备... 激活 Mac GPU 加速: {device}")

# 2. 从虚拟环境中加载官方 CLIP 模型并转换为 float32 防止 Mac 精度冲突
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.float()


# -------------------------------------------------------------
# 🧩 2. 核心学术还原：实现官方位置拼接的 PromptLearner
# -------------------------------------------------------------
class StandardPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=4):
        super().__init__()
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  # 512

        # 🚨 核心可学习参数（Generic 模式）：4 * 512 矩阵
        self.ctx = nn.Parameter(torch.empty(n_ctx, ctx_dim, dtype=dtype))
        nn.init.normal_(self.ctx, std=0.02)

        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        # 记录 Token ID 用来在 TextEncoder 里定位 [EOS]
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

        with torch.no_grad():
            embedding = clip_model.token_embedding(self.tokenized_prompts).float()

        # 提取固定的 SOS 和 包含 CLS/EOS 的后缀
        self.register_buffer("token_prefix", embedding[:, :1, :])  # [SOS]
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # [CLS] + [EOS] + padding

    def forward(self):
        # 广播给所有类别
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        # 官方标准的 "end"（前缀提示）拼接模式
        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
        return prompts


# -------------------------------------------------------------
# 🔄 3. 数据流与前向传播还原（对应官方 CustomCLIP 与 TextEncoder）
# -------------------------------------------------------------
class CoOpSystem(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = StandardPromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_projection = clip_model.text_projection
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.logit_scale = clip_model.logit_scale

    def forward(self, image):
        # A. 提取图像特征
        image_features = self.image_encoder(image.float())

        # B. 动态拼接并提取文本特征（还原官方 TextEncoder 逻辑）
        prompts = self.prompt_learner()
        x = prompts + self.positional_embedding.type(model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(model.dtype)

        # 精准抠出 [EOS] 位置特征进行矩阵乘法
        text_features = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        # C. L2 归一化与相似度计算 (Logits)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = self.logit_scale.exp() * image_features @ text_features.t()
        return logits


# -------------------------------------------------------------
# 🧪 4. 跑起来微调实验（全流程闭环）
# -------------------------------------------------------------
classnames = ["airplane", "automobile", "bird"]
coop_model = CoOpSystem(classnames, model).to(device)

# 🚨 冻结大模型，只把 prompt_learner 的 ctx 参数喂给优化器
for name, param in coop_model.named_parameters():
    if "prompt_learner.ctx" not in name:
        param.requires_grad = False

optimizer = optim.AdamW([coop_model.prompt_learner.ctx], lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("\n🚀 2. 模拟真实数据输入，开始执行前向与反向传播迭代...")

for epoch in range(1, 6):
    mock_image = torch.randn(1, 3, 224, 224).to(device)
    mock_label = torch.tensor([0]).to(device)

    outputs = coop_model(mock_image)
    loss = criterion(outputs, mock_label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(
        f" 📈 迭代 [Epoch {epoch}/5] -> 仿真 Loss: {loss.item():.4f} | ctx 梯度范数: {coop_model.prompt_learner.ctx.grad.norm().item():.2f}")

print("\n🎉 3. 实验成功闭环！数据流完全跑通，可学习参数已成功更新。")