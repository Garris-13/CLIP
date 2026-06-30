import torch
import torch.nn as nn
import clip

# 1. 检测并激活 Mac M5 的 GPU 加速设备 (MPS)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🎉 成功检测并激活设备: {device}")

# 2. 从虚拟环境中加载官方的标准 CLIP 模型
print("正在加载官方 CLIP ViT-B/32 模型...")
model, preprocess = clip.load("ViT-B/32", device=device)

# 3. 🎯 核心科研动作：全量冻结原生 CLIP 的所有参数（不让它们参与反向传播）
for param in model.parameters():
    param.requires_grad = False


# -------------------------------------------------------------
# 🧩 核心模块：定义一个可学习的 Prompt 模块 (CoOp 的核心逻辑)
# -------------------------------------------------------------
class SimplePromptLearner(nn.Module):
    def __init__(self, num_tokens=4, embedding_dim=512):
        super().__init__()
        # 连续的、可学习的 Token 向量。这就是 CoOp 的灵魂！
        # 在训练时，只有这部分参数会计算梯度并被 Optimizer 更新。
        self.ctx = nn.Parameter(torch.empty(num_tokens, embedding_dim))
        # 使用正态分布随机初始化这个可学习的矩阵
        nn.init.normal_(self.ctx, std=0.02)

    def forward(self, class_embeddings):
        # class_embeddings 的 shape 是 [Num_Classes, 512]
        # 核心逻辑：这里我们计算可学习 ctx 向量的平均值，并与类别向量结合
        ctx_mean = self.ctx.mean(dim=0).unsqueeze(0)  # 变换维度为 [1, 512]

        # 利用 PyTorch 的广播机制相加，得到注入了“可学习上下文”的新特征
        prompted_embeddings = class_embeddings + ctx_mean  # 结果为 [Num_Classes, 512]
        return prompted_embeddings


# -------------------------------------------------------------
# 🧪 模拟运行：验证 Shape 和可训练参数
# -------------------------------------------------------------
# 假设我们下游任务有 3 个类别
class_names = ["cat", "dog", "sports car"]
tokenized_classes = clip.tokenize(class_names).to(device)

with torch.no_grad():
    # 提取纯类别未加 Prompt 的原始文本特征
    raw_class_features = model.encode_text(tokenized_classes)  # 维度：[3, 512]

# 实例化我们的 Prompt 学习器
prompt_learner = SimplePromptLearner(num_tokens=4, embedding_dim=512).to(device)

# 获得经过 CoOp 思想调整后的新文本特征
enhanced_text_features = prompt_learner(raw_class_features)

print("\n--- 📊 维度与参数检查 ---")
print(f"原始类别特征 Shape (raw_class_features): {raw_class_features.shape}")
print(f"注入可学习 Prompt 后 Shape (enhanced_text_features): {enhanced_text_features.shape}")

print("\n检查当前网络中真正【可被梯度训练（Requires Grad）】的参数：")
trainable_params_count = 0
for name, param in prompt_learner.named_parameters():
    if param.requires_grad:
        print(f" -> 参数名: {name} | Shape: {param.shape} | 是否需要梯度: {param.requires_grad}")
        trainable_params_count += param.numel()

print(f"\n💡 结论：整个巨大的 CLIP 模型已被完全冻结。")
print(f"   我们现在只需要训练这短短 {trainable_params_count} 个浮点数参数（仅占极小存储），")
print(f"   就能实现针对特定下游任务的微调！这就是 Parameter-Efficient Tuning 的魅力。")
print("--------------------------------------------------\n")
