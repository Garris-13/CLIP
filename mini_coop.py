import torch
import torch.nn as nn
import clip

device = "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 冻结整个 CLIP 模型的参数，不让它们参与梯度更新
for param in model.parameters():
    param.requires_grad = False


# -------------------------------------------------------------
# 🧩 核心：定义一个可学习的 Prompt 模块
# -------------------------------------------------------------
class SimplePromptLearner(nn.Module):
    def __init__(self, num_tokens=4, embedding_dim=512):
        super().__init__()
        # 1. 虚拟定义 4 个连续的、可学习的 Token 向量（这就是 CoOp 的灵魂）
        self.ctx = nn.Parameter(torch.empty(num_tokens, embedding_dim))
        # 随机初始化这些向量
        nn.init.normal_(self.ctx, std=0.02)

    def forward(self, class_embeddings):
        # class_embeddings 的 shape 是 [Num_Classes, 512]
        # 把可学习的 ctx 拼接到每一个类别的向量前面
        # 简化版逻辑：这里我们把 ctx 的平均值和类别向量相加，模拟 Prompt 融合
        num_classes = class_embeddings.shape[0]
        ctx_mean = self.ctx.mean(dim=0).unsqueeze(0)  # [1, 512]

        # 融合后的文本特征
        prompted_embeddings = class_embeddings + ctx_mean  # 广播机制变成 [Num_Classes, 512]
        return prompted_embeddings


# -------------------------------------------------------------
# 🧪 模拟运行
# -------------------------------------------------------------
# 假设我们有 3 个类别
class_names = ["cat", "dog", "car"]
tokenized_classes = clip.tokenize(class_names).to(device)

with torch.no_grad():
    # 提取纯类别的原始文本特征
    raw_class_features = model.encode_text(tokenized_classes)  # [3, 512]

# 实例化我们的 Prompt 学习器
prompt_learner = SimplePromptLearner(num_tokens=4, embedding_dim=512).to(device)

# 获得注入了“可学习 Prompt”后的新文本特征
enhanced_text_features = prompt_learner(raw_class_features)

print("\n--- 📊 CoOp 维度检查 ---")
print("原始类别特征 Shape:", raw_class_features.shape)  # [3, 512]
print("注入可学习 Prompt 后 Shape:", enhanced_text_features.shape)  # [3, 512]
print("查看哪些参数可以被梯度训练：")
for name, param in prompt_learner.named_parameters():
    print(f" -> 参数名: {name}, Shape: {param.shape}, 是否需要梯度: {param.requires_grad}")