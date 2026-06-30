import torch
from PIL import Image
import urllib.request

# 1. 自动检测并指定 Mac 的 GPU 加速设备 (MPS)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🎉 成功检测到 Mac M5 GPU 加速 (MPS)！")
else:
    device = torch.device("cpu")
    print("⚠️ 未检测到 MPS，将使用 CPU 运行。")

# 2. 加载模型 (下载可能需要 1-2 分钟，请保持网络畅通)
print("正在加载 CLIP ViT-B/32 模型...")
model, preprocess = clip.load("ViT-B/32", device=device)

# 3. 下载一张测试图片（这里下载一张小猫的图存到本地）
img_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
urllib.request.urlretrieve(img_url, "test_dog.jpg")

# 4. 准备输入数据
image = preprocess(Image.open("test_dog.jpg")).unsqueeze(0).to(device)
text_inputs = ["a photo of a cat", "a photo of a dog", "a photo of a sports car"]
text = clip.tokenize(text_inputs).to(device)

# 5. 前向传播（核心逻辑）
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # 打印导师要求关注的 Tensor Shape
    print("\n--- 关键维度检查 ---")
    print(f"图片特征 (image_features) 的 Shape: {image_features.shape}")
    print(f"文本特征 (text_features) 的 Shape: {text_features.shape}")
    print("---------------------------------\n")

    # 计算相似度 logits
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# 6. 打印结果
print("预测概率：")
for label, prob in zip(text_inputs, probs[0]):
    print(f" 预测为 [{label}]: {prob * 100:.2f}%")