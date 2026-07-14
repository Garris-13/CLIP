# TDA 源码学习导读

论文：Efficient Test-Time Adaptation of Vision-Language Models, CVPR 2024

官方源码已克隆到：`learn/TDA`

## 1. 先读 README，建立整体图景

入口文件：

- `learn/TDA/README.md`
- `learn/TDA/docs/main_figure.png`

这篇工作的核心思想是：不训练、不反传，在测试过程中维护一个动态 key-value cache。key 是测试样本的图像特征，value 是伪标签或负标签掩码。后续测试样本会查询这个缓存，把 CLIP 原始 logits 和缓存 logits 融合起来。

## 2. 核心源码阅读顺序

### Step 1：运行入口

文件：`learn/TDA/tda_runner.py`

重点函数：

- `main()`
- `run_test_tda()`

`main()` 做四件事：

1. 读取命令行参数。
2. 加载 CLIP backbone。
3. 根据数据集构建测试集和文本分类器权重。
4. 调用 `run_test_tda()` 执行测试时自适应。

真正的算法循环在 `run_test_tda()`。

### Step 2：理解单张测试图像如何进入 CLIP

文件：`learn/TDA/utils.py`

重点函数：

- `clip_classifier()`
- `get_clip_logits()`

`clip_classifier()` 把类别名通过 prompt template 编码成文本特征，得到 CLIP 的分类权重。

`get_clip_logits()` 做图像编码，并计算：

- `image_features`：归一化后的图像特征。
- `clip_logits`：原始 CLIP 分类分数。
- `loss`：熵，用来判断预测是否可靠。
- `prob_map`：类别概率分布。
- `pred`：当前预测类别。

如果输入包含多视角增强结果，它会选熵最低的前 10% 视图做聚合。

### Step 3：理解正缓存 Positive Cache

文件：`learn/TDA/tda_runner.py`

重点位置：

- `update_cache()`
- `compute_cache_logits()`
- `run_test_tda()` 中 positive cache 相关逻辑

正缓存保存模型比较确信的伪标签样本：

- key：图像特征。
- value：预测类别。
- 每个类别最多保存 `shot_capacity` 个样本。
- 如果缓存已满，会优先保留熵更低、更可靠的样本。

使用时，当前图像特征会和缓存 key 做相似度匹配，再把匹配结果转成 cache logits，加到原始 CLIP logits 上。

### Step 4：理解负缓存 Negative Cache

文件：`learn/TDA/tda_runner.py`

重点位置：

- `run_test_tda()` 中 negative cache 的熵阈值判断。
- `compute_cache_logits(..., neg_mask_thresholds=...)`

负缓存处理“模型不够确定，但又有一些类别可以排除”的情况。

进入负缓存需要满足：

```python
lower_entropy < normalized_entropy < upper_entropy
```

负缓存保存的是概率分布 `prob_map`，之后通过 `mask_threshold` 生成负类掩码。最终负缓存 logits 会从原始结果中减掉，从而抑制某些不可信类别。

### Step 5：看配置如何控制算法

目录：`learn/TDA/configs`

示例：`learn/TDA/configs/imagenet.yaml`

关键配置：

- `positive.shot_capacity`
- `positive.alpha`
- `positive.beta`
- `negative.shot_capacity`
- `negative.alpha`
- `negative.beta`
- `negative.entropy_threshold`
- `negative.mask_threshold`

可以把配置理解成 TDA 的旋钮：

- `shot_capacity` 控制每类缓存多少测试样本。
- `alpha` 控制缓存 logits 的影响强度。
- `beta` 控制相似度到权重的尖锐程度。
- `entropy_threshold` 控制哪些样本可以进入负缓存。
- `mask_threshold` 控制哪些类别会被视为负类。

## 3. 最小核心调用链

```text
scripts/*.sh
  -> tda_runner.py main()
    -> build_test_data_loader()
    -> clip_classifier()
    -> run_test_tda()
      -> get_clip_logits()
      -> update_cache(pos_cache)
      -> update_cache(neg_cache)
      -> compute_cache_logits(pos_cache)
      -> compute_cache_logits(neg_cache)
      -> final_logits = clip_logits + pos_logits - neg_logits
```

## 4. 建议你重点对照论文看的代码段

1. Dynamic Adapter / key-value cache：
   `tda_runner.py` 的 `update_cache()` 和 `compute_cache_logits()`。

2. Progressive pseudo label refinement：
   `run_test_tda()` 中每个测试样本都会更新 cache，然后立刻用 cache 改善后续预测。

3. Negative pseudo labeling：
   `run_test_tda()` 中 negative cache 的熵筛选，以及 `compute_cache_logits()` 中根据概率阈值生成 mask 的逻辑。

4. Efficient / no backpropagation：
   整个 `run_test_tda()` 都包在 `torch.no_grad()` 里，没有优化器，没有梯度更新。

## 5. 下一步学习建议

如果只想先吃透核心算法，可以暂时不看 `clip/` 和 `datasets/` 目录，先集中读：

1. `learn/TDA/tda_runner.py`
2. `learn/TDA/utils.py`
3. `learn/TDA/configs/imagenet.yaml`
4. `learn/TDA/scripts/run_ood_benchmark_rn50.sh`

读完这四处，基本就能把 TDA 的源码主线串起来。
