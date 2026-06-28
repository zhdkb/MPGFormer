# MPGFormer 整体架构流程说明

本文档根据 `models/MPGFormer/mpgformer.py`、`modules/` 目录中的核心模块，以及 `main.tex` 方法部分的撰写逻辑，整理 MPGFormer 的完整前向流程。该流程可直接用于绘制模型架构图。

## 1. 整体输入与任务形式

MPGFormer 面向云工作负载预测任务。模型输入是一批滑动窗口样本：

```text
X ∈ R^{B × 72 × C}
```

其中：

- `B` 表示 batch size，也就是一次送入模型的窗口数量；
- `72 = 48 + 24`，前 48 个时间点是历史输入，后 24 个时间点是预测区间；
- `C` 表示动态特征通道数量，默认第 0 个通道是目标 workload；
- 目标是预测未来 24 个时间点的 target workload：

```text
ŷ ∈ R^{B × 24}
```

在数据构造阶段，未来 24 个时间点的 target 通道会被置为 0，防止模型看到真实未来值。其他已知动态特征可以保留。

## 2. Encoder / Decoder 输入切分

模型内部首先将输入窗口切分成 encoder 输入和 decoder 输入。

Encoder 输入是前 48 个历史时间点：

```text
X_enc ∈ R^{B × 48 × C}
```

Decoder 输入由两部分组成：

```text
X_dec ∈ R^{B × 36 × C}
```

其中：

- 前 12 个时间点来自历史窗口最后 12 个点；
- 后 24 个时间点是未来预测位置；
- 未来 24 个位置的 target 通道不会使用真实值，后续会被原型分支生成的 `future_padding` 替换。

因此，decoder input 可以理解为：

```text
Decoder Input = Last 12 History + Future 24 Masked Tokens
```

## 3. Target RevIN：目标通道窗口级归一化

MPGFormer 只对 target 通道做窗口级可逆归一化，而不是对所有动态特征都做归一化。

首先从 encoder input 中取出目标通道：

```text
target_history ∈ R^{B × 48}
```

对每个窗口单独计算均值和标准差：

```text
μ_w = mean(target_history)
σ_w = std(target_history)
```

然后归一化 target 通道：

```text
target_norm = (target_history - μ_w) / (σ_w + ε)
```

归一化后的 target 通道会替换回 encoder input，形成：

```text
X̃_enc ∈ R^{B × 48 × C}
```

这里需要注意：

- 只有目标通道被归一化；
- 其他动态特征保持原样；
- `μ_w` 和 `σ_w` 会保存下来，用于最后把预测结果反归一化回原始 workload 尺度。

## 4. 历史表示生成：从动态特征到隐藏表示

归一化后的 encoder input 会进入历史表示生成模块。

第一步是线性投影：

```text
Z = Linear(X̃_enc)
Z ∈ R^{B × 48 × 128}
```

这里的作用是把原始 `C` 维动态特征映射到统一的 128 维隐藏空间。

第二步加入位置编码：

```text
Z_pos = Z + PE
```

位置编码用于告诉模型每个 token 在 48 个历史时间点中的相对位置。

第三步融合静态上下文：

```text
Z_static = GatedStaticContext(Z_pos, s)
```

其中 `s` 是静态上下文，例如机器、应用、实例类型等信息。如果数据集中没有静态上下文，该模块退化为恒等映射。

最后经过 LayerNorm，得到历史隐藏表示：

```text
H_s ∈ R^{B × 48 × 128}
```

`H_s` 是后续时间证据分支、decoder memory 的基础表示。

## 5. Temporal Evidence Branch：时间证据分支

时间证据分支用于提取当前 workload 自身历史中的时间依赖、趋势变化和局部突发信息。

输入包括：

```text
H_s ∈ R^{B × 48 × 128}
target_history ∈ R^{B × 48}
```

### 5.1 非平稳注意力

该模块首先根据 target_history 的窗口统计量生成注意力调制参数：

```text
stats = [mean(target_history), std(target_history)]
[τ, δ] = MLP(stats)
```

其中：

- `τ` 是注意力温度，用来调节注意力分布的尖锐程度；
- `δ` 是注意力偏置，用来根据当前窗口状态调整注意力得分。

然后对 `H_s` 做 self-attention：

```text
Q, K, V = Linear(H_s)
A = softmax(τ · QK^T / sqrt(d_h) + δ)
AttnOut = A · V
```

这一步的作用是让模型根据当前窗口的负载状态，自适应地聚合不同历史时间点的信息。

### 5.2 Moving-average Decomposition

在每一层非平稳注意力中，模型还会对隐藏序列做移动平均分解：

```text
Trend = MovingAverage(X)
Seasonal = X - Trend
```

其中：

- `Trend` 表示低频趋势，反映平滑变化方向；
- `Seasonal` 表示高频残差，保留短期波动、突发和振荡。

Trend 会作为残差偏置加入注意力更新，Seasonal 会进入 FFN 更新。

### 5.3 Short-term Temporal Encoder

经过若干层非平稳注意力后，模型再使用短期时序编码器增强局部模式。

该模块包含：

```text
Conv1D(k=3)
Conv1D(k=5)
BiGRU
Concat + Linear
LayerNorm
```

其中：

- `Conv1D(k=3)` 捕捉非常短的局部变化；
- `Conv1D(k=5)` 捕捉稍宽的短期波动；
- `BiGRU` 捕捉相邻时间点之间的顺序依赖；
- `Concat + Linear` 将局部卷积信息和循环状态融合回 128 维。

最终得到：

```text
encoded ∈ R^{B × 48 × 128}
c_time = mean(encoded, dim=time)
c_time ∈ R^{B × 128}
```

其中：

- `encoded` 保留每个历史时间点的高维表示，后面作为 decoder memory 的主体；
- `c_time` 是窗口级时间证据向量，表示当前窗口自身历史能够提供的预测依据。

## 6. Multi-scale Prototype Branch：多尺度时频原型分支

多尺度原型分支用于检索训练过程中学到的典型 workload 形态。

输入是归一化后的目标历史：

```text
target_history ∈ R^{B × 48}
```

### 6.1 多尺度切片

模型使用四个尺度：

```text
p ∈ {6, 12, 24, 48}
```

以 `p=6` 为例，48 个历史点会被切成 8 个 patch：

```text
48 / 6 = 8
```

不同尺度对应不同粒度的工作负载形态：

- `p=6` 更关注短周期局部变化；
- `p=12` 关注中短期片段；
- `p=24` 关注半窗口级模式；
- `p=48` 关注整个历史窗口的整体形态。

### 6.2 时域 token 与频域 token

每个尺度下，模型会同时提取时域和频域信息。

时域路径：

```text
patches -> Linear -> patch tokens -> mean -> time token
```

频域路径：

```text
patches -> rFFT -> magnitude -> Linear -> mean -> frequency token
```

然后两者相加并经过 `tanh`：

```text
scale_token = tanh(time_token + frequency_token)
```

因此，每个尺度最终得到一个 scale token。

### 6.3 Prototype Memory 匹配

模型内部维护一个可学习的原型库：

```text
Prototype Memory M ∈ R^{4 × K × 128}
```

其中：

- 4 表示四个尺度；
- `K` 表示每个尺度的原型数量，训练默认是 48；
- 每个 prototype 是一个 128 维向量；
- prototype memory 是可学习参数，训练开始时随机初始化，之后通过反向传播更新。

每个 scale token 会与同尺度的 prototype 计算相似度：

```text
prototype_weights = softmax(scale_token · prototype^T)
```

然后加权读取原型：

```text
proto = prototype_weights · prototype_memory
```

再与当前 scale token 融合：

```text
proto_recovered = Recover(proto + scale_token)
```

注意：这里不是简单用 prototype 替换当前窗口，而是把检索到的 prototype 和当前 scale token 融合。

### 6.4 原型证据与未来初始填充

四个尺度的原型响应会通过 scale weights 聚合：

```text
c_proto = Σ scale_weight_p · proto_recovered_p
c_proto ∈ R^{B × 128}
```

同时，模型会用 `c_proto` 生成未来 24 个 target 位置的初始填充：

```text
future_padding = Linear(c_proto)
future_padding ∈ R^{B × 24}
```

这里需要强调：

- `c_proto` 是原型证据向量，给 router 使用；
- `future_padding` 是 decoder 的未来初始轮廓；
- `future_padding` 不是最终预测结果，最终预测还要经过 Transformer Decoder 和 Prediction Head。

## 7. Dynamic Graph Branch：动态图证据分支

动态图分支用于捕捉 batch 内不同窗口之间的跨序列协同关系。

输入包括：

```text
c_time ∈ R^{B × 128}
target_history ∈ R^{B × 48}
static_context optional
```

### 7.1 Workload Graph

模型根据 target_history 计算窗口之间的余弦相似度：

```text
sim_ij = cosine(target_history_i, target_history_j)
```

然后保留每个窗口最相似的 top-k 个邻居，得到 workload adjacency：

```text
workload_adj ∈ R^{B × B}
```

这张图表示：哪些 workload 窗口在近期形态上相似。

### 7.2 Context Graph

如果数据中存在静态上下文，模型使用静态上下文构建 context graph。

如果没有静态上下文，模型使用 target_history 的统计特征：

```text
mean
std
mean(|Δ|)
last - first
```

同样经过余弦相似度和 top-k 稀疏化，得到：

```text
context_adj ∈ R^{B × B}
```

### 7.3 GCN / Message Passing

两张图分别对 `c_time` 做图卷积：

```text
z_w = GCN(c_time, workload_adj)
z_c = GCN(c_time, context_adj)
```

直观理解：

- `z_w` 表示从负载形态相似窗口中聚合后的证据；
- `z_c` 表示从上下文相似窗口中聚合后的证据。

最后拼接并映射：

```text
c_graph = Linear(Concat(z_w, z_c))
c_graph ∈ R^{B × 128}
```

动态图分支还会计算 contrastive loss，使 workload graph 和 context graph 两种视角下的同一窗口表示更加一致。

## 8. Drift-aware Router：漂移感知路由器

MPGFormer 的核心思想是：不同窗口应该依赖不同证据。

Router 输入三类证据：

```text
c_time  ∈ R^{B × 128}
c_graph ∈ R^{B × 128}
c_proto ∈ R^{B × 128}
```

同时输入漂移特征：

```text
last-value z-score
mean absolute difference
target std
prototype entropy
```

其中：

- `last-value z-score` 衡量最后一个历史点是否偏离当前窗口均值；
- `mean absolute difference` 衡量短期变化幅度；
- `target std` 衡量窗口整体波动强度；
- `prototype entropy` 衡量原型匹配是否确定。

Router 先生成 drift score：

```text
drift_score ∈ R^B
```

然后通过 MLP + softmax 输出三类证据权重：

```text
[w_time, w_graph, w_proto] ∈ R^{B × 3}
```

最后进行加权融合：

```text
c_fused = w_time · c_time + w_graph · c_graph + w_proto · c_proto
c_fused ∈ R^{B × 128}
```

这一步表示模型根据当前窗口状态选择更可信的预测证据。

## 9. Enhanced Memory 与 Transformer Decoder

Router 输出的融合上下文会用于增强 encoder memory。

时间分支得到的逐时间点表示是：

```text
encoded ∈ R^{B × 48 × 128}
```

融合上下文经过线性映射并扩展到每个历史时间点：

```text
Proj(c_fused) ∈ R^{B × 128}
Broadcast(Proj(c_fused)) ∈ R^{B × 48 × 128}
```

得到增强记忆：

```text
memory = encoded + Broadcast(Proj(c_fused))
memory ∈ R^{B × 48 × 128}
```

Decoder 输入是：

```text
last 12 history + future_padding 24
X_dec ∈ R^{B × 36 × C}
```

其中未来 24 个 target 位置由 `future_padding` 填入。

然后进入 Transformer Decoder：

```text
decoded = TransformerDecoder(X_dec, memory)
decoded ∈ R^{B × 24 × 128}
```

Decoder 会根据：

- 最后 12 个历史点；
- 原型分支给出的未来初始轮廓；
- encoder memory；
- router 融合后的上下文；

共同生成未来 24 个时间点的隐藏状态。

## 10. Prediction Head：点预测与区间预测

Prediction Head 对 decoder 输出的未来隐藏状态做映射。

它输出四个通道：

```text
median
lower distance
upper distance
raw point
```

然后构造分位数区间：

```text
lower = median - softplus(lower distance)
upper = median + softplus(upper distance)
```

点预测由 raw point 和 median 通过门控融合：

```text
point = gate · raw_point + (1 - gate) · median
```

最终输出包括：

```text
point forecast ∈ R^{B × 24}
quantiles ∈ R^{B × 24 × 3}
lower ∈ R^{B × 24}
upper ∈ R^{B × 24}
```

如果启用了 Target RevIN，模型会使用前面保存的 `μ_w` 和 `σ_w` 将预测结果反归一化：

```text
ŷ_raw = ŷ_norm · σ_w + μ_w
```

因此，最终评估时的 MSE、MAE 和 SMAPE 都是在原始 workload 尺度上计算的。

## 11. 训练目标

训练时，MPGFormer 使用复合损失函数：

```text
Loss =
  point loss
  + quantile loss
  + peak underestimate loss
  + residual tail MSE loss
  + target tail MSE loss
  + graph contrastive loss
```

其中：

- point loss 同时包含 MAE 和 MSE；
- quantile loss 用于训练预测区间；
- peak underestimate loss 强调高负载低估的代价；
- tail MSE loss 强调大残差样本；
- target tail MSE loss 强调高负载目标；
- contrastive loss 约束动态图两种视角的一致性。

这使模型不仅追求平均误差较低，也更关注云工作负载中重要的高峰值和突发区间。

## 12. 画图时推荐的总流程

架构图可以画成三大区域。

### 左侧：Input Preparation & History Representation

```text
Raw Window [B×72×C]
  -> Split: 48 history + 24 future
  -> Future target masked
  -> Target RevIN
  -> Linear Projection
  -> Positional Encoding
  -> Gated Static Context
  -> H_s [B×48×128]
```

### 中间：Multi-Evidence Extraction

```text
Temporal Evidence Branch:
H_s + target_history
  -> Non-stationary Attention
  -> Moving-average Decomposition
  -> Short-term Temporal Encoder
  -> encoded [B×48×128], c_time [B×128]

Multi-scale Prototype Branch:
target_history
  -> Multi-scale Patching p={6,12,24,48}
  -> Time/Frequency Tokens
  -> Prototype Matching
  -> c_proto [B×128]
  -> future_padding [B×24]

Dynamic Graph Branch:
target_history + static/context stats + c_time
  -> workload_adj + context_adj
  -> GCN / Message Passing
  -> c_graph [B×128]
```

### 右侧：Drift-aware Fusion, Decoder & Outputs

```text
c_time, c_graph, c_proto + drift features
  -> Drift-aware Router
  -> w_time, w_graph, w_proto
  -> c_fused [B×128]

encoded + Proj(c_fused)
  -> Enhanced Memory [B×48×128]

last 12 history + future_padding 24
  -> Transformer Decoder
  -> Prediction Head
  -> point forecast + quantile interval + diagnostics
```

## 13. 一句话总结

MPGFormer 先对目标通道进行窗口级归一化并生成历史隐藏表示，然后并行提取时间证据、原型证据和图证据，再由漂移感知路由器根据当前窗口状态自适应融合三类证据，最后通过 Transformer Decoder 输出未来 24 步工作负载预测及可解释辅助信息。

