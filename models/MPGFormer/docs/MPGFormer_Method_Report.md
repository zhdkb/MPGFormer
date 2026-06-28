# MPG-Former 实现方法详解

MPGFormer 是为云工作负载预测设计的多上下文融合模型。它的出发点很直接：云负载序列往往不是平滑、单尺度、稳定分布的时间序列，而是同时包含局部分布漂移、多尺度周期、短时尖峰以及跨机器或跨任务的相似模式。因此，MPGFormer 没有把预测问题完全交给一个标准 Transformer，而是把“时间依赖”“多尺度形态”“跨序列关系”和“漂移状态”分别建模，最后再通过一个动态路由机制把这些信息合成用于预测的上下文。

从实现上看，MPGFormer 的主流程集中在 `models/MPGFormer/mpgformer.py`，数据构造在 `data.py`，训练与评估在 `train_mpgformer.py`，损失函数在 `losses.py`，其余核心组件被拆到 `modules/` 目录中。模型的输入是形状为 `[num_series, num_periods, num_features]` 的工作负载数组，输出则包括未来 `pred_len` 个时间步的点预测、分位数区间，以及路由权重、漂移分数、原型权重和动态图等辅助分析信息。

## 1. 数据进入模型之前

训练脚本首先读取 `.npy` 或 `.pkl` 格式的数据，并检查其维度是否为三维。默认情况下，第 0 列是待预测的 target，前 `input_size` 列作为动态输入特征，从 `static_start` 开始的列作为静态上下文。这个约定使 MPGFormer 可以兼容 Google v1、Azure/VM 以及 ECW 这类数据，只需要通过命令行参数指定动态特征范围和静态特征起点。

在进入数据集类之前，脚本会对动态特征做 MinMax 归一化，并单独保存 target 的 scaler。这里特意保留 target scaler，是为了在验证阶段把预测值和真实值都反归一化回原始尺度后再计算 MSE 和 MAE。这样 MPGFormer 的指标口径与 Informer、DynEformer 等 baseline 保持一致，不会因为归一化空间不同导致结果不可比。

`MPGFormerWindowDataset` 负责把每条序列切成滑动窗口。每个样本由一个长度为 `enc_len + pred_len` 的输入窗口和一个长度为 `pred_len` 的预测目标组成。窗口前半段是历史输入，后半段是预测区间。为了防止未来 target 泄露，预测区间中的 target 会被置为 0，但其他已知动态特征会保留下来。换句话说，decoder 可以看到未来协变量，却看不到未来真实负载。

```text
x: [enc_len + pred_len, input_size]
y: [pred_len]
static: [static_dim]
```

在模型内部，`_split_inputs` 会进一步把输入切成 encoder 使用的 `enc_input` 和 decoder 使用的 `dec_input`。`dec_input` 由历史最后 `label_len` 个 token 加上未来 `pred_len` 个 masked token 构成。相比直接给 decoder 全零未来输入，MPGFormer 后面还会用多尺度原型库生成的 `future_padding` 替换未来 target 的初始值，让 decoder 得到一个更合理的预测起点。

## 2. Target RevIN：先处理局部分布漂移

云负载的一个典型问题是，同一条序列在不同窗口里的尺度可能完全不同。全局 MinMaxScaler 可以缓解特征量纲问题，但不能解决窗口内部均值和方差变化。MPGFormer 因此在模型内部加入 `TargetRevIN`，只对 target 通道做窗口级可逆归一化。

具体来说，模型对每个样本窗口计算 target 的均值和标准差：

```text
target_norm = (target - mean) / std
```

后续编码器看到的是局部标准化后的 target。预测完成后，模型再用同一个 `mean` 和 `std` 把 point forecast 和 quantile forecast 还原到原始窗口尺度。这里没有对所有特征都做 RevIN，而是只处理 target，因为其他动态特征可能有自己的语义范围，强行按 target 的局部尺度变换反而会引入噪声。

这一步的作用不是提升模型复杂度，而是让后面的注意力层不必同时承受大幅尺度变化和模式学习的压力。对云负载这种非平稳数据来说，它相当于先把“当前窗口处在什么量级”这件事从编码难题中剥离出来。

## 3. 时间编码：长程依赖和短期突变同时建模

经过 Target RevIN 后，动态输入先通过线性层投影到 `d_model` 维空间，并加入 sinusoidal positional encoding。位置编码保留了时间顺序，线性投影则把不同动态特征映射到统一的隐空间。

如果输入中存在静态上下文，MPGFormer 不会简单地把静态特征拼接到每个时间点上，而是使用 `StaticContextSelector` 做门控注入。每个时间 token 会生成自己的 query，再决定静态上下文应该以多大强度进入当前 token。这样做的好处是，静态信息只在有用的时间位置发挥作用，而不是在整个序列中机械重复。

时间编码的主体是 `NonStationaryAttentionLayer`。它仍然使用 self-attention 建模长程依赖，但 attention logits 不再是固定形式，而是由当前窗口 target 的均值和标准差动态调制：

```text
attention_logits = tau * (QK^T / sqrt(d_head)) + delta
```

其中 `tau` 和 `delta` 由窗口统计量生成。直观上，负载平稳时模型可以使用较稳定的注意力分布；负载波动变强或尺度变化时，attention 可以改变分布锐度和偏置。这比标准 attention 更适合非平稳工作负载。

在同一层中，MPGFormer 还加入 moving average decomposition，把表示拆成平滑趋势和季节残差：

```text
trend = moving_average(x)
seasonal = x - trend
```

trend 作为残差偏置保留，seasonal residual 进入 feed-forward 分支。这样模型不必用同一个子结构同时拟合趋势和高频扰动，时间表示会更清晰。

长程 attention 之后，`ShortTermTemporalEncoder` 再用两条卷积分支和一个双向 GRU 捕捉短期变化。`kernel_size=3` 和 `kernel_size=5` 的卷积负责局部尖峰和短片段模式，BiGRU 负责邻近时间步之间的顺序关系。这个模块是对 Transformer 的补充，因为尖峰往往发生在很短的局部范围内，单靠全局 attention 不一定能稳定抓住。

## 4. 多尺度时频原型库：把典型负载形态显式学出来

MPGFormer 中最重要的结构之一是 `MultiScaleTimeFrequencyBank`。它的设计动机是，云负载中经常同时存在不同尺度的节奏：短期抖动、周期性业务波动、调度周期和更长的趋势变化。如果只让模型隐式学习这些模式，训练会更难，泛化也不稳定。

该模块会按照多个 patch size 切分 target history，例如 6、12、24、48。每个尺度上都会提取两类 token：一类来自原始 patch 的时域形态，另一类来自 `torch.fft.rfft` 得到的频域幅值。时域 token 告诉模型这一段负载“长什么样”，频域 token 则告诉模型其中是否存在周期性成分。

随后，每个尺度 token 会去匹配一个可学习的 prototype memory。可以把这些 prototype 理解为模型在训练过程中学到的典型负载形态。当前窗口与这些原型的匹配权重不仅用于生成 `prototype_context`，也用于估计模型对当前形态的确定程度。若原型分配很分散，说明当前负载形态不典型，这个信号会进入后面的漂移路由器。

多尺度原型库还会输出 `future_padding`。它不是最终预测，而是给 decoder 未来 target 位置的初始填充值。相比把未来 target 全部置 0，这个由原型上下文生成的初始轨迹可以让 decoder 从一个更接近真实负载形态的位置开始修正。

## 5. 动态图对比编码：利用 batch 内相似序列

很多 baseline 默认 batch 中的不同序列相互独立，只是共享模型参数。MPGFormer 进一步利用 batch 内序列之间的相似性，构建 `DynamicGraphContrastiveEncoder`。

这个模块会生成两个图。第一个图由 target history 的 cosine similarity 构建，表示哪些序列在近期负载形态上相似。第二个图优先由静态上下文构建，如果没有静态上下文，则使用 target 的统计特征构建。每个图都会保留 top-k 邻居、加入 self-loop，并做行归一化。

时间编码得到的 `time_context` 会分别经过两个图卷积，形成两个视图。模型再用对比损失约束这两个视图对同一条序列的表示保持一致。这样，MPGFormer 不只是看一条序列自己的历史，还会从 batch 中相似的机器、任务或应用那里获得补充信息。对于稀疏尖峰来说，这一点尤其重要，因为单条序列中尖峰样本可能很少，而相似序列可能提供额外参考。

## 6. 漂移感知路由：不是简单相加，而是按状态选择信息来源

MPGFormer 会同时得到三个上下文：时间编码器输出的 `time_context`，动态图模块输出的 `graph_context`，以及多尺度原型库输出的 `prototype_context`。这些信息来源并不总是同等重要。稳定窗口可能更依赖时间历史，漂移窗口可能更依赖图上下文或原型模式，原型不确定时又需要降低原型分支的影响。

因此，MPGFormer 使用 `DriftAwareRouter` 生成三个专家权重：

```text
router_weights = [w_time, w_graph, w_proto]
```

路由器输入除了三个上下文本身，还包含当前窗口的漂移特征：最后一个值相对窗口均值的 z-score、一阶差分绝对值均值、target 标准差，以及 prototype assignment entropy。这些特征共同描述了当前窗口是否稳定、是否波动剧烈、是否处于不典型模式。

最终融合方式为：

```text
fused_context =
    w_time  * time_context
  + w_graph * graph_context
  + w_proto * prototype_context
```

这使 MPGFormer 的融合过程是动态的，而不是固定地把几个模块结果拼接或相加。`drift_score` 和 `router_weights` 也会被输出，便于后续分析模型在某个窗口中为什么更依赖某一类信息。

## 7. Decoder、区间预测和点预测稳定化

MPGFormer 使用 Transformer decoder 生成未来 `pred_len` 个时间步的隐状态。decoder 的 memory 是历史编码表示加上 fused context 的投影，decoder input 则由历史 label token 和带 `future_padding` 的未来 token 构成。这样 decoder 既能回看历史，也能使用融合后的全局上下文。

预测头输出四个通道：median、lower distance、upper distance 和 raw point。lower 和 upper 通过 softplus 保证区间有序：

```text
lower = median - softplus(lower_distance)
upper = median + softplus(upper_distance)
```

早期版本中，point forecast 完全来自独立 point head。后来在 Google v1 实验中发现，MPGFormer 的 MAE 很低但 MSE 偏高，说明多数点预测稳定，但少数点可能偏离较大。为此，模型加入了 `point_gate`，让 point forecast 在 raw point 和 quantile median 之间自适应融合：

```text
gate = sigmoid(point_gate(decoded))
point = gate * raw_point + (1 - gate) * median
```

median 通常比独立 point head 更稳定，而 raw point 保留了额外表达能力。这个 gate 的目标是减少偶发偏离，同时不把点预测完全限制成 median。

## 8. 损失函数与 checkpoint 选择

MPGFormer 的损失函数不是单一 MSE，而是由多个目标组成：

```text
loss =
  point_weight * MAE
+ mse_weight * MSE
+ quantile_weight * QuantileLoss
+ peak_weight * PeakUnderestimateLoss
+ tail_mse_weight * ResidualTailMSE
+ target_tail_mse_weight * TargetTailMSE
+ contrastive_weight * GraphContrastiveLoss
```

其中 MAE 保证大多数普通时间点的稳定性，MSE 对齐主要评估指标，QuantileLoss 训练预测区间，PeakUnderestimateLoss 减少高负载低估，GraphContrastiveLoss 稳定两个动态图视图。后续新增的 ResidualTailMSE 和 TargetTailMSE 是为了处理 Google v1 这类数据中的 MSE 问题：前者专门惩罚 batch 中残差最大的区域，后者专门关注真实 target 较高的区域。

当前默认权重采用温和的 MSE 优化策略。tail 相关损失没有设得很大，因为过强的尖峰惩罚会让模型过度补偿高负载，反而破坏普通时间点的预测。现在的目标是在保持 MPGFormer 原本 MAE 优势的同时，把少数大误差压下来。

checkpoint 选择也做了调整。旧版本默认按复合 `val_loss` 保存最优模型，但复合 loss 不一定对应最低 MSE。新版本默认使用 `--select_metric mse`，因此最终保存的 `MPGFormer_best.pt` 对应验证 MSE 最低的 epoch。这一点使 MPGFormer 与 Informer 这类主要按 MSE 优化和选择模型的 baseline 更可比。

## 9. 和其他方法相比的改进与目标

MPGFormer 和 Informer、Autoformer、DynEformer、Deep Transformer、DeepAR、MQRNN 的区别，主要不在于“用了更多层”，而在于它把云工作负载中的几个结构性问题拆开处理。Informer 和 Deep Transformer 更依赖 attention 本身去学习时间关系，MPGFormer 在 attention 之外加入了 Target RevIN、窗口统计调制和短期 Conv/GRU，使模型更能适应非平稳和局部尖峰。Autoformer 强调趋势和周期分解，MPGFormer 保留了 moving average decomposition 的思想，但进一步加入 FFT token 和 learnable prototype memory，让周期与典型负载形态可以端到端学习。DynEformer 使用 global pooling 引入全局模式，MPGFormer 则用多尺度原型库和动态图对比编码替代对预构建全局池的依赖，使模式学习和跨序列关系都能随训练自适应调整。

与 DeepAR 和 MQRNN 这类 RNN 系方法相比，MPGFormer 不再主要依赖自回归隐藏状态去记住历史，而是同时使用 attention 的长程建模、卷积和 GRU 的局部建模、原型库的形态记忆以及动态图的跨序列信息。MQRNN 的分位数预测思想在 MPGFormer 中被保留下来，但 MPGFormer 额外加入 point-median gate 和 MSE/tail-MSE 目标，使区间预测和点预测指标可以同时优化。

这些改进的共同目标是让模型更贴近云负载数据本身：Target RevIN 和非平稳 attention 处理分布漂移，多尺度时频原型处理周期和典型形态，动态图对比编码处理跨序列相似性，漂移感知路由处理信息来源选择，point-median gate 与 tail-MSE 系列损失则专门降低少数尖峰误差对 MSE 的影响。换句话说，MPGFormer 的目标不是在所有 baseline 上简单叠加模块，而是把每个模块对应到一个明确的数据难点上。

## 10. 适合写入论文的方法概括

MPGFormer 可以概括为一个面向云工作负载预测的漂移感知多专家融合框架。模型首先通过目标通道可逆归一化缓解局部分布漂移，然后使用由窗口统计量调制的非平稳注意力提取长程时间依赖，并通过卷积与 GRU 捕捉短期尖峰。在此基础上，模型构建多尺度时频原型库以学习典型负载形态，同时构建动态图对比编码器以利用 batch 内跨序列相似性。随后，漂移感知路由器根据负载波动、趋势变化和原型不确定性，自适应融合时间上下文、图上下文和原型上下文。最后，Transformer decoder 输出点预测和分位数区间。为了降低稀疏尖峰造成的 MSE，模型进一步引入 point-median gate、ResidualTailMSE、TargetTailMSE，并默认按验证 MSE 选择最佳 checkpoint。

## 11. 小规模验证观察

在本地小规模配置中，使用 `d_model=32`、`d_ff=64`、`epochs=60`、`batch_size=64` 的温和 MSE/tail-MSE 配置，观察到：

```text
best_mse = 0.7262327075004578
best_mae = 0.1294773370027542
```

该结果只作为趋势验证，不代表正式大模型最终结果。它说明 point-median gate、MSE 选模和温和 tail-MSE 方向能够缓解旧版本中“MAE 很低但 MSE 偏高”的问题，同时保留 MPGFormer 在 MAE 上的优势。

## 12. 总结

MPGFormer 的实现可以理解为一条从“局部尺度校正”到“多上下文融合”再到“尖峰误差控制”的完整预测流程。Target RevIN 先缓解窗口尺度漂移，非平稳 attention 和短期编码器提取时间表示，多尺度时频原型库提供典型形态和未来初始化，动态图对比编码利用 batch 内相似序列，漂移感知路由器决定不同信息来源的权重，decoder 最后输出点预测和区间预测。

相对其他方法，MPGFormer 的主要改进不是单点结构，而是围绕云工作负载的非平稳、多尺度、跨序列和稀疏尖峰特性形成了一套连贯的实现。它的最终目标是在保持 MAE 优势的同时降低 MSE，并让模型输出具备更好的可分析性。
