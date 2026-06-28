# MPGFormer Introduction 中 Figure 2 的动机图画法说明

本文档只说明 MPGFormer 论文 Introduction 中 Figure 2 应该如何绘制。需要特别注意的是，Figure 2 不是 MPGFormer 的模型结构图，也不是模块流程图。它不应该展示 Transformer、Prototype Bank、Graph Encoder 或 Router 的内部细节。Figure 2 的作用是解释 Introduction 中提出的研究动机和问题：**同一个云工作负载在不同运行窗口中会呈现不同预测状态，因此最可靠的预测证据会随状态变化。**

换句话说，这张图要回答的问题不是“MPGFormer 怎么做”，而是“为什么云工作负载预测需要被看作状态依赖的证据选择问题”。

## 1. 图的核心标题

建议中文标题：

```text
Figure 2: 云工作负载预测中的状态依赖证据选择问题
```

如果论文正文使用英文，可以写成：

```text
Figure 2: Workload forecasting as a state-contingent evidence selection problem.
```

这个标题要突出两个关键词：

1. **state-contingent**：预测依赖当前运行状态。
2. **evidence selection**：预测不是只依赖固定历史，而是要选择可信证据。

## 2. 整体布局建议

Figure 2 建议画成一个从左到右的四段式动机图，也可以画成 2 × 2 布局。推荐从左到右，因为它更像一个故事展开过程：

```text
(a) 同一条 workload trace
        ↓
(b) 不同时间窗口呈现不同运行状态
        ↓
(c) 不同状态对应不同可靠证据
        ↓
(d) 问题定义：预测需要做状态依赖证据选择
```

注意：第 (d) 部分只画“问题定义”，不要画 MPGFormer 的具体模块。可以用一个抽象的 “Forecasting decision” 或 “Evidence selection” 框表示问题，但不要画成方法架构。

## 3. 子图 (a)：同一条长期运行的 workload trace

子图标题：

```text
(a) Same long-running workload trace
```

中文含义：

```text
同一条长期运行的工作负载序列
```

画法：

画一条连续的 workload 曲线，横轴是时间，纵轴是负载值，例如 CPU usage、QPS 或请求量。曲线应表现为一条真实云负载，而不是规则正弦波。可以包含缓慢上升、周期起伏、尖峰和局部尺度变化。

这个子图要表达：

```text
观察对象是同一个服务、同一个实例或同一条 workload，而不是多个不同任务。
```

建议在图中标注：

```text
Same service / same instance
```

中文可标注：

```text
同一个服务 / 同一个实例
```

这句标注很重要，因为它说明本文关注的问题不是“新应用”或“新机器”，而是同一个已知实体内部的运行状态变化。

## 4. 子图 (b)：同一 workload 的不同运行状态

子图标题：

```text
(b) Different operational states within the same trace
```

中文含义：

```text
同一条序列中的不同运行状态
```

画法：

在子图 (a) 的 workload 曲线上框出四个时间窗口，用不同颜色或不同边框表示四种状态：

1. **Stable periodic window**  
   稳定周期窗口。曲线有较明显的重复节律，例如平稳上升下降或日周期模式。

2. **Bursty high-load window**  
   突发高负载窗口。曲线在短时间内快速升高，形成尖峰或突然的高负载区间。

3. **Co-fluctuation window**  
   协同波动窗口。当前 workload 与旁边几条浅色参考曲线同步上升或下降。这里可以额外画 2 到 3 条灰色曲线，表示其他实例或相关服务。

4. **Drifted / atypical window**  
   漂移或非典型窗口。曲线的均值、方差或形态与前面窗口明显不同，例如整体抬升、尺度变大、波动变强或进入不规则形态。

这个子图的核心标注建议写：

```text
The entity is not new, but its forecasting state changes over time.
```

中文可写：

```text
实体并不是新的，但预测状态会随时间变化。
```

这句话应放在子图 (b) 附近，作为整张 Figure 2 的关键解释。

## 5. 子图 (c)：不同状态需要不同预测证据

子图标题：

```text
(c) Reliable evidence changes with workload state
```

中文含义：

```text
可靠预测证据随运行状态变化
```

画法：

将子图 (b) 中的四种窗口分别连到不同的证据类型。注意，这里画的是“问题中的证据来源”，不是 MPGFormer 的模块。

建议画四类证据：

1. **Temporal rhythm / history**  
   时间节律 / 历史记忆。对应稳定周期窗口。

2. **Recent local change**  
   近期局部变化。对应突发高负载窗口。

3. **Similar workloads**  
   相似工作负载。对应协同波动窗口。

4. **Prototype uncertainty / distribution shift**  
   原型不确定性 / 分布漂移。对应漂移或非典型窗口。

推荐在图中用箭头表示：

```text
Stable periodic window      -> temporal rhythm
Bursty high-load window     -> recent local change
Co-fluctuation window       -> similar workloads
Drifted / atypical window   -> uncertainty and drift awareness
```

中文可以写：

```text
稳定周期窗口      -> 时间节律更可靠
突发高负载窗口    -> 近期局部变化更可靠
协同波动窗口      -> 相似工作负载更可靠
漂移非典型窗口    -> 需要关注不确定性和漂移
```

这个子图最重要的是表达“证据会变化”。不要在这里写具体网络名称，例如不要写 Non-stationary Attention、Prototype Bank、Graph Encoder 或 Router。

## 6. 子图 (d)：问题定义，而不是方法结构

子图标题：

```text
(d) Forecasting requires state-contingent evidence selection
```

中文含义：

```text
预测需要状态依赖的证据选择
```

画法：

画一个抽象的问题框，不要画成 MPGFormer 的方法框。可以画成：

```text
Current workload window
        ↓
Which evidence should be trusted?
        ↓
Forecast
```

中文可写成：

```text
当前 workload 窗口
        ↓
应该相信哪一种预测证据？
        ↓
未来负载预测
```

旁边可以列出四种候选证据：

```text
history?
local change?
similar workloads?
uncertainty?
```

中文可写：

```text
历史节律？
局部变化？
相似序列？
漂移不确定性？
```

这个子图应当让读者一眼看出本文的研究问题：

```text
Cloud workload forecasting is not only temporal extrapolation,
but also state-contingent evidence selection.
```

中文可写：

```text
云工作负载预测不仅是时间外推，更是状态依赖的证据选择。
```

## 7. 视觉风格建议

整张图建议使用简洁的论文风格，而不是复杂架构图风格。

1. workload 曲线使用深蓝色或黑色主线。
2. 四个窗口使用不同浅色边框，例如蓝色、红色、绿色、橙色。
3. 其他相似 workload 曲线使用浅灰色，避免抢占主线视觉。
4. 子图 (c) 的证据类型可以使用简单图标：
   - 时间节律：小钟表或周期波形。
   - 局部变化：尖峰图标。
   - 相似 workload：多条相邻曲线。
   - 漂移不确定性：问号或分布偏移箭头。
5. 子图 (d) 不要出现 MPGFormer 的模块名称，只保留抽象问题框。

## 8. 推荐图注

中文图注建议：

```text
Figure 2 展示了云工作负载预测中的状态依赖证据选择问题。同一条长期运行的 workload trace 在不同时间窗口中可能呈现稳定周期、突发高负载、协同波动以及漂移或非典型状态。虽然预测对象本身并不是新的，但不同状态下最可靠的预测证据并不相同：稳定窗口更依赖时间节律，突发窗口更依赖近期局部变化，协同波动窗口更依赖相似 workload，而漂移窗口需要关注不确定性和分布变化。因此，云工作负载预测不仅是时间外推问题，也是根据当前运行状态选择可信预测证据的问题。
```

英文图注可以写：

```text
Figure 2 illustrates the state-contingent evidence selection problem in cloud workload forecasting. A long-running workload trace can exhibit stable periodic, bursty, co-fluctuating, and drifted states over time. Although the target entity is not new, the most reliable forecasting evidence changes across states. Stable windows rely more on temporal rhythm, bursty windows require recent local changes, co-fluctuating windows benefit from similar workloads, and drifted windows require uncertainty and distribution-shift awareness. Therefore, cloud workload forecasting is not only temporal extrapolation, but also the selection of trustworthy evidence according to the current operational state.
```

## 9. 一句话总结

这张 Figure 2 要传达的核心不是 MPGFormer 的结构，而是 MPGFormer 这篇论文为什么要提出新的研究视角：

```text
同一个云工作负载在不同时间窗口中会切换运行状态，因此预测模型需要先判断当前窗口应该相信哪一种证据。
```

英文一句话可以写成：

```text
The same cloud workload can switch forecasting states over time, requiring the model to decide which evidence should be trusted in the current window.
```
