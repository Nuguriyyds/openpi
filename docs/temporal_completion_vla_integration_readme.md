# Temporal Completion Head：VLA 集成说明

1. **History prefix（主方案）**：使用当前子任务内连续三个 2 Hz prefix 特征。
2. **Current-only prefix（严格消融对照）**：与 History prefix 使用完全相同的 MLP、数据、sampler、seed 和训练预算，只把前两个历史槽固定置零。

## 1. 已训练权重

两个权重都是 JAX checkpoint，选中的 checkpoint step 均为 2000。

### 1.1 History prefix

- Config：`pi05_agilex_breakfast_temporal_completion_head`
- Checkpoint：

  ```text
  /mnt/data/wyt/checkpoints/completion_head/history_prefix/2000
  ```

- 参数目录：

  ```text
  /mnt/data/wyt/checkpoints/completion_head/history_prefix/2000/params
  ```

### 1.2 Current-only prefix（对照）

- Config：`pi05_agilex_breakfast_temporal_completion_current_only_head`
- Checkpoint：

  ```text
  /mnt/data/wyt/checkpoints/completion_head/current_only_prefix/2000
  ```

- 参数目录：

  ```text
  /mnt/data/wyt/checkpoints/completion_head/current_only_prefix/2000/params
  ```

这就是与 History prefix 直接做 matched ablation 的“最干净”版本。本文所称 Current-only 均指这个 checkpoint，**不指** `fit_temporal_current_only_probe.py` 产生的线性 probe，也不使用旧的单帧 completion head。

二者使用相同的 clean Pi0.5 backbone、模型结构、数据划分、采样器和训练预算，只是 completion head 的输入不同。完整 checkpoint 中 completion head 的参数子树是 `completion_head/*`。如果只把头合入另一个 VLA，需要从对应 checkpoint 恢复这棵参数子树，并保证结构完全一致。

共同的 clean backbone 来源为：

```text
/mnt/data/models/wyt/checkpoints/pi05_730_breakfast_subtasks/breakfast_subtasks_bs64_50k/49999
```

## 2. 在不同 Pi0.5 VLA 间是否可插拔

需要区分“**代码/结构可插拔**”和“**现有权重可直接复用**”：

- Completion head 的代码和结构可以接入任何 Pi0.5 VLA。
- 现有 `completion_head/*` 权重能否直接复用，不由训练数据集名称决定，而由产生 `prefix_out` 的 VLM/PaliGemma 参数是否相同决定。

### 2.1 可以直接复用现有 head 权重

满足以下条件时，可以把本文的 `completion_head/*` 直接合入另一个 Pi0.5 checkpoint：

1. 两个 VLA 使用相同的 Pi0.5/PaliGemma 架构和 tokenizer。
2. 图像预处理、prompt 模板、prefix token 构造和 mask 语义相同。
3. `prefix_feature_dim`相同；当前模型是2048。
4. 目标 VLA 的 PaliGemma/VLM prefix encoder 参数与训练 completion head 时使用的参数完全相同。
5. 不同数据训练只更新了 action expert/action projection 等动作分支，没有更新会产生 `prefix_out` 的视觉语言主干。

这种情况下，动作策略可以因数据不同而不同，但 completion head 看到的特征坐标系不变，权重可以直接插拔。

### 2.2 结构可复用，但 head 必须重新训练

如果两个 VLA 都叫 Pi0.5、隐藏维度也都是2048，但目标 VLA 在自己的数据上更新过以下任一部分：

- PaliGemma LLM；
- SigLIP/image encoder；
- 作用于上述模块的 LoRA；
- tokenizer、prompt模板或图像预处理；

那么 `prefix_out` 的特征空间可能已经变化。此时旧head在形状上可以正常加载，但其权重不再有可靠语义，不能因为“架构相同、场景相同、维度相同”就认为可以直接复用。

推荐做法是：

1. 复用本文的 masked-mean、三帧输入和 `TemporalCompletionHead` 结构。
2. 冻结目标 VLA。
3. 用目标 VLA 重新提取 prefix feature。
4. 使用相同 completion 标签和sampler，只重新训练很小的 `completion_head/*`。

因此最准确的结论是：

> 不同数据训练的 Pi0.5 可以直接复用 completion head 的代码和训练方案；只有在 prefix encoder 没有变化时，才能直接复用本文已经训练好的 head 权重。

### 2.3 合并不同 VLA 时不要覆盖动作模型

本文列出的目录是完整训练checkpoint。如果目标系统已有另一个Pi0.5 VLA权重，不要直接用本文完整checkpoint覆盖目标VLA，否则会同时替换其动作模型和backbone。

应当：

1. 先加载目标 VLA checkpoint。
2. 在模型中创建相同结构的 `TemporalCompletionHead`。
3. 只从本文checkpoint复制 `completion_head/*` 参数子树。
4. 保留目标 VLA 的其余所有参数。
5. 将 completion head 参数保持为FP32。

如果目标 VLA 的 prefix encoder 已经改变，则执行相同结构的重新训练，而不是复制旧 `completion_head/*`。

## 3. Prefix 特征定义

每次 VLA prefix forward 得到：

```text
prefix_out: [B, token_count, D]
prefix_mask: [B, token_count]
```

单帧特征 `z_t` 是有效 prefix token 的 FP32 masked mean：

```python
z_t = masked_mean(prefix_out, prefix_mask)  # [B, D]
```

约束如下：

- pooling 必须是 `masked_mean`，不能换成 last token、attention pooling 或 action hidden。
- pooling 和 completion head 都使用 FP32。
- prefix 特征对 completion loss 是 stop-gradient；训练时只更新 completion head。
- 当前 clean Pi0.5 的 `D=2048`，但集成代码应从模型的 `prefix_feature_dim` 推导，不要硬编码。
- prefix 必须在正确的当前子任务 prompt 下生成。

模型已有两种特征获取入口：

```python
model.compute_prefix_feature(rng, observation, train=False)
```

该入口会独立执行 prefix forward。正式 VLA 推理更推荐复用动作推理已有的 prefix forward：

```python
actions, z_t = model.sample_actions_with_prefix_feature(
    rng,
    observation,
    num_steps=10,
)
```

通过 policy 接口时：

```python
result = policy.infer(observation, return_prefix_feature=True)
actions = result["actions"]
z_t = result["prefix_feature"]  # [D], numpy.float32
```

这样不会为了 completion head 再运行一次 VLM prefix forward。

## 4. Completion head 结构

两个方案使用完全相同的 `TemporalCompletionHead`：

```text
输入                         [B, 3, D]
共享 FP32 LayerNorm          分别作用于三个时间槽，参数共享
按时间顺序展平               [z_(t-2), z_(t-1), z_t] -> [B, 3D]
Linear                       3D -> 128
GELU
Dropout                      0.1，仅训练时启用
Linear                       128 -> 1
输出                         logit [B]
```

概率为：

```python
score = sigmoid(logit)
```

模型入口为：

```python
logits = model.compute_temporal_completion_logits(
    rng,
    prefix_history,  # [B, 3, D], oldest -> newest
    train=False,
)
```

policy 对单条轨迹提供：

```python
score = policy.score_temporal_completion(prefix_history)  # [3, D] -> float
```

`score_temporal_completion()`返回 sigmoid 后的概率；传入 `return_logit=True` 可获取原始 logit。

## 5. 方案一：History prefix

### 5.1 输入

每个判断样本使用同一子任务、同一 prompt 下的三个 prefix：

```text
[z_(t-2), z_(t-1), z_t]
```

数据与推理频率是 2 Hz。原始视频为 30 fps，因此相邻特征严格相隔 15 帧：

```text
[frame(t-30), frame(t-15), frame(t)]
```

### 5.2 在线状态

每条 rollout 维护一个长度为 3 的 FIFO/deque：

```python
history.append(z_t)
if len(history) > 3:
    history.pop(0)
```

只有积累到三个特征后才能评分。若新子任务从相对 frame 0 开始，理论上的前三个 completion tick 为：

```text
0, 15, 30
```

因此首次评分发生在相对 frame 30。

### 5.3 Prompt 切换

这个模型使用的是 **subtask-local history**，严禁把前一个 prompt 下的 prefix 留给下一个子任务：

1. 当前三个 prefix 都必须由当前 active prompt 生成。
2. completion score 达到阈值后，当前子任务完成。
3. 新 prompt 从下一次 VLA forward 开始生效。
4. prompt 切换时立即清空 prefix deque。
5. 新任务重新积累三个 2 Hz prefix 后再评分。
6. 如果动作系统缓存了旧 prompt 生成的 action chunk，切换时应同时使该 pending plan 失效。

仓库中的 `TemporalCompletionController` 和 `TemporalCompletionPolicy` 实现了上述 history-prefix 状态语义，可作为接入参考：

```text
src/openpi/policies/temporal_completion_controller.py
```

该 wrapper 是单 rollout 可变状态，不应在多个并发客户端之间共享。

## 6. 方案二：Current-only prefix

Current-only 是与 History prefix 严格匹配的同结构、同参数量对照方案。它不是线性 probe，也不是另一个单输入 MLP，而是继续使用 `[B, 3, D]` 的相同 temporal MLP，只把前两个时间槽置零：

```python
prefix_history = stack([z_t_minus_2, z_t_minus_1, z_t])
current_only_input = concatenate([
    zeros_like(prefix_history[:, :2, :]),
    prefix_history[:, 2:, :],
], axis=1)

# 等价表示
current_only_input = [0, 0, z_t]
```

对应实现位于：

```text
src/openpi/training/completion.py::apply_temporal_input_mode
```

### 6.1 重要集成差异

`Policy.score_temporal_completion()`只负责执行head，**不会根据config自动把前两个槽置零**。因此使用current-only权重时，VLA接入层必须显式构造：

```python
head_input = np.stack([
    np.zeros_like(z_t),
    np.zeros_like(z_t),
    z_t,
], axis=0).astype(np.float32)

score = policy.score_temporal_completion(head_input)
```

如果直接把真实的三个历史prefix送入current-only checkpoint，输入语义与训练不一致，结果无效。

虽然current-only在数值上只需要当前帧，但为了与训练/评估协议一致，建议仍按2 Hz运行，并在当前子任务开始后的第三个tick才启用判断。prompt切换后也应重置计时状态。

## 7. 加载示例

History prefix：

```python
from openpi.policies import policy_config
from openpi.training import config as training_config

config = training_config.get_config(
    "pi05_agilex_breakfast_temporal_completion_head"
)
policy = policy_config.create_trained_policy(
    config,
    "/mnt/data/wyt/checkpoints/completion_head/history_prefix/2000",
)
```

Current-only prefix：

```python
config = training_config.get_config(
    "pi05_agilex_breakfast_temporal_completion_current_only_head"
)
policy = policy_config.create_trained_policy(
    config,
    "/mnt/data/wyt/checkpoints/completion_head/current_only_prefix/2000",
)
```

加载逻辑会保持 frozen VLA/action参数为BF16，并把 `completion_head/*` 保持为FP32。若在其他加载器中手工合并参数，也必须保留这一混合精度约定。

## 8. 训练语义摘要

两个方案共享相同训练样本和batch：

```text
batch_size = 64
positive = 32
hard_negative = 16
ordinary_negative = 16
BCEWithLogits
pos_weight = 1
```

对每个子任务，人工标注最后一帧为 inclusive endpoint `E`：

- Positive：`[E-30, E-15, E]`，label 1。
- Hard negative：`[E-45, E-30, E-15]`，label 0。
- Ordinary negative：从 `E-30` 开始按15帧间隔反向采样，label 0。

所有样本都限制在同一个raw subtask episode和同一个prompt内。训练和部署都不应跨prompt保留历史。

## 9. 主体代码文件

### 9.1 必须合入 VLA 模型的代码

| 优先级 | 文件 | 需要合入的主体功能 |
|---|---|---|
| 必须 | `src/openpi/models/completion.py` | `CompletionHeadConfig`、`masked_mean_pool`、`TemporalCompletionHead`。这里定义了head本身。 |
| 必须 | `src/openpi/models/pi0.py` | 创建`completion_head`；从prefix forward取得`prefix_out`；计算`z_t`；实现`compute_prefix_feature()`、`compute_temporal_completion_logits()`和`sample_actions_with_prefix_feature()`。这是head接入Pi0.5主模型的核心。 |
| 必须 | `src/openpi/models/pi0_config.py` | 把`CompletionHeadConfig`加入`Pi0Config`，并声明completion head参数子树。 |

### 9.2 使用 OpenPI Policy API 时需要的代码

| 文件 | 作用 |
|---|---|
| `src/openpi/policies/policy.py` | 暴露`return_prefix_feature=True`和`score_temporal_completion()`；使用独立completion RNG，不改变动作采样随机流。 |
| `src/openpi/policies/policy_config.py` | 加载checkpoint时保持backbone BF16、`completion_head/*` FP32。 |

如果目标VLA系统不用OpenPI的`Policy`封装，可以不照搬这两个文件，但必须实现等价接口和FP32加载语义。

### 9.3 History在线状态与prompt切换

| 文件 | 作用 |
|---|---|
| `src/openpi/policies/temporal_completion_controller.py` | 维护2 Hz三特征deque、阈值判断、prompt切换、history清空和pending action plan失效标志。 |

这个文件不属于MLP本体，但History方案上线时需要同等状态机语义。可以复用，也可以在现有VLA调度器中重写。

### 9.4 仅训练/复现实验需要的代码

| 功能 | 文件/入口 |
|---|---|
| History/current-only输入变换 | `src/openpi/training/completion.py::apply_temporal_input_mode` |
| 两个训练配置 | `src/openpi/training/config.py` |
| Temporal数据与标签索引 | `src/openpi/training/temporal_completion_data.py` |
| 32/16/16 batch sampler | `src/openpi/training/temporal_completion_sampler.py` |
| Prefix feature cache | `src/openpi/training/temporal_completion_features.py`、`scripts/extract_temporal_completion_features.py` |
| Head训练入口 | `scripts/train.py` |

## 10. 集成检查清单

- [ ] 使用正确方案对应的checkpoint，不能混用两个head权重。
- [ ] Prefix pooling与训练一致：有效token的FP32 masked mean。
- [ ] Head输入为FP32，形状严格为 `[B, 3, D]` 或单样本 `[3, D]`。
- [ ] History顺序严格为oldest-to-newest。
- [ ] Completion每15帧运行一次，即30 fps下2 Hz。
- [ ] History方案前三个prefix使用同一active prompt。
- [ ] Prompt切换后清空history，并丢弃旧prompt的pending action plan。
- [ ] Current-only方案显式输入 `[0, 0, z_t]`。
- [ ] Head输出先取logit，再做sigmoid；阈值作为部署配置，不写死在模型里。
- [ ] Completion head参数保持FP32。
- [ ] 若目标Pi0.5由不同数据训练，确认其prefix encoder是否与源模型相同；不能只比较架构名和特征维度。
- [ ] 合并到已有VLA时只覆盖`completion_head/*`，不覆盖目标动作模型和backbone。
