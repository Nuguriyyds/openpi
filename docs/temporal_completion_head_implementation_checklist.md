# Temporal completion head 代码修改清单

> 状态：本地第一阶段代码已实现；运行顺序见 `temporal_completion_head_runbook.md`。真实 full-video 边界对齐、生产 scheduler 接线和 closed-loop 评测仍是后续阶段，当前 evaluator 只声明 oracle-prompt 结果。  
> 日期：2026-08-16  
> 当前阶段只实现 `prefix(t-2, t-1, t) -> MLP -> completion logit`。不加入 action、TTRTC、progress 或 ranking loss。

## 1. 已锁定的目标

在完整早餐轨迹上，以 2 Hz 运行一个因果的子任务完成检测器。检测器持有当前子任务 prompt；检测到完成事件后，prompt 控制器切到下一子任务并清空历史缓存。

单个可判定 tick 的输入为：

```text
z_t     = masked_mean(prefix_out(observation_t, current_prompt))
x_t     = concat(z_(t-2), z_(t-1), z_t)
logit_t = TemporalCompletionMLP(x_t)
```

其中：

- 原始数据是 30 fps；completion tick 严格为 2 Hz，因此相邻 tick 相差 15 帧。
- `t-1`、`t-2` 表示前一个、前两个 2 Hz tick，即逻辑帧 `t-15`、`t-30`，不是原视频的相邻帧。
- 时间顺序固定为 oldest-to-newest：`[z_(t-30), z_(t-15), z_t]`。
- backbone 使用 clean pi0.5 checkpoint，完全冻结；第一版不使用 S1/TTRTC checkpoint。
- completion head 和损失全程 FP32。

## 2. 固定的数据与模型来源

训练/虚拟完整轨迹来源：

```text
/mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_subtask_730
```

之后的真实完整轨迹闭环评测来源：

```text
/mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_730
```

clean checkpoint：

```text
/mnt/data/models/wyt/checkpoints/pi05_730_breakfast_subtasks/breakfast_subtasks_bs64_50k/49999
```

clean config：

```text
pi05_730_breakfast_subtasks
```

已知数据事实，代码必须再次自动审计而不能只依赖本文：

- subtask 数据集有 2944 个 episode、4 个 task，理论上组成 736 条四阶段轨迹。
- episode 顺序预期为每四个一组：`4g+0, 4g+1, 4g+2, 4g+3`，对应 task 0、1、2、3。
- 完整轨迹数据集有 737 个 episode，因此至少存在一条无法直接由 736 个 subtask group 对应的轨迹。
- 两个数据集总帧数不同，不能假设四段 subtask 的简单拼接与完整视频逐帧完全相同。
- 任何缺段、task 顺序错误、空段、异常短段或无法映射的完整轨迹都必须显式记录并从监督指标中排除，不能静默修补。
- **必须先建立 full-trajectory episode 到四个 subtask episode 的映射，再做 train/val/test 划分。**不能先按 subtask group 随机拆分、之后再挑 full test；否则 test 完整视频对应的 subtask 图像可能已经进入 train，形成最严重的数据泄漏。

## 3. 完整轨迹、边界与 2 Hz 标签的精确定义

### 3.1 逻辑轨迹

对一个合法 group `g`：

```text
source episodes = [4g, 4g+1, 4g+2, 4g+3]
lengths         = [L0, L1, L2, L3]
```

虚拟逻辑轨迹是四段按顺序连接后的索引空间。定义：

```text
B0 = L0
B1 = L0 + L1
B2 = L0 + L1 + L2
B3 = L0 + L1 + L2 + L3
```

`Bi` 表示 task `i` 结束后第一个逻辑帧的位置；它不是 task `i` 的最后一帧索引。task `i` 的最后一帧是 `Bi - 1`。

### 3.2 全轨迹统一的 2 Hz 时钟

2 Hz tick 使用完整逻辑轨迹的统一相位：

```text
T = {0, 15, 30, 45, ...}
```

禁止在每个原始 subtask episode 开头重新设置采样相位。prompt 切换只发生在一个 2 Hz tick 上，因此切换后继续使用同一全局相位。

### 3.3 唯一正事件

task `i` 的唯一正 tick。实现时使用纯整数式，避免浮点 `ceil`：

```text
C_i = ((B_i + 14) // 15) * 15
```

等价的区间标签是：

```text
y_t = 1  iff  B_i in (t - 15, t]
```

每个合法 `(trajectory, task)` 必须恰好有一个正样本，不能保留旧方案中连续 10 帧的正标签。

边界情况必须单测：

- `Bi % 15 == 0` 时，`C_i == Bi`。
- `Bi % 15 == 1` 时，`C_i == Bi + 14`。
- `Bi % 15 == 14` 时，`C_i == Bi + 1`。

### 3.4 正 tick 的观测和 prompt

对 task 0/1/2：

- 在 `C_i` 计算完成分数时仍使用 task `i` 的旧 prompt。
- 若 `C_i >= B_i`，观测来自下一个 subtask episode，source frame 为 `C_i - B_i`。
- 这不是把下一任务 prompt 泄漏给模型；它模拟“已经看到过渡后的画面，但控制器尚未切 prompt”。
- source observation 走 clean pi0.5 的正常 image/state/prompt transform；是否把 state 离散编码进 prefix 以 clean config 为准。raw `actions` 不进入第一版 head。

对 task 3：

- 没有下一 subtask。
- 从 `B3` 到 `C_3` 使用 task 3 最后一帧的 observation 做 terminal hold。
- terminal hold 只为覆盖第一个边界后 2 Hz tick，不生成额外正样本。
- task 3 必须单独分层报告，避免模型只凭重复 terminal frame 形成捷径而被总体指标掩盖。

### 3.5 prompt 激活与历史重置

在线语义固定如下：

1. task 0 prompt 从轨迹开始生效。
2. 在 `C_i` 用旧 prompt 计算 task `i` 的 completion logit。
3. 若触发，则下一次 policy forward 使用 task `i+1` prompt。
4. completion 历史 deque 立即清空；不把旧 prompt 的 prefix 放进新任务历史。
5. 不在同一个 raw frame 上为了新 prompt 额外重跑一次 VLM。
6. 新任务积累满三个 2 Hz prefix 后才允许判断；缺历史时直接 suppress，不做零填充、首帧复制或当前帧复制。
7. task 3 触发后进入整条任务 `done` 状态。

可执行的 feature 激活区间：

```text
task 0 feature ticks: 0, 15, 30, ... , C_0
task 0 first eligible decision tick: 30

task i>0 feature ticks after previous switch:
    C_(i-1)+15, C_(i-1)+30, C_(i-1)+45, ... , C_i
task i>0 first eligible decision tick:
    C_(i-1)+45
```

因此每个纳入训练的 trajectory 必须满足：

```text
C_0 >= 30
C_i - C_(i-1) >= 45, for i=1,2,3
```

否则该 task 的唯一正 tick 无法构成同 prompt 的三帧历史。默认整条 trajectory 进入 quarantine，并记录 `unreachable_positive_after_history_reset`，不能一边要求“每 task 恰好一个正样本”一边静默丢掉该正样本。

离线 oracle 数据构造也必须遵守同一 prompt/reset 语义，而不是为方便而跨 prompt 拼历史。

prompt 必须在 `PromptFromLeRobotTask`、`InjectDefaultPrompt` 等自动 prompt transform 之前显式覆盖为 logical current task prompt。尤其在：

- 正 tick 已经读取下一 raw episode observation 时，不能让 source episode 的 task 自动把 prompt 换成下一任务。
- 真实完整轨迹自带完整早餐 prompt 时，不能依赖 `InjectDefaultPrompt` 覆盖；它通常只在 prompt 缺失时注入默认值。

第一版每个边界只有一个受监督正 tick。如果该 tick 未超过阈值，closed-loop 中应记为 false negative，并暴露后续 prompt 未切换造成的错误传播；不能在 evaluator 中悄悄把更晚的 old-prompt tick 改成正样本。若之后需要“迟到一 tick 可恢复”，必须另立 recovery/tolerance 标签方案。

## 4. 数据划分

### 4.1 划分单位

唯一划分单位是已完成映射的真实完整 trajectory。每个 full episode 对应的四个 subtask episode、由它们生成的所有 tick、缓存特征和增强视图必须进入同一个 split。

划分前置条件：

1. 先对齐 full episode 与四个 subtask episode。
2. 只对 `mapping_status == matched` 且四段审计通过的 trajectory 划分。
3. full-only、subtask-only、歧义匹配或异常轨迹进入 quarantine，不参与监督 train/val/test。
4. 映射不能只依赖 `episode_id // 4`；还要用 metadata、task 顺序以及可用的状态/动作/图像边界证据校验。

禁止：

- 按帧随机划分。
- 按 subtask episode 独立划分。
- 先生成 triplet 后再随机拆分。
- 让同一 raw frame 的不同 prompt 视图跨 split。

### 4.2 比例

应用于通过数据审计后的合法 trajectory：

```text
train = 72%
val   =  8%
test  = 20%
```

实现要求：

- 固定 `split_seed = 42`。
- 先封存 test，再从剩余 80% 中确定 val/train。
- 固定取整规则（round-half-up）：

  ```text
  n_test = floor(0.20 * N + 0.5)
  n_val  = floor(0.10 * (N - n_test) + 0.5)
  n_train = N - n_test - n_val
  ```

  对合法 trajectory 做固定 seed shuffle 后，依次分配 test、val、train，并把实际数量写入 manifest。不能依赖不同库的默认 `round` 行为。
- 精确数量由合法 trajectory 数量决定；不要在代码中硬编码 736、530 等计数。
- 若最终恰有 736 个合法 matched trajectory，预期计数是 `train=530, val=59, test=147`；代码仍需按通用取整规则计算并审计总和。
- test manifest 创建后不可在训练脚本中重写。
- 任何为超参数、checkpoint、阈值或模型选择读取 test label 的路径都应直接报错。

### 4.3 manifest 最低字段

建议新增版本化 JSON manifest，至少包含：

```text
schema_version
source_subtask_repo_id / absolute_root
source_full_repo_id / absolute_root
fps
tick_stride_frames
split_seed
split_ratios
trajectory_id
source_episode_ids[4]
task_indices[4]
lengths[4]
boundaries[4]
positive_ticks[4]
split
full_episode_id
mapping_status
exclusion_reason (nullable)
```

manifest 还应记录输入 metadata 文件的 hash 或稳定指纹，避免数据改变后继续使用旧划分。

## 5. 训练样本索引

不要修改或复制原始数据集。优先构造只含索引和标签的虚拟样本表。

每条 candidate row 至少包含：

```text
trajectory_id
task_index
split
logical_tick
label
sample_kind                 # positive / hard_negative / ordinary_negative
boundary_tick
prompt_index
history_logical_ticks[3]    # [t-30, t-15, t]
source_episode_ids[3]
source_frame_indices[3]
terminal_hold_flags[3]
```

构造不变量：

- 三个历史 tick 的差值严格为 `[15, 15]`。
- 三个 prefix 都以同一个 current task prompt 提取。
- 三个 tick 都必须位于该 prompt 的同一次激活区间内。
- 任一历史缺失则整条 candidate 不进入 train/val/test；不能 padding。
- 不能让 LeRobot 的 raw-episode delta timestamp/clamp 机制自动生成跨边界历史；三个 source reference 必须由 logical indexer 显式给出。
- positive row 的 `logical_tick == boundary_tick`。
- hard negative 是同一边界前 1～4 个合法 tick，即提前 0.5、1.0、1.5、2.0 秒。
- ordinary negative 排除 positive pool 和 hard-negative pool。
- val/test 保存自然 2 Hz candidate 全集，不做类别重采样。

## 6. prefix 特征提取与缓存

### 6.1 特征定义

从 frozen pi0.5 prefix forward 得到：

```text
prefix_out:  [B, tokens, D]
prefix_mask: [B, tokens]
```

固定使用 masked mean，不使用现有 completion head 的 learned attention query：

```text
z = sum(prefix_out.float32 * mask) / max(sum(mask), 1)
```

当前 clean config 审计得到 `D=2048`，但实现必须从 PaliGemma config/实际 tensor 推导并在启动时 assert，不能在通用 helper 中硬编码 2048。输出 `z` 维度为 `D`。

### 6.2 缓存键与精度

缓存键必须至少包含：

```text
(source_episode, source_frame, prompt_index, checkpoint_fingerprint)
```

同一 raw frame 在不同 prompt 下不是同一个特征，禁止只按 `(episode, frame)` 去重。

要求：

- 模型处于 eval 模式，第一版不做 image augmentation。
- prefix 输出在 pooling 前转 FP32。
- 缓存可用 FP16 节约磁盘，但进入 LayerNorm/MLP 前必须转回 FP32。
- cache manifest 记录 checkpoint、config、数据 root、prompt 文本、pooling 方法和代码版本。
- 当前 `clean_completion_features.../features.npz` 使用旧的 10 帧正窗口和旧采样集合，不能直接作为新训练集；只能用于调试或数值一致性检查。
- 新正 tick 可能需要下一 subtask 的 source frame `0..14` 在旧 prompt 下提取，旧 `copy_frames=5` cache 覆盖不足。
- 可以为所有 split 预提取 frozen feature；但 feature cache 的存在不授权训练/阈值选择读取 test label 或 test 指标。

### 6.3 训练和部署复用

- 训练时优先离线提取并缓存每个唯一 tick/prompt 的 `z`，避免一条 triplet 重跑三次 VLM。
- 部署时从正常 action-policy prefix forward 复用当前 `z_t`，completion head 只维护三个向量的 deque；不得为了 head 再额外运行一遍 VLM。

## 7. completion head

### 7.1 新 head 的固定结构

新增 temporal 变体，并保留 legacy attention head 以免破坏旧 config/checkpoint。

新 head 输入：

```text
[B, 3, 2048]
```

结构：

```text
shared LayerNorm(2048) applied to each time step
reshape / concat -> [B, 6144]
Linear(6144, 128)
GELU
Dropout(0.1) during train only
Linear(128, 1)
squeeze -> [B] FP32 logits
```

必须满足：

- 三个时间步共享同一个 LayerNorm 参数。
- 不在 head 内 sigmoid；sigmoid 只用于指标和推理阈值。
- 输入通过 `stop_gradient`，只有 temporal head 参数可训练。
- head 参数和计算 dtype 均为 FP32。
- 参数量和路径在启动时审计；VLM/action parameter gradient 必须为零或不存在。

### 7.2 推荐 API 边界

不要让 JAX model 在内部保存有状态 deque。建议拆成：

```text
Pi0.compute_prefix_feature(observation) -> [B, 2048]
Pi0.compute_temporal_completion_logits(prefix_history, train, rng) -> [B]
```

在线 deque/prompt 状态由 policy/controller 层维护。训练数据可以直接提供缓存的 `[B, 3, 2048]`。

现有 `compute_completion_logits(observation)` 和 legacy head 保持兼容；新 config 明确选择 temporal variant，不能根据输入 shape 静默猜测。

### 7.3 checkpoint 兼容

- 新训练从 clean checkpoint 加载完整 backbone/action 参数。
- 只允许 temporal completion head 参数缺失并随机初始化。
- 不加载旧 S2 attention-head 参数到 temporal MLP。
- 严格检查除此之外没有 missing/unexpected parameter。

## 8. 损失函数与 batch sampler

### 8.1 损失

第一版固定使用无权重 BCE with logits：

```text
loss_i = y_i * softplus(-logit_i) + (1-y_i) * softplus(logit_i)
loss   = mean(loss_i)
```

配置要求：

- `objective = binary`
- `focal_gamma = 0`
- `pos_weight = 1.0`
- 禁止在 batch 平衡的同时再按自然类频率设置 `pos_weight`。
- 禁止在 loss 前显式调用 sigmoid。
- 第一版不加入 focal、ranking、temporal smoothness 或 progress loss。

由于训练 batch 的正比例是 1/3，而线上自然正比例更低，raw sigmoid 不被解释为校准概率；阈值只从自然分布 val 集选择。

### 8.2 batch 组成

head-level batch size 固定为 64 个 triplet：

```text
21 positive
21 hard negative
22 ordinary negative
```

采样流程：

1. 先均匀采样 21 个 `(trajectory, task)` 正事件；同一个 batch 内正事件不可重复。
2. 对每个正事件，优先从其前 1～4 个合法 2 Hz tick 中随机取一个 event-local paired hard negative。若合法边界正好是 prompt reset 后的首个可判定 tick（间隔 45 帧），该事件没有更早的合法 decision row；此时保留该唯一正样本，并从同一 task、另一条 train trajectory 的真实 hard-negative pool 回退采样。禁止跨 task、伪造 padding 或跨 prompt 取历史。
3. 从其余普通负池中采样 22 个 ordinary negative，尽量来自不同 trajectory。
4. task 0/1/2/3 在连续 batch 中轮换成近似均匀的 5/5/5/6 分配，避免长任务主导训练。
5. 允许正事件跨 batch 重复，这是处理自然类别不平衡的预期行为。

实现要求：

- 使用真正的 batch sampler，不能只产生一个全局 index 列表后依赖 DataLoader 偶然组成比例。
- `set_epoch`/resume 后采样顺序可复现。
- train sampler 可 replacement；val/test 不 replacement、不 shuffle 标签比例。
- 分布式训练时不同 rank 不得重复整个 batch；若第一版仅单设备训练，需要在 config 中显式限制并报清楚。
- 日志记录实际 batch 正/困难负/普通负数量和 task 分布。
- sampler audit 必须分别记录 `event_local_hard_count`、`same_task_fallback_hard_count` 和无本地 hard 的 event 列表；val/test 不做这种回退采样。

## 9. optimizer 与训练运行

固定初始配置：

```text
optimizer           = AdamW
peak learning rate  = 1e-4
weight decay        = 1e-4
gradient clip norm  = 1.0
dropout              = 0.1
```

运行要求：

- 至少三个训练 seed，报告均值、标准差和逐 seed 结果。
- checkpoint 选择只使用 val。
- train loss、balanced-batch accuracy 不能作为主要模型选择指标。
- 最大 step、warmup、validation interval 和 early-stopping patience 属于可在 val 上调整的运行超参数，不改变本文的数据/标签语义。
- 建议初始预算以每个 train 正事件期望被看到约 10～20 次为准，而不是把过采样数据伪装成传统 epoch。

## 10. 验证、测试和模型选择

### 10.1 两种评测模式

必须同时支持：

1. `oracle_prompt`：按真实 completion tick 切 prompt，用来隔离 head 的判别能力。
2. `closed_loop_prompt`：只根据模型触发切 prompt，用来测部署时的错误累积。

测试集只允许在所有实现和阈值规则冻结后运行一次正式结果。

### 10.2 threshold-free 主指标

本节 top-1、AUPRC、margin 和 current-only paired ablation 都在 `oracle_prompt` 的固定 natural candidate set 上计算；closed-loop 不用这些逐样本指标掩盖 prompt 错位，而主要报告实际触发序列、early trigger、recovery/delay、miss 和最终 done。

模型/checkpoint 选择优先使用：

1. episode-task top-1 boundary rate：一个 task 激活区间内最高分是否位于唯一正 tick。
2. hard-local AUPRC：正 tick 对比边界前 0.5～2 秒负 tick。
3. boundary margin：`score(C_i) - max(score(C_i-60:C_i-15))` 的均值、中位数和大于零比例。
4. natural-distribution AUPRC。

同时报告：

- 按 task 0/1/2/3 分层的上述指标。
- 负区间相邻 2 Hz 分数 `abs(delta)` 的 mean/p95，监控曲线波动。
- 三个 seed 的均值和标准差。
- 与 current-only `z_t` baseline 在完全相同 eligible rows/split 上的 paired 对照，不能直接引用旧标签/旧样本集结果。

### 10.3 阈值指标

用户已确认：提前触发的代价高于晚一个 2 Hz tick。

阈值选择规则必须在 val 上预先定义并记录，至少报告：

- early-trigger trajectory/task rate。
- event recall。
- detection delay（以帧和秒计，2 Hz 分辨率）。
- false triggers per minute。
- task 3 的 final-done recall。

具体阈值数值后续由 val 决定，不硬编码 0.5，也不能从 test 曲线人工挑选。

closed-loop 的迟到恢复只是一项操作性指标，不改变监督标签：

- 在 `C_i` 触发记为 on-time。
- 若 `C_i` 漏检，controller 可继续以旧 prompt 评分；后续首次触发可记为 delayed recovery，但这些后续 tick 的训练标签仍不是新的正样本。
- task 0/1/2 的 recovery 最晚只匹配到下一个 GT boundary 之前；超过该窗口仍未触发则记为 miss，不能把更后面的触发强行匹配回旧事件。
- task 3 只 hold 到 `C_3`；若 `C_3` 未触发，第一版直接记 final miss。延长 terminal hold 属于未来单独的 recovery 方案。
- oracle 模式无论模型是否越阈值都按 GT `C_i` 切 prompt，因此只用于隔离分类器，不代表真实闭环恢复能力。

## 11. prompt controller

建议新增无模型依赖的状态机，最小状态：

```text
current_task_index
current_prompt
prefix_deque(maxlen=3)
last_completion_score
triggered_boundaries
done
```

每个 2 Hz tick：

```text
1. 使用 current_prompt 得到当前 z_t。
2. append 到 deque。
3. deque 未满时不调用 head/不触发。
4. deque 已满时计算 logit/score。
5. 若满足冻结后的触发规则：
   - latch 当前边界，防止重复触发；
   - current_task_index += 1；
   - 更新 prompt，供下一次 policy forward 使用；
   - clear deque；
   - task 3 后设置 done。
```

controller 必须能运行在：

- oracle 模式（测试 head 本身）。
- predicted/closed-loop 模式（真实部署）。

触发切 prompt 时，部署层还必须丢弃或失效化由旧 prompt 生成、尚未执行的 action chunk；否则 completion controller 已切换但机器人仍继续执行旧任务动作。

“零额外 VLM forward”有前提：action policy 的有效 prefix forward cadence 至少达到 2 Hz，并能在 completion tick 提供当前 observation/prompt 对应的 `z_t`。如果只有过期 `z`：

- 不得把 stale feature 冒充当前 tick。
- 必须执行一次 prefix-only forward，或调整 completion tick 与 policy inference 调度对齐。
- 集成验收需记录 observation timestamp、feature timestamp、score timestamp 和实际延迟。

## 12. 预计修改/新增的代码位置

以下是实现导航，不要求所有逻辑硬塞进现有文件；优先保持 legacy 路径不变。

### 模型

- `src/openpi/models/completion.py`
  - 保留 `CompletionHead` legacy 行为。
  - 新增 `TemporalCompletionHead` 或显式 temporal variant。
  - 新增 masked-mean helper（若不放在 `pi0.py`）。
- `src/openpi/models/pi0_config.py`
  - 为 completion head 增加显式 variant/temporal_steps/hidden_dim/pooling 配置。
  - 参数分类仍把新 head 归入 completion group。
  - 在 `src/openpi/training/config_test.py` / model tests 中验证新的参数路径仍被 completion-only freeze/filter 正确分类。
- `src/openpi/models/pi0.py`
  - 新增无状态 `compute_prefix_feature`。
  - 新增接收 `[B,3,D]` 的 temporal logits API。
  - legacy `compute_completion_logits` 保持兼容。
  - 若训练不使用 feature cache，可将 `[B,3,...]` flatten 为 `3B` 做一次 batched prefix forward，不能用 Python 循环重跑三次。

### 数据与采样

- 建议新增 `src/openpi/training/temporal_completion_data.py`
  - subtask group 审计。
  - logical trajectory/boundary/tick/source-frame 映射。
  - 72/8/20 split manifest。
  - temporal sample index 和自然分布 eval index。
- `src/openpi/training/data_loader.py`
  - 新增 temporal cached-feature dataset/data loader。
  - 新增严格 21/21/22 的 batch sampler。
  - 不复用旧 `BoundaryCompletionSampler` 的“连续10帧正样本”语义。
- `src/openpi/training/completion_data.py`
  - 若 temporal manifest/index 放入新模块，现有 completion data 入口仍需显式路由到新 schema，并拒绝把旧 boundary manifest 当 temporal manifest。
- `src/openpi/training/completion.py`
  - 新增 temporal 配置字段和校验。
  - 明确 balanced batch 时使用 unweighted BCE。
- `src/openpi/training/config.py`
  - 新增独立的 clean temporal completion config。
  - 显式设置 clean checkpoint、`training_time_rtc.enabled=False`、`batch_size=64`，并用整数配置固定 `positive=21 / hard=21 / ordinary=22`，避免浮点 fraction 取整产生不同 batch。
  - 不改变旧 S1/S2/TTRTC config。

### 训练、提取和评测

- `scripts/train.py`
  - temporal feature batch 分支。
  - unweighted BCE 路径。
  - val checkpoint selection 所需的序列指标。
- 建议新增 `scripts/build_temporal_completion_manifest.py`
  - 只读源数据，生成版本化 split/sample manifest。
- 建议新增 `scripts/align_breakfast_trajectory_identity.py`
  - 在 split 前建立 full episode 与 subtask group 的身份映射；输出 matched/quarantine 清单。
- 建议新增 `scripts/extract_temporal_prefix_features.py`
  - clean checkpoint 特征缓存。
- 建议新增 `scripts/evaluate_temporal_completion.py`
  - oracle/closed-loop、自然分布、边界指标和曲线输出。
- 后续新增 `scripts/align_full_breakfast_boundaries.py`
  - 在已确定 identity 的 pair 内，把四个 subtask 首尾帧对齐到 full-video frame index，用于真实完整视频评测。
  - full-video 的 `B_i/C_i` 必须从 full-frame alignment 重新计算，不能直接复用虚拟 concat 的累计长度。
- policy/deploy 层新增 prompt controller；具体文件在接入线上推理代码时确定。
  - 接入点预计涉及 `src/openpi/policies/policy.py` 或在线执行脚本。
  - 通过兼容 API/aux return 暴露首次 prefix/KV forward 的 pooled `z`，不能破坏既有 `sample_actions` 默认返回协议。

## 13. 必须补的测试

### 数据构造单测

- 每四个 episode 组成一个 trajectory，task 顺序必须严格为 0/1/2/3。
- 缺 episode、重复 task、错误 task 顺序、空 episode、异常短 episode 必须 fail closed。
- `Bi`、`Ci` 的 off-by-one 和余数 0/1/14 情况。
- `C_i-C_(i-1)=44` 必须不可达并 quarantine，`=45` 必须刚好能组成第一条合法 triplet。
- 2 Hz 相位在四段拼接后保持全局一致，不在 subtask 处 reset。
- logical tick 到 source episode/frame 的映射跨 task 边界正确。
- task 3 terminal hold 正确且只产生一个正 tick。
- 每个 `(trajectory, task)` 恰好一个正样本。
- hard negative 恰好来自正 tick 前 1～4 个合法 tick。
- history gap 恰好 15/15，缺历史整条丢弃且无 padding。
- positive triplet 的三个 prefix 都使用旧 prompt。
- prompt override 必须发生在自动 task/default-prompt transform 之前。
- full observation 已有完整早餐 prompt 时，controller 仍显式写入 current subtask prompt。
- prompt 切换后历史清空，不跨 prompt 拼 triplet。

### split/manifest 单测

- 固定 seed 重复生成内容相同。
- train/val/test trajectory 集合两两不交。
- 同 trajectory 的所有 source episodes/ticks/prompts 同 split。
- 比例取整和最终计数写入 manifest。
- metadata fingerprint 变化时旧 manifest 被拒绝。
- 737/736 的 unmatched 情况显式记录，不静默按 episode id 偏移。
- full test episode 对应的四个 subtask episode 与 train subtask 集合交集必须为空。

### sampler 单测

- 每个完整 batch 精确为 21/21/22。
- 同 batch 无重复 positive event。
- 有合法 pre-boundary decision row 时，paired hard negative 与 positive 属于同一 trajectory/task/boundary；45 帧最短可达事件只能使用显式审计的 same-task cross-trajectory fallback。
- ordinary pool 不含 positive/hard rows。
- 长时间运行后 task 分布近似均匀。
- seed、epoch、resume 状态可复现。
- val/test loader 保持自然分布且不 replacement。

### 模型/损失单测

- `[B,3,2048] -> [B]` shape。
- masked mean 对 padding token 不敏感，并拒绝 malformed mask/shape。
- time step 共享 LayerNorm。
- oldest-to-newest 拼接顺序不会被反转。
- head 输入/参数/logit 为 FP32，即使缓存是 FP16。
- dropout 只在 train 启用，eval 确定性。
- backbone/action 无梯度，只有 completion 参数更新。
- BCE-with-logits 数值与参考实现一致，且 `pos_weight == 1`。
- loss 路径不包含 sigmoid/focal/progress。
- temporal checkpoint 只允许新 head 参数缺失。
- action-forward 复用得到的 `z` 与独立 `compute_prefix_feature` 在同 observation/prompt 下数值一致。

### controller/evaluator 单测

- deque 未满不触发。
- 触发后 prompt 只前进一步、只 latch 一次并清空 deque。
- 不在同一 raw frame 下重跑新 prompt。
- task 3 触发进入 done。
- oracle 和 closed-loop 模式产生预期不同的错误传播。
- 唯一正 tick 漏检后的 delayed recovery/miss 匹配窗口符合 10.3，且不会改写监督标签。
- top-1、hard AUPRC、margin、early trigger、delay 的 synthetic 示例数值正确。
- 阈值只能由 val 计算；test API 不接受“自动搜索最佳 test 阈值”。
- 集成测试记录并校验 observation/feature/score timestamp；报告 completion/controller 增量 latency 的 p50/p95。

### 最小集成测试

构造一个四段 synthetic trajectory，覆盖：

- 一个边界正好在 15 的倍数。
- 一个边界后延 14 帧才命中 tick。
- 一个 task 3 terminal hold。
- 一个早触发导致 closed-loop prompt 错位。

从 manifest -> sample index -> feature triplet -> batch -> loss -> evaluator 全链路跑通。

## 14. 实现顺序

1. **只读数据审计与 manifest**
   - 审计 2944 subtask episode、736 group 和 737 full episode。
   - 先完成 full episode ↔ subtask group 的 trajectory identity mapping；歧义/unmatched 进入 quarantine。
   - 生成固定 72/8/20 split，不读取 test 特征做选择。
2. **纯索引标签器**
   - 实现 logical trajectory、`Bi/Ci`、source mapping、prompt/reset 和 triplet index。
   - 先在 synthetic 数据上通过全部时间语义单测。
3. **clean prefix cache**
   - 提取 masked-mean FP32 prefix，缓存按 prompt 区分。
   - 用少量样本验证 batch/单样本提取一致。
4. **TemporalCompletionHead + sampler + BCE**
   - 先单设备跑通，只更新 head。
   - 验证每 batch 21/21/22 和 FP32/stop-gradient 审计。
5. **自然分布 oracle 评测**
   - 三个 seed。
   - 同一 eligible sample set 上对照 current-only 与三帧历史，避免集合差异。
6. **prompt controller 与 closed-loop 评测**
   - 先在虚拟拼接轨迹上验证。
7. **真实完整轨迹对齐与最终 test**
   - 在 step 1 已确定的 identity pair 内做 full-video frame boundary alignment。
   - 按 full-frame 边界重新计算真实完整轨迹的 `B_i/C_i`，不复用虚拟 concat 累计长度。
   - 冻结 threshold/trigger 规则后才运行 20% test。

## 15. 验收条件

数据层：

- 所有纳入轨迹都有四个顺序正确的 subtask。
- 每个 task 恰好一个正 tick，历史精确为前两个 2 Hz tick。
- split 在 trajectory 层完全隔离；test 未被训练/阈值选择读取。
- full test 对应的所有 subtask source episode 与 train source episode 零交集。
- 737/736 mismatch 有明确审计报告和 exclusion/mapping 记录。

训练层：

- 每个 batch 严格 21 positive / 21 hard negative / 22 ordinary negative。
- 实际 loss 是 unweighted BCE-with-logits。
- 只有 temporal completion head 参数发生变化。
- 至少三个 seed，结果不只报告最好一次。

评测层：

- val/test 使用自然 2 Hz 分布。
- 主报告包含 top-1 boundary rate、hard AUPRC、boundary margin、early trigger、delay、负区间波动和按 task 分层结果。
- 同时提供 oracle-prompt 与 closed-loop-prompt 结果。
- test threshold 不从 test 标签选择。
- current-only 与三帧 temporal 模型在同 eligible rows、同 split、同 seed 上完成 paired ablation。

部署层：

- 当前 prefix 从 action-policy forward 复用，无第二次 VLM forward。
- prompt 切换后 deque reset，三 tick warm-up，无 padding。
- task 3 能可靠进入 done 状态。
- 报告 completion/controller 增量 latency 的 p50/p95；若 cadence 不满足复用条件，明确计入 prefix-only fallback 延迟。

## 16. 明确不做的事情

- 不修复或更新任务服务器 NVIDIA 驱动/环境。
- 不修改原始数据集、视频或已有 checkpoint。
- 第一版不使用 TTRTC、历史 action、action hidden 或 progress。
- 不沿用旧的“最后5帧+下一段前5帧，共10个正样本”标签。
- 不用每个 subtask 自己的 frame 0 重置 2 Hz 相位。
- 不用 test 集选 checkpoint、正则、阈值或 MLP 结构。
- 不把 balanced train sigmoid 当作真实完成概率。

## 17. 结果解释限制

clean pi0.5 backbone 本身曾使用 breakfast 数据训练。72/8/20 trajectory split 保证的是：completion head、batch sampler、checkpoint 选择和阈值不会看到 held-out trajectory 的 completion 监督；它不代表 frozen backbone 从未见过这些视觉轨迹。最终报告必须明确区分“completion-head held-out 泛化”和“backbone 对完全未见数据的泛化”。
