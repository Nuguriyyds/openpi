# Temporal completion head 实现清单（subtask 反向采样版）

本阶段只验证 clean pi0.5 的 frozen prefix 是否能判断单个 subtask 的人工完成事件。完整早餐 trajectory 只作为 split 单位；它不再被拼成逻辑连续视频，也不参与 prompt 或历史构造。

## 固定输入

- 数据：`agilex_make_breakfast_subtask_730`。
- clean config：`pi05_730_breakfast_subtasks`。
- clean checkpoint：`/mnt/data/models/wyt/checkpoints/pi05_730_breakfast_subtasks/breakfast_subtasks_bs64_50k/49999`。
- TTRTC、action hidden、progress、full-video prompt、closed-loop controller 均关闭。
- backbone 完全冻结；completion head 为 FP32。
- 每四个连续 subtask episode 为一个 `group_id = episode_id // 4`，按 group 用 seed 42 做 72%/8%/20% train/val/test split。

## 单 subtask 结束帧和样本

审计 parquet 后确认 `frame_index` 连续为 `0..L-1` 时，人工完成帧是 inclusive 的 `E=L-1`。`E<45` 的 episode 不产生样本；它不使同组其他 episode 改变 split，也不做 padding 或跨 episode 补历史。

所有窗口都在同一个 subtask episode 内，三个 frame 使用完全相同的该 subtask prompt，帧间隔严格为 15：

| 样本 | 当前帧 | history（最旧到最新） | label |
|---|---:|---|---:|
| positive | `E` | `[E-30, E-15, E]` | 1 |
| hard | `E-15` | `[E-45, E-30, E-15]` | 0 |
| ordinary | `E-30, E-45, ...` 且 `current>=30` | `[t-30,t-15,t]` | 0 |

因此 `45<=E<60` 的 episode 只有 positive 和 hard，没有 ordinary candidate；`E>=60` 才进入 ordinary episode pool。每个合法 episode 恰好一个 positive 和一个 hard，ordinary candidate 可以有多个，训练时每次重新抽取一个。

每帧特征为：

```text
z_t = FP32 masked_mean(prefix_out(image_t, subtask_prompt))
```

prompt 必须在自动 prompt/repack/normalize transform 之前显式写入。completion head 输入是 `[z_(t-30), z_(t-15), z_t]`；current-only 对照使用同一个 head 和参数量，但把输入变成 `[0,0,z_t]`。特征 stop-gradient，不使用 action/state。

## 训练 batch 和损失

batch 固定 64，由 32 个不同 subtask episode pair 展开：每个 task 恰好 8 个 episode，其中 4 个 hard pair、4 个 ordinary pair。展开后严格为：

```text
32 positive + 16 hard negative + 16 ordinary negative
```

同一 batch 内 episode 不重复，跨 batch 可以重复。hard 固定为该 episode 的 `E-15`；ordinary 从该 episode 的全部 ordinary candidates 均匀随机重采样。任一 task 的 ordinary pool 少于 4 个不同 episode，或无法凑出 8 个不重复 episode，直接报清晰错误，不跨 task 补样本。

head 结构保持现有 temporal MLP：shared LayerNorm → flatten concat → Linear(3D,128) → GELU → Dropout(0.1) → Linear(128,1)。损失固定为 `binary_cross_entropy_with_logits`，`pos_weight=1`，不使用 class weight、focal、ranking 或 phase augmentation。

配置名保持：

- `pi05_agilex_breakfast_temporal_completion_head`
- `pi05_agilex_breakfast_temporal_completion_current_only_head`

## val/test 和 checkpoint 选择

val/test 使用自然 candidate set，每个合法 episode 的 positive、hard 和全部 ordinary candidate 各评估一次，不使用平衡 sampler。报告 overall 和 task 0/1/2/3：natural AUPRC、ROC-AUC、positive-vs-hard-only AUPRC、paired ordering accuracy、paired hard margin（mean/median/p25/p75）、ordinary score mean/p95。

checkpoint 只按 val threshold-free 指标排序：

1. natural AUPRC 越高越好；
2. hard-only AUPRC 越高越好；
3. paired hard margin median 越高越好。

训练不根据 early-trigger 或复杂 threshold 搜索选择 checkpoint。固定 0.5 只用于兼容性的报告字段。

## 本阶段明确不做

- 四段 subtask 逻辑拼接、全局 `0,15,30` 相位和 terminal hold；
- next-subtask 图像配 old prompt；
- 跨 subtask 历史、prompt reset 后补历史；
- full-video identity map/evidence、真实 full-video 对齐；
- closed-loop 控制、phase augmentation、action hidden/TTRTC/progress；
- 新增 hash、fingerprint、manifest evidence 或复杂 cache 绑定。

## 最小验收

只保留三个 focused test：

1. `E=91` 数据索引：positive `[61,76,91]`、hard `[46,61,76]`、ordinary current `[61,46,...]`，三帧同 episode/同 prompt；
2. sampler：每 batch `32/16/16`，每 task `8 positive/4 hard/4 ordinary`，32 个 episode 不重复；
3. cached-feature train step：loss finite、head 变化、frozen backbone 不变。

相位鲁棒性、真实 full-video 和控制逻辑留到后续阶段。
