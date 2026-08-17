# Temporal completion head 运行手册

本文对应 `temporal_completion_head_implementation_checklist.md` 中已经实现的第一阶段：在 subtask 数据构成的逻辑完整轨迹上训练和评估
`[prefix(t-30), prefix(t-15), prefix(t)] -> MLP -> completion logit`。

## 当前交付边界

已实现：

- 四段 subtask group 构成逻辑轨迹并做 72/8/20 trajectory split；
- 严格全局 2 Hz tick、每个边界唯一正标签、同 prompt 三帧历史；
- clean pi0.5 masked-mean prefix cache（feature-cache schema v3）；
- FP32 temporal MLP、unweighted BCE、21/21/22 train sampler、自然 val/test；
- history 与 same-head current-only 两个完全同结构消融；
- train-only current-prefix 线性信息 probe；
- validation-only checkpoint/threshold 选择及 oracle-prompt evaluator；
- 可复用 action forward prefix 的 prompt controller 基础组件。

尚未宣称完成：

- 真实 full-video 内四个边界的逐帧对齐；当前监督和 oracle 评估来自四段 subtask 的逻辑拼接；
- 漏检后继续使用旧 prompt 的 closed-loop 特征提取与评估；
- 与实际 30 fps action scheduler/`ActionChunkBroker` 的接线、pending action chunk 清理和端到端延迟测量。

因此，在完成最后三项前，报告必须写成“逻辑完整轨迹、oracle prompt 结果”，不能写成“真实完整视频 closed-loop 结果”。

## 只读输入

```text
subtask dataset:
/mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_subtask_730

clean checkpoint:
/mnt/data/models/wyt/checkpoints/pi05_730_breakfast_subtasks/breakfast_subtasks_bs64_50k/49999
```

真实 full-video 评测阶段才额外读取
`/mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_730`。

所有生成物必须写到 `/mnt/data/models/wyt/...` 的新目录，不得写入或覆盖上述 dataset/checkpoint。

本流程按当前决策不计算或校验 SHA-256。数据变化后需要主动重新生成 manifest/cache；加载时只校验
schema、显式路径/config、prompt、canonical rows、shape、dtype 和数值有限性。

## 运行顺序

### 1. 封存逻辑轨迹 manifest

当前训练阶段不需要 identity map，也不读取 full dataset。脚本直接将每四个
subtask episode 组成一条逻辑轨迹，再以轨迹为单位封存 train/val/test split。

```bash
uv run scripts/build_temporal_completion_manifest.py \
  --subtask-root /mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_subtask_730 \
  --subtask-repo-id modanqing/agilex_make_breakfast_subtask_730 \
  --output /mnt/data/models/wyt/split_manifests/agilex_make_breakfast_temporal_completion_v3.json \
  --audit-summary /mnt/data/models/wyt/split_manifests/agilex_make_breakfast_temporal_completion_v3_audit.json
```

必须人工检查 audit summary 中的 group 数量、reachability exclusion 和最终 train/val/test 数量。
只有以后进行真实 full-video 评测时，才同时传入 `--full-root`、`--full-repo-id`
和 `--identity-map`，生成独立的 `full_identity` manifest。

### 2. 提取一次共享 prefix cache

必须在最终代码版本上提取。代码不计算内容哈希；运行时通过 manifest 行、prompt、config 名称和 checkpoint 路径检查 cache。

```bash
uv run scripts/extract_temporal_completion_features.py \
  --manifest /mnt/data/models/wyt/split_manifests/agilex_make_breakfast_temporal_completion_v3.json \
  --config-name pi05_730_breakfast_subtasks \
  --checkpoint /mnt/data/models/wyt/checkpoints/pi05_730_breakfast_subtasks/breakfast_subtasks_bs64_50k/49999 \
  --dataset-root /mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_subtask_730 \
  --output /mnt/data/models/wyt/evaluations/temporal_completion_prefix_features_v3/features.npz
```

history/current-only 共用这一份 cache；不要复制数据集，也不要复用旧 10-frame positive cache。

### 3. 拟合线性 current-prefix 信息 probe

```bash
uv run scripts/fit_temporal_current_only_probe.py \
  --output /mnt/data/models/wyt/evaluations/temporal_completion_prefix_features_v3/current_only_linear_probe.npz
```

该 probe 仅用自然 train 拟合、自然 val 选 L2；不会索引 test。它用于表示信息量诊断，不替代下面的 same-head MLP 消融。

### 5. 训练三个 seed 的 history 与 current-only MLP

history config：`pi05_agilex_breakfast_temporal_completion_head`  
current-only config：`pi05_agilex_breakfast_temporal_completion_current_only_head`

对 seed `42/43/44` 分别运行，且每个实验使用独立 `exp-name`：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_agilex_breakfast_temporal_completion_head \
  --exp-name=history_seed42 --seed=42 --overwrite

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_agilex_breakfast_temporal_completion_current_only_head \
  --exp-name=current_only_seed42 --seed=42 --overwrite
```

将 `42` 替换为 `43/44`。两种 config 的模型结构、参数数、sample rows、sampler 和 seed 完全相同；current-only 只把前两段历史槽置零。

### 6. 最终 oracle-prompt 评估

每个 run 只允许使用训练生成的 `best_temporal_validation.json`；evaluator 会要求 validation 已覆盖配置的最后一步，并拒绝搜索 latest checkpoint 或 test threshold。

```bash
uv run scripts/evaluate_temporal_completion.py \
  --config-name pi05_agilex_breakfast_temporal_completion_head \
  --checkpoint-root /mnt/data/models/wyt/checkpoints/pi05_agilex_breakfast_temporal_completion_head/history_seed42 \
  --current-only-cache /mnt/data/models/wyt/evaluations/temporal_completion_prefix_features_v3/features.npz \
  --current-only-probe-weights /mnt/data/models/wyt/evaluations/temporal_completion_prefix_features_v3/current_only_linear_probe.npz \
  --output /mnt/data/models/wyt/evaluations/temporal_completion_reports/history_seed42.json
```

current-only MLP 使用其自己的 config/checkpoint-root 运行同一 evaluator。三个 seed 全部报告均值和标准差，不能只选最好的 seed。

## 验收时必须核对

- manifest 行、prompt、cache schema、config 名称和 checkpoint 路径匹配；
- 每个 train batch 为 21 positive、21 hard、22 ordinary；
- test 没参与训练、L2、checkpoint 或 threshold 选择；
- history 与 current-only 的 `temporal_input_mode` 和 checkpoint artifact 匹配；
- 主报告包含 natural/hard-local AUPRC、boundary top-1、margin、early trigger、false trigger/min、负区间相邻 2 Hz 波动和四个 task 分层结果；
- 报告明确写 `closed_loop_evaluated=false`，直到真实 scheduler/full-video 对齐完成。
