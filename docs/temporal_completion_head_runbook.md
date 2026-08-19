# Temporal completion head 运行手册

本手册对应 subtask 反向 2 Hz 采样版实现。完整 breakfast trajectory 只用于 group split，不参与训练样本的图像拼接或 prompt 构造。

## 固定路径

```text
数据：/mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_subtask_730
clean config：pi05_730_breakfast_subtasks
clean checkpoint：/mnt/data/models/wyt/checkpoints/pi05_730_breakfast_subtasks/breakfast_subtasks_bs64_50k/49999
manifest：/mnt/data/models/wyt/split_manifests/agilex_make_breakfast_temporal_completion_v4.json
cache：/mnt/data/models/wyt/evaluations/temporal_completion_prefix_features_v5/features.npz
```

不传 `--full-root`、`--full-repo-id` 或 `--identity-map`。训练路径不需要 full-video identity map。

## 1. 构造 subtask manifest

```bash
uv run scripts/build_temporal_completion_manifest.py \
  --subtask-root /mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_subtask_730 \
  --subtask-repo-id modanqing/agilex_make_breakfast_subtask_730 \
  --output /mnt/data/models/wyt/split_manifests/agilex_make_breakfast_temporal_completion_v4.json \
  --audit-summary /mnt/data/models/wyt/split_manifests/agilex_make_breakfast_temporal_completion_v4_audit.json
```

审计摘要会报告每个 task 的 `E<45` 排除数和保留 episode 中 `45<=E<60`（无 ordinary candidate）数量。`E` 是 parquet 最后一行的 inclusive `frame_index`。

## 2. 提取共享 prefix cache

```bash
uv run scripts/extract_temporal_completion_features.py \
  --manifest /mnt/data/models/wyt/split_manifests/agilex_make_breakfast_temporal_completion_v4.json \
  --config-name pi05_730_breakfast_subtasks \
  --checkpoint /mnt/data/models/wyt/checkpoints/pi05_730_breakfast_subtasks/breakfast_subtasks_bs64_50k/49999 \
  --dataset-root /mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_subtask_730 \
  --output /mnt/data/models/wyt/evaluations/temporal_completion_prefix_features_v5/features.npz \
  --storage-dtype float16 \
  --batch-size 32
```

提取只运行 clean prefix forward，不运行 action sampling，不读取 action hidden。history/current-only 共用这份 cache。

提取过程会在目标旁边增量写入 `<output>.partial/unique_features.npy` 和
`<output>.partial/progress.json`。每个 batch 完成后才推进进度；如果进程中断，重新执行同一条命令会从最近一个已完成 batch 继续，不会重新提取已经落盘的 prefix。最终 `features.npz` 写成功后，目标文件仍按不可覆盖处理。

## 3. 训练 history head

```bash
uv run scripts/train.py \
  pi05_agilex_breakfast_temporal_completion_head \
  --exp-name history_seed42
```

## 4. 训练 current-only 对照

```bash
uv run scripts/train.py \
  pi05_agilex_breakfast_temporal_completion_current_only_head \
  --exp-name current_only_seed42
```

两者使用相同 manifest/cache、split、seed、MLP 和 32/16/16 sampler，区别只有 temporal input mode。

## 5. val/test 报告

```bash
uv run scripts/evaluate_temporal_completion.py \
  --config-name pi05_agilex_breakfast_temporal_completion_head \
  --checkpoint-root /mnt/data/models/wyt/checkpoints/pi05_agilex_breakfast_temporal_completion_head/history_seed42 \
  --output /mnt/data/models/wyt/evaluations/temporal_completion_reports/history_seed42.json

uv run scripts/evaluate_temporal_completion.py \
  --config-name pi05_agilex_breakfast_temporal_completion_current_only_head \
  --checkpoint-root /mnt/data/models/wyt/checkpoints/pi05_agilex_breakfast_temporal_completion_current_only_head/current_only_seed42 \
  --output /mnt/data/models/wyt/evaluations/temporal_completion_reports/current_only_seed42.json
```

evaluator 只读取自然 val/test candidate set；checkpoint 由 val 的 natural AUPRC、hard AUPRC、paired hard margin median 选定，test 不重新选阈值。报告是 oracle prompt、非 closed-loop 结果。

## 6. focused checks

```bash
uv run pytest -q src/openpi/training/temporal_completion_subtask_test.py scripts/train_temporal_integration_test.py
uvx ruff check src/openpi/training/temporal_completion_data.py src/openpi/training/temporal_completion_sampler.py src/openpi/training/temporal_completion_features.py src/openpi/training/completion.py src/openpi/training/data_loader.py src/openpi/training/config.py scripts/train.py scripts/extract_temporal_completion_features.py scripts/evaluate_temporal_completion.py
git diff --check
```

val/test 只评估自然 candidate set；本阶段不做 closed-loop 控制和真实 full-video 评估。
