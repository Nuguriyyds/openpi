# Raw-prefix Completion Transformer Expert 运行手册

这条路径只训练 `completion_head/*`。clean Pi0.5 的 VLA 和 action expert 从固定 checkpoint 加载并冻结；训练阶段只读取已经提取的当前帧 raw-prefix cache，不运行 VLA。以下命令是远程训练机上的模板，本地开发环境不要执行提取、训练或完整评测命令。

## 固定输入和输出

```text
clean config:
  pi05_730_breakfast_subtasks
clean checkpoint:
  /mnt/data/models/wyt/checkpoints/pi05_730_breakfast_subtasks/breakfast_subtasks_bs64_50k/49999
subtask dataset:
  /mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_subtask_730
full episode dataset:
  /mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_730
manifest:
  /mnt/data/models/wyt/split_manifests/agilex_make_breakfast_temporal_completion_v4.json
raw cache:
  /mnt/data/models/wyt/evaluations/current_raw_prefix_tokens_v1
```

manifest 的 `subtask_local` canonical rows 是唯一标签来源：每个 task 使用 `positive=E`、`hard_negative=E-15`、`ordinary_negative=E-30,E-45,...`，不含 history、transition 或 terminal carry。训练 batch 固定为 `32/16/16/0`，validation/test 使用自然候选顺序。

## 1. 提取 current raw-prefix cache

```bash
uv run scripts/extract_current_raw_prefix_tokens.py \
  --config-name pi05_730_breakfast_subtasks \
  --checkpoint /mnt/data/models/wyt/checkpoints/pi05_730_breakfast_subtasks/breakfast_subtasks_bs64_50k/49999 \
  --dataset-root /mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_subtask_730 \
  --manifest /mnt/data/models/wyt/split_manifests/agilex_make_breakfast_temporal_completion_v4.json \
  --output /mnt/data/models/wyt/evaluations/current_raw_prefix_tokens_v1 \
  --batch-size 16
```

脚本会复用现有 evaluation repack/norm/tokenizer，`train=False`，只读取每个 canonical row 的当前 source frame。cache 中 `prefix_out` 为 float16、`prefix_mask` 为 bool，并保存一份共享的 `prefix_segment_ids` / `prefix_position_ids`。目标目录已存在时命令会拒绝覆盖。

## 2. 启动 4000-step head training

```bash
uv run scripts/train.py \
  pi05_agilex_breakfast_raw_prefix_completion_head \
  --exp-name raw_prefix_seed42
```

配置已经固定 `batch_size=64`、`num_train_steps=4000`、`warmup_steps=100`、`peak_lr=3e-5`、`decay_lr=3e-6`、`weight_decay=1e-4`、`gradient_clip_norm=1.0`、`seed=42`、`ema_decay=None`、`num_workers=0`、`fsdp_devices=1`。checkpoint 应保存为 200、400、…、4000 step。

## 3. 对 200–4000 step 做自然 validation

下面循环只读取 `val` 自然候选分布；它不使用训练时的 32/16/16 重采样。先比较每份 JSON 的 `metrics.macro_task_auprc`，再人工选择 validation 最佳 step，不要在代码中预设 1000 或 2000 为最佳。

```bash
for step in $(seq 200 200 4000); do
  uv run scripts/evaluate_raw_prefix_completion.py \
    --config-name pi05_agilex_breakfast_raw_prefix_completion_head \
    --checkpoint /mnt/data/models/wyt/checkpoints/pi05_agilex_breakfast_raw_prefix_completion_head/raw_prefix_seed42/${step} \
    --cache /mnt/data/models/wyt/evaluations/current_raw_prefix_tokens_v1 \
    --split val \
    --output /mnt/data/models/wyt/evaluations/raw_prefix_reports/raw_prefix_seed42_step${step}_val.json \
    --predictions-output /mnt/data/models/wyt/evaluations/raw_prefix_reports/raw_prefix_seed42_step${step}_val_predictions.json
done
```

报告包含 overall/per-task sample count、AUPRC、AUROC、BCE、positive/hard/ordinary score mean、positive-hard ordering accuracy 以及 margin mean/median。

## 4. 对 validation 选出的 checkpoint 做 test

把 `<BEST_STEP>` 替换成第 3 步根据 validation macro task AUPRC 选出的 step：

```bash
uv run scripts/evaluate_raw_prefix_completion.py \
  --config-name pi05_agilex_breakfast_raw_prefix_completion_head \
  --checkpoint /mnt/data/models/wyt/checkpoints/pi05_agilex_breakfast_raw_prefix_completion_head/raw_prefix_seed42/<BEST_STEP> \
  --cache /mnt/data/models/wyt/evaluations/current_raw_prefix_tokens_v1 \
  --split test \
  --output /mnt/data/models/wyt/evaluations/raw_prefix_reports/raw_prefix_seed42_step<BEST_STEP>_test.json \
  --predictions-output /mnt/data/models/wyt/evaluations/raw_prefix_reports/raw_prefix_seed42_step<BEST_STEP>_test_predictions.json
```

test 只报告自然分布结果，不参与 checkpoint 或 threshold 选择。

## 5. 完整 episode 半闭环评测

半闭环使用完整 episode 数据进行时间轴映射，初始只用 task0 prompt；每个 2 Hz 判断点用当前 active-task prompt 重新计算一次 raw prefix，不维护 prefix history。仍保留 gated terminal hold 和 2 秒 timeout。

```bash
uv run scripts/evaluate_temporal_completion_semiclosed.py \
  --mode raw_prefix_current \
  --config-name pi05_agilex_breakfast_raw_prefix_completion_head \
  --checkpoint /mnt/data/models/wyt/checkpoints/pi05_agilex_breakfast_raw_prefix_completion_head/raw_prefix_seed42/<BEST_STEP> \
  --full-dataset-root /mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_730 \
  --subtask-dataset-root /mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_subtask_730 \
  --manifest /mnt/data/models/wyt/split_manifests/agilex_make_breakfast_temporal_completion_v4.json \
  --threshold <VALIDATION_SELECTED_THRESHOLD> \
  --output /mnt/data/models/wyt/evaluations/raw_prefix_reports/raw_prefix_seed42_step<BEST_STEP>_semiclosed.json
```

`--threshold` 可替换为用户指定的自定义阈值。输出的每个 tick 保留 active task/prompt、source frame、target boundary、logit、sigmoid score、threshold、classification 和 switch reason；summary 包含 early 0.5s、early 至少 1s、on-time、late trigger、timeout forced、timing error mean/median/MAE 及对应 episode IDs。

## 6. 本地验证边界

本地只运行不需要远程数据、远程 checkpoint 或 GPU 的定向单元测试。完整 cache 提取、4000-step 训练、自然 val/test 推理和 full-episode 半闭环命令留给远程训练机执行。

