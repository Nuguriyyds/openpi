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

提取过程会在目标旁边增量写入 `<output>.partial/batches/*.npy` 和
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

## 7. History-carry transition 消融

该消融保留子任务切换时上一任务已经生成的两个 prefix。普通
positive/hard/ordinary 样本仍使用单个子任务自己的 prompt；只有 transition
negative 的历史槽位使用混合 prompt：

- step0: `[previous(E-15), previous(E), current(0)]`
- step1: `[previous(E), current(0), current(15)]`

旧的 v5 cache 和 `pi05_agilex_breakfast_temporal_completion_head` 配置不变。
消融使用独立 cache：

```text
/mnt/data/models/wyt/evaluations/temporal_completion_prefix_features_history_carry_v1/features.npz
```

```bash
uv run scripts/extract_temporal_completion_features.py \
  --manifest /mnt/data/models/wyt/split_manifests/agilex_make_breakfast_temporal_completion_v4.json \
  --config-name pi05_730_breakfast_subtasks \
  --checkpoint /mnt/data/models/wyt/checkpoints/pi05_730_breakfast_subtasks/breakfast_subtasks_bs64_50k/49999 \
  --dataset-root /mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_subtask_730 \
  --sampling-protocol history_carry \
  --output /mnt/data/models/wyt/evaluations/temporal_completion_prefix_features_history_carry_v1/features.npz \
  --storage-dtype float16 \
  --batch-size 32
```

训练配置固定为 `16 positive / 16 hard / 28 ordinary / 4 transition`、
batch size 64、2000 optimizer steps，并使用未加权 BCE：

```bash
uv run scripts/train.py \
  pi05_agilex_breakfast_temporal_completion_history_carry_head \
  --exp-name history_carry_seed42
```

```bash
uv run scripts/evaluate_temporal_completion.py \
  --config-name pi05_agilex_breakfast_temporal_completion_history_carry_head \
  --checkpoint-root /mnt/data/models/wyt/checkpoints/pi05_agilex_breakfast_temporal_completion_history_carry_head/history_carry_seed42 \
  --output /mnt/data/models/wyt/evaluations/temporal_completion_reports/history_carry_seed42.json
```

## 8. 生成逐样本曲线 HTML

评估命令只生成汇总 JSON。需要查看类似“target / predicted score / threshold”
的曲线时，先完成第 7 节的评估，再运行独立可视化脚本。它复用同一个
validation-selected checkpoint、feature cache 和 threshold，不重新搜索 test
threshold。默认绘制 test split 中的所有 trajectory/task，可在 HTML 下拉框中切换。

```bash
uv run scripts/visualize_temporal_completion.py \
  --report /mnt/data/models/wyt/evaluations/temporal_completion_reports/history_carry_seed42.json \
  --output /mnt/data/models/wyt/evaluations/temporal_completion_reports/history_carry_seed42_curves.html \
  --split both \
  --predictions-json /mnt/data/models/wyt/evaluations/temporal_completion_reports/history_carry_seed42_curves_predictions.json
```

该命令会复用评估报告中记录的 checkpoint、cache 和 validation threshold，
并在 HTML 旁边生成可复用的 `.predictions.npz` sidecar。以后再次运行相同
报告时，如果 sidecar 仍匹配，会直接复用预测，不重复做 prefix-head 推理。

只查看某一个 task 时，可以追加：

```bash
  --task-index 2
```

`--task-index` 取值为 `0/1/2/3`。如果不传该参数，HTML 会为所有 trajectory/task
建立下拉选项；打开页面后可在 Episode 下拉框中选择具体曲线。若只想生成
少量均匀分布的 episode 预览，可追加 `--max-episodes 20`。

输出文件：

- `*_curves_predictions.json`：每个候选行的 current frame、label、score、
  `sample_kind` 和 episode/task 信息。
- `*_curves.html`：单文件离线页面，浏览器直接打开即可，不依赖 Plotly 或网络。
- `*.predictions.npz`：脚本内部复用的预测 sidecar；不需要手工打开。

横轴是当前 task 内的 source frame（30 fps，候选点按 15 帧即 2 Hz 采样），
不是完整视频的每一个 30 fps 图像。曲线含义如下：

- 绿色阶梯线：人工 label；
- 橙色线：completion head 的 sigmoid score；
- 紫色虚线：只由 val 选择的 threshold；
- 彩色圆点：`positive`、`hard_negative`、`ordinary_negative` 和
  `transition_negative`，history-carry 的 transition 会保留在图中。

该页面仍然是 oracle-prompt candidate-set 可视化，不代表 closed-loop 部署
曲线；history-carry 的 transition 行会显示旧 prompt 与新 prompt 混合历史的
样本语义。

current-only 或 subtask-local baseline 的可视化只需替换
`--report` 和输出文件名，
命令结构完全相同。

## 9. History-carry 的 `pos_weight=2.0` 消融

原始 `pi05_agilex_breakfast_temporal_completion_history_carry_head` 保持
`pos_weight=1.0`。`pos_weight=2.0` 使用独立配置和独立实验目录，避免覆盖
原实验的 checkpoint 或 validation artifact：

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/train.py \
  pi05_agilex_breakfast_temporal_completion_history_carry_posweight2_head \
  --exp-name history_carry_seed42_single_gpu_posweight2
```

该配置仍使用相同的 history-carry cache、16/16/28/4 batch、2000 steps，
只有 BCE 正类权重改为 `2.0`。评估时也必须使用新的 config 和 checkpoint root：

```bash
uv run scripts/evaluate_temporal_completion.py \
  --config-name pi05_agilex_breakfast_temporal_completion_history_carry_posweight2_head \
  --checkpoint-root /mnt/data/models/wyt/checkpoints/pi05_agilex_breakfast_temporal_completion_history_carry_posweight2_head/history_carry_seed42_single_gpu_posweight2 \
  --output /mnt/data/models/wyt/evaluations/temporal_completion_reports/history_carry_seed42_single_gpu_posweight2.json
```

## 10. History-carry 的 32/16/12/4 匹配消融

该消融复用已有 history-carry cache，不需要重新提取 prefix。它保留原方案的
32 条 positive 和16条 hard negative，只把4条 ordinary negative替换成4条
transition negative：

```text
positive=32, hard=16, ordinary=12, transition=4
batch_size=64, num_train_steps=2000, pos_weight=1.0
```

训练：

```bash
uv run scripts/train.py \
  pi05_agilex_breakfast_temporal_completion_history_carry_32_16_12_4_head \
  --exp-name history_carry_32_16_12_4_seed42_single_gpu
```

评估：

```bash
uv run scripts/evaluate_temporal_completion.py \
  --config-name pi05_agilex_breakfast_temporal_completion_history_carry_32_16_12_4_head \
  --checkpoint-root /mnt/data/models/wyt/checkpoints/pi05_agilex_breakfast_temporal_completion_history_carry_32_16_12_4_head/history_carry_32_16_12_4_seed42_single_gpu \
  --output /mnt/data/models/wyt/evaluations/temporal_completion_reports/history_carry_32_16_12_4_seed42_single_gpu.json
```
