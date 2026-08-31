# Progress+Done Head：联合训练与评测记录

## 1. 结论

当前联合预测基线为 **h768 token-query attention，step 1400**。VLA backbone 与 action head
保持冻结，只训练独立的 Progress+Done Head。

| 项目 | 配置 |
|---|---|
| Backbone | Pi0.5 checkpoint `39999` |
| Head | hidden dim 768、32 learned queries、12 attention heads、3 transformer layers |
| 参数量 | 30,351,618（约 0.030B） |
| 输入 | 三个时刻的完整 VLM prefix tokens |
| 输出 | 一个 done logit、一个 progress logit |
| 训练 | batch size 48、2 epochs、seed 42、确定性 GPU 算子 |
| 推理 | 2Hz；done sigmoid 阈值 0.5；progress sigmoid 映射到 `[0,1]` |

最佳 head-only 参数：

```text
/home/geek/share3/vla_done/v2/progress_done_head_h768_seed42_20260831/
checkpoints/step_001400/params
```

`best/` 当前也是 step 1400。

## 2. 模型结构

每个时刻仍使用当前子任务原有指令和单帧观测运行冻结的 Pi0.5 prefix forward：

```text
三路相机图像 + 当前任务指令
    │
    ├─ SigLIP 视觉编码
    ├─ 文本 token embedding
    └─ PaliGemma 联合上下文化
             ↓
prefix tokens [968, 2048] + valid mask [968]
```

图像与文本起初分别编码，但 Head 使用的是经过 PaliGemma 全注意力融合后的
`prefix_out`，所以 prefix tokens 与 prompt 有关。

每个训练样本使用三个时刻：

```text
[t-1.0s, t-0.5s, t]
tokens: [B, 3, 968, 2048]
mask:   [B, 3, 968]
```

Head 内部流程：

1. 对 2048 维 prefix tokens 做 FP32 LayerNorm。
2. Key/value 投影到 768 维，并加入三个时刻的 time embedding。
3. 32 个 learned queries 对所有有效 token 做 cross-attention。
4. Queries 经过 3 层、12 头 self-attention transformer。
5. 第一个 query 经过 LayerNorm，分别送入 Done 和 Progress 两个线性输出层。

两个输出共享全部 token-query 特征，仅最后一层独立。联合 Head 比 Done-only Head
多 769 个参数。Head 与 action head 并行，不修改 prompt，也不更新 VLA encoder 或
action head。

## 3. 标签与数据

数据来源与 Done-only 基线一致：

```text
LeRobot 轨迹：
/home/geek/share3/breakfest_data/agilex_make_breakfast_330-2

子任务边界：
/home/geek/share3/breakfest_data/rule_split_330-2_action_state_delay05_cut2dist10_overlap10/split
```

Done 标签：

- `continue` 为 0。
- `transition` 和 `terminal` 为 1。

Progress 标签根据当前子任务的原始边界线性计算：

```text
progress = clip((query_frame - task_start) / (task_end - task_start), 0, 1)
```

最后一个子任务使用 `task_end`；缺少有效 `task_end` 的 Progress 样本不参与 Progress
loss。Progress 是离散 query 点上的连续监督，不要求预测值严格单调。

每个 query 独立构造历史：

```text
[max(0, q-30), max(0, q-15), q]
```

三帧全部使用该样本当前任务的 prompt 编码。Token cache 由 OpenPI 从 LeRobot、边界
标注和 VLA checkpoint 生成；联合训练直接读取 cache，不读取 Qwen 训练 manifest。

| Split | 总数 | Done 正样本 | Done 负样本 |
|---|---:|---:|---:|
| Train | 34,925 | 3,013（8.63%） | 31,912（91.37%） |
| Val | 2,512 | 111（4.42%） | 2,401（95.58%） |

## 4. 训练配置

```text
epochs                 2
batch_size             48
eval_batch_size        16
learning_rate          5e-5
min_learning_rate      1e-6
warmup_ratio           0.03
weight_decay           0.01
max_grad_norm          1.0
dropout_rate           0.1
progress_loss_weight   1.0
progress_huber_delta   0.1
seed                    42
```

总损失：

```text
loss = done BCE + normalized progress Huber loss
```

VLA prefix tokens以FP16缓存，Head参数和计算保持FP32。训练默认启用OpenXLA确定性GPU
算子。

训练入口：

```bash
uv run python scripts/train_token_progress_head.py
```

## 5. 完整开环结果

| Step | Val loss | Done Acc. | Done P | Done R | Done F1 | Progress MAE | Progress RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 0.1441 | 0.9737 | 0.6596 | 0.8378 | 0.7381 | 0.1264 | 0.1747 |
| 400 | 0.1317 | 0.9689 | 0.5882 | 0.9910 | 0.7383 | 0.0961 | 0.1295 |
| 600 | 0.0836 | 0.9877 | 0.8226 | 0.9189 | 0.8681 | 0.0924 | 0.1256 |
| 800 | 0.1088 | 0.9757 | 0.6471 | 0.9910 | 0.7829 | 0.0912 | 0.1238 |
| 1000 | 0.0503 | **0.9908** | **0.9151** | 0.8739 | 0.8940 | 0.0598 | 0.0830 |
| 1200 | 0.0534 | 0.9893 | 0.8088 | **0.9910** | 0.8907 | 0.0585 | 0.0811 |
| 1400 | **0.0491** | 0.9904 | 0.8372 | 0.9730 | **0.9000** | **0.0570** | **0.0792** |
| 1456 | 0.0532 | 0.9893 | 0.8088 | **0.9910** | 0.8907 | 0.0592 | 0.0819 |

Step 1400 的联合验证损失最低，同时具有最佳 Done F1 和最低 Progress MAE/RMSE，
因此选为最终 checkpoint。

## 6. 半闭环协议

正式报告采用训练对齐协议：

- 每个子任务从验证集中已有的首个 `query_frame` 开始。
- 首次推理重新编码 `[q-30, q-15, q]`，三帧使用当前任务 prompt。
- 后续推理按2Hz滚动历史。
- 到达终点仍未触发时重复终点帧，最多等待2秒。
- Done决定任务切换；Progress仅记录，不参与控制。
- 提前或延迟只计入当前任务；下一任务重新对齐其首个query，因此误差不跨任务传播。

这是一种按任务重新对齐的半闭环协议，不等价于连续物理时间轴上的完整闭环。

## 7. Step 1400 半闭环结果

18个具有有效`task_end`的验证episode，共72个子任务：

| 指标 | 结果 |
|---|---:|
| 准时 | 59/72 |
| 提前 | 12/72 |
| 延迟 | 1/72 |
| 漏检 | 0/72 |
| 全自主完成 | 18/18 episode |
| 四个任务全部准时 | 9/18 episode |

12个提前均为15帧，即0.5秒；唯一延迟也是0.5秒。确定性配置下两次完整运行的
报告字节一致。

Progress结果：

| 指标 | 结果 |
|---|---:|
| 评分点数 | 1,463 |
| MAE | 0.05875 |
| RMSE | 0.07955 |
| Spearman | 0.96272 |
| 前段 MAE | 0.06401 |
| 中段 MAE | 0.06540 |
| 后段 MAE | 0.04780 |
| 相邻点回撤率 | 11.21% |
| Done触发时平均Progress | 0.99419 |

正式报告：

```text
/home/geek/share3/vla_done/v2/progress_done_head_h768_seed42_20260831/
semiclosed_step1400_training_aligned/report.json
```

运行入口：

```bash
uv run python scripts/evaluate_breakfast_progress_done_semiclosed.py
```

## 8. 错误 Case 可视化

可视化将Done和Progress画在同一张简图中，仅保留以下片段：

- Done存在误报或漏报。
- Progress最大绝对误差超过阈值。
- Progress出现超过阈值的回撤。

每个问题片段保存所有推理点的三路相机图像；页面一次展示一个推理点，并支持在片段和
推理点之间翻页。最终生成内嵌图像的单文件HTML索引。

```bash
uv run python scripts/visualize_breakfast_progress_done.py
```

当前输出：

```text
/home/geek/share3/vla_done/v2/progress_done_head_h768_seed42_20260831/
openloop_error_cases_step1400/index.html
```

## 9. 当前集成状态

联合权重仍以head-only checkpoint保存，参数位于`completion_head/*`。评测脚本会创建
`TokenQueryProgressDoneHead`并单独恢复该参数树，原始VLA checkpoint保持不变。

当前联合Head尚未接入`policy_config.create_trained_policy`的常规部署路径，也不能直接
按Done-only Head配置加载；正式部署前需要让模型配置显式创建联合Head并暴露两个输出。

## 10. 主要文件

| 功能 | 文件 |
|---|---|
| 联合Head | `src/openpi/models/progress_done.py` |
| 联合训练 | `scripts/train_token_progress_head.py` |
| 联合半闭环评测 | `scripts/evaluate_breakfast_progress_done_semiclosed.py` |
| 错误Case可视化 | `scripts/visualize_breakfast_progress_done.py` |
| 三帧控制器 | `src/openpi/training/temporal_completion_semiclosed.py` |
| 通用半闭环回放 | `scripts/evaluate_temporal_completion_semiclosed.py` |
| 数据与边界标签 | `src/openpi/training/breakfast_done_data.py` |
| Prefix token抽取 | `scripts/extract_breakfast_done_tokens.py` |

## 11. 使用检查

- [ ] 使用当前任务原有指令，不增加Progress或Done专用prompt。
- [ ] 三帧history严格按oldest-to-newest排列，并保持2Hz。
- [ ] prompt切换后，用新prompt重新编码该任务首个query的三帧历史。
- [ ] 只用Done输出控制切换，Progress输出用于状态估计和监控。
- [ ] 加载step 1400权重，Done阈值使用0.5。
- [ ] 使用与VLA checkpoint `39999`一致的norm stats。
- [ ] 区分当前按任务重对齐的半闭环与连续时间轴全闭环。
