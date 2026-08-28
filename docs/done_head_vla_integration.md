# Token-query Done Head：VLA 集成与复现

## 1. 结论

当前基线为 **h768 token-query attention，step 1400**：

| 项目 | 配置 |
|---|---|
| Config | `pi05_agilex_breakfast_token_query_completion_head_h768` |
| Head | hidden dim 768、32 learned queries、12 attention heads、3 transformer layers |
| 参数量 | 30,350,849（约0.030B） |
| 训练 | 冻结VLA，只训练done head；batch size 48，2 epochs，确定性GPU算子 |
| 推理 | 2Hz，sigmoid阈值0.5 |

最终head-only参数：

```text
/home/geek/share3/vla_done/v2/done_head_h768_deterministic_full_seed42_20260828/
checkpoints/step_001400/params
```

## 2. 模型结构

每个时刻使用当前子任务原有指令和单帧观测运行冻结的Pi0.5 prefix forward：

```text
三路相机图像 + 当前任务指令
    │
    ├─ SigLIP视觉编码
    ├─ 文本token embedding
    └─ PaliGemma联合上下文化
             ↓
prefix tokens [968, 2048] + valid mask [968]
```

Done head维护同一子任务内三个时刻的特征：

```text
[t-1.0s, t-0.5s, t]
tokens: [B, 3, 968, 2048]
mask:   [B, 3, 968]
```

Head内部流程：

1. 对2048维prefix token做FP32 LayerNorm。
2. Key/value投影到768维，并加入三个时刻各自的time embedding。
3. 32个learned queries对全部有效prefix token做cross-attention。
4. Query经过3层、12头self-attention transformer。
5. 第一个query经过LayerNorm和线性层，输出一个done logit。

Padding位置由valid mask排除。Head与action head并行，不修改prompt，不更新VLA
encoder，也不更新action head。

VLA本身仍按单帧推理；外部控制器负责保存最近三个时刻的prefix tokens。任务切换后
必须清空history，避免上一任务的视觉上下文进入下一任务。

## 3. 数据来源

训练样本直接由以下两类原始数据生成：

```text
LeRobot轨迹：
/home/geek/share3/breakfest_data/agilex_make_breakfast_330-2

子任务边界：
/home/geek/share3/breakfest_data/rule_split_330-2_action_state_delay05_cut2dist10_overlap10/split
```

边界是根据LeRobot中的action、joint state和gripper state自动检测得到的规则标注，
不是Qwen输出。四个主要边界分别对应：

1. 右臂开始执行取面包动作。
2. 右夹爪释放面包并离开至少10cm。
3. 右臂回位并稳定停止。
4. 左臂回位并稳定停止。

`task_end`表示最后右臂动作完成并稳定后的整条任务结束点。329个原始episode中，
规则质检保留313个、剔除16个。

OpenPI直接从LeRobot轨迹和边界JSON生成训练样本，不读取Qwen manifest。

## 4. 采样与标签

- 原始轨迹：30FPS。
- Done调度频率：2Hz，即每15帧判断一次。
- History：当前帧、前15帧、前30帧，对应`[-1.0s, -0.5s, 0s]`。
- Train划分：282个episode，使用offset 0和7。
- Val划分：31个episode，使用offset 0。
- 训练集额外加入每个真实边界帧上的exact-boundary正样本。
- 固定seed 42，从每段中间15%至85%的普通负样本中删除25%。
- `continue`标签为0，`transition`和`terminal`标签为1。
- Prompt直接使用边界JSON中的当前子任务描述，不增加done专用描述。

| Split | 总数 | 正样本 | 负样本 |
|---|---:|---:|---:|
| Train | 34,925 | 3,013（8.63%） | 31,912（91.37%） |
| Val | 2,512 | 111（4.42%） | 2,401（95.58%） |

## 5. 特征抽取与训练

Token缓存包含52,855个去重的单帧prefix特征，使用FP16保存为4个shard。特征抽取
使用原始VLA checkpoint `/home/geek/share3/vla_done/v2/39999` 中的norm stats。

```text
/home/geek/share3/vla_done/v2/39999/assets/modanqing/
agilex_make_breakfast_generalize_720_subtasks_pickbread600_water400_button300_putbread200/
norm_stats.json
```

Head训练使用BCE loss、AdamW、初始学习率`5e-5`、最终学习率`1e-6`、
weight decay `0.01`、gradient clip `1.0`。训练入口默认启用OpenXLA确定性GPU算子；
同一软件和硬件环境下，相同seed可逐参数复现。单张A800训练1456步约15分钟，
不包含token抽取时间。

验证集开环结果：

| Step | Val loss | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 200 | 0.0822 | 0.9594 | **1.0000** | 0.0811 | 0.1500 |
| 400 | 0.0423 | 0.9809 | 0.7006 | **0.9910** | 0.8209 |
| 600 | 0.0311 | 0.9865 | 0.9529 | 0.7297 | 0.8265 |
| 800 | 0.0361 | 0.9841 | 0.7415 | 0.9820 | 0.8450 |
| 1000 | **0.0214** | **0.9920** | 0.9027 | 0.9189 | **0.9107** |
| 1200 | 0.0249 | 0.9889 | 0.8074 | 0.9820 | 0.8862 |
| 1400 | 0.0222 | 0.9908 | 0.8438 | 0.9730 | 0.9038 |
| 1456 | 0.0238 | 0.9896 | 0.8244 | 0.9730 | 0.8926 |

Step 1000的val loss最低，但最终按半闭环表现选择step 1400。

## 6. 半闭环结果

评测集31个episode中，18个具有有效`task_end`，共评测72个子任务。控制器不再进行
超时强制切换；head未触发时记为`missed/stalled`。

| Checkpoint | 准时 | 提前 | 延迟 | 漏检 | 全准时episode |
|---|---:|---:|---:|---:|---:|
| step 1400 | **60/72** | 11/72 | 1/72 | 0/72 | **9/18** |
| step 1456 | 59/72 | 12/72 | 1/72 | 0/72 | 8/18 |

Step 1400的详细误差：

- 11个提前全部只提前15帧，即0.5秒。
- 唯一延迟只延迟一个2Hz tick，即0.5秒。
- 72/72子任务均自主切换，18/18 episode完成。
- prompt mismatch为0。

最终报告：

```text
/home/geek/share3/vla_done/v2/done_head_h768_deterministic_full_seed42_20260828/
semiclosed_step1400/report.json
```

## 7. 加载方式

Head-only权重通过`completion_head_params`加载到已有VLA，只替换
`completion_head/*`，不会覆盖VLA encoder或action head：

```python
policy = policy_config.create_trained_policy(
    config,
    vla_checkpoint,
    completion_head_params=done_head_params,
)
```

如需生成完整checkpoint，也可使用`scripts/splice_done_head.py`。

## 8. 主要文件

| 功能 | 文件 |
|---|---|
| Token-query head | `src/openpi/models/completion.py` |
| Pi0 prefix token接口 | `src/openpi/models/pi0.py` |
| Policy三帧history | `src/openpi/policies/policy.py` |
| 独立head加载 | `src/openpi/policies/policy_config.py` |
| 边界样本生成 | `src/openpi/training/breakfast_done_data.py` |
| Token抽取 | `scripts/extract_breakfast_done_tokens.py` |
| Head训练 | `scripts/train_token_done_head.py` |
| Head拼接 | `scripts/splice_done_head.py` |
| 半闭环评测 | `scripts/evaluate_breakfast_done_semiclosed.py` |

## 9. 集成检查

- [ ] 使用当前任务原有指令，不增加done专用prompt。
- [ ] 每次VLA forward复用完整prefix tokens和valid mask。
- [ ] 三帧history严格按oldest-to-newest排列，并保持2Hz。
- [ ] 切换子任务时清空history和旧prompt对应的pending action plan。
- [ ] Done head保持FP32，VLA和action head保持原精度与参数。
- [ ] 只加载`completion_head/*`，不覆盖目标VLA参数。
- [ ] 使用step 1400权重和0.5阈值。
