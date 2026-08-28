# Token-query Done Head：VLA 集成与复现

## 1. 结论

当前基线为 **h768 token-query attention，step 1400**：

| 项目 | 配置 |
|---|---|
| Config | `pi05_agilex_breakfast_token_query_completion_head_h768` |
| Head | hidden dim 768、32 learned queries、12 attention heads、3 transformer layers |
| 参数量 | 30,350,849（约0.030B） |
| 训练 | 冻结VLA，只训练done head；batch size 48，2 epochs |
| 推理 | 2Hz，sigmoid阈值0.5 |

最终head-only参数：

```text
/home/geek/share3/vla_done/v2/qwen_done_v2/
qwen_style_done_head_v2_token_query_h768/step_1400_loadable/params
```

目录中的`qwen_done_v2`只是历史实验命名；当前代码和模型均不依赖Qwen模型、
Qwen仓库或Qwen训练数据目录。

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
weight decay `0.01`、gradient clip `1.0`。单张A800训练1456步约14分43秒，
不包含token抽取时间。

验证集开环结果：

| Step | Val loss | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 200 | 0.0611 | 0.9713 | 0.6226 | 0.8919 | 0.7333 |
| 400 | 0.0868 | 0.9590 | 0.5187 | 1.0000 | 0.6831 |
| 600 | 0.0277 | 0.9900 | 0.9479 | 0.8198 | 0.8792 |
| 800 | 0.0331 | 0.9857 | 0.7622 | 0.9820 | 0.8583 |
| 1000 | **0.0215** | **0.9912** | **0.9159** | 0.8829 | 0.8991 |
| 1200 | 0.0238 | 0.9893 | 0.8134 | **0.9820** | 0.8898 |
| 1400 | 0.0221 | 0.9904 | 0.8372 | 0.9730 | **0.9000** |
| 1456 | 0.0235 | 0.9893 | 0.8134 | **0.9820** | 0.8898 |

Step 1000的val loss最低，但最终按半闭环表现选择step 1400。

## 6. 半闭环结果

评测集31个episode中，18个具有有效`task_end`，共评测72个子任务。控制器不再进行
超时强制切换；head未触发时记为`missed/stalled`。

| Checkpoint | 准时 | 提前 | 延迟 | 漏检 | 全准时episode |
|---|---:|---:|---:|---:|---:|
| step 1400 | **61/72** | 10/72 | 1/72 | 0/72 | **10/18** |
| step 1456 | 58/72 | 14/72 | 0/72 | 0/72 | 6/18 |

Step 1400的详细误差：

- 10个提前全部只提前15帧，即0.5秒。
- 唯一延迟只延迟一个2Hz tick，即0.5秒。
- 72/72子任务均自主切换，18/18 episode完成。
- prompt mismatch为0。

最终报告：

```text
/home/geek/share3/vla_done/v2/qwen_done_v2/
qwen_done_semiclosed_v2_token_query_h768_step1400/report.json
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
