# Pi0.5 三帧 Completion Head 真机集成部署

本文档说明新增的真机推理部署模块：在保留 Training-Paper RTC 动作推理与机械臂控制逻辑的基础上，复用同一次 Pi0.5 VLM prefix forward 得到的原始 token，使用三帧 Transformer completion head 判断当前子任务是否完成，并自动切换下一条 prompt。

## 1. 代码来源与分支关系

本部署实现以 GitHub `ljc/completion_head_frozen` 为训练和模型代码基线。三帧 raw-token Transformer head、输入结构和权重格式均直接沿用该分支，不另建训练实现。

真机部署逻辑先在服务器隔离仓库 `/home/geekplus/develop/ra_ttrtc/openpi_completion_integrated` 的 `completion-integrated` 分支完成验证，再以新增部署文件和一个可选 Policy 接口的形式整理回当前分支。原 `ljc/completion_head_frozen` 分支保持不动。

## 2. 实现目标

部署需要同时满足：

1. 动作继续使用原 Training-Paper RTC 异步 action chunk 推理与执行逻辑。
2. completion head 使用未池化的原始 prefix token，而不是 2048 维均值池化特征。
3. action 和 completion 共享同一次 VLA prefix forward，不额外执行一次 VLM。
4. 原始 prefix 只保留在策略服务端 GPU 上，不经网络发送给客户端。
5. completion 达到阈值后自动切换 prompt，并拒绝执行同一响应中属于旧 prompt 的动作。
6. 保存完整事件日志、顶部相机录像和可交互 HTML，便于回放切换过程。

## 3. 总体架构

```text
顶部/左右相机 + 机械臂状态
          │
          ▼
真机客户端（30 Hz 控制循环）
          │ 异步 action request：观测、prompt、generation、时间戳
          ▼
集成策略服务（8001）
          │
          ├─ Pi0.5 prefix forward（只计算一次）
          │      ├─ KV cache → flow action expert → action chunk
          │      └─ raw prefix tokens → 服务端时间历史
          │                              └─ 选 t-1.0、t-0.5、t
          │                                  → completion Transformer
          ▼
联合响应：actions + completion
          │
          ▼
客户端 generation 检查与 prompt 状态机
          ├─ score < threshold：安装并执行 action chunk
          └─ score ≥ threshold：丢弃旧 action，切换 prompt，重新请求
```

真机控制服务监听 `127.0.0.1:9901`，集成策略服务监听 `127.0.0.1:8001`。

## 4. Completion 输入与时间历史

每个时刻的原始 prefix 包含：

```text
prefix_out：约 [968, 2048]
prefix_mask：约 [968]
```

completion head 的一次有效输入为：

```text
[t-1.0 s, t-0.5 s, t] × [968 tokens, 2048 dims]
```

这里的三帧不是由一个独立的 2 Hz completion 请求产生。服务端只在 action inference 时得到 prefix，并按 observation 的单调时间戳缓存最近 2 秒的数据。对当前 prefix，服务端寻找：

- 最接近 `t-1.0 s` 的历史 prefix；
- 最接近 `t-0.5 s` 的历史 prefix；
- 当前 `t` 的 prefix。

前两帧与目标时间的误差都必须不超过 `0.2 s`，而且三帧时间严格递增。满足后 `history_ready=true`，否则本次只返回动作，不进行有效 completion 判定。

当前策略服务的历史有两个边界：

- 历史仅保存在内存/GPU 中，不写 feature cache。
- `prompt_generation` 改变时历史会清空。因此每次 prompt 切换后，需要重新积累大约 1 秒历史，completion 才会再次 ready。

## 5. Completion Head

部署加载的是冻结的 `token_query_attention` 三帧 Transformer head。当前固定结构为：

```text
temporal_steps = 3
token_count = 968
feature_dim = 2048
hidden_dim = 768
query_count = 32
attention_heads = 12
temporal_layers = 3
dropout_rate = 0.1（推理时关闭）
history_times_seconds = [-1.0, -0.5, 0.0]
```

head 输出一个 logit，经 sigmoid 得到 `[0, 1]` completion score。它只负责状态判断，不参与 action flow matching，也不修改 VLA 权重。

## 6. 客户端调度逻辑

客户端只有一个异步 action planner。每个请求携带：

- 当前 prompt；
- `task_index`；
- `prompt_generation`；
- observation 单调时间戳；
- RTC 所需的剩余动作前缀和 delay 信息。

联合响应到达后按以下顺序处理：

1. 读取响应中的 request generation 和 server generation。
2. 如果 generation 已过期，丢弃整个响应。
3. 更新最新 completion score 和 history 状态。
4. 当 `history_ready=true` 且 `score >= threshold` 时，丢弃同一响应中的 action chunk。
5. 切换至下一条 prompt，增加 generation，使所有旧请求失效。
6. 清除当前旧 action chunk，并立即为新 prompt 发起 action request。
7. 如果没有触发完成，才安装并执行响应里的 action chunk。

这样可以避免模型已经判定 task 0 完成，却继续执行同一响应中 task 0 的动作。

当前自动切换条件是单次有效 score 达到阈值，没有额外连续帧确认、迟滞或 timeout 自动切换。若模型没有触发，可按 `n` 手动切换下一条 prompt；不需要按 Enter。

四个 prompt 的顺序从 completion head 的 `metadata.json` 读取，预期顺序为：

1. `load_bread_into_toaster`
2. `activate_toaster`
3. `pour_drink_into_cup`
4. `place_toasted_bread_on_plate`

客户端启动时只使用 task 0 prompt。完成最后一个 task 后进入 episode complete，保持最后姿态，不再请求下一任务。

## 7. 新增文件

| 文件 | 作用 |
|---|---|
| `deploy_completion/completion_head_runtime.py` | 加载独立 completion 权重，在 GPU 上执行三帧 token-query head |
| `deploy_completion/serve_integrated_completion_policy.py` | 集成策略服务；维护 prefix 时间历史并返回 actions + completion |
| `deploy_completion/integrated_completion_online_inference_execution.py` | 真机客户端、prompt 状态机、generation 管理、日志和录像 |
| `deploy_completion/integrated_completion_online_inference.yaml` | 客户端默认参数 |
| `deploy_completion/start_integrated_completion_policy_server.sh` | 启动集成策略服务 |
| `deploy_completion/start_integrated_completion_client.sh` | 启动集成真机客户端 |
| `deploy_completion/render_integrated_completion_session.py` | 根据 session 日志生成视频/分数同步 HTML |
| `deploy_completion/smoke_test_action_equivalence.py` | 对比原 action 路径与同前向 action 路径 |
| `deploy_completion/smoke_test_joint_response.py` | 验证 actions + completion 联合响应和三帧历史 |

为了暴露同一次 action forward 的原始 prefix，并让联合字段通过原 WebSocket 返回，集成仓库还对以下三个既有文件做了小范围接口修改：

| 文件 | 必要修改 |
|---|---|
| `src/openpi/policies/policy.py` | 新增 `infer_with_raw_prefix()`，返回动作以及仍在设备上的 raw prefix/mask |
| `scripts/serve_training_paper_rtc_policy.py` | 同一次 prefix forward 同时返回 KV cache 和 raw prefix；原 action 接口保持不变 |
| `scripts/serve_training_paper_rtc_base.py` | 增加联合推理扩展点，并允许 response/metadata 携带 completion 字段 |

## 8. 默认运行参数

客户端配置位于 `deploy_completion/integrated_completion_online_inference.yaml`。当前主要默认值：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `control_hz` | 30 | 机械臂控制频率 |
| `obs_send_hz` | 3.75 | 兼容配置项；当前请求实际由 action chunk 消耗与 replan 时机驱动，不是独立定时器 |
| `client_chunk_fixed_delay_steps` | 6 | RTC 固定延迟步数 |
| `client_chunk_delay_mode` | `realtime_ceil` | 根据实际推理耗时向上取整 delay |
| `client_chunk_max_delay_steps` | 6 | 客户端允许的最大 delay |
| `completion_threshold` | 0.6 | 自动切换阈值 |
| `completion history tolerance` | 0.2 s | 历史帧与目标时间的最大误差 |
| `completion history window` | 2.0 s | 服务端 prefix 缓存窗口 |

命令行参数会覆盖 YAML 默认值。例如：

```bash
bash deploy_completion/start_integrated_completion_client.sh \
  --completion-threshold 0.7 \
  --record-top-video
```

因此 completion 的实际检测频率等于异步 action response 的到达频率，不保证严格为 `3.75 Hz` 或 `2 Hz`；三帧选择依靠服务端时间戳和 `0.2 s` 容差完成。

## 9. 正确启动顺序

必须依次启动 ROS 基础服务、集成策略服务、集成客户端。`ros_start.sh start` 会清理旧推理链及冲突端口，因此不能先启动策略服务，再补启动 ROS。

### 终端 1：ROS、相机和机械臂控制服务

```bash
cd /home/geekplus/develop/ra_ttrtc/openpi_completion_integrated
OPENPI_DIR="$PWD" bash scripts/deploy/ros_start.sh start
```

等待输出 validation passed，并保持终端运行。显式设置 `OPENPI_DIR` 是为了确保 `9901` 启动的是集成仓库中的新版 ROS-only service，而不是旧仓库服务。

### 终端 2：集成策略服务

```bash
cd /home/geekplus/develop/ra_ttrtc/openpi_completion_integrated
bash deploy_completion/start_integrated_completion_policy_server.sh
```

### 终端 3：集成真机客户端

```bash
cd /home/geekplus/develop/ra_ttrtc/openpi_completion_integrated
bash deploy_completion/start_integrated_completion_client.sh \
  --completion-threshold 0.6 \
  --record-top-video
```

运行期间：

- 自动切换：有效 completion score 达到阈值；
- 人工切换：按 `n`；
- 结束运行：在客户端终端按 `Ctrl+C`。

客户端退出流程会调用 RobotArmService stop，使机械臂回零，并完成 MP4 封装和 HTML 生成。看到 `[session report] .../report.html` 后，再关闭终端。随后可以停止策略服务和 ROS 基础服务。

## 10. 日志、视频和 HTML

按当前 YAML 配置，每次运行创建：

```text
logs/integrated_completion_deploy/session_YYYYMMDD_HHMMSS_microseconds/
├── session.json
├── events.jsonl
├── top_camera.mp4
├── top_camera_frames.jsonl
├── top_camera_ffmpeg.log
└── report.html
```

其中：

- `session.json`：模型、head、阈值、prompt 和运行配置；
- `events.jsonl`：请求、响应、score、history、action 安装/执行/丢弃和切换事件；
- `top_camera.mp4`：H.264、yuv420p、faststart 格式的顶部相机录像；
- `top_camera_frames.jsonl`：视频帧与 control step 的对应关系；
- `report.html`：上方视频、下方 completion score 曲线和切换标记。

HTML 中播放视频时，白色游标会沿分数曲线同步移动；点击曲线可跳转到对应视频位置。`report.html` 使用相对路径引用同目录的 `top_camera.mp4`，复制结果时应保留整个 session 文件夹。

查看最新 session：

```bash
ls -td logs/integrated_completion_deploy/session_* | head -1
```

若一次 session 正常保存了日志但没有生成 HTML，可手动生成：

```bash
/home/geekplus/miniforge3/envs/piper_ros/bin/python \
  deploy_completion/render_integrated_completion_session.py \
  logs/integrated_completion_deploy/session_对应目录
```

## 11. 关键日志事件

| event | 含义 |
|---|---|
| `action_chunk_request_queued` | 已排队一个异步动作请求 |
| `action_chunk_response` | 动作/完成联合响应到达 |
| `completion_result` | completion score、logit、三帧时间和匹配误差 |
| `action_chunk_install` | 未触发完成，安装当前 action chunk |
| `action_chunk_drop_completed` | completion 达阈值，丢弃同响应旧 prompt 动作 |
| `action_chunk_drop_stale` | generation 已变化，丢弃过期响应 |
| `auto_prompt_switch` | completion head 自动切换 prompt |
| `manual_prompt_switch` | 操作者按 `n` 切换 prompt |
| `episode_complete` | 最后一个任务完成 |
| `control_step` | 30 Hz 控制步、当前 task、动作来源和最近 score |

`completion_result` 中最重要的诊断字段：

```text
score
history_ready
history_size
history_not_ready_reason
relative_times
target_time_errors
head_score_ms
```

## 12. 常见报错

### `ConnectionRefusedError`，连接 `127.0.0.1:9901` 失败

ROS-only RobotArmService 未启动。先在终端 1 启动 `scripts/deploy/ros_start.sh start`，等 validation passed 后再启动客户端。

### `ROS-only service does not support 'absolute_joint': []`

客户端连接到了旧仓库的 ROS-only service。旧服务的 ping 协议没有 `supported_action_spaces`。停止旧部署链，然后从集成仓库执行：

```bash
OPENPI_DIR="$PWD" bash scripts/deploy/ros_start.sh start
```

### `8001` 端口冲突

启动顺序错误或旧策略服务仍在运行。先停止旧客户端/策略服务，再按“ROS → 策略服务 → 客户端”的顺序启动。

### 一段时间没有 completion score

查看 `completion_result.history_not_ready_reason`。常见原因是缺少 `t-1.0 s` 或 `t-0.5 s` 附近的 prefix。每次 prompt 切换后历史会清空，前约 1 秒没有有效三帧是当前设计的正常行为。

### HTML 或 MP4 不完整

不要用强制结束进程的方式退出客户端。按 `Ctrl+C`，等待 FFmpeg 关闭并输出 `[session report]`。只有新版本生成的 MP4 是 H.264；旧的 `mp4v` 文件不会自动转换。

## 13. 对原训练代码的影响

当前集成采用最小侵入方式：

- `src/openpi/models/pi0.py` 不修改；
- `src/openpi/models/completion.py` 不修改，部署 head 与训练 head 保持同一结构；
- `src/openpi/training/` 不修改；
- `src/openpi/policies/policy.py` 只增加可选的 `infer_with_raw_prefix()` 部署入口，原 `infer()`、训练和普通 action 推理路径保持不变；
- Training-Paper RTC、ROS adapter、completion 调度、录像和 HTML 均作为新增文件加入。

`scripts/serve_training_paper_rtc_policy.py` 只在该部署服务进程内安装同前向 runtime 方法：一次 prefix forward 同时得到 action 所需 KV cache，以及 completion 所需 raw tokens/mask。没有启动该服务时，不会改变普通训练或推理行为。

因此当前分支可以同时承担原有 completion head 训练和新增真机部署；原 `ljc/completion_head_frozen` 分支无需改动。
