# M18 Hybrid Streaming SMPL-X Choreography

## 1. 目标

M18 的目标是把 M17 的 `repeat12_total5_center` 视觉基线升级成一套更接近端侧实时场景的通用编舞策略：

- 输入端统一模拟 `2s initial buffer + 2s planner lookahead + 1s transport lookfront`。
- 前 `5s` 不检索舞蹈动作，使用合成的自然直立 idle。
- 正常舞蹈阶段用 FineDance rhythmic-first 中段动作库检索 SMPL-X 动作片段。
- 根据流式可见的音乐状态维护编舞状态机，不做单曲特调。
- 动作速度只做保守节奏适配，优先保证动作稳定和人体自然度。
- 歌曲尾声停止普通检索，进入 recover outro，逐渐回到初始 neutral idle。

M18 仍然只走 FineDance SMPL-X 路线，不引入 Willa，也不处理真实蓝牙/系统音频权限。当前实现是本地歌曲的端侧流式模拟。

## 2. 整体流程

```mermaid
flowchart LR
  A["Local audio stream"] --> B["Streaming event rail"]
  B --> C["Visible music_state"]
  C --> D["Hybrid choreography state machine"]
  D --> E["Style/energy cohort selection"]
  E --> F["FineDance unit retrieval"]
  F --> G["Conservative beat retime"]
  G --> H["SMPL-X stitch manifest"]
  H --> I["Mesh stitch renderer"]
  I --> J["MP4 / HTML review / reports"]
```

端到端分为五层：

1. 流式音乐事件轨：只使用当前播放点前后允许可见的音频窗口，生成 beat、downbeat、drum hit、accent 和音乐状态。
2. 状态机：根据音乐状态和当前动作上下文决定当前编舞意图，例如 groove、accent hit、transition 或 recover outro。
3. 动作检索：在 FineDance rhythmic-first 动作库中先缩小候选歌曲/风格池，再选择具体动作单元。
4. 节奏适配与转场约束：用保守 retime、重复约束、转场分和速度硬门槛过滤候选。
5. SMPL-X 视觉拼接：按计划加载真实 FineDance SMPL-X mesh cache，做 root 连续、recover blend、transition smoothing，并输出 HTML/MP4。

## 3. 输入策略

M18 固定使用以下流式输入契约：

- `initial_buffer_sec = 2.0`
- `lookahead_sec = 2.0`
- `lookfront_sec = 1.0`
- `total_future_sec = 3.0`

含义：

- 播放开始前允许先积累 `2s` 音频。
- 播放中 planner 最多看到播放点之后 `2s` 的规划音频。
- 额外 `1s lookfront` 模拟端侧/蓝牙传输缓存，可以帮助边界决策更稳定。
- planner 的每个 decision 都记录 `future_visibility_guard`，检查 `available_audio_until_sec <= playhead_sec + total_future_sec`。

M18 的验收中，`audio0-audio4` 的 `future_visibility_violations` 都是 `0`。

## 4. 流式音乐事件轨

M18 继续使用 M17 的在线 tempo tracker 和 hysteresis，避免 tempo 在 half-time / double-time 之间频繁跳变。事件轨输出：

- `beats`
- `downbeats`
- `drum_hits`
- `accents`
- `segment_hypotheses`
- `beat_phase`
- `music_state`

`music_state` 是 M18 新增的流式安全音乐状态摘要，只来自当前可见音频窗口：

| 字段 | 含义 |
| --- | --- |
| `energy_level` | `low / mid / high`，由 low band、onset、high attack 和 accent density 估计 |
| `accent_density` | 当前可见窗口内 onset / low / high peak 的密度 |
| `beat_confidence` | 当前 tempo/phase 可信度 |
| `phrase_phase` | 可靠时给出 count / beat phase / downbeat confidence |
| `melodic_motion_proxy` | low-mid 与 high 频段变化量的轻量 proxy，不做完整旋律/和弦分析 |

这个设计的重点是端侧可用：不用离线全曲分析，也不依赖人工标定作为主路径。

## 5. 动作库前提

M18 使用已经构建并标注过的 FineDance rhythmic-first 中段动作库：

- 来源优先为 rhythmic-first 子集。
- 去掉每首源歌前 `10s`，避免 intro 静止/弱动作进入主池。
- 生成 `2 / 4 / 8 / 16 beat` 动作单元。
- 每个 unit 带有检索所需标注：
  - `source_sequence`
  - `duration_beats`
  - `accent_lock_frames`
  - `entry_pose_anchor`
  - `exit_pose_anchor`
  - `root_velocity`
  - `yaw_delta`
  - `energy_curve`
  - `movement_quality`
  - `style_tags`
  - `safe_retime_range`
  - `compatible_next_units`

动作库不是只存 mesh，而是把“动作是否能踩点、能否接得顺、适合什么音乐状态”提前标出来。M18 planner 的质量很大程度依赖这些标注。

## 6. Hybrid Choreography State Machine

M18 新增 `state_machine_policy=hybrid`。状态机不直接生成动作，而是决定“当前应该跳什么意图”，具体动作仍由检索 scorer 选择。

状态集合：

| 状态 | 使用场景 | 对检索的影响 |
| --- | --- | --- |
| `intro_idle` | 前 `5s` 初始等待 | 不检索舞蹈动作，使用 synthetic neutral idle |
| `groove_low` | 低能量/低密度段 | 偏向 smooth、small motion、稳定动作 |
| `groove_mid` | 默认律动段 | 偏向 mid energy、medium motion |
| `groove_high` | 高能量/高密度段 | 偏向 sharp、large motion、高能动作 |
| `accent_prepare` | 可见窗口内有强鼓点/强拍即将到来 | 偏向 accent lock 更丰富的 unit |
| `accent_hit` | 当前边界附近有强 kick/downbeat/accent | 强化 count 1/downbeat 或 backbeat lock |
| `transition` | 弱拍/小节尾且能量或源序列需要切换 | 偏向转场兼容、同风格/同能量候选 |
| `recover_outro` | 歌曲尾声 | 停止普通检索，渐变回 neutral idle |

这种结构的好处是把“音乐语义/编舞意图”和“具体动作检索”拆开：后续扩展到更多音乐风格时，可以加强状态定义，而不必重写整个 planner。

## 7. 动作选择策略

M18 的动作选择仍是两阶段：

### 7.1 先缩池

根据当前 `music_state` 和目标 BPM/能量/风格，从多首 rhythmic-first 源歌中选出候选 cohort。缩池维度包括：

- BPM 接近度
- source song energy
- movement quality
- style tags
- source quality weight
- 上一个动作的 source sequence continuity

这避免每一步都在全库硬搜，也减少跨歌跳切。

### 7.2 再做 unit 级检索

在 cohort 内，对具体 unit 计算综合分：

- `rhythm_lock`: 0.60
- `transition_smoothness`: 0.25
- `style_energy_bpm`: 0.10
- `source_quality_weight`: 0.03
- `diversity`: 0.02

M18 比 M15/M16 更偏向节奏稳定，而不是追求风格变化。它的原则是：

- 节奏优先于多样性。
- 稳定优先于频繁切换。
- 不合适时宁可延续或提前进入 recover，也不强行塞一个过快/过短动作。

## 8. 节奏适配

M18 的节奏适配叫 `speed_retime_policy=conservative_lock`。

核心规则：

- 优先选择能让动作速度落在 `0.90-1.10` 的目标长度。
- 非尾段硬约束避免超过 `0.85-1.15`。
- 如果自动 beat rail 过密，M18 不再盲目把动作压短，而是修正 target spacing，让动作回到自然速度。
- 高置信可见 lock 使用约 `2 frames` 的严格误差窗口评分。

这解决了之前用户观察到的关键问题：当 beat tracker 把节拍估得太密时，旧 planner 会让动作速度暴冲，视觉上就像“抽筋”和“不踩点”。M18 的做法是把“节拍密度错误”从动作速度里隔离出来，优先保证动作自然，再在可信强拍上锁点。

## 9. 重复控制

M18 继承 M17 的 exact unit diversity 规则：

- 相同动作连续出现 `1-2` 次最佳。
- 连续相同动作最多不超过 `3` 次。
- 同一个 exact `unit_id` 整首歌使用不超过 `5` 次。

报告字段：

- `max_consecutive_motion_unit_run`
- `max_total_motion_unit_uses`
- `repeat_unit_hard_reject_count`
- `repeat_unit_preferred_reject_count`
- `repeat_unit_total_hard_reject_count`

当前 M18 五首测试中，`max_consecutive_motion_unit_run <= 2`，`max_total_motion_unit_uses <= 5`。

## 10. 转场与拼接

动作边界主要使用以下信息控制：

- entry/exit pose anchor
- root planar velocity
- yaw delta
- foot contact overlap proxy
- compatible next units
- same sequence / cross sequence stability

渲染阶段做：

- pelvis/root XYZ 连续对齐；
- transition window 内 mesh/joint smoothing；
- recover outro 时从上一段 danced mesh 长窗口 blend 到 neutral mesh；
- final neutral hold 使用 synthetic `smplx_neutral_idle`。

当前实现仍是视觉验证优先，转场 smoothing 主要在 mesh/joint 层完成。下一阶段如果进入 runtime-grade，需要升级到 pose-space root trajectory blend、quaternion blend 和 contact-aware foot locking。

## 11. 开头与结尾策略

### 开头

前 `5s` 固定 synthetic neutral idle：

- `pose_source = smplx_neutral_idle`
- `switch_reason.mode = initial_upright_hold`
- 不复用第一段舞蹈动作首帧
- 第一段舞蹈从 `5s` 后可信 beat/downbeat 边界开始

### 结尾

M18 不再让普通舞蹈动作硬撑到最后一帧。尾声策略：

- 到 `duration_sec - ending_hold_sec` 附近，尽量对齐可信边界；
- 停止普通 FineDance unit 检索；
- 插入 synthetic `smplx_neutral_recover`；
- 从上一段 danced mesh 渐变回 neutral；
- 最后使用 `smplx_neutral_idle` hold。

注意：有些歌曲的 recover window 会长于 `5s`。这是刻意的：如果剩余空间不足以放入自然速度的舞蹈动作，M18 宁可提前 recover，也不压缩动作造成过快/抽动。

## 12. HTML 审查页

M18 HTML review 在 M17 基础上新增：

- state-colored timeline bands；
- streaming decision 的 `choreography_state`；
- retime policy 和 selected beats；
- source sequence / source frames；
- cohort songs；
- rhythm / transition scores；
- rejected top candidates；
- outro recover marker 和 final neutral status 可从 report/decision 字段追踪。

生成文件：

- `outputs/renders/unity_audio0_m18_state_retime_outro_mesh_review.html`
- `outputs/renders/unity_audio1_m18_state_retime_outro_mesh_review.html`
- `outputs/renders/unity_audio2_m18_state_retime_outro_mesh_review.html`
- `outputs/renders/unity_audio3_m18_state_retime_outro_mesh_review.html`
- `outputs/renders/unity_audio4_m18_state_retime_outro_mesh_review.html`

## 13. 当前 M18 验收结果

| Song | Gap | Future violations | Outro recover | Final pose | Max consecutive run | Max total uses | Max non-tail speed | High-conf lock err | Max temporal vertex delta |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `unity_audio0` | 0 | 0 | 11.81997s | `smplx_neutral_idle` | 2 | 5 | 1.09157 | 0.84 frames | 0.040092 |
| `unity_audio1` | 0 | 0 | 5.00000s | `smplx_neutral_idle` | 2 | 5 | 1.06666 | 0.8001 frames | 0.045980 |
| `unity_audio2` | 0 | 0 | 7.33997s | `smplx_neutral_idle` | 2 | 4 | 1.09157 | 0.48 frames | 0.045042 |
| `unity_audio3` | 0 | 0 | 5.92522s | `smplx_neutral_idle` | 2 | 5 | 1.09157 | 0.8214 frames | 0.044800 |
| `unity_audio4` | 0 | 0 | 10.21401s | `smplx_neutral_idle` | 2 | 5 | 1.09157 | 0.7893 frames | 0.045896 |

核心通过项：

- `gap_count = 0`
- `future_visibility_violations = 0`
- `final_pose_source = smplx_neutral_idle`
- `max_consecutive_motion_unit_run <= 2`
- `max_total_motion_unit_uses <= 5`
- `max_non_tail_speed_scale <= 1.15`
- high-confidence lock error 在 `2 frames` 内
- mesh temporal delta 约 `0.040-0.046`

## 14. 与 M17 的主要差异

| 方向 | M17 | M18 |
| --- | --- | --- |
| 编舞控制 | 主要靠 retrieval scorer | 加入 hybrid state machine |
| 音乐状态 | beat/energy 直接进 scorer | 生成 streaming-safe `music_state` |
| 速度控制 | 强约束但可能被 fallback 突破 | conservative retime + spacing correction |
| 尾声 | 普通动作尽量铺满 | recover outro + final neutral hold |
| HTML 审查 | 来源、节奏、转场 | 新增 state/retime 可解释性 |

## 15. 已知限制与下一步

M18 解决的是“通用稳定可看”的第一版，不是最终 runtime-grade 编舞系统。剩余问题：

- 目前重复控制基于 exact `unit_id`，还没有做视觉相似动作家族去重。
- 状态机的 melodic proxy 还比较轻量，没有真实 melody/chord/section transcription。
- 转场仍以 mesh/joint smoothing 为主，下一步应升级到 pose-space quaternion blend、root trajectory blend 和 foot-contact-aware IK。
- 端侧 runtime bundle 还需要把 Python/SMPL-X 实时依赖替换为预计算压缩 pose/mesh cache。

建议后续 M19 优先做：

1. motion-family diversity，减少不同 `unit_id` 但视觉相似的重复。
2. contact-aware transition，让脚底和 root acceleration 更稳。
3. 更可靠的在线 beat/downbeat tracker，减少需要 retime spacing correction 的场景。
4. Unity runtime bundle contract，把 M18 plan + compressed motion cache 导出为端侧可播放资产。
