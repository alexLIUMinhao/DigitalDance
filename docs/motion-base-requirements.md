# Motion Base 分支需求、实施规格与人机协作平台说明

更新时间：2026-03-29

仓库：`alexLIUMinhao/DigitalDance`

当前分支：`codex/motion_base`

## 1. 需求背景

当前项目中的舞蹈动作库主要位于 `Assets/Resources/Dance/Clips`。现阶段这套动作库存在两个核心问题：

- 动作数量不够丰富，难以支撑更长时间、更高变化度的音乐驱动编排
- 现有动作的风格表达和过渡质量还不理想，整体观感偏生硬，容易出现重复、突兀和不够贴合音乐的问题

因此，这个分支的目标不只是“多找一些 FBX 文件”，而是围绕“音乐驱动舞蹈编排”建立一套长期可扩展、结构科学、可持续迭代的动作数据库与处理流水线。

## 2. Motion Base 的核心定义

`Motion Base` 的核心单位不是单个 `FBX` 文件，而是可供音乐驱动编舞系统消费的 `dance phrase`。

也就是说：

- 一个原始视频或原始动作资产只是来源素材
- 一个来源素材可以切分成多个候选短语
- 最终入库和运行时编排消费的对象，应当是“可审核、可切片、可语义检索、可稳定衔接”的动作短语

`Motion Base` 的目标应被定义为：

一个面向音乐驱动舞蹈系统的动作短语数据库与处理平台。

## 3. 本分支要解决的核心问题

本分支聚焦解决以下四类问题：

### 3.1 新舞蹈动作从哪里来

当前最关键的问题之一，是新的舞蹈动作到底应该从哪里获取。

从需求角度，本分支接受多种来源路线，包括但不限于：

- AIGC 动作生成
- 基于舞蹈视频的 3D motion 提取 / 重建
- 商业动作库、公开动作库或可合法使用的现成动作资源
- 对现有动作进行切片、重组、再整理得到的新动作短语

但 `v1` 不同时铺开所有路线，而是先锁定一条主线打通闭环。

### 3.2 新的数据结构怎么设计

如果只是简单把更多动作文件放进 `Assets/Resources/Dance/Clips`，动作数量可能会增加，但编排质量不会自动提升。

因此需要构建一套更适合音乐驱动编舞的数据结构，核心要求是：

- 区分来源素材、候选短语、审核结果、正式库动作、运行时导出
- 能承载风格、能量、乐句、角色、过渡等编排语义
- 能持续扩充，而不是每次加动作都重构一遍逻辑

### 3.3 新音乐来了以后，如何更流畅地编排动作

数据库必须帮助系统回答以下问题：

- 这个动作适合哪类音乐段落
- 它属于哪种风格和子风格
- 它的能量等级、节奏范围和切片长度是什么
- 它更适合做主循环、重拍强调、桥接过渡还是停顿收势
- 它与哪些动作属于同类，哪些动作适合轮换，哪些动作不适合连续出现

### 3.4 如何建立一套高效合理的 pipeline

未来动作扩充不能依赖“每次人工试一遍”。本分支需要把以下环节标准化：

- 来源发现
- 来源登记
- 技术预检
- 原始视频审核
- 提取 / 重定向
- 片段切分
- 静态校验
- 候选动作审核
- 正式入库
- 运行时预览导出

## 4. V1 范围与默认假设

本分支当前的 `v1` 目标，不是立即替换正式 runtime，而是先打通一条真实可用的离线动作生产线。

### 4.1 V1 主路线

`v1` 默认主路线固定为：

`dataset native motion -> dataset adapter -> rawFbx/datasets -> candidate phrases -> review -> formal library -> runtime preview`

其中第一优先来源不再是零散 source-video，而是带 `SMPL / SMPLH / SMPL-style joints` 与配套标注的舞蹈数据集。

### 4.2 V1 风格范围

首批试点固定为单风格：

- `styleFamily = contemporary`
- `styleSubstyle = commercial_contemporary`

这样做的原因是：

- 素材更容易获取
- 单人全身镜头更常见
- 视频转动作时遮挡和服装干扰更少
- 更适合先验证 workflow，而不是一开始追求多风格覆盖

### 4.3 当前系统状态

截至当前分支状态：

- runtime 仍然使用旧的 `Assets/StreamingAssets/DanceData/motion_manifest.json`
- `motion_base/library/motion_index.json` 中的 `48` 条记录主要是 seed records
- `Motion Base` 已具备 editor review 和 isolated library 能力，但真实 source-video 主链路尚处于首版落地阶段

因此，本文件既是愿景说明，也是 `v1` 的实施规格。

## 5. 动作来源策略

### 5.1 允许的 V1 来源

`v1` 的第一优先来源固定为公开可下载的舞蹈数据集：

- `AIST++`
- `FineDance`
- `PhantomDance`

后续保留来源：

- `AIOZ-GDANCE`
- `SoulDance`
- 官方许可清晰的 `sourceVideo` 平台，如 [Pexels API](https://www.pexels.com/api/documentation/) 和 [Pixabay API](https://pixabay.com/api/docs/)

额外说明：

- `Mixkit` 只保留为人工补充来源，且仅允许 `Free License`
- `YouTube` 及其他未确认授权的平台，不作为 `v1` 自动抓取来源

### 5.2 来源选择标准

后续选择动作来源时，判断标准不应只看“能不能拿到视频”，而应综合评估：

- 风格质量
- 可切片性
- 过渡质量
- Unity Humanoid / Mixamo 兼容潜力
- 提取和人工处理成本
- 版权与追溯安全性
- 可持续扩充性

## 6. 数据结构要求

`Motion Base` 希望构建的是一个分层数据库，而不是简单文件夹。

推荐逻辑层级固定为：

1. 来源层
2. intake 层
3. 候选层
4. 正式库层
5. 运行时导出层

### 6.1 来源层

来源层记录数据集序列或原始视频候选及其 provenance。

对应文件：

- `motion_base/intake/source_catalog.json`
- `motion_base/intake/dataset_catalog.json`

其中 `dataset_catalog.json` 的核心字段包括：

- `datasetName`
- `sequenceId`
- `motionPath`
- `musicPath`
- `labelPath`
- `fps`
- `sourceFormat`
- `license`
- `styleLabels`
- `convertedMotionRelPath`
- `conversionStatus`

`source_catalog.json` 仍保留给 `sourceVideo` 路线，核心字段包括：

- `sourceId`
- `provider`
- `remoteAssetId`
- `sourcePageUrl`
- `downloadUrl`
- `creatorName`
- `licenseName`
- `query`
- `localVideoRelPath`
- `targetStyleFamily`
- `targetStyleSubstyle`
- `expectedMetaAction`
- `downloadedAtUtc`
- `notes`

### 6.2 Intake 层

intake 层记录“已进入系统、等待预检 / 审核 / 提取”的工作状态。

对应文件：

- `motion_base/intake/intake_queue.json`

`MotionBaseIntakeJob` 除原有字段外，新增两组结构化信息：

- `sourceProvenance`
- `sourceReview`

其中：

- `sourceProvenance` 用于保存 provider、page、license、query、creator 等来源信息
- `sourceReview` 用于保存原始视频的人工质量判断

### 6.3 候选层

候选层保存从原始视频提取并切出来的 candidate phrases。

对应文件：

- `motion_base/review/review_feed.json`
- `motion_base/review/candidate_review.json`
- `motion_base/validation/candidate_metrics.json`

### 6.4 正式库层

正式库层保存后续真正供编排系统使用的动作短语。

对应文件：

- `motion_base/library/motion_index.json`
- `motion_base/library/reviews.json`

该层重点围绕以下字段：

- `styleFamily`
- `styleSubstyle`
- `metaAction`
- `energyBand`
- `preferredSegments`
- `nativeBpm`
- `phraseBeats`
- `entryOffsetsBeats`
- `sliceBeatsOptions`
- `transitionProfile`
- `varietyGroup`
- `role`
- `qualityTier`
- `licenseTier`

### 6.5 运行时导出层

运行时导出层是 formal library 到 runtime 的投影。

当前对应文件：

- `motion_base/preview/runtime_preview/motion_manifest.preview.json`

这一层的目标是：

- 只保留 runtime 真正需要的字段
- 保证读取稳定
- 不把审核噪声和来源细节直接带进 runtime

## 7. 人机协作分工

本分支的 `v1` 明确采用人机协作模式。

### 7.1 系统负责

系统负责：

- 数据集下载与登记
- 数据集原始 motion 到 bridge asset 的转换
- 官方来源搜索
- 原始视频候选下载
- provenance 记录
- intake queue 同步
- 技术 precheck
- candidate 切片和静态校验
- review queue 构建
- formal library ingest
- preview 导出

### 7.2 人工负责

人工负责两层质量闸门：

1. 原始视频审核
2. 候选动作审核

原始视频审核的目标是回答：

- 这条视频是否值得送去提取
- 风格标签和 `metaAction` 是否合理
- 是否存在遮挡、硬切、多人、镜头不稳等问题

候选动作审核的目标是回答：

- 提取后的动作是否保留了预期风格
- loop seam 是否可接受
- foot stability 是否可接受
- tempo tolerance 是否可接受
- 是否值得进入正式库

## 8. 平台形态

`v1` 平台载体固定为 Unity Editor，不另起 Web 系统。

平台名称定义为：

`Motion Base Studio`

菜单入口：

- `Tools > Motion Base > Studio`
- `Tools > Motion Base > Review Queue` 作为兼容入口，指向同一窗口

### 8.1 Source Intake

`Source Intake` 页负责原始视频审核。

界面分三栏：

- 左栏：原始视频队列
- 中栏：source video 预览
- 右栏：provenance、precheck、source review、style/metaAction 覆写、备注

用户按钮：

- `Approve For Extraction`
- `Hold`
- `Reject`

状态语义固定为：

- `precheck_passed`：待人工审核
- `extraction_pending`：已批准，等待外部动作提取
- `hold / rejected`：本轮不送提取

### 8.2 Candidate Review

`Candidate Review` 页复用现有双预览能力：

- 左侧 source video
- 右侧 avatar preview

这一页继续承担最终入库审核。

## 9. V1 平台工作流

`v1` 的完整 workflow 固定为：

1. 配置 `motion_base/config/asset_roots.local.json`
2. 运行 `tools/motion_base/fetch_source_candidates.py`
3. 生成并更新 `motion_base/intake/source_catalog.json`
4. 运行 `tools/motion_base/sync_intake_queue.py`
5. 运行 `tools/motion_base/precheck_sources.py`
6. 在 Unity `Motion Base Studio -> Source Intake` 审核原始视频
7. 将批准的视频送到 Rokoko 或其他动作提取环节
8. 运行 `tools/motion_base/register_rokoko_export.py`
9. 运行 `tools/motion_base/retarget_and_slice.py`
10. 运行 `tools/motion_base/validate_fbx_candidates.py`
11. 运行 `tools/motion_base/build_review_queue.py`
12. 在 Unity `Motion Base Studio -> Candidate Review` 审核候选动作
13. 运行 `tools/motion_base/ingest_motion.py --sync-approved`
14. 运行 `tools/motion_base/validate_motion_base.py`
15. 运行 `tools/motion_base/export_runtime_preview.py`

## 10. 首批素材规范

首批 `sourceVideo` 的目录结构固定为：

- `sourceVideo/contemporary/commercial_contemporary/basic_step/`
- `sourceVideo/contemporary/commercial_contemporary/travel_step/`
- `sourceVideo/contemporary/commercial_contemporary/turn_phrase/`
- `sourceVideo/contemporary/commercial_contemporary/accent_hit/`
- `sourceVideo/contemporary/commercial_contemporary/pose_hold/`

下载文件命名规范：

- `<provider>_<remoteAssetId>_<slug>.mp4`

首批默认 query：

- `contemporary dance solo`
- `modern dance solo`
- `dancer rehearsal full body`
- `dancer spin solo`
- `dancer pose hold`
- `dancer movement across stage`

视频准入硬规则：

- 单人
- 全身可见
- 固定机位或极轻微移动
- 无硬切
- 低遮挡
- 无大件道具
- 无明显 logo / 品牌 / 水印
- 分辨率至少 `1280x720`
- 优选 `1920x1080`
- 帧率至少 `30fps`
- 优选 `60fps`
- 时长 `5-15s`

## 11. 阶段性成功标准

`v1` 至少达到以下标准，才算完成首轮使命：

- 已形成 `source_catalog -> intake_queue -> review_feed -> formal library -> runtime preview` 的完整链路
- 首批成功收集 `8-12` 条 source videos
- 至少 `6` 条通过 precheck
- 至少 `3` 条经人工批准进入 extraction
- 至少 `2` 条 candidate motion 经人工批准并成功 ingest
- `validate_motion_base.py` 通过
- `export_runtime_preview.py` 产出非空 preview manifest
- 当前正式 runtime 的 `motion_manifest.json` 保持不被替换

## 12. 一句话总结

这个分支的本质，不是“再加一些 FBX”，而是构建一个面向音乐驱动编舞的人机协作动作生产平台：由系统负责找来源、拉素材、建流程，由人工负责做质量闸门，最终把原始视频持续转化成可编排、可审核、可扩展的 dance phrases。
