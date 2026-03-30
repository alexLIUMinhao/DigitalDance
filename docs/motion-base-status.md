# Motion Base 分支状态报告

更新时间：2026-03-29

仓库：`alexLIUMinhao/DigitalDance`

当前分支：`codex/motion_base`

## 1. 背景与目标

`Motion Base` 这一条分支的目标，是在不影响当前运行时舞蹈主路径的前提下，先搭建一套独立的动作库工作流，用于后续接入真实外部动捕或视频转动作素材。

这套工作流当前强调的是：

- 将动作库数据、审核记录、预览导出从主运行时路径隔离
- 在仓库内保留元数据、审核结果和预览清单
- 通过本地忽略配置连接仓库外的 FBX / 视频素材根目录
- 在 Unity 中提供候选动作审核入口，而不是直接改运行时资源

## 2. 这条分支新增了什么

### 2.1 独立的 Motion Base 数据目录

新增 `motion_base/` 目录，包含以下子结构：

- `config/`：资产根目录配置模板
- `intake/`：待接入素材队列
- `library/`：正式动作索引和 review 记录
- `review/`：候选审核 feed 和审核结果
- `validation/`：候选静态验证结果
- `preview/runtime_preview/`：运行时兼容的预览清单和导出报告
- `taxonomy/`：受控词表和默认策略
- `docs/`：审核 SOP

### 2.2 Motion Base 工具链脚本

新增 `tools/motion_base/`，覆盖这套工作流的主要步骤：

- `sync_intake_queue.py`
- `precheck_sources.py`
- `register_rokoko_export.py`
- `retarget_and_slice.py`
- `validate_fbx_candidates.py`
- `build_review_queue.py`
- `ingest_motion.py`
- `validate_motion_base.py`
- `export_runtime_preview.py`

这些脚本现在已经能支撑“发现素材 -> 预检 -> 生成候选 -> 审核 -> 写入库 -> 导出预览”的离线流程骨架。

### 2.3 Unity 编辑器审核入口

新增 `Assets/Scripts/MotionBase/Editor/`：

- `MotionBase.Editor.asmdef`
- `MotionBaseReviewModels.cs`
- `MotionBaseReviewWindow.cs`

Unity 中新增菜单：

- `Tools > Motion Base > Review Queue`

当前定位是编辑器内候选动作审核面板，而不是运行时播放功能。

### 2.4 本地配置与缓存隔离

`.gitignore` 新增了以下忽略规则：

- `motion_base/config/asset_roots.local.json`
- `Assets/MotionBaseReviewCache/`

这保证了本地素材根目录配置和 Unity 审核缓存不会污染仓库。

## 3. 当前进展

### 3.1 已完成并验证的部分

已确认以下能力可用：

- `motion_base` 目录结构、taxonomy、preview 导出物和工具脚本均已落地
- Unity 可以成功编译 `MotionBase.Editor` 程序集
- Unity PlayMode 自动化测试已恢复到可批处理执行状态
- 当前分支已经具备提交与推送到远端的技术条件

### 3.2 当前 Motion Base 库的实际状态

当前 `motion_base/library/` 中的 48 条动作和 48 条 review 记录，主要是 seed 数据，不是已经完成真实外部素材审核后的正式库。

当前可明确看到的状态是：

- `motion_base/library/motion_index.json` 的 `batchId` 为 `motion_base_seed_v1`
- `motion_base/review/review_feed.json` 当前为空
- `motion_base/review/candidate_review.json` 当前为空
- `motion_base/validation/candidate_metrics.json` 当前为空

这说明：

- 动作库骨架和示例数据已经建立
- 真实候选素材的 intake / review / ingest 全链路还没有完成

## 4. 本轮测试结果

本轮按“聚焦 Motion Base”执行了以下验证。

### 4.1 运行时数据校验

执行：

```bash
python3 tools/dance_pipeline/validate_runtime_inputs.py --song-id audio_mp3
python3 tools/dance_pipeline/validate_runtime_inputs.py --song-id audio1_mp3
```

结果：

- `audio_mp3`：通过
- `audio1_mp3`：通过

说明：

- `song_catalog.json`
- `Songs/audio_mp3.json`
- `Songs/audio1_mp3.json`
- `motion_manifest.json`

在当前仓库内保持一致，运行时基础数据没有被这条分支破坏。

### 4.2 Motion Base 数据层校验

执行：

```bash
python3 tools/motion_base/validate_motion_base.py
python3 tools/motion_base/export_runtime_preview.py
```

结果：

- `validate_motion_base.py`：通过
- `export_runtime_preview.py`：通过，导出 `48` 个 preview clips

注意：

- 当前校验仍然提示未配置本地 asset root
- 因此外部 FBX 存在性检查被跳过

### 4.3 Unity batch compile

执行：

```bash
/Applications/Unity/Unity.app/Contents/MacOS/Unity -batchmode -quit -nographics -projectPath "/Users/alex/Desktop/codex project/3d-digital/3d-digital-human" -logFile /tmp/motion_base_unity_compile.log
```

结果：

- 通过
- `MotionBase.Editor.dll` 已成功编译生成

### 4.4 EditMode 自动化测试

执行：

```bash
/Applications/Unity/Unity.app/Contents/MacOS/Unity -batchmode -nographics -projectPath "/Users/alex/Desktop/codex project/3d-digital/3d-digital-human" -runTests -testPlatform editmode -assemblyNames "DanceDemo.EditorTests" -testResults /tmp/motion_base_editmode_results.xml -logFile /tmp/motion_base_editmode.log
```

结果：

- `8 / 8` 通过

### 4.5 PlayMode 自动化测试

执行：

```bash
/Applications/Unity/Unity.app/Contents/MacOS/Unity -batchmode -nographics -projectPath "/Users/alex/Desktop/codex project/3d-digital/3d-digital-human" -runTests -testPlatform playmode -assemblyNames "DanceDemo.PlayModeTests" -testResults /tmp/motion_base_playmode_results.xml -logFile /tmp/motion_base_playmode.log
```

结果：

- `3 / 3` 通过

本轮还顺手修复了一个测试稳定性问题：

- PlayMode tests 的 `SetUp` 会删除已有 `DanceDemoBootstrap`
- 但命令行 PlayMode 场景切换不会自动再次触发 bootstrap 的运行时创建
- 现已在测试里补上显式重建逻辑，使批处理模式下的 PlayMode 测试可以稳定通过

## 5. 未完成项与阻塞

当前还不能把 `Motion Base` 定义为“全链路调试完成”，主要原因有三点：

- 没有 `motion_base/config/asset_roots.local.json`
- 本机未发现与 `approvedFbx/rawFbx/extractedMotion/sourceVideo` 对应的真实素材根目录
- `review_feed / candidate_review / candidate_metrics` 仍为空，说明真实候选素材还没有经过 intake、预检、审核和 ingest

因此，当前更准确的表述是：

- 编辑器工具已能编译
- 数据骨架与 seed library 已就绪
- 自动化验证已通过
- 真实外部素材驱动的 Motion Base 工作流尚未完成联调

## 6. 建议下一步

建议按下面顺序继续：

1. 创建 `motion_base/config/asset_roots.local.json`
2. 接上真实 `rawFbx / sourceVideo / extractedMotion / approvedFbx / previewCache` 路径
3. 跑一遍 `sync_intake_queue.py -> precheck_sources.py -> retarget_and_slice.py -> validate_fbx_candidates.py -> build_review_queue.py`
4. 在 Unity 中通过 `Tools > Motion Base > Review Queue` 做真实候选审核
5. 执行 `ingest_motion.py --sync-approved`
6. 在真实外部素材接通后，重新执行严格校验并更新这份报告

## 7. 当前结论

截至 2026-03-29，`codex/motion_base` 已经是一个可提交、可推送、可继续协作的分支状态。

它已经完成了：

- Motion Base 工作流骨架
- Unity 审核窗口
- seed 数据库
- 预览导出能力
- 自动化测试与编译验证

但它还没有完成：

- 真实外部素材根目录接入
- 真实候选动作审核链路
- 严格素材存在性校验
- 基于真实资产的最终入库闭环
