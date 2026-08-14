# 文件分块与 Asset 生成开发方案

## 方案状态

本方案覆盖 Markdown、图片、视频从文件输入到分块，再转换为统一 Asset 的第一阶段链路。
核心数据契约已经确认。PostgreSQL、Markdown、图片和批处理入口已实现；视频视觉资产化等待
参数与 Worker 运行位置确认。

第一阶段包含 Source File、Asset 和处理任务在 PostgreSQL 中的持久化；不包含 Embedding、
Milvus 入库和聚类。视频仅处理视觉信息，不实现 OCR、ASR，也不安装对应依赖。

## 总体链路

```text
输入文件或目录
  → 文件发现与格式校验
  → 建立 Source File 上下文
  → 按格式解析为中间 Chunk
      ├── Markdown → 多个 MarkdownChunk
      ├── Image    → 一个 ImageChunk
      └── Video    → 多个 VideoSegmentChunk
  → Chunk 完整性校验
  → AssetFactory 补齐统一字段
  → PostgreSQL 事务写入 Source File / Asset / Job 状态
  → Asset 列表 / 单文件失败结果
```

解析器只负责“怎样分块及提取客观信息”；`AssetFactory` 统一负责 ID、来源字段、状态、版本和
时间，避免三个解析器分别拼装 Asset 导致字段漂移。

## 分层数据对象

### SourceFileContext

代表本次输入的来源文件，由上游文件发现和 Source File 创建阶段提供：

```json
{
  "workspace_id": "workspace_demo",
  "source_file_id": "src_01J...",
  "absolute_path": "/input/episode-01.mp4",
  "relative_path": "project/episode-01.mp4",
  "file_name": "episode-01.mp4",
  "file_type": ".mp4",
  "mime_type": "video/mp4",
  "file_size_bytes": 102400,
  "folder_context": ["project"]
}
```

### AssetChunk

解析器的中间输出，不包含数据库 ID、revision 和时间戳。公共字段为：

```json
{
  "asset_type": "video_segment",
  "source_locator": {},
  "raw_content": null,
  "file_info": {},
  "derived_file_uri": null,
  "preview_uri": null,
  "model_inputs": {}
}
```

`model_inputs` 只在内存中传递给后续理解模型，不进入最终 Asset。例如 Markdown 的正文、图片
原图、视频关键帧列表。这样可以避免把临时路径和模型张量写进 Asset。

### Asset

`AssetFactory` 将 `SourceFileContext + AssetChunk + 理解结果` 转成统一 Asset：

```json
{
  "asset_id": "asset_01J...",
  "workspace_id": "workspace_demo",
  "source_file_id": "src_01J...",
  "asset_type": "video_segment",
  "file_name": "episode-01.mp4",
  "file_type": ".mp4",
  "asset_name": "雨夜街道中的银发角色",
  "asset_description": "……",
  "asset_features": {},
  "file_info": {},
  "source_locator": {},
  "raw_content": null,
  "derived_file_uri": null,
  "preview_uri": null,
  "processing_status": "completed",
  "feature_revision": 1,
  "embedding_revision": 1,
  "created_at": "2026-07-22T12:00:00+08:00",
  "updated_at": "2026-07-22T12:00:00+08:00"
}
```

`file_name` 始终是原始来源文件名；分块序号和定位信息只放在 `source_locator`，不能把
`file_name` 改成派生片段名。

ID 使用带业务前缀的 ULID，例如 `asset_01J...`、`src_01J...`，当前由 `python-ulid` 生成。

Asset 的结构化创建与模型理解解耦：分块完成后即可创建 Asset，所有字段必须存在，但模型尚未
完成时 `asset_name`、`asset_description` 可以为 null，`asset_features` 为 `{}`。模型开始与
完成时再推进 `processing_status`。理解模型不得阻塞文件分块结果落地。

`asset_name` 支持用户修改，人工名称优先于模型名称。重新理解或重复处理时不得覆盖人工名称；
通过 `asset_name_source` 持久化名称来源；值为 `user` 时自动重处理不得覆盖。

## PostgreSQL 职责

POC 文档已经明确：PostgreSQL 是业务数据的事实来源，保存 Asset 完整信息、Feature、用户修改
记录、Job 状态和模型调用日志；Milvus 只保存向量索引和搜索过滤所需字段。

本阶段至少需要 PostgreSQL 表：

```text
workspaces
source_files
assets
processing_jobs
```

后续理解和向量阶段再接入 `asset_features`、`model_call_logs`、`embedding_records`。是否将
`asset_features` 作为 `assets` 的 JSONB 字段还是独立版本表，需要在 Feature 阶段前结合用户修改
和 revision 需求确认；对外 Asset 字段名称采用远程契约的复数 `asset_features`。

职责边界：

- `source_files` 保存原始文件身份、路径、哈希、文件元数据和总体处理状态；
- `assets` 保存三个解析器生成的可理解最小单元；
- `processing_jobs` 保存导入/分块任务的阶段、总数、完成数、失败数、错误、重试次数和时间；
- Redis 后续只承担长任务分发、租约和可重建的运行时状态，不能成为 Asset 的唯一存储；
- 当前不上传 MinIO，`storage_uri` 保存规范化的本地 `file://` URI，不伪造 S3 URI。

单个 Source File 的 Asset 替换使用一个数据库事务：新分块全部验证成功后，原子替换该来源下
的自动生成 Asset；任何分块失败则回滚，避免旧、新 Asset 混合。人工名称等用户修改数据需要
在替换时按稳定 `asset_key` 映射保留。Source File 身份键为 `(workspace_id, relative_path)`；
`asset_key` 由 Asset 类型和规范化 `source_locator` 生成。同一来源的旧 Asset 在新结果完整验证后
于单个事务中替换。

## 三类文件处理流程

### Markdown

```text
读取 UTF-8 原文
  → Markdown AST / token 节点
  → Heading 层级树
  → 批量 Tokenization
  → 超过 400 tokens 时递归切分
  → 按内容节点安全拆分
  → 相邻短块合并
  → MarkdownChunk 列表
```

- 保留 `heading_path` 和原文 `char_start` / `char_end`。
- 不从句子中间截断，不拆代码块、完整表格和短列表。
- 超长不可拆分节点保留，并在 `file_info` 标记 `oversized` 和原因。
- 每个最终 Block 生成一个 `markdown_block` Asset。
- `asset_name` 根据该 Block 内容生成，不能使用整个文件统一标题替代所有 Block 名称。
- `raw_content` 直接保存该 Block 对应的原始 Markdown 字符串片段，不再嵌套
  `markdown` / `text` 两个字段；模型需要的纯文本只作为临时派生输入，不写入 Asset。
- Tokenization 批量调用、任务内缓存；单个 Markdown 处理失败时记录并返回失败结果。

建议的 `source_locator`：

```json
{
  "type": "text_range",
  "block_index": 0,
  "heading_path": ["第一章", "安装"],
  "char_start": 120,
  "char_end": 860
}
```

### 图片

```text
打开并验证图片
  → 提取尺寸、格式、色彩模式和 EXIF
  → 生成缩略图
  → 一个 ImageChunk
  → 一个 image Asset
```

- 图片不做区域切分，一张图片对应一个 Asset。
- EXIF 不存在的值不推断；原图不修改。
- `source_locator={"type": "whole_file"}`。
- `asset_name` 根据图片主体生成；`file_name` 仍为原始图片名。
- 当前阶段不上传对象存储，`preview_uri` 为空，不能写伪 URI；缩略图是否在本地暂存由图片模块
  开发前另行确认。

### 视频

```text
FFprobe 获取客观元数据
  → PySceneDetect 自然镜头切分
  → 超长 Shot 按时间窗口继续切分
  → 区间内均匀采样候选帧（包含首尾）
  → 无效帧与近重复帧过滤
  → MobileCLIP-S0 批量生成视觉特征
  → 动态聚类并选择最多 3 张代表帧
  → 基于视觉连续性合并相邻候选区间
  → VideoSegmentChunk 列表
```

- 当前只保存原视频上的逻辑时间区间，不立即重编码出物理短视频。
- 每个最终 Segment 生成一个 `video_segment` Asset。
- `asset_name` 和 `asset_description` 只基于 Segment 的代表帧等视觉内容生成。
- 不调用 OCR 或 ASR，不提取或使用视频文字、语音信息。
- 当前阶段不上传对象存储；逻辑片段的 `derived_file_uri` 和 `preview_uri` 均为空。

建议的 `source_locator`：

```json
{
  "type": "time_range",
  "segment_index": 0,
  "start_ms": 42000,
  "end_ms": 72000,
  "source": "scene_then_window",
  "scene_indices": [3],
  "window_indices": [0, 1]
}
```

视频参数在编码前单独评测确认，不在本方案中提前写死。

## 失败与状态规则

- 文件发现层遇到不支持格式时返回跳过结果并记录日志。
- 单个文件解析失败时返回结构化失败结果，不向批任务抛异常，不影响其他文件。
- 同一文件内若一个分块失败，需要明确是整文件失败还是保留成功分块；默认建议原子化处理，
  即该文件不产出半套 Asset，避免重复覆盖时出现旧、新分块混合。
- 不生成虚假的 `derived_file_uri`、`preview_uri`、EXIF、描述或特征。
- `processing_status=completed` 只应表示该 Asset 当前阶段要求的处理均已完成；若模型理解尚未
  执行，应使用 `pending` 或 `processing`，不能提前标记完成。

## 开发顺序

### 1. PostgreSQL 契约与统一 AssetFactory

- 字段统一为 `asset_features`；ID 使用带前缀 ULID；Markdown `raw_content` 为原始 Markdown
  字符串；图片和视频 `raw_content` 为 null。
- 确认状态转换细节和人工名称来源/锁定信息的存储方式。
- 建立 `SourceFileContext`、三种 `AssetChunk` 和完整 Asset Schema。
- 让 AssetFactory 统一补齐公共字段。
- 调整 PostgreSQL `source_files`、`assets`，新增 `processing_jobs` migration 和 Repository。
- 编写 Schema、字段映射、事务回滚、时间、ULID 和 Repository 测试。

### 2. Markdown 端到端

- 重构现有 MarkdownParser 为结构化节点与标题树。
- 接入批量 Tokenization、递归切分、不可拆分节点和相邻合并。
- 从一个 Markdown 文件生成多个完整 Asset。
- 覆盖代码块、表格、长短列表、中文标点、无标题正文和超长原子节点测试。

### 3. 图片端到端

- 补齐 MIME、文件大小、EXIF 时间/软件、文件夹上下文和缩略图。
- 从一张图片生成一个完整 Asset。
- 覆盖无 EXIF、损坏图片、不同色彩模式、Orientation 和极端宽高比测试。

### 4. 视频视觉端到端

- 先用真实长短视频确认场景、窗口、采样和过滤参数。
- 实现逻辑区间、候选帧过滤、MobileCLIP 特征、动态聚类和视觉相邻合并。
- 从一个视频生成多个完整 Asset。
- 覆盖无镜头变化、频繁转场、长镜头、黑场和单个有效帧测试。

### 5. 批处理编排与集成测试

- 将三个 Handler 注册到统一 Assetizer。
- CLI 输入文件或目录，按文件输出成功、失败与 Asset 数量汇总。
- 验证单文件失败不终止批次、结果顺序稳定、重复运行结果可复现。
- 验证 Source File、Asset、processing job 正确落库；同一来源的自动 Asset 原子覆盖，失败时
  不留下半套记录。

## 暂不实施

- 普通文本文件；
- 视频 OCR、ASR 及其依赖；
- HTTP API；
- Embedding、Milvus、聚类结果；
- 未确认前的视频物理切片与转码。

## 已确认的数据契约

1. 最终字段使用远程契约的复数 `asset_features`，现有代码已统一。
2. `asset_id`、`source_file_id` 等业务 ID 使用带前缀 ULID。
3. Asset 创建不等待模型理解；字段必须存在，模型产物在异步阶段回填。
4. 人工修改的 `asset_name` 优先，自动重处理不得覆盖。
5. Markdown 的 `raw_content` 直接保存原始 Markdown 片段字符串；图片和视频为 null。
6. 当前阶段不上传 MinIO，`derived_file_uri` 和 `preview_uri` 允许为 null。

## 长视频任务分发器参考

已阅读 `/Users/gao/work/videoTask` 原型。该原型基于 Redis Streams Consumer Group，已经实现
Worker 心跳、Task attempt fencing、FFmpeg 真实进度监测、动态超时、硬截止时间、PEL
`XCLAIM` 恢复、延迟重试、DLQ 和幂等收尾，并有 Redis 恢复集成测试。

适合后续复用并按 Capsule 修改的设计原则：

- PostgreSQL 保存业务任务事实，Redis 只保存可重建的分发与运行状态；
- 使用稳定 `task_id` 和 `result_version` 保证重复投递、重复执行可收敛；
- `task_id + attempt + worker_instance_id` 作为执行栅栏，拒绝失效 Worker 的迟到结果；
- Worker 心跳、任务进度超时、延迟重试和任务收尾保持独立；
- 外部进程使用独立进程组，先 `SIGTERM`、超时后 `SIGKILL`，最终 `wait()` 回收；
- 进度必须来自实际 frame/time/stage 前进，不能仅靠 Worker 仍有心跳判断任务健康；
- 长任务失败按可恢复、不可恢复分类，不让单个视频阻塞整个批次。

Capsule 中的修改：

- 删除 SQLite 模拟后端和 Java HTTP 回调；Worker 直接通过 Repository 在事务中更新 Capsule
  PostgreSQL 的 `processing_jobs`、`source_files` 和 `assets`；
- Worker 先可靠提交 PostgreSQL 结果，再 `XACK` Redis 消息。若数据库提交成功但 ACK 失败，
  消息重投时按稳定 job ID、stage 和 result version 识别结果已经完成，只补 ACK，不重复生成
  Asset；
- Producer/CLI 先在 PostgreSQL 创建 Job，再投递 Redis。Redis 投递失败时由数据库中的 queued
  Job 补偿投递；
- 固定间隔 FFmpeg 抽帧要替换为场景切分、时间窗口、无效帧过滤和 MobileCLIP 聚类；
- Capsule 当前文件到 Asset 流程先保持本地同步/批处理边界，长视频分发器在视频算法稳定后再接；
- 当前不把 Redis 和 Scheduler 代码提前塞入三个 Parser，Parser 只返回 Chunk。
