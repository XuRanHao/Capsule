# 文件资产化实现记录

## 已实现链路

```text
文件/目录输入
  → Discovery
  → SourceFile 持久化
  → Markdown / Image Parser
  → AssetFactory
  → 单来源事务覆盖 Asset
  → ProcessingJob 汇总
  → CLI JSON 结果
```

- Markdown：标题递归切分、400 token 阈值、批量官方 Tokenization、代码块/表格/短列表不可拆、
  长列表按顶层项拆分、长段落按句子边界拆分、相邻短块合并、保留标题路径和原文字符范围。
- 图片：一图一个 Asset，提取尺寸、宽高比、真实 MIME、格式、色彩模式、文件大小、JSON 安全
  EXIF、拍摄时间和软件信息；缺失 EXIF 不补全。
- 视频：本段记录最初的实体媒体实现。当前默认模式仍进行内容切分和代表画面选择，但只保存原视频
  区间与代表时间戳，不再生成分段 MP4、封面或关键帧文件；资产仍正常进入理解、向量和聚类。只有
  显式切换到旧实体模式时，才沿用下文的对象存储写入方式。
- Asset：使用带前缀 ULID；`asset_features` 采用远程数据契约的复数字段；Markdown `raw_content` 保存原始片段；
  模型理解不阻塞入库。所有 Asset 均有一级 `file_tree_context: list[str]`，由 `relative_path` 的父
  目录生成；它与 `SourceFile.file_tree_context` 保持同值。图片不再在 `file_info.folder_context`
  重复保存该信息。
- PostgreSQL：Alembic 初始 migration 已执行；Source File、Asset、ProcessingJob 已接 Repository。
- 失败隔离：文件在进入 Parser 前先创建 Source File；单文件失败写入状态和错误，批次继续；
  Job 最终为 `completed`、`partial_failed` 或 `failed`。

## 覆盖与人工数据规则

- Source File 唯一身份为 `(workspace_id, relative_path)`。
- 同内容不同路径是不同 Source File。
- 每个 Asset 的 `asset_key` 由 `asset_type + source_locator` 稳定生成。
- 同来源重跑时先完整解析，再在一个事务中替换 Asset，失败不会混入半套新结果。
- 匹配到相同 `asset_key` 时复用 Asset ID。
- 内容变化会清空旧模型描述/特征并递增 revision。
- `asset_name_source=user` 的人工名称在自动重处理时保留。

当前限制：Markdown 的字符偏移发生大幅变化时，后续块的 `asset_key` 也可能变化，因此只能保留
定位仍相同的人工名称。若产品要求跨大幅编辑保持人工名称，需要另行设计内容指纹与块匹配策略。

## 验证记录

- Markdown 单元测试通过：6 项。
- 完整测试通过：49 passed，2 skipped。
- Ruff 全量检查通过。
- mypy 对 `src/capsule` 全量检查通过。
- PostgreSQL 集成测试覆盖稳定 Asset ID、人工名称保留、内容变化清理模型数据、revision 递增、
  单文件失败继续批次和 Job 状态。
- 视频单元测试覆盖长镜头窗口、候选帧首尾与上限、无效帧过滤、代表帧聚类和真实 MP4 解析；
  集成测试覆盖 `video_segment` Asset 入库、MP4/JPEG 派生文件、`file_info.keyframes` 及 MPS 缺失时
  同批图片继续成功。
- 宿主 MPS 真实导入已验证：3 秒视频生成 1 个 `video_segment`、1 个 H.264 MP4 和预览/关键帧 JPEG，
  MinIO 对象已读回校验文件头后清理。
- 完整检查：62 passed、2 skipped；Ruff 与 mypy 均通过。
- `20260728_0004` 迁移已在空的 `capsule_assetization_verify` 测试库完成从零验证：已有 Asset
  通过关联 SourceFile 回填 `file_tree_context`，新建和覆盖 Asset 也会写入该字段。

## 豆包 Embedding 已验证兼容项

- 方舟当前响应的 `data` 可能是对象而非 OpenAI 风格列表；客户端已同时兼容两种响应形状。
- `doubao-embedding-vision-250615` 默认可返回 2048 维；客户端必须传递配置中的
  `dimensions`，项目当前固定请求 1024 维。
- 宿主环境若设置了 SOCKS 代理，`httpx` 需要可选 `socksio` 才能使用；当前不添加该依赖，测试时
  临时绕过代理直连。生产运行前需统一代理方案。

## 下一步边界

下一步：

1. 使用正式数据集回调无效帧过滤与代表帧聚类阈值；
2. 设计相邻区间视觉连续性合并阈值并接入最终 Segment 合并；
3. 长视频算法稳定后，再接入独立宿主 Worker 的任务分发与恢复机制。

视频 OCR、ASR 继续搁置，不引入对应依赖。
