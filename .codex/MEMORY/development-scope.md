# 第一阶段开发范围

## 已确认范围

- 完整实现：原始文件 → Asset → Description / Feature → Embedding → PostgreSQL + Milvus → 聚类结果。
- 输入格式：Markdown、普通文本、图片、视频。
- Embedding：Seed-1.6-Embedding，Model ID 为 `doubao-embedding-vision-250615`。
- 输出多套聚类结果：
  1. 原始内容（`native_multimodal`）；
  2. 模型生成描述（`asset_description`）；
  3. 各 Feature 标准分别聚类。
- 重复文件提交采用覆盖策略：Source File 身份键为 `(workspace_id, relative_path)`；同内容不同路径
  视为不同来源。Asset 使用由类型和 `source_locator` 生成的稳定 `asset_key` 映射。
- 先交付服务层、CLI 批处理入口和集成测试，HTTP API 后接。
- 用户的正式数据集正在下载；本地先使用 `data/dev-fixtures/` 小数据集。
- 当前支持 Markdown、普通文本、图片和视频；普通文本采用无标题的确定性版式切分。
- 统一资产化入口按文件返回结构化结果；单文件处理失败时记录文件信息和异常日志，
  返回失败结果而不抛异常，是否继续由未来的 Pipeline 层决定。
- Markdown 切分方案已经确定，详见 `MEMORY/markdown-splitting.md`。
- Markdown 与 TXT 的统一分片上限为 400 tokens，可通过
  `CAPSULE_DOCUMENT_CHUNK_MAX_TOKENS` 调整。
- 图片资产化基础方案已经确定，详见 `MEMORY/image-assetization.md`。

## 现有实现差距

- `.txt` 已接入 Discovery、Assetization、PostgreSQL 和 native Embedding；日志、OCR 和代码型 TXT
  尚未做专门的格式策略。
- Pipeline runner 已接通 Markdown、图片、视频、PostgreSQL、MinIO 和 CLI；视频 Segment 会生成
  持久化 MP4、封面与关键帧。
- 初始 Alembic migration 已生成并在空开发库执行通过。
- `Asset → EmbeddingRecord → Milvus` 已接成服务层和 CLI；`ClusterRun / ClusterCapsule /
  ClusterMembership` 仍未接成服务层，当前只有 PCA/HDBSCAN 算法原语。

## 待确认设计点

- “原始文件聚类”是 Source File 粒度，还是文档定义的 Asset 粒度。
- 第一阶段是否启用全部十个 Feature Embedding，或先选择高价值子集。
- 视频原始向量的回退顺序与 Seed-1.6 当前账号能力探测。
- 视频采用“PySceneDetect 自然镜头优先、超长 Shot 时间窗口兜底、关键帧与多维信息相邻合并”
  的候选主方案，详见 `MEMORY/video-segmentation.md`。具体参数与本地模型尚未确定；进入视频
  资产化编码前必须先提醒用户，并结合其已有长视频处理经验进行实测。

## 已确认的聚类预处理

- 对每一个 Embedding Type 独立执行 L2 归一化与 PCA，统一降至最多 64 维后再运行 HDBSCAN。
- 实际维度为 `min(64, 样本数 - 1, 原始维度)`；原始 Seed-1.6 Embedding 按当前项目配置为 1024 维。
- 虽然模型支持跨模态统一向量空间，`native_multimodal`、`asset_description` 和各 Feature 通道
  仍分别聚类，不把不同语义通道混为一个数据集。
- 本地 MobileCLIP 的 512 维关键帧向量只用于视频内部代表帧选择，不能与豆包的 1024 维向量混合
  入库或聚类。
- 已使用 18 条本地 Markdown 文本调用真实 Seed-1.6-Embedding 验证：显式请求 `dimensions=1024`
  后返回 1024 维向量；18 样本会因 `sample_count - 1` 被 PCA 到 17 维，HDBSCAN 得到 6 个三样本簇、
  0 噪声。该小样本仅证明链路和参数可运行，不能评估正式聚类质量。
