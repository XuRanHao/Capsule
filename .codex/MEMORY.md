# 项目记忆索引

## 项目上下文

- 项目：个人资产 Capsule 冷启动聚类 POC。
- 当前负责角色：角色 A——Embedding 聚合入库。
- 责任链路：原始文件 → Asset → Description / Feature → Embedding → PostgreSQL + Milvus → 聚类结果。

## 主题索引

- [Docker 开发环境](MEMORY/docker-development.md)：容器组成、验证结果和 ARM64 构建注意事项。
- [第一阶段开发范围](MEMORY/development-scope.md)：已确认交付边界、现有差距和待确认设计点。
- [Markdown 切分方案](MEMORY/markdown-splitting.md)：标题递归切分、不可拆分项、列表与批量 Tokenization 规则。
- [TXT 切分方案](MEMORY/text-splitting.md)：无标题纯文本的版式识别、段落合并和句子边界规则。
- [图片资产化方案](MEMORY/image-assetization.md)：整图 Asset、客观元数据、EXIF 与缩略图规则。
- [视频切分与关键帧方案](MEMORY/video-segmentation.md)：镜头优先、时间窗口兜底、关键帧聚类与多信号合并。
- [音频切分与共享媒体 Worker](MEMORY/audio-segmentation-and-media-worker.md)：1 秒无重叠声学窗口、连续聚类、3 秒短片段合并、EfficientAT 临时向量、豆包直传转写及视频/音频共用 Worker。
- [可靠整视频任务与逻辑分段架构](MEMORY/reliable-video-task-runtime.md)：无分段文件和关键帧落盘、源区间播放、按需兼容转码，以及 PostgreSQL 权威任务状态机。
- [逻辑视频 50 样本性能测试](MEMORY/logical-video-performance-50.md)：分层样本、阶段平均耗时、实时系数、临时取帧、峰值内存和并发 1/2/4 压测结果。
- [通用任务运行时不入库压测](MEMORY/processing-task-runtime-load-test-2026-08-12.md)：真实 Redis 下的发布吞吐、完成吞吐、延迟、积压和异常消息隔离结果。
- [视频任务与资产入库压测](MEMORY/video-asset-persistence-load-test-50.md)：50 个分层视频、并发 2、真实 PostgreSQL 资产入库、阶段耗时和下游零调用验收结果。
- [多模态可靠任务处理架构](MEMORY/durable-multimodal-processing-tasks.md)：视频、图片和文本的三路耐久任务、租约围栏、原子入库、下游边界与浏览器接入现状。
- [多模态可靠任务 200 文件压测](MEMORY/mixed-durable-task-load-test-200.md)：50 个视频、100 张图片、50 个文本的真实入库压测、阶段耗时、资源占用和零下游调用结果。
- [文件分块与 Asset 生成开发方案](MEMORY/assetization-development-plan.md)：三类文件的统一中间层、Asset 字段映射、开发顺序与待确认契约。
- [文件资产化实现记录](MEMORY/file-assetization-implementation.md)：已落地模块、数据库覆盖语义、测试结果和剩余视频边界。
- [Embedding 持久化](MEMORY/embedding-persistence.md)：Asset 到 PostgreSQL EmbeddingRecord、Seed 和 Milvus 的幂等入库规则。
- [Asset 理解生成](MEMORY/asset-understanding.md)：Description / Feature 的 Lite 模型、官方调用方式、严格证据规则与真实验证结果。
- [Cluster Capsule](MEMORY/cluster-capsules.md)：代表资产选择、摘要模型输入、持久化与人工覆盖规则。
- [聚类算法评测](MEMORY/clustering-evaluation.md)：算法对比指标、初测结果与真实数据复测要求。
- [长文本切块评测](MEMORY/text-chunking-evaluation.md)：LongBench 中文长文上 300–800 token
  切块的 Recall、MRR、证据完整性和重复率结论。
- [浏览器文件夹导入](MEMORY/browser-folder-import.md)：showDirectoryPicker 优先、逐文件可重试上传与后台资产化协议。
- [前端开发环境](MEMORY/frontend-development.md)：Node/npm 要求、构建插件与本地启动注意事项。
- [工作区清库](MEMORY/workspace-clear.md)：Assets 页清库范围、确认词与运行中任务限制。

## 未完成事项

- 确认原始聚类粒度、首批 Feature 维度和视频 Embedding 回退策略。
- 对象存储中的浏览器不兼容视频目前不做无落盘转码；本地按需转码也不缓存结果。

## 已知问题与踩坑

- Docker Registry 网络不稳定时可能出现 TLS handshake timeout；重试会复用已下载的镜像层。
- Compose 启动时若端口冲突，失败容器可能未加入项目网络；释放端口后应强制重建该服务。
- GitHub 当前凭据 `Gao-LKe` 已成功推送到 `XuRanHao/Capsule` 的 `main`；后续功能完成后仍需
  由用户明确授权再提交和推送。

## 已验证经验

- Docker 应用环境使用容器内的 uv；需要 MPS 的视频/音频媒体 Worker 则必须在 macOS 宿主机运行。
- 宿主机没有全局 uv 时，使用项目忽略目录 `tmp/tools/uv/uv`；MPS Worker 使用 uv 创建的
  `tmp/mobileclip-uv311-env/`，并遵守仓库 `.python-version` 的 Python 3.11 约束。
- MPS Worker 的最小推理依赖为项目依赖、PyTorch、torchvision、timm、Apple MobileCLIP 源码及
  `open-clip-torch`；这些依赖和模型不进入 Docker 镜像。
- ARM64 环境安装 `hdbscan` 时需要源码编译，因此镜像必须包含编译工具链。
- Ark ClusterSummary 可能返回 JSON 正确但 description 不足 50 字；保持 Schema 严格校验，并在
  `DoubaoClient.summarize_cluster` 中追加一次明确格式修正请求后，真实端到端验证已通过。

## 用户意见

- 架构文档应以中文为主，先讲清总体流程、数据边界和新旧兼容关系，避免大量中英混写。
- 视频默认只记录原视频区间与代表画面位置，不保存分段视频或关键帧文件；资产仍正常入库并进入理解、向量和聚类链路。
- 多视频性能测试可以只测切分与临时取帧，不入库、不跑向量；这只是测试范围，不能改变生产流程。
- 前端必须同时兼容旧实体片段与新逻辑区间播放。
