# 视频切分与关键帧方案

## 当前方案状态

视频处理采用“自然镜头边界优先、固定时间窗口兜底、多维信息相邻合并”的分层方案。
首轮参数已由用户确认（2026-07-28），后续可基于真实视频调整：最短镜头 1 秒、超过 45 秒
的 Shot 按 20 秒窗口拆分、候选帧包含首尾且每 5 秒一帧（最多 12 帧）、每段最多 3 张代表帧。
场景阈值初始采用 PySceneDetect `ContentDetector` 默认值 27；质量过滤和相邻合并的数值阈值仍为
可配置试验值，需在真实数据集上回调。

当前开发边界：视频 OCR 与 ASR 方案暂时搁置，不开发、不安装相关依赖，也不为其预先添加
实现代码。现阶段只推进视觉链路；后续仅在用户明确恢复这两个方向后再讨论和实现。

## 总体流程

```text
原始视频
  → PySceneDetect 检测自然镜头边界
  → 得到 Shot 时间区间
  → 超长 Shot 按固定时间窗口继续划分
  → 每个候选区间均匀采样候选帧
  → 过滤无效帧
  → 轻量聚类，选取最多 3 张关键帧
  → 汇总视觉、OCR、ASR 等多维信息
  → 判断相邻区间语义连续性
  → 相邻合并
  → 最终 video_segment
```

## 两级初始切分

### 自然镜头切分

- 第一层使用 PySceneDetect 检测镜头变化。
- PySceneDetect 只提供候选边界，不将其输出直接视为最终语义 Segment。
- 检测过碎的相邻 Shot 可以在语义合并阶段恢复。

### 时间窗口兜底

- 对超过最大时长的 Shot 再按时间窗口划分。
- 解决固定机位、会议、课程和长镜头中视觉边界不足的问题。
- 时间窗口是计算单位，不是最终语义边界。
- 最终合并必须允许跨 Shot 和跨窗口。

## 逻辑区间原则

初始切分阶段只保存原视频上的逻辑时间范围，不立即生成多个物理短视频，避免重复解码、
重编码和存储：

```json
{
  "source": "scene_then_window",
  "scene_index": 3,
  "window_index": 1,
  "start_ms": 42000,
  "end_ms": 72000
}
```

最终 Segment 确定后，默认只保存原视频区间和代表画面时间戳，不生成派生视频或关键帧文件。
只有显式切换到旧实体模式时才生成派生媒体。完整运行与播放方案见
[可靠整视频任务与逻辑分段架构](reliable-video-task-runtime.md)。

## 关键帧抽取

### 候选帧

- 在每个候选区间内均匀采样。
- 不主动跳过区间开头和结尾，保留标题卡、片头文字和结尾画面的可能性。
- 候选数量应随区间长度变化并设置上限，具体采样频率尚待测试。
- 尽量使用 FFmpeg 顺序解码，避免为每个时间点重复打开视频。

### 无效帧过滤

在聚类前过滤：

- 全黑或接近全黑帧；
- 全白或严重过曝帧；
- 明显模糊帧；
- 转场混合帧；
- 与相邻候选几乎完全相同的帧。

可使用平均亮度、亮度方差、Laplacian 清晰度、感知相似度和相邻帧差异等轻量指标。

### 轻量聚类

- 对有效候选帧生成视觉特征并聚类。
- 每个候选区间最多产生 3 个视觉聚类，不强制固定为 3 个。
- 静态或高度相似的区间可以只保留 1 张关键帧。
- 每个聚类选择距离中心最近且图像质量合格的真实帧，不使用数学聚类中心作为图片。
- 最终关键帧按原视频时间戳排序。

## 多维信息提取

本节仅保留为远期设计备忘。当前阶段不实现 OCR 和 ASR，也不把它们纳入视频处理链路。

候选方向包括：

- 视觉镜头与关键帧特征；
- OCR 文字和文字变化时间点；
- ASR 语句、时间范围和静音信息；
- 相邻区间的时间连续性；
- 后续可选的语义描述或文本向量。

各能力应作为可选信号源，通过统一时间线融合，不能让单一能力成为所有视频的强制依赖。

ASR 优先对原始音轨执行一次并获得全局时间戳，再将结果分配到候选区间，避免逐窗口识别时
从句子中间截断。OCR 采用稀疏抽帧和文字变化检测，避免对每一帧执行识别。

## 本地轻量模型候选

以下作为后续评测候选，尚未确定正式引入：

- 视觉特征：MobileCLIP-S0；
- 镜头检测增强：TransNetV2；
- ASR：whisper.cpp base 或 small；
- OCR：PaddleOCR Mobile；
- 中文文本语义：BGE-small-zh-v1.5。

本地推理运行在 macOS 宿主机，使用 MPS。Docker Linux 容器不能直接利用 macOS 的 MPS/ANE，
因此不把 PyTorch 或 MobileCLIP 强行加入当前应用镜像；后续以独立宿主 Worker 连接 Capsule
PostgreSQL。

## MobileCLIP-S0 宿主机冒烟测试（2026-07-28）

- 使用 Apple 官方 `ml-mobileclip` 主分支和 `apple/MobileCLIP-S0` 权重完成试跑。
- 权重大小为 215,934,653 bytes（约 216 MB / 206 MiB），SHA-256 为
  `809b408eff74f8058843e86a1f92967097d42ba782450e85b8f4867b7f0ca0b7`。
- 输出为归一化的 512 维图片向量。
- M4 Mac mini、PyTorch 2.13.0、9 张 640×360 图片单批测试：
  - CPU 中位耗时约 335.8 ms/批，约 37.31 ms/图；
  - MPS 中位耗时约 50.1 ms/批，约 5.57 ms/图；
  - 当前小批量下 MPS 约为 CPU 的 6.7 倍。
- 9 张开发图片最近邻类别命中 9/9；但同类别测试图共用近似底图，只改变少量图形或文字，
  该结果只能证明链路可运行，不能代表真实关键帧聚类准确率。
- 试验环境位于被忽略的 `tmp/mobileclip-env/`，约占 760 MB；权重位于被忽略的
  `data/models/mobileclip-s0/`。尚未加入项目依赖或 Docker 镜像。
- 官方代码为 MIT License，模型权重使用 Apple ML Research Model Terms（`apple-amlr`），
  正式采用前需要确认使用场景符合条款。
- `create_model_and_transforms()` 默认已经执行重参数化，调用方不得再次调用
  `reparameterize_model()`，否则当前实现会在 `RepMixer` 上重复处理并报属性不存在。

## 降级原则

- PySceneDetect 检测不到边界时，将整个视频交给时间窗口兜底。
- 单个信号源失败时，使用其他可用信号继续处理并记录日志。
- 所有语义信号不可用时，至少保留可定位的时间区间，不伪造 OCR、ASR 或内容描述。

## 长任务分发参考

已阅读旧项目 `/Users/gao/work/videoTask`。其 Redis Streams 分发器可作为视频算法稳定后的执行层
参考，重点保留：稳定任务标识、结果格式版本、执行轮次隔离、工作进程心跳、FFmpeg 实际进度、
无进展超时、最长执行时间、进程组终止、待确认消息恢复、延迟重试和死信流。

当前不直接复制代码或引入 Redis。第一阶段先完成本地“视频 → 逻辑 Segment → Asset →
PostgreSQL”流程；真实长视频验证通过后，再把整个视频作为长任务交给独立 Worker。

旧原型需要按 Capsule 改造：删除 SQLite 模拟后端和 Java 回调，工作进程直接事务更新 Capsule 的
PostgreSQL `processing_jobs`、`source_files` 和 `assets`。数据库提交成功后才 XACK；ACK 失败导致
重投时，通过稳定任务标识、处理阶段、执行轮次和结果格式版本识别已提交结果并只做收尾。固定间隔
抽帧替换为当前场景切分与关键帧视觉方案。

## 可靠整视频任务已落地（2026-08-11）

可靠任务提交、Redis Streams 消息投递、PostgreSQL 写入保护、真实进度与取消、故障恢复和死信补发
已完成并通过架构验收。完整方案、状态机、一致性规则、运行命令和验证记录见
[可靠整视频任务架构](reliable-video-task-runtime.md)。

## 待确认与实测

- PySceneDetect 使用的 Detector、阈值和最短场景长度。
- 多长的 Shot 判定为超长。
- 超长 Shot 的时间窗口长度。
- 过短 Shot 是否在关键帧处理前合并。
- 候选帧采样频率和数量上限。
- 黑帧、过曝、模糊和重复帧的过滤阈值。
- 轻量聚类使用的视觉特征、聚类算法和动态聚类数判断。
- 相邻区间多信号融合与合并规则。
- 本地 OCR、ASR、视觉模型的最终选型、运行位置和资源消耗。

## 已验证的最小开发环境

- FFmpeg / FFprobe 5.1.9；
- PySceneDetect 0.7.1；
- OpenCV 5.0.0（由 PySceneDetect 依赖提供）；
- Pillow 11.3.0；
- NumPy 2.4.6；
- scikit-learn 1.9.0。

FFmpeg 已确认包含 `fps`、`select`、`thumbnail` 和 `blackdetect` 滤镜。测试 MP4 已完成
PySceneDetect 场景检测、OpenCV 均匀采样 9 帧和 scikit-learn 3 中心聚类的内存冒烟测试。

MobileCLIP 仅在宿主机的隔离目录完成试验；Whisper、PaddleOCR 仍不安装且对应方案已搁置。
运行位置已由用户确认（2026-07-28）：macOS 宿主 Worker + MPS。视频核心解析和可选
`MobileClipMpsEmbedder` 已接入代码；Docker 使用假向量器完成入库集成测试，真实调用必须在宿主
MPS 环境执行。

当前宿主检查（2026-07-28）：arm64 macOS、Apple M4 和 Metal 均可用。相同的 PyTorch 2.13.0
环境在 Codex 工作区沙箱内显示 `is_built()=True`、`is_available()=False`，但在沙箱外显示
`is_available()=True` 且 `torch.mps.device_count()=1`；根因是 Codex 沙箱不授予 Metal/GPU 访问，
并非宿主 MPS 故障。宿主仍缺少 FFmpeg/FFprobe，真实 Worker 需要先安装这两个二进制；MPS Worker
必须由普通宿主 Terminal/服务进程运行，不能由 Codex 终端直接运行。

后续已完成：从 FFmpeg 官方 8.1.2 源码在被忽略的 `tmp/tools/ffmpeg/` 构建原生 arm64
`ffmpeg` 与 `ffprobe`（含 VideoToolbox），`VideoParser` 在 macOS 会自动发现该路径；Docker
继续使用容器自身的 Linux 二进制。将当前 Capsule 以可编辑方式装入隔离宿主环境后，
在沙箱外对 `hiking-trip.mp4` 完成了真实链路：FFprobe → PySceneDetect/OpenCV → MPS
MobileCLIP-S0 → PostgreSQL；结果为 1 个完成 Job 和 1 个 `video_segment`，代表帧时间戳 0 ms。
隔离验证库的测试工作区随后已删除。

## uv / Python 3.11 全链路验证（2026-07-28）

- 宿主机原本没有全局 `uv`；已将官方 uv 0.11.32 安装至被忽略的
  `tmp/tools/uv/`，不修改系统 Python 或 shell 配置。
- 仓库固定 `.python-version = 3.11`。首次以 Python 3.12 运行虽能成功，但 uv 会提示版本不一致；
  已改为 uv 下载的 CPython 3.11.15，并创建 `tmp/mobileclip-uv311-env/`。
- 该环境通过 uv 安装 Capsule、PyTorch 2.13.0、torchvision、timm、MobileCLIP 与
  `open-clip-torch`；后者是 MobileCLIP 推理的直接运行时依赖。
- 在沙箱外使用 `uv run --active --no-sync capsule mps-video` 完成真实链路，确认
  Python 3.11、MPS、FFmpeg、MobileCLIP-S0 与 PostgreSQL 均可用，导入 3 秒开发视频得到
  1 个 `video_segment` Asset。临时工作区随后已删除。
- 已删除被替代的 `tmp/mobileclip-env/`（旧 pip / Python 3.12）和
  `tmp/mobileclip-uv-env/`（旧 uv / Python 3.12），只保留当前 Python 3.11 的 uv 环境。
- 新逻辑视频 Asset 的 `processing_status` 从 `pending` 开始，随后正常进入内容理解、向量和聚类；
  模型画面按代表时间戳从原视频临时提取，用完释放，不保存关键帧文件。

## 旧实体媒体持久化记录（2026-07-28）

- 这一节记录旧实体模式，不再代表默认流程。当前默认逻辑模式不生成视频、封面或关键帧文件；需要
  兼容旧流程时可显式设置 `CAPSULE_VIDEO_OUTPUT_MODE=materialized`。
- 视频 Asset 的通用外部契约不增加 `keyframe_uris` 一级字段：`derived_file_uri` 指向 MP4，
  `preview_uri` 指向封面，`file_info.keyframes` 保存带 URI、时间戳、质量指标和角色的关键帧列表。
- 对象键使用 `workspace_id/source_file_id/generation/asset_key`，同 generation 的同一 Segment 重跑会
  覆盖对应对象，但旧 generation 不会污染新 generation。Segment 数量减少时可能留下未引用对象，
  后续需补对象存储的垃圾回收/生命周期策略。
- 宿主 Worker 使用 `http://127.0.0.1:9000` 访问本机映射的 MinIO；Docker app 使用
  `http://minio:9000`。两者不能混用。真实验证生成了 MP4（`ftyp` 头）和 JPEG（`FFD8` 头），随后
  已删除临时工作区与对象。
