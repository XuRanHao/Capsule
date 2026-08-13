# Capsule

Capsule 是一个面向个人多模态素材的整理、理解与检索系统。

它可以统一处理文档、图片、视频和音频，将原始文件转换为结构化素材，并提供内容理解、
自动切分、聚类、关系图和多模态搜索能力。

## 主要能力

- 导入 Markdown、TXT、Word、PDF、图片、视频和常见音频。
- 自动提取文档中的正文、表格和图片。
- 统一切分不同格式的文档，并保留父子层级和来源信息。
- 对视频进行内容感知切分，生成适合检索和浏览的片段。
- 对音频做连续声学聚类，并将片段转写接入现有文本检索链。
- 为素材生成内容描述与多维语义特征。
- 对相近素材进行自动聚类，并支持增量归类和人工调整。
- 构建素材与实体之间的关系图。
- 使用文字、图片或图文组合搜索素材。
- 在网页中查看素材、任务、聚类、关系图和搜索结果。

## 处理流程

```text
导入文件
  -> 识别文件类型
  -> 文档 / 图片 / 视频 / 音频处理
  -> 生成结构化素材
  -> 内容理解与向量化
  -> 聚类和关系图更新
  -> 搜索与浏览
```

### 文档

TXT、Word 和 PDF 会先转换为统一的 Markdown 结构，再进入公共切块流程。

- 普通内容以约 400 Token 为目标切分。
- 理想范围为 250～500 Token。
- 过短内容会与相邻内容合并，合并后最多约 600 Token。
- 表格保持完整，不在表格内部切分。
- 文档中的图片会作为独立素材处理，并保留原文档来源。
- 子块用于精确检索，父块用于补充完整上下文。
- Token 数量默认由项目内置的 DeepSeek V3 tokenizer 在本地计算。

### 图片

图片会生成适合模型处理的统一尺寸版本，并保留其文件、文档或页面来源。
文档内嵌图片会先过滤无效小图，再按需要进行 OCR 和内容理解。

### 视频

视频先按固定时间间隔采样，再结合连续画面内容和变化强度生成时间片段。
关键帧直接复用采样结果，最终保留轻量图片，避免重复解码原视频。

当前方案的重点是：先尽量识别真实内容边界，再合并过碎且内容连续的片段，
而不是简单地把所有视频固定切成相同时长。

### 音频

音频由宿主机上的 EfficientAT `mn10_as` 生成临时声学特征：使用独立的 1 秒窗口、窗口间
不重叠，先进行时间连续聚类，再把不足 3 秒的区域并入声学上最接近的相邻区域。最终片段
不设最长时长，声学向量不写入索引或数据库。

每个逻辑音频片段会直接交给豆包 `doubao-seed-2-0-lite-260428` 理解并转写。转写保存在同一个
`audio_segment` 的 `raw_content` 中，随后与图片 OCR 文本一样进入现有文本向量化链。

视频和音频共用同一套 PostgreSQL fenced task、Redis Stream 和 macOS MPS Worker；仅根据 task
kind 在 Worker 内分别调用 MobileCLIP 或 EfficientAT。

## 聚类与关系图

系统可以从不同语义维度组织素材，例如主体内容、场景主题和视觉呈现。
新素材可先尝试加入现有聚类；积累到一定数量后，再触发整体重聚类。

关系图用于展示素材、人物、物体或其他实体之间的联系，并支持持久化和增量更新。

## 搜索

搜索支持：

- 纯文本查询。
- 纯图片查询。
- 图片与文字组合查询。
- 按 Workspace、文件类型、时间和聚类过滤。
- 从多个语义维度召回并合并结果。

## 本地启动

### 环境要求

- Python 3.11+
- `uv`
- Node.js 22+
- Docker Desktop
- FFmpeg 与 FFprobe
- Apple Silicon + MPS（视频和音频媒体 Worker 需要）

宿主媒体 Worker 还需要 PyTorch、MobileCLIP、`librosa` 与 EfficientAT 官方源码。EfficientAT
源码目录默认为 `data/models/EfficientAT`，其中 `resources/mn10_as_mAP_471.pt` 为默认权重。
这些宿主推理依赖和模型不进入 Docker 镜像，也不由 Git 跟踪。

### 初始化

```bash
make setup
```

首次运行时，在项目根目录的 `.env` 中填写模型服务密钥：

```dotenv
CAPSULE_ARK_API_KEY=your-ark-api-key
CAPSULE_DEEPSEEK_API_KEY=your-deepseek-api-key
```

`.env` 不会被 Git 跟踪；其他可配置项可参考 `.env.example`。

### 启动开发环境

```bash
make dev
```

默认入口：

- Web：<http://localhost:3000>
- API：<http://localhost:8010>
- API 文档：<http://localhost:8010/docs>

### 常用命令

```bash
make status     # 检查本地服务状态
make test       # 运行测试
make down       # 停止本地基础设施
```

媒体 Worker 与恢复调度器可在宿主推理环境中运行：

```bash
capsule media-worker
capsule media-scheduler
```

兼容命令 `video-worker` 和 `video-scheduler` 仍然可用，并指向同一实现。

## 页面

- `/import`：导入文件。
- `/tasks`：查看处理任务和进度。
- `/assets`：浏览已生成的素材。
- `/clusters`：查看和调整聚类。
- `/graph`：查看素材关系图。
- `/search`：执行多模态搜索。
- `/capsules`：查看保存的搜索结果。

## 开发检查

后端：

```bash
uv run ruff check .
uv run mypy src/capsule
uv run pytest
```

前端：

```bash
cd frontend
npm run lint
npm test
```

数据库结构更新后执行：

```bash
uv run alembic upgrade head
```

## 数据说明

本地运行产生的原文件、缓存、模型、日志、基准数据和 Demo 数据不会随代码提交。
仓库只保存程序、迁移、测试、配置示例和必要文档。
