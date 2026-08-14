# 音频切分与共享媒体 Worker（2026-08-13）

## 已确认算法契约

- 音频声学特征使用独立的 1 秒窗口，窗口之间不重叠。
- 使用 EfficientAT `mn10_as` 的音频特征，仅用于当前文件内部的时间连续聚类。
- 先进行带时间邻接约束的连续聚类，再把不足 3 秒的最终区域并入声学上最接近的相邻区域。
- 最终片段不设置最长时长，也不以目标片段数量或目标时长调参。
- EfficientAT 窗口和向量不写入 PostgreSQL、Milvus 或搜索索引。
- 静音窗口产生零向量时，公共聚类核心将其映射到同一个临时单位方向，使连续静音自然聚类。

## 任务与数据链

- 新任务类型为 `audio`，新 Asset 类型为 `audio_segment`。
- 音频与视频共用现有 `mps_video` ResourceClass、Redis Stream、consumer group、租约围栏和宿主
  Worker；Worker 根据 task kind 分派到 MobileCLIP 视频解析器或 EfficientAT 音频解析器。
- `video-worker` / `video-scheduler` 保持兼容，同时提供 `media-worker` / `media-scheduler` 命令。
- 音频 Asset 只保存原文件时间范围与切分参数，不生成或保存实体分段音频。

## 理解、转写与检索

- 默认理解模型更新为 `doubao-seed-2-0-lite-260428`。
- 理解阶段按 `audio_segment` 的时间范围临时抽取 16 kHz mono WAV，通过 Responses API 的
  `input_audio.audio_url` 直接发送给豆包。
- 同一次结构化理解返回 `transcript`、名称、描述和三类 Feature。
- 转写写回同一个 `audio_segment.raw_content`，处理方式与图片 OCR 文本一致。
- 音频的 `native_multimodal` 搜索通道实际使用转写文本与 `original_text` source mode；不会把
  EfficientAT 向量与 MobileCLIP 或豆包 Embedding 混入同一索引空间。

## 宿主环境与验证

- EfficientAT 官方源码默认目录为 `data/models/EfficientAT`，默认权重为
  `resources/mn10_as_mAP_471.pt`；宿主环境额外需要 PyTorch 与 librosa，Docker 不引入这些依赖。
- 忽略目录 `tmp/EfficientAT`、`tmp/mobileclip-uv311-env` 和 WAV 完成真实 MPS 冒烟：`mn10_as`
  输出 960 维、L2 norm=1 的特征。初版错误复用了 MobileCLIP 的 `0.08–0.25` 阈值范围；实测
  EfficientAT 1 秒相邻窗口距离为 `0.000068–0.013994`，导致所有窗口必然合并。音频专用默认值
  已校准为 `0.002–0.02`，75% 分位数规则不变。
- 40 秒明显变化样本由地铁、中文语音、音乐、白噪声各 10 秒组成；默认配置产生
  `0–5 / 5–10 / 10–13 / 13–20 / 20–30 / 30–40` 秒片段，三个真实场景切换点
  `10 / 20 / 30` 秒全部命中，另保留两个场景内部声学变化点。
- 原 49.065 秒样本在修正默认阈值后重新运行，产生
  `0–6 / 6–10 / 10–15 / 15–37 / 37–40 / 40–49.065` 秒六个片段；此前“一整片”是阈值量纲
  错误的结果，不应作为模型能力结论。
- 2026-08-13 最终检查：Ruff、96 个源文件 Mypy、481 个后端测试（另 2 个跳过）、前端 ESLint、
  build 和 6 个渲染测试均通过。
