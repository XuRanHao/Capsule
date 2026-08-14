# Embedding 持久化

## 已实现链路

```text
PostgreSQL Asset
  → AssetEmbeddingService
  → Seed-1.6 Embedding（1024 维）
  → PostgreSQL embedding_records
  → Milvus asset_embeddings_seed16_1024
```

- CLI：`capsule embed --workspace <id>`；默认通道为 `native_multimodal`，可使用
  `--embedding-type` 选择描述或单个 Feature 通道，`--asset-id` 限制处理对象，`--force`
  才重算已索引的逻辑输入。
- 每条向量都以 `embedding_id` 为 Milvus 主键；Milvus upsert 可安全重试。
- PostgreSQL 在模型调用前创建/复用 `EmbeddingRecord(status=processing)`，成功写 Milvus 后标记
  `indexed`；失败标记为 `failed`。单个 Asset 失败会汇总到 CLI 结果，其他 Asset 继续。
- 幂等键：`asset_id + embedding_type + model_name + dimension + source_content_hash`。
  相同原始内容重复运行不会产生重复记录或重复模型调用；`--force` 保持同一个 `embedding_id`
  并覆盖该 Milvus 向量。

## 原始输入规则

- Markdown：`raw_content` → `text`。
- 图片：从 `SourceFile.storage_uri` 读取原始图片，作为 `data:image/...;base64,...` 传给
  `image_url`。
- 视频：使用真实切片的 `derived_file_uri`；`s3://` URI 通过 Object Storage 生成临时签名 HTTP URL，
  作为 `video_url` 传给模型。不是关键帧聚合向量。

## 已验证

- Docker 隔离库的服务测试：重复运行跳过、`--force` 重写、单个失败不阻断批次。
- 真实 E2E：导入本地 Markdown、调用 `doubao-embedding-vision-250615`、写入 PostgreSQL 与 Milvus
  均成功；模型实际返回 1024 维，PostgreSQL `embedding_id` 与 Milvus 主键一致。临时 workspace 已清理。
- 完整检查：`ruff check .`、`mypy src/capsule`、`pytest` 通过（62 passed, 2 skipped）。

## 当前限制

- Ark 必须能下载视频 URL；Docker 本地 MinIO (`minio` / `localhost`) 仅本机可达，视频正式运行前
  要设置 Ark 可访问的 `CAPSULE_OBJECT_STORAGE_PUBLIC_ENDPOINT`（通常为生产对象存储的签名 URL 域名）。
- Description / Feature 的生成器尚未接入；服务可在字段已有内容后索引各自的独立通道。
- 向量删除与 Asset 覆盖之间的生命周期清理尚未实现；生产上应在 Asset 删除/重分片时删除对应
  Milvus 向量，避免孤儿向量。
