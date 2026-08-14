# Asset 理解生成

## 已确定模型与职责

- POC 指定多模态理解模型为 `doubao-seed-2-0-lite-260215`（Doubao-Seed-2.0-lite），用于
  `asset_name`、`asset_description` 与十项 Asset Feature 的生成。
- `doubao-embedding-vision-250615` 仅负责后续 Embedding，不生成自然语言描述或 Feature。
- `asset_features` 保持当前复数命名；数据库 JSONB 不需要迁移即可保存新 Feature 结构。

## 生成与持久化契约

- 模型只生成 `model_value`、`status`、`confidence`、`evidence`。
- 服务层保留人工 `user_value`，用 `user_value ?? model_value` 计算 `effective_value`，并写入
  `updated_by`、`updated_at`；模型不可覆盖人工值或伪造时间。
- 严格字段 `asset_usage`、`target_audience`、`provenance`、`rights_version_authorship` 没有来自
  Asset 内容、`file_tree_context` 或文件元数据的明确证据时，必须为
  `model_value: null, status: unknown, confidence: 0, evidence: []`。
- Markdown 的 `visual_style` 与 `color_composition` 通常为 `not_applicable`。

## 方舟调用方式

- 当前官方 Lite 示例使用在线 `POST /api/v3/responses`，而非项目原有的 `/chat/completions` 路径。
- 用于固定 JSON 抽取时传入 `thinking: {"type": "disabled"}`；不需要深度思考，延迟更可控。
- 先以在线单 Asset 调用，并由应用层限并发。方舟批量推理是依赖 TOS JSONL 输入/输出的异步任务；
  本地 MinIO 不能直接替代，接入 TOS 前不作为首版实现。

## 已验证

- 2026-07-28：本地 `tmp/markdown-chunk-demo.md` 的第一章 Markdown Asset 真实调用 Lite。
- 输入：`raw_content`、`heading_path`、`file_tree_context`；未写 PostgreSQL。
- 响应：HTTP 200，合法 JSON，Description 197 字，十项 Feature 齐全。
- 四个严格字段全部正确返回 `null + unknown`，且没有缺失 evidence 的非空严格值。

## 结构化输出约束（2026-07-29）

- `DoubaoClient.understand_asset` 会在调用方消息前自动注入完整的
  `AssetUnderstanding.model_json_schema()`，明确要求根节点与 `features` 均为对象、十个 Feature
  必须按名称完整出现，且每项必须包含 `value/status/confidence/evidence`；Prompt 同时保留一份完整
  手工 JSON 结构示例，并明确示例只约束结构、实际内容不得照抄。
- 若首次返回不是合法 JSON 或未通过 Pydantic Schema（实测模型曾把 `features` 输出为数组），客户端
  会携带具体校验错误自动修正一次；第二次仍不合法则向上抛错，不会落库不完整结构。
- 使用 `analytics-dashboard.png` 去掉手工 JSON 示例后真实复验通过，返回
  `asset_description` 与完整十项 Feature。
