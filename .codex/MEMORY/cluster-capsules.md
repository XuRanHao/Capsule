# Cluster Capsule

## 已确认的代表资产规则

- 每个非噪声 Cluster 在 HDBSCAN 实际使用的 PCA 后向量空间中计算 Medoid；Medoid 必是实际 Asset。
- 先按到 Medoid 距离选择中心代表，再可补充 1～2 个低 membership 的边缘 Asset。
- 每个 Cluster 目标展示约 5～10 个代表；同一 `source_file_id` 最多 2 个；无法同时满足时优先遵守
  同源上限。
- 代表记录仅保存 `asset_id`，不存名称、描述或原文。关联表必须使用 Asset 外键，并保存
  `role=medoid|core|edge`、`rank`、到 Medoid 的距离和 membership。

## Cluster Capsule 摘要

- 调用 `doubao-seed-2-0-lite-260215`，模型输入只包含已选代表 Asset，不传完整 Cluster。
- 每个代表携带角色、Asset 类型、Description、与当前 `embedding_type` 对应的有效 Feature、
  文件树上下文、membership、距 Medoid 距离；输入还含 Cluster 的规模与统计信息。
- 输出：中文 `name`、50～150 字 `description`、3～8 个 `keywords`、`common_features`、
  `internal_variance=low|medium|high`。名称必须反映当前 embedding 维度。
- 不得添加输入没有的作者、版权、项目或人物身份；代表差异明显时描述中需说明。

## 持久化与人工覆盖

- `ClusterCapsule` 保存模型生成的名称和描述、人工覆盖的名称和描述及相应的 effective 值。
- 人工覆盖优先；清空人工值后 effective 值自动回退到模型生成值。
- 共同特征、内部差异、Medoid Asset ID 与代表关联作为 Cluster Capsule 的持久化数据。

## 已落地（2026-07-28）

- 迁移 `20260728_0005` 新建 `cluster_representative_assets`：`asset_id` 为外键，且每个
  Capsule 的代表 Asset、rank 均唯一；`representative_asset_ids` 仅保留为兼容用的 ID 缓存。
- `ClusterRepository.upsert_capsule` 会验证同源上限、唯一 Medoid、连续 rank 和 workspace，
  重生成模型摘要时保留人工覆盖；分别传入 `None` 可清空名称或描述覆盖。
- `select_cluster_representatives` 在 PCA 后空间选择 Medoid，随后选中心/边缘代表，未擅自添加
  membership 阈值；边缘资产依低 membership 排序。
- `build_cluster_summary_messages` 只序列化已选代表和聚类统计，Lite 摘要调用使用 Ark
  Responses API 并禁用 thinking。
- 隔离验证库升级成功；相关测试 11 通过，全量结果为 66 passed、2 skipped，Ruff 与 Mypy 通过。

## 单类型聚类入口（2026-07-28）

- `ClusterService.run(workspace_id, embedding_type=...)` 一次只执行一个 Embedding Type，绝不
  自动遍历全部类型；默认 `native_multimodal`。CLI 为
  `capsule cluster --workspace <id> --embedding-type <type>`，省略 Type 时同样使用该默认值。
- 每次调用从 PostgreSQL 读取该 Type 当前已索引的 EmbeddingRecord，并按 ID 精确从 Milvus 取向量；
  PCA、HDBSCAN、ClusterRun、ClusterMembership、Capsule 均以该 Type 独立保存。
- 少于 15 个有效向量时持久化 `insufficient_data` Run，不调用摘要模型，也不影响其他 Type 后续的
  手动调用；Milvus 缺失向量数量写入 Run preprocessing。
- 通过：单类型隔离、代表资产模型输入、成员持久化、Milvus 精确取向量测试；全量为
  69 passed、2 skipped，Ruff/Mypy 通过。

## 前端异步 API（2026-07-28）

- `POST /api/v1/cluster-runs` 接收 `workspace_id` 和可选 `embedding_type`；默认
  `native_multimodal`，立即返回 `202` 和持久化的 `cluster_run_id`。
- 后台任务按该 Type 运行，前端通过 `GET /api/v1/cluster-runs/{id}?workspace_id=...` 轮询，
  通过 `GET /api/v1/cluster-runs/{id}/capsules?workspace_id=...` 获取结果。
- `PATCH /api/v1/cluster-capsules/{id}?workspace_id=...` 可分别写入 `name`、`description`；字段
  显式传 `null` 清除人工覆盖并回退模型值。
- 初版异步采用 FastAPI 进程内 BackgroundTasks：不会阻塞 HTTP 响应，但服务重启不会恢复未完成的
  任务。多进程部署前需接可靠队列。API 测试和全量回归通过：70 passed、2 skipped。

## 真实端到端验证（2026-07-28）

- 使用隔离 PostgreSQL Workspace、临时 2 维 Milvus collection 和真实 Ark Lite Responses API，写入
  20 条 `visual_style` EmbeddingRecord 与对应向量。
- 通过 `POST /api/v1/cluster-runs` 显式提交 `visual_style`，再经两个 GET 接口读取结果：Run 为
  `completed`，`sample_count=20`、`cluster_count=2`，两个 Capsule 均返回真实的 Medoid 与
  `representative_asset_ids`。
- 初次真实调用发现模型偶尔会输出少于 50 字的 description；不能放宽持久化契约。`DoubaoClient`
  现对 ClusterSummary 的 Pydantic 校验失败额外执行一次格式修正请求，携带原始代表资产与明确校验
  错误。真实复验通过；临时 Workspace 和 collection 已删除。

## 可选参数择优（2026-07-29）

- Cluster API 请求支持 `optimize_parameters`，默认 `false`。默认只运行按样本量确定的一组
  HDBSCAN 参数；显式传 `true` 时才运行多组候选并按内部质量分数择优。
- `ClusterService.run` 与 CLI `cluster --optimize-parameters` 使用同一开关；Run preprocessing 分别记录
  `parameter_selection=size_based_default|adaptive_dbcv_silhouette`，parameters 记录实际候选数量。
