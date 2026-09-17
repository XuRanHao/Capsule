# Agent 记忆系统

## 范围与层级

系统将“可见会话历史”和“可跨会话复用的信息”分开管理：

| 层级 | 作用域 | 权威存储 | 典型内容 |
| --- | --- | --- | --- |
| 短期记忆 | 单个 `thread_id` | `agent_messages` + `agent_threads` 摘要 | 当前目标、最近决定、工具结果、对话上下文。 |
| 工作区长期记忆 | 一个工作区 | `agent_memories(scope=workspace)` | 项目规则、约束、事实、跨会话偏好。 |
| 通用记忆 | 同一 `user_id` 的所有工作区 | `agent_memories(scope=global)` | 明确跨工作区稳定的表达偏好或操作习惯。 |
| 运行快照 | 单次 Agent 流程 | LangGraph checkpoint | 待确认动作、当前计划、工具队列；不是业务记忆。 |

原始会话消息永久保留，摘要只是一层可重建压缩视图。`workspace_memory_profiles` 只保存有限的活跃主题，帮助 Worker 判断会话情景；它不存储规则、偏好或细节，不能成为第二份长期记忆。

## 写入原则

长期与通用记忆**完全自动**整理，不打断用户要求确认。自动并不代表任意写入：候选必须具备稳定性、可复用性、足够证据和明确作用域；一次性讨论、模型猜测、工具报错、未解决问题、密钥和口令必须丢弃。

Agent 图只进行同步读取，规划合约不含 `memory_writes`，图末端也没有记忆写入节点。唯一可提交长期/通用记忆变更的组件是独立的 Memory Worker。

## 异步整理链路

```text
原始消息写入 PostgreSQL
  │
  ├─ 常规阈值达到：enqueue_consolidation_if_needed
  └─ 上下文超窗：指定旧前缀 through_sequence 的 enqueue_consolidation
       │
       ▼
agent_memory_outbox（事务 Outbox）
       │
       ▼
MemoryOutboxDispatcher → Redis Stream: agent-memory
       │
       ▼
MemoryWorker（按 thread_id 领取并续租）
       │
       ├─ 第一事务：提交短期摘要、主题和工作区侧写
       └─ 第二事务：提交长期/通用记忆变更、向量 Outbox、整理游标
```

队列消息只携带会话、用户、工作区和截止序号。Worker 必须从 PostgreSQL 读取源消息，不应把完整对话复制进 Redis。

同一会话的重复投递是安全的：`last_consolidated_sequence` 与 `through_sequence` 组成幂等水位；数据库提交成功而 Stream 尚未确认时，重投只会发现已完成状态并确认消息。

## 两阶段提交与主题侧写

第一阶段先生成一次 `ConversationSummary(summary, topic, covered_sequence)` 并提交：

1. 摘要输入包含旧的活跃工作区主题和待整理消息；近期消息权重更高。
2. 若仍属于已有情景，模型必须复用原主题；只有新情景才生成新主题。
3. 服务端限制主题字符数，并确定性合并到 `workspace_memory_profiles`：按活跃时间与覆盖会话数排序，只保留有限数量。
4. `summary_covered_sequence` 与 `memory_revision` 推进后，Runtime 可立即用新摘要恢复短期上下文。

第二阶段在第一阶段成功后进行。失败重试会复用已经提交的摘要和主题，不重复增加侧写主题计数。

## 长期/通用候选与整合

第二阶段的工作区和通用分支并发执行：

```text
已提交摘要 + 源消息 + 活跃主题
  ├─ 工作区候选提取
  └─ 通用候选提取
          │
          └─ 合并按初始置信度排序，单批最多 agent_memory_batch_max_mutations 条（默认 3）
                    │
                    ├─ 每条候选独立检索既有记忆，最多取 3 条
                    └─ 每条候选独立输出受限变更动作
```

候选整合不是面向用户的语义召回，因此不使用向量相似度。每条候选在同一作用域内：

1. 强制加入 `scope + kind + memory_key` 的精确命中；
2. 通过 PostgreSQL `pg_search` BM25 召回词法相关条目；
3. 以精确键/BM25 相关度、有效置信度与时间衰减重排，最多交给模型三条；
4. 仅允许输出 `create`、`merge`、`deactivate`、`lower_confidence`。

冲突默认不做不可逆真假裁决。无法判断真伪时降低旧记忆置信度并保留来源；明确替代时停用旧记录；普通冲突不物理删除。

工作区与通用记忆的默认每日衰减率分别是 `0.002` 和 `0.0005`，因此通用记忆衰减较慢。用户直接否定或纠正时，Worker 应将其作为降低置信度/更新的证据，而不是弹出记忆确认框。

## 同步召回

每个 Agent 请求首次进入 `load_context` 时：

1. 以当前问题文本作为查询。
2. 并发检索工作区范围与同一用户的通用范围，各自最多 `agent_memory_context_per_scope` 条（默认 3）。
3. 正常路径：文本嵌入 → 独立 Memory Milvus collection 的 ANN 候选 → PostgreSQL 按用户、工作区、状态和向量版本回填授权 → 置信度与时间衰减排序。
4. 向量库或嵌入服务异常时，安全降级至 PostgreSQL BM25。

Milvus 永远不是权威来源。`agent_memory_vector_outbox` 在记忆变更事务中生成，专用 Dispatcher/Worker 将最新版本写入或从 Milvus 删除；查询时 PostgreSQL 会拒绝过期向量版本。

同一 `request_id` 只召回一次，后续工具循环复用这批结果。上下文预算导致的本轮摘要等待不会重新注入刚写入的长期/通用记忆。

## Worker 运行与配置

常用命令：

```powershell
capsule agent-memory-dispatcher
capsule agent-memory-worker
capsule agent-memory-vector-dispatcher
capsule agent-memory-vector-worker
```

所有命令支持 `--once`，适用于探针和一次性运维。重要配置包括：

| 配置 | 默认值 | 说明 |
| --- | ---: | --- |
| `agent_memory_max_active_topics` | 5 | 工作区侧写主题上限。 |
| `agent_memory_max_topic_chars` | 32 | 单个主题最大字符数。 |
| `agent_memory_batch_max_mutations` | 3 | 一批最多长期/通用变更数。 |
| `agent_memory_worker_lease_seconds` | 120 | 摘要/记忆 Worker 会话租约。 |
| `agent_memory_vector_worker_lease_seconds` | 120 | 向量投影租约。 |

当一个用户同时有多个会话需要整理时，可以横向运行多个 Worker。每个 Worker 通过会话租约安全竞争；被占用的消息不会提前确认。用户级 Worker 并发上限属于部署调度策略，不由 Memory Worker 自己隐式决定。

## 诊断与测试

排查时先看 `agent_memory_outbox` 的投递状态、会话的 `last_consolidated_sequence`/`summary_covered_sequence`、Worker 租约，再看向量 Outbox；不要直接从 Milvus 推断记忆是否存在。

主要测试：

- `tests/test_agent_memory_worker.py`：队列消费、租约和确认。
- `tests/test_agent_memory_consolidator.py`：摘要、并发候选与整合策略。
- `tests/test_agent_memory_vector_worker.py`：向量 Outbox 投影。
- `tests/test_agent_milvus_memory_store.py`：同步召回与 BM25 降级。
