# Agent 会话系统

## 会话的职责边界

会话是用户可见的短期记忆和历史事实来源，以 `thread_id` 标识，与 HTTP 连接无关。LangGraph checkpoint 保存的是可恢复流程，不是聊天列表；工具审计保存的是操作记录，不是会话消息。

会话业务表由 Alembic 管理，checkpoint 表由 `AsyncPostgresSaver.setup()` 管理。两者可以使用同一 PostgreSQL 实例，但迁移和清理职责彼此独立。

## 核心数据模型

| 表/状态 | 关键内容 | 用途 |
| --- | --- | --- |
| `agent_threads` | 用户、工作区、标题、状态、消息序号、水位、摘要、主题、回合与记忆租约 | 会话元数据与一致性栅栏。 |
| `agent_messages` | 追加式 `sequence`、角色、内容、`turn_id`、`request_id` | 完整对话历史与 Worker 输入。 |
| `workspace_memory_profiles` | 有限数量的近期主题 | 跨会话工作区侧写。 |
| checkpoint 表 | 图节点状态、待确认动作、工具历史 | 中断恢复，不面向前端读取。 |

消息序号在一个会话内单调递增。`(thread_id, role, request_id)` 唯一约束使客户端对同一请求的网络重试不会重复追加同一角色消息。

## 双写与热镜像

PostgreSQL 是权威，图状态只是可丢弃的热镜像。

```text
用户输入
  → 追加 PostgreSQL 用户消息（request_id 幂等）
  → 更新 LangGraph AgentState.messages
  → 图执行
  → 追加 PostgreSQL 助手消息（request_id 幂等）
  → 更新热镜像水位并返回
```

`working_context` 保存以下轻量水位：

| 字段 | 作用 |
| --- | --- |
| `hot_mirror_through_sequence` | 图内消息镜像已包含到的数据库序号。 |
| `summary_covered_sequence` | 短期摘要已覆盖到的序号。 |
| `memory_revision` | Worker 更新摘要后递增，用于失效热镜像。 |
| `conversation_summary`、`conversation_topic` | 已提交短期摘要和主题。 |

普通回合开始时，Runtime 只读取 `last_message_sequence` 与 `memory_revision`。缓存缺失、实例切换、其他进程推进消息，或摘要被 Worker 更新时，才从数据库重新装载“摘要 + 摘要水位后的消息”。待确认回合不做这类重建，以免破坏恢复语义。

## 单活跃回合租约

一次 Runtime 调用可能包含模型等待、多个工具和确认恢复。为防止两个 API 实例交叉改写同一图状态，`AgentRuntime.invoke` 必须先取得 PostgreSQL 回合租约。

| 字段 | 默认值/语义 |
| --- | --- |
| `turn_lease_owner` | 本次 Runtime 随机所有者标识。 |
| `turn_lease_expires_at` | 有效期；默认由 `agent_turn_lease_seconds=300` 控制。 |
| 心跳 | 每约三分之一租期续租，最长 30 秒一次。 |
| 重叠请求 | 抛出 `AgentTurnBusyError`，HTTP API 映射为 409。 |
| 释放 | 仅所有者能清空租约；进程异常后等待过期可恢复。 |

首次 `/invoke` 可能还没有 `agent_threads` 行。租约获取使用 PostgreSQL `INSERT ... ON CONFLICT DO NOTHING` 后再锁行，避免并发首请求的创建竞争。

回合租约不等于 Memory Worker 的 `memory_lease_*`：前者保护用户请求与图状态，后者按会话串行化摘要和长期记忆整理。两者必须各自续租。

## 生命周期

会话状态只有 `active`、`archived`、`deleted`：

| 操作 | 数据保留 | 后续输入 | checkpoint/队列处理 |
| --- | --- | --- | --- |
| 归档 | 保留消息和摘要 | 禁止 | API 删除图 checkpoint；恢复后不恢复旧待确认动作。 |
| 恢复 | 保留全部历史 | 允许 | 清除过期租约，按摘要 + 最近消息开始。 |
| 软删除 | 保留可审计历史但默认隐藏 | 禁止 | 清除未处理记忆 Outbox 与 checkpoint；已沉淀的跨会话记忆不自动撤销。 |

归档和软删除会拒绝存在有效回合租约的会话，避免在模型或工具正在执行时删除 checkpoint。重命名不改变历史或摘要。

## API 与错误语义

Agent 路由位于 `/api/v1/agent`：

| 路径 | 作用 |
| --- | --- |
| `POST /invoke` | 发送新输入、确认或取消待确认动作。 |
| `POST /threads` | 创建会话。 |
| `GET /threads` | 按工作区、状态和标题查询。 |
| `PATCH /threads/{thread_id}` | 重命名。 |
| `POST /threads/{thread_id}/archive` | 归档并丢弃 checkpoint。 |
| `POST /threads/{thread_id}/restore` | 恢复。 |
| `DELETE /threads/{thread_id}` | 软删除并丢弃 checkpoint。 |

线程状态或活跃回合冲突返回 409；身份/工作区不匹配和不存在的资源返回 404。`request_id` 应由客户端在网络重试时复用，不能为每次重试生成新值。

## 已有迁移

| 迁移 | 内容 |
| --- | --- |
| `20260917_0029` | 会话、消息、记忆、侧写和 Outbox 基础表。 |
| `20260917_0032` | 热镜像消息序号水位。 |
| `20260917_0033` | 软删除时间、状态约束和生命周期索引。 |
| `20260917_0034` | `turn_lease_owner` 与 `turn_lease_expires_at`。 |

部署包含本次功能时必须执行 `uv run alembic upgrade head`，并确保所有 API 实例使用同一 PostgreSQL 数据库。

## 验证与待办

真实 PostgreSQL 生命周期、摘要水位和回合租约由 `tests/test_agent_conversation_management_integration.py` 覆盖；跨 Runtime checkpoint 确认恢复由 `tests/test_agent_postgres_checkpoint_integration.py` 覆盖。

尚未实现的事项：消息编辑和分支会话的不可变版本模型、会话导出/保留期/物理清理策略。实现这些能力前必须先确定它们对摘要水位、长期记忆来源和 checkpoint 的继承或失效语义。
