# Agent 开发文档

本文档集描述 Capsule 当前 Agent 的运行边界与实现事实。它面向后端开发、工具开发和运维人员；产品设想或未实现能力会明确标为“待办”，不能据此修改生产行为。

## 文档导航

| 主题 | 关注的问题 | 文档 |
| --- | --- | --- |
| 工具 | 模型如何选择、获得定义、确认并执行工具 | [工具系统](tools.md) |
| 记忆 | 三层记忆如何整理、存储、召回与投影 | [记忆系统](memory.md) |
| 上下文 | 超出模型窗口时如何压缩而不删除源数据 | [上下文预算](context.md) |
| 会话 | 消息权威来源、热镜像、生命周期与并发 | [会话系统](conversations.md) |
| 循环 | LangGraph 节点、状态与恢复路径 | [图循环](graph-cycle.md) |

## 系统总览

```text
HTTP /invoke
  │
  ├─ PostgreSQL 会话回合租约 ──> 防止同一 thread_id 的重叠执行
  │
  ├─ PostgreSQL 消息历史 ──────> 权威的短期记忆与前端会话来源
  │                                 │
  │                                 ├─ LangGraph 热状态 / checkpoint
  │                                 └─ Memory Outbox ─> Redis Stream ─> Memory Worker
  │                                                                    │
  │                                                                    ├─ 短期摘要 + 主题
  │                                                                    ├─ 工作区长期记忆
  │                                                                    └─ 用户通用记忆
  │
  └─ LangGraph：准备 → 召回 → 预算治理 → 规划 → 工具/回复
                              │
                              └─ 可同步等待“摘要水位”，不等待长期记忆写入
```

## 权威来源与派生数据

| 数据 | 权威来源 | 可丢失/可重建的副本 | 写入者 |
| --- | --- | --- | --- |
| 可见会话消息 | PostgreSQL `agent_messages` | LangGraph `messages` 热镜像 | Agent Runtime |
| 暂停的 Agent 流程 | LangGraph PostgreSQL checkpoint | 无 | LangGraph |
| 短期摘要与主题 | PostgreSQL `agent_threads` | 运行时 `working_context` | Memory Worker |
| 工作区/通用记忆 | PostgreSQL `agent_memories` | Milvus 向量 | Memory Worker |
| 工具审计与幂等 | PostgreSQL `agent_tool_executions` | 无 | Tool Registry |

PostgreSQL 是业务数据唯一权威来源。Milvus 只提供召回候选，Redis Stream 只传递可恢复任务，LangGraph checkpoint 只保存流程状态；三者都不能替代会话、记忆或审计表。

## 关键不变量

1. 原始会话消息只追加，不因摘要、上下文压缩或记忆整理而删除。
2. Agent 图只读长期/通用记忆；长期记忆的创建、合并、失效和降置信度只能由 Memory Worker 提交。
3. 工具必须先完成名称选择与完整 Schema 披露，才允许生成调用参数。
4. 所有工具调用必须在服务端重新做参数、权限、确认、幂等和并发校验；模型输出不是授权。
5. 单个 `thread_id` 同时最多一个活跃 Runtime 回合；业务回合租约与 Memory Worker 租约互不替代。
6. 一次 Runtime 请求内长期/通用记忆最多同步召回一次，工具循环复用同一批结果。

## 配置与数据库升级

Agent 配置位于 `src/capsule/config.py`，环境变量使用 `CAPSULE_` 前缀。例如：

```dotenv
CAPSULE_AGENT_CONTEXT_WINDOW_TOKENS=32000
CAPSULE_AGENT_CONTEXT_OUTPUT_RESERVE_TOKENS=4000
CAPSULE_AGENT_TURN_LEASE_SECONDS=300
```

修改模型、数据库或部署环境后，先执行：

```powershell
uv run alembic upgrade head
uv run pytest tests/test_agent_framework.py tests/test_agent_context_budget.py
```

涉及 PostgreSQL 持久化、租约或 checkpoint 时，再运行真实集成测试：

```powershell
$env:CAPSULE_RUN_POSTGRES_INTEGRATION = '1'
uv run pytest tests/test_agent_conversation_management_integration.py tests/test_agent_postgres_checkpoint_integration.py
```

## 文档维护规则

- 先修改实现和测试，再更新对应主题文档；不要把计划写成“已实现”。
- 修改状态字段、迁移、默认配置、节点路由或 Worker 提交顺序时，必须同步更新本文档集。
- 设计决策应写清楚作用域、失败语义和恢复方式，而不只记录接口名称。
