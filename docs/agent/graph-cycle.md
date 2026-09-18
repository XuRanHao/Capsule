# Agent LangGraph 循环

## 图的职责

LangGraph 管理一条可中断、可恢复的 Agent 执行流程。它协调规划、上下文、工具和确认，但不替代 PostgreSQL 会话历史、工具审计或 Memory Worker。图的 `thread_id` 与业务会话 `thread_id` 一致；在持久化部署中 checkpoint 使用 PostgreSQL Saver。

## 模型 Planner 与应用注入

正常 API 启动且配置了 Ark 密钥时，应用创建一个 `DoubaoClient`，并将
`ModelAgentPlanner` 注入默认 `AgentRuntime`。Planner 通过该客户端的严格 JSON
输出能力生成 `PlanDecision`；模型名称与最大输出 Token 分别由
`CAPSULE_AGENT_PLANNER_MODEL` 和 `CAPSULE_AGENT_PLANNER_MAX_OUTPUT_TOKENS` 配置。

模型只会收到治理后的白名单投影：会话摘要/主题和消息、冻结的记忆召回、近期工具
结果、当前工具目录与已披露 Schema，以及已授予权限。checkpoint 控制字段、待执行
动作和内部请求标识不会进入提示词。`PlanDecision` 仍会经过图路由及 `ToolRegistry`
的服务端校验，因此模型既不能自行授权，也不能直接执行副作用。

无 Ark 密钥时，或测试显式传入自定义 `AgentRuntime` 时，不替换其 Planner；默认
`ReadyPlanner` 保留为安全降级。模型请求失败会转换为 `planner_unavailable`，图以
“对话规划服务暂不可用，请稍后重试。”结束本轮，而不会尝试执行工具。

## 节点与路由

当前图由以下节点组成：

```text
START
  → prepare
      ├─ 已取消 / 超步数 ────────────────> finalize → END
      ├─ 已确认的 pending_action ───────> execute_tools
      └─ 普通回合 ──────────────────────> load_context
                                              → manage_context
                                                  ├─ 失败 ─> finalize → END
                                                  └─ plan
                                                       ├─ select_tools → load_tool_details → manage_context
                                                       ├─ tool → execute_tools
                                                       ├─ confirm → await_confirmation → END
                                                       ├─ 未披露调用 → reject_undisclosed_tools → finalize → END
                                                       └─ respond / finish → finalize → END

execute_tools
  ├─ awaiting_confirmation ─────────────> END
  ├─ cancelled ─────────────────────────> finalize → END
  └─ 有工具结果 ───────────────────────> prepare → load_context → … → plan
```

`finalize` 只组织最终用户回复并更新图内消息。它不写长期/通用记忆；记忆整理始终由异步 Worker 接管。

## 状态契约

核心 `AgentState` 字段：

| 字段 | 语义 |
| --- | --- |
| `thread_id`、`turn_id`、`request_id` | 会话、单次完整输出回合、客户端重试身份。 |
| `messages`、`working_context` | 热镜像的短期会话投影与水位。 |
| `memory_context`、`memory_context_request_id` | 一次请求冻结的长期/通用召回。 |
| `tool_catalog`、`tool_details`、`deferred_tool_names` | 工具渐进披露状态。 |
| `plan` | 当前 `PlanDecision` 的 JSON 投影。 |
| `pending_action`、`approved_action` | 暂停确认与已确认但尚未执行的受控动作。 |
| `pending_tool_calls`、`tool_history`、`last_tool_results` | 工具队列、近期历史和当前输出。 |
| `context_budget` | Token 计量、治理轮数与是否达标。 |
| `status`、`error`、`step_count` | 流程终态、诊断与循环上限。 |

`turn_id` 覆盖一个完整用户输出，内部工具循环不改变它；确认恢复沿用原 `turn_id`。`request_id` 用于一次 HTTP 请求及数据库消息幂等，确认恢复是新的请求，因此通常有新的 `request_id`。

`graph_id` 不是会话或素材标识，而是用户在当前工作区选定的 `NarrativeGraph.graph_id`。
若未创建或未选择图谱，模型仍可自然语言回复；图谱工具在服务端领域校验处返回“必须先选择图谱”，不会回退到任意素材。

## 普通文本回复

1. Runtime 获取回合租约，读取 checkpoint，生成 `turn_id` 与 `request_id`。
2. Runtime 以幂等方式持久化用户消息，并按需要重建热镜像。
3. `prepare` 追加本轮输入、清理新回合遗留队列、填充轻量工具目录。
4. `load_context` 同步召回一次长期/通用记忆。
5. `manage_context` 在输入过大时治理投影。
6. `plan` 返回 `respond` 或 `finish`；`finalize` 形成助手消息。
7. Runtime 持久化助手消息，推进热镜像水位，释放回合租约。

## 工具回合

```text
plan(select_tools)
  → load_tool_details
  → manage_context
  → plan(tool)
  → execute_tools
  → prepare（input_message 已清空）
  → load_context（复用冻结的记忆）
  → manage_context → plan(respond) → finalize
```

这使规划器能先按描述选择工具，再根据精确 Schema 构造参数。若工具结果需要多步处理，循环继续，但 `max_steps`（请求默认 8）会阻止无限回合。

## 确认恢复

确认回合有特殊路由：

1. `await_confirmation` 或 Registry 的确认保护将完整待执行计划保存为 `pending_action`，图暂停。
2. 下一次请求携带 `confirmation=true`；Runtime 从 checkpoint 识别恢复语义，沿用原 `turn_id`。
3. `prepare` 将 `pending_action` 转成 `approved_action`。
4. 路由立即进入 `execute_tools`；不先做记忆召回、上下文治理或模型规划。
5. Registry 仍重新执行服务端参数、权限、确认状态、幂等和锁校验。
6. 工具完成后才回到普通 `prepare -> load_context -> manage_context -> plan` 路径，生成自然语言结果。

恢复时将已存在的 `memory_context` 标记为本次请求已冻结，避免后台记忆写入在确认执行前后改变原计划依据。

## 终态与错误

| `status` | 产生条件 | 后续行为 |
| --- | --- | --- |
| `completed` | 已形成最终回复 | 持久化助手消息。 |
| `awaiting_confirmation` | 计划或 Registry 要求确认 | 保存 checkpoint，等待下一请求。 |
| `cancelled` | 用户拒绝/取消或图检测到取消 | 取消未开始调用并形成取消回复。 |
| `failed` | 上下文不可压缩、工具未披露、未知工具选择或 Planner 不可用 | 形成明确失败回复，不执行后续操作。 |
| `max_steps` | 节点步数超过请求上限 | 形成最大步数回复，停止循环。 |

工具失败本身通常作为结构化 `ToolExecutionResult` 返回给后续规划器；是否终止由规划器和图状态决定。权限、参数或确认拒绝不会被模型绕过。

## 扩展规则

新增节点或路由前先回答：

1. 状态是否能序列化进 PostgreSQL checkpoint，且跨 Runtime 恢复后语义不变？
2. 节点会不会写业务事实？若会，是否应改由数据库仓储或异步 Worker 提交？
3. 是否会使同一 `request_id` 的记忆重新召回，或意外泄露完整工具 Schema？
4. 被取消、超时、重试、确认恢复或进程重启时，是否仍保持幂等？
5. 需要在何处增加图级和 PostgreSQL checkpoint 集成测试？

不要为了增加新能力把外部副作用放进 `plan` 或 `finalize`。规划应保持纯决策；副作用应落在经过服务端校验的工具、仓储或 Worker 中。

## 测试入口

- `tests/test_agent_framework.py`：图路由、渐进工具披露、确认恢复、取消和回合租约调用。
- `tests/test_agent_model_planner.py`：模型输入白名单、远端模型失败与 Runtime 重新装配。
- `tests/test_agent_context_budget.py`：治理边界和不可压缩失败。
- `tests/test_agent_postgres_checkpoint_integration.py`：跨 Runtime 恢复待确认动作。
- `tests/test_agent_conversation_management_integration.py`：真实 PostgreSQL 会话与租约。
