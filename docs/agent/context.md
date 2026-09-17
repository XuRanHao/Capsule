# Agent 上下文预算

## 目标

上下文预算控制的是“即将送给规划器的投影”，不是数据清理机制。它可以用已提交的会话摘要替代旧消息、把工具结果缩成结构化预览、延后完整 Schema；它不会删除 PostgreSQL 原始消息、工具审计或长期/通用记忆。

治理节点固定放在图中的 `load_context -> manage_context -> plan` 边界，并在工具 Schema 披露后再次经过同一节点。

## 默认预算

| 配置 | 默认值 | 含义 |
| --- | ---: | --- |
| `agent_context_window_tokens` | 32,000 | 规划模型总窗口。 |
| `agent_context_output_reserve_tokens` | 4,000 | 保留给模型输出，不能用于输入。 |
| 有效输入上限 | 28,000 | `window - output_reserve`。 |
| `agent_context_short_term_ratio` | 0.30 | 超窗时短期会话触发摘要的占比线，默认 8,400 Token。 |
| `agent_context_tool_result_ratio` | 0.50 | 超窗时工具结果触发压缩的占比线，默认 14,000 Token。 |
| `agent_context_raw_tail_tokens` | 4,000 | 摘要后优先保留的最新原始消息尾部。 |
| `agent_context_max_reduction_rounds` | 3 | 最多治理循环次数。 |
| `agent_context_summary_wait_seconds` | 12 | 等待短期摘要水位的最长时间。 |

30% 与 50% 不是每轮固定分区。系统先检查整体 Token；总量不超窗时直接放行。它们只决定超窗后哪一类上下文有资格被处理。

## 计量对象

`ContextTokenUsage` 会记录并写入 checkpoint：

| 类别 | 内容 | 是否允许压缩 |
| --- | --- | --- |
| 短期 | 摘要、主题、未被摘要的原始消息 | 是；只整理旧前缀，保留最新尾部。 |
| 工具结果 | 当前/近期工具的上下文投影 | 是；原始审计不变。 |
| 长期 | 工作区与通用记忆召回 | 否。 |
| 工作上下文 | 侧写、水位、运行元数据 | 不作为常规压缩目标。 |
| 工具定义 | 轻量目录和已选择工具的完整 Schema | 不截断 Schema，只能整条延后。 |

Token 是确定性近似估计，用于预算决策而不是账单统计。

## 治理顺序

```text
组装候选上下文
  │
  ├─ 总量 ≤ 输入上限 ───────────────────────────> 直接规划
  │
  └─ 总量超限
       ├─ 短期 > 短期阈值：投递摘要任务并等待摘要水位
       ├─ 工具结果 > 工具阈值：生成结构化结果预览
       └─ 重新计量
             ├─ 已达标：规划
             ├─ 还有轮次：重复，最多 3 次
             └─ 仍超限：确定性整条裁剪或明确失败
```

### 短期会话压缩

1. 控制器在完整预检后选择一个旧消息前缀的 `through_sequence`，并保留接近 `raw_tail_tokens` 的最新消息。
2. 在同一 PostgreSQL 事务中写入/复用 `agent_memory_outbox` 事件；API 可立即发布到 Redis Stream，失败时由 Outbox Dispatcher 补发。
3. Runtime 仅等待 Worker 将 `summary_covered_sequence` 推进到目标水位。
4. Worker 先提交摘要和主题；控制器重新从数据库读取“摘要 + 水位后的原始尾部”。
5. Worker 之后继续长期/通用记忆提炼；本轮不会重新召回新写入的长期记忆。

等待只针对短期摘要提交，不等待完整记忆整理，因此可恢复上下文又不让用户请求等待长期抽取。

### 工具结果压缩

结果投影保留调用标识、工具名、成功/失败状态、错误码、关键字段和受限预览；长字符串、嵌套列表和大对象会被缩小。完整输出仍保存在 `agent_tool_executions` 的审计记录中。

### 工具定义处理

工具目录始终保持轻量。完整定义只在 `select_tools` 后出现；若预算硬裁剪需要腾出空间，完整 Schema 会整条从 `tool_details` 移到 `deferred_tool_names`，不会截断 JSON Schema。规划器必须重新选择该工具后才能再次获得 Schema。

## 硬裁剪与失败

治理最多循环三次。仍超窗时，按以下顺序移除整个可延后项目：

1. 最旧工具结果投影；
2. 最旧的非当前会话消息；
3. 非当前调用所需的完整工具定义。

当前用户输入、已召回的长期/通用记忆和单条已选择 Schema 不被半截裁剪。若这些不可压缩内容本身已超过输入上限，图以明确的 `failed` 状态结束；绝不把超窗请求发送给模型。

## 与记忆、会话和循环的关系

- `agent_messages` 原文保持追加式；摘要水位只改变运行时读取范围。
- 同一 `request_id` 的长期/通用记忆只召回一次。工具循环、Schema 披露和确认恢复不会让后台 Worker 改变该次判断。
- 确认恢复会复用 checkpoint 中冻结的记忆上下文，先直接执行待确认动作；工具结果出来后才进入后续回复规划。
- `context_budget` 是 checkpoint 状态的一部分，便于诊断 Token 构成、治理轮数和失败原因。

## 调参与排障

先按真实规划模型的窗口调整 `window_tokens` 与输出预留，再评估短期和工具结果比例。不要仅提高总窗口来掩盖工具输出或历史消息无界增长。

排查超窗失败时依次查看：

1. checkpoint 中的 `context_budget` 计量；
2. `working_context.summary_covered_sequence` 与 `agent_threads` 水位；
3. `agent_memory_outbox` 是否已投递、Worker 是否完成摘要第一阶段；
4. `deferred_tool_names` 是否说明 Schema 被延后；
5. 工具审计中的原始输出是否异常膨胀。

对应单元测试在 `tests/test_agent_context_budget.py`，图级回归在 `tests/test_agent_framework.py`。
