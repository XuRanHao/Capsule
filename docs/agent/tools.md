# Agent 工具系统

## 目标与边界

工具系统允许规划器提出受限的操作请求，但模型从不直接获得执行权。`ToolRegistry` 是唯一执行入口，负责参数验证、权限、用户确认、幂等、并发互斥、超时、重试和审计。

工具调用与会话消息不同：消息属于可见对话历史；工具调用的权威审计记录属于 `agent_tool_executions`。不要用工具审计表重建聊天记录，也不要用聊天历史判断一个操作是否已经完成。

## 工具定义

每个 `AgentTool` 至少包含名称、描述、Pydantic 入参模型和处理器。可选字段定义了风险和执行策略：

| 字段 | 作用 |
| --- | --- |
| `required_permission` | 服务端根据工作区成员权限校验；不匹配时返回 `permission_denied`。 |
| `requires_confirmation` | 未确认时不运行处理器，持久化为待确认操作。 |
| `validate_input` | 在 Pydantic 入参校验后执行的领域校验器。 |
| `output_schema`、`max_output_bytes` | 限定输出结构与体积。 |
| `timeout_seconds`、`max_attempts`、重试参数 | 控制单次执行及可重试错误。 |
| `concurrency_mode`、`lock_scope` | 控制工具自身的串行/并行及锁键范围。 |

注册时会校验工具名唯一、超时和重试参数合法、Schema 是 Pydantic 模型、锁模式有效。工具名是模型唯一可选择的标识，处理器函数不会暴露给模型。

## 图谱工具的生产装配

正常 API 启动会创建 `RelationGraphRepository` 与持久化的
`AgentToolExecutionRepository`，并通过 `build_graph_tool_registry` 装配到默认
`AgentRuntime`。当前目录包含 11 个图谱工具：4 个读取工具和 7 个写入或结构调整
工具。模型收到的 `tools.catalog` 来自这个 Registry；空 Runtime 只允许在单元测试或
显式轻量注入时使用，不能作为生产应用装配的替代。

图谱工具的 `ToolContext.graph_id` 必须是 `narrative_graphs.graph_id`。前端的素材
`asset_id` 只用于素材浏览，不能作为图谱 ID。工作台通过“新建图谱”创建空图谱并将
返回的真实 ID 注入会话请求；图谱仍为空时，Agent 可以读取其上下文或创建实体，但
不会自动把当前查看的素材写入图谱。

工具执行还要求用户在 `workspace_users` 中具有对应工作区权限。图谱创建不改变成员
权限，也不会为了演示界面自动授予 `graph:write` 或更高权限。

## 渐进式披露与规划合约

规划器在首次调用时只收到轻量目录：

```json
{
  "name": "asset_update",
  "description": "更新素材元数据",
  "requires_confirmation": true,
  "required_permission": "asset:write"
}
```

它必须先输出 `PlanDecision(action="select_tools")`：

```json
{
  "action": "select_tools",
  "selected_tool_names": ["asset_update"]
}
```

图中的 `load_tool_details` 节点随后返回完整定义，包括 `args_schema`、`output_schema`、输出上限、重试和锁策略。只有之后的规划调用，才能输出 `tool` 或 `confirm` 及 `tool_calls`。

| `PlanDecision.action` | 允许的字段 | 图行为 |
| --- | --- | --- |
| `respond` / `finish` | 可选 `message`，不能携带选择或调用 | 进入最终回复。 |
| `select_tools` | 至少一个 `selected_tool_names`，不能携带调用 | 披露这些工具的完整 Schema 后重新规划。 |
| `tool` | 至少一个 `tool_calls`，不能携带选择 | 要求每个调用都已有完整 Schema，然后交给 Registry。 |
| `confirm` | 至少一个 `tool_calls`，可选确认提示 | 要求完整 Schema，保存待确认动作并暂停。 |

`PlanDecision` 的结构校验不把“已披露”写进模型，因为这是图状态条件；`route_after_plan` 会在服务端再次检查 `tool_details`。若工具未被披露，图以 `failed` 结束，不会自动替模型补选工具。

```text
目录
  → select_tools
  → 完整 Schema
  → tool / confirm
  → 服务端校验与执行
```

未知工具的选择同样失败，不会返回空 Schema 后继续规划。

## 执行顺序

每个 `ToolCall` 进入 `ToolRegistry.execute` 后按以下顺序处理：

1. 创建或读取持久化操作记录，建立 `operation_id` 和审计边界。
2. 解析 Pydantic 入参；失败返回 `invalid_arguments`。
3. 运行可选的领域入参校验器；失败同样返回结构化错误。
4. 运行前置 Hook；Hook 可拦截并返回受控结果。
5. 校验 `required_permission`。
6. 若同一幂等操作已成功，直接返回既有输出。
7. 若需要确认而本次未确认，标记 `awaiting_confirmation` 并返回 `needs_confirmation=true`，绝不调用处理器。
8. 领取操作执行权；重复中的操作返回 `operation_in_progress`。
9. 按工具并发策略取得本地或分布式锁，运行超时与重试控制下的处理器。
10. 运行后置 Hook，完成审计记录，并返回 `ToolExecutionResult`。

`ToolExecutionResult` 始终带有 `call_id`、`operation_id`、`name`、`ok`、`attempts`；失败时带 `error_code` 和受限错误文本。图只把结构化结果写入短期上下文，完整审计不因上下文压缩而丢失。

## 确认、取消与恢复

确认有两道防线：

- 规划器可主动返回 `confirm`，在调用前向用户展示确认提示。
- 即使规划器错误地直接返回 `tool`，Registry 仍会阻止 `requires_confirmation=true` 的处理器，改为持久化待确认操作。

用户确认时，Runtime 从 checkpoint 中取得 `pending_action`，将其变为受确认的 `tool` 动作，**直接交给 Registry 校验和执行**。执行前不再调用规划器，也不重新披露 Schema。工具完成后，图才进入普通规划节点，以根据结果组织最终回复。

用户拒绝、取消或发起新的普通回合时，尚未运行的调用会标记为 `cancelled`；运行中的处理器仅支持协作式取消，处理器应通过 `ToolContext.raise_if_cancelled()` 主动退出。

## 新增工具的最小示例

```python
class RenameArgs(BaseModel):
    asset_id: str
    title: str

registry.register(
    AgentTool(
        name="rename_asset",
        description="修改一个素材的显示标题",
        args_schema=RenameArgs,
        handler=rename_asset,
        required_permission="asset:write",
        requires_confirmation=True,
        concurrency_mode="exclusive",
        lock_scope="asset",
    )
)
```

新增工具时必须同时完成：

1. 明确权限和是否产生外部/不可逆影响；默认倾向于要求确认。
2. 让 Pydantic Schema 限定必要字段、类型和长度，不把自由文本直接交给处理器。
3. 设计稳定的幂等键和锁键，特别是写同一素材、实体或图时。
4. 为参数错误、权限拒绝、超时、重试、确认恢复和取消增加测试。
5. 确认描述足够让模型从轻量目录中正确选择，不依赖完整 Schema 才能理解作用。

## 禁止事项

- 不要根据模型说“已确认”就跳过 Registry 的 `confirmed` 参数。
- 不要在规划器中直接访问数据库、网络或处理器。
- 不要把完整工具目录的 Schema 永久放入每轮上下文。
- 不要把工具原始输出作为唯一会话记录；审计必须落库。
- 不要把业务回合租约当作工具实体锁。前者串行会话，后者保护资源操作，两者粒度不同。

## 验证入口

基础行为位于 `tests/test_agent_framework.py`；持久化操作、租约和 API 行为应结合 `tests/test_agent_postgres_checkpoint_integration.py` 与相关工具执行测试验证。
