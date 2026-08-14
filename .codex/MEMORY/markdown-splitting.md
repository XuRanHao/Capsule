# Markdown 切分方案

## 目标

将一个 Markdown Source File 稳定、可复现地转换成多个语义完整的
`markdown_block` Asset。第一版使用确定性规则，不调用理解模型参与切分；
方舟 Tokenization API 只负责精确计算 token 数。

## 总体流程

```text
Markdown 原文
  → 解析结构化节点
  → 构建 Heading 层级树
  → 按标题层级递归切分
  → 超长叶子章节按内容节点切分
  → 合并相邻短块
  → 生成 markdown_block Asset
```

结构化节点包括 Heading、Paragraph、List、Blockquote、Code Fence、Table、
Image Reference、Link 和 Horizontal Rule。节点是中间结构，不直接等同于 Asset。

## 标题递归切分

1. 先按一级标题形成章节。
2. 章节不超过 400 tokens 时整体保留。
3. 超过 400 tokens 时，按下一级标题继续递归切分。
4. H1 到 H6 使用相同规则，不将实现写死在 H1/H2。
5. 没有更低级标题且仍超长时，才按内容节点切分。
6. Heading 用于维护 `heading_path`，没有正文的标题不单独生成 Asset。
7. 标题之前的正文属于虚拟根章节，`heading_path=[]`。

## 长度规则

- 目标长度：100～300 tokens。
- 允许范围：50～400 tokens。
- 只有超过 400 tokens 时才继续拆分。
- 100～300 是拆分和合并时的目标，不是强制边界。
- 不使用固定字符数，也不从句子中间截断。
- 语义完整性和不可拆分项优先于长度限制。

## 内容节点规则

### Paragraph

章节已经无法按标题切分且仍超过 400 tokens 时，Paragraph 可以在句子边界切分，
禁止从句子中间截断。

### Code Fence

代码块是不可拆分原子节点。即使自身超过 400 tokens，也必须保持完整并独立输出。

### Table

完整表格是不可拆分原子节点。即使自身超过 400 tokens，也必须保持完整并独立输出。

### List

- 列表整体不超过 400 tokens 时完整保留，不拆开列表项。
- 列表整体超过 400 tokens 时，只能在顶层列表项之间分组。
- 嵌套列表跟随所属顶层列表项。
- 分组目标为 100～300 tokens，每组最多 400 tokens。
- 单个顶层列表项自身超过 400 tokens 时保持完整，并标记为超长。

### 其他节点

- Blockquote 优先保持完整，必要时按其内部段落边界处理。
- Horizontal Rule 是强边界，本身不生成 Asset。
- Image Reference 保留引用和附近文本上下文，不单独生成文本 Asset。

## 相邻短块合并

只有同时满足以下条件时才合并：

- 内容相邻；
- 属于同一个父标题；
- 中间没有 Horizontal Rule 等强边界；
- 合并后不超过 400 tokens；
- 不破坏代码块、表格和列表的完整性。

无法安全合并时允许保留不足 50 tokens 的独立 Block，不为了凑长度跨无关章节合并。

## 超长原子节点

不可拆分节点超过限制时，在 `file_info` 中记录：

```json
{
  "token_count": 1850,
  "oversized": true,
  "oversized_reason": "indivisible_code_fence"
}
```

解析阶段不截断超长原子节点；后续模型调用是否摘要、跳过或采用其他输入方式另行设计。

## Token 计算方案

使用方舟官方 `/api/v3/tokenization`，模型为
`doubao-embedding-vision-250615`。已经过实际验证：接口返回 HTTP 200，响应包含
`total_tokens`、`token_ids` 和 `offset_mapping`。

不采用“每判断一个候选块就立即请求一次”的方式。批量流程为：

```text
收集待计数文本
  → 按 model_name + sha256(text) 查缓存
  → 对未命中文本批量调用 Tokenization
  → 按响应 index 回填计数与偏移
  → 本地执行递归切分和合并
```

- 第一阶段使用单次任务内存缓存。
- 接入 PostgreSQL 后再评估持久化缓存。
- 批量大小需要通过接口测试确定，当前不写死。
- `offset_mapping` 可辅助长 Paragraph 在句子边界定位。

## Asset 输出

每个最终 Block 至少输出：

```json
{
  "asset_type": "markdown_block",
  "source_locator": {
    "block_index": 0,
    "heading_path": ["第一章", "安装说明"],
    "char_start": 120,
    "char_end": 860
  },
  "raw_content": "## 安装说明\n\n原始 Markdown 片段……",
  "file_info": {
    "token_count": 320,
    "oversized": false
  }
}
```

`char_start` 和 `char_end` 必须基于原始 Markdown，而不是清洗或重新拼接后的文本。
`raw_content` 直接保存该范围对应的原始 Markdown 字符串；提取后的纯文本只作为 Tokenization
或模型理解的临时输入，不在 Asset 中重复保存。

## 尚待确定

- Tokenization API 的安全批量上限、限流表现和失败重试策略。
- Markdown 图片引用如何与独立图片 Asset 做路径归一化和上下文合并。
