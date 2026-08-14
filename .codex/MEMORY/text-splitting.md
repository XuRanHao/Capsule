# TXT 切分方案

## 目标

将无结构 UTF-8 `.txt` 文件稳定转换为多个 `text_block` Asset。TXT 没有可靠标题，不尝试推断
`heading_path`，只保留基于原文的 `char_start`、`char_end` 和 `raw_content`。

## 总体流程

```text
TXT 原文
  → 识别空行段落、列表、明显代码块与简单表格
  → 将没有空行的连续换行视为同一自然段（保留原始换行）
  → 按相邻原子块贪心合并
  → 仅在超过 400 tokens 的普通段落按句子边界拆分
  → 生成 text_block Asset
```

## 规则

- 使用方舟 `/tokenization` 批量计算 token；不调用理解模型参与切分。
- 空行是段落边界；没有空行的连续行视为硬换行的同一段，不重写 `raw_content`。
- 明显的 fenced/缩进代码、连续列表、两行及以上的 pipe/tabular 表格先作为原子块。
- 普通段落超过 400 tokens 时只允许在中英文句末标点边界拆分；不能从句子中间截断。
- 列表超过 400 tokens 时仅在顶层列表项之间拆分；单项超过限制则完整保留。
- 代码块和表格永不拆分。任何不可拆分项超长时记录 `oversized=true` 与原因。
- 相邻原子块可在合计不超过 400 tokens 时合并；没有标题层级时不会跨文件或重排块顺序。

## 输出

```json
{
  "asset_type": "text_block",
  "source_locator": {
    "type": "text_range",
    "block_index": 0,
    "char_start": 0,
    "char_end": 824
  },
  "raw_content": "原始 TXT 片段……",
  "file_info": {
    "token_count": 320,
    "node_kinds": ["paragraph"],
    "oversized": false
  }
}
```

## 范围限制

当前方案面向自然语言笔记、文章和导出的说明文本。大量日志、OCR 断行、终端输出或混合代码的
TXT，后续需要结合真实数据决定专门识别策略。
