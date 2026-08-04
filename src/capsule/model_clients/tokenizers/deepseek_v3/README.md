# DeepSeek V3 tokenizer

Capsule uses `tokenizer.json` directly through Hugging Face `tokenizers` for
offline document chunk sizing. Raw document text is encoded with
`add_special_tokens=False`; model weights and remote code are not loaded.

Bundled file checksums:

- `tokenizer.json`: `ecb6f9fc369894346f0511f4074ca75cee5cd5f3b06d02f1ba35fcd39f8e121d`
- `tokenizer_config.json`: `144a6d92b6012baeb4f2ac41d48ed3458e758f977a0fb5caf75ff07698fc844c`
