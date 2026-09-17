# Search-R1 v0.2 最小复现测试

本目录只验证官方 rollout 的确定性语义，不加载模型，也不启动 Retriever：

- v0.2 Prompt 使用最多四次 Search 的原文；
- `<information>` 的外部双换行、内部 `.strip()` 和 token 边界；
- strict、case-sensitive、first-match parser；
- generation 完成后优先按 `</search>` 裁剪并重新 tokenize；
- 非法 action 的官方纠错 observation；
- rolling context 保留最后 4096 token 的规则；
- sampling 参数不含 closing-tag stop，也不在每个 turn 注入 seed。

执行：

```bash
uv run pytest main/tests/search_r1_v0_2
```

