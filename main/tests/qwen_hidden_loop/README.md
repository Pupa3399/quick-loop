# Qwen Hidden-State Residual Loop 测试

本目录验证 `QwenHiddenLoopEngine` 的核心不变量：

- 纯 R1 输出与 Hugging Face 标准 greedy generation 一致；
- `alpha=0` 时 R2 与 R1 logits 数值一致；
- R1/R2 使用相同 logical position；
- R2 只令 KV cache 长度增加 1；
- 当前位置最终保留第二次 forward 的 KV；
- backend 没有增加可训练参数；
- `depth_by_token` 使用 0-based token index。

`smoke.py` 使用本地 Search-R1 Qwen2.5-3B，在此前发生 soft-token phase shift 的同一个
`<answer>` 边界上比较 R1、旧 soft-token R2 和新 hidden-state R2。它只生成一个 token，
不启动 Retriever，也不下载模型。

```bash
uv run --no-sync pytest main/tests/qwen_hidden_loop/test_engine.py
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python main/tests/qwen_hidden_loop/smoke.py
```

smoke 结果保存到 `results/single_token_smoke.json`。
