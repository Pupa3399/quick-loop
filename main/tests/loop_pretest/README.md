# Loop 预实验

本实验使用人工筛选的 20 个 Search-R1 Qwen2.5-3B 决策点，比较目标 turn 使用 R1 与
R2 时的后续完整 Agent 轨迹。这里只编写与保存实验代码；正式执行需要另行启动。

## 输入

输入文件：

`main/data/qwen_prefilt/search_r1_qwen3b_decision_turns_human_20.jsonl`

20 条样本均满足：

- `human_annotation.decision=keep`
- 0-based `target_turn=1`
- `prefix_history` 恰好包含第 0 轮
- 原始目标 action 为 Search
- 10 条 `over_search`，10 条 `search_stagnation`

筛选文件没有保存原始标签周围的空白格式，因此 prefix 会从 `thinking`、`query` 和
`snippets` 字段重建成规范 Search-R1 格式。模型生成内容不会被重写。

## 对照设计

每个决策点使用 4 个配对 seed，每个 seed 分别生成一条 R1 和一条 R2 轨迹：

- R1：目标 turn 的所有生成 token 使用 depth 1。
- R2：目标 turn 的所有生成 token 使用 depth 2。
- 目标 turn 之后的模型生成统一恢复 depth 1。
- R1/R2 使用相同 prefix、Retriever、采样参数和配对 seed。

总计生成：

```text
20 个决策点 × 2 个 depth × 4 条轨迹 = 160 条轨迹
```

采样参数与现有 Search GRPO rollout 对齐为 `temperature=1.0`、`top_p=1.0`。每轮最多
生成 500 token，模型生成与 Retriever observation 的累计 budget 为 3000 token，最多
进行 4 次 Search（包含固定 prefix 中已经发生的 Search）。

## R2 与缓存

R2 使用 top-100 logits 的 FP32 softmax，通过 `embed_tokens.weight` 得到 soft embedding，
在同一 logical position 再执行一次 Qwen，最终直接从 z2 采样。

目标 turn 与后续 turn 共用同一个 `LatentLoopState`。Retriever observation 通过
`append_visible_tokens()` 追加到现有 cache，不重新 prefill，因此目标轮写入的
`latest_depth_kv` 会继续影响后续 Agent 轮。

## 运行方式

先启动 Wiki18 + E5 + GPU FAISS Retriever，再在另一张 GPU 上执行：

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python main/tests/loop_pretest/run.py
```

脚本使用本地模型与离线 Hugging Face 模式，不会下载权重。运行前会检查 Retriever
确实使用 GPU FAISS，且索引规模为 21,015,324。

## 输出

结果保存于 `main/tests/loop_pretest/results/`：

- `r1_trajectories.json`：80 条 R1 分支轨迹
- `r2_trajectories.json`：80 条 R2 分支轨迹
- `summary.json`：轮次数、Search 次数、answer rate、EM 和终止原因汇总

每条轨迹明确记录：

- `target_turn` 与 `target_depth`
- 配对采样 seed
- 完整 prefix 与后续 Agent turns
- 每一轮的 `loop_depth`、完整模型输出、Search query 和 Retriever information
- `num_agent_turns`、`num_generated_agent_turns`、`num_search_turns`
- 目标 turn 的逐 token R1/R2 top-1 诊断
- 最终答案、EM 与终止原因
