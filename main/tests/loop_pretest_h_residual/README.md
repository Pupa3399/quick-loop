# Hidden-State Residual Loop 预实验

本实验在人工筛选的 20 个 Search-R1 Qwen2.5-3B 决策点上，对比目标 turn 使用标准 R1
和 Hidden-State Residual R2 后的完整 Agent 轨迹。

## 固定条件

- 输入：`main/data/qwen_prefilt/search_r1_qwen3b_decision_turns_human_20.jsonl`
- 20 个样本，0-based `target_turn=1`
- 每个样本、每个 depth 采样 4 条轨迹
- R1/R2 使用配对 seed 42 至 121
- `temperature=1.0`、`top_p=1.0`
- 每轮最多 500 token，总 response budget 3000 token
- 最多 4 次 Search，包含固定 prefix 已发生的 Search
- Wiki18 + E5 + GPU FAISS Flat，`top_k=3`
- R2 仅用于目标 turn，后续生成恢复 R1

总计：

```text
20 × 2 depths × 4 draws = 160 条轨迹
```

## R2 机制

```text
h1 = Qwen(x)
scaled_h1 = h1 * RMS(x) / (RMS(h1) + eps)
x2 = x + alpha * scaled_h1
z2 = lm_head(Qwen(x2))
```

本实验固定 `alpha=0.5`、`eps=1e-6`。RMS 使用 FP32。R1/R2 使用相同 position，
当前位置的 R1 KV 被删除并替换为 R2 KV。Retriever observation 追加到同一个 cache，
不会重新 prefill 丢失目标轮的 R2 KV。

## 运行

GPU 0 启动 Retriever，GPU 1 执行模型：

```bash
OURO_RETRIEVER_CUDA_VISIBLE_DEVICES=0 bash main/scripts/launch_e5_retriever.sh
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python main/tests/loop_pretest_h_residual/run.py
```

所有模型和语料均从本地加载，Hugging Face 处于 offline 模式。

## 输出

结果位于 `results/`：

- `r1_trajectories.json`
- `r2_trajectories.json`
- `summary.json`

每条轨迹记录完整模型输出、Search query、Retriever information、Agent/Search 轮数、
目标 turn 的逐 token R1/R2 top-1、logits 差值、RMS 以及最终答案和 EM。

去除 prefix 重建阶段人为换行的对照配置为 `config_no_newlines.yaml`，结果写入
`results_no_added_newlines/`。该配置保留 Prompt 文本和 Retriever 文档内部换行，只去掉
continuation 前后的空行以及 `</think>` 与 `<search>` 之间由重建代码添加的换行。
