# Search-R1 R1 重新 Rollout

本实验对人工选中的 20 条 Search-R1 Qwen2.5-3B 样本从问题开始重新 rollout。每题生成
4 条轨迹，只运行原始 R1，不使用任何 Hidden R2 或 latent loop。

## 固定条件

- 模型：`SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo-v0.2`
- 原始 Prompt 从 `main/data/searchr1/train.parquet` 按 `dataset + orig_index` 读取
- Prompt 必须包含 `You can search at most four times.`
- vLLM 0.10.0，BF16，temperature=1.0，top_p=1.0
- 当前服务器缺少 Python 3.10 开发头文件，使用 vLLM eager 模式避免 `torch.compile`
- 每轮最多生成 500 token，response budget 为 3000 token
- Wiki18 + E5 + GPU FAISS Flat，top-k=3
- 最多调用 Retriever 4 次
- stop 为 `</search>` 和 `</answer>`，stop string 保留在输出中

Agent 拼接严格按 token ID 执行：初始 Prompt 通过 chat template 编码；每轮生成后直接
追加 `generated_token_ids`；Search 后直接追加 `information_token_ids`。代码不会插入、删除
或重建任何换行。

每条 trajectory 使用固定 seed。离线 vLLM 的每次生成调用都使用该 trajectory 的同一 seed，
以保证本实验可重复。原始历史 rollout 没有保存 sampler RNG 状态，因此本实验不能复现其
逐 token 随机状态，只复现可确认的模型、Prompt、采样参数、Agent 和 Retriever 环境。

## 运行

GPU 0 启动 Retriever：

```bash
OURO_RETRIEVER_CUDA_VISIBLE_DEVICES=0 bash main/scripts/launch_e5_retriever.sh
```

GPU 1 运行模型：

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python main/tests/rerollout/run.py
```

## 输出

- `results/trajectories.json`：80 条完整轨迹
- `results/summary.json`：Search/Answer、目标 action 复现率和 information 边界统计

每轮保存生成前的完整 prefix 文本和 token IDs、原始生成文本和 token IDs、Retriever
information 文本和 token IDs，以及解析后的 thinking、query、answer 和 action。
