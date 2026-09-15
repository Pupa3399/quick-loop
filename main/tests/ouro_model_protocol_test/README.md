# Ouro 模型与协议指令遵循测试

本实验在不训练、不更新参数的前提下，对比以下四组组合：

- Ouro Base + Search-R1
- Ouro Base + Hermes
- Ouro Thinking + Search-R1
- Ouro Thinking + Hermes

## 固定样本

`samples.json` 固定为此前 `zero_shot_agent_v1` 正式测试使用的前 10 条 NQ 与前 10 条
HotpotQA，共 20 条。四组必须使用完整、顺序一致的 `sample_id`，运行脚本在合并汇总时会检查这一点。

## 共享推理条件

- Ouro Base revision：`1ed04250da1a9936042725d302e81c8fa2ab5abd`
- Ouro Thinking revision：`f1edd81e7ac41355db670500ceaf204e0f73af68`
- 循环深度：R3
- 解码：greedy，`temperature=0`，`top_p=1`
- 最大 Search 次数：4
- Retriever：Wiki18 + E5 + GPU FAISS Flat，`top_k=3`
- 模型上下文长度：7096 token
- 不设置额外 `max_new_tokens` 或累计 response token 上限
- Retriever observation 不截断

没有额外生成上限并不代表无限生成：模型仍受 7096 token 上下文限制。汇总中的
`length_limited_generation_count` 和 `context_exhausted_sample_count` 用于区分模型主动结束与上下文耗尽。

## 运行

先在一张独立 GPU 上启动真实 Retriever：

```bash
OURO_RETRIEVER_CUDA_VISIBLE_DEVICES=0 bash main/scripts/launch_e5_retriever.sh
```

再在另一张 GPU 上依次运行两种模型；每个模型只加载一次并连续运行两个协议：

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python \
  main/tests/ouro_model_protocol_test/run.py --model base

CUDA_VISIBLE_DEVICES=1 uv run --no-sync python \
  main/tests/ouro_model_protocol_test/run.py --model thinking
```

运行脚本强制检查 Retriever `/health` 中的 GPU FAISS 标记和 21,015,324 条索引规模。
模型目录也会核对各自的固定 revision，避免 Base 与 Thinking 权重混用。

## 输出

输出位于 `results/`：

- `base_search_r1.json`
- `base_hermes.json`
- `thinking_search_r1.json`
- `thinking_hermes.json`
- `summary.json`

四个轨迹文件均为带缩进的 JSON 数组。每条只保存问题、实际初始 prompt、每轮完整原始输出、
实际 Search query、未缩略 observation、最终答案和 EM，便于直接人工阅读。

指标口径如下：

- `format_valid_rate`：一条样本的所有生成轮均通过对应协议的严格格式检查。
- `search_tool_call_rate`：至少产生一次可解析的合法 Search/tool action 的样本比例。
- `retriever_invocation_rate`：至少实际调用一次 Retriever 的样本比例。
- `post_observation_continuation_rate`：得到 observation 后确实产生下一轮输出的比例。
- `post_observation_valid_action_rate`：下一轮输出还能解析成合法 action 的比例。
- `average_search_turns`：每条样本的真实 Retriever 调用次数均值。
- `multi_search_rate`：实际调用 Retriever 至少两次的样本比例。
- `final_answer_rate`：最终输出能由对应协议解析为 answer 的样本比例。
- `exact_match`：解析后的最终答案对固定 aliases 的标准化精确匹配率。
