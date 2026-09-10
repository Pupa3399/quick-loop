# Search-R1 v0.2 Baselines

## Immutable Artifacts

| Model | Revision | Local path | Parameters | Weight storage |
| --- | --- | --- | ---: | --- |
| Search-R1 Qwen2.5-3B | `7ff32234abe1e1bf1dbbbf3a1385686b52ecfea2` | `/data2/wuguanting/quick_loop/data/models/searchr1-qwen2.5-3b-grpo-v0.2` | 3,397,103,616 | FP32 weights 13,588,414,464 bytes; complete directory 13,604,381,044 bytes (13G `du`) |
| Search-R1 Qwen2.5-7B | `deaa9d14b92dd4d481414e77b9f733c934df830a` | `/data2/wuguanting/quick_loop/data/models/searchr1-qwen2.5-7b-grpo-v0.2` | 7,615,616,512 | FP32 weights 30,462,466,048 bytes; complete directory 30,478,413,878 bytes (29G `du`) |

The model indexes report FP32 checkpoint storage. Evaluation explicitly loads both models
as BF16. `scripts/download_searchr1_baselines.py` refuses any `HF_ENDPOINT` other than
`https://hf-mirror.com`, pins both revisions, and has no overseas fallback.

## Unified Evaluation Matrix

All four targets run through `scripts/evaluate_agent.py`; only Hydra's `model=` selection
changes:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 \
  uv run --no-sync python scripts/evaluate_agent.py \
  model=searchr1_qwen2_5_3b_grpo_v0_2
```

The other model selections are `searchr1_qwen2_5_7b_grpo_v0_2`, `ouro_2_6b_r3`, and
`ouro_2_6b_r4`. The entry point always uses the shared vLLM engine, HTTP Retriever client,
Search-R1 protocol runner, pure final-answer EM reward, and JSONL trajectory writer.

| Setting | Search-R1 3B | Search-R1 7B | Ouro-2.6B | Source of difference |
| --- | --- | --- | --- | --- |
| Backbone | Qwen2.5-3B | Qwen2.5-7B | Ouro-2.6B-Thinking | Model/checkpoint |
| Parameters | 3,397,103,616 | 7,615,616,512 | 2,667,974,657 | Model/checkpoint |
| Search GRPO | Released v0.2 NQ + HotpotQA outcome-only GRPO | Released v0.2 NQ + HotpotQA outcome-only GRPO | Project outcome-only GRPO pipeline; long-training checkpoint pending | Model/checkpoint |
| Reward | Final-answer EM only | Final-answer EM only | Final-answer EM only | Shared experiment setting |
| User prompt | Search-R1 v0.2 prompt | Search-R1 v0.2 prompt | Search-R1 v0.2 prompt | Shared experiment setting |
| Tokenizer template | Native Qwen2 chat template | Native Qwen2 chat template | Native Ouro chat template | Model/checkpoint |
| Recurrent depth | N/A | N/A | Fixed R3 or R4 config | Model architecture/config |
| Evaluation decoding | BF16, greedy, temperature 0, top-p 1 | Same | Same | Shared experiment setting |
| Per-turn generation | 500 tokens | 500 tokens | 500 tokens | Shared Search-R1 setting |
| Total response budget | 3000 tokens | 3000 tokens | 3000 tokens | Shared veRL aggregation limit |
| Search budget | At most 4 actions | At most 4 actions | At most 4 actions | Shared Search-R1 setting |
| Information budget | 500 tokens per result block | Same | Same | Shared Search-R1 setting |
| Retriever | E5 + Wiki18 + FAISS Flat, top-k=3, HTTP | Same | Same | Shared experiment setting |

The 500-token Search-R1 value is the cap for one model generation. The current 3000-token
value is the aggregate multi-turn response buffer containing generations and inserted
information; it is not an expansion of the per-turn decoding cap. Four searches plus a
final answer cannot be represented safely in a single 500-token aggregate buffer, so the
3000-token buffer is retained and must not be increased without measured truncation data.

## Runtime Acceptance

| Model | BF16 vLLM / ordinary generation | KV cache | GPU allocation | Real Search-R1 trajectory |
| --- | --- | --- | --- | --- |
| Search-R1 3B | Pass; `The answer is four.` in 0.102s | Pass; 20.30 GiB, 591,376 tokens, 83.25x max concurrency at 7,096 tokens | 33,689 MiB sampled peak; 5.7918 GiB weights + 0.52 GiB CUDA graphs | Pass; natural 1-search action, top-3 HTTP retrieval, information continuation, legal answer in 12.775s; 550 response tokens; `outputs/trajectories/searchr1-3b-acceptance.jsonl` |
| Search-R1 7B | Pass; nonempty response beginning `Two plus two is four.` in 0.754s | Pass; 11.77 GiB, 220,336 tokens, 31.02x max concurrency at 7,096 tokens | 33,705 MiB sampled peak; 14.2717 GiB weights + 0.46 GiB CUDA graphs | Pass; natural 1-search action, top-3 HTTP retrieval, information continuation, legal answer in 20.919s; 914 response tokens; `outputs/trajectories/searchr1-7b-acceptance.jsonl` |

The 3B trajectory searched for the filming location, retrieved three entries from *The
Flight of the Phoenix (1965 film)*, then emitted `<answer> California </answer>`. The
protocol path is valid, though the answer is not the reference and therefore correctly
receives EM 0.

The 7B trajectory issued the same natural search, consumed the same top-3 result source,
and emitted `<answer> Pilot Knob Mesa </answer>`. It also receives EM 0 against the fixed
`20th Century-Fox Studios` reference. Both runs used the released native Qwen chat
template and completed without NaN, Inf, or OOM.

## Stage-Three Constraint

The Search-R1 releases are comparison-ready. The Ouro model configs currently point to
fixed-depth views of the official base snapshot. The completed 20-step Ouro checkpoint is
an integration acceptance artifact, not a converged comparison checkpoint, and remains
in veRL's sharded format. A long-trained Ouro checkpoint must be produced and selected in
the R3/R4 model configs before a model-quality baseline comparison is meaningful; this
task intentionally does not start that long RL run.
