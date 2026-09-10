# ouro-search

`ouro-search` is a research stack for Search-R1-style reinforcement learning with
`ByteDance/Ouro-2.6B-Thinking`. Ouro is fixed to BF16 and R3 for all RL rollouts. The
official `modeling_ouro.py` and `early_exit_gate` structure are untouched; the real gate
is retained and frozen.

The repository contains an E5 + Wikipedia-2018 + FAISS HTTP retriever, NQ/HotpotQA
Parquet preparation, a vLLM Ouro R3 plugin, a veRL asynchronous search AgentLoop, pure
final-answer EM reward, information-token masking, JSONL trajectories, and GRPO configs.

## Environment And Domestic Mirrors

The tested environment is Python 3.10.12, PyTorch 2.7.1+cu126, Transformers 4.54.1,
vLLM 0.10.0, veRL 0.6.0 at commit
`ddd86f527a4af75095e4677b02b5aa272913a088`, CUDA 12.6 wheels, and 8 x A800 80GB.
No Conda environment is used.

Every download must use these domestic endpoints; scripts reject a different HF endpoint:

```bash
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data2/wuguanting/quick_loop/data/huggingface
export HF_MODULES_CACHE=$HF_HOME/modules

uv sync --extra retriever --extra train --index "$UV_INDEX_URL"
bash scripts/install_train_stack.sh
```

`scripts/install_train_stack.sh` obtains veRL from the Gitee mirror
`https://gitee.com/mirrors/verl.git`, checks out the pinned SHA, and applies
`patches/verl-ouro-r3.patch`. The Python development headers needed by vLLM JIT on this
server were obtained from the Tsinghua Ubuntu mirror
`https://mirrors.tuna.tsinghua.edu.cn/ubuntu/` and extracted under
`/data2/wuguanting/quick_loop/data/python-dev`; the system CUDA driver is not modified.

## Ouro R3 And vLLM

The Transformers wrapper uses `trust_remote_code=True`, supports R3/R4 inference, finds
the official gate at `model.early_exit_gate`, freezes every gate parameter, and fails if
the real gate cannot be found. Run its greedy smoke test with:

```bash
HF_ENDPOINT=https://hf-mirror.com uv run --no-sync python scripts/smoke_test_ouro.py
```

The recurrent model has 48 shared parameter layers. `scripts/prepare_vllm_ouro.py`
creates a local metadata view with 144 effective attention calls for R3 so vLLM assigns
an independent KV-cache slot to each `(loop_step, logical_layer)` pair. Weights and
official remote code remain symlinks to the immutable HF snapshot.

```bash
HF_ENDPOINT=https://hf-mirror.com uv run --no-sync python scripts/prepare_vllm_ouro.py
C_INCLUDE_PATH=/data2/wuguanting/quick_loop/data/python-dev/usr/include/python3.10:/data2/wuguanting/quick_loop/data/python-dev/usr/include \
LIBRARY_PATH=/data2/wuguanting/quick_loop/data/python-dev/usr/lib/x86_64-linux-gnu \
VLLM_PLUGINS=ouro_search \
uv run --no-sync python scripts/smoke_test_vllm_ouro.py
```

The vLLM smoke test has been run on this server: Ouro R3 loaded in BF16, generated two
responses, and allocated/exercised an 8.99 GiB KV cache (8176 tokens at the smoke-test
memory setting).

## Search-R1 Data

Data comes from `RUC-NLPIR/FlashRAG_datasets` through `hf-mirror.com`. Each row contains
`id`, `question`, `reference_answer`, `answer_aliases`, `dataset`, the Search-R1 prompt,
and veRL reward metadata.

```bash
HF_ENDPOINT=https://hf-mirror.com uv run --no-sync python scripts/prepare_searchr1_data.py
```

Actual prepared counts in `/data2/wuguanting/quick_loop/data/searchr1` are:

| Dataset | Train | Test/dev | Filtered | Total |
| --- | ---: | ---: | ---: | ---: |
| NQ | 79,168 | 3,610 | 0 | 82,778 |
| HotpotQA | 90,447 | 7,405 | 0 | 97,852 |
| Combined | 169,615 | 11,015 | 0 | 180,630 |

## Search-R1 GRPO Baselines

The official Search-R1 v0.2 outcome-only GRPO checkpoints for Qwen2.5-3B and
Qwen2.5-7B are pinned by immutable revision and downloaded only through `hf-mirror.com`:

```bash
HF_ENDPOINT=https://hf-mirror.com uv run --no-sync python \
  scripts/download_searchr1_baselines.py
```

Both baselines, Ouro R3, and Ouro R4 share `scripts/evaluate_agent.py`. Select one of
`searchr1_qwen2_5_3b_grpo_v0_2`, `searchr1_qwen2_5_7b_grpo_v0_2`,
`ouro_2_6b_r3`, or `ouro_2_6b_r4` through Hydra's `model=` override. They use the same
Search-R1 prompt/protocol, vLLM backend, E5 + Wiki18 + FAISS Flat HTTP Retriever,
top-k=3, four-search limit, 3000-token aggregate response budget, outcome-only EM, and
trajectory schema. Each model keeps its native tokenizer chat template.

Artifact revisions, paths, and the complete configuration comparison are in
[`docs/searchr1_baselines.md`](docs/searchr1_baselines.md).

## E5 Wikipedia-2018 Retriever

The retriever pins `intfloat/e5-base-v2` at revision
`f52bf8ec8c7124536f0efb74aca902b2995e5bcd`, and uses Search-R1's
`PeterJinGo/wiki-18-e5-index` FAISS flat index, and
`PeterJinGo/wiki-18-corpus`. Artifacts are pinned by revision in
`scripts/download_wiki18.py` and stored under
`/data2/wuguanting/quick_loop/data/wiki18`. The corpus download is a tar.gz despite its
`.jsonl.gz` name; the script verifies its SHA-256, safely extracts `wiki_dump.jsonl`, and
checks the 14,393,573,105-byte result before the service loads it.

```bash
HF_ENDPOINT=https://hf-mirror.com uv run --no-sync python scripts/download_wiki18.py
bash scripts/launch_e5_retriever.sh
curl -X POST http://127.0.0.1:8000/search \
  -H 'content-type: application/json' \
  -d '{"query":"Who wrote Hamlet?","top_k":3}'
uv run --no-sync python scripts/benchmark_retriever.py --queries 100 --top-k 3 --batch-size 100
```

The Agent only calls `POST /search` over HTTP, so the retriever stays replaceable. The
benchmark uses the compatible `POST /search/batch` endpoint to avoid 100 separate HTTP
round trips. A mock service remains available through
`bash scripts/launch_mock_retriever.sh` for unit tests.
Set `OURO_RETRIEVER_DEVICE=cpu OURO_RETRIEVER_DTYPE=float32` to benchmark without using a
GPU; the default service uses `cuda:0` and FP16 query encoding.

The real CPU/FP32 deployment on this server reports 21,015,324 index entries. For the
query `Who wrote the play Hamlet?`, top-1 is the `Hamlet` article and its text identifies
William Shakespeare. The 100-query, top-3, batch-size-100 benchmark completed in 116.802
seconds: 116,741.34 ms per HTTP batch, 1,167.41 ms amortized mean latency per query, and
0.85615 QPS. These CPU Flat-index numbers are a correctness baseline; production RL needs
a dedicated GPU retriever or a separately deployed retrieval node.

## Agent, Reward, Mask, And Trajectories

The asynchronous veRL loop implements the Search-R1 protocol:

```text
<think>...</think><search>query</search>
<information>top-3 documents</information>
<think>...</think><answer>final answer</answer>
```

It allows at most four search actions. All turn IDs are 0-based and every rollout writes
a JSONL record under `outputs/trajectories/verl/` with question, references, prediction,
pure EM reward, search count, and per-turn R3/query/documents/information fields.
Model-generated tokens use policy mask 1; all inserted `<information>` tokens use mask 0.
The reward normalizes case, punctuation, English articles, and whitespace, checks all
aliases, and gives no search or formatting bonus.

## Search-R1 To Ouro GRPO Configuration

The reference is Search-R1 commit `598e61bd1d36895726d28a8d06b3a15bed19f5d3`.
Comments in `configs/train/grpo_r3.yaml` identify original and changed values.

| Parameter | Search-R1 original | Ouro current | Modified | Reason |
| --- | ---: | ---: | --- | --- |
| Actor | Qwen2.5-3B | Ouro-2.6B-Thinking | Yes | Research target |
| Recurrent steps | N/A | fixed R3 | Yes | Ouro compute setting |
| Exit gate | N/A | retained, frozen | Yes | Fixed-depth training |
| Precision | BF16 | BF16 | No | Same training precision |
| Algorithm / group size | GRPO / 5 | GRPO / 5 | No | Search-R1 baseline |
| Learning rate / warmup | 5e-7 / 0.285 | 5e-7 / 0.285 | No | Search-R1 baseline |
| Global / validation batch | 512 / 256 | 512 / 256 | No | Full config baseline |
| PPO mini batch | 256 | 256 | No | Search-R1 baseline |
| Micro batch | global 64 | 8 per GPU | Yes | veRL 0.6 API and memory |
| Prompt / per-turn output | 4096 / 500 | 4096 / 500 | No | Search-R1 limits |
| Total rollout response budget | implicit multi-turn | 3000 | Yes | Four searches plus evidence |
| Information limit | 500 | 500 | No | Search-R1 state masking |
| Search actions / top-k | 4 / 3 | 4 / 3 | No | Search-R1 environment |
| Rollout temperature / top-p | 1.0 / 1.0 | 1.0 / 1.0 | No | Search-R1 sampling |
| KL loss | 0.001, low-var | 0.001, low-var | No | Search-R1 baseline |
| Entropy coefficient | 0 | 0 | No | Search-R1 baseline |
| FSDP offload | actor/ref enabled | actor/ref enabled | No | Search-R1 baseline |
| Rollout TP / GPUs | 1 / 8 | 1 / 8 | No | Search-R1 baseline |
| Save / eval / total steps | 100 / 100 / 1005 | 100 / 100 / 1005 | No | Full config baseline |
| vLLM | older Search-R1 stack | 0.10.0 | Yes | Torch 2.7.1 compatibility |
| veRL | Search-R1 fork | v0.6.0 pinned SHA | Yes | Async AgentLoop and Torch compatibility |
| Reward | final-answer outcome | pure final-answer EM | No bonus | Explicit project requirement |

The short-run launcher uses batch/mini-batch 8 and micro-batch 1 only to validate the
end-to-end update economically; all algorithmic settings remain the same. It requires
eight GPUs with at least 70GB free each and a healthy real retriever, then saves a final
checkpoint and offline W&B/console metrics:

```bash
bash scripts/run_grpo_r3.sh 1
bash scripts/run_grpo_r3.sh 5
bash scripts/run_grpo_r3.sh 20
```

Logged metrics include reward/EM accuracy, search action rate, mean search turns, format
validity, response length, KL, policy loss, and gradient norm. The launcher refuses to
disturb GPUs occupied by another job.

The 1-, 5-, and 20-step acceptance runs completed on eight A800s. Every rank reported a
nonzero Ouro trunk gradient and parameter update while both real exit-gate parameters
remained frozen with no gradient. The 20-step checkpoint, including model, optimizer,
extra state, config, and tokenizer, is under
`outputs/checkpoints/grpo-r3-20-step/global_step_20`. These short runs validate plumbing;
they are not a replacement for the long comparison training stage.

## Quality Checks

```bash
uv lock --check
uv run ruff check .
uv run pytest
```

Default tests are offline and do not download Ouro, Wikipedia, or the FAISS index.

## Layout

```text
configs/                 Hydra model, retriever, AgentLoop, and GRPO configs
patches/                 Reproducible adaptation for the pinned veRL commit
src/ouro_search/data/    Search-R1 NQ/HotpotQA normalization
src/ouro_search/rewards/ Pure final-answer EM reward
src/ouro_search/verl/    Async search loop and Ouro model hooks
src/ouro_search/vllm_model/ Ouro R3 vLLM implementation and plugin
src/ouro_search/trajectory/ 0-based JSONL trajectory records
retriever/               Mock and real E5 + FAISS HTTP services
scripts/                 Data, deployment, benchmark, smoke, and training commands
tests/                   Offline unit and state-machine tests
outputs/                 Ignored trajectories, logs, and checkpoints
```
