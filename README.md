# ouro-search

`ouro-search` is a minimal research scaffold for testing whether extra Ouro loop
computation can repair weak search-agent turns. The current baseline runs
`ByteDance/Ouro-2.6B-Thinking` with Transformers and records 0-based search turns so
that selected turns can be changed from R3 to R4.

This stage deliberately contains only a mock HTTP retriever. It is not a Search-RL
training stack.

## Environment

Requirements:

- Python 3.10
- an NVIDIA driver compatible with the locked PyTorch wheel for GPU inference
- enough GPU or host memory for the 2.67B-parameter BF16 model
- `uv`

All package and model downloads on the current server must use domestic mirrors.
The project configures the Aliyun PyPI mirror in `pyproject.toml`. Configure the
Hugging Face mirror before model commands:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$PWD/.cache/huggingface"
uv sync --frozen
```

No Conda environment is used. On this server a local bootstrap executable is also
available at `.uv-bootstrap/uv-0.9.9.data/scripts/uv`; it is intentionally not
committed.

## Ouro Model

The default model is `ByteDance/Ouro-2.6B-Thinking`, loaded in BF16 with
`trust_remote_code=True`. The wrapper does not modify the downloaded
`modeling_ouro.py`. Inspection of the official remote code at commit
`f1edd81e7ac41355db670500ceaf204e0f73af68` shows that the exit gate is
`model.early_exit_gate`. That revision is pinned in the model config. The wrapper
requires that exact module to exist, freezes its weight and bias, asserts their
`requires_grad` values, and prints their state.

Generation currently uses `use_cache: false`. The pinned official remote code assigns
to legacy `Cache.key_cache` and `Cache.value_cache` attributes, which are read-only in
Transformers 4.54.1. Disabling the KV cache avoids that upstream incompatibility without
editing `modeling_ouro.py`; it preserves loop computation but makes decoding slower.

Runtime loop switching updates both the decoder execution field and its config:

```python
model.set_loop_steps(3)
model.set_loop_steps(4)
```

Run the real model smoke test with greedy decoding:

```bash
uv run python scripts/smoke_test_ouro.py
```

## Mock Retriever

Start the FastAPI service:

```bash
bash scripts/launch_mock_retriever.sh
```

The Agent talks to `POST http://127.0.0.1:8000/search` over HTTP. It never imports
the mock backend, so a later E5 + FAISS service can preserve the same contract.

## Search Agent

With the retriever running in another shell:

```bash
export HF_ENDPOINT=https://hf-mirror.com
uv run python scripts/smoke_test_agent.py
```

The protocol is `<think>...</think>` followed by either `<search>...</search>` or
`<answer>...</answer>`. Retriever evidence is inserted as
`<information>...</information>`. The parser tolerates missing closing tags, and
the runner stops after at most four retrievals.

Every turn index is 0-based. For example, `loop_steps_by_turn={2: 4}` changes the
third search-generation turn to R4 and leaves all other turns at the default R3.
Trajectories are appended to one JSONL file per sample under
`outputs/trajectories/`.

Ouro has not been trained for this search protocol. If the real zero-shot model
does not emit `<search>`, the smoke test preserves its actual output and reports an
unparseable termination instead of manufacturing an Agent transition.

## Quality Checks

```bash
uv run ruff check .
uv run pytest
```

Default tests use a scripted inference stub for parser and Agent state-machine
coverage. They do not download or pretend to execute Ouro.

## Layout

```text
configs/                 Hydra model, Agent, search, and evaluation configuration
src/ouro_search/models/  Official-model wrapper and exit-gate validation
src/ouro_search/inference/ Transformers text-generation engine
src/ouro_search/agent/   Protocol parser, prompt construction, and state machine
src/ouro_search/search/  HTTP retriever client and shared response types
src/ouro_search/trajectory/ JSONL trajectory schema and writer
retriever/               FastAPI mock service and mock backend
scripts/                 Model, Agent, and retriever smoke commands
tests/                   Offline unit and service tests
outputs/                 Ignored trajectories and logs
```

## Not Implemented Yet

- E5
- Wikipedia-2018
- FAISS
- veRL
- GRPO
- production Search RL

## Next Stage

1. Deploy a real E5 + FAISS retriever backed by Wikipedia-2018.
2. Prepare NQ and HotpotQA evaluation/training splits.
3. Integrate veRL and GRPO.
4. Keep Ouro fixed at R3 while applying Search-R1-style reinforcement learning.
