# GPU FAISS And 7-GPU Resource Validation

Validated on 2026-09-10 with eight NVIDIA A800-SXM4-80GB GPUs. GPU 0 was dedicated to
the Retriever, and physical GPUs 1-7 were exposed to veRL through `CUDA_VISIBLE_DEVICES`.

## Retriever

The corpus and algorithm are unchanged: Wiki18 with 21,015,324 passages,
`intfloat/e5-base-v2`, inner-product FAISS Flat, and top-k=3. E5 encodes in FP16 on GPU 0.
FAISS loads the CPU index and explicitly clones it to GPU 0 as an FP32 `GpuIndexFlat`.

`GET /health` returned:

```json
{
  "status": "ok",
  "index_size": 21015324,
  "index_type": "GpuIndexFlat",
  "faiss_gpu": true,
  "faiss_index_device": "cuda:0",
  "faiss_index_precision": "float32",
  "faiss_gpu_memory_bytes": 66270003200
}
```

The FAISS transfer reduced free GPU memory by 66.270 GB (61.719 GiB). `nvidia-smi`
reported 63,844 MiB for the full Retriever process and 63,853 MiB total GPU use. The
additional process memory includes E5 and CUDA/FAISS runtime state.

The real query `Who wrote the play Hamlet?` returned the `Hamlet` article at rank 0 with
score 0.86808574; the text states that William Shakespeare wrote the tragedy.

## Fixed 100-Query Benchmark

Both runs used the first 100 questions from the same Search-R1 test parquet, one HTTP
batch of 100, and top-k=3. Latencies below are batch latency divided by 100; therefore
mean, P50, and P95 are identical for this single-batch measurement.

| Backend | Mean (ms/query) | P50 | P95 | HTTP batch | QPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU `IndexFlatIP` | 1,099.314 | 1,099.314 | 1,099.314 | 109.931 s | 0.909 |
| GPU `GpuIndexFlat` | 4.283 | 4.283 | 4.283 | 0.428 s | 203.141 |

GPU versus CPU result parity was 98/100 for ordered top-3 IDs, 100/100 for unordered
top-3 sets, and 100% mean top-3 overlap. The two ordered mismatches were ties: query 63
swapped passages `1121347` and `244444`, and query 93 swapped `4719305` and `4719285`.
No retrieved passage changed.

## One Retriever GPU Plus Seven Training GPUs

A real veRL/FSDP world-size-7 step completed on GPUs 1-7 while the GPU 0 Retriever
remained allocated and healthy. The short-run launcher used 7 prompts, group size 5,
7 AgentLoop workers, and produced 35 rollouts. It completed rollout, log-prob/reference
passes, backward/optimizer handling, and wrote seven model, optimizer, and extra-state
checkpoint shards plus Hugging Face metadata.

| Metric | 7 GPU smoke | Existing 8 GPU smoke | Difference |
| --- | ---: | ---: | ---: |
| Step time | 115.159 s | 181.398 s | -36.52% |
| veRL throughput | 28.133 | 20.390 | +37.97% |
| PyTorch max allocated | 62.390 GiB | 62.035 GiB | +0.355 GiB |
| PyTorch max reserved | 66.529 GiB | 66.061 GiB | +0.469 GiB |

The 0.5-second external sampler observed 50,044 MiB on physical GPUs 1 and 7, and
50,140 MiB on GPUs 2-6. The in-process PyTorch peak counter is the conservative capacity
number because a polling sampler can miss short allocation peaks.

The 7-GPU rollout happened to produce reward 0, no legal search action, and zero policy
gradient, so that particular step did not change a parameter. This is a sampling outcome,
not an FSDP failure: all seven ranks completed collective work and checkpoint writing,
with no OOM, NaN, Inf, or communication error. It also means this one step did not exercise
Retriever HTTP calls during rollout; prior 8-GPU acceptance runs already exercised real
multi-turn retrieval and nonzero updates.

The short runs are not an apples-to-apples scaling benchmark: the 7-GPU batch contained
35 rollouts with mean response length 491.657 and no search, while the prior 8-GPU batch
contained 40 rollouts with mean response length 584.375 and search action rate 0.15.
Use a fixed replayable rollout set for a defensible long-run throughput comparison.

Recommendation: the 1+7 layout fits and has no observed FSDP communication or memory
blocker. Use it for long training only after a multi-step soak confirms concurrent
Retriever traffic and tail latency under representative search-action rates.
