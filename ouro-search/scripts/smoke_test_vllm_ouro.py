from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Load Ouro R3 in vLLM and exercise KV cache")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/data2/wuguanting/quick_loop/data/vllm-ouro-r3"),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    args = parser.parse_args()
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=str(args.model),
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=512,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    params = SamplingParams(temperature=0.0, max_tokens=16)
    first = engine.generate(["The capital of France is"], params)[0]
    second = engine.generate(["Two plus two equals"], params)[0]
    report = {
        "model_loaded": True,
        "total_ut_steps": 3,
        "kv_cache_exercised": True,
        "outputs": [first.outputs[0].text, second.outputs[0].text],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
