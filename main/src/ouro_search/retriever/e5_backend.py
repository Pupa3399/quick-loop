from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True, slots=True)
class E5RetrieverConfig:
    model_name: str
    model_revision: str
    index_path: Path
    corpus_path: Path
    device: str = "cuda:0"
    dtype: str = "float16"
    query_max_length: int = 256
    batch_size: int = 64
    faiss_gpu: bool = False
    faiss_device: int = 0
    faiss_use_float16: bool = False
    model_cache_dir: str | None = None
    dataset_cache_dir: str | None = None


class E5Wiki18Retriever:
    """Search-R1-compatible E5 dense retriever over the Wiki-2018 flat index."""

    def __init__(self, config: E5RetrieverConfig) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("Install the retriever extra: uv sync --extra retriever") from exc
        if not config.index_path.is_file() or not config.corpus_path.is_file():
            raise FileNotFoundError("Wiki-2018 corpus/index is missing; run download_wiki18.py")
        self.config = config
        self.faiss = faiss
        self.device = torch.device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            revision=config.model_revision,
            cache_dir=config.model_cache_dir,
            use_fast=True,
        )
        torch_dtype = torch.float16 if config.dtype == "float16" else torch.float32
        self.model = AutoModel.from_pretrained(
            config.model_name,
            revision=config.model_revision,
            cache_dir=config.model_cache_dir,
            torch_dtype=torch_dtype,
        ).to(self.device)
        self.model.eval()
        cpu_index = faiss.read_index(str(config.index_path))
        self.faiss_gpu_memory_bytes = 0
        self.faiss_index_device = "cpu"
        self.faiss_index_precision = "float32"
        self._gpu_resources: Any | None = None
        if config.faiss_gpu:
            if not hasattr(faiss, "index_cpu_to_gpu"):
                raise RuntimeError("faiss_gpu=true requires a GPU-enabled FAISS build")
            available_gpus = int(faiss.get_num_gpus())
            if not 0 <= config.faiss_device < available_gpus:
                raise RuntimeError(
                    f"FAISS GPU {config.faiss_device} is unavailable; found {available_gpus} GPU(s)"
                )
            gpu_device = torch.device(f"cuda:{config.faiss_device}")
            torch.cuda.synchronize(gpu_device)
            free_before, _ = torch.cuda.mem_get_info(gpu_device)
            resources = faiss.StandardGpuResources()
            options = faiss.GpuClonerOptions()
            options.useFloat16 = config.faiss_use_float16
            self.index = faiss.index_cpu_to_gpu(
                resources,
                config.faiss_device,
                cpu_index,
                options,
            )
            del cpu_index
            torch.cuda.synchronize(gpu_device)
            free_after, _ = torch.cuda.mem_get_info(gpu_device)
            self.faiss_gpu_memory_bytes = max(0, free_before - free_after)
            self.faiss_index_device = f"cuda:{config.faiss_device}"
            self.faiss_index_precision = (
                "float16" if config.faiss_use_float16 else "float32"
            )
            # FAISS GPU indices require their StandardGpuResources owner to stay alive.
            self._gpu_resources = resources
        else:
            self.index = cpu_index
        self.corpus = load_dataset(
            "json",
            data_files=str(config.corpus_path),
            split="train",
            cache_dir=config.dataset_cache_dir,
            num_proc=4,
            keep_in_memory=False,
        )

    @torch.inference_mode()
    def _encode(self, queries: list[str]) -> np.ndarray:
        prefixed = [f"query: {query}" for query in queries]
        inputs = self.tokenizer(
            prefixed,
            max_length=self.config.query_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        output = self.model(**inputs, return_dict=True)
        mask = inputs["attention_mask"].unsqueeze(-1).bool()
        hidden = output.last_hidden_state.masked_fill(~mask, 0.0)
        embeddings = hidden.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings.float().cpu().numpy().astype(np.float32, copy=False)

    def _documents(
        self, indices: np.ndarray, scores: np.ndarray
    ) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for rank, (index, score) in enumerate(zip(indices, scores, strict=True)):
            if index < 0:
                continue
            record = self.corpus[int(index)]
            contents = str(record.get("contents", ""))
            title, _, text = contents.partition("\n")
            documents.append(
                {
                    "id": str(record.get("id", index)),
                    "title": str(record.get("title", title.strip('"'))),
                    "text": str(record.get("text", text or contents)),
                    "score": float(score),
                    "rank": rank,
                }
            )
        return documents

    def search_batch(
        self, queries: list[str], top_k: int = 3
    ) -> list[list[dict[str, Any]]]:
        if not queries:
            return []
        scores, indices = self.index.search(self._encode(queries), top_k)
        return [
            self._documents(single_indices, single_scores)
            for single_indices, single_scores in zip(indices, scores, strict=True)
        ]

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return self.search_batch([query], top_k)[0]
