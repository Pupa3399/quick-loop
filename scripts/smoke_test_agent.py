from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from ouro_search.agent import AgentRunner  # noqa: E402
from ouro_search.inference import TransformersEngine  # noqa: E402
from ouro_search.models import OuroModel  # noqa: E402
from ouro_search.search import HttpSearchClient  # noqa: E402
from ouro_search.trajectory import JsonlTrajectoryWriter  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    if os.environ["HF_ENDPOINT"] != config.model.hf_endpoint:
        raise RuntimeError("HF_ENDPOINT must use the configured domestic mirror")
    model = OuroModel(
        model_name=config.model.model_name,
        revision=config.model.revision,
        dtype=config.model.dtype,
        trust_remote_code=config.model.trust_remote_code,
        total_ut_steps=config.model.total_ut_steps,
        freeze_exit_gate=config.model.freeze_exit_gate,
        device=config.model.device,
        cache_dir=str(PROJECT_ROOT / config.model.cache_dir),
    )
    runner = AgentRunner(
        TransformersEngine(model, use_cache=config.model.use_cache),
        HttpSearchClient(
            config.search.retriever_url,
            top_k=config.search.top_k,
            timeout_seconds=config.search.timeout_seconds,
        ),
        max_search_turns=config.agent.max_search_turns,
        default_loop_steps=config.agent.default_loop_steps,
        max_new_tokens=config.agent.max_new_tokens,
        temperature=config.agent.temperature,
        top_p=config.agent.top_p,
        trajectory_writer=JsonlTrajectoryWriter(PROJECT_ROOT / config.eval.trajectory_dir),
    )
    trajectory = runner.run(
        "Use the retriever to search for 'Ouro model'. According to the retrieved document, "
        "what evidence does it contain?",
        sample_id="smoke-agent",
    )
    print(trajectory.to_dict())
    if trajectory.termination_reason != "answer":
        print(
            "The real zero-shot Ouro output did not complete the search protocol; "
            "the trajectory above was preserved without fabrication."
        )


if __name__ == "__main__":
    main()
