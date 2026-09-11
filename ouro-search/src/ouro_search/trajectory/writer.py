from __future__ import annotations

import json
from pathlib import Path

from ouro_search.trajectory.schema import Trajectory


class JsonlTrajectoryWriter:
    def __init__(self, output_dir: str | Path = "outputs/trajectories") -> None:
        self.output_dir = Path(output_dir)

    def write(self, trajectory: Trajectory) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in trajectory.id
        )
        path = self.output_dir / f"{safe_id or 'trajectory'}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            json.dump(trajectory.to_dict(), handle, ensure_ascii=False)
            handle.write("\n")
        return path
