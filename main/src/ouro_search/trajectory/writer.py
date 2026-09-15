from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path

from ouro_search.trajectory.schema import Trajectory


class JsonlTrajectoryWriter:
    def __init__(
        self,
        output_dir: str | Path = "main/data/trajectories",
        *,
        atomic_replace: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.atomic_replace = atomic_replace

    def write(self, trajectory: Trajectory) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in trajectory.id
        )
        path = self.output_dir / f"{safe_id or 'trajectory'}.jsonl"
        if not self.atomic_replace:
            with path.open("a", encoding="utf-8") as handle:
                json.dump(trajectory.to_dict(), handle, ensure_ascii=False)
                handle.write("\n")
            return path
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.output_dir, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(trajectory.to_dict(), handle, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name)
            raise
        return path
