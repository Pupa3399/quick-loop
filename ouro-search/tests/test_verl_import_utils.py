from __future__ import annotations

import sys

from omegaconf import OmegaConf
from verl.utils.import_utils import import_external_libs


def test_import_external_libs_accepts_omegaconf_list() -> None:
    external_libs = OmegaConf.create(["json", "pathlib"])

    import_external_libs(external_libs)

    assert "json" in sys.modules
    assert "pathlib" in sys.modules
