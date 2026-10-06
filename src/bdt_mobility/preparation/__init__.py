from __future__ import annotations

import importlib
from pathlib import Path


def _discover_steps() -> list:
    pkg_path = Path(__file__).parent
    return [
        importlib.import_module(f".prepare_road_network", package=__name__)
    ]


STEPS = _discover_steps()