from __future__ import annotations

import importlib
from pathlib import Path


def _discover_steps() -> list:
    pkg_path = Path(__file__).parent
    return [
        importlib.import_module(f".{f.stem}", package=__name__)
        for f in sorted(pkg_path.glob("step_*.py"))
    ]


def __getattr__(name: str):
    if name == "STEPS":
        steps = _discover_steps()
        globals()["STEPS"] = steps
        return steps
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
