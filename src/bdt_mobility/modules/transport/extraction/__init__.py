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
        from .od_routing import STEPS as _OD_STEPS
        from .transport_efficiency import STEPS as _TE_STEPS
        steps = [*_discover_steps(), *_OD_STEPS, *_TE_STEPS]
        globals()["STEPS"] = steps
        return steps
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
