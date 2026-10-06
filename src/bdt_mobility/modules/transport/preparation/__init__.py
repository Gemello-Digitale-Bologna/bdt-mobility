from __future__ import annotations

import importlib
from pathlib import Path


def _discover_submodules() -> list:
    """Discover all submodule STEPS and aggregate them."""
    pkg_path = Path(__file__).parent
    all_steps = []
    for submodule_dir in sorted(pkg_path.glob("*/__init__.py")):
        submodule_name = submodule_dir.parent.name
        try:
            submodule = importlib.import_module(f".{submodule_name}", package=__name__)
            if hasattr(submodule, "STEPS"):
                all_steps.extend(submodule.STEPS)
        except (ImportError, AttributeError):
            pass
    return all_steps


def __getattr__(name: str):
    if name == "STEPS":
        steps = _discover_submodules()
        globals()["STEPS"] = steps
        return steps
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")