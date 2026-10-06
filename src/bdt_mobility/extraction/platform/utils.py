from __future__ import annotations

from importlib import import_module

from bdt_mobility.extraction.pipeline import PIPELINE

MODULES = [step.__name__ for step in PIPELINE]


def exec_main(module_name: str):
    """Dynamically import the specified module and execute its ``main()``.

    Returns whatever ``main()`` returns (some steps return generated table
    names so the caller can log them).
    """
    m = import_module(module_name)
    return getattr(m, "main")()


def parse_labels(labels: list[str] | str | None) -> list[str] | None:
    """Normalise input labels (a string, list of strings, or None)."""
    if labels is None:
        return None
    if isinstance(labels, str):
        return [labels]
    if isinstance(labels, list):
        if not all(isinstance(label, str) for label in labels):
            raise ValueError("All labels must be strings.")
        return labels
    raise ValueError("Labels must be a string, a list of strings, or None.")
