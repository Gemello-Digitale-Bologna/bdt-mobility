from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class PlatformOutput:
    """A single static output logged to DigitalHub after a pipeline step runs.

    ``name``  — the DigitalHub dataitem/artifact name (also used in ``@handler(outputs=[...])``)
    ``path``  — local file path, resolved from ``PathRegistry``
    ``kind``  — ``"table"`` for tabular parquet files, ``"artifact"`` for binary/geo files
    """

    name: str
    path: Path
    kind: Literal["table", "artifact"]


@dataclass
class DynamicPlatformOutputs:
    """Marker for steps whose output names are only known at runtime.

    The step's ``main()`` return value provides the names; the generator
    produces a loop that logs each one from ``directory / f"{name}.parquet"``.

    ``directory`` — base directory for all outputs (from ``PathRegistry``)
    ``kind``      — ``"table"`` or ``"artifact"``, applied to every output
    """

    directory: Path
    kind: Literal["table", "artifact"]
