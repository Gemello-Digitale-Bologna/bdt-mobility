"""Download all pipeline data from DigitalHub to local PathRegistry paths.

Mirrors ingest_data.py: first downloads raw inputs (step_00), then pulls all
static MAIN_OUTPUTS produced by subsequent steps.

Dynamic outputs (DynamicPlatformOutputs steps) are skipped: their names are
only known at runtime. This is a temporary limitation TO BE FIXED.

Usage:
    python download_data.py
"""

from __future__ import annotations

import importlib
import logging

import digitalhub as dh

from bdt_mobility.extraction import step_00_get_input_data
from bdt_mobility.extraction.pipeline import PIPELINE
from bdt_mobility.extraction.platform.outputs import DynamicPlatformOutputs

PROJECT = "mobility"

logging.basicConfig(level=logging.INFO)

logging.info("=== step_00: downloading raw inputs ===")
step_00_get_input_data.main()

project = dh.get_project(PROJECT)

for step in PIPELINE[1:]:
    m = importlib.import_module(step.__name__)
    outputs = getattr(m, "MAIN_OUTPUTS", [])

    if not outputs:
        continue

    if isinstance(outputs, DynamicPlatformOutputs):
        logging.warning(
            "Skipping dynamic outputs for %s "
            "(names only known at runtime — download manually from the DigitalHub UI)",
            step.__name__,
        )
        continue

    for out in outputs:
        logging.info("Downloading %r → %s", out.name, out.path)
        if out.kind == "artifact":
            entity = project.get_artifact(out.name)
        else:
            entity = project.get_dataitem(out.name)
        entity.download(str(out.path.parent), overwrite=True)
