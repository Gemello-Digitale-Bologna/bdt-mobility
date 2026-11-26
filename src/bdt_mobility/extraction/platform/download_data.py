"""Download all pipeline data from DigitalHub to local PathRegistry paths.

Mirrors ingest_data.py: first downloads raw inputs (step_00), then pulls every
output produced by subsequent steps, restricted to entities labelled DATA_TAG
(latest labelled version of each). Dynamic outputs are found through the
manifest their step logs at ingestion time (see restore.py).

Usage:
    python download_data.py
"""

from __future__ import annotations

import logging

from bdt_mobility.extraction import step_00_get_input_data
from bdt_mobility.extraction.pipeline import PIPELINE
from bdt_mobility.extraction.platform.restore import restore_outputs
from bdt_mobility.utils.config import DATA_TAG

logging.basicConfig(level=logging.INFO)

logging.info("=== step_00: downloading raw inputs ===")
step_00_get_input_data.main()

logging.info("=== downloading pipeline outputs labelled %r ===", DATA_TAG)
missing = restore_outputs(PIPELINE[1:], DATA_TAG, strict=False)
if missing:
    logging.warning("%d output(s) not downloaded: %s", len(missing), missing)
