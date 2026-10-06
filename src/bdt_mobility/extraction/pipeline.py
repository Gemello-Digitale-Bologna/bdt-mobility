"""Ordered extraction pipeline assembled from per-module STEPS lists."""
from __future__ import annotations

from bdt_mobility.extraction import step_00_get_input_data
from bdt_mobility.modules.transport import extraction as transport_extraction
from bdt_mobility.modules.ethics import extraction as ethics_extraction

PIPELINE = [
    step_00_get_input_data,
    *transport_extraction.STEPS,
    *ethics_extraction.STEPS,
]