"""Ordered preparation pipeline assembled from per-module STEPS lists."""
from __future__ import annotations

from bdt_mobility.preparation import prepare_road_network
from bdt_mobility.modules.transport import preparation as transport_preparation
from bdt_mobility.modules.ethics import preparation as ethics_preparation

PIPELINE = [
    prepare_road_network,
    *transport_preparation.STEPS,
    *ethics_preparation.STEPS,
]
