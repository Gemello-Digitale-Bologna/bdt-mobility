from .config_vars import (
    ENSEMBLE_SIZE,
    ETHICAL_YEAR,
    behavioral_params,
    config_params,
    policy_params,
)

__all__ = [
    "ENSEMBLE_SIZE",
    "ETHICAL_YEAR",
    "MobilityModel",
    "behavioral_params",
    "compute_kpis",
    "config_params",
    "evaluate_model",
    "policy_params",
]

# `model.model` is imported lazily: the transport/ethics sub-models import
# `bdt_mobility.model.axis`, so an eager import here would be circular.
_LAZY = {"MobilityModel", "compute_kpis", "evaluate_model"}


def __getattr__(name):
    if name in _LAZY:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
