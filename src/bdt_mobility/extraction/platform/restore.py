"""Restore pipeline outputs from DigitalHub onto the local data tree.

Shared by:
  * ``download_data.py`` — pull everything locally (lenient: warn and continue);
  * ``workflow_handlers.step_0`` — when the ingestion workflow is resumed from
    ``start_step > 0``, put back on the shared volume every output produced by
    steps ``1 .. start_step-1`` (strict: fail fast if anything is missing).

Only entities carrying the data label are considered, and for each output the
most recently created labelled version is used. All versions of an entity are
listed through the paginated ``get_*_versions`` SDK calls (``search_entity``
returns a single unordered page, so it could miss the newest version).

Dynamic steps (``DynamicPlatformOutputs``) log their outputs under names known
only at runtime. Alongside them they log a *manifest* artifact (see
``manifest_name``) listing ``[name, path relative to PathRegistry.DATA]``; it is
read back here to restore those outputs too.
"""

from __future__ import annotations

import importlib
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import digitalhub as dh
from digitalhub.utils.exceptions import EntityNotExistsError

from bdt_mobility.extraction.config import PROJECT_NAME
from bdt_mobility.extraction.platform.outputs import DynamicPlatformOutputs
from bdt_mobility.utils.paths import PathRegistry

logger = logging.getLogger(__name__)

_EPOCH = datetime.min.replace(tzinfo=timezone.utc)


class RestoreError(RuntimeError):
    """Raised in strict mode when a required output cannot be restored."""


def manifest_name(module_name: str) -> str:
    """DigitalHub name of the manifest artifact logged by a dynamic step."""
    return f"{module_name.rsplit('.', 1)[-1]}_manifest"


def _created_at(entity) -> datetime:
    """Creation timestamp of an entity version (oldest possible if missing/unparsable)."""
    created = getattr(entity.metadata, "created", None)
    if not created:
        return _EPOCH
    try:
        dt = datetime.fromisoformat(created)
    except ValueError:
        return _EPOCH
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _all_versions(name: str, kind: str) -> list:
    """Every version of the dataitem (kind 'table') or artifact called ``name``."""
    getter = dh.get_dataitem_versions if kind == "table" else dh.get_artifact_versions
    try:
        return getter(name, project=PROJECT_NAME)
    except EntityNotExistsError:  # never logged under this name
        return []


def restore_latest(name: str, kind: str, target: Path, tag: str) -> Path | None:
    """Download the latest version of ``name`` labelled ``tag`` to ``target``.

    ``kind`` is the output kind ("table" → dataitem, "artifact" → artifact).
    ``target`` is the exact file path the pipeline expects. Returns the local
    path, or ``None`` if no labelled version exists.
    """
    versions = _all_versions(name, kind)
    existing = [v for v in versions if tag in (v.metadata.labels or [])]
    if not existing:
        # Entities ingested before labelling was introduced have no labels.
        available = sorted({lbl for v in versions for lbl in (v.metadata.labels or [])})
        logger.warning(
            "No %s %r with label %r among %d version(s) (available labels: %s)",
            kind, name, tag, len(versions), available or "none",
        )
        return None

    # No ordering guarantee on the listing: pick the most recently created version.
    latest = max(existing, key=_created_at)
    logger.debug(
        "Selected %r version %s (created %s) out of %d labelled version(s)",
        name, latest.id, latest.metadata.created, len(existing),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(latest.download(str(target), overwrite=True))


def _read_manifest(module_name: str, kind: str, tag: str) -> list[tuple[str, str, Path]] | None:
    name = manifest_name(module_name)
    with tempfile.TemporaryDirectory() as tmp:
        local = restore_latest(name, "artifact", Path(tmp) / f"{name}.json", tag)
        if local is None:
            return None
        entries = json.loads(local.read_text())
    return [(n, kind, PathRegistry.DATA / rel) for n, rel in entries]


def _step_outputs(step: ModuleType, tag: str) -> list[tuple[str, str, Path]] | None:
    """(name, kind, local path) triples a step produced; ``None`` if they cannot be determined."""
    m = importlib.import_module(step.__name__)
    outputs = getattr(m, "MAIN_OUTPUTS", [])
    if isinstance(outputs, DynamicPlatformOutputs):
        return _read_manifest(step.__name__, outputs.kind, tag)
    return [(o.name, o.kind, Path(o.path)) for o in outputs]


def restore_outputs(steps: list[ModuleType], tag: str, *, strict: bool) -> list[str]:
    """Restore every output of ``steps`` labelled ``tag``.

    Returns the names that could not be restored. In strict mode, raises
    ``RestoreError`` listing all of them instead (after trying every output,
    so a single run reports everything that is missing).
    """
    missing: list[str] = []
    downloaded = 0
    for step in steps:
        pairs = _step_outputs(step, tag)
        if pairs is None:
            logger.warning(
                "No manifest %r with label %r: cannot restore dynamic outputs of %s",
                manifest_name(step.__name__), tag, step.__name__,
            )
            missing.append(manifest_name(step.__name__))
            continue
        for name, kind, path in pairs:
            restored = restore_latest(name, kind, path, tag)
            if restored is None:
                missing.append(name)
                continue
            downloaded += 1
            logger.info("Downloaded %s", restored.name)
            logger.debug("Saved %s to %s", restored.name, restored)

    logger.debug("Downloaded %d file(s), %d missing", downloaded, len(missing))

    if missing and strict:
        raise RestoreError(
            f"Cannot restore {len(missing)} output(s) labelled {tag!r}: {missing}. "
            "Run those steps first (lower start_step) or check the label."
        )
    return missing
