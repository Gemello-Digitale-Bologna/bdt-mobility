import logging
import os
from pathlib import Path

# Tag corresponding to the version of datasets used by the project
DATA_TAG = os.environ.get("BDT_DATA_TAG", "san_felice")

# Project root path — override with BDT_DATA_ROOT to point at a test fixture tree
_default_root = Path(__file__).parents[3].resolve()
ROOT_PATH = Path(os.environ["BDT_DATA_ROOT"]) if "BDT_DATA_ROOT" in os.environ else _default_root

# Path to the tagged data dir
DATA_PATH = ROOT_PATH / "data" / DATA_TAG

# Path to the tmp dir
TMP_PATH = (
  Path(os.environ["BDT_TMP_PATH"]) if "BDT_TMP_PATH" in os.environ
  else ROOT_PATH / "data" / "tmp"
)

_level_name = os.environ.get("BDT_LOG_LEVEL", "INFO").upper()
logging_level = getattr(logging, _level_name, logging.INFO)
