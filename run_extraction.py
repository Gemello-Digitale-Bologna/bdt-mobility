"""
Run the full data extraction pipeline in sequence.
Stops immediately if any step fails (non-zero exit code).

Usage:
    uv run python run_extraction.py
    uv run python run_extraction.py --list
    uv run python run_extraction.py --steps 3:       # skip first 3 steps
    uv run python run_extraction.py --steps 3:8      # run steps 3 through 8 (inclusive)
    uv run python run_extraction.py --steps :4       # run only first 5 steps (0-4)
    uv run python run_extraction.py --steps 9        # run only step 9
"""

import argparse
import subprocess
import sys
import os

from bdt_mobility.extraction.pipeline import PIPELINE

# To ensure the full reproducibility of the pipeline, set the pythonseed before running it
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

def parse_slice(s: str) -> slice:
    if ":" not in s:
        n = int(s)
        return slice(n, n + 1)
    parts = s.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"expected N, START:END or START: (e.g. '3', '2:8', '3:'), got {s!r}"
        )
    start = int(parts[0]) if parts[0] else None
    stop = int(parts[1]) + 1 if parts[1] else None  # END is inclusive
    return slice(start, stop)


parser = argparse.ArgumentParser(description="Run the bdt_mobility extraction pipeline.")
parser.add_argument(
    "--list", action="store_true", help="Print all steps with their index and exit."
)
parser.add_argument(
    "--steps",
    type=parse_slice,
    metavar="START:END",
    help="Slice of steps to run, e.g. '3:' skips the first 3.",
)
args = parser.parse_args()

if args.list:
    for i, step in enumerate(PIPELINE):
        print(f"  {i:3d}  {step.__name__}")
    sys.exit(0)

steps = PIPELINE[args.steps] if args.steps else PIPELINE

for step in steps:
    name = step.__name__
    print(f"\n{'=' * 60}")
    print(f"Running: {name}")
    print(f"{'=' * 60}")
    result = subprocess.run(["uv", "run", "python", "-m", name])
    if result.returncode != 0:
        print(f"\nFAILED: {name} (exit code {result.returncode})")
        sys.exit(result.returncode)

print("\nAll extraction steps completed successfully.")
