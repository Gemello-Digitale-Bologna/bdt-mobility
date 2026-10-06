"""
Run the full data preparation pipeline in sequence.
Stops immediately if any step fails (non-zero exit code).

Usage:
    uv run python run_preparation.py
    uv run python run_preparation.py --list
    uv run python run_preparation.py --steps 1:       # skip first step
    uv run python run_preparation.py --steps 1:3      # run steps 1 through 3 (inclusive)
    uv run python run_preparation.py --steps :2       # run only first 3 steps (0-2)
    uv run python run_preparation.py --steps 0        # run only step 0
"""

import argparse
import subprocess
import sys

from bdt_mobility.preparation.pipeline import PIPELINE


def parse_slice(s: str) -> slice:
    if ":" not in s:
        n = int(s)
        return slice(n, n + 1)
    parts = s.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"expected N, START:END or START: (e.g. '0', '1:3', '1:'), got {s!r}"
        )
    start = int(parts[0]) if parts[0] else None
    stop = int(parts[1]) + 1 if parts[1] else None  # END is inclusive
    return slice(start, stop)


parser = argparse.ArgumentParser(description="Run the bdt_mobility preparation pipeline.")
parser.add_argument(
    "--list", action="store_true", help="Print all steps with their index and exit."
)
parser.add_argument(
    "--steps",
    type=parse_slice,
    metavar="START:END",
    help="Slice of steps to run, e.g. '1:' skips the first step.",
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

print("\nAll preparation steps completed successfully.")
