from __future__ import annotations

import argparse
from pathlib import Path

from aq.reporting import generate_global_far_round_sweep_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Global Far-Round sweep results")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = generate_global_far_round_sweep_report(Path(args.output))
    print(f"wrote sweep report for {len(summary['runs'])} runs to {args.output}")


if __name__ == "__main__":
    main()
