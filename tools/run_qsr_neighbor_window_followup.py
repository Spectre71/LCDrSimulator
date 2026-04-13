#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFSETS = "4.5e-8,5.0e-8,5.5e-8,6.0e-8,6.5e-8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full neighboring-window follow-up on a completed transient weak-anchor sweep: "
            "audit retained snapshot distinctness, build shell-depth outputs, decompose shell bands, "
            "and generate the focus-summary figure/table."
        )
    )
    parser.add_argument("sweep_root", type=Path, help="Completed sweep root to analyze.")
    parser.add_argument(
        "--offsets",
        type=str,
        default=DEFAULT_OFFSETS,
        help="Comma-separated offsets after Tc in seconds.",
    )
    parser.add_argument(
        "--allow-aliased",
        action="store_true",
        help="Continue even if the snapshot feasibility audit reports aliasing.",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        help="Optional PNG path for the focus-summary figure.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Optional PDF path for the focus-summary figure.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def read_blocked_flag(summary_path: Path) -> bool:
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if "|" not in line or line.startswith("|") is False:
            continue
    return False


def main() -> int:
    args = parse_args()
    sweep_root = args.sweep_root.resolve()
    if not sweep_root.exists():
        raise FileNotFoundError(f"Sweep root not found: {sweep_root}")

    sweep_stem = sweep_root.name
    png_path = args.png.resolve() if args.png else (ROOT / "pics" / f"{sweep_stem}_shell_focus_summary.png")
    pdf_path = args.pdf.resolve() if args.pdf else (ROOT / "pics" / f"{sweep_stem}_shell_focus_summary.pdf")

    python = sys.executable
    audit_cmd = [
        python,
        str(ROOT / "tools" / "audit_qsr_snapshot_window_feasibility.py"),
        str(sweep_root),
        "--offsets",
        args.offsets,
    ]
    run_command(audit_cmd)

    summary_csv = sweep_root / "analysis_snapshot_feasibility" / "snapshot_window_feasibility_summary.csv"
    if summary_csv.exists() and not args.allow_aliased:
        import csv

        with summary_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if any(row.get("all_offsets_distinct") == "False" for row in rows):
            raise RuntimeError(
                "Snapshot feasibility audit found aliased windows. Re-run with --allow-aliased only if that is intentional."
            )

    run_command(
        [
            python,
            str(ROOT / "tools" / "analyze_shell_depth.py"),
            str(sweep_root),
            "--offsets",
            args.offsets,
        ]
    )
    run_command(
        [
            python,
            str(ROOT / "tools" / "analyze_shell_band_decomposition.py"),
            str(sweep_root),
            "--offsets",
            args.offsets,
        ]
    )
    run_command(
        [
            python,
            str(ROOT / "tools" / "plot_shell_focus_summary.py"),
            str(sweep_root),
            "--offsets",
            args.offsets,
            "--png",
            str(png_path),
            "--pdf",
            str(pdf_path),
        ]
    )

    print(f"analysis_complete_for={sweep_root}")
    print(f"png={png_path}")
    print(f"pdf={pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
