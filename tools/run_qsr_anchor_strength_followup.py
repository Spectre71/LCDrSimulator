#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFSETS = "4.5e-8,5.0e-8,5.5e-8,6.0e-8,6.5e-8"
DEFAULT_RAMP_VALUES = "25,35,50,70,100,140,200,300,500,700"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the matched dense weak-anchoring point-2 follow-up: execute a dense rate sweep from a supplied config, "
            "run the neighboring-window shell analysis, and extract a fixed-band focus exponent for apples-to-apples comparison "
            "against the active [2,10) baseline."
        )
    )
    parser.add_argument("base_config", type=Path, help="Validation config to sweep.")
    parser.add_argument("--output-root", type=Path, required=True, help="Sweep output directory.")
    parser.add_argument("--binary", type=Path, default=ROOT / "QSR_cuda", help="Path to the CUDA solver binary.")
    parser.add_argument("--ramp-values", type=str, default=DEFAULT_RAMP_VALUES, help="Comma-separated dense ramp ladder.")
    parser.add_argument("--offsets", type=str, default=DEFAULT_OFFSETS, help="Comma-separated after-Tc offsets in seconds.")
    parser.add_argument("--focus-start-index", type=int, default=1, help="Fixed focus-band start index.")
    parser.add_argument("--focus-end-index", type=int, default=4, help="Fixed focus-band end index.")
    parser.add_argument("--focus-band-label", type=str, default="[2,10)", help="Fixed focus-band label.")
    parser.add_argument(
        "--analysis-subdir",
        type=str,
        default="focus_exponent_fixed_2_10",
        help="Analysis subdirectory name under analysis_shell_depth for the fixed-band exponent outputs.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    base_config = args.base_config.resolve()
    output_root = args.output_root.resolve()
    binary = args.binary.resolve()
    if not base_config.exists():
        raise FileNotFoundError(f"Config not found: {base_config}")
    if not binary.exists():
        raise FileNotFoundError(f"Binary not found: {binary}")

    python = sys.executable
    sweep_name = output_root.name
    analysis_dir = output_root / "analysis_shell_depth" / args.analysis_subdir
    auto_png_path = ROOT / "pics" / f"{sweep_name}_focus_exponent.png"
    auto_pdf_path = ROOT / "pics" / f"{sweep_name}_focus_exponent.pdf"
    png_path = ROOT / "pics" / f"{sweep_name}_{args.analysis_subdir}.png"
    pdf_path = ROOT / "pics" / f"{sweep_name}_{args.analysis_subdir}.pdf"

    run_command(
        [
            python,
            str(ROOT / "tools" / "quench_rate_sweep.py"),
            "run",
            str(base_config),
            "--output-root",
            str(output_root),
            "--ramp-values",
            args.ramp_values,
            "--binary",
            str(binary),
            "--keep-going",
        ]
    )
    run_command(
        [
            python,
            str(ROOT / "tools" / "run_qsr_neighbor_window_followup.py"),
            str(output_root),
            "--offsets",
            args.offsets,
        ]
    )
    run_command(
        [
            python,
            str(ROOT / "tools" / "analyze_shell_focus_exponent.py"),
            str(output_root),
            "--offsets",
            args.offsets,
            "--png",
            str(auto_png_path),
            "--pdf",
            str(auto_pdf_path),
        ]
    )
    run_command(
        [
            python,
            str(ROOT / "tools" / "analyze_shell_focus_exponent.py"),
            str(output_root),
            "--offsets",
            args.offsets,
            "--focus-start-index",
            str(args.focus_start_index),
            "--focus-end-index",
            str(args.focus_end_index),
            "--focus-band-label",
            args.focus_band_label,
            "--analysis-dir",
            str(analysis_dir),
            "--png",
            str(png_path),
            "--pdf",
            str(pdf_path),
        ]
    )

    print(f"point2_complete_for={output_root}")
    print(f"auto_focus_png={auto_png_path}")
    print(f"auto_focus_pdf={auto_pdf_path}")
    print(f"fixed_band_analysis_dir={analysis_dir}")
    print(f"fixed_band_png={png_path}")
    print(f"fixed_band_pdf={pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())