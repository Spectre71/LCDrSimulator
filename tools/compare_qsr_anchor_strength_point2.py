#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare completed confined weak-anchoring sweeps at a fixed shell band, so anchoring dependence is measured "
            "against the same [2,10) late-window readout rather than against per-branch auto-selected bands."
        )
    )
    parser.add_argument("sweep_roots", nargs="+", type=Path, help="Completed sweep roots to compare.")
    parser.add_argument(
        "--analysis-subdir",
        type=str,
        default="focus_exponent_fixed_2_10",
        help="Analysis subdirectory under analysis_shell_depth containing the fixed-band exponent outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation/anchoring_strength_point2"),
        help="Directory where summary CSV/markdown outputs are written.",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=Path("pics/qsr_anchor_strength_point2_fixed_2_10.png"),
        help="Output PNG path for the anchoring comparison figure.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("pics/qsr_anchor_strength_point2_fixed_2_10.pdf"),
        help="Output PDF path for the anchoring comparison figure.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_kv_config(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cfg[key.strip()] = value.strip()
    return cfg


def pick_focus_row(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    for row in rows:
        if row.get("region") == "focus" and row.get("observable") == "density":
            return row
    raise ValueError(f"Could not find focus density row in {path}")


def pick_band_row(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    for row in rows:
        if row.get("qualifies_focus_band") == "True":
            return row
    if rows:
        return rows[0]
    raise ValueError(f"No band rows found in {path}")


def ensure_auto_focus_outputs(sweep_root: Path) -> None:
    common_fit = sweep_root / "analysis_shell_depth" / "focus_exponent" / "focus_exponent_common_fit.csv"
    if common_fit.exists():
        return
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "analyze_shell_focus_exponent.py"),
            str(sweep_root),
        ],
        check=True,
    )


def fmt(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def load_summary_row(sweep_root: Path, analysis_subdir: str) -> dict[str, float | str]:
    ensure_auto_focus_outputs(sweep_root)
    sweep_plan = read_csv_rows(sweep_root / "sweep_plan.csv")
    generated_config = Path(sweep_plan[0]["generated_config_path"])
    cfg = parse_kv_config(generated_config)
    fixed_focus = pick_focus_row(sweep_root / "analysis_shell_depth" / analysis_subdir / "focus_exponent_common_fit.csv")
    auto_focus = pick_focus_row(sweep_root / "analysis_shell_depth" / "focus_exponent" / "focus_exponent_common_fit.csv")
    auto_band = pick_band_row(sweep_root / "analysis_shell_depth" / "band_decomposition" / "shell_band_common_summary.csv")
    return {
        "sweep_root": str(sweep_root),
        "sweep_name": sweep_root.name,
        "W": float(cfg["W"]),
        "fixed_band_label": "[2,10)",
        "fixed_alpha": float(fixed_focus["alpha_rate_common"]),
        "fixed_ci95": float(fixed_focus["alpha_rate_ci95_norm"]),
        "fixed_half_range": float(fixed_focus["window_half_range_alpha_rate"]),
        "auto_band_label": str(auto_band["band_label"]),
        "auto_alpha": float(auto_focus["alpha_rate_common"]),
        "auto_ci95": float(auto_focus["alpha_rate_ci95_norm"]),
        "auto_half_range": float(auto_focus["window_half_range_alpha_rate"]),
    }


def write_markdown(path: Path, rows: list[dict[str, float | str]]) -> None:
    lines: list[str] = []
    lines.append("# Anchoring-Strength Point-2 Summary")
    lines.append("")
    lines.append("## Fixed-band comparison")
    lines.append("")
    lines.append("| W | Fixed band | Fixed alpha | 95% fit CI | Window half-range | Auto-selected band | Auto alpha |")
    lines.append("|---:|---|---:|---:|---:|---|---:|")
    for row in rows:
        lines.append(
            "| "
            f"{fmt(float(row['W']), 3)} | {row['fixed_band_label']} | {fmt(float(row['fixed_alpha']))} | {fmt(float(row['fixed_ci95']))} | "
            f"{fmt(float(row['fixed_half_range']))} | {row['auto_band_label']} | {fmt(float(row['auto_alpha']))} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This table fixes the confined readout to [2,10) so the anchoring-strength dependence is compared on one common observable. "
        "The auto-selected band is shown only as a control for whether the preferred shell-depth support shifts with W."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(rows: list[dict[str, float | str]], png_path: Path, pdf_path: Path) -> None:
    x = np.array([float(row["W"]) for row in rows], dtype=float)
    y_fixed = np.array([float(row["fixed_alpha"]) for row in rows], dtype=float)
    yerr_fixed = np.array([float(row["fixed_ci95"]) for row in rows], dtype=float)
    y_auto = np.array([float(row["auto_alpha"]) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.errorbar(x, y_fixed, yerr=yerr_fixed, fmt="o-", color="#b91c1c", capsize=4, lw=1.8, label="fixed [2,10) alpha")
    ax.plot(x, y_auto, "s--", color="#1d4ed8", lw=1.5, ms=5.5, label="auto-selected alpha")
    ax.set_xscale("log")
    ax.set_xlabel("Weak anchoring strength W [J/m^2]")
    ax.set_ylabel("Late-window density exponent alpha")
    ax.set_title("Point 2: anchoring dependence at fixed confined readout")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False)

    for row in rows:
        ax.text(
            float(row["W"]),
            float(row["fixed_alpha"]) + float(row["fixed_ci95"]) + 0.01,
            str(row["auto_band_label"]),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = [load_summary_row(path.resolve(), args.analysis_subdir) for path in args.sweep_roots]
    rows.sort(key=lambda row: float(row["W"]))

    output_dir = args.output_dir.resolve()
    csv_path = output_dir / "anchoring_strength_point2_summary.csv"
    md_path = output_dir / "anchoring_strength_point2_summary.md"
    png_path = args.png.resolve()
    pdf_path = args.pdf.resolve()

    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    plot_summary(rows, png_path, pdf_path)

    print(f"summary_csv={csv_path}")
    print(f"summary_md={md_path}")
    print(f"png={png_path}")
    print(f"pdf={pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())