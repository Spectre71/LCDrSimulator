from __future__ import annotations

import argparse
import csv
from pathlib import Path

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORE_SUMMARY = ROOT / "validation" / "sanity3_rate_sweep_merged_analysis" / "core" / "core_cohort_summary.csv"
DEFAULT_SKELETON_SUMMARY = ROOT / "validation" / "sanity3_rate_sweep_merged_analysis" / "skeleton" / "skeleton_cohort_summary.csv"
DEFAULT_PNG = ROOT / "pics" / "confined_transient_bridge_summary_2026-04-10.png"
DEFAULT_PDF = ROOT / "pics" / "confined_transient_bridge_summary_2026-04-10.pdf"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f64(value: str) -> float:
    return float(value)


def select_summary_row(rows: list[dict[str, str]], cohort: str, target_after_tc_s: float) -> dict[str, str]:
    for row in rows:
        if row["cohort"] != cohort:
            continue
        if abs(f64(row["target_after_Tc_s"]) - target_after_tc_s) <= 1e-18:
            return row
    raise KeyError(f"Missing summary row for cohort={cohort} offset={target_after_tc_s}")


def load_run_series(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(csv_path)
    pairs = sorted((f64(row["t_ramp_s"]), f64(row["defect_density_per_system_vox"])) for row in rows)
    t_ramp_ns = np.array([pair[0] * 1e9 for pair in pairs], dtype=float)
    defect = np.array([pair[1] for pair in pairs], dtype=float)
    return t_ramp_ns, defect


def plot_top_panel(ax: Axes, summary_rows: list[dict[str, str]], cohort: str, offsets_s: list[float], title: str) -> None:
    colors = {
        3.4e-8: "#0f766e",
        3.6e-8: "#c2410c",
        4.0e-8: "#334155",
    }
    markers = {
        3.4e-8: "o",
        3.6e-8: "s",
        4.0e-8: "^",
    }
    for off_s in offsets_s:
        row = select_summary_row(summary_rows, cohort, off_s)
        csv_path = ROOT / row["csv_path"]
        t_ramp_ns, defect = load_run_series(csv_path)
        label = f"after Tc = {off_s * 1e9:.0f} ns"
        ax.plot(
            t_ramp_ns,
            defect,
            marker=markers[off_s],
            ms=5,
            lw=1.8,
            color=colors[off_s],
            label=label,
        )

    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel(r"$t_{ramp}$ [ns]")


def plot_bottom_panel(ax: Axes, summary_rows: list[dict[str, str]], title: str) -> None:
    cohorts = [
        ("all11", "All 11 runs", "#111827", "o", "-"),
        ("no1000_10", "25 to 700", "#0f766e", "s", "--"),
        ("slow5", "200 to 1000", "#c2410c", "^", ":"),
    ]
    for cohort, label, color, marker, linestyle in cohorts:
        cohort_rows = [row for row in summary_rows if row["cohort"] == cohort]
        cohort_rows.sort(key=lambda row: f64(row["target_after_Tc_s"]))
        x_ns = np.array([f64(row["target_after_Tc_s"]) * 1e9 for row in cohort_rows], dtype=float)
        slopes = np.array([f64(row["slope_defect_vs_rate"]) for row in cohort_rows], dtype=float)
        ax.plot(x_ns, slopes, color=color, marker=marker, ms=5, lw=1.8, linestyle=linestyle, label=label)

    ax.axhline(0.0, color="#475569", lw=1.0, alpha=0.8)
    ax.axvspan(34.0, 36.0, color="#fde68a", alpha=0.25)
    ax.text(34.15, ax.get_ylim()[0] * 0.85 if ax.get_ylim()[0] < 0 else 0.0, "active window", fontsize=9, color="#7c2d12")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel(r"measurement offset after $T_c$ [ns]")
    ax.set_ylabel("slope of defect observable vs rate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a merged confined transient-bridge summary figure from existing cohort analyses.")
    parser.add_argument("--core-summary", default=str(DEFAULT_CORE_SUMMARY), help="Core-cohort summary CSV")
    parser.add_argument("--skeleton-summary", default=str(DEFAULT_SKELETON_SUMMARY), help="Skeleton-cohort summary CSV")
    parser.add_argument("--png", default=str(DEFAULT_PNG), help="PNG output path")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF), help="PDF output path")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively")
    args = parser.parse_args()

    core_summary_rows = read_csv_rows(Path(args.core_summary))
    skeleton_summary_rows = read_csv_rows(Path(args.skeleton_summary))
    offsets_top = [3.4e-8, 3.6e-8, 4.0e-8]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax_core_runs, ax_skel_runs, ax_core_slopes, ax_skel_slopes = axes.ravel()

    plot_top_panel(ax_core_runs, core_summary_rows, "all11", offsets_top, "Whole-volume core density across merged runs")
    ax_core_runs.set_ylabel("core density [core vox / system vox]")
    ax_core_runs.legend(loc="best")

    plot_top_panel(ax_skel_runs, skeleton_summary_rows, "all11", offsets_top, "Whole-volume line density across merged runs")
    ax_skel_runs.set_ylabel("line density [length / system vox]")
    ax_skel_runs.legend(loc="best")

    plot_bottom_panel(ax_core_slopes, core_summary_rows, "Core metric slope decay by cohort")
    ax_core_slopes.legend(loc="best")

    plot_bottom_panel(ax_skel_slopes, skeleton_summary_rows, "Skeleton metric slope decay by cohort")
    ax_skel_slopes.legend(loc="best")

    fig.suptitle(
        "Confined transient bridge under strong radial confinement\n"
        "Slower quenches transiently retain more 3D defect structure before the signal collapses",
        fontsize=14,
    )

    png_path = Path(args.png)
    pdf_path = Path(args.pdf)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()