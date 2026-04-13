from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = (
    ROOT
    / "validation"
    / "sanity3_rate_sweep_weak_anchor_W3em6_finalhold"
    / "analysis_option2_log_windows"
    / "transient_log_window_summary.csv"
)
DEFAULT_PNG = ROOT / "pics" / "sanity3_weak_anchor_finalhold_option2_2026-04-11.png"
DEFAULT_PDF = ROOT / "pics" / "sanity3_weak_anchor_finalhold_option2_2026-04-11.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the option-2 transient log-window follow-up for the no-early-stop W=3e-6 branch."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Summary CSV produced by tools/analyze_qsr_transient_log_windows.py.",
    )
    parser.add_argument(
        "--top-offsets",
        default="4.0e-8,5.0e-8,6.0e-8",
        help="Comma-separated after-Tc offsets to highlight in the per-run panels.",
    )
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG, help="PNG output path")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="PDF output path")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively")
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f64(value: str) -> float:
    return float(value)


def select_summary_row(rows: list[dict[str, str]], observable: str, target_after_tc_s: float) -> dict[str, str]:
    for row in rows:
        if row["observable"] != observable:
            continue
        if abs(f64(row["target_after_Tc_s"]) - target_after_tc_s) <= 1e-18:
            return row
    raise KeyError(f"Missing summary row for observable={observable} offset={target_after_tc_s}")


def load_detail_series(path: Path) -> dict[str, np.ndarray]:
    rows = read_csv_rows(path)
    rows.sort(key=lambda row: f64(row["t_ramp_s"]))
    return {
        "t_ramp_ns": np.array([f64(row["t_ramp_s"]) * 1e9 for row in rows], dtype=float),
        "rate": np.array([f64(row["rate_K_per_s"]) for row in rows], dtype=float),
        "observable": np.array([f64(row["observable_value"]) for row in rows], dtype=float),
        "selected_avg_S": np.array([f64(row["selected_avg_S"]) for row in rows], dtype=float),
        "selected_xi_grad_proxy": np.array([f64(row["selected_xi_grad_proxy"]) for row in rows], dtype=float),
    }


def ratio(values: np.ndarray) -> float:
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite) / np.min(finite))


def main() -> None:
    args = parse_args()
    summary_rows = read_csv_rows(args.summary)
    defect_rows = sorted(
        [row for row in summary_rows if row["observable"] == "defect_density_per_plaquette"],
        key=lambda row: f64(row["target_after_Tc_s"]),
    )
    avg_rows = sorted(
        [row for row in summary_rows if row["observable"] == "avg_S"],
        key=lambda row: f64(row["target_after_Tc_s"]),
    )
    xi_rows = sorted(
        [row for row in summary_rows if row["observable"] == "xi_grad_proxy"],
        key=lambda row: f64(row["target_after_Tc_s"]),
    )

    offsets_s = np.array([f64(row["target_after_Tc_s"]) for row in defect_rows], dtype=float)
    offsets_ns = offsets_s * 1e9
    defect_slopes = np.array([f64(row["slope_vs_rate"]) for row in defect_rows], dtype=float)
    avg_slopes = np.array([f64(row["slope_vs_rate"]) for row in avg_rows], dtype=float)
    xi_slopes = np.array([f64(row["slope_vs_rate"]) for row in xi_rows], dtype=float)
    defect_ratios = np.array([f64(row["value_ratio_max_over_min"]) for row in defect_rows], dtype=float)
    positive_cases = np.array([int(float(row["positive_cases"])) for row in defect_rows], dtype=int)

    avg_s_ratios: list[float] = []
    xi_ratios: list[float] = []
    for row in defect_rows:
        detail = load_detail_series(Path(row["detail_csv"]))
        avg_s_ratios.append(ratio(detail["selected_avg_S"]))
        xi_ratios.append(ratio(detail["selected_xi_grad_proxy"]))

    top_offsets = parse_float_list(args.top_offsets)
    top_colors = {
        4.0e-8: "#0f766e",
        5.0e-8: "#c2410c",
        6.0e-8: "#334155",
    }
    top_markers = {
        4.0e-8: "o",
        5.0e-8: "s",
        6.0e-8: "^",
    }

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.4))
    ax_defect_runs, ax_slopes, ax_order_runs, ax_ratios = axes.ravel()

    for off_s in top_offsets:
        row = select_summary_row(summary_rows, "defect_density_per_plaquette", off_s)
        detail = load_detail_series(Path(row["detail_csv"]))
        label = f"after Tc = {off_s * 1e9:.0f} ns"
        color = top_colors.get(off_s, "#1f2937")
        marker = top_markers.get(off_s, "o")
        ax_defect_runs.plot(
            detail["t_ramp_ns"],
            detail["observable"],
            color=color,
            marker=marker,
            ms=5,
            lw=1.8,
            label=label,
        )
        ax_order_runs.plot(
            detail["t_ramp_ns"],
            detail["selected_avg_S"],
            color=color,
            marker=marker,
            ms=5,
            lw=1.8,
            label=label,
        )

    ax_defect_runs.set_xscale("log")
    ax_defect_runs.grid(True, which="both", alpha=0.25)
    ax_defect_runs.set_title("2D defect density across the no-early-stop rate ladder")
    ax_defect_runs.set_xlabel(r"$t_{ramp}$ [ns]")
    ax_defect_runs.set_ylabel("defect density per plaquette")
    ax_defect_runs.legend(loc="best")

    ax_order_runs.set_xscale("log")
    ax_order_runs.grid(True, which="both", alpha=0.25)
    ax_order_runs.set_title("Selected scalar order at the same absolute windows")
    ax_order_runs.set_xlabel(r"$t_{ramp}$ [ns]")
    ax_order_runs.set_ylabel(r"selected $\langle S \rangle$")
    ax_order_runs.legend(loc="best")

    ax_slopes.axhline(0.0, color="#475569", lw=1.0, alpha=0.8)
    ax_slopes.axvspan(50.0, 60.0, color="#fde68a", alpha=0.25)
    ax_slopes.plot(offsets_ns, defect_slopes, color="#0f766e", marker="o", ms=5, lw=1.8, label="2D defect slope vs rate")
    ax_slopes.plot(offsets_ns, avg_slopes, color="#c2410c", marker="s", ms=5, lw=1.8, label=r"selected $\langle S \rangle$ slope vs rate")
    ax_slopes.plot(offsets_ns, xi_slopes, color="#334155", marker="^", ms=5, lw=1.8, label=r"$\xi_{grad}$ slope vs rate")
    for x_ns, y_val, count in zip(offsets_ns, defect_slopes, positive_cases, strict=True):
        if count < 4:
            ax_slopes.text(x_ns, y_val + 0.05, f"{count}/4", color="#0f766e", fontsize=8, ha="center")
    ax_slopes.text(50.5, ax_slopes.get_ylim()[1] * 0.82, "cleaner window", fontsize=9, color="#7c2d12")
    ax_slopes.grid(True, alpha=0.25)
    ax_slopes.set_title("Window slopes and state-lag diagnostics")
    ax_slopes.set_xlabel(r"measurement offset after $T_c$ [ns]")
    ax_slopes.set_ylabel("slope vs rate")
    ax_slopes.legend(loc="best")

    ax_ratios.axvspan(50.0, 60.0, color="#fde68a", alpha=0.25)
    ax_ratios.plot(offsets_ns, defect_ratios, color="#0f766e", marker="o", ms=5, lw=1.8, label="defect ratio max/min")
    ax_ratios.plot(offsets_ns, np.array(avg_s_ratios, dtype=float), color="#c2410c", marker="s", ms=5, lw=1.8, label=r"selected $\langle S \rangle$ ratio max/min")
    ax_ratios.plot(offsets_ns, np.array(xi_ratios, dtype=float), color="#334155", marker="^", ms=5, lw=1.8, label=r"selected $\xi_{grad}$ ratio max/min")
    ax_ratios.axhline(1.0, color="#475569", lw=1.0, alpha=0.8)
    ax_ratios.grid(True, alpha=0.25)
    ax_ratios.set_title("Spread collapse with later windows")
    ax_ratios.set_xlabel(r"measurement offset after $T_c$ [ns]")
    ax_ratios.set_ylabel("max/min across runs")
    ax_ratios.legend(loc="best")

    fig.suptitle(
        "Option-2 follow-up on the no-early-stop weak-anchor branch\n"
        "The 40 ns canonical sign is strong but state-mismatched; 50-60 ns remains weaker and cleaner",
        fontsize=14,
    )
    fig.text(
        0.06,
        0.012,
        (
            "The finalhold sweep used snapshot_mode=0, so 3D after-Tc aggregation on field files is degenerate.\n"
            "Meaningful option 2 on this branch comes from transient quench-log windows: 40 ns is lag-dominated, while 50-60 ns is weaker but cleaner."
        ),
        ha="left",
        va="bottom",
        fontsize=10,
        wrap=True,
    )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.17, wspace=0.24, hspace=0.22)

    png_path = args.png
    pdf_path = args.pdf
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