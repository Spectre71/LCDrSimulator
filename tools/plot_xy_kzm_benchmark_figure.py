from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ObservableSeries:
    tau_q_s: np.ndarray
    observable_value: np.ndarray


@dataclass
class SummaryRow:
    observable: str
    slope_vs_tauq: float
    expected_slope_vs_tauq: float
    corr_vs_tauq: float
    actual_after_tc_min_s: float
    actual_after_tc_max_s: float
    measure_mode: str
    target_after_tc_s: float | None
    detail_csv: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the proving-ground KZM benchmark figure from existing analysis artifacts."
    )
    parser.add_argument(
        "--final-summary",
        type=Path,
        default=Path("validation/kzm_prooving_ground_xy_base_sweep/analysis_final/kzm_prooving_ground_summary.csv"),
        help="Summary CSV for the final-state analysis.",
    )
    parser.add_argument(
        "--fixed-summary",
        type=Path,
        default=Path("validation/kzm_prooving_ground_xy_base_sweep/analysis_fixed_0p4/kzm_prooving_ground_summary.csv"),
        help="Summary CSV for the early fixed-after-Tc control analysis.",
    )
    parser.add_argument(
        "--observable",
        default="vortex_line_density",
        help="Observable to plot from the summary files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pics/kzm_prooving_ground_xy_kzm_2026-04-11.png"),
        help="Main output image path.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("pics/kzm_prooving_ground_xy_kzm_2026-04-11.pdf"),
        help="Optional PDF output path.",
    )
    return parser.parse_args()


def read_summary_row(summary_path: Path, observable: str) -> SummaryRow:
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("observable") != observable:
                continue
            target_raw = row.get("target_after_Tc_s", "")
            target_after_tc_s = None
            if target_raw and target_raw.lower() != "nan":
                target_after_tc_s = float(target_raw)
            return SummaryRow(
                observable=row["observable"],
                slope_vs_tauq=float(row["slope_vs_tauQ"]),
                expected_slope_vs_tauq=float(row["expected_slope_vs_tauQ"]),
                corr_vs_tauq=float(row["corr_vs_tauQ"]),
                actual_after_tc_min_s=float(row["actual_after_Tc_min_s"]),
                actual_after_tc_max_s=float(row["actual_after_Tc_max_s"]),
                measure_mode=row["measure_mode"],
                target_after_tc_s=target_after_tc_s,
                detail_csv=Path(row["detail_csv"]),
            )
    raise ValueError(f"Observable {observable!r} not found in {summary_path}")


def read_detail_csv(path: Path) -> ObservableSeries:
    tau_q: list[float] = []
    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tau_q.append(float(row["tau_Q_s"]))
            values.append(float(row["observable_value"]))
    order = np.argsort(tau_q)
    tau_q_arr = np.asarray(tau_q, dtype=float)[order]
    values_arr = np.asarray(values, dtype=float)[order]
    return ObservableSeries(tau_q_s=tau_q_arr, observable_value=values_arr)


def fit_loglog(series: ObservableSeries) -> tuple[float, float]:
    logx = np.log(series.tau_q_s)
    logy = np.log(series.observable_value)
    slope, intercept = np.polyfit(logx, logy, deg=1)
    return float(slope), float(intercept)


def intercept_with_fixed_slope(series: ObservableSeries, slope: float) -> float:
    logx = np.log(series.tau_q_s)
    logy = np.log(series.observable_value)
    return float(np.mean(logy - slope * logx))


def plot_series_with_fit(
    ax: Axes,
    series: ObservableSeries,
    slope: float,
    intercept: float,
    *,
    label: str,
    color: str,
    marker: str,
) -> None:
    xfit = np.geomspace(series.tau_q_s.min(), series.tau_q_s.max(), 200)
    yfit = np.exp(intercept) * np.power(xfit, slope)
    ax.loglog(series.tau_q_s, series.observable_value, linestyle="none", marker=marker, ms=8.0, color=color, label=label)
    ax.loglog(xfit, yfit, linewidth=2.0, color=color)


def main() -> int:
    args = parse_args()

    final_row = read_summary_row(args.final_summary, args.observable)
    fixed_row = read_summary_row(args.fixed_summary, args.observable)

    final_series = read_detail_csv(final_row.detail_csv)
    fixed_series = read_detail_csv(fixed_row.detail_csv)

    final_fit_slope, final_fit_intercept = fit_loglog(final_series)
    fixed_fit_slope, fixed_fit_intercept = fit_loglog(fixed_series)
    expected_intercept = intercept_with_fixed_slope(final_series, final_row.expected_slope_vs_tauq)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )

    fig, (ax_main, ax_control) = plt.subplots(
        1,
        2,
        figsize=(13.5, 6.4),
        gridspec_kw={"width_ratios": [1.65, 1.0]},
    )

    plot_series_with_fit(
        ax_main,
        final_series,
        final_fit_slope,
        final_fit_intercept,
        label="Measured final-state vortex-line density",
        color="#0b6e4f",
        marker="o",
    )
    xfit = np.geomspace(final_series.tau_q_s.min(), final_series.tau_q_s.max(), 200)
    y_expected = np.exp(expected_intercept) * np.power(xfit, final_row.expected_slope_vs_tauq)
    ax_main.loglog(xfit, y_expected, linestyle="--", linewidth=2.0, color="#9a3412", label="3D XY Model-A expectation")
    ax_main.set_title("Recovered KZM signal in the 64^3 periodic XY benchmark")
    ax_main.set_xlabel(r"Quench time $\tau_Q$ [simulation time]")
    ax_main.set_ylabel("Final vortex-line density")
    ax_main.grid(True, which="both", alpha=0.22)
    ax_main.legend(loc="lower left", frameon=False)
    ax_main.text(
        0.03,
        0.97,
        (
            f"Fit slope: {final_row.slope_vs_tauq:.3f}\n"
            f"Expected: {final_row.expected_slope_vs_tauq:.3f}\n"
            f"corr(log-log): {final_row.corr_vs_tauq:.3f}\n"
            f"final readout occurs {final_row.actual_after_tc_min_s:.1f}-{final_row.actual_after_tc_max_s:.1f} after $T_c$"
        ),
        transform=ax_main.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cbd5e1"},
    )

    plot_series_with_fit(
        ax_control,
        final_series,
        final_fit_slope,
        final_fit_intercept,
        label=f"final readout ({final_row.slope_vs_tauq:.3f})",
        color="#0b6e4f",
        marker="o",
    )
    plot_series_with_fit(
        ax_control,
        fixed_series,
        fixed_fit_slope,
        fixed_fit_intercept,
        label=f"fixed +0.4 after Tc ({fixed_row.slope_vs_tauq:.3f})",
        color="#2563eb",
        marker="s",
    )
    ax_control.set_title("Why this readout was chosen")
    ax_control.set_xlabel(r"Quench time $\tau_Q$ [simulation time]")
    ax_control.set_ylabel("Vortex-line density")
    ax_control.grid(True, which="both", alpha=0.22)
    ax_control.legend(loc="lower left", frameon=False)

    fig.suptitle("Kibble-Zurek proving ground before the confined-droplet branch", fontsize=16, y=0.98)
    fig.text(
        0.06,
        0.03,
        (
            "Left: the main benchmark result. Slower quenches leave fewer final vortex lines, and the fitted power law is already close to the 3D XY "
            "Kibble-Zurek expectation. Right: an early fixed-after-Tc readout is nearly flat, so the proof-of-concept signal is currently the later "
            "topological readout rather than an arbitrary early snapshot. This is the benchmark figure that motivates the next ladder step: periodic bulk LdG, then confined droplets."
        ),
        ha="left",
        va="bottom",
        fontsize=10.5,
        wrap=True,
    )

    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    if args.output_pdf:
        args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_pdf)
    plt.close(fig)

    print(f"[plot-kzm] Wrote {args.output}")
    if args.output_pdf:
        print(f"[plot-kzm] Wrote {args.output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())