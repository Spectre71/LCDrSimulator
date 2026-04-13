from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ObservableSeries:
    tau_q_s: np.ndarray
    observable_value: np.ndarray


@dataclass
class SummaryRow:
    observable: str
    measure_mode: str
    target_after_tc_s: float | None
    target_avg_s: float | None
    actual_after_tc_min_s: float
    actual_after_tc_max_s: float
    selected_avg_s_min: float
    selected_avg_s_max: float
    positive_case_count: int
    slope_vs_tauq: float
    corr_vs_tauq: float
    observable_ratio: float
    detail_csv: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the first periodic bulk-LdG KZM window-scan figure from existing analysis artifacts."
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("validation/kzm_bulk_ldg_rate_sweep_initial"),
        help="Sweep root containing the analysis_final, analysis_avg_s, and analysis_fixed_* outputs.",
    )
    parser.add_argument(
        "--observable",
        default="defect_line_density",
        help="Observable to plot from the summary files.",
    )
    parser.add_argument(
        "--fixed-offsets",
        default="3.6e-8,4.0e-8,4.4e-8,5.0e-8,6.0e-8,7.0e-8",
        help="Comma-separated fixed after-Tc offsets to include in the window comparison.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pics/kzm_bulk_ldg_initial_scan_2026-04-11.png"),
        help="Main output image path.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("pics/kzm_bulk_ldg_initial_scan_2026-04-11.pdf"),
        help="Optional PDF output path.",
    )
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_string_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def fixed_dir_label(raw_offset: str) -> str:
    safe = raw_offset.replace(".", "p").replace("-", "m").replace("+", "p")
    safe = safe.replace("em0", "em").replace("ep0", "ep")
    return safe


def read_summary_rows(summary_path: Path) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            target_after_tc_s = None
            if row["target_after_Tc_s"] and row["target_after_Tc_s"].lower() != "nan":
                target_after_tc_s = float(row["target_after_Tc_s"])
            target_avg_s = None
            if row["target_avg_S"] and row["target_avg_S"].lower() != "nan":
                target_avg_s = float(row["target_avg_S"])
            rows.append(
                SummaryRow(
                    observable=row["observable"],
                    measure_mode=row["measure_mode"],
                    target_after_tc_s=target_after_tc_s,
                    target_avg_s=target_avg_s,
                    actual_after_tc_min_s=float(row["actual_after_Tc_min_s"]),
                    actual_after_tc_max_s=float(row["actual_after_Tc_max_s"]),
                    selected_avg_s_min=float(row["selected_avg_S_min"]),
                    selected_avg_s_max=float(row["selected_avg_S_max"]),
                    positive_case_count=int(row["positive_case_count"]),
                    slope_vs_tauq=float(row["slope_vs_tauQ"]),
                    corr_vs_tauq=float(row["corr_vs_tauQ"]),
                    observable_ratio=float(row["observable_ratio_max_over_min"]),
                    detail_csv=Path(row["detail_csv"]),
                )
            )
    return rows


def find_row(
    rows: list[SummaryRow],
    observable: str,
    measure_mode: str,
    *,
    target_after_tc_s: float | None = None,
    target_avg_s: float | None = None,
) -> SummaryRow:
    for row in rows:
        if row.observable != observable or row.measure_mode != measure_mode:
            continue
        if target_after_tc_s is not None:
            if row.target_after_tc_s is not None and abs(row.target_after_tc_s - target_after_tc_s) <= 1e-18:
                return row
            continue
        if target_avg_s is not None:
            if row.target_avg_s is not None and abs(row.target_avg_s - target_avg_s) <= 1e-12:
                return row
            continue
        return row
    raise ValueError(
        f"No summary row found for observable={observable}, measure_mode={measure_mode}, "
        f"target_after_tc_s={target_after_tc_s}, target_avg_s={target_avg_s}"
    )


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


def main() -> int:
    args = parse_args()
    sweep_root = args.sweep_root.resolve()

    final_rows = read_summary_rows(sweep_root / "analysis_final" / "kzm_bulk_ldg_summary.csv")
    avg_rows = read_summary_rows(sweep_root / "analysis_avg_s" / "kzm_bulk_ldg_summary.csv")
    fixed_offset_strings = parse_string_list(args.fixed_offsets)
    fixed_offsets = [float(item) for item in fixed_offset_strings]
    fixed_rows = [
        find_row(
            read_summary_rows(
                sweep_root / f"analysis_fixed_{fixed_dir_label(raw_off)}" / "kzm_bulk_ldg_summary.csv"
            ),
            args.observable,
            "fixed",
            target_after_tc_s=off,
        )
        for raw_off, off in zip(fixed_offset_strings, fixed_offsets, strict=True)
    ]

    final_row = find_row(final_rows, args.observable, "final")
    final_series = read_detail_csv(final_row.detail_csv)
    final_fit_slope, final_fit_intercept = fit_loglog(final_series)

    avg_target_rows = [
        find_row(avg_rows, args.observable, "avg_s", target_avg_s=target)
        for target in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
    ]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9.5,
        }
    )

    fig = plt.figure(figsize=(14.2, 8.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], hspace=0.28, wspace=0.24)
    ax_main = fig.add_subplot(grid[:, 0])
    ax_fixed = fig.add_subplot(grid[0, 1])
    ax_avg = fig.add_subplot(grid[1, 1])

    xfit = np.geomspace(final_series.tau_q_s.min(), final_series.tau_q_s.max(), 200)
    yfit = np.exp(final_fit_intercept) * np.power(xfit, final_fit_slope)
    ax_main.loglog(final_series.tau_q_s, final_series.observable_value, linestyle="none", marker="o", ms=8.0, color="#0b6e4f")
    ax_main.loglog(xfit, yfit, linewidth=2.2, color="#0b6e4f", label="final-state fit")
    ax_main.set_title("Current best bulk-LdG readout: final defect-line density")
    ax_main.set_xlabel(r"Quench time $\tau_Q$ [s]")
    ax_main.set_ylabel("Final defect-line density")
    ax_main.grid(True, which="both", alpha=0.22)
    ax_main.legend(loc="lower left", frameon=False)
    ax_main.text(
        0.03,
        0.97,
        (
            f"fit slope vs $\\tau_Q$: {final_row.slope_vs_tauq:.3f}\n"
            f"corr(log-log): {final_row.corr_vs_tauq:.3f}\n"
            f"spread across rates: {final_row.observable_ratio:.2f}x\n"
            f"final readout occurs {final_row.actual_after_tc_min_s * 1e9:.1f}-{final_row.actual_after_tc_max_s * 1e9:.1f} ns after $T_c$"
        ),
        transform=ax_main.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cbd5e1"},
    )

    fixed_x_ns = np.array([row.target_after_tc_s * 1e9 for row in fixed_rows], dtype=float)
    fixed_y = np.array([row.slope_vs_tauq for row in fixed_rows], dtype=float)
    fixed_counts = [row.positive_case_count for row in fixed_rows]
    ax_fixed.plot(fixed_x_ns, fixed_y, color="#2563eb", marker="s", ms=6.0, linewidth=1.8)
    ax_fixed.axhline(final_row.slope_vs_tauq, color="#0b6e4f", linestyle="--", linewidth=1.5, label="final-state slope")
    ax_fixed.set_title("Fixed after-$T_c$ windows are transient")
    ax_fixed.set_xlabel("Measurement offset after $T_c$ [ns]")
    ax_fixed.set_ylabel(r"Fitted slope vs $\tau_Q$")
    ax_fixed.grid(True, alpha=0.22)
    for x_ns, y_val, count in zip(fixed_x_ns, fixed_y, fixed_counts, strict=True):
        if count < 7:
            ax_fixed.text(x_ns, y_val + 0.08, f"{count}/7", color="#1d4ed8", fontsize=8, ha="center")
    ax_fixed.legend(loc="lower right", frameon=False)

    avg_x = np.array([row.target_avg_s for row in avg_target_rows], dtype=float)
    avg_y = np.array([row.slope_vs_tauq for row in avg_target_rows], dtype=float)
    ax_avg.plot(avg_x, avg_y, color="#c2410c", marker="o", ms=6.0, linewidth=1.8, label="matched-<S> windows")
    ax_avg.axhline(final_row.slope_vs_tauq, color="#0b6e4f", linestyle="--", linewidth=1.5, label="final-state slope")
    ax_avg.set_title("Matched-<S> windows are smoother but weaker")
    ax_avg.set_xlabel(r"Target average order $\langle S \rangle$")
    ax_avg.set_ylabel(r"Fitted slope vs $\tau_Q$")
    ax_avg.grid(True, alpha=0.22)
    ax_avg.legend(loc="lower right", frameon=False)

    fig.suptitle("First periodic bulk-LdG KZM window scan before confinement", fontsize=16, y=0.98)
    fig.text(
        0.06,
        0.02,
        (
            "Left: the first seven-rate periodic bulk-LdG scan already gives a clean final-state topology signal. "
            "Top-right: fixed absolute-time windows near defect turn-on are too transient and can be sparse or over-steep. "
            "Bottom-right: matched-order windows are more coherent, but weaker than the final-state readout. "
            "Current conclusion: use the final bulk defect readout as the bridge between the XY proving ground and the later confined-droplet branch."
        ),
        ha="left",
        va="bottom",
        fontsize=10.2,
        wrap=True,
    )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.18, wspace=0.24, hspace=0.30)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    if args.output_pdf:
        args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_pdf)
    plt.close(fig)

    print(f"[plot-kzm-bulk-ldg] Wrote {args.output}")
    if args.output_pdf:
        print(f"[plot-kzm-bulk-ldg] Wrote {args.output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())