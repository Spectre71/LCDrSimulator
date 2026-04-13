#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

XY_FINAL_SUMMARY = ROOT / "validation" / "kzm_prooving_ground_xy_base_sweep" / "analysis_final" / "kzm_prooving_ground_summary.csv"
XY_FIXED_SUMMARY = ROOT / "validation" / "kzm_prooving_ground_xy_base_sweep" / "analysis_fixed_0p4" / "kzm_prooving_ground_summary.csv"

BULK_SWEEP_ROOT = ROOT / "validation" / "kzm_bulk_ldg_rate_sweep_initial"
BULK_FINAL_SUMMARY = BULK_SWEEP_ROOT / "analysis_final" / "kzm_bulk_ldg_summary.csv"
BULK_AVG_SUMMARY = BULK_SWEEP_ROOT / "analysis_avg_s" / "kzm_bulk_ldg_summary.csv"

BULK_PROTOCOL_SUMMARY = ROOT / "validation" / "kzm_bulk_ldg_protocol_convergence" / "protocol_comparison.csv"
BULK_PROTOCOL_FINAL = ROOT / "validation" / "kzm_bulk_ldg_protocol_convergence" / "final_state_comparison.csv"
BULK_PROTOCOL_RAMP600_SUMMARY = ROOT / "validation" / "kzm_bulk_ldg_protocol_convergence_ramp600" / "protocol_comparison.csv"
BULK_PROTOCOL_RAMP600_FINAL = ROOT / "validation" / "kzm_bulk_ldg_protocol_convergence_ramp600" / "final_state_comparison.csv"

CONFINED_CORE_SUMMARY = ROOT / "validation" / "sanity3_rate_sweep_merged_analysis" / "core" / "core_cohort_summary.csv"
CONFINED_SKELETON_SUMMARY = ROOT / "validation" / "sanity3_rate_sweep_merged_analysis" / "skeleton" / "skeleton_cohort_summary.csv"

CONFINED_BAND_SUMMARY = (
    ROOT
    / "validation"
    / "sanity3_rate_sweep_weak_anchor_W3em6_qtensor_window65_dense"
    / "analysis_shell_depth"
    / "band_decomposition"
    / "shell_band_common_summary.csv"
)
CONFINED_REGION_SUMMARY = (
    ROOT
    / "validation"
    / "sanity3_rate_sweep_weak_anchor_W3em6_qtensor_window65_dense"
    / "analysis_shell_depth"
    / "band_decomposition"
    / "shell_focus_region_summary.csv"
)
CONFINED_FOCUS_COMMON = (
    ROOT
    / "validation"
    / "sanity3_rate_sweep_weak_anchor_W3em6_qtensor_window65_dense"
    / "analysis_shell_depth"
    / "focus_exponent_fixed_2_10"
    / "focus_exponent_common_fit.csv"
)
CONFINED_FOCUS_WINDOW = (
    ROOT
    / "validation"
    / "sanity3_rate_sweep_weak_anchor_W3em6_qtensor_window65_dense"
    / "analysis_shell_depth"
    / "focus_exponent_fixed_2_10"
    / "focus_exponent_window_fits.csv"
)
CONFINED_FOCUS_DETAIL = (
    ROOT
    / "validation"
    / "sanity3_rate_sweep_weak_anchor_W3em6_qtensor_window65_dense"
    / "analysis_shell_depth"
    / "focus_exponent_fixed_2_10"
    / "focus_exponent_detail.csv"
)

ANCHOR_STRENGTH_SUMMARY = ROOT / "validation" / "anchoring_strength_point2" / "anchoring_strength_point2_summary.csv"

SPARSE100_SHELL_ROOT = ROOT / "validation" / "sanity3_rate_sweep_weak_anchor_W3em6_qtensor_window65"
SPARSE100_BAND_SUMMARY = SPARSE100_SHELL_ROOT / "analysis_shell_depth" / "band_decomposition" / "shell_band_common_summary.csv"
SPARSE100_FOCUS_COMMON = SPARSE100_SHELL_ROOT / "analysis_shell_depth" / "focus_exponent" / "focus_exponent_common_fit.csv"

SIZE200_MATCHED_RATE_SUMMARY = (
    ROOT
    / "validation"
    / "size200_pilot_W3em6_window65_sparse100"
    / "analysis_matched_rate"
    / "size200_vs_size100_ramp050_shell_depth.csv"
)
SIZE200_SHELL_ROOT = ROOT / "validation" / "sanity3_rate_sweep_weak_anchor_W3em6_qtensor_window65_size200_sparse100_4rate"
SIZE200_BAND_SUMMARY = SIZE200_SHELL_ROOT / "analysis_shell_depth" / "band_decomposition" / "shell_band_common_summary.csv"
SIZE200_FOCUS_COMMON = SIZE200_SHELL_ROOT / "analysis_shell_depth" / "focus_exponent" / "focus_exponent_common_fit.csv"
SIZE200_FOCUS_FIXED_COMMON = (
    SIZE200_SHELL_ROOT / "analysis_shell_depth" / "focus_exponent_fixed_2_10" / "focus_exponent_common_fit.csv"
)

XY_FIXED_OFFSETS = [3.6e-8, 4.0e-8, 4.4e-8, 5.0e-8, 6.0e-8, 7.0e-8]

COHORT_STYLES = {
    "all11": {"label": "all 11 runs", "color": "#111827", "marker": "o", "linestyle": "-"},
    "no1000_10": {"label": "25 to 700", "color": "#0f766e", "marker": "s", "linestyle": "--"},
    "slow5": {"label": "200 to 1000", "color": "#c2410c", "marker": "^", "linestyle": ":"},
}

REGION_COLORS = {
    "skin": "#64748b",
    "focus": "#b91c1c",
    "bulk": "#0f766e",
}


BASE_REQUIRED_ARTIFACTS: tuple[Path, ...] = (
    XY_FINAL_SUMMARY,
    XY_FIXED_SUMMARY,
    BULK_FINAL_SUMMARY,
    BULK_AVG_SUMMARY,
    BULK_PROTOCOL_SUMMARY,
    BULK_PROTOCOL_FINAL,
    BULK_PROTOCOL_RAMP600_SUMMARY,
    BULK_PROTOCOL_RAMP600_FINAL,
    CONFINED_CORE_SUMMARY,
    CONFINED_SKELETON_SUMMARY,
    CONFINED_BAND_SUMMARY,
    CONFINED_REGION_SUMMARY,
    CONFINED_FOCUS_COMMON,
    CONFINED_FOCUS_WINDOW,
    CONFINED_FOCUS_DETAIL,
    ANCHOR_STRENGTH_SUMMARY,
)

SIZE200_REQUIRED_ARTIFACTS: tuple[Path, ...] = (
    SPARSE100_BAND_SUMMARY,
    SPARSE100_FOCUS_COMMON,
    SIZE200_MATCHED_RATE_SUMMARY,
    SIZE200_BAND_SUMMARY,
    SIZE200_FOCUS_COMMON,
    SIZE200_FOCUS_FIXED_COMMON,
)


@dataclass
class ObservableSeries:
    x: np.ndarray
    y: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate production-grade benchmark and confined-bridge review figures from existing reduced artifacts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "pics" / "production",
        help="Directory where production PNG figures are written.",
    )
    parser.add_argument(
        "--skip-size200",
        action="store_true",
        help="Generate only the benchmark-to-dense-100^3 ladder and omit the 200^3 control figures.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str) -> float:
    text = value.strip()
    if not text:
        return float("nan")
    return float(text)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_detail_series(path: Path) -> ObservableSeries:
    rows = read_csv_rows(path)
    x = np.array([to_float(row["tau_Q_s"]) for row in rows], dtype=float)
    y = np.array([to_float(row["observable_value"]) for row in rows], dtype=float)
    order = np.argsort(x)
    return ObservableSeries(x=x[order], y=y[order])


def fit_loglog(series: ObservableSeries) -> tuple[float, float]:
    log_x = np.log(series.x)
    log_y = np.log(series.y)
    slope, intercept = np.polyfit(log_x, log_y, deg=1)
    return float(slope), float(intercept)


def intercept_with_fixed_slope(series: ObservableSeries, slope: float) -> float:
    log_x = np.log(series.x)
    log_y = np.log(series.y)
    return float(np.mean(log_y - slope * log_x))


def fixed_dir_label(raw_offset: float) -> str:
    text = f"{raw_offset:.1e}"
    return text.replace(".", "p").replace("-", "m").replace("+", "p").replace("em0", "em").replace("ep0", "ep")


def find_row(
    rows: list[dict[str, str]],
    observable: str,
    measure_mode: str,
    *,
    target_after_tc_s: float | None = None,
    target_avg_s: float | None = None,
) -> dict[str, str]:
    for row in rows:
        if row.get("observable") != observable:
            continue
        if row.get("measure_mode") != measure_mode:
            continue
        if target_after_tc_s is not None:
            row_target = to_float(row.get("target_after_Tc_s", "nan"))
            if np.isfinite(row_target) and abs(row_target - target_after_tc_s) <= 1e-18:
                return row
            continue
        if target_avg_s is not None:
            row_target = to_float(row.get("target_avg_S", row.get("target_avg_amp", "nan")))
            if np.isfinite(row_target) and abs(row_target - target_avg_s) <= 1e-12:
                return row
            continue
        return row
    raise ValueError(
        f"Could not find row for observable={observable}, measure_mode={measure_mode}, "
        f"target_after_tc_s={target_after_tc_s}, target_avg_s={target_avg_s}"
    )


def find_exact_row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    details = ", ".join(f"{key}={value}" for key, value in criteria.items())
    raise ValueError(f"Could not find row with {details}")


def required_input_paths(include_size200: bool) -> tuple[Path, ...]:
    if include_size200:
        return BASE_REQUIRED_ARTIFACTS + SIZE200_REQUIRED_ARTIFACTS
    return BASE_REQUIRED_ARTIFACTS


def validate_required_inputs(include_size200: bool) -> None:
    missing = [path for path in required_input_paths(include_size200) if not path.exists()]
    if not missing:
        return

    ladder_note = (
        "The full ladder is sensible only after the periodic benchmarks, the dense 100^3 fixed-[2,10) confined baseline, and the size-200 controls you want to cite have already been reduced and interpreted."
        if include_size200
        else "The benchmark-to-confined ladder is sensible only after the periodic benchmarks and the dense 100^3 fixed-[2,10) confined baseline have already been reduced and interpreted."
    )
    missing_lines = "\n".join(f"  - {display_path(path)}" for path in missing)
    raise FileNotFoundError(
        "Production review figures are built from existing reduced artifacts, not raw run directories.\n"
        f"Missing inputs:\n{missing_lines}\n\n{ladder_note}"
    )


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "savefig.dpi": 260,
        }
    )


def style_axis(ax: Axes, *, grid_axis: str = "both") -> None:
    if grid_axis == "both":
        ax.grid(True, which="both", alpha=0.18, linewidth=0.8)
    elif grid_axis == "y":
        ax.grid(True, axis="y", alpha=0.18, linewidth=0.8)
    else:
        ax.grid(True, axis="x", alpha=0.18, linewidth=0.8)


def save_figure(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def production_path(output_dir: Path, name: str) -> Path:
    return output_dir / name


def plot_xy_benchmark(output_dir: Path) -> Path:
    rows_final = read_csv_rows(XY_FINAL_SUMMARY)
    rows_fixed = read_csv_rows(XY_FIXED_SUMMARY)
    final_row = find_row(rows_final, "vortex_line_density", "final")
    fixed_row = find_row(rows_fixed, "vortex_line_density", "fixed", target_after_tc_s=0.4)

    final_series = read_detail_series(Path(final_row["detail_csv"]))
    fixed_series = read_detail_series(Path(fixed_row["detail_csv"]))

    final_slope, final_intercept = fit_loglog(final_series)
    fixed_slope, fixed_intercept = fit_loglog(fixed_series)
    expected_slope = to_float(final_row["expected_slope_vs_tauQ"])
    expected_intercept = intercept_with_fixed_slope(final_series, expected_slope)
    xfit = np.geomspace(final_series.x.min(), final_series.x.max(), 200)

    fig, (ax_main, ax_control) = plt.subplots(
        1,
        2,
        figsize=(11.8, 4.8),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )

    ax_main.loglog(final_series.x, final_series.y, linestyle="none", marker="o", ms=6.5, color="#0b6e4f")
    ax_main.loglog(
        xfit,
        np.exp(final_intercept) * np.power(xfit, final_slope),
        color="#0b6e4f",
        linewidth=2.0,
        label=fr"measured fit $m={final_slope:.3f}$",
    )
    ax_main.loglog(
        xfit,
        np.exp(expected_intercept) * np.power(xfit, expected_slope),
        color="#9a3412",
        linewidth=1.8,
        linestyle="--",
        label=fr"3D XY expectation $m={expected_slope:.3f}$",
    )
    ax_main.set_title("A. Final-state benchmark")
    ax_main.set_xlabel(r"Quench time $\tau_Q$")
    ax_main.set_ylabel("Final vortex-line density")
    style_axis(ax_main)
    ax_main.legend(loc="lower left", frameon=False)

    ax_control.loglog(final_series.x, final_series.y, linestyle="none", marker="o", ms=6.0, color="#0b6e4f")
    ax_control.loglog(
        xfit,
        np.exp(final_intercept) * np.power(xfit, final_slope),
        color="#0b6e4f",
        linewidth=1.8,
        label=fr"final-state $m={final_slope:.3f}$",
    )
    ax_control.loglog(fixed_series.x, fixed_series.y, linestyle="none", marker="s", ms=5.8, color="#2563eb")
    ax_control.loglog(
        xfit,
        np.exp(fixed_intercept) * np.power(xfit, fixed_slope),
        color="#2563eb",
        linewidth=1.6,
        label=fr"fixed $+0.4$ after $T_c$ $m={fixed_slope:.3f}$",
    )
    ax_control.set_title("B. Early-time control")
    ax_control.set_xlabel(r"Quench time $\tau_Q$")
    ax_control.set_ylabel("Vortex-line density")
    style_axis(ax_control)
    ax_control.legend(loc="lower left", frameon=False)

    path = production_path(output_dir, "01_periodic_xy_benchmark.png")
    save_figure(fig, path)
    return path


def plot_bulk_benchmark(output_dir: Path) -> Path:
    final_rows = read_csv_rows(BULK_FINAL_SUMMARY)
    avg_rows = read_csv_rows(BULK_AVG_SUMMARY)
    fixed_rows = [
        find_row(
            read_csv_rows(BULK_SWEEP_ROOT / f"analysis_fixed_{fixed_dir_label(offset)}" / "kzm_bulk_ldg_summary.csv"),
            "defect_line_density",
            "fixed",
            target_after_tc_s=offset,
        )
        for offset in XY_FIXED_OFFSETS
    ]

    final_row = find_row(final_rows, "defect_line_density", "final")
    final_series = read_detail_series(Path(final_row["detail_csv"]))
    final_slope, final_intercept = fit_loglog(final_series)
    xfit = np.geomspace(final_series.x.min(), final_series.x.max(), 200)

    avg_targets = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    avg_target_rows = [find_row(avg_rows, "defect_line_density", "avg_s", target_avg_s=value) for value in avg_targets]

    fig = plt.figure(figsize=(12.4, 6.6))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], hspace=0.32, wspace=0.24)
    ax_main = fig.add_subplot(grid[:, 0])
    ax_fixed = fig.add_subplot(grid[0, 1])
    ax_avg = fig.add_subplot(grid[1, 1])

    ax_main.loglog(final_series.x, final_series.y, linestyle="none", marker="o", ms=6.5, color="#0b6e4f")
    ax_main.loglog(
        xfit,
        np.exp(final_intercept) * np.power(xfit, final_slope),
        color="#0b6e4f",
        linewidth=2.0,
        label=fr"final-state fit $m={final_slope:.3f}$",
    )
    ax_main.set_title("A. Final-state bulk readout")
    ax_main.set_xlabel(r"Quench time $\tau_Q$ [s]")
    ax_main.set_ylabel("Final defect-line density")
    style_axis(ax_main)
    ax_main.legend(loc="lower left", frameon=False)

    fixed_x = np.array([to_float(row["target_after_Tc_s"]) * 1e9 for row in fixed_rows], dtype=float)
    fixed_alpha = -np.array([to_float(row["slope_vs_tauQ"]) for row in fixed_rows], dtype=float)
    final_alpha = -to_float(final_row["slope_vs_tauQ"])
    ax_fixed.plot(fixed_x, fixed_alpha, color="#2563eb", marker="s", ms=5.5, linewidth=1.8)
    ax_fixed.axhline(final_alpha, color="#0b6e4f", linestyle="--", linewidth=1.4)
    ax_fixed.set_title("B. Fixed after-$T_c$ windows")
    ax_fixed.set_xlabel(r"Measurement offset after $T_c$ [ns]")
    ax_fixed.set_ylabel(r"Apparent exponent $\alpha=-m$")
    style_axis(ax_fixed)

    avg_x = np.array(avg_targets, dtype=float)
    avg_alpha = -np.array([to_float(row["slope_vs_tauQ"]) for row in avg_target_rows], dtype=float)
    ax_avg.plot(avg_x, avg_alpha, color="#c2410c", marker="o", ms=5.8, linewidth=1.8)
    ax_avg.axhline(final_alpha, color="#0b6e4f", linestyle="--", linewidth=1.4)
    ax_avg.set_title("C. Matched-order windows")
    ax_avg.set_xlabel(r"Target average order $\langle S \rangle$")
    ax_avg.set_ylabel(r"Apparent exponent $\alpha=-m$")
    style_axis(ax_avg)

    path = production_path(output_dir, "02_periodic_bulk_ldg_benchmark.png")
    save_figure(fig, path)
    return path


def plot_bulk_protocol_convergence(output_dir: Path) -> Path:
    standard_rows = read_csv_rows(BULK_PROTOCOL_SUMMARY)
    ramp600_rows = read_csv_rows(BULK_PROTOCOL_RAMP600_SUMMARY)
    final_standard_rows = read_csv_rows(BULK_PROTOCOL_FINAL)
    final_ramp600_rows = read_csv_rows(BULK_PROTOCOL_RAMP600_FINAL)

    fig = plt.figure(figsize=(12.2, 7.0))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.34)
    ax_top = fig.add_subplot(grid[0, 0])
    ax_bottom = fig.add_subplot(grid[1, 0])

    metric_specs = [
        ("rel_total_diff", "total", "#334155"),
        ("rel_avg_S_diff", r"avg $S$", "#0f766e"),
        ("rel_xi_grad_proxy_diff", r"$\xi_{grad}$", "#b45309"),
    ]

    for rows, linestyle, suffix in ((standard_rows, "-", "dt/2"), (ramp600_rows, "--", "ramp600 dt/2")):
        x_ns = np.array([to_float(row["offset_s"]) * 1e9 for row in rows], dtype=float)
        for column, label, color in metric_specs:
            y = np.array([to_float(row[column]) for row in rows], dtype=float)
            mask = np.isfinite(y) & (y > 0.0)
            ax_top.semilogy(
                x_ns[mask],
                y[mask],
                color=color,
                linestyle=linestyle,
                linewidth=1.7,
                label=f"{label} ({suffix})",
            )

    ax_top.set_title("A. Bulk protocol refinement over the post-$T_c$ timeline")
    ax_top.set_xlabel(r"Offset after $T_c$ [ns]")
    ax_top.set_ylabel("Relative coarse/fine mismatch")
    style_axis(ax_top)
    ax_top.legend(loc="upper right", frameon=False, ncol=2)

    observables = ["avg_S", "xi_grad_proxy", "defect_density_per_plaquette", "defect_line_density"]
    labels = [r"avg $S$", r"$\xi_{grad}$", "plaquette defect", "line defect"]

    def final_rel(rows: list[dict[str, str]], observable: str) -> float:
        for row in rows:
            if row["observable"] == observable:
                return to_float(row["rel_diff"])
        raise ValueError(f"Missing final-state comparison for {observable}")

    standard_values = np.array([final_rel(final_standard_rows, observable) for observable in observables], dtype=float)
    ramp600_values = np.array([final_rel(final_ramp600_rows, observable) for observable in observables], dtype=float)
    display_floor = 1e-5
    x = np.arange(len(observables), dtype=float)
    width = 0.36

    standard_heights = np.where(standard_values > 0.0, standard_values, display_floor)
    ramp600_heights = np.where(ramp600_values > 0.0, ramp600_values, display_floor)
    bars_standard = ax_bottom.bar(x - width / 2.0, standard_heights, width, color="#475569", alpha=0.88, label="standard pair")
    bars_ramp600 = ax_bottom.bar(x + width / 2.0, ramp600_heights, width, color="#0f766e", alpha=0.88, label="ramp600 pair")
    ax_bottom.set_yscale("log")
    ax_bottom.set_ylim(display_floor, 2e-2)
    ax_bottom.set_xticks(x, labels)
    ax_bottom.set_ylabel("Final-state relative mismatch")
    ax_bottom.set_title("B. Final-state agreement after matched protocol refinement")
    style_axis(ax_bottom, grid_axis="y")
    ax_bottom.legend(loc="upper right", frameon=False)

    for bars, values in ((bars_standard, standard_values), (bars_ramp600, ramp600_values)):
        for bar, value in zip(bars, values, strict=True):
            if value == 0.0:
                ax_bottom.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    display_floor * 1.15,
                    "0",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color="#0f172a",
                )

    path = production_path(output_dir, "03_periodic_bulk_ldg_protocol_convergence.png")
    save_figure(fig, path)
    return path


def plot_confined_global_control(output_dir: Path) -> Path:
    core_rows = read_csv_rows(CONFINED_CORE_SUMMARY)
    skeleton_rows = read_csv_rows(CONFINED_SKELETON_SUMMARY)

    fig, (ax_core, ax_skeleton) = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)

    for ax, rows, title in (
        (ax_core, core_rows, "A. Whole-volume core proxy"),
        (ax_skeleton, skeleton_rows, "B. Whole-volume line proxy"),
    ):
        for cohort, style in COHORT_STYLES.items():
            subset = [row for row in rows if row["cohort"] == cohort]
            subset.sort(key=lambda row: to_float(row["target_after_Tc_s"]))
            x_ns = np.array([to_float(row["target_after_Tc_s"]) * 1e9 for row in subset], dtype=float)
            y = np.array([to_float(row["slope_defect_vs_rate"]) for row in subset], dtype=float)
            ax.plot(
                x_ns,
                y,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                ms=5.3,
                label=style["label"],
            )
        ax.axhline(0.0, color="#475569", linewidth=1.0)
        ax.axvspan(34.0, 36.0, color="#fde68a", alpha=0.28)
        ax.set_title(title)
        ax.set_xlabel(r"Measurement offset after $T_c$ [ns]")
        style_axis(ax)

    ax_core.set_ylabel("Slope of defect observable vs rate")
    ax_skeleton.legend(loc="lower right", frameon=False)

    path = production_path(output_dir, "04_confined_global_transient_control.png")
    save_figure(fig, path)
    return path


def plot_confined_localization(output_dir: Path) -> Path:
    band_rows = read_csv_rows(CONFINED_BAND_SUMMARY)
    region_rows = read_csv_rows(CONFINED_REGION_SUMMARY)

    fig, (ax_bands, ax_regions) = plt.subplots(1, 2, figsize=(12.4, 5.1))

    support = np.array([to_float(row["mean_support_fraction_across_offsets"]) for row in band_rows], dtype=float)
    slope = np.array([to_float(row["mean_density_slope_vs_rate"]) for row in band_rows], dtype=float)
    ax_bands.scatter(support, slope, color="#cbd5e1", edgecolor="none", s=38)

    highlight_labels = {
        "[2,10)": ("#b91c1c", "s", (6, 8)),
        "[2,6)": ("#2563eb", "o", (6, -10)),
        "[4,6)": ("#c2410c", "^", (6, 8)),
        "[0,10)": ("#0f766e", "D", (6, -10)),
        "[10,inf)": ("#475569", "P", (6, 8)),
    }
    for row in band_rows:
        label = row["band_label"]
        if label not in highlight_labels:
            continue
        color, marker, offset = highlight_labels[label]
        x_val = to_float(row["mean_support_fraction_across_offsets"])
        y_val = to_float(row["mean_density_slope_vs_rate"])
        ax_bands.scatter([x_val], [y_val], color=color, marker=marker, s=88, zorder=3)
        ax_bands.annotate(label, (x_val, y_val), xytext=offset, textcoords="offset points", fontsize=9, color=color)
    ax_bands.set_title("A. Candidate shell bands")
    ax_bands.set_xlabel("Mean support fraction across windows")
    ax_bands.set_ylabel(r"Mean density exponent $\alpha$")
    style_axis(ax_bands)

    for region in ("skin", "focus", "bulk"):
        subset = [row for row in region_rows if row["region"] == region]
        subset.sort(key=lambda row: to_float(row["target_after_Tc_s"]))
        x_ns = np.array([to_float(row["target_after_Tc_s"]) * 1e9 for row in subset], dtype=float)
        alpha = np.array([to_float(row["density_slope_vs_rate"]) for row in subset], dtype=float)
        ax_regions.plot(
            x_ns,
            alpha,
            color=REGION_COLORS[region],
            marker="o",
            ms=5.2,
            linewidth=1.8,
            label=row_label(region),
        )
    ax_regions.set_title("B. Region-resolved late-window exponents")
    ax_regions.set_xlabel(r"Measurement offset after $T_c$ [ns]")
    ax_regions.set_ylabel(r"Density exponent $\alpha$")
    style_axis(ax_regions)
    ax_regions.legend(loc="upper right", frameon=False)

    path = production_path(output_dir, "05_confined_shell_localization.png")
    save_figure(fig, path)
    return path


def row_label(region: str) -> str:
    return {
        "skin": "skin [0,2)",
        "focus": "focus [2,10)",
        "bulk": "bulk [10,inf)",
    }[region]


def plot_confined_exponent(output_dir: Path) -> Path:
    common_rows = read_csv_rows(CONFINED_FOCUS_COMMON)
    window_rows = read_csv_rows(CONFINED_FOCUS_WINDOW)
    detail_rows = read_csv_rows(CONFINED_FOCUS_DETAIL)

    focus_density_common = next(row for row in common_rows if row["region"] == "focus" and row["observable"] == "density")
    focus_share_common = next(row for row in common_rows if row["region"] == "focus" and row["observable"] == "defect_share")
    focus_window_rows = [row for row in window_rows if row["region"] == "focus"]
    focus_window_rows.sort(key=lambda row: to_float(row["offset_ns"]))

    colors = matplotlib.colormaps["plasma"](np.linspace(0.15, 0.85, len(focus_window_rows)))

    fig, (ax_main, ax_window) = plt.subplots(1, 2, figsize=(12.6, 5.0), gridspec_kw={"width_ratios": [1.45, 1.0]})

    for color, window_row in zip(colors, focus_window_rows, strict=True):
        offset_ns = int(round(to_float(window_row["offset_ns"])))
        subset = [
            row
            for row in detail_rows
            if row["region"] == "focus" and int(round(to_float(row["offset_ns"]))) == offset_ns
        ]
        subset.sort(key=lambda row: to_float(row["tau_Q_s"]))
        tau_q = np.array([to_float(row["tau_Q_s"]) for row in subset], dtype=float)
        density = np.array([to_float(row["density"]) for row in subset], dtype=float)
        xfit = np.geomspace(tau_q.min(), tau_q.max(), 120)
        intercept_key = f"window_{offset_ns}ns_intercept_tau"
        intercept = to_float(focus_density_common[intercept_key])
        slope_tau = to_float(focus_density_common["slope_tauQ_common"])
        ax_main.loglog(tau_q, density, linestyle="none", marker="o", ms=5.7, color=color)
        ax_main.loglog(
            xfit,
            np.exp(intercept) * np.power(xfit, slope_tau),
            color=color,
            linewidth=1.5,
            label=f"{offset_ns} ns",
        )

    ax_main.set_title("A. Fixed [2,10) annulus across late windows")
    ax_main.set_xlabel(r"Quench time $\tau_Q$ [s]")
    ax_main.set_ylabel("Focus-band defect density")
    style_axis(ax_main)
    ax_main.legend(loc="lower left", frameon=False, ncol=2)

    x_ns = np.array([to_float(row["offset_ns"]) for row in focus_window_rows], dtype=float)
    alpha_density = np.array([to_float(row["alpha_rate_density"]) for row in focus_window_rows], dtype=float)
    alpha_share = np.array([to_float(row["alpha_rate_defect_share"]) for row in focus_window_rows], dtype=float)
    pooled_density = to_float(focus_density_common["alpha_rate_common"])
    pooled_share = to_float(focus_share_common["alpha_rate_common"])
    half_range = to_float(focus_density_common["window_half_range_alpha_rate"])
    ci95 = to_float(focus_density_common["alpha_rate_ci95_norm"])

    ax_window.plot(x_ns, alpha_density, color="#b91c1c", marker="o", ms=5.6, linewidth=1.8, label="density")
    ax_window.plot(x_ns, alpha_share, color="#1d4ed8", marker="s", ms=5.0, linewidth=1.6, label="defect share")
    ax_window.axhline(pooled_density, color="#7f1d1d", linestyle="--", linewidth=1.4)
    ax_window.axhline(pooled_share, color="#1e3a8a", linestyle=":", linewidth=1.4)
    ax_window.fill_between(x_ns, pooled_density - half_range, pooled_density + half_range, color="#fecaca", alpha=0.48)
    ax_window.fill_between(x_ns, pooled_density - ci95, pooled_density + ci95, color="#fca5a5", alpha=0.22)
    ax_window.set_title("B. Pooled exponent stability")
    ax_window.set_xlabel(r"Measurement offset after $T_c$ [ns]")
    ax_window.set_ylabel(r"Positive exponent $\alpha$")
    style_axis(ax_window)
    ax_window.legend(loc="upper right", frameon=False)

    path = production_path(output_dir, "06_confined_fixed_band_exponent.png")
    save_figure(fig, path)
    return path


def plot_anchor_strength(output_dir: Path) -> Path:
    rows = read_csv_rows(ANCHOR_STRENGTH_SUMMARY)
    rows.sort(key=lambda row: to_float(row["W"]))
    x = np.array([to_float(row["W"]) for row in rows], dtype=float)
    fixed_alpha = np.array([to_float(row["fixed_alpha"]) for row in rows], dtype=float)
    fixed_ci = np.array([to_float(row["fixed_ci95"]) for row in rows], dtype=float)
    auto_alpha = np.array([to_float(row["auto_alpha"]) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.errorbar(x, fixed_alpha, yerr=fixed_ci, fmt="o-", color="#b91c1c", capsize=4, linewidth=1.8, label="fixed [2,10)")
    ax.plot(x, auto_alpha, "s--", color="#2563eb", linewidth=1.5, ms=5.2, label="auto-selected band")
    ax.set_xscale("log")
    ax.set_title("Anchoring dependence on a fixed confined readout")
    ax.set_xlabel(r"Weak anchoring strength $W$")
    ax.set_ylabel(r"Late-window density exponent $\alpha$")
    style_axis(ax)
    ax.legend(loc="upper right", frameon=False)

    for row in rows:
        ax.annotate(
            row["auto_band_label"],
            (to_float(row["W"]), to_float(row["auto_alpha"])),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color="#1d4ed8",
        )

    path = production_path(output_dir, "07_confined_anchor_strength_comparison.png")
    save_figure(fig, path)
    return path


def plot_size200_matched_rate_shift(output_dir: Path) -> Path:
    rows = read_csv_rows(SIZE200_MATCHED_RATE_SUMMARY)
    rows.sort(key=lambda row: to_float(row["offset_ns"]))

    x_ns = np.array([to_float(row["offset_ns"]) for row in rows], dtype=float)

    fig = plt.figure(figsize=(13.4, 4.8))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 0.95], wspace=0.28)
    ax_density = fig.add_subplot(grid[0, 0])
    ax_weights = fig.add_subplot(grid[0, 1])
    ax_depth = fig.add_subplot(grid[0, 2])

    density_specs = (
        ("total_density_200", "total_density_100", "all defects", "#111827", "o"),
        ("focus_density_200", "focus_density_100", "focus [2,10)", "#b91c1c", "s"),
        ("skin_density_200", "skin_density_100", "skin [0,2)", "#64748b", "^"),
        ("bulk_density_200", "bulk_density_100", "bulk [10,inf)", "#0f766e", "D"),
    )
    for num_key, den_key, label, color, marker in density_specs:
        ratio = np.array([to_float(row[num_key]) / to_float(row[den_key]) for row in rows], dtype=float)
        ax_density.plot(x_ns, ratio, color=color, marker=marker, ms=5.3, linewidth=1.8, label=label)
    ax_density.axhline(1.0, color="#475569", linewidth=1.0, linestyle="--")
    ax_density.set_title("A. Matched-rate density ratios")
    ax_density.set_xlabel(r"Late window after $T_c$ [ns]")
    ax_density.set_ylabel(r"$200^3 / 100^3$")
    style_axis(ax_density)
    ax_density.legend(loc="lower right", frameon=False)

    support_ratio = np.array(
        [to_float(row["focus_support_frac_200"]) / to_float(row["focus_support_frac_100"]) for row in rows],
        dtype=float,
    )
    defect_ratio = np.array(
        [to_float(row["focus_defect_frac_200"]) / to_float(row["focus_defect_frac_100"]) for row in rows],
        dtype=float,
    )
    ax_weights.plot(x_ns, support_ratio, color="#1d4ed8", marker="o", ms=5.2, linewidth=1.8, label="focus support")
    ax_weights.plot(x_ns, defect_ratio, color="#c2410c", marker="s", ms=5.2, linewidth=1.8, label="focus defect share")
    ax_weights.axhline(1.0, color="#475569", linewidth=1.0, linestyle="--")
    ax_weights.set_title("B. Shell-adjacent weight loss")
    ax_weights.set_xlabel(r"Late window after $T_c$ [ns]")
    ax_weights.set_ylabel(r"$200^3 / 100^3$")
    style_axis(ax_weights)
    ax_weights.legend(loc="upper right", frameon=False)

    support_depth = np.array([to_float(row["support_depth_mean_200"]) for row in rows], dtype=float)
    defect_depth = np.array([to_float(row["defect_depth_mean_200"]) for row in rows], dtype=float)
    ax_depth.plot(x_ns, support_depth, color="#2563eb", marker="o", ms=5.2, linewidth=1.8, label="support-weighted")
    ax_depth.plot(x_ns, defect_depth, color="#b91c1c", marker="s", ms=5.2, linewidth=1.8, label="defect-weighted")
    ax_depth.set_title(r"C. $200^3$ shell-depth moments")
    ax_depth.set_xlabel(r"Late window after $T_c$ [ns]")
    ax_depth.set_ylabel("Mean shell depth [layers]")
    style_axis(ax_depth)
    ax_depth.legend(loc="upper left", frameon=False)

    path = production_path(output_dir, "08_confined_size200_matched_rate_shift.png")
    save_figure(fig, path)
    return path


def plot_size200_sparse_ladder(output_dir: Path) -> Path:
    sparse100_band_rows = read_csv_rows(SPARSE100_BAND_SUMMARY)
    sparse100_common_rows = read_csv_rows(SPARSE100_FOCUS_COMMON)
    size200_band_rows = read_csv_rows(SIZE200_BAND_SUMMARY)
    size200_common_rows = read_csv_rows(SIZE200_FOCUS_COMMON)
    size200_fixed_common_rows = read_csv_rows(SIZE200_FOCUS_FIXED_COMMON)
    dense100_fixed_rows = read_csv_rows(CONFINED_FOCUS_COMMON)

    fig, (ax_bands, ax_compare) = plt.subplots(1, 2, figsize=(13.0, 5.1), gridspec_kw={"width_ratios": [1.2, 1.0]})

    sparse100_support = np.array(
        [to_float(row["mean_support_fraction_across_offsets"]) for row in sparse100_band_rows],
        dtype=float,
    )
    sparse100_alpha = np.array([to_float(row["mean_density_slope_vs_rate"]) for row in sparse100_band_rows], dtype=float)
    size200_support = np.array(
        [to_float(row["mean_support_fraction_across_offsets"]) for row in size200_band_rows],
        dtype=float,
    )
    size200_alpha = np.array([to_float(row["mean_density_slope_vs_rate"]) for row in size200_band_rows], dtype=float)

    ax_bands.scatter(sparse100_support, sparse100_alpha, color="#93c5fd", edgecolor="none", s=36, alpha=0.85, label="100^3 sparse bands")
    ax_bands.scatter(size200_support, size200_alpha, color="#fca5a5", edgecolor="none", s=40, alpha=0.8, label="200^3 sparse bands")

    highlight_specs = (
        (sparse100_band_rows, "[2,6)", "#1d4ed8", "o", "100^3 [2,6)", (-58, 10)),
        (size200_band_rows, "[2,6)", "#b91c1c", "s", "200^3 [2,6)", (8, 8)),
        (sparse100_band_rows, "[2,10)", "#0f766e", "D", "100^3 [2,10)", (8, 8)),
        (size200_band_rows, "[2,10)", "#c2410c", "^", "200^3 [2,10)", (8, -14)),
    )
    for rows, label, color, marker, note, offset in highlight_specs:
        row = find_exact_row(rows, band_label=label)
        x_val = to_float(row["mean_support_fraction_across_offsets"])
        y_val = to_float(row["mean_density_slope_vs_rate"])
        ax_bands.scatter([x_val], [y_val], color=color, marker=marker, s=92, zorder=3)
        ax_bands.annotate(note, (x_val, y_val), xytext=offset, textcoords="offset points", fontsize=8.8, color=color)

    ax_bands.set_title("A. Sparse-ladder band selection by droplet size")
    ax_bands.set_xlabel("Mean support fraction across windows")
    ax_bands.set_ylabel(r"Mean density exponent $\alpha$")
    style_axis(ax_bands)
    ax_bands.legend(loc="lower right", frameon=False)

    sparse100_focus = find_exact_row(sparse100_common_rows, region="focus", observable="density")
    size200_focus = find_exact_row(size200_common_rows, region="focus", observable="density")
    dense100_fixed = find_exact_row(dense100_fixed_rows, region="focus", observable="density")
    size200_fixed = find_exact_row(size200_fixed_common_rows, region="focus", observable="density")
    size200_bulk = find_exact_row(size200_common_rows, region="bulk", observable="density")

    compare_cases = (
        ("100^3 sparse\nauto [2,6)", to_float(sparse100_focus["alpha_rate_common"]), to_float(sparse100_focus["alpha_rate_ci95_norm"]), "#2563eb"),
        ("200^3 sparse\nauto [2,6)", to_float(size200_focus["alpha_rate_common"]), to_float(size200_focus["alpha_rate_ci95_norm"]), "#b91c1c"),
        ("100^3 dense\nfixed [2,10)", to_float(dense100_fixed["alpha_rate_common"]), to_float(dense100_fixed["alpha_rate_ci95_norm"]), "#0f766e"),
        ("200^3 sparse\nfixed [2,10)", to_float(size200_fixed["alpha_rate_common"]), to_float(size200_fixed["alpha_rate_ci95_norm"]), "#c2410c"),
        ("200^3 sparse\nbulk [10,inf)", to_float(size200_bulk["alpha_rate_common"]), to_float(size200_bulk["alpha_rate_ci95_norm"]), "#475569"),
    )
    x = np.arange(len(compare_cases), dtype=float)
    values = np.array([case[1] for case in compare_cases], dtype=float)
    errors = np.array([case[2] for case in compare_cases], dtype=float)
    colors = [case[3] for case in compare_cases]

    bars = ax_compare.bar(x, values, color=colors, alpha=0.9)
    ax_compare.errorbar(x, values, yerr=errors, fmt="none", ecolor="#0f172a", elinewidth=1.0, capsize=3)
    ax_compare.axhline(0.0, color="#475569", linewidth=1.0)
    ax_compare.set_xticks(x, [case[0] for case in compare_cases])
    ax_compare.set_title("B. Size-200 sparse ladder versus the active dense baseline")
    ax_compare.set_ylabel(r"Positive exponent $\alpha$")
    style_axis(ax_compare, grid_axis="y")

    for bar, value in zip(bars, values, strict=True):
        ax_compare.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.018,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.7,
            color="#0f172a",
        )

    path = production_path(output_dir, "09_confined_size200_sparse_ladder.png")
    save_figure(fig, path)
    return path


def main() -> int:
    args = parse_args()
    setup_matplotlib()
    output_dir = args.output_dir.resolve()
    include_size200 = not args.skip_size200

    validate_required_inputs(include_size200)

    figure_paths = [
        plot_xy_benchmark(output_dir),
        plot_bulk_benchmark(output_dir),
        plot_bulk_protocol_convergence(output_dir),
        plot_confined_global_control(output_dir),
        plot_confined_localization(output_dir),
        plot_confined_exponent(output_dir),
        plot_anchor_strength(output_dir),
    ]

    if include_size200:
        figure_paths.extend(
            [
                plot_size200_matched_rate_shift(output_dir),
                plot_size200_sparse_ladder(output_dir),
            ]
        )

    for path in figure_paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())