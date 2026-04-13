#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import QSRvis as qv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose the existing shell-depth outputs into defendable shell bands, choose a stable "
            "common focus annulus across multiple post-Tc windows, and compare that focus band against "
            "the outer skin and deeper bulk interior."
        ),
    )
    parser.add_argument("sweep_root", type=Path, help="Sweep root containing analysis_shell_depth outputs.")
    parser.add_argument(
        "--offsets",
        type=str,
        default="5.0e-8,6.0e-8",
        help="Comma-separated offsets after Tc in seconds.",
    )
    parser.add_argument(
        "--shell-analysis-dir",
        type=Path,
        default=None,
        help="Directory containing shell_depth_after_Tc_*.csv. Defaults to <sweep_root>/analysis_shell_depth.",
    )
    parser.add_argument(
        "--min-support-fraction",
        type=float,
        default=0.10,
        help="Minimum mean plaquette support fraction required for a candidate focus band.",
    )
    parser.add_argument(
        "--min-density-ratio",
        type=float,
        default=1.10,
        help="Minimum max/min band-density ratio required for a candidate focus band.",
    )
    parser.add_argument(
        "--min-focus-width-bins",
        type=int,
        default=2,
        help="Minimum number of adjacent shell-depth bins in the chosen focus band.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Directory where the band-decomposition CSVs are written.",
    )
    return parser.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated value")
    return items


def parse_inf_float(raw: str) -> float:
    text = str(raw).strip().lower()
    if text in ("inf", "+inf", "infinity", "+infinity"):
        return float("inf")
    if text in ("-inf", "-infinity"):
        return float("-inf")
    return float(text)


def fit_loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    slope, _ = qv._fit_powerlaw_loglog(x, y)
    return float(slope)


def positive_log_correlation(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if int(np.count_nonzero(mask)) < 2:
        return float("nan")
    log_x = np.log(x[mask])
    log_y = np.log(y[mask])
    if np.allclose(log_x, log_x[0]) or np.allclose(log_y, log_y[0]):
        return float("nan")
    return float(np.corrcoef(log_x, log_y)[0, 1])


def compute_ratio(values: np.ndarray) -> float:
    finite = values[np.isfinite(values) & (values >= 0.0)]
    if finite.size == 0:
        return float("nan")
    vmax = float(np.max(finite))
    vmin = float(np.min(finite))
    if vmin > 0.0:
        return vmax / vmin
    if vmax > 0.0:
        return float("inf")
    return float("nan")


def monotonic_with_rate(rate: np.ndarray, values: np.ndarray, *, tol: float = 1e-15) -> bool:
    mask = np.isfinite(rate) & np.isfinite(values)
    if int(np.count_nonzero(mask)) < 2:
        return False
    order = np.argsort(rate[mask])
    vals = values[mask][order]
    return bool(np.all(np.diff(vals) >= -float(tol)))


def safe_nanmean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def format_band_label(lo: float, hi: float) -> str:
    if math.isinf(hi):
        return f"[{lo:g},inf)"
    return f"[{lo:g},{hi:g})"


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_shell_detail_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_per_run_bins(detail_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_run: dict[str, list[dict[str, str]]] = {}
    for row in detail_rows:
        by_run.setdefault(row["run"], []).append(row)

    runs: list[dict[str, object]] = []
    for run_name, rows in by_run.items():
        rows_sorted = sorted(rows, key=lambda row: int(row["depth_bin_index"]))
        bins: list[dict[str, float | int | str | bool]] = []
        for row in rows_sorted:
            lo = float(row["depth_bin_lo"])
            hi = parse_inf_float(row["depth_bin_hi"])
            bins.append(
                {
                    "depth_bin_index": int(row["depth_bin_index"]),
                    "depth_bin_lo": lo,
                    "depth_bin_hi": hi,
                    "depth_bin_label": str(row["depth_bin_label"]),
                    "depth_bin_center_plot": float(row["depth_bin_center_plot"]),
                    "depth_bin_used_plaquettes": int(row["depth_bin_used_plaquettes"]),
                    "depth_bin_defect_count": int(row["depth_bin_defect_count"]),
                }
            )
        runs.append(
            {
                "run": run_name,
                "rate_K_per_s": float(rows_sorted[0]["rate_K_per_s"]),
                "t_ramp_s": float(rows_sorted[0]["t_ramp_s"]),
                "target_after_Tc_s": float(rows_sorted[0]["target_after_Tc_s"]),
                "actual_after_Tc_s": float(rows_sorted[0]["actual_after_Tc_s"]),
                "selected_time_s": float(rows_sorted[0]["selected_time_s"]),
                "selected_iter": int(rows_sorted[0]["selected_iter"]),
                "field_file": str(rows_sorted[0]["field_file"]),
                "selected_avg_S": float(rows_sorted[0]["selected_avg_S"]),
                "total_used_plaquettes": int(rows_sorted[0]["total_used_plaquettes"]),
                "total_defect_count": int(rows_sorted[0]["total_defect_count"]),
                "total_density": float(rows_sorted[0]["total_density"]),
                "bins": bins,
            }
        )
    runs.sort(key=lambda row: float(row["rate_K_per_s"]))
    return runs


def summarize_metric(rate: np.ndarray, values: np.ndarray) -> dict[str, float | bool | int]:
    return {
        "positive_cases": int(np.count_nonzero(np.isfinite(values) & (values > 0.0))),
        "mean_value": float(safe_nanmean(values)),
        "slope_vs_tauQ": float("nan"),
        "slope_vs_rate": fit_loglog_slope(rate, values),
        "corr_vs_tauQ": float("nan"),
        "corr_vs_rate": positive_log_correlation(rate, values),
        "value_min": float(np.nanmin(values)),
        "value_max": float(np.nanmax(values)),
        "value_ratio_max_over_min": float(compute_ratio(values)),
        "monotonic_vs_rate": bool(monotonic_with_rate(rate, values)),
    }


def build_band_scan_rows(runs: list[dict[str, object]]) -> list[dict[str, float | int | str | bool]]:
    if not runs:
        return []
    first_bins = runs[0]["bins"]
    assert isinstance(first_bins, list)
    number_of_bins = len(first_bins)
    rate = np.array([float(run["rate_K_per_s"]) for run in runs], dtype=float)
    tau_q = np.array([float(run["t_ramp_s"]) for run in runs], dtype=float)

    scan_rows: list[dict[str, float | int | str | bool]] = []
    for start_index in range(number_of_bins):
        for end_index in range(start_index, number_of_bins):
            density_vals: list[float] = []
            defect_share_vals: list[float] = []
            support_share_vals: list[float] = []
            enrichment_vals: list[float] = []
            mean_center_vals: list[float] = []

            for run in runs:
                bins = run["bins"]
                assert isinstance(bins, list)
                selected_bins = bins[start_index : end_index + 1]
                used = float(sum(int(bin_row["depth_bin_used_plaquettes"]) for bin_row in selected_bins))
                defect = float(sum(int(bin_row["depth_bin_defect_count"]) for bin_row in selected_bins))
                total_used = float(run["total_used_plaquettes"])
                total_defect = float(run["total_defect_count"])
                total_density = float(run["total_density"])

                density = defect / used if used > 0.0 else float("nan")
                defect_share = defect / total_defect if total_defect > 0.0 else float("nan")
                support_share = used / total_used if total_used > 0.0 else float("nan")
                enrichment = density / total_density if density > 0.0 and total_density > 0.0 else float("nan")
                mean_center = (
                    sum(float(bin_row["depth_bin_center_plot"]) * int(bin_row["depth_bin_defect_count"]) for bin_row in selected_bins) / defect
                    if defect > 0.0
                    else float("nan")
                )

                density_vals.append(density)
                defect_share_vals.append(defect_share)
                support_share_vals.append(support_share)
                enrichment_vals.append(enrichment)
                mean_center_vals.append(mean_center)

            start_bin = first_bins[start_index]
            end_bin = first_bins[end_index]
            assert isinstance(start_bin, dict)
            assert isinstance(end_bin, dict)
            lo = float(start_bin["depth_bin_lo"])
            hi = float(end_bin["depth_bin_hi"])
            label = format_band_label(lo, hi)

            density_arr = np.array(density_vals, dtype=float)
            defect_share_arr = np.array(defect_share_vals, dtype=float)
            support_share_arr = np.array(support_share_vals, dtype=float)
            enrichment_arr = np.array(enrichment_vals, dtype=float)
            mean_center_arr = np.array(mean_center_vals, dtype=float)

            density_summary = summarize_metric(rate, density_arr)
            defect_share_summary = summarize_metric(rate, defect_share_arr)
            enrichment_summary = summarize_metric(rate, enrichment_arr)

            scan_rows.append(
                {
                    "band_start_index": int(start_index),
                    "band_end_index": int(end_index),
                    "band_bin_count": int(end_index - start_index + 1),
                    "band_lo": float(lo),
                    "band_hi": float(hi),
                    "band_label": label,
                    "mean_support_fraction": float(safe_nanmean(support_share_arr)),
                    "support_fraction_min": float(np.nanmin(support_share_arr)),
                    "support_fraction_max": float(np.nanmax(support_share_arr)),
                    "density_positive_cases": int(density_summary["positive_cases"]),
                    "density_mean": float(density_summary["mean_value"]),
                    "density_slope_vs_tauQ": fit_loglog_slope(tau_q, density_arr),
                    "density_slope_vs_rate": float(density_summary["slope_vs_rate"]),
                    "density_corr_vs_tauQ": positive_log_correlation(tau_q, density_arr),
                    "density_corr_vs_rate": float(density_summary["corr_vs_rate"]),
                    "density_value_min": float(density_summary["value_min"]),
                    "density_value_max": float(density_summary["value_max"]),
                    "density_ratio_max_over_min": float(density_summary["value_ratio_max_over_min"]),
                    "density_monotonic_vs_rate": bool(density_summary["monotonic_vs_rate"]),
                    "defect_share_positive_cases": int(defect_share_summary["positive_cases"]),
                    "defect_share_mean": float(defect_share_summary["mean_value"]),
                    "defect_share_slope_vs_tauQ": fit_loglog_slope(tau_q, defect_share_arr),
                    "defect_share_slope_vs_rate": float(defect_share_summary["slope_vs_rate"]),
                    "defect_share_corr_vs_tauQ": positive_log_correlation(tau_q, defect_share_arr),
                    "defect_share_corr_vs_rate": float(defect_share_summary["corr_vs_rate"]),
                    "defect_share_value_min": float(defect_share_summary["value_min"]),
                    "defect_share_value_max": float(defect_share_summary["value_max"]),
                    "defect_share_ratio_max_over_min": float(defect_share_summary["value_ratio_max_over_min"]),
                    "defect_share_monotonic_vs_rate": bool(defect_share_summary["monotonic_vs_rate"]),
                    "enrichment_mean": float(enrichment_summary["mean_value"]),
                    "enrichment_slope_vs_tauQ": fit_loglog_slope(tau_q, enrichment_arr),
                    "enrichment_slope_vs_rate": float(enrichment_summary["slope_vs_rate"]),
                    "enrichment_corr_vs_tauQ": positive_log_correlation(tau_q, enrichment_arr),
                    "enrichment_corr_vs_rate": float(enrichment_summary["corr_vs_rate"]),
                    "enrichment_ratio_max_over_min": float(enrichment_summary["value_ratio_max_over_min"]),
                    "enrichment_monotonic_vs_rate": bool(enrichment_summary["monotonic_vs_rate"]),
                    "band_center_weighted_by_defects_mean": float(safe_nanmean(mean_center_arr)),
                    "band_center_weighted_by_defects_slope_vs_rate": fit_loglog_slope(rate, mean_center_arr),
                    "band_center_weighted_by_defects_corr_vs_rate": positive_log_correlation(rate, mean_center_arr),
                }
            )
    return scan_rows


def band_row_key(row: dict[str, float | int | str | bool]) -> tuple[int, int]:
    return int(row["band_start_index"]), int(row["band_end_index"])


def build_common_band_summary(
    band_scans_by_offset: dict[str, list[dict[str, float | int | str | bool]]],
    *,
    min_support_fraction: float,
    min_density_ratio: float,
    min_focus_width_bins: int,
) -> tuple[list[dict[str, float | int | str | bool]], dict[str, float | int | str | bool] | None]:
    if not band_scans_by_offset:
        return [], None

    by_key: dict[tuple[int, int], dict[str, dict[str, float | int | str | bool]]] = {}
    for offset_key, rows in band_scans_by_offset.items():
        for row in rows:
            by_key.setdefault(band_row_key(row), {})[offset_key] = row

    offsets_sorted = sorted(band_scans_by_offset)
    summary_rows: list[dict[str, float | int | str | bool]] = []
    for key, rows_by_offset in by_key.items():
        if any(offset_key not in rows_by_offset for offset_key in offsets_sorted):
            continue
        rows = [rows_by_offset[offset_key] for offset_key in offsets_sorted]
        first = rows[0]
        qualifies = all(
            int(row["band_bin_count"]) >= int(min_focus_width_bins)
            and float(row["mean_support_fraction"]) >= float(min_support_fraction)
            and bool(row["density_monotonic_vs_rate"])
            and np.isfinite(float(row["density_slope_vs_rate"]))
            and float(row["density_slope_vs_rate"]) > 0.0
            and np.isfinite(float(row["density_corr_vs_rate"]))
            and float(row["density_corr_vs_rate"]) > 0.0
            and np.isfinite(float(row["density_ratio_max_over_min"]))
            and float(row["density_ratio_max_over_min"]) >= float(min_density_ratio)
            for row in rows
        )
        summary_rows.append(
            {
                "band_start_index": int(first["band_start_index"]),
                "band_end_index": int(first["band_end_index"]),
                "band_bin_count": int(first["band_bin_count"]),
                "band_lo": float(first["band_lo"]),
                "band_hi": float(first["band_hi"]),
                "band_label": str(first["band_label"]),
                "qualifies_focus_band": bool(qualifies),
                "mean_support_fraction_across_offsets": float(np.mean([float(row["mean_support_fraction"]) for row in rows])),
                "min_support_fraction_across_offsets": float(min(float(row["mean_support_fraction"]) for row in rows)),
                "mean_density_slope_vs_rate": float(np.mean([float(row["density_slope_vs_rate"]) for row in rows])),
                "min_density_slope_vs_rate": float(min(float(row["density_slope_vs_rate"]) for row in rows)),
                "min_density_corr_vs_rate": float(min(float(row["density_corr_vs_rate"]) for row in rows)),
                "min_density_ratio_max_over_min": float(min(float(row["density_ratio_max_over_min"]) for row in rows)),
                "mean_defect_share_slope_vs_rate": float(np.mean([float(row["defect_share_slope_vs_rate"]) for row in rows])),
                "min_defect_share_slope_vs_rate": float(min(float(row["defect_share_slope_vs_rate"]) for row in rows)),
                "min_defect_share_corr_vs_rate": float(min(float(row["defect_share_corr_vs_rate"]) for row in rows)),
                "min_defect_share_ratio_max_over_min": float(min(float(row["defect_share_ratio_max_over_min"]) for row in rows)),
                "mean_enrichment_slope_vs_rate": float(np.mean([float(row["enrichment_slope_vs_rate"]) for row in rows])),
                "min_enrichment_corr_vs_rate": float(min(float(row["enrichment_corr_vs_rate"]) for row in rows)),
            }
        )

    summary_rows.sort(
        key=lambda row: (
            bool(row["qualifies_focus_band"]),
            float(row["mean_density_slope_vs_rate"]),
            float(row["min_density_slope_vs_rate"]),
            float(row["mean_support_fraction_across_offsets"]),
        ),
        reverse=True,
    )

    focus_row = next((row for row in summary_rows if bool(row["qualifies_focus_band"])), None)
    return summary_rows, focus_row


def aggregate_region_from_indices(
    run: dict[str, object],
    *,
    start_index: int,
    end_index: int,
) -> dict[str, float]:
    bins = run["bins"]
    assert isinstance(bins, list)
    selected_bins = bins[start_index : end_index + 1]
    used = float(sum(int(bin_row["depth_bin_used_plaquettes"]) for bin_row in selected_bins))
    defect = float(sum(int(bin_row["depth_bin_defect_count"]) for bin_row in selected_bins))
    total_used = float(run["total_used_plaquettes"])
    total_defect = float(run["total_defect_count"])
    total_density = float(run["total_density"])
    return {
        "used": used,
        "defect": defect,
        "density": defect / used if used > 0.0 else float("nan"),
        "defect_share": defect / total_defect if total_defect > 0.0 else float("nan"),
        "support_share": used / total_used if total_used > 0.0 else float("nan"),
        "enrichment": (defect / used) / total_density if used > 0.0 and total_density > 0.0 else float("nan"),
    }


def build_focus_region_rows(
    runs: list[dict[str, object]],
    *,
    focus_row: dict[str, float | int | str | bool],
) -> list[dict[str, float | int | str | bool]]:
    if not runs:
        return []

    bins = runs[0]["bins"]
    assert isinstance(bins, list)
    max_index = len(bins) - 1
    focus_start = int(focus_row["band_start_index"])
    focus_end = int(focus_row["band_end_index"])

    region_specs: list[tuple[str, int, int]] = []
    if focus_start > 0:
        region_specs.append(("skin", 0, focus_start - 1))
    region_specs.append(("focus", focus_start, focus_end))
    if focus_end < max_index:
        region_specs.append(("bulk", focus_end + 1, max_index))

    rate = np.array([float(run["rate_K_per_s"]) for run in runs], dtype=float)
    tau_q = np.array([float(run["t_ramp_s"]) for run in runs], dtype=float)
    summary_rows: list[dict[str, float | int | str | bool]] = []
    for region_name, start_index, end_index in region_specs:
        density_vals: list[float] = []
        defect_share_vals: list[float] = []
        support_share_vals: list[float] = []
        enrichment_vals: list[float] = []
        for run in runs:
            agg = aggregate_region_from_indices(run, start_index=start_index, end_index=end_index)
            density_vals.append(float(agg["density"]))
            defect_share_vals.append(float(agg["defect_share"]))
            support_share_vals.append(float(agg["support_share"]))
            enrichment_vals.append(float(agg["enrichment"]))

        density_arr = np.array(density_vals, dtype=float)
        defect_share_arr = np.array(defect_share_vals, dtype=float)
        support_share_arr = np.array(support_share_vals, dtype=float)
        enrichment_arr = np.array(enrichment_vals, dtype=float)
        lo = float(bins[start_index]["depth_bin_lo"])
        hi = float(bins[end_index]["depth_bin_hi"])
        summary_rows.append(
            {
                "region": region_name,
                "region_start_index": int(start_index),
                "region_end_index": int(end_index),
                "region_bin_count": int(end_index - start_index + 1),
                "region_lo": float(lo),
                "region_hi": float(hi),
                "region_label": format_band_label(lo, hi),
                "mean_support_fraction": float(safe_nanmean(support_share_arr)),
                "support_fraction_min": float(np.nanmin(support_share_arr)),
                "support_fraction_max": float(np.nanmax(support_share_arr)),
                "density_slope_vs_tauQ": fit_loglog_slope(tau_q, density_arr),
                "density_slope_vs_rate": fit_loglog_slope(rate, density_arr),
                "density_corr_vs_tauQ": positive_log_correlation(tau_q, density_arr),
                "density_corr_vs_rate": positive_log_correlation(rate, density_arr),
                "density_ratio_max_over_min": float(compute_ratio(density_arr)),
                "density_monotonic_vs_rate": bool(monotonic_with_rate(rate, density_arr)),
                "defect_share_slope_vs_tauQ": fit_loglog_slope(tau_q, defect_share_arr),
                "defect_share_slope_vs_rate": fit_loglog_slope(rate, defect_share_arr),
                "defect_share_corr_vs_tauQ": positive_log_correlation(tau_q, defect_share_arr),
                "defect_share_corr_vs_rate": positive_log_correlation(rate, defect_share_arr),
                "defect_share_ratio_max_over_min": float(compute_ratio(defect_share_arr)),
                "enrichment_slope_vs_tauQ": fit_loglog_slope(tau_q, enrichment_arr),
                "enrichment_slope_vs_rate": fit_loglog_slope(rate, enrichment_arr),
                "enrichment_corr_vs_tauQ": positive_log_correlation(tau_q, enrichment_arr),
                "enrichment_corr_vs_rate": positive_log_correlation(rate, enrichment_arr),
                "enrichment_ratio_max_over_min": float(compute_ratio(enrichment_arr)),
            }
        )
    return summary_rows


def build_moment_rows(runs: list[dict[str, object]]) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
    moment_rows: list[dict[str, float | int | str]] = []
    rate = np.array([float(run["rate_K_per_s"]) for run in runs], dtype=float)
    tau_q = np.array([float(run["t_ramp_s"]) for run in runs], dtype=float)
    defect_depth_vals: list[float] = []
    support_depth_vals: list[float] = []
    depth_shift_vals: list[float] = []

    for run in runs:
        bins = run["bins"]
        assert isinstance(bins, list)
        centers = np.array([float(bin_row["depth_bin_center_plot"]) for bin_row in bins], dtype=float)
        defect_weights = np.array([int(bin_row["depth_bin_defect_count"]) for bin_row in bins], dtype=float)
        support_weights = np.array([int(bin_row["depth_bin_used_plaquettes"]) for bin_row in bins], dtype=float)
        defect_mean_depth = (
            float(np.sum(centers * defect_weights) / np.sum(defect_weights))
            if np.sum(defect_weights) > 0.0
            else float("nan")
        )
        support_mean_depth = (
            float(np.sum(centers * support_weights) / np.sum(support_weights))
            if np.sum(support_weights) > 0.0
            else float("nan")
        )
        depth_shift = defect_mean_depth - support_mean_depth if np.isfinite(defect_mean_depth) and np.isfinite(support_mean_depth) else float("nan")
        defect_depth_vals.append(defect_mean_depth)
        support_depth_vals.append(support_mean_depth)
        depth_shift_vals.append(depth_shift)
        moment_rows.append(
            {
                "run": str(run["run"]),
                "rate_K_per_s": float(run["rate_K_per_s"]),
                "t_ramp_s": float(run["t_ramp_s"]),
                "target_after_Tc_s": float(run["target_after_Tc_s"]),
                "actual_after_Tc_s": float(run["actual_after_Tc_s"]),
                "selected_iter": int(run["selected_iter"]),
                "field_file": str(run["field_file"]),
                "selected_avg_S": float(run["selected_avg_S"]),
                "defect_mean_depth_band_center": float(defect_mean_depth),
                "support_mean_depth_band_center": float(support_mean_depth),
                "defect_minus_support_mean_depth": float(depth_shift),
            }
        )

    defect_depth_arr = np.array(defect_depth_vals, dtype=float)
    support_depth_arr = np.array(support_depth_vals, dtype=float)
    depth_shift_arr = np.array(depth_shift_vals, dtype=float)
    summary_row: dict[str, float | int | str] = {
        "defect_mean_depth_mean": float(safe_nanmean(defect_depth_arr)),
        "defect_mean_depth_slope_vs_tauQ": fit_loglog_slope(tau_q, defect_depth_arr),
        "defect_mean_depth_slope_vs_rate": fit_loglog_slope(rate, defect_depth_arr),
        "defect_mean_depth_corr_vs_tauQ": positive_log_correlation(tau_q, defect_depth_arr),
        "defect_mean_depth_corr_vs_rate": positive_log_correlation(rate, defect_depth_arr),
        "support_mean_depth_mean": float(safe_nanmean(support_depth_arr)),
        "support_mean_depth_slope_vs_tauQ": fit_loglog_slope(tau_q, support_depth_arr),
        "support_mean_depth_slope_vs_rate": fit_loglog_slope(rate, support_depth_arr),
        "support_mean_depth_corr_vs_tauQ": positive_log_correlation(tau_q, support_depth_arr),
        "support_mean_depth_corr_vs_rate": positive_log_correlation(rate, support_depth_arr),
        "defect_minus_support_depth_mean": float(safe_nanmean(depth_shift_arr)),
        "defect_minus_support_depth_slope_vs_tauQ": fit_loglog_slope(tau_q, depth_shift_arr),
        "defect_minus_support_depth_slope_vs_rate": fit_loglog_slope(rate, depth_shift_arr),
        "defect_minus_support_depth_corr_vs_tauQ": positive_log_correlation(tau_q, depth_shift_arr),
        "defect_minus_support_depth_corr_vs_rate": positive_log_correlation(rate, depth_shift_arr),
    }
    return moment_rows, summary_row


def main() -> int:
    args = parse_args()
    sweep_root = args.sweep_root.resolve()
    shell_analysis_dir = args.shell_analysis_dir.resolve() if args.shell_analysis_dir else (sweep_root / "analysis_shell_depth").resolve()
    analysis_dir = args.analysis_dir.resolve() if args.analysis_dir else (shell_analysis_dir / "band_decomposition").resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    offsets = [float(item) for item in parse_csv_list(args.offsets)]
    band_scans_by_offset: dict[str, list[dict[str, float | int | str | bool]]] = {}
    runs_by_offset: dict[str, list[dict[str, object]]] = {}
    region_summary_all: list[dict[str, float | int | str | bool]] = []
    moment_summary_rows: list[dict[str, float | int | str]] = []

    for offset in offsets:
        offset_key = f"{offset:.2e}"
        detail_path = shell_analysis_dir / f"shell_depth_after_Tc_{offset_key}.csv"
        if not detail_path.exists():
            raise FileNotFoundError(f"Missing shell-depth detail CSV: {detail_path}")

        detail_rows = load_shell_detail_rows(detail_path)
        runs = build_per_run_bins(detail_rows)
        runs_by_offset[offset_key] = runs

        band_scan_rows = build_band_scan_rows(runs)
        band_scans_by_offset[offset_key] = band_scan_rows
        write_csv(analysis_dir / f"shell_band_scan_after_Tc_{offset_key}.csv", band_scan_rows)

        moment_rows, moment_summary = build_moment_rows(runs)
        write_csv(analysis_dir / f"shell_depth_moments_after_Tc_{offset_key}.csv", moment_rows)
        moment_summary_rows.append(
            {
                "target_after_Tc_s": float(offset),
                **moment_summary,
            }
        )

    write_csv(analysis_dir / "shell_depth_moment_summary.csv", moment_summary_rows)

    common_band_rows, focus_row = build_common_band_summary(
        band_scans_by_offset,
        min_support_fraction=float(args.min_support_fraction),
        min_density_ratio=float(args.min_density_ratio),
        min_focus_width_bins=int(args.min_focus_width_bins),
    )
    write_csv(analysis_dir / "shell_band_common_summary.csv", common_band_rows)

    if focus_row is not None:
        for offset in offsets:
            offset_key = f"{offset:.2e}"
            region_rows = build_focus_region_rows(runs_by_offset[offset_key], focus_row=focus_row)
            for row in region_rows:
                region_summary_all.append(
                    {
                        "target_after_Tc_s": float(offset),
                        "focus_band_label": str(focus_row["band_label"]),
                        **row,
                    }
                )
            write_csv(analysis_dir / f"shell_focus_regions_after_Tc_{offset_key}.csv", region_rows)
        write_csv(analysis_dir / "shell_focus_region_summary.csv", region_summary_all)

        print(
            f"focus_band={focus_row['band_label']} | width_bins={int(focus_row['band_bin_count'])} | "
            f"mean_support={float(focus_row['mean_support_fraction_across_offsets']):.6g} | "
            f"mean_density_slope={float(focus_row['mean_density_slope_vs_rate']):.6g} | "
            f"mean_share_slope={float(focus_row['mean_defect_share_slope_vs_rate']):.6g}"
        )
    else:
        print("focus_band=NONE")

    for row in moment_summary_rows:
        print(
            f"moment after_Tc={float(row['target_after_Tc_s']):.3e} | "
            f"defect_mean_depth_slope_vs_rate={float(row['defect_mean_depth_slope_vs_rate']):.6g} | "
            f"defect_minus_support_depth_slope_vs_rate={float(row['defect_minus_support_depth_slope_vs_rate']):.6g}"
        )

    print(f"analysis_dir={analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())