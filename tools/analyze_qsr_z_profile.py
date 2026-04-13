#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import QSRvis as qv


class SliceProfileRow(TypedDict):
    slice_index: int
    density: float
    defect_count: int
    used_plaquettes: int


class ProfileRunEntry(TypedDict):
    label: str
    metrics: dict[str, float]
    field_path: Path
    selected_iter: int
    selected_time_s: float
    actual_after_tc_s: float
    selected_avg_s: float
    Nx: int
    Ny: int
    Nz: int
    z_center: int
    rows: list[SliceProfileRow]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure full z-slice defect profiles over a confined rate sweep using real post-Tc snapshots, "
            "and summarize how thick the positive equatorial band remains around the midplane."
        ),
    )
    parser.add_argument("sweep_root", type=Path, help="Sweep root containing cases/*/output and rate_sweep_metrics.csv.")
    parser.add_argument(
        "--offsets",
        type=str,
        default="5.0e-8,6.0e-8",
        help="Comma-separated offsets after Tc in seconds.",
    )
    parser.add_argument(
        "--S-threshold",
        type=float,
        default=0.1,
        help="Minimum scalar order S for a plaquette corner to count as valid.",
    )
    parser.add_argument(
        "--charge-cutoff",
        type=float,
        default=0.25,
        help="Minimum absolute nematic charge per plaquette used to count a defect.",
    )
    parser.add_argument(
        "--center-index",
        type=int,
        default=None,
        help="Optional z index to treat as the equatorial center. Defaults to the geometric midplane.",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=1.10,
        help="Minimum max/min density ratio required for a slice to count toward the central positive band.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Directory where summary/detail CSVs and plots are written.",
    )
    return parser.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated value")
    return items


def list_run_dirs(sweep_root: Path) -> list[Path]:
    run_dirs: list[Path] = []
    for case_dir in sorted((sweep_root / "cases").glob("*")):
        output_dir = case_dir / "output"
        if (output_dir / "quench_log.dat").exists():
            run_dirs.append(output_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No cases/*/output directories with quench_log.dat found in {sweep_root}")
    return run_dirs


def load_rate_metrics(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    metrics: dict[str, dict[str, float]] = {}
    for row in rows:
        metrics[row["label"]] = {
            "ramp_iters": float(row["ramp_iters"]),
            "t_ramp_s": float(row["ramp_iters"]) * float(row["dt_config_s"]),
            "rate_K_per_s": abs(float(row["cooling_rate_K_per_s"])),
            "tc_crossing_time_s": float(row["tc_crossing_time_s"]),
            "final_logged_time_s": float(row["final_logged_time_s"]),
            "final_avg_S": float(row["final_avg_S"]),
        }
    return metrics


def extract_iter(path: Path) -> int:
    match = re.search(r"(\d+)", path.name)
    return int(match.group(1)) if match else -1


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
    mask = np.isfinite(values) & (values >= 0.0)
    finite = values[mask]
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
    diffs = np.diff(vals)
    return bool(np.all(diffs >= -float(tol)))


def clamp_center_index(center_index: int | None, Nz: int) -> int:
    if center_index is None:
        return int(Nz) // 2
    return max(0, min(int(Nz) - 1, int(center_index)))


def build_center_band_summary(
    slab_scan_rows: list[dict[str, float | int | str]],
    *,
    z_center: int,
    min_ratio: float,
) -> dict[str, float | int]:
    def qualifies(row: dict[str, float | int | str]) -> bool:
        return (
            bool(row.get("monotonic_vs_rate", False))
            and int(row.get("positive_cases", 0)) >= 2
            and np.isfinite(float(row.get("slope_vs_rate", float("nan"))))
            and float(row.get("slope_vs_rate", float("nan"))) > 0.0
            and np.isfinite(float(row.get("corr_vs_rate", float("nan"))))
            and float(row.get("corr_vs_rate", float("nan"))) > 0.0
            and np.isfinite(float(row.get("value_ratio_max_over_min", float("nan"))))
            and float(row.get("value_ratio_max_over_min", float("nan"))) >= float(min_ratio)
        )

    qualifying = [row for row in slab_scan_rows if qualifies(row)]
    if not qualifying:
        return {
            "z_center": int(z_center),
            "band_half_width": -1,
            "band_slice_count": 0,
            "band_z_min": int(z_center),
            "band_z_max": int(z_center),
            "band_min_ratio": float("nan"),
            "band_min_corr": float("nan"),
            "band_min_slope": float("nan"),
        }

    best = max(qualifying, key=lambda row: int(row["slab_half_width"]))
    return {
        "z_center": int(z_center),
        "band_half_width": int(best["slab_half_width"]),
        "band_slice_count": int(best["slab_slice_count"]),
        "band_z_min": int(best["slab_z_min"]),
        "band_z_max": int(best["slab_z_max"]),
        "band_min_ratio": float(best["value_ratio_max_over_min"]),
        "band_min_corr": float(best["corr_vs_rate"]),
        "band_min_slope": float(best["slope_vs_rate"]),
    }


def compute_profile_rows(
    field_path: Path,
    *,
    S_threshold: float,
    charge_cutoff: float,
) -> tuple[int, int, int, list[SliceProfileRow]]:
    Nx, Ny, Nz = qv.infer_grid_dims_from_nematic_field_file(str(field_path))
    S, nx, ny, nz = qv.load_nematic_field_volume(str(field_path), Nx, Ny, Nz)
    raw_rows = qv._slice_profile_rows(
        S,
        nx,
        ny,
        nz,
        axis="z",
        S_threshold=float(S_threshold),
        charge_cutoff=float(charge_cutoff),
    )
    rows: list[SliceProfileRow] = []
    for row in raw_rows:
        rows.append(
            {
                "slice_index": int(row["slice_index"]),
                "density": float(row["density"]),
                "defect_count": int(row["defect_count"]),
                "used_plaquettes": int(row["used_plaquettes"]),
            }
        )
    return Nx, Ny, Nz, rows


def build_slab_scan_rows(
    detail_rows: list[dict[str, float | int | str]],
    *,
    z_center: int,
) -> list[dict[str, float | int | str]]:
    by_run: dict[str, list[dict[str, float | int | str]]] = {}
    for row in detail_rows:
        by_run.setdefault(str(row["run"]), []).append(row)

    if not by_run:
        return []

    max_half_width = min(
        max(abs(int(row["slice_offset_from_center"])) for row in rows)
        for rows in by_run.values()
    )

    scan_rows: list[dict[str, float | int | str]] = []
    for half_width in range(int(max_half_width) + 1):
        run_rows: list[dict[str, float | int | str]] = []
        for run_name, rows in by_run.items():
            selected = [
                row
                for row in rows
                if abs(int(row["slice_offset_from_center"])) <= int(half_width)
            ]
            selected_indices = [int(row["slice_index"]) for row in selected]
            total_defects = int(sum(int(row["defect_count"]) for row in selected))
            total_used = int(sum(int(row["used_plaquettes"]) for row in selected))
            slab_density = float(total_defects) / float(total_used) if total_used > 0 else float("nan")
            base = selected[0] if selected else rows[0]
            run_rows.append(
                {
                    "run": run_name,
                    "rate_K_per_s": float(base["rate_K_per_s"]),
                    "t_ramp_s": float(base["t_ramp_s"]),
                    "slab_density": slab_density,
                    "total_defects": total_defects,
                    "total_plaquettes": total_used,
                        "slab_z_min": min(selected_indices) if selected_indices else int(base["slice_index"]),
                        "slab_z_max": max(selected_indices) if selected_indices else int(base["slice_index"]),
                        "slab_slice_count_actual": len(selected_indices) if selected_indices else 1,
                }
            )

        run_rows_sorted = sorted(run_rows, key=lambda row: float(row["rate_K_per_s"]))
        rate = np.array([float(row["rate_K_per_s"]) for row in run_rows_sorted], dtype=float)
        tau_q = np.array([float(row["t_ramp_s"]) for row in run_rows_sorted], dtype=float)
        density = np.array([float(row["slab_density"]) for row in run_rows_sorted], dtype=float)
        slab_z_min = int(min(int(row["slab_z_min"]) for row in run_rows_sorted))
        slab_z_max = int(max(int(row["slab_z_max"]) for row in run_rows_sorted))
        slab_slice_count_actual = int(max(int(row["slab_slice_count_actual"]) for row in run_rows_sorted))
        scan_rows.append(
            {
                "slab_half_width": int(half_width),
                "slab_slice_count": int(slab_slice_count_actual),
                "slab_z_min": int(slab_z_min),
                "slab_z_max": int(slab_z_max),
                "positive_cases": int(np.count_nonzero(np.isfinite(density) & (density > 0.0))),
                "mean_density": float(np.nanmean(density)),
                "slope_vs_tauQ": fit_loglog_slope(tau_q, density),
                "slope_vs_rate": fit_loglog_slope(rate, density),
                "corr_vs_tauQ": positive_log_correlation(tau_q, density),
                "corr_vs_rate": positive_log_correlation(rate, density),
                "value_min": float(np.nanmin(density)),
                "value_max": float(np.nanmax(density)),
                "value_ratio_max_over_min": float(compute_ratio(density)),
                "monotonic_vs_rate": bool(monotonic_with_rate(rate, density)),
            }
        )
    return scan_rows


def plot_slab_scan(
    path: Path,
    *,
    offset_s: float,
    slab_scan_rows: list[dict[str, float | int | str]],
    band_summary: dict[str, float | int],
) -> None:
    if not slab_scan_rows:
        return
    rows_sorted = sorted(slab_scan_rows, key=lambda row: int(row["slab_half_width"]))
    h = np.array([int(row["slab_half_width"]) for row in rows_sorted], dtype=float)
    slope = np.array([float(row["slope_vs_rate"]) for row in rows_sorted], dtype=float)
    ratio = np.array([float(row["value_ratio_max_over_min"]) for row in rows_sorted], dtype=float)
    corr = np.array([float(row["corr_vs_rate"]) for row in rows_sorted], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(h, slope, color="#1f77b4", linewidth=1.6, label="slope vs rate")
    ax.plot(h, corr, color="#2ca02c", linewidth=1.4, linestyle="-.", label="corr vs rate")
    ax.plot(h, ratio, color="#d62728", linewidth=1.2, linestyle="--", label="ratio max/min")
    if int(band_summary["band_half_width"]) >= 0:
        ax.axvline(float(band_summary["band_half_width"]), color="#444444", linestyle=":", linewidth=1.2)
    ax.set_xlabel("slab half-width")
    ax.set_ylabel("summary value")
    ax.set_title(f"Symmetric slab scan after Tc={offset_s:.2e} s")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_offset_profiles(
    path: Path,
    *,
    offset_s: float,
    z_center: int,
    detail_rows: list[dict[str, float | int | str]],
    summary_rows: list[dict[str, float | int | str]],
    band_summary: dict[str, float | int],
) -> None:
    per_run: dict[str, list[dict[str, float | int | str]]] = {}
    for row in detail_rows:
        per_run.setdefault(str(row["run"]), []).append(row)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    def run_sort_key(item: tuple[str, list[dict[str, float | int | str]]]) -> float:
        rows = item[1]
        return float(rows[0]["rate_K_per_s"]) if rows else float("inf")

    for run_name, rows in sorted(per_run.items(), key=run_sort_key):
        rows_sorted = sorted(rows, key=lambda row: int(row["slice_index"]))
        z = np.array([int(row["slice_offset_from_center"]) for row in rows_sorted], dtype=float)
        dens = np.array([float(row["density"]) for row in rows_sorted], dtype=float)
        rate = float(rows_sorted[0]["rate_K_per_s"]) if rows_sorted else float("nan")
        axes[0].plot(z, dens, marker="o", markersize=2.5, linewidth=1.2, label=f"{run_name} ({rate:.3g} K/s)")

    sum_sorted = sorted(summary_rows, key=lambda row: int(row["slice_index"]))
    z_sum = np.array([int(row["slice_offset_from_center"]) for row in sum_sorted], dtype=float)
    slope = np.array([float(row["slope_vs_rate"]) for row in sum_sorted], dtype=float)
    ratio = np.array([float(row["value_ratio_max_over_min"]) for row in sum_sorted], dtype=float)
    axes[1].plot(z_sum, slope, color="#1f77b4", linewidth=1.6, label="slope vs rate")
    axes[1].plot(z_sum, ratio, color="#d62728", linewidth=1.2, linestyle="--", label="ratio max/min")

    if int(band_summary["band_slice_count"]) > 0:
        x0 = int(band_summary["band_z_min"]) - int(z_center) - 0.5
        x1 = int(band_summary["band_z_max"]) - int(z_center) + 0.5
        axes[0].axvspan(x0, x1, color="#999999", alpha=0.12)
        axes[1].axvspan(x0, x1, color="#999999", alpha=0.12)

    axes[0].axvline(0.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[1].axvline(0.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[0].set_ylabel("slice defect density")
    axes[1].set_ylabel("slice summary")
    axes[1].set_xlabel(r"$z - z_{mid}$")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].legend(fontsize=9)
    axes[0].grid(True, alpha=0.25)
    axes[1].grid(True, alpha=0.25)
    fig.suptitle(
        f"Equatorial z-profile after Tc={offset_s:.2e} s | center={z_center} | band={int(band_summary['band_slice_count'])} slices"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    sweep_root = args.sweep_root.resolve()
    offsets = [float(item) for item in parse_csv_list(args.offsets)]
    analysis_dir = args.analysis_dir.resolve() if args.analysis_dir else (sweep_root / "analysis_z_profile").resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = list_run_dirs(sweep_root)
    rate_metrics = load_rate_metrics(sweep_root / "rate_sweep_metrics.csv")

    band_summary_rows: list[dict[str, float | int | str]] = []

    for offset_s in offsets:
        per_run_profiles: list[ProfileRunEntry] = []
        z_center_use: int | None = None

        for run_dir in run_dirs:
            label = run_dir.parent.name
            metrics = rate_metrics.get(label)
            if metrics is None:
                raise KeyError(f"Missing rate-sweep metrics for {label}")

            log_data, _ = qv.load_quench_log(str(run_dir))
            it = np.atleast_1d(log_data["iteration"]).astype(float)
            t = np.atleast_1d(log_data["time_s"]).astype(float)
            target_time_s = float(metrics["tc_crossing_time_s"]) + float(offset_s)
            field_path = Path(qv._select_snapshot_by_time(str(run_dir), target_time_s, it, t))
            selected_iter = extract_iter(field_path)
            selected_time_s = float(qv._nearest_from_log(it, t, selected_iter)) if selected_iter >= 0 else float("nan")
            actual_after_tc_s = selected_time_s - float(metrics["tc_crossing_time_s"])

            names = log_data.dtype.names or ()
            if "avg_S" in names and selected_iter >= 0:
                avg_s_arr = np.atleast_1d(log_data["avg_S"]).astype(float)
                selected_avg_s = float(qv._nearest_from_log(it, avg_s_arr, selected_iter))
            else:
                selected_avg_s = float("nan")

            Nx, Ny, Nz, rows = compute_profile_rows(
                field_path,
                S_threshold=float(args.S_threshold),
                charge_cutoff=float(args.charge_cutoff),
            )
            z_center = clamp_center_index(args.center_index, Nz)
            if z_center_use is None:
                z_center_use = z_center

            per_run_profiles.append(
                {
                    "label": label,
                    "metrics": metrics,
                    "field_path": field_path,
                    "selected_iter": selected_iter,
                    "selected_time_s": selected_time_s,
                    "actual_after_tc_s": actual_after_tc_s,
                    "selected_avg_s": selected_avg_s,
                    "Nx": Nx,
                    "Ny": Ny,
                    "Nz": Nz,
                    "z_center": z_center,
                    "rows": rows,
                }
            )

        if z_center_use is None:
            raise RuntimeError(f"No profiles were computed for offset {offset_s}")

        detail_rows: list[dict[str, float | int | str]] = []
        for entry in per_run_profiles:
            label = str(entry["label"])
            metrics = entry["metrics"]
            assert isinstance(metrics, dict)
            for row in entry["rows"]:
                detail_rows.append(
                    {
                        "run": label,
                        "ramp_iters": float(metrics["ramp_iters"]),
                        "t_ramp_s": float(metrics["t_ramp_s"]),
                        "rate_K_per_s": float(metrics["rate_K_per_s"]),
                        "target_after_Tc_s": float(offset_s),
                        "actual_after_Tc_s": float(entry["actual_after_tc_s"]),
                        "selected_time_s": float(entry["selected_time_s"]),
                        "selected_iter": int(entry["selected_iter"]),
                        "field_file": str(Path(entry["field_path"]).name),
                        "slice_index": int(row["slice_index"]),
                        "slice_offset_from_center": int(row["slice_index"]) - int(z_center_use),
                        "density": float(row["density"]),
                        "defect_count": int(row["defect_count"]),
                        "used_plaquettes": int(row["used_plaquettes"]),
                        "selected_avg_S": float(entry["selected_avg_s"]),
                    }
                )

        detail_path = analysis_dir / f"z_profile_after_Tc_{offset_s:.2e}.csv"
        write_csv(detail_path, detail_rows)

        by_slice: dict[int, list[dict[str, float | int | str]]] = {}
        for row in detail_rows:
            by_slice.setdefault(int(row["slice_index"]), []).append(row)

        summary_rows: list[dict[str, float | int | str]] = []
        for slice_index in sorted(by_slice):
            rows = by_slice[slice_index]
            rows_sorted = sorted(rows, key=lambda row: float(row["rate_K_per_s"]))
            rate = np.array([float(row["rate_K_per_s"]) for row in rows_sorted], dtype=float)
            density = np.array([float(row["density"]) for row in rows_sorted], dtype=float)
            used = np.array([float(row["used_plaquettes"]) for row in rows_sorted], dtype=float)
            summary_rows.append(
                {
                    "target_after_Tc_s": float(offset_s),
                    "slice_index": int(slice_index),
                    "slice_offset_from_center": int(slice_index) - int(z_center_use),
                    "positive_cases": int(np.count_nonzero(np.isfinite(density) & (density > 0.0))),
                    "mean_density": float(np.nanmean(density)),
                    "mean_used_plaquettes": float(np.nanmean(used)),
                    "slope_vs_tauQ": fit_loglog_slope(np.array([float(row["t_ramp_s"]) for row in rows_sorted], dtype=float), density),
                    "slope_vs_rate": fit_loglog_slope(rate, density),
                    "corr_vs_tauQ": positive_log_correlation(np.array([float(row["t_ramp_s"]) for row in rows_sorted], dtype=float), density),
                    "corr_vs_rate": positive_log_correlation(rate, density),
                    "value_min": float(np.nanmin(density)),
                    "value_max": float(np.nanmax(density)),
                    "value_ratio_max_over_min": float(compute_ratio(density)),
                    "monotonic_vs_rate": bool(monotonic_with_rate(rate, density)),
                }
            )

        summary_path = analysis_dir / f"z_profile_summary_after_Tc_{offset_s:.2e}.csv"
        write_csv(summary_path, summary_rows)

        slab_scan_rows = build_slab_scan_rows(detail_rows, z_center=int(z_center_use))
        slab_scan_path = analysis_dir / f"z_profile_slab_scan_after_Tc_{offset_s:.2e}.csv"
        write_csv(slab_scan_path, slab_scan_rows)

        band_summary = build_center_band_summary(
            slab_scan_rows,
            z_center=int(z_center_use),
            min_ratio=float(args.min_ratio),
        )
        band_summary_rows.append(
            {
                "target_after_Tc_s": float(offset_s),
                "z_center": int(band_summary["z_center"]),
                "band_half_width": int(band_summary["band_half_width"]),
                "band_slice_count": int(band_summary["band_slice_count"]),
                "band_z_min": int(band_summary["band_z_min"]),
                "band_z_max": int(band_summary["band_z_max"]),
                "band_min_ratio": float(band_summary["band_min_ratio"]),
                "band_min_corr": float(band_summary["band_min_corr"]),
                "band_min_slope": float(band_summary["band_min_slope"]),
                "min_ratio_threshold": float(args.min_ratio),
                "detail_csv": str(detail_path.as_posix()),
                "summary_csv": str(summary_path.as_posix()),
                "slab_scan_csv": str(slab_scan_path.as_posix()),
                "plot_png": str((analysis_dir / f"z_profile_after_Tc_{offset_s:.2e}.png").as_posix()),
                "slab_scan_plot_png": str((analysis_dir / f"z_profile_slab_scan_after_Tc_{offset_s:.2e}.png").as_posix()),
            }
        )

        plot_offset_profiles(
            analysis_dir / f"z_profile_after_Tc_{offset_s:.2e}.png",
            offset_s=float(offset_s),
            z_center=int(z_center_use),
            detail_rows=detail_rows,
            summary_rows=summary_rows,
            band_summary=band_summary,
        )
        plot_slab_scan(
            analysis_dir / f"z_profile_slab_scan_after_Tc_{offset_s:.2e}.png",
            offset_s=float(offset_s),
            slab_scan_rows=slab_scan_rows,
            band_summary=band_summary,
        )

        print(
            f"z_profile after_Tc={float(offset_s):.3e} | center={int(z_center_use)} | "
            f"band_slices={int(band_summary['band_slice_count'])} "
            f"(z={int(band_summary['band_z_min'])}..{int(band_summary['band_z_max'])}) | "
            f"band_min_ratio={float(band_summary['band_min_ratio']):.6g}"
        )

    band_summary_path = analysis_dir / "z_profile_band_summary.csv"
    write_csv(band_summary_path, band_summary_rows)
    print(f"band_summary_csv={band_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())