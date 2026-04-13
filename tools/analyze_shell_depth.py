#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import QSRvis as qv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure shell-versus-interior localization of the confined post-Tc defect signal by "
            "binning XY defect plaquettes with the minimum inward distance of their corners from the "
            "filled droplet interface, and by scanning how the rate trend changes when the outer shell "
            "layers are excluded."
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
        "--bin-edges",
        type=str,
        default="0,2,4,6,8,10,12,16,24",
        help="Comma-separated inward-distance bin edges in voxel layers. The last bin is open-ended.",
    )
    parser.add_argument(
        "--S-threshold",
        type=float,
        default=0.1,
        help="Minimum scalar order S for a plaquette corner to count as valid.",
    )
    parser.add_argument(
        "--S-droplet",
        type=float,
        default=0.1,
        help="Minimum scalar order S used to seed the droplet support before filling holes.",
    )
    parser.add_argument(
        "--charge-cutoff",
        type=float,
        default=0.25,
        help="Minimum absolute nematic charge per plaquette used to count a defect.",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=1.10,
        help="Minimum max/min density ratio required for a shell-excluded interior trend to count as positive.",
    )
    parser.add_argument(
        "--min-remaining-fraction",
        type=float,
        default=0.05,
        help="Minimum mean retained plaquette fraction required for a shell-excluded interior trend to count as robust.",
    )
    parser.add_argument(
        "--max-exclude-layers",
        type=int,
        default=None,
        help="Optional cap on the shell-exclusion scan depth. Defaults to the common maximum supported by all runs.",
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


def parse_float_edges(raw: str) -> list[float]:
    vals = sorted({max(0.0, float(item)) for item in parse_csv_list(raw)})
    if not vals:
        raise ValueError("Expected at least one shell-depth bin edge")
    if vals[0] > 0.0:
        vals.insert(0, 0.0)
    return vals


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


def safe_nanmean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def safe_nanmin(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else float("nan")


def safe_nanmax(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else float("nan")


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def iter_shell_bins(edges: list[float]) -> list[tuple[int, float, float | None, str]]:
    rows: list[tuple[int, float, float | None, str]] = []
    for idx, lo in enumerate(edges):
        hi = edges[idx + 1] if idx + 1 < len(edges) else None
        if hi is None:
            label = f"[{lo:g},inf)"
        else:
            label = f"[{lo:g},{hi:g})"
        rows.append((idx, float(lo), None if hi is None else float(hi), label))
    return rows


def build_droplet_distance(
    S: np.ndarray,
    *,
    S_droplet: float,
) -> tuple[np.ndarray, np.ndarray]:
    if qv.ndi is None:
        raise ImportError("scipy is required for shell-depth analysis (install scipy)")
    finite = np.isfinite(S)
    droplet_seed = finite & (S > float(S_droplet))
    droplet = qv._largest_connected_component_3d(droplet_seed)
    droplet = qv.ndi.binary_fill_holes(droplet)
    distance_inside = qv.ndi.distance_transform_edt(droplet)
    return np.asarray(droplet, dtype=bool), np.asarray(distance_inside, dtype=float)


def compute_shell_depth_data(
    field_path: Path,
    *,
    S_threshold: float,
    S_droplet: float,
    charge_cutoff: float,
    bin_edges: list[float],
) -> dict[str, object]:
    Nx, Ny, Nz = qv.infer_grid_dims_from_nematic_field_file(str(field_path))
    S, nx, ny, _ = qv.load_nematic_field_volume(str(field_path), Nx, Ny, Nz)
    droplet, distance_inside = build_droplet_distance(S, S_droplet=float(S_droplet))

    valid_corner = (
        (S > float(S_threshold))
        & np.isfinite(S)
        & np.isfinite(nx)
        & np.isfinite(ny)
        & droplet
    )
    psi = 2.0 * np.arctan2(ny, nx)

    depth_parts: list[np.ndarray] = []
    defect_parts: list[np.ndarray] = []
    used_total = 0
    defect_total = 0

    for z_idx in range(Nz):
        mask = valid_corner[:, :, z_idx]
        plaq_mask = mask[:-1, :-1] & mask[1:, :-1] & mask[1:, 1:] & mask[:-1, 1:]
        if not np.any(plaq_mask):
            continue

        p00 = psi[:-1, :-1, z_idx]
        p10 = psi[1:, :-1, z_idx]
        p11 = psi[1:, 1:, z_idx]
        p01 = psi[:-1, 1:, z_idx]
        dsum = (
            qv._wrap_to_pi(p10 - p00)
            + qv._wrap_to_pi(p11 - p10)
            + qv._wrap_to_pi(p01 - p11)
            + qv._wrap_to_pi(p00 - p01)
        )
        s_map = 0.5 * (dsum / (2.0 * np.pi))
        defects = (np.abs(s_map) > float(charge_cutoff)) & plaq_mask

        depth_min = np.minimum.reduce(
            [
                distance_inside[:-1, :-1, z_idx],
                distance_inside[1:, :-1, z_idx],
                distance_inside[1:, 1:, z_idx],
                distance_inside[:-1, 1:, z_idx],
            ]
        )

        depth_parts.append(np.asarray(depth_min[plaq_mask], dtype=float))
        defect_parts.append(np.asarray(defects[plaq_mask], dtype=bool))
        used_total += int(np.count_nonzero(plaq_mask))
        defect_total += int(np.count_nonzero(defects))

    if depth_parts:
        plaquette_depth = np.concatenate(depth_parts)
        plaquette_defect = np.concatenate(defect_parts)
    else:
        plaquette_depth = np.array([], dtype=float)
        plaquette_defect = np.array([], dtype=bool)

    bin_rows: list[dict[str, float | int | str]] = []
    for bin_index, lo, hi, label in iter_shell_bins(bin_edges):
        if hi is None:
            sel = plaquette_depth >= float(lo)
        else:
            sel = (plaquette_depth >= float(lo)) & (plaquette_depth < float(hi))
        used = int(np.count_nonzero(sel))
        defects = int(np.count_nonzero(plaquette_defect[sel])) if used > 0 else 0
        density = float(defects) / float(used) if used > 0 else float("nan")
        bin_rows.append(
            {
                "depth_bin_index": int(bin_index),
                "depth_bin_lo": float(lo),
                "depth_bin_hi": float(hi) if hi is not None else float("inf"),
                "depth_bin_label": label,
                "depth_bin_is_open": bool(hi is None),
                "depth_bin_center_plot": float(lo) if hi is None else 0.5 * float(lo + hi),
                "depth_bin_used_plaquettes": int(used),
                "depth_bin_defect_count": int(defects),
                "depth_bin_density": float(density),
            }
        )

    max_depth = float(np.max(plaquette_depth)) if plaquette_depth.size else float("nan")
    return {
        "Nx": int(Nx),
        "Ny": int(Ny),
        "Nz": int(Nz),
        "droplet_voxels": int(np.count_nonzero(droplet)),
        "total_used_plaquettes": int(used_total),
        "total_defect_count": int(defect_total),
        "total_density": float(defect_total) / float(used_total) if used_total > 0 else float("nan"),
        "plaquette_depth": plaquette_depth,
        "plaquette_defect": plaquette_defect,
        "max_depth": float(max_depth),
        "bin_rows": bin_rows,
    }


def summarize_bin_rows(detail_rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    by_bin: dict[int, list[dict[str, float | int | str]]] = {}
    for row in detail_rows:
        by_bin.setdefault(int(row["depth_bin_index"]), []).append(row)

    summary_rows: list[dict[str, float | int | str]] = []
    for bin_index in sorted(by_bin):
        rows = sorted(by_bin[bin_index], key=lambda row: float(row["rate_K_per_s"]))
        rate = np.array([float(row["rate_K_per_s"]) for row in rows], dtype=float)
        tau_q = np.array([float(row["t_ramp_s"]) for row in rows], dtype=float)
        density = np.array([float(row["depth_bin_density"]) for row in rows], dtype=float)
        used = np.array([float(row["depth_bin_used_plaquettes"]) for row in rows], dtype=float)
        base = rows[0]
        summary_rows.append(
            {
                "depth_bin_index": int(bin_index),
                "depth_bin_lo": float(base["depth_bin_lo"]),
                "depth_bin_hi": float(base["depth_bin_hi"]),
                "depth_bin_label": str(base["depth_bin_label"]),
                "depth_bin_is_open": bool(base["depth_bin_is_open"]),
                "depth_bin_center_plot": float(base["depth_bin_center_plot"]),
                "positive_cases": int(np.count_nonzero(np.isfinite(density) & (density > 0.0))),
                "mean_density": float(safe_nanmean(density)),
                "mean_used_plaquettes": float(safe_nanmean(used)),
                "slope_vs_tauQ": fit_loglog_slope(tau_q, density),
                "slope_vs_rate": fit_loglog_slope(rate, density),
                "corr_vs_tauQ": positive_log_correlation(tau_q, density),
                "corr_vs_rate": positive_log_correlation(rate, density),
                "value_min": float(safe_nanmin(density)),
                "value_max": float(safe_nanmax(density)),
                "value_ratio_max_over_min": float(compute_ratio(density)),
                "monotonic_vs_rate": bool(monotonic_with_rate(rate, density)),
            }
        )
    return summary_rows


def build_shell_exclusion_scan(
    per_run_entries: list[dict[str, object]],
    *,
    max_exclude_layers: int | None,
) -> list[dict[str, float | int | str]]:
    finite_depths = [
        int(np.floor(float(entry["shell"]["max_depth"])))
        for entry in per_run_entries
        if np.isfinite(float(entry["shell"]["max_depth"]))
    ]
    if not finite_depths:
        return []

    common_max = int(min(finite_depths))
    if max_exclude_layers is not None:
        common_max = min(common_max, int(max_exclude_layers))
    common_max = max(0, int(common_max))

    scan_rows: list[dict[str, float | int | str]] = []
    for cutoff in range(common_max + 1):
        run_rows: list[dict[str, float | int | str]] = []
        for entry in per_run_entries:
            metrics = entry["metrics"]
            assert isinstance(metrics, dict)
            shell = entry["shell"]
            assert isinstance(shell, dict)
            depth = np.asarray(shell["plaquette_depth"], dtype=float)
            defects = np.asarray(shell["plaquette_defect"], dtype=bool)
            keep = depth > float(cutoff)
            used = int(np.count_nonzero(keep))
            defect_count = int(np.count_nonzero(defects[keep])) if used > 0 else 0
            density = float(defect_count) / float(used) if used > 0 else float("nan")
            total_used = int(shell["total_used_plaquettes"])
            retained_fraction = float(used) / float(total_used) if total_used > 0 else float("nan")
            run_rows.append(
                {
                    "run": str(entry["label"]),
                    "rate_K_per_s": float(metrics["rate_K_per_s"]),
                    "t_ramp_s": float(metrics["t_ramp_s"]),
                    "remaining_used_plaquettes": int(used),
                    "remaining_defect_count": int(defect_count),
                    "remaining_density": float(density),
                    "remaining_fraction_of_total": float(retained_fraction),
                }
            )

        run_rows_sorted = sorted(run_rows, key=lambda row: float(row["rate_K_per_s"]))
        rate = np.array([float(row["rate_K_per_s"]) for row in run_rows_sorted], dtype=float)
        tau_q = np.array([float(row["t_ramp_s"]) for row in run_rows_sorted], dtype=float)
        density = np.array([float(row["remaining_density"]) for row in run_rows_sorted], dtype=float)
        used = np.array([float(row["remaining_used_plaquettes"]) for row in run_rows_sorted], dtype=float)
        frac = np.array([float(row["remaining_fraction_of_total"]) for row in run_rows_sorted], dtype=float)
        scan_rows.append(
            {
                "shell_exclude_layers": int(cutoff),
                "positive_cases": int(np.count_nonzero(np.isfinite(density) & (density > 0.0))),
                "mean_remaining_density": float(safe_nanmean(density)),
                "mean_remaining_used_plaquettes": float(safe_nanmean(used)),
                "mean_remaining_fraction_of_total": float(safe_nanmean(frac)),
                "slope_vs_tauQ": fit_loglog_slope(tau_q, density),
                "slope_vs_rate": fit_loglog_slope(rate, density),
                "corr_vs_tauQ": positive_log_correlation(tau_q, density),
                "corr_vs_rate": positive_log_correlation(rate, density),
                "value_min": float(safe_nanmin(density)),
                "value_max": float(safe_nanmax(density)),
                "value_ratio_max_over_min": float(compute_ratio(density)),
                "monotonic_vs_rate": bool(monotonic_with_rate(rate, density)),
            }
        )
    return scan_rows


def build_shell_band_summary(
    scan_rows: list[dict[str, float | int | str]],
    *,
    min_ratio: float,
    min_remaining_fraction: float,
) -> dict[str, float | int]:
    def qualifies(row: dict[str, float | int | str]) -> bool:
        return (
            bool(row.get("monotonic_vs_rate", False))
            and int(row.get("positive_cases", 0)) >= 2
            and np.isfinite(float(row.get("mean_remaining_fraction_of_total", float("nan"))))
            and float(row.get("mean_remaining_fraction_of_total", float("nan"))) >= float(min_remaining_fraction)
            and np.isfinite(float(row.get("slope_vs_rate", float("nan"))))
            and float(row.get("slope_vs_rate", float("nan"))) > 0.0
            and np.isfinite(float(row.get("corr_vs_rate", float("nan"))))
            and float(row.get("corr_vs_rate", float("nan"))) > 0.0
            and np.isfinite(float(row.get("value_ratio_max_over_min", float("nan"))))
            and float(row.get("value_ratio_max_over_min", float("nan"))) >= float(min_ratio)
        )

    qualifying = [row for row in scan_rows if qualifies(row)]
    if not qualifying:
        return {
            "max_excluded_layers": -1,
            "remaining_mean_fraction": float("nan"),
            "remaining_min_ratio": float("nan"),
            "remaining_min_corr": float("nan"),
            "remaining_min_slope": float("nan"),
        }

    best = max(qualifying, key=lambda row: int(row["shell_exclude_layers"]))
    return {
        "max_excluded_layers": int(best["shell_exclude_layers"]),
        "remaining_mean_fraction": float(best["mean_remaining_fraction_of_total"]),
        "remaining_min_ratio": float(best["value_ratio_max_over_min"]),
        "remaining_min_corr": float(best["corr_vs_rate"]),
        "remaining_min_slope": float(best["slope_vs_rate"]),
    }


def plot_shell_profile(
    path: Path,
    *,
    offset_s: float,
    detail_rows: list[dict[str, float | int | str]],
    summary_rows: list[dict[str, float | int | str]],
) -> None:
    if not detail_rows or not summary_rows:
        return

    per_run: dict[str, list[dict[str, float | int | str]]] = {}
    for row in detail_rows:
        per_run.setdefault(str(row["run"]), []).append(row)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    def run_sort_key(item: tuple[str, list[dict[str, float | int | str]]]) -> float:
        rows = item[1]
        return float(rows[0]["rate_K_per_s"]) if rows else float("inf")

    for run_name, rows in sorted(per_run.items(), key=run_sort_key):
        rows_sorted = sorted(rows, key=lambda row: int(row["depth_bin_index"]))
        x = np.array([float(row["depth_bin_center_plot"]) for row in rows_sorted], dtype=float)
        y = np.array([float(row["depth_bin_density"]) for row in rows_sorted], dtype=float)
        rate = float(rows_sorted[0]["rate_K_per_s"]) if rows_sorted else float("nan")
        axes[0].plot(x, y, marker="o", markersize=3.0, linewidth=1.2, label=f"{run_name} ({rate:.3g} K/s)")

    sum_sorted = sorted(summary_rows, key=lambda row: int(row["depth_bin_index"]))
    x_sum = np.array([float(row["depth_bin_center_plot"]) for row in sum_sorted], dtype=float)
    slope = np.array([float(row["slope_vs_rate"]) for row in sum_sorted], dtype=float)
    ratio = np.array([float(row["value_ratio_max_over_min"]) for row in sum_sorted], dtype=float)
    corr = np.array([float(row["corr_vs_rate"]) for row in sum_sorted], dtype=float)
    axes[1].plot(x_sum, slope, color="#1f77b4", linewidth=1.6, label="slope vs rate")
    axes[1].plot(x_sum, corr, color="#2ca02c", linewidth=1.2, linestyle="-.", label="corr vs rate")
    axes[1].plot(x_sum, ratio, color="#d62728", linewidth=1.2, linestyle="--", label="ratio max/min")

    axes[0].set_ylabel("bin defect density")
    axes[1].set_ylabel("bin summary")
    axes[1].set_xlabel("minimum interface depth [layers]")
    axes[0].grid(True, alpha=0.25)
    axes[1].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].legend(fontsize=9)
    fig.suptitle(f"Shell-depth profile after Tc={offset_s:.2e} s")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_shell_scan(
    path: Path,
    *,
    offset_s: float,
    scan_rows: list[dict[str, float | int | str]],
    band_summary: dict[str, float | int],
) -> None:
    if not scan_rows:
        return
    rows_sorted = sorted(scan_rows, key=lambda row: int(row["shell_exclude_layers"]))
    x = np.array([int(row["shell_exclude_layers"]) for row in rows_sorted], dtype=float)
    slope = np.array([float(row["slope_vs_rate"]) for row in rows_sorted], dtype=float)
    corr = np.array([float(row["corr_vs_rate"]) for row in rows_sorted], dtype=float)
    ratio = np.array([float(row["value_ratio_max_over_min"]) for row in rows_sorted], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(x, slope, color="#1f77b4", linewidth=1.6, label="slope vs rate")
    ax.plot(x, corr, color="#2ca02c", linewidth=1.4, linestyle="-.", label="corr vs rate")
    ax.plot(x, ratio, color="#d62728", linewidth=1.2, linestyle="--", label="ratio max/min")
    if int(band_summary["max_excluded_layers"]) >= 0:
        ax.axvline(float(band_summary["max_excluded_layers"]), color="#444444", linestyle=":", linewidth=1.2)
    ax.set_xlabel("excluded shell depth [layers]")
    ax.set_ylabel("summary value")
    ax.set_title(f"Shell-excluded interior scan after Tc={offset_s:.2e} s")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    sweep_root = args.sweep_root.resolve()
    offsets = [float(item) for item in parse_csv_list(args.offsets)]
    bin_edges = parse_float_edges(args.bin_edges)
    analysis_dir = args.analysis_dir.resolve() if args.analysis_dir else (sweep_root / "analysis_shell_depth").resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = list_run_dirs(sweep_root)
    rate_metrics = load_rate_metrics(sweep_root / "rate_sweep_metrics.csv")

    band_summary_rows: list[dict[str, float | int | str]] = []

    for offset_s in offsets:
        per_run_entries: list[dict[str, object]] = []
        detail_rows: list[dict[str, float | int | str]] = []

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

            shell = compute_shell_depth_data(
                field_path,
                S_threshold=float(args.S_threshold),
                S_droplet=float(args.S_droplet),
                charge_cutoff=float(args.charge_cutoff),
                bin_edges=bin_edges,
            )
            per_run_entries.append(
                {
                    "label": label,
                    "metrics": metrics,
                    "field_path": field_path,
                    "selected_iter": selected_iter,
                    "selected_time_s": selected_time_s,
                    "actual_after_tc_s": actual_after_tc_s,
                    "selected_avg_s": selected_avg_s,
                    "shell": shell,
                }
            )

            shell_bin_rows = shell["bin_rows"]
            assert isinstance(shell_bin_rows, list)
            for row in shell_bin_rows:
                detail_rows.append(
                    {
                        "run": label,
                        "ramp_iters": float(metrics["ramp_iters"]),
                        "t_ramp_s": float(metrics["t_ramp_s"]),
                        "rate_K_per_s": float(metrics["rate_K_per_s"]),
                        "target_after_Tc_s": float(offset_s),
                        "actual_after_Tc_s": float(actual_after_tc_s),
                        "selected_time_s": float(selected_time_s),
                        "selected_iter": int(selected_iter),
                        "field_file": str(field_path.name),
                        "selected_avg_S": float(selected_avg_s),
                        "Nx": int(shell["Nx"]),
                        "Ny": int(shell["Ny"]),
                        "Nz": int(shell["Nz"]),
                        "droplet_voxels": int(shell["droplet_voxels"]),
                        "max_depth": float(shell["max_depth"]),
                        "total_used_plaquettes": int(shell["total_used_plaquettes"]),
                        "total_defect_count": int(shell["total_defect_count"]),
                        "total_density": float(shell["total_density"]),
                        "depth_bin_index": int(row["depth_bin_index"]),
                        "depth_bin_lo": float(row["depth_bin_lo"]),
                        "depth_bin_hi": float(row["depth_bin_hi"]),
                        "depth_bin_label": str(row["depth_bin_label"]),
                        "depth_bin_is_open": bool(row["depth_bin_is_open"]),
                        "depth_bin_center_plot": float(row["depth_bin_center_plot"]),
                        "depth_bin_used_plaquettes": int(row["depth_bin_used_plaquettes"]),
                        "depth_bin_defect_count": int(row["depth_bin_defect_count"]),
                        "depth_bin_density": float(row["depth_bin_density"]),
                    }
                )

        detail_path = analysis_dir / f"shell_depth_after_Tc_{offset_s:.2e}.csv"
        write_csv(detail_path, detail_rows)

        summary_rows = summarize_bin_rows(detail_rows)
        summary_path = analysis_dir / f"shell_depth_summary_after_Tc_{offset_s:.2e}.csv"
        write_csv(summary_path, summary_rows)

        scan_rows = build_shell_exclusion_scan(
            per_run_entries,
            max_exclude_layers=args.max_exclude_layers,
        )
        scan_path = analysis_dir / f"shell_depth_scan_after_Tc_{offset_s:.2e}.csv"
        write_csv(scan_path, scan_rows)

        band_summary = build_shell_band_summary(
            scan_rows,
            min_ratio=float(args.min_ratio),
            min_remaining_fraction=float(args.min_remaining_fraction),
        )
        band_summary_rows.append(
            {
                "target_after_Tc_s": float(offset_s),
                "max_excluded_layers": int(band_summary["max_excluded_layers"]),
                "remaining_mean_fraction": float(band_summary["remaining_mean_fraction"]),
                "remaining_min_ratio": float(band_summary["remaining_min_ratio"]),
                "remaining_min_corr": float(band_summary["remaining_min_corr"]),
                "remaining_min_slope": float(band_summary["remaining_min_slope"]),
                "min_ratio_threshold": float(args.min_ratio),
                "min_remaining_fraction_threshold": float(args.min_remaining_fraction),
                "detail_csv": str(detail_path.as_posix()),
                "summary_csv": str(summary_path.as_posix()),
                "scan_csv": str(scan_path.as_posix()),
                "profile_plot_png": str((analysis_dir / f"shell_depth_after_Tc_{offset_s:.2e}.png").as_posix()),
                "scan_plot_png": str((analysis_dir / f"shell_depth_scan_after_Tc_{offset_s:.2e}.png").as_posix()),
            }
        )

        plot_shell_profile(
            analysis_dir / f"shell_depth_after_Tc_{offset_s:.2e}.png",
            offset_s=float(offset_s),
            detail_rows=detail_rows,
            summary_rows=summary_rows,
        )
        plot_shell_scan(
            analysis_dir / f"shell_depth_scan_after_Tc_{offset_s:.2e}.png",
            offset_s=float(offset_s),
            scan_rows=scan_rows,
            band_summary=band_summary,
        )

        print(
            f"shell_depth after_Tc={float(offset_s):.3e} | max_excluded_layers={int(band_summary['max_excluded_layers'])} | "
            f"remaining_fraction={float(band_summary['remaining_mean_fraction']):.6g} | "
            f"remaining_min_ratio={float(band_summary['remaining_min_ratio']):.6g}"
        )

    band_summary_path = analysis_dir / "shell_depth_band_summary.csv"
    write_csv(band_summary_path, band_summary_rows)
    print(f"band_summary_csv={band_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())