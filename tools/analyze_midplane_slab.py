#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import QSRvis as qv


type SliceRow = dict[str, float | int]


@dataclass(frozen=True)
class SelectedRun:
    label: str
    metrics: dict[str, float]
    field_path: Path
    selected_iter: int
    selected_time_s: float
    actual_after_tc_s: float
    selected_avg_s: float
    z_center: int
    slice_rows: list[SliceRow]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Screen a localized midplane slab defect observable over a confined rate sweep. "
            "The observable is the total number of nematic defect plaquettes across a contiguous "
            "midplane slab divided by the total number of valid plaquettes in that slab."
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
        "--half-widths",
        type=str,
        default="0,1,2,4",
        help="Comma-separated slab half-widths in slices. half_width=0 means the single midplane slice.",
    )
    parser.add_argument(
        "--center-index",
        type=int,
        default=None,
        help="Optional z index for the slab center. Defaults to the geometric midplane for each selected snapshot.",
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
        "--analysis-dir",
        type=Path,
        default=None,
        help="Directory where the summary CSV and per-window detail CSVs are written.",
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


def slice_defect_counts(
    S2: np.ndarray,
    nx2: np.ndarray,
    ny2: np.ndarray,
    *,
    S_threshold: float,
    charge_cutoff: float,
) -> tuple[float, int, int]:
    mask = (S2 > float(S_threshold)) & np.isfinite(S2) & np.isfinite(nx2) & np.isfinite(ny2)
    psi = 2.0 * np.arctan2(ny2, nx2)
    p00 = psi[:-1, :-1]
    p10 = psi[1:, :-1]
    p11 = psi[1:, 1:]
    p01 = psi[:-1, 1:]
    dsum = (
        qv._wrap_to_pi(p10 - p00)
        + qv._wrap_to_pi(p11 - p10)
        + qv._wrap_to_pi(p01 - p11)
        + qv._wrap_to_pi(p00 - p01)
    )
    s_map = 0.5 * (dsum / (2.0 * np.pi))
    plaq_mask = mask[:-1, :-1] & mask[1:, :-1] & mask[1:, 1:] & mask[:-1, 1:]
    n_plaquettes = int(np.count_nonzero(plaq_mask))
    if n_plaquettes <= 0:
        return float("nan"), 0, 0
    defects = (np.abs(s_map) > float(charge_cutoff)) & plaq_mask
    n_defects = int(np.count_nonzero(defects))
    density = float(n_defects) / float(n_plaquettes)
    return density, n_defects, n_plaquettes


def clamp_center_index(center_index: int | None, Nz: int) -> int:
    if center_index is None:
        return int(Nz) // 2
    return max(0, min(int(Nz) - 1, int(center_index)))


def compute_slice_rows(
    field_path: Path,
    *,
    center_index: int | None,
    max_half_width: int,
    S_threshold: float,
    charge_cutoff: float,
) -> tuple[int, list[SliceRow]]:
    Nx, Ny, Nz = qv.infer_grid_dims_from_nematic_field_file(str(field_path))
    z_center = clamp_center_index(center_index, Nz)
    S, nx, ny, _ = qv.load_nematic_field_volume(str(field_path), Nx, Ny, Nz)

    z_lo = max(0, z_center - int(max_half_width))
    z_hi = min(Nz - 1, z_center + int(max_half_width))
    rows: list[SliceRow] = []
    for z_idx in range(z_lo, z_hi + 1):
        density, n_defects, n_plaquettes = slice_defect_counts(
            S[:, :, z_idx],
            nx[:, :, z_idx],
            ny[:, :, z_idx],
            S_threshold=S_threshold,
            charge_cutoff=charge_cutoff,
        )
        rows.append(
            {
                "slice_index": int(z_idx),
                "slice_density": float(density),
                "slice_defects": int(n_defects),
                "slice_plaquettes": int(n_plaquettes),
            }
        )
    return z_center, rows


def combine_slab_rows(
    slice_rows: list[SliceRow],
    *,
    z_center: int,
    half_width: int,
) -> dict[str, float | int]:
    selected = [
        row
        for row in slice_rows
        if abs(int(row["slice_index"]) - int(z_center)) <= int(half_width)
    ]

    total_defects = int(sum(int(row["slice_defects"]) for row in selected))
    total_plaquettes = int(sum(int(row["slice_plaquettes"]) for row in selected))
    slice_densities = np.array(
        [float(row["slice_density"]) for row in selected if np.isfinite(float(row["slice_density"]))],
        dtype=float,
    )

    slab_density = float(total_defects) / float(total_plaquettes) if total_plaquettes > 0 else float("nan")
    slice_mean = float(np.mean(slice_densities)) if slice_densities.size else float("nan")
    slice_std = float(np.std(slice_densities)) if slice_densities.size else float("nan")
    return {
        "slab_density": slab_density,
        "slice_density_mean": slice_mean,
        "slice_density_std": slice_std,
        "total_defects": total_defects,
        "total_plaquettes": total_plaquettes,
        "slices_used": int(len(selected)),
    }


def write_detail_csv(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    sweep_root = args.sweep_root.resolve()
    offsets = [float(item) for item in parse_csv_list(args.offsets)]
    half_widths = sorted({max(0, int(item)) for item in parse_csv_list(args.half_widths)})
    max_half_width = max(half_widths)

    analysis_dir = args.analysis_dir.resolve() if args.analysis_dir else (sweep_root / "analysis_midplane_slab").resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = list_run_dirs(sweep_root)
    rate_metrics = load_rate_metrics(sweep_root / "rate_sweep_metrics.csv")

    summary_rows: list[dict[str, float | str | int]] = []
    for offset_s in offsets:
        selected_runs: list[SelectedRun] = []
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

            z_center, slice_rows = compute_slice_rows(
                field_path,
                center_index=args.center_index,
                max_half_width=max_half_width,
                S_threshold=float(args.S_threshold),
                charge_cutoff=float(args.charge_cutoff),
            )
            selected_runs.append(
                SelectedRun(
                    label=label,
                    metrics=metrics,
                    field_path=field_path,
                    selected_iter=selected_iter,
                    selected_time_s=selected_time_s,
                    actual_after_tc_s=actual_after_tc_s,
                    selected_avg_s=selected_avg_s,
                    z_center=z_center,
                    slice_rows=slice_rows,
                )
            )

        for half_width in half_widths:
            detail_rows: list[dict[str, float | str | int]] = []
            tau_q_vals: list[float] = []
            rate_vals: list[float] = []
            density_vals: list[float] = []
            actual_offsets: list[float] = []
            selected_avg_s_vals: list[float] = []

            for selected in selected_runs:
                slab = combine_slab_rows(
                    selected.slice_rows,
                    z_center=selected.z_center,
                    half_width=int(half_width),
                )

                tau_q_vals.append(float(selected.metrics["t_ramp_s"]))
                rate_vals.append(float(selected.metrics["rate_K_per_s"]))
                density_vals.append(float(slab["slab_density"]))
                actual_offsets.append(float(selected.actual_after_tc_s))
                selected_avg_s_vals.append(float(selected.selected_avg_s))

                detail_rows.append(
                    {
                        "run": selected.label,
                        "ramp_iters": float(selected.metrics["ramp_iters"]),
                        "t_ramp_s": float(selected.metrics["t_ramp_s"]),
                        "rate_K_per_s": float(selected.metrics["rate_K_per_s"]),
                        "target_after_Tc_s": float(offset_s),
                        "actual_after_Tc_s": float(selected.actual_after_tc_s),
                        "selected_time_s": float(selected.selected_time_s),
                        "selected_iter": int(selected.selected_iter),
                        "field_file": selected.field_path.name,
                        "z_center": int(selected.z_center),
                        "slab_half_width": int(half_width),
                        "slab_slice_count": int(2 * half_width + 1),
                        "slices_used": int(slab["slices_used"]),
                        "slab_density": float(slab["slab_density"]),
                        "slice_density_mean": float(slab["slice_density_mean"]),
                        "slice_density_std": float(slab["slice_density_std"]),
                        "total_defects": int(slab["total_defects"]),
                        "total_plaquettes": int(slab["total_plaquettes"]),
                        "selected_avg_S": float(selected.selected_avg_s),
                        "final_after_Tc_s": float(selected.metrics["final_logged_time_s"] - selected.metrics["tc_crossing_time_s"]),
                        "final_avg_S": float(selected.metrics["final_avg_S"]),
                    }
                )

            detail_path = analysis_dir / f"midplane_slab_hw{half_width:02d}_after_Tc_{offset_s:.2e}.csv"
            write_detail_csv(detail_path, detail_rows)

            tau_q = np.array(tau_q_vals, dtype=float)
            rate = np.array(rate_vals, dtype=float)
            density = np.array(density_vals, dtype=float)
            actual_arr = np.array(actual_offsets, dtype=float)
            avg_s_arr = np.array(selected_avg_s_vals, dtype=float)

            summary_rows.append(
                {
                    "sweep_root": str(sweep_root.as_posix()),
                    "observable": "midplane_slab_defect_density",
                    "target_after_Tc_s": float(offset_s),
                    "slab_half_width": int(half_width),
                    "slab_slice_count": int(2 * half_width + 1),
                    "positive_cases": int(np.count_nonzero(np.isfinite(density) & (density > 0.0))),
                    "slope_vs_tauQ": fit_loglog_slope(tau_q, density),
                    "slope_vs_rate": fit_loglog_slope(rate, density),
                    "corr_vs_tauQ": positive_log_correlation(tau_q, density),
                    "corr_vs_rate": positive_log_correlation(rate, density),
                    "value_min": float(np.nanmin(density)),
                    "value_max": float(np.nanmax(density)),
                    "value_ratio_max_over_min": float(compute_ratio(density)),
                    "actual_after_Tc_min_s": float(np.nanmin(actual_arr)),
                    "actual_after_Tc_max_s": float(np.nanmax(actual_arr)),
                    "selected_avg_S_min": float(np.nanmin(avg_s_arr)),
                    "selected_avg_S_max": float(np.nanmax(avg_s_arr)),
                    "detail_csv": str(detail_path.as_posix()),
                }
            )

    summary_path = analysis_dir / "midplane_slab_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    for row in summary_rows:
        print(
            f"midplane_slab hw={int(row['slab_half_width'])} after_Tc={float(row['target_after_Tc_s']):.3e} | "
            f"slope_rate={float(row['slope_vs_rate']):.6g} corr_rate={float(row['corr_vs_rate']):.6g} "
            f"ratio={float(row['value_ratio_max_over_min']):.6g} avgS={float(row['selected_avg_S_min']):.6g}-{float(row['selected_avg_S_max']):.6g}"
        )
    print(f"summary_csv={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())