#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import QSRvis as qv


PRESETS: dict[str, dict[str, int | float | bool]] = {
    "default": {
        "S_droplet": 0.1,
        "S_core": 0.05,
        "dilate_iters": 2,
        "fill_holes": False,
        "core_erosion_iters": 0,
        "min_core_voxels": 30,
    },
    "conservative": {
        "S_droplet": 0.1,
        "S_core": 0.05,
        "dilate_iters": 0,
        "fill_holes": True,
        "core_erosion_iters": 1,
        "min_core_voxels": 30,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate 3D KZ metrics over a quench screen and write a compact summary table.",
    )
    parser.add_argument("sweep_root", type=Path, help="Sweep root containing cases/*/output directories.")
    parser.add_argument(
        "--offsets",
        type=str,
        default="3.4e-8,3.6e-8",
        help="Comma-separated offsets after Tc in seconds.",
    )
    parser.add_argument(
        "--proxies",
        type=str,
        default="core,skeleton",
        help="Comma-separated defect proxies to aggregate.",
    )
    parser.add_argument(
        "--measure-mode",
        type=str,
        default="fixed",
        choices=("fixed", "avg_s", "auto"),
        help="How to choose the measurement point: fixed after-Tc offset, fixed avg_S milestone, or auto.",
    )
    parser.add_argument(
        "--avgs-targets",
        type=str,
        default="0.1",
        help="Comma-separated avg_S milestone values for measure-mode=avg_s or auto.",
    )
    parser.add_argument(
        "--extra-after-target-s",
        type=float,
        default=0.0,
        help="Optional extra delay after the avg_S milestone when measure-mode=avg_s or auto.",
    )
    parser.add_argument(
        "--beta-min",
        type=float,
        default=0.5,
        help="Minimum biaxiality beta for biaxial core proxies.",
    )
    parser.add_argument(
        "--core-region-mode",
        type=str,
        default="morphology",
        choices=("morphology", "distance"),
        help="How to define the 3D core search region.",
    )
    parser.add_argument(
        "--shell-exclude-layers",
        type=float,
        default=0.0,
        help="For core-region-mode=distance, exclude this many voxel layers from the droplet interface.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        choices=sorted(PRESETS.keys()),
        help="Morphology preset for the 3D low-S core extraction.",
    )
    parser.add_argument("--tc", type=float, default=310.2, help="Transition temperature in K.")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Directory where per-offset CSVs and the summary CSV are written.",
    )
    return parser.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated value")
    return items


def list_run_dirs(sweep_root: Path) -> list[str]:
    run_dirs = []
    for case_dir in sorted((sweep_root / "cases").glob("*")):
        output_dir = case_dir / "output"
        if (output_dir / "quench_log.dat").exists():
            run_dirs.append(str(output_dir))
    if not run_dirs:
        raise FileNotFoundError(f"No cases/*/output run directories with quench_log.dat found in {sweep_root}")
    return run_dirs


def load_metric_table(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def positive_log_correlation(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if int(np.count_nonzero(mask)) < 2:
        return float("nan")
    return float(np.corrcoef(np.log(x[mask]), np.log(y[mask]))[0, 1])


def compute_summary_rows(
    sweep_root: Path,
    analysis_dir: Path,
    offsets: list[float],
    avgS_targets: list[float],
    proxies: list[str],
    preset_name: str,
    tc_k: float,
    measure_mode: str,
    extra_after_target_s: float,
    beta_min: float,
    core_region_mode: str,
    shell_exclude_layers: float,
) -> list[dict[str, str | float]]:
    params = PRESETS[preset_name]
    run_dirs = list_run_dirs(sweep_root)
    summary_rows: list[dict[str, str | float]] = []

    for proxy in proxies:
        proxy_dir = analysis_dir / proxy
        proxy_dir.mkdir(parents=True, exist_ok=True)
        targets = offsets if measure_mode == "fixed" else avgS_targets
        for target in targets:
            if measure_mode == "fixed":
                agg_result = qv.aggregate_kz_scaling_3d(
                    parent_dir=str(sweep_root),
                    run_dirs=run_dirs,
                    out_dir=str(proxy_dir),
                    x_axis="rate",
                    measure="after_Tc",
                    Tc=float(tc_k),
                    after_Tc_s=float(target),
                    S_threshold_xi=0.1,
                    S_droplet=float(params["S_droplet"]),
                    S_core=float(params["S_core"]),
                    beta_min=float(beta_min),
                    core_region_mode=str(core_region_mode),
                    shell_exclude_layers=float(shell_exclude_layers),
                    dilate_iters=int(params["dilate_iters"]),
                    fill_holes=bool(params["fill_holes"]),
                    core_erosion_iters=int(params["core_erosion_iters"]),
                    min_core_voxels=int(params["min_core_voxels"]),
                    defect_proxy=proxy,
                    show=False,
                    plot=False,
                    write_files=True,
                    close=True,
                )
            else:
                agg_result = qv.aggregate_kz_scaling_3d(
                    parent_dir=str(sweep_root),
                    run_dirs=run_dirs,
                    out_dir=str(proxy_dir),
                    x_axis="rate",
                    measure="after_Tc",
                    Tc=float(tc_k),
                    after_Tc_mode=measure_mode,
                    avgS_target=float(target),
                    extra_after_target_s=float(extra_after_target_s),
                    S_threshold_xi=0.1,
                    S_droplet=float(params["S_droplet"]),
                    S_core=float(params["S_core"]),
                    beta_min=float(beta_min),
                    core_region_mode=str(core_region_mode),
                    shell_exclude_layers=float(shell_exclude_layers),
                    dilate_iters=int(params["dilate_iters"]),
                    fill_holes=bool(params["fill_holes"]),
                    core_erosion_iters=int(params["core_erosion_iters"]),
                    min_core_voxels=int(params["min_core_voxels"]),
                    defect_proxy=proxy,
                    show=False,
                    plot=False,
                    write_files=True,
                    close=True,
                )
            csv_path = agg_result[2]
            if not csv_path:
                raise RuntimeError(f"aggregate_kz_scaling_3d did not produce a CSV for proxy={proxy}, target={target}")

            rows = load_metric_table(Path(csv_path))
            rate = np.array([float(row["rate_K_per_s"]) for row in rows], dtype=float)
            xi = np.array([float(row["xi3d_lattice"]) for row in rows], dtype=float)
            defect_col = 'defect_metric' if ('defect_metric' in rows[0]) else 'defect_density_per_system_vox'
            defect = np.array([float(row[defect_col]) for row in rows], dtype=float)
            actual_after = np.array([float(row["after_Tc_actual_s"]) for row in rows], dtype=float)

            slope_xi, _ = qv._fit_powerlaw_loglog(rate, xi)
            slope_defect, _ = qv._fit_powerlaw_loglog(rate, defect)
            corr_xi = positive_log_correlation(rate, xi)
            corr_defect = positive_log_correlation(rate, defect)

            defect_pos = defect[np.isfinite(defect) & (defect >= 0.0)]
            defect_min = float(np.min(defect_pos)) if defect_pos.size else float("nan")
            defect_max = float(np.max(defect_pos)) if defect_pos.size else float("nan")
            if np.isfinite(defect_min) and defect_min > 0.0 and np.isfinite(defect_max):
                defect_ratio = float(defect_max / defect_min)
            elif np.isfinite(defect_min) and defect_min == 0.0 and np.isfinite(defect_max) and defect_max > 0.0:
                defect_ratio = float("inf")
            else:
                defect_ratio = float("nan")

            summary_rows.append(
                {
                    "preset": preset_name,
                    "proxy": proxy,
                    "measure_mode": measure_mode,
                    "target_after_Tc_s": float(target) if measure_mode == "fixed" else float("nan"),
                    "target_avg_S": float(target) if measure_mode != "fixed" else float("nan"),
                    "extra_after_target_s": float(extra_after_target_s) if measure_mode != "fixed" else 0.0,
                    "actual_after_Tc_min_s": float(np.nanmin(actual_after)),
                    "actual_after_Tc_max_s": float(np.nanmax(actual_after)),
                    "slope_xi_vs_rate": float(slope_xi),
                    "slope_defect_vs_rate": float(slope_defect),
                    "corr_xi_vs_rate": float(corr_xi),
                    "corr_defect_vs_rate": float(corr_defect),
                    "defect_min": defect_min,
                    "defect_max": defect_max,
                    "defect_ratio_max_over_min": defect_ratio,
                    "csv_path": str(Path(csv_path).as_posix()),
                }
            )
    return summary_rows


def write_summary_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    fieldnames = [
        "preset",
        "proxy",
        "measure_mode",
        "target_after_Tc_s",
        "target_avg_S",
        "extra_after_target_s",
        "actual_after_Tc_min_s",
        "actual_after_Tc_max_s",
        "slope_xi_vs_rate",
        "slope_defect_vs_rate",
        "corr_xi_vs_rate",
        "corr_defect_vs_rate",
        "defect_min",
        "defect_max",
        "defect_ratio_max_over_min",
        "csv_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    sweep_root = args.sweep_root.resolve()
    offsets = [float(item) for item in parse_csv_list(args.offsets)]
    avgS_targets = [float(item) for item in parse_csv_list(args.avgs_targets)]
    proxies = parse_csv_list(args.proxies)
    analysis_dir = args.analysis_dir.resolve() if args.analysis_dir else (sweep_root / "analysis" / args.preset).resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows = compute_summary_rows(
        sweep_root=sweep_root,
        analysis_dir=analysis_dir,
        offsets=offsets,
        avgS_targets=avgS_targets,
        proxies=proxies,
        preset_name=args.preset,
        tc_k=float(args.tc),
        measure_mode=str(args.measure_mode),
        extra_after_target_s=float(args.extra_after_target_s),
        beta_min=float(args.beta_min),
        core_region_mode=str(args.core_region_mode),
        shell_exclude_layers=float(args.shell_exclude_layers),
    )
    summary_path = analysis_dir / f"{args.preset}_summary.csv"
    write_summary_csv(summary_path, rows)

    for row in rows:
        if str(row["measure_mode"]) == "fixed":
            target_desc = f"after_Tc={float(row['target_after_Tc_s']):.3e} s"
        else:
            target_desc = f"avg_S={float(row['target_avg_S']):.6g}"
            extra = float(row["extra_after_target_s"])
            if extra != 0.0:
                target_desc += f" + {extra:.3e} s"
        print(
            f"[{row['preset']}] {row['proxy']} {target_desc} | "
            f"slope_xi={row['slope_xi_vs_rate']:.6g} slope_defect={row['slope_defect_vs_rate']:.6g} "
            f"corr_defect={row['corr_defect_vs_rate']:.6g} ratio={row['defect_ratio_max_over_min']:.6g}"
        )
    print(f"summary_csv={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())