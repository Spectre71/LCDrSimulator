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
        description="Aggregate final-state 3D confined-droplet defect metrics over a quench sweep.",
    )
    parser.add_argument("sweep_root", type=Path, help="Sweep root containing cases/*/output directories.")
    parser.add_argument(
        "--proxies",
        type=str,
        default="skeleton_droplet,core_droplet",
        help="Comma-separated defect proxies to aggregate.",
    )
    parser.add_argument(
        "--presets",
        type=str,
        default="default",
        help="Comma-separated morphology presets to evaluate.",
    )
    parser.add_argument("--tc", type=float, default=310.2, help="Transition temperature in K.")
    parser.add_argument(
        "--beta-min",
        type=float,
        default=0.5,
        help="Minimum biaxiality beta for biaxial-core proxies.",
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
        "--analysis-dir",
        type=Path,
        default=None,
        help="Directory where per-proxy CSVs and the summary CSV are written.",
    )
    return parser.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated value")
    return items


def list_run_dirs(sweep_root: Path) -> list[str]:
    run_dirs: list[str] = []
    for case_dir in sorted((sweep_root / "cases").glob("*")):
        output_dir = case_dir / "output"
        if (output_dir / "quench_log.dat").exists():
            run_dirs.append(str(output_dir))
    if not run_dirs:
        raise FileNotFoundError(f"No cases/*/output run directories with quench_log.dat found in {sweep_root}")
    return run_dirs


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_rate_metrics(path: Path) -> dict[str, dict[str, float]]:
    rows = load_csv_rows(path)
    metrics: dict[str, dict[str, float]] = {}
    for row in rows:
        label = row["label"]
        metrics[label] = {
            "final_avg_S": float(row["final_avg_S"]),
            "final_time_s": float(row["final_logged_time_s"]),
            "tc_cross_s": float(row["tc_crossing_time_s"]),
        }
    return metrics


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


def aggregate_proxy(
    sweep_root: Path,
    analysis_dir: Path,
    run_dirs: list[str],
    proxy: str,
    preset_name: str,
    tc_k: float,
    beta_min: float,
    core_region_mode: str,
    shell_exclude_layers: float,
) -> tuple[list[dict[str, str]], Path]:
    params = PRESETS[preset_name]
    proxy_dir = analysis_dir / preset_name / proxy
    proxy_dir.mkdir(parents=True, exist_ok=True)
    _, _, csv_path = qv.aggregate_kz_scaling_3d(
        parent_dir=str(sweep_root),
        run_dirs=run_dirs,
        out_dir=str(proxy_dir),
        x_axis="t_ramp",
        measure="final",
        Tc=float(tc_k),
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
    if not csv_path:
        raise RuntimeError(f"aggregate_kz_scaling_3d did not produce a CSV for proxy={proxy}, preset={preset_name}")
    csv_path_obj = Path(csv_path)
    return load_csv_rows(csv_path_obj), csv_path_obj


def compute_summary_row(
    sweep_root: Path,
    preset_name: str,
    proxy: str,
    rows: list[dict[str, str]],
    csv_path: Path,
    rate_metrics: dict[str, dict[str, float]],
) -> dict[str, str | float]:
    tau_q = np.array([float(row["t_ramp_s"]) for row in rows], dtype=float)
    rate = np.array([float(row["rate_K_per_s"]) for row in rows], dtype=float)
    defect = np.array([float(row["defect_metric"]) for row in rows], dtype=float)
    xi3d = np.array([float(row["xi3d_lattice"]) for row in rows], dtype=float)
    after_tc_actual = np.array([float(row["after_Tc_actual_s"]) for row in rows], dtype=float)

    slope_defect_vs_tau_q, _ = qv._fit_powerlaw_loglog(tau_q, defect)
    slope_defect_vs_rate, _ = qv._fit_powerlaw_loglog(rate, defect)
    slope_xi_vs_tau_q, _ = qv._fit_powerlaw_loglog(tau_q, xi3d)
    corr_defect_vs_tau_q = positive_log_correlation(tau_q, defect)
    corr_defect_vs_rate = positive_log_correlation(rate, defect)

    final_avg_s_values: list[float] = []
    final_after_tc_values: list[float] = []
    for row in rows:
        label = row["run"]
        metric = rate_metrics.get(label)
        if metric is None:
            raise KeyError(f"Missing rate-sweep metrics for run label {label} in {sweep_root}")
        final_avg_s_values.append(float(metric["final_avg_S"]))
        final_after_tc_values.append(float(metric["final_time_s"] - metric["tc_cross_s"]))

    final_avg_s_arr = np.array(final_avg_s_values, dtype=float)
    final_after_tc_arr = np.array(final_after_tc_values, dtype=float)
    finite_after_tc_actual = after_tc_actual[np.isfinite(after_tc_actual)]

    if finite_after_tc_actual.size:
        actual_after_tc_min_s = float(np.min(finite_after_tc_actual))
        actual_after_tc_max_s = float(np.max(finite_after_tc_actual))
    else:
        actual_after_tc_min_s = float("nan")
        actual_after_tc_max_s = float("nan")

    positive_cases = int(np.count_nonzero(np.isfinite(defect) & (defect > 0.0)))
    defect_ratio = compute_ratio(defect)
    defect_metric_kind = rows[0].get("defect_metric_kind", "unknown") if rows else "unknown"

    return {
        "sweep_root": str(sweep_root.as_posix()),
        "preset": preset_name,
        "proxy": proxy,
        "defect_metric_kind": defect_metric_kind,
        "positive_cases": positive_cases,
        "slope_defect_vs_tauQ": float(slope_defect_vs_tau_q),
        "slope_defect_vs_rate": float(slope_defect_vs_rate),
        "slope_xi_vs_tauQ": float(slope_xi_vs_tau_q),
        "corr_defect_vs_tauQ": float(corr_defect_vs_tau_q),
        "corr_defect_vs_rate": float(corr_defect_vs_rate),
        "defect_min": float(np.nanmin(defect)),
        "defect_max": float(np.nanmax(defect)),
        "defect_ratio_max_over_min": float(defect_ratio),
        "actual_after_Tc_min_s": actual_after_tc_min_s,
        "actual_after_Tc_max_s": actual_after_tc_max_s,
        "final_after_Tc_min_s": float(np.nanmin(final_after_tc_arr)),
        "final_after_Tc_max_s": float(np.nanmax(final_after_tc_arr)),
        "final_avg_S_min": float(np.nanmin(final_avg_s_arr)),
        "final_avg_S_max": float(np.nanmax(final_avg_s_arr)),
        "csv_path": str(csv_path.as_posix()),
    }


def write_summary_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    fieldnames = [
        "sweep_root",
        "preset",
        "proxy",
        "defect_metric_kind",
        "positive_cases",
        "slope_defect_vs_tauQ",
        "slope_defect_vs_rate",
        "slope_xi_vs_tauQ",
        "corr_defect_vs_tauQ",
        "corr_defect_vs_rate",
        "defect_min",
        "defect_max",
        "defect_ratio_max_over_min",
        "actual_after_Tc_min_s",
        "actual_after_Tc_max_s",
        "final_after_Tc_min_s",
        "final_after_Tc_max_s",
        "final_avg_S_min",
        "final_avg_S_max",
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
    analysis_dir = args.analysis_dir.resolve() if args.analysis_dir else (sweep_root / "analysis_final").resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    proxies = parse_csv_list(args.proxies)
    presets = parse_csv_list(args.presets)
    unknown_presets = [preset for preset in presets if preset not in PRESETS]
    if unknown_presets:
        raise ValueError(f"Unknown presets: {unknown_presets}. Available presets: {sorted(PRESETS)}")

    run_dirs = list_run_dirs(sweep_root)
    rate_metrics = load_rate_metrics(sweep_root / "rate_sweep_metrics.csv")

    summary_rows: list[dict[str, str | float]] = []
    for preset_name in presets:
        for proxy in proxies:
            rows, csv_path = aggregate_proxy(
                sweep_root=sweep_root,
                analysis_dir=analysis_dir,
                run_dirs=run_dirs,
                proxy=proxy,
                preset_name=preset_name,
                tc_k=float(args.tc),
                beta_min=float(args.beta_min),
                core_region_mode=str(args.core_region_mode),
                shell_exclude_layers=float(args.shell_exclude_layers),
            )
            summary_rows.append(
                compute_summary_row(
                    sweep_root=sweep_root,
                    preset_name=preset_name,
                    proxy=proxy,
                    rows=rows,
                    csv_path=csv_path,
                    rate_metrics=rate_metrics,
                )
            )

    summary_path = analysis_dir / "confined_final_summary.csv"
    write_summary_csv(summary_path, summary_rows)

    for row in summary_rows:
        print(
            f"[{row['preset']}] {row['proxy']} | "
            f"slope_tauQ={row['slope_defect_vs_tauQ']:.6g} slope_rate={row['slope_defect_vs_rate']:.6g} "
            f"corr_tauQ={row['corr_defect_vs_tauQ']:.6g} ratio={row['defect_ratio_max_over_min']:.6g} "
            f"final_avgS={row['final_avg_S_min']:.6g}-{row['final_avg_S_max']:.6g}"
        )
    print(f"summary_csv={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())