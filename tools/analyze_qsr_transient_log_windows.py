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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen transient fixed-after-Tc windows using logged 2D quench observables over a rate sweep.",
    )
    parser.add_argument("sweep_root", type=Path, help="Sweep root containing cases/*/output and rate_sweep_metrics.csv.")
    parser.add_argument(
        "--offsets",
        type=str,
        default="3.4e-8,3.6e-8,4.0e-8,5.0e-8,6.0e-8,7.0e-8,8.0e-8",
        help="Comma-separated offsets after Tc in seconds.",
    )
    parser.add_argument(
        "--observables",
        type=str,
        default="defect_density_per_plaquette,xi_grad_proxy,avg_S",
        help="Comma-separated quench-log observable columns to screen.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Directory where the summary CSV and per-window detail CSV are written.",
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
            "rate_K_per_s": float(row["cooling_rate_K_per_s"]),
            "tc_crossing_time_s": float(row["tc_crossing_time_s"]),
            "final_logged_time_s": float(row["final_logged_time_s"]),
            "final_avg_S": float(row["final_avg_S"]),
        }
    return metrics


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


def nearest_row_for_time(log_data: np.ndarray, target_time_s: float) -> tuple[int, float]:
    time_arr = np.atleast_1d(log_data["time_s"]).astype(float)
    finite = np.isfinite(time_arr)
    if not np.any(finite):
        raise ValueError("Quench log has no finite time values")
    idxs = np.nonzero(finite)[0]
    sub = time_arr[finite]
    j = int(np.argmin(np.abs(sub - target_time_s)))
    idx = int(idxs[j])
    return idx, float(time_arr[idx])


def write_detail_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
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
    observables = parse_csv_list(args.observables)

    analysis_dir = args.analysis_dir.resolve() if args.analysis_dir else (sweep_root / "analysis_transient_log_windows").resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = list_run_dirs(sweep_root)
    rate_metrics = load_rate_metrics(sweep_root / "rate_sweep_metrics.csv")

    summary_rows: list[dict[str, float | str]] = []
    for observable in observables:
        observable_dir = analysis_dir / observable
        observable_dir.mkdir(parents=True, exist_ok=True)
        for offset_s in offsets:
            detail_rows: list[dict[str, float | str]] = []
            tau_q_vals: list[float] = []
            rate_vals: list[float] = []
            obs_vals: list[float] = []
            actual_offsets: list[float] = []
            final_offsets: list[float] = []
            final_avg_s_vals: list[float] = []

            for run_dir in run_dirs:
                log_data, _ = qv.load_quench_log(str(run_dir))
                label = run_dir.parent.name
                metrics = rate_metrics.get(label)
                if metrics is None:
                    raise KeyError(f"Missing rate-sweep metrics for {label}")

                if observable not in (log_data.dtype.names or ()): 
                    raise KeyError(f"Observable {observable} not present in {run_dir / 'quench_log.dat'}")

                target_time_s = float(metrics["tc_crossing_time_s"]) + float(offset_s)
                idx, actual_time_s = nearest_row_for_time(log_data, target_time_s)
                actual_offset_s = actual_time_s - float(metrics["tc_crossing_time_s"])

                obs_value = float(np.atleast_1d(log_data[observable]).astype(float)[idx])
                xi_value = float(np.atleast_1d(log_data["xi_grad_proxy"]).astype(float)[idx]) if "xi_grad_proxy" in (log_data.dtype.names or ()) else float("nan")
                avg_s_value = float(np.atleast_1d(log_data["avg_S"]).astype(float)[idx]) if "avg_S" in (log_data.dtype.names or ()) else float("nan")
                iter_value = int(np.atleast_1d(log_data["iteration"]).astype(float)[idx])

                tau_q_vals.append(float(metrics["t_ramp_s"]))
                rate_vals.append(abs(float(metrics["rate_K_per_s"])))
                obs_vals.append(obs_value)
                actual_offsets.append(actual_offset_s)
                final_offsets.append(float(metrics["final_logged_time_s"] - metrics["tc_crossing_time_s"]))
                final_avg_s_vals.append(float(metrics["final_avg_S"]))

                detail_rows.append(
                    {
                        "run": label,
                        "ramp_iters": float(metrics["ramp_iters"]),
                        "t_ramp_s": float(metrics["t_ramp_s"]),
                        "rate_K_per_s": abs(float(metrics["rate_K_per_s"])),
                        "target_after_Tc_s": float(offset_s),
                        "actual_after_Tc_s": float(actual_offset_s),
                        "selected_time_s": float(actual_time_s),
                        "selected_iter": iter_value,
                        "observable": observable,
                        "observable_value": obs_value,
                        "selected_avg_S": avg_s_value,
                        "selected_xi_grad_proxy": xi_value,
                        "final_after_Tc_s": float(metrics["final_logged_time_s"] - metrics["tc_crossing_time_s"]),
                        "final_avg_S": float(metrics["final_avg_S"]),
                    }
                )

            detail_path = observable_dir / f"{observable}_after_Tc_{offset_s:.2e}.csv"
            write_detail_csv(detail_path, detail_rows)

            tau_q = np.array(tau_q_vals, dtype=float)
            rate = np.array(rate_vals, dtype=float)
            obs_arr = np.array(obs_vals, dtype=float)
            actual_arr = np.array(actual_offsets, dtype=float)
            final_offset_arr = np.array(final_offsets, dtype=float)
            final_avg_s_arr = np.array(final_avg_s_vals, dtype=float)

            summary_rows.append(
                {
                    "sweep_root": str(sweep_root.as_posix()),
                    "observable": observable,
                    "target_after_Tc_s": float(offset_s),
                    "positive_cases": int(np.count_nonzero(np.isfinite(obs_arr) & (obs_arr > 0.0))),
                    "slope_vs_tauQ": fit_loglog_slope(tau_q, obs_arr),
                    "slope_vs_rate": fit_loglog_slope(rate, obs_arr),
                    "corr_vs_tauQ": positive_log_correlation(tau_q, obs_arr),
                    "corr_vs_rate": positive_log_correlation(rate, obs_arr),
                    "value_min": float(np.nanmin(obs_arr)),
                    "value_max": float(np.nanmax(obs_arr)),
                    "value_ratio_max_over_min": float(compute_ratio(obs_arr)),
                    "actual_after_Tc_min_s": float(np.nanmin(actual_arr)),
                    "actual_after_Tc_max_s": float(np.nanmax(actual_arr)),
                    "final_after_Tc_min_s": float(np.nanmin(final_offset_arr)),
                    "final_after_Tc_max_s": float(np.nanmax(final_offset_arr)),
                    "final_avg_S_min": float(np.nanmin(final_avg_s_arr)),
                    "final_avg_S_max": float(np.nanmax(final_avg_s_arr)),
                    "detail_csv": str(detail_path.as_posix()),
                }
            )

    summary_path = analysis_dir / "transient_log_window_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    for row in summary_rows:
        print(
            f"{row['observable']} after_Tc={float(row['target_after_Tc_s']):.3e} | "
            f"slope_rate={float(row['slope_vs_rate']):.6g} corr_rate={float(row['corr_vs_rate']):.6g} "
            f"ratio={float(row['value_ratio_max_over_min']):.6g} final_avgS={float(row['final_avg_S_min']):.6g}-{float(row['final_avg_S_max']):.6g}"
        )
    print(f"summary_csv={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())