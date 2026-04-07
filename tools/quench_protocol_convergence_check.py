#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


OBSERVABLE_FIELDS = (
    "bulk",
    "elastic",
    "anchoring",
    "total",
    "radiality",
    "avg_S",
    "max_S",
    "defect_density_per_plaquette",
    "xi_grad_proxy",
)


@dataclass(frozen=True)
class RunMetrics:
    label: str
    config_path: str
    run_dir: str
    output_dir: str
    tc_K: float
    pre_equil_iters: int
    ramp_iters: int
    total_iters: int
    dt_config_s: float
    cooling_rate_K_per_s: float
    first_logged_time_s: float
    final_logged_time_s: float
    first_logged_T_K: float
    final_logged_T_K: float
    sample_count: int
    post_crossing_samples: int
    tc_crossing_time_s: float
    dt_min_s: float
    dt_max_s: float
    dt_span_s: float
    fixed_dt: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-dt quench protocol-convergence checks and compare post-crossing observables.",
    )
    parser.add_argument(
        "configs",
        nargs="+",
        type=Path,
        help="Config files to run, ordered from coarser protocol resolution to finer.",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("QSR_cuda"),
        help="Path to the QSR CUDA binary.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("validation/quench_protocol_convergence"),
        help="Directory where per-run outputs and comparison CSVs are written.",
    )
    parser.add_argument(
        "--tc",
        type=float,
        default=float("nan"),
        help="Transition temperature used for the crossing-time comparison. Defaults to Tc_KZ from each config.",
    )
    return parser.parse_args()


def parse_kv_config(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cfg[key.strip()] = value.strip()
    return cfg


def cfg_int(cfg: dict[str, str], key: str) -> int:
    return int(float(cfg[key]))


def cfg_float(cfg: dict[str, str], key: str) -> float:
    return float(cfg[key])


def clean_run_dir(run_dir: Path) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def load_quench_log(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if getattr(data, "size", 0) == 0:
        raise ValueError(f"Quench log is empty: {path}")
    return np.array(data, ndmin=1)


def series(data: np.ndarray, *names: str) -> np.ndarray:
    field_names = set(data.dtype.names or ())
    for name in names:
        if name in field_names:
            return np.atleast_1d(data[name]).astype(float)
    raise ValueError(f"Missing required columns {names} in log with fields {sorted(field_names)}")


def crossing_time(time_s: np.ndarray, temperature_K: np.ndarray, tc_K: float) -> float:
    if not np.isfinite(tc_K):
        raise ValueError("Non-finite Tc supplied to crossing_time")

    for idx in range(time_s.size):
        if np.isfinite(temperature_K[idx]) and abs(temperature_K[idx] - tc_K) <= 1e-12:
            return float(time_s[idx])

    for idx in range(1, time_s.size):
        T0 = temperature_K[idx - 1]
        T1 = temperature_K[idx]
        if not (np.isfinite(T0) and np.isfinite(T1)):
            continue
        if (T0 - tc_K) * (T1 - tc_K) < 0.0:
            t0 = time_s[idx - 1]
            t1 = time_s[idx]
            if not (np.isfinite(t0) and np.isfinite(t1)):
                continue
            return float(t0 + (tc_K - T0) * (t1 - t0) / (T1 - T0))

    raise ValueError(f"Could not find a Tc crossing for Tc={tc_K} K")


def interpolate_at(time_s: np.ndarray, values: np.ndarray, target_time_s: float) -> float:
    if target_time_s < time_s[0] - 1e-18 or target_time_s > time_s[-1] + 1e-18:
        return float("nan")

    idx = int(np.searchsorted(time_s, target_time_s, side="left"))
    if idx < time_s.size and abs(time_s[idx] - target_time_s) <= 1e-18:
        return float(values[idx])
    if idx <= 0:
        return float(values[0])
    if idx >= time_s.size:
        return float(values[-1])

    t0 = time_s[idx - 1]
    t1 = time_s[idx]
    v0 = values[idx - 1]
    v1 = values[idx]
    if not (np.isfinite(v0) and np.isfinite(v1) and np.isfinite(t0) and np.isfinite(t1)):
        return float("nan")
    if abs(t1 - t0) <= 1e-30:
        return float(v1)
    weight = (target_time_s - t0) / (t1 - t0)
    return float(v0 + weight * (v1 - v0))


def choose_tc(cfg: dict[str, str], tc_override: float) -> float:
    if np.isfinite(tc_override):
        return float(tc_override)
    if "Tc_KZ" in cfg:
        return float(cfg["Tc_KZ"])
    raise ValueError("Tc was not provided and Tc_KZ is missing from the config")


def reference_offsets(time_s: np.ndarray, tc_crossing_time_s: float, horizon_s: float) -> list[float]:
    offsets: list[float] = [0.0]
    for value in time_s:
        offset = float(value - tc_crossing_time_s)
        if offset <= 1e-18 or offset > horizon_s + 1e-18:
            continue
        if abs(offset - offsets[-1]) > 1e-18:
            offsets.append(offset)
    return offsets


def run_case(binary: Path, config_path: Path, output_root: Path, tc_override: float) -> tuple[RunMetrics, np.ndarray]:
    cfg = parse_kv_config(config_path)
    label = config_path.stem.replace("quench_protocol_convergence_", "")
    run_dir = output_root / label
    clean_run_dir(run_dir)

    result = subprocess.run(
        [str(binary), "--config", str(config_path)],
        cwd=run_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    (run_dir / "solver_stdout.log").write_text(result.stdout, encoding="utf-8")
    (run_dir / "solver_stderr.log").write_text(result.stderr, encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(
            f"Solver failed for {config_path} with code {result.returncode}. See {run_dir / 'solver_stderr.log'}"
        )

    output_dir = run_dir / cfg.get("out_dir", "output")
    log_path = output_dir / "quench_log.dat"
    if not log_path.exists():
        raise FileNotFoundError(f"Expected quench log not found: {log_path}")

    data = load_quench_log(log_path)
    time_s = series(data, "time_s", "time")
    temperature_K = series(data, "T_K")
    dt_s = series(data, "dt_s", "dt")
    tc_K = choose_tc(cfg, tc_override)
    tc_cross = crossing_time(time_s, temperature_K, tc_K)

    pre_equil_iters = cfg_int(cfg, "pre_equil_iters")
    ramp_iters = cfg_int(cfg, "ramp_iters")
    total_iters = cfg_int(cfg, "total_iters")
    dt_config_s = cfg_float(cfg, "dt")
    T_high_K = cfg_float(cfg, "T_high")
    T_low_K = cfg_float(cfg, "T_low")
    cooling_rate = 0.0
    if ramp_iters > 0 and dt_config_s > 0.0:
        cooling_rate = (T_low_K - T_high_K) / (float(ramp_iters) * dt_config_s)

    post_crossing_samples = int(np.count_nonzero(time_s >= tc_cross - 1e-18))
    dt_min_s = float(np.nanmin(dt_s))
    dt_max_s = float(np.nanmax(dt_s))
    dt_span_s = dt_max_s - dt_min_s

    metrics = RunMetrics(
        label=label,
        config_path=str(config_path),
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        tc_K=tc_K,
        pre_equil_iters=pre_equil_iters,
        ramp_iters=ramp_iters,
        total_iters=total_iters,
        dt_config_s=dt_config_s,
        cooling_rate_K_per_s=cooling_rate,
        first_logged_time_s=float(time_s[0]),
        final_logged_time_s=float(time_s[-1]),
        first_logged_T_K=float(temperature_K[0]),
        final_logged_T_K=float(temperature_K[-1]),
        sample_count=int(time_s.size),
        post_crossing_samples=post_crossing_samples,
        tc_crossing_time_s=tc_cross,
        dt_min_s=dt_min_s,
        dt_max_s=dt_max_s,
        dt_span_s=dt_span_s,
        fixed_dt=bool(dt_span_s <= 1e-18),
    )
    return metrics, data


def write_metrics_csv(path: Path, metrics: list[RunMetrics]) -> None:
    fieldnames = list(RunMetrics.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            writer.writerow(item.__dict__)


def build_offset_rows(metrics: list[RunMetrics], logs: list[np.ndarray]) -> tuple[list[dict[str, float | str]], list[float]]:
    reference_metrics = metrics[0]
    reference_log = logs[0]
    reference_time = series(reference_log, "time_s", "time")
    horizon_s = min(item.final_logged_time_s - item.tc_crossing_time_s for item in metrics)
    offsets = reference_offsets(reference_time, reference_metrics.tc_crossing_time_s, horizon_s)

    rows: list[dict[str, float | str]] = []
    for item, data in zip(metrics, logs):
        time_s = series(data, "time_s", "time")
        for offset_s in offsets:
            target_time_s = item.tc_crossing_time_s + offset_s
            row: dict[str, float | str] = {
                "label": item.label,
                "offset_s": offset_s,
                "time_s": target_time_s,
                "T_K": interpolate_at(time_s, series(data, "T_K"), target_time_s),
            }
            for field in OBSERVABLE_FIELDS:
                row[field] = interpolate_at(time_s, series(data, field), target_time_s)
            rows.append(row)
    return rows, offsets


def write_offsets_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = ["label", "offset_s", "time_s", "T_K", *OBSERVABLE_FIELDS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def relative_difference(a: float, b: float, floor: float = 1e-18) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    scale = max(abs(a), abs(b), floor)
    return abs(a - b) / scale


def absolute_difference(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return abs(a - b)


def write_comparison_csv(path: Path, metrics: list[RunMetrics], offset_rows: list[dict[str, float | str]], offsets: list[float]) -> None:
    reference = metrics[-1]
    fieldnames = [
        "reference_label",
        "test_label",
        "offset_s",
        "tc_crossing_time_diff_s",
        "abs_T_K_diff",
        "abs_total_diff",
        "rel_total_diff",
        "abs_radiality_diff",
        "abs_avg_S_diff",
        "abs_max_S_diff",
        "abs_defect_density_diff",
        "abs_xi_grad_proxy_diff",
    ]

    rows_by_label: dict[str, dict[float, dict[str, float | str]]] = {}
    for row in offset_rows:
        rows_by_label.setdefault(str(row["label"]), {})[float(row["offset_s"])] = row

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        reference_rows = rows_by_label[reference.label]
        for item in metrics[:-1]:
            test_rows = rows_by_label[item.label]
            for offset_s in offsets:
                ref_row = reference_rows[offset_s]
                test_row = test_rows[offset_s]
                writer.writerow(
                    {
                        "reference_label": reference.label,
                        "test_label": item.label,
                        "offset_s": offset_s,
                        "tc_crossing_time_diff_s": abs(item.tc_crossing_time_s - reference.tc_crossing_time_s),
                        "abs_T_K_diff": absolute_difference(float(test_row["T_K"]), float(ref_row["T_K"])),
                        "abs_total_diff": absolute_difference(float(test_row["total"]), float(ref_row["total"])),
                        "rel_total_diff": relative_difference(float(test_row["total"]), float(ref_row["total"])),
                        "abs_radiality_diff": absolute_difference(float(test_row["radiality"]), float(ref_row["radiality"])),
                        "abs_avg_S_diff": absolute_difference(float(test_row["avg_S"]), float(ref_row["avg_S"])),
                        "abs_max_S_diff": absolute_difference(float(test_row["max_S"]), float(ref_row["max_S"])),
                        "abs_defect_density_diff": absolute_difference(
                            float(test_row["defect_density_per_plaquette"]),
                            float(ref_row["defect_density_per_plaquette"]),
                        ),
                        "abs_xi_grad_proxy_diff": absolute_difference(
                            float(test_row["xi_grad_proxy"]),
                            float(ref_row["xi_grad_proxy"]),
                        ),
                    }
                )


def max_finite(values: list[float]) -> float:
    finite_values = [value for value in values if np.isfinite(value)]
    return max(finite_values) if finite_values else float("nan")


def print_summary(metrics: list[RunMetrics], offset_rows: list[dict[str, float | str]], offsets: list[float]) -> None:
    rows_by_label: dict[str, dict[float, dict[str, float | str]]] = {}
    for row in offset_rows:
        rows_by_label.setdefault(str(row["label"]), {})[float(row["offset_s"])] = row

    for item in metrics:
        print(
            f"[protocol-check] {item.label}: "
            f"tcross={item.tc_crossing_time_s:.8e} s, "
            f"dt=[{item.dt_min_s:.8e}, {item.dt_max_s:.8e}] s, "
            f"fixed_dt={'yes' if item.fixed_dt else 'no'}, "
            f"offset_samples={item.post_crossing_samples}"
        )

    reference = metrics[-1]
    reference_rows = rows_by_label[reference.label]
    for item in metrics[:-1]:
        test_rows = rows_by_label[item.label]
        total_abs_diffs = [
            absolute_difference(float(test_rows[offset]["total"]), float(reference_rows[offset]["total"]))
            for offset in offsets
        ]
        total_diffs = [
            relative_difference(float(test_rows[offset]["total"]), float(reference_rows[offset]["total"]))
            for offset in offsets
        ]
        radiality_diffs = [
            absolute_difference(float(test_rows[offset]["radiality"]), float(reference_rows[offset]["radiality"]))
            for offset in offsets
        ]
        avg_s_diffs = [
            absolute_difference(float(test_rows[offset]["avg_S"]), float(reference_rows[offset]["avg_S"]))
            for offset in offsets
        ]
        defect_diffs = [
            absolute_difference(
                float(test_rows[offset]["defect_density_per_plaquette"]),
                float(reference_rows[offset]["defect_density_per_plaquette"]),
            )
            for offset in offsets
        ]
        xi_diffs = [
            absolute_difference(float(test_rows[offset]["xi_grad_proxy"]), float(reference_rows[offset]["xi_grad_proxy"]))
            for offset in offsets
        ]
        print(
            f"[protocol-check] {item.label} vs {reference.label}: "
            f"dTcross={abs(item.tc_crossing_time_s - reference.tc_crossing_time_s):.6e} s, "
            f"max_abs_total={max_finite(total_abs_diffs):.6e}, "
            f"max_rel_total={max_finite(total_diffs):.6e}, "
            f"max_dRbar={max_finite(radiality_diffs):.6e}, "
            f"max_d<S>={max_finite(avg_s_diffs):.6e}, "
            f"max_dDefect2D={max_finite(defect_diffs):.6e}, "
            f"max_dXi={max_finite(xi_diffs):.6e}"
        )


def main() -> int:
    args = parse_args()
    binary = args.binary.resolve()
    output_root = args.output_root.resolve()
    configs = [path.resolve() for path in args.configs]

    if not binary.exists():
        raise FileNotFoundError(f"Binary not found: {binary}")
    for cfg in configs:
        if not cfg.exists():
            raise FileNotFoundError(f"Config not found: {cfg}")

    output_root.mkdir(parents=True, exist_ok=True)

    metrics: list[RunMetrics] = []
    logs: list[np.ndarray] = []
    for cfg in configs:
        metric, log = run_case(binary, cfg, output_root, args.tc)
        metrics.append(metric)
        logs.append(log)

    offset_rows, offsets = build_offset_rows(metrics, logs)
    write_metrics_csv(output_root / "protocol_metrics.csv", metrics)
    write_offsets_csv(output_root / "protocol_offsets.csv", offset_rows)
    if len(metrics) > 1:
        write_comparison_csv(output_root / "protocol_comparison.csv", metrics, offset_rows, offsets)
    print_summary(metrics, offset_rows, offsets)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[protocol-check] ERROR: {exc}", file=sys.stderr)
        raise