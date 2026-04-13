#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


@dataclass(frozen=True)
class RunCase:
    label: str
    generated_config_path: Path
    output_dir: Path
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze rate-sweep outputs from the periodic XY KZM proving ground.",
    )
    parser.add_argument("sweep_root", type=Path, help="Sweep root containing cases/*/output/quench_log.dat.")
    parser.add_argument(
        "--measure-mode",
        type=str,
        default="avg_amp",
        choices=("final", "fixed", "avg_amp"),
        help="How to select the measurement point from each run log.",
    )
    parser.add_argument(
        "--after-tc-s",
        type=float,
        default=0.0,
        help="For measure-mode=fixed, select the first sample at or after this offset from Tc.",
    )
    parser.add_argument(
        "--avg-amp-targets",
        type=str,
        default="0.25,0.35",
        help="Comma-separated avg amplitude milestones for measure-mode=avg_amp.",
    )
    parser.add_argument(
        "--extra-after-target-s",
        type=float,
        default=0.0,
        help="Optional delay added after the avg amplitude milestone time.",
    )
    parser.add_argument(
        "--observables",
        type=str,
        default="vortex_line_density,xi_grad_proxy",
        help="Comma-separated observables to fit.",
    )
    parser.add_argument(
        "--tc",
        type=float,
        default=float("nan"),
        help="Override Tc used for crossing-time selection. Defaults to Tc_KZ or Tc from each config.",
    )
    parser.add_argument(
        "--nu-critical",
        type=float,
        default=0.6717,
        help="Static critical exponent nu used for the expected KZM slope.",
    )
    parser.add_argument(
        "--z-dynamic",
        type=float,
        default=2.0,
        help="Dynamic critical exponent z used for the expected KZM slope.",
    )
    parser.add_argument(
        "--defect-codimension",
        type=float,
        default=2.0,
        help="Codimension D-d for defect-density observables. Use 2 for vortex lines in 3D.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Directory where per-target detail CSVs and the summary CSV are written.",
    )
    return parser.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated value")
    return items


def parse_float_list(raw: str) -> list[float]:
    return [float(item) for item in parse_csv_list(raw)]


def parse_kv_config(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cfg[key.strip()] = value.strip()
    return cfg


def list_cases(sweep_root: Path) -> list[RunCase]:
    cases_root = sweep_root / "cases"
    if not cases_root.exists():
        raise FileNotFoundError(f"Expected sweep root with cases/* under {sweep_root}")

    cases: list[RunCase] = []
    for case_dir in sorted(cases_root.glob("*")):
        generated_config_path = case_dir / "generated_config.cfg"
        output_dir = case_dir / "output"
        log_path = output_dir / "quench_log.dat"
        if generated_config_path.exists() and log_path.exists():
            cases.append(
                RunCase(
                    label=case_dir.name,
                    generated_config_path=generated_config_path,
                    output_dir=output_dir,
                    log_path=log_path,
                )
            )
    if not cases:
        raise FileNotFoundError(f"No cases with generated_config.cfg and output/quench_log.dat found in {sweep_root}")
    return cases


def load_log(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if getattr(data, "size", 0) == 0:
        raise ValueError(f"Quench log is empty: {path}")
    return np.array(data, ndmin=1)


def series(data: np.ndarray, *names: str) -> np.ndarray:
    fields = set(data.dtype.names or ())
    for name in names:
        if name in fields:
            return np.atleast_1d(data[name]).astype(float)
    raise ValueError(f"Missing required columns {names} in log with fields {sorted(fields)}")


def choose_tc(cfg: dict[str, str], tc_override: float) -> float:
    if np.isfinite(tc_override):
        return float(tc_override)
    if "Tc_KZ" in cfg:
        return float(cfg["Tc_KZ"])
    if "Tc" in cfg:
        return float(cfg["Tc"])
    raise ValueError("Tc not provided and config has neither Tc_KZ nor Tc")


def crossing_time(time_s: np.ndarray, temperature_K: np.ndarray, tc_K: float) -> float:
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
            return float(t0 + (tc_K - T0) * (t1 - t0) / (T1 - T0))
    raise ValueError(f"Could not find a Tc crossing for Tc={tc_K}")


def first_index_at_or_after(time_s: np.ndarray, target_time_s: float) -> int:
    idx = int(np.searchsorted(time_s, target_time_s, side="left"))
    return min(idx, time_s.size - 1)


def select_index(
    data: np.ndarray,
    tc_cross_s: float,
    measure_mode: str,
    after_tc_s: float,
    avg_amp_target: float,
    extra_after_target_s: float,
) -> int:
    time_s = series(data, "time_s", "time")
    avg_amp = series(data, "avg_S")

    if measure_mode == "final":
        return int(time_s.size - 1)

    if measure_mode == "fixed":
        return first_index_at_or_after(time_s, tc_cross_s + after_tc_s)

    if measure_mode == "avg_amp":
        mask = (time_s >= tc_cross_s - 1e-18) & np.isfinite(avg_amp) & (avg_amp >= avg_amp_target)
        if not np.any(mask):
            raise ValueError(f"avg_S never reaches target {avg_amp_target}")
        first_idx = int(np.where(mask)[0][0])
        target_time_s = float(time_s[first_idx] + extra_after_target_s)
        return first_index_at_or_after(time_s, target_time_s)

    raise ValueError(f"Unsupported measure_mode: {measure_mode}")


def fit_powerlaw_loglog(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if int(np.count_nonzero(mask)) < 2:
        return float("nan"), float("nan")
    logx = np.log(x[mask])
    logy = np.log(y[mask])
    slope, _intercept = np.polyfit(logx, logy, 1)
    corr = float(np.corrcoef(logx, logy)[0, 1])
    return float(slope), corr


def expected_slopes(observable: str, nu_critical: float, z_dynamic: float, defect_codimension: float) -> tuple[float, float]:
    xi_exponent_tau = nu_critical / (1.0 + z_dynamic * nu_critical)
    lower = observable.lower()
    if "xi" in lower:
        return xi_exponent_tau, -xi_exponent_tau
    if "vortex" in lower or "defect" in lower:
        defect_exponent_tau = -defect_codimension * xi_exponent_tau
        return defect_exponent_tau, -defect_exponent_tau
    return float("nan"), float("nan")


def cooling_rate_abs(cfg: dict[str, str]) -> float:
    T_high = float(cfg["T_high"])
    T_low = float(cfg["T_low"])
    ramp_iters = int(float(cfg["ramp_iters"]))
    dt_s = float(cfg["dt"])
    return abs((T_low - T_high) / (float(ramp_iters) * dt_s))


def quench_time_s(cfg: dict[str, str]) -> float:
    return int(float(cfg["ramp_iters"])) * float(cfg["dt"])


def detail_filename(observable: str, measure_mode: str, target_desc: str) -> str:
    safe_observable = observable.replace("/", "_")
    safe_target = target_desc.replace(".", "p").replace("-", "m")
    return f"{safe_observable}_{measure_mode}_{safe_target}.csv"


def write_detail_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    sweep_root = args.sweep_root.resolve()
    analysis_dir = args.analysis_dir.resolve() if args.analysis_dir else (sweep_root / "analysis_kzm_pg").resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    observables = parse_csv_list(args.observables)
    targets = [float("nan")] if args.measure_mode in ("final", "fixed") else parse_float_list(args.avg_amp_targets)
    cases = list_cases(sweep_root)

    summary_rows: list[dict[str, float | str]] = []

    for observable in observables:
        for target in targets:
            detail_rows: list[dict[str, float | str]] = []
            rate_values: list[float] = []
            tau_values: list[float] = []
            obs_values: list[float] = []
            selected_times: list[float] = []

            for case in cases:
                cfg = parse_kv_config(case.generated_config_path)
                log = load_log(case.log_path)
                time_s = series(log, "time_s", "time")
                temperature_K = series(log, "T_K")
                tc_K = choose_tc(cfg, float(args.tc))
                tc_cross_s = crossing_time(time_s, temperature_K, tc_K)
                idx = select_index(
                    log,
                    tc_cross_s,
                    str(args.measure_mode),
                    float(args.after_tc_s),
                    float(target),
                    float(args.extra_after_target_s),
                )

                fields = set(log.dtype.names or ())
                if observable not in fields:
                    raise ValueError(f"Observable {observable} not found in {case.log_path}")

                observable_value = float(series(log, observable)[idx])
                rate_abs = cooling_rate_abs(cfg)
                tau_q_s = quench_time_s(cfg)
                after_tc_actual_s = float(time_s[idx] - tc_cross_s)
                avg_amp_value = float(series(log, "avg_S")[idx])

                detail_rows.append(
                    {
                        "run": case.label,
                        "observable": observable,
                        "measure_mode": str(args.measure_mode),
                        "target_after_Tc_s": float(args.after_tc_s) if args.measure_mode == "fixed" else float("nan"),
                        "target_avg_amp": float(target) if args.measure_mode == "avg_amp" else float("nan"),
                        "extra_after_target_s": float(args.extra_after_target_s) if args.measure_mode == "avg_amp" else 0.0,
                        "rate_abs": rate_abs,
                        "tau_Q_s": tau_q_s,
                        "observable_value": observable_value,
                        "avg_amp": avg_amp_value,
                        "selected_time_s": float(time_s[idx]),
                        "after_Tc_actual_s": after_tc_actual_s,
                        "tc_cross_s": tc_cross_s,
                        "log_path": str(case.log_path.as_posix()),
                    }
                )

                rate_values.append(rate_abs)
                tau_values.append(tau_q_s)
                obs_values.append(observable_value)
                selected_times.append(after_tc_actual_s)

            rate_arr = np.array(rate_values, dtype=float)
            tau_arr = np.array(tau_values, dtype=float)
            obs_arr = np.array(obs_values, dtype=float)
            after_tc_arr = np.array(selected_times, dtype=float)

            slope_tau, corr_tau = fit_powerlaw_loglog(tau_arr, obs_arr)
            slope_rate, corr_rate = fit_powerlaw_loglog(rate_arr, obs_arr)
            expected_tau, expected_rate = expected_slopes(
                observable,
                float(args.nu_critical),
                float(args.z_dynamic),
                float(args.defect_codimension),
            )

            finite_positive = obs_arr[np.isfinite(obs_arr) & (obs_arr > 0.0)]
            obs_min = float(np.min(finite_positive)) if finite_positive.size else float("nan")
            obs_max = float(np.max(finite_positive)) if finite_positive.size else float("nan")
            obs_ratio = float(obs_max / obs_min) if np.isfinite(obs_min) and obs_min > 0.0 and np.isfinite(obs_max) else float("nan")

            target_desc = (
                f"afterTc_{args.after_tc_s:.3e}s" if args.measure_mode == "fixed"
                else "final" if args.measure_mode == "final"
                else f"avgAmp_{target:.3f}"
            )
            detail_path = analysis_dir / detail_filename(observable, str(args.measure_mode), target_desc)
            write_detail_csv(detail_path, detail_rows)

            summary_rows.append(
                {
                    "observable": observable,
                    "measure_mode": str(args.measure_mode),
                    "target_after_Tc_s": float(args.after_tc_s) if args.measure_mode == "fixed" else float("nan"),
                    "target_avg_amp": float(target) if args.measure_mode == "avg_amp" else float("nan"),
                    "extra_after_target_s": float(args.extra_after_target_s) if args.measure_mode == "avg_amp" else 0.0,
                    "actual_after_Tc_min_s": float(np.nanmin(after_tc_arr)),
                    "actual_after_Tc_max_s": float(np.nanmax(after_tc_arr)),
                    "slope_vs_tauQ": slope_tau,
                    "slope_vs_rate": slope_rate,
                    "corr_vs_tauQ": corr_tau,
                    "corr_vs_rate": corr_rate,
                    "expected_slope_vs_tauQ": expected_tau,
                    "expected_slope_vs_rate": expected_rate,
                    "observable_min": obs_min,
                    "observable_max": obs_max,
                    "observable_ratio_max_over_min": obs_ratio,
                    "detail_csv": str(detail_path.as_posix()),
                }
            )

    summary_path = analysis_dir / "kzm_prooving_ground_summary.csv"
    fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    for row in summary_rows:
        if row["measure_mode"] == "fixed":
            target_desc = f"after_Tc={float(row['target_after_Tc_s']):.3e}s"
        elif row["measure_mode"] == "avg_amp":
            target_desc = f"avg_amp={float(row['target_avg_amp']):.3f}"
        else:
            target_desc = "final"
        print(
            f"[{row['observable']}] {target_desc} | "
            f"slope_vs_tauQ={float(row['slope_vs_tauQ']):.6g} expected_tauQ={float(row['expected_slope_vs_tauQ']):.6g} | "
            f"slope_vs_rate={float(row['slope_vs_rate']):.6g} expected_rate={float(row['expected_slope_vs_rate']):.6g}"
        )
    print(f"summary_csv={summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[kzm-proving-ground] ERROR: {exc}", file=sys.stderr)
        raise