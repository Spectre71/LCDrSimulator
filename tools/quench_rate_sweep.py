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

from prune_qsr_snapshots_to_offsets import parse_csv_list as parse_prune_csv_list, prune_run_dir


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
class SweepCase:
    label: str
    ramp_iters: int
    pre_equil_iters: int
    post_ramp_hold_iters: int
    total_iters: int
    dt_s: float
    cooling_rate_K_per_s: float
    generated_config_path: str
    run_dir: str
    output_dir: str


@dataclass(frozen=True)
class SweepMetrics:
    label: str
    ramp_iters: int
    pre_equil_iters: int
    post_ramp_hold_iters: int
    total_iters: int
    dt_config_s: float
    cooling_rate_K_per_s: float
    tc_K: float
    tc_crossing_time_s: float
    final_logged_time_s: float
    fixed_dt: bool
    dt_min_s: float
    dt_max_s: float
    sample_count: int
    defect_nonzero_count: int
    first_nonzero_iter: int
    first_nonzero_time_s: float
    first_nonzero_offset_from_tc_s: float
    first_nonzero_T_K: float
    peak_defect_density: float
    peak_defect_iter: int
    peak_defect_time_s: float
    peak_defect_offset_from_tc_s: float
    eligible_plaquettes_at_peak: int
    estimated_charged_plaquettes_at_peak: int
    unique_nonzero_defect_levels: int
    defect_signal_kind: str
    final_defect_density: float
    final_avg_S: float
    final_max_S: float
    final_radiality: float
    final_xi_grad_proxy: float
    run_dir: str
    output_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare, run, and summarize fixed-dt quench-rate sweeps from a baseline config.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("prepare", "run"):
        sub = subparsers.add_parser(name, help=f"{name.capitalize()} a quench-rate sweep from a baseline config.")
        sub.add_argument("base_config", type=Path, help="Baseline config to clone and modify.")
        sub.add_argument(
            "--output-root",
            type=Path,
            default=Path("validation/quench_rate_sweep"),
            help="Directory where generated configs, run outputs, and summary CSVs are written.",
        )
        sub.add_argument("--ramp-start", type=int, default=1, help="First ramp_iters value in the sweep.")
        sub.add_argument("--ramp-stop", type=int, default=101, help="Final ramp_iters value in the sweep, inclusive.")
        sub.add_argument("--ramp-step", type=int, default=10, help="Step between ramp_iters values.")
        sub.add_argument(
            "--ramp-values",
            type=str,
            default="",
            help="Comma-separated explicit ramp_iters values. Overrides --ramp-start/--ramp-stop/--ramp-step.",
        )
        sub.add_argument(
            "--snapshot-mode",
            type=int,
            default=None,
            help="Override snapshot_mode for all generated configs. Use 0 for lightweight screening.",
        )
        sub.add_argument(
            "--snapshot-freq",
            type=int,
            default=None,
            help="Override snapshotFreq for all generated configs.",
        )
        sub.add_argument(
            "--log-freq",
            type=int,
            default=None,
            help="Override logFreq for all generated configs.",
        )
        sub.add_argument(
            "--label-prefix",
            type=str,
            default="",
            help="Optional prefix for case labels. Defaults to the base config stem.",
        )
        if name == "run":
            sub.add_argument(
                "--binary",
                type=Path,
                default=Path("QSR_cuda"),
                help="Path to the QSR CUDA binary.",
            )
            sub.add_argument(
                "--keep-going",
                action="store_true",
                help="Continue the sweep after individual run failures and summarize successful cases.",
            )
            sub.add_argument(
                "--retain-offsets",
                type=str,
                default="",
                help=(
                    "Optional comma-separated after-Tc offsets in seconds. When set, iterator snapshots are pruned "
                    "after each completed case so that only the nearest files to those offsets are retained."
                ),
            )
            sub.add_argument(
                "--retain-iter-prefixes",
                type=str,
                default="nematic_field_iter_,Qtensor_output_iter_",
                help="Comma-separated iterator snapshot filename prefixes to prune when --retain-offsets is used.",
            )

    summarize = subparsers.add_parser("summarize", help="Summarize an existing quench-rate sweep output directory.")
    summarize.add_argument(
        "--output-root",
        type=Path,
        default=Path("validation/quench_rate_sweep"),
        help="Directory containing sweep_plan.csv and per-case run outputs.",
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


def inclusive_range(start: int, stop: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("ramp-step must be positive")
    if stop < start:
        raise ValueError("ramp-stop must be greater than or equal to ramp-start")
    values: list[int] = []
    current = start
    while current <= stop:
        values.append(current)
        current += step
    return values


def parse_explicit_ramp_values(raw_values: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for chunk in raw_values.split(","):
        item = chunk.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("All explicit ramp_iters values must be positive")
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("--ramp-values did not contain any valid integers")
    return values


def resolve_ramp_values(args: argparse.Namespace) -> list[int]:
    if getattr(args, "ramp_values", "").strip():
        return parse_explicit_ramp_values(args.ramp_values)
    return inclusive_range(args.ramp_start, args.ramp_stop, args.ramp_step)


def replace_config_keys(base_text: str, replacements: dict[str, str], header_lines: list[str]) -> str:
    remaining = dict(replacements)
    rendered: list[str] = [*header_lines, ""] if header_lines else []

    for raw_line in base_text.splitlines():
        code = raw_line.split("#", 1)[0]
        if "=" not in code:
            rendered.append(raw_line)
            continue
        key, _ = code.split("=", 1)
        normalized_key = key.strip()
        if normalized_key not in remaining:
            rendered.append(raw_line)
            continue

        new_line = f"{normalized_key} = {remaining.pop(normalized_key)}"
        rendered.append(new_line)

    if remaining:
        rendered.append("")
        for key in sorted(remaining):
            rendered.append(f"{key} = {remaining[key]}")

    return "\n".join(rendered).rstrip() + "\n"


def prepare_output_root(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    cases_dir = output_root / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)


def build_case_label(prefix: str, ramp_iters: int) -> str:
    return f"{prefix}_ramp{ramp_iters:03d}"


def choose_label_prefix(base_config: Path, explicit_prefix: str) -> str:
    return explicit_prefix.strip() if explicit_prefix.strip() else base_config.stem


def prepare_cases(args: argparse.Namespace) -> list[SweepCase]:
    base_config = args.base_config.resolve()
    if not base_config.exists():
        raise FileNotFoundError(f"Config not found: {base_config}")

    cfg = parse_kv_config(base_config)
    base_text = base_config.read_text(encoding="utf-8")
    output_root = args.output_root.resolve()
    prepare_output_root(output_root)

    pre_equil_iters = cfg_int(cfg, "pre_equil_iters")
    base_ramp_iters = cfg_int(cfg, "ramp_iters")
    base_total_iters = cfg_int(cfg, "total_iters")
    post_ramp_hold_iters = base_total_iters - pre_equil_iters - base_ramp_iters
    if post_ramp_hold_iters < 0:
        raise ValueError("Baseline config has negative post-ramp hold iterations")

    dt_s = cfg_float(cfg, "dt")
    T_high_K = cfg_float(cfg, "T_high")
    T_low_K = cfg_float(cfg, "T_low")
    ramp_values = resolve_ramp_values(args)
    label_prefix = choose_label_prefix(base_config, args.label_prefix)

    cases: list[SweepCase] = []
    for ramp_iters in ramp_values:
        if ramp_iters <= 0:
            raise ValueError("All ramp_iters values must be positive")

        total_iters = pre_equil_iters + post_ramp_hold_iters + ramp_iters
        label = build_case_label(label_prefix, ramp_iters)
        run_dir = output_root / "cases" / label
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        cooling_rate = (T_low_K - T_high_K) / (float(ramp_iters) * dt_s)
        replacements = {
            "ramp_iters": str(ramp_iters),
            "total_iters": str(total_iters),
            "out_dir": "output",
            "overwrite_out_dir": "true",
        }
        if args.snapshot_mode is not None:
            replacements["snapshot_mode"] = str(args.snapshot_mode)
        if args.snapshot_freq is not None:
            replacements["snapshotFreq"] = str(args.snapshot_freq)
        if args.log_freq is not None:
            replacements["logFreq"] = str(args.log_freq)

        header_lines = [
            f"# Auto-generated from {base_config.name} by tools/quench_rate_sweep.py.",
            f"# ramp_iters = {ramp_iters}, total_iters = {total_iters}, post_ramp_hold_iters = {post_ramp_hold_iters}",
            f"# cooling_rate_K_per_s = {cooling_rate:.8e}",
        ]
        generated_config = run_dir / "generated_config.cfg"
        generated_config.write_text(
            replace_config_keys(base_text, replacements, header_lines),
            encoding="utf-8",
        )

        cases.append(
            SweepCase(
                label=label,
                ramp_iters=ramp_iters,
                pre_equil_iters=pre_equil_iters,
                post_ramp_hold_iters=post_ramp_hold_iters,
                total_iters=total_iters,
                dt_s=dt_s,
                cooling_rate_K_per_s=cooling_rate,
                generated_config_path=str(generated_config),
                run_dir=str(run_dir),
                output_dir=str(run_dir / "output"),
            )
        )

    write_plan_csv(output_root / "sweep_plan.csv", cases)
    write_prepare_summary(output_root / "prepare_summary.txt", base_config, post_ramp_hold_iters, cases)
    return cases


def write_plan_csv(path: Path, cases: list[SweepCase]) -> None:
    fieldnames = list(SweepCase.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow(case.__dict__)


def write_prepare_summary(path: Path, base_config: Path, post_ramp_hold_iters: int, cases: list[SweepCase]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"base_config={base_config}\n")
        handle.write(f"case_count={len(cases)}\n")
        handle.write(f"post_ramp_hold_iters={post_ramp_hold_iters}\n")
        if cases:
            handle.write(f"first_ramp_iters={cases[0].ramp_iters}\n")
            handle.write(f"last_ramp_iters={cases[-1].ramp_iters}\n")
            handle.write(f"dt_s={cases[0].dt_s}\n")


def load_cases_from_plan(path: Path) -> list[SweepCase]:
    if not path.exists():
        raise FileNotFoundError(f"Sweep plan not found: {path}")

    cases: list[SweepCase] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cases.append(
                SweepCase(
                    label=str(row["label"]),
                    ramp_iters=int(row["ramp_iters"]),
                    pre_equil_iters=int(row["pre_equil_iters"]),
                    post_ramp_hold_iters=int(row["post_ramp_hold_iters"]),
                    total_iters=int(row["total_iters"]),
                    dt_s=float(row["dt_s"]),
                    cooling_rate_K_per_s=float(row["cooling_rate_K_per_s"]),
                    generated_config_path=str(row["generated_config_path"]),
                    run_dir=str(row["run_dir"]),
                    output_dir=str(row["output_dir"]),
                )
            )
    return cases


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


def choose_tc(generated_config_path: Path) -> float:
    cfg = parse_kv_config(generated_config_path)
    if "Tc_KZ" in cfg:
        return float(cfg["Tc_KZ"])
    raise ValueError(f"Tc_KZ missing from generated config: {generated_config_path}")


def estimate_charged_plaquettes(defect_density: float, eligible_plaquettes: float) -> int:
    if not (np.isfinite(defect_density) and np.isfinite(eligible_plaquettes) and eligible_plaquettes > 0.0):
        return 0
    return int(round(defect_density * eligible_plaquettes))


def classify_defect_signal(nonzero_values: np.ndarray) -> tuple[int, str]:
    if nonzero_values.size == 0:
        return 0, "none"
    rounded = np.unique(np.round(nonzero_values, 12))
    if rounded.size == 1:
        return int(rounded.size), "single_plateau"
    return int(rounded.size), "variable"


def extract_metrics(case: SweepCase, log_path: Path) -> SweepMetrics:
    data = load_quench_log(log_path)
    time_s = series(data, "time_s", "time")
    temperature_K = series(data, "T_K")
    dt_s = series(data, "dt_s", "dt")
    defect_density = series(data, "defect_density_per_plaquette")
    eligible_plaquettes = series(data, "defect_plaquettes_used")
    tc_K = choose_tc(Path(case.generated_config_path))
    tc_cross = crossing_time(time_s, temperature_K, tc_K)

    nonzero_mask = np.isfinite(defect_density) & (defect_density > 0.0)
    first_nonzero_iter = -1
    first_nonzero_time_s = float("nan")
    first_nonzero_offset = float("nan")
    first_nonzero_T_K = float("nan")
    peak_defect_density = 0.0
    peak_defect_iter = -1
    peak_defect_time_s = float("nan")
    peak_defect_offset = float("nan")
    eligible_plaquettes_at_peak = 0
    estimated_charged_at_peak = 0

    if nonzero_mask.any():
        first_idx = int(np.where(nonzero_mask)[0][0])
        first_nonzero_iter = int(series(data, "iteration")[first_idx])
        first_nonzero_time_s = float(time_s[first_idx])
        first_nonzero_offset = first_nonzero_time_s - tc_cross
        first_nonzero_T_K = float(temperature_K[first_idx])

        peak_idx = int(np.nanargmax(defect_density))
        peak_defect_density = float(defect_density[peak_idx])
        peak_defect_iter = int(series(data, "iteration")[peak_idx])
        peak_defect_time_s = float(time_s[peak_idx])
        peak_defect_offset = peak_defect_time_s - tc_cross
        eligible_plaquettes_at_peak = int(round(float(eligible_plaquettes[peak_idx])))
        estimated_charged_at_peak = estimate_charged_plaquettes(
            peak_defect_density,
            float(eligible_plaquettes[peak_idx]),
        )

    unique_levels, defect_signal_kind = classify_defect_signal(defect_density[nonzero_mask])
    dt_min_s = float(np.nanmin(dt_s))
    dt_max_s = float(np.nanmax(dt_s))

    return SweepMetrics(
        label=case.label,
        ramp_iters=case.ramp_iters,
        pre_equil_iters=case.pre_equil_iters,
        post_ramp_hold_iters=case.post_ramp_hold_iters,
        total_iters=case.total_iters,
        dt_config_s=case.dt_s,
        cooling_rate_K_per_s=case.cooling_rate_K_per_s,
        tc_K=tc_K,
        tc_crossing_time_s=tc_cross,
        final_logged_time_s=float(time_s[-1]),
        fixed_dt=bool(dt_max_s - dt_min_s <= 1e-18),
        dt_min_s=dt_min_s,
        dt_max_s=dt_max_s,
        sample_count=int(time_s.size),
        defect_nonzero_count=int(np.count_nonzero(nonzero_mask)),
        first_nonzero_iter=first_nonzero_iter,
        first_nonzero_time_s=first_nonzero_time_s,
        first_nonzero_offset_from_tc_s=first_nonzero_offset,
        first_nonzero_T_K=first_nonzero_T_K,
        peak_defect_density=peak_defect_density,
        peak_defect_iter=peak_defect_iter,
        peak_defect_time_s=peak_defect_time_s,
        peak_defect_offset_from_tc_s=peak_defect_offset,
        eligible_plaquettes_at_peak=eligible_plaquettes_at_peak,
        estimated_charged_plaquettes_at_peak=estimated_charged_at_peak,
        unique_nonzero_defect_levels=unique_levels,
        defect_signal_kind=defect_signal_kind,
        final_defect_density=float(defect_density[-1]) if np.isfinite(defect_density[-1]) else float("nan"),
        final_avg_S=float(series(data, "avg_S")[-1]),
        final_max_S=float(series(data, "max_S")[-1]),
        final_radiality=float(series(data, "radiality")[-1]),
        final_xi_grad_proxy=float(series(data, "xi_grad_proxy")[-1]),
        run_dir=case.run_dir,
        output_dir=case.output_dir,
    )


def run_case(binary: Path, case: SweepCase) -> None:
    run_dir = Path(case.run_dir)
    output_dir = Path(case.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    result = subprocess.run(
        [str(binary), "--config", case.generated_config_path],
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
            f"Solver failed for {case.label} with code {result.returncode}. See {run_dir / 'solver_stderr.log'}"
        )


def write_metrics_csv(path: Path, metrics: list[SweepMetrics]) -> None:
    fieldnames = list(SweepMetrics.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric.__dict__)


def write_failures_csv(path: Path, failures: list[dict[str, str]]) -> None:
    fieldnames = ["label", "run_dir", "reason"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)


def write_summary_txt(path: Path, metrics: list[SweepMetrics], failures: list[dict[str, str]]) -> None:
    ranked = sorted(
        metrics,
        key=lambda item: (
            item.estimated_charged_plaquettes_at_peak,
            item.peak_defect_density,
            -item.first_nonzero_offset_from_tc_s if np.isfinite(item.first_nonzero_offset_from_tc_s) else float("-inf"),
        ),
        reverse=True,
    )

    with path.open("w", encoding="utf-8") as handle:
        handle.write("[overview]\n")
        handle.write(f"successful_cases={len(metrics)}\n")
        handle.write(f"failed_cases={len(failures)}\n")
        if ranked:
            best = ranked[0]
            handle.write(
                "best_peak_case="
                f"{best.label} ramp_iters={best.ramp_iters} charged_plaquettes={best.estimated_charged_plaquettes_at_peak} "
                f"peak_defect_density={best.peak_defect_density} defect_signal_kind={best.defect_signal_kind}\n"
            )

        handle.write("[cases]\n")
        for item in sorted(metrics, key=lambda metric: metric.ramp_iters):
            handle.write(
                f"{item.label}: ramp_iters={item.ramp_iters} cooling_rate_K_per_s={item.cooling_rate_K_per_s:.8e} "
                f"fixed_dt={item.fixed_dt} defect_signal_kind={item.defect_signal_kind} "
                f"peak_defect_density={item.peak_defect_density:.9g} charged_plaquettes={item.estimated_charged_plaquettes_at_peak} "
                f"first_nonzero_offset_from_tc_s={item.first_nonzero_offset_from_tc_s:.9g}\n"
            )

        if failures:
            handle.write("[failures]\n")
            for failure in failures:
                handle.write(
                    f"{failure['label']}: reason={failure['reason']} run_dir={failure['run_dir']}\n"
                )


def summarize_cases(output_root: Path, cases: list[SweepCase], failures: list[dict[str, str]] | None = None) -> list[SweepMetrics]:
    if failures is None:
        failures = []
    metrics: list[SweepMetrics] = []
    for case in cases:
        log_path = Path(case.output_dir) / "quench_log.dat"
        if not log_path.exists():
            failures.append(
                {
                    "label": case.label,
                    "run_dir": case.run_dir,
                    "reason": f"missing quench log: {log_path}",
                }
            )
            continue
        metrics.append(extract_metrics(case, log_path))

    metrics.sort(key=lambda item: item.ramp_iters)
    write_metrics_csv(output_root / "rate_sweep_metrics.csv", metrics)
    write_failures_csv(output_root / "rate_sweep_failures.csv", failures)
    write_summary_txt(output_root / "rate_sweep_summary.txt", metrics, failures)
    return metrics


def print_run_summary(metrics: list[SweepMetrics], failures: list[dict[str, str]]) -> None:
    for item in metrics:
        first_offset = item.first_nonzero_offset_from_tc_s
        first_offset_text = f"{first_offset:.6e}" if np.isfinite(first_offset) else "nan"
        print(
            f"[rate-sweep] {item.label}: ramp_iters={item.ramp_iters}, "
            f"cooling_rate_K_per_s={item.cooling_rate_K_per_s:.6e}, fixed_dt={'yes' if item.fixed_dt else 'no'}, "
            f"signal={item.defect_signal_kind}, peak_defect={item.peak_defect_density:.9g}, "
            f"charged_plaquettes={item.estimated_charged_plaquettes_at_peak}, "
            f"first_nonzero_offset_from_tc_s={first_offset_text}"
        )
    for failure in failures:
        print(
            f"[rate-sweep] FAILED {failure['label']}: {failure['reason']}",
            file=sys.stderr,
        )


def command_prepare(args: argparse.Namespace) -> int:
    cases = prepare_cases(args)
    print(
        f"[rate-sweep] prepared {len(cases)} cases in {args.output_root.resolve()} "
        f"from {args.base_config.resolve()}"
    )
    return 0


def command_run(args: argparse.Namespace) -> int:
    binary = args.binary.resolve()
    if not binary.exists():
        raise FileNotFoundError(f"Binary not found: {binary}")

    cases = prepare_cases(args)
    output_root = args.output_root.resolve()
    failures: list[dict[str, str]] = []
    retain_offsets = [float(item) for item in parse_prune_csv_list(args.retain_offsets)] if args.retain_offsets.strip() else []
    retain_iter_prefixes = parse_prune_csv_list(args.retain_iter_prefixes) if retain_offsets else []

    for case in cases:
        try:
            print(f"[rate-sweep] running {case.label} (ramp_iters={case.ramp_iters})")
            run_case(binary, case)
            if retain_offsets:
                prune_summary, _ = prune_run_dir(
                    Path(case.output_dir),
                    offsets=retain_offsets,
                    tc_K=choose_tc(Path(case.generated_config_path)),
                    iter_prefixes=retain_iter_prefixes,
                    dry_run=False,
                )
                retained_files = sum(int(row["retained_snapshot_files"]) for row in prune_summary)
                deleted_files = sum(int(row["deleted_snapshot_files"]) for row in prune_summary)
                print(
                    f"[rate-sweep] pruned {case.label}: retained_iter_snapshots={retained_files}, "
                    f"deleted_iter_snapshots={deleted_files}"
                )
        except Exception as exc:
            failures.append(
                {
                    "label": case.label,
                    "run_dir": case.run_dir,
                    "reason": str(exc),
                }
            )
            if not args.keep_going:
                summarize_cases(output_root, cases, failures)
                raise

    metrics = summarize_cases(output_root, cases, failures)
    print_run_summary(metrics, failures)
    return 0


def command_summarize(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    cases = load_cases_from_plan(output_root / "sweep_plan.csv")
    failures: list[dict[str, str]] = []
    metrics = summarize_cases(output_root, cases, failures)
    print_run_summary(metrics, failures)
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        return command_prepare(args)
    if args.command == "run":
        return command_run(args)
    if args.command == "summarize":
        return command_summarize(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[rate-sweep] ERROR: {exc}", file=sys.stderr)
        raise