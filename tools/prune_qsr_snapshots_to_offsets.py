#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import QSRvis as qv


DEFAULT_OFFSETS = "4.5e-8,5.0e-8,5.5e-8,6.0e-8,6.5e-8"
DEFAULT_ITER_PREFIXES = "nematic_field_iter_,Qtensor_output_iter_"
FINAL_NAME_BY_PREFIX = {
    "nematic_field_iter_": "nematic_field_final.dat",
    "Qtensor_output_iter_": "Qtensor_output_final.dat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trim retained KZ snapshots down to the nearest files for a chosen set of after-Tc windows. "
            "Accepts either a single run output directory or a sweep root containing cases/*/output."
        )
    )
    parser.add_argument("target", type=Path, help="Run output directory, case directory, or sweep root to prune.")
    parser.add_argument(
        "--offsets",
        type=str,
        default=DEFAULT_OFFSETS,
        help="Comma-separated after-Tc offsets in seconds to retain.",
    )
    parser.add_argument(
        "--tc",
        type=float,
        default=310.2,
        help="Tc used to locate the crossing time from quench_log.dat.",
    )
    parser.add_argument(
        "--iter-prefixes",
        type=str,
        default=DEFAULT_ITER_PREFIXES,
        help="Comma-separated iterator snapshot filename prefixes to prune.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Directory where pruning detail/summary CSVs are written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be kept/deleted without removing files.",
    )
    return parser.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated value")
    return items


def write_csv(path: Path, rows: list[dict[str, float | int | str | bool]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary_rows: list[dict[str, float | int | str | bool]]) -> None:
    lines: list[str] = []
    lines.append("# Snapshot Pruning Summary")
    lines.append("")
    lines.append("| Run | Prefix | Before | After | Deleted | Retained files |")
    lines.append("|---|---|---:|---:|---:|---|")
    for row in summary_rows:
        lines.append(
            "| "
            f"{row['run']} | {row['iter_prefix']} | {row['snapshot_files_before']} | {row['retained_snapshot_files']} | {row['deleted_snapshot_files']} | {row['retained_files']} |"
        )
    lines.append("")
    lines.append("Only iterator snapshots nearest the requested after-Tc windows are kept. Final-state files are not touched.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_iter(path: Path) -> int:
    match = re.search(r"(\d+)", path.name)
    return int(match.group(1)) if match else -1


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


def resolve_output_dirs(target: Path) -> list[Path]:
    resolved = target.resolve()
    if (resolved / "quench_log.dat").exists():
        return [resolved]
    if (resolved / "output" / "quench_log.dat").exists():
        return [(resolved / "output").resolve()]

    run_dirs = [path.resolve() for path in sorted((resolved / "cases").glob("*/output")) if (path / "quench_log.dat").exists()]
    if run_dirs:
        return run_dirs

    raise FileNotFoundError(f"Could not resolve any run output directories from {resolved}")


def default_analysis_dir(target: Path) -> Path:
    resolved = target.resolve()
    if (resolved / "cases").exists():
        return resolved / "analysis_snapshot_pruning"
    if resolved.name == "output":
        return resolved.parent / "analysis_snapshot_pruning"
    return resolved / "analysis_snapshot_pruning"


def prune_run_dir(
    run_dir: Path,
    *,
    offsets: list[float],
    tc_K: float,
    iter_prefixes: list[str],
    dry_run: bool = False,
) -> tuple[list[dict[str, float | int | str | bool]], list[dict[str, float | int | str | bool]]]:
    data, _ = qv.load_quench_log(str(run_dir))
    it = np.atleast_1d(data["iteration"]).astype(float)
    t = np.atleast_1d(data["time_s"]).astype(float)
    temperature = np.atleast_1d(data["T_K"]).astype(float)
    tc_cross = crossing_time(t, temperature, float(tc_K))

    run_label = run_dir.parent.name if run_dir.name == "output" else run_dir.name
    summary_rows: list[dict[str, float | int | str | bool]] = []
    detail_rows: list[dict[str, float | int | str | bool]] = []

    for iter_prefix in iter_prefixes:
        snapshot_files = sorted(run_dir.glob(f"{iter_prefix}*.dat"))
        if not snapshot_files:
            continue

        final_name = FINAL_NAME_BY_PREFIX.get(iter_prefix, "")
        retained_names: set[str] = set()
        before_count = len(snapshot_files)

        for offset in offsets:
            target_time_s = tc_cross + float(offset)
            selected_path = Path(
                qv._select_snapshot_by_time(
                    str(run_dir),
                    target_time_s,
                    it,
                    t,
                    iter_prefix=iter_prefix,
                    final_name=final_name,
                )
            )
            if selected_path.name not in {path.name for path in snapshot_files}:
                continue

            selected_iter = extract_iter(selected_path)
            selected_time_s = float(qv._nearest_from_log(it, t, selected_iter)) if selected_iter >= 0 else float("nan")
            actual_after_tc_s = selected_time_s - tc_cross if np.isfinite(selected_time_s) else float("nan")
            delta_s = actual_after_tc_s - float(offset) if np.isfinite(actual_after_tc_s) else float("nan")
            retained_names.add(selected_path.name)
            detail_rows.append(
                {
                    "run": run_label,
                    "iter_prefix": iter_prefix,
                    "requested_after_Tc_s": float(offset),
                    "requested_after_Tc_ns": float(offset) * 1e9,
                    "selected_file": selected_path.name,
                    "selected_iter": int(selected_iter),
                    "selected_time_s": float(selected_time_s),
                    "actual_after_Tc_s": float(actual_after_tc_s),
                    "actual_after_Tc_ns": float(actual_after_tc_s) * 1e9 if np.isfinite(actual_after_tc_s) else float("nan"),
                    "delta_s": float(delta_s),
                    "delta_ns": float(delta_s) * 1e9 if np.isfinite(delta_s) else float("nan"),
                }
            )

        deleted_count = 0
        for path in snapshot_files:
            if path.name in retained_names:
                continue
            deleted_count += 1
            if not dry_run:
                path.unlink()

        summary_rows.append(
            {
                "run": run_label,
                "iter_prefix": iter_prefix,
                "snapshot_files_before": int(before_count),
                "retained_snapshot_files": int(len(retained_names)),
                "deleted_snapshot_files": int(deleted_count),
                "dry_run": bool(dry_run),
                "retained_files": "; ".join(sorted(retained_names)),
            }
        )

    return summary_rows, detail_rows


def main() -> int:
    args = parse_args()
    offsets = [float(item) for item in parse_csv_list(args.offsets)]
    iter_prefixes = parse_csv_list(args.iter_prefixes)
    run_dirs = resolve_output_dirs(args.target)
    analysis_dir = args.analysis_dir.resolve() if args.analysis_dir else default_analysis_dir(args.target)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | int | str | bool]] = []
    detail_rows: list[dict[str, float | int | str | bool]] = []
    for run_dir in run_dirs:
        run_summary, run_detail = prune_run_dir(
            run_dir,
            offsets=offsets,
            tc_K=float(args.tc),
            iter_prefixes=iter_prefixes,
            dry_run=bool(args.dry_run),
        )
        summary_rows.extend(run_summary)
        detail_rows.extend(run_detail)

    summary_path = analysis_dir / "snapshot_pruning_summary.csv"
    detail_path = analysis_dir / "snapshot_pruning_detail.csv"
    markdown_path = analysis_dir / "snapshot_pruning_summary.md"
    write_csv(summary_path, summary_rows)
    write_csv(detail_path, detail_rows)
    write_markdown(markdown_path, summary_rows)

    print(f"run_count={len(run_dirs)}")
    print(f"summary_csv={summary_path}")
    print(f"detail_csv={detail_path}")
    print(f"summary_md={markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())