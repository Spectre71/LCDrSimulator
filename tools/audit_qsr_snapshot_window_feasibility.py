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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether requested post-Tc windows can be measured honestly from the snapshots currently "
            "present in a confined rate-sweep output, or whether they alias onto the same retained files."
        )
    )
    parser.add_argument("sweep_root", type=Path, help="Sweep root containing cases/*/output and rate_sweep_metrics.csv.")
    parser.add_argument(
        "--offsets",
        type=str,
        default="4.5e-8,5.0e-8,5.5e-8,6.0e-8,6.5e-8",
        help="Comma-separated offsets after Tc in seconds to audit.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Directory where feasibility CSV/Markdown outputs are written.",
    )
    return parser.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated value")
    return items


def extract_iter(path: Path) -> int:
    match = re.search(r"(\d+)", path.name)
    return int(match.group(1)) if match else -1


def read_rate_metrics(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    metrics: dict[str, dict[str, float]] = {}
    for row in rows:
        metrics[row["label"]] = {
            "tc_crossing_time_s": float(row["tc_crossing_time_s"]),
            "rate_K_per_s": abs(float(row["cooling_rate_K_per_s"])),
            "t_ramp_s": float(row["ramp_iters"]) * float(row["dt_config_s"]),
        }
    return metrics


def write_csv(path: Path, rows: list[dict[str, float | int | str | bool]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary_rows: list[dict[str, float | int | str | bool]]) -> None:
    lines: list[str] = []
    lines.append("# Snapshot Window Feasibility")
    lines.append("")
    lines.append("| Run | Requested offsets [ns] | Distinct retained files | Fully distinct? | Alias groups |")
    lines.append("|---|---|---:|---|---|")
    for row in summary_rows:
        lines.append(
            "| "
            f"{row['run']} | {row['requested_offsets_ns']} | {row['distinct_file_count']} | {row['all_offsets_distinct']} | {row['alias_groups']} |"
        )
    lines.append("")
    lines.append(
        "Interpretation: if multiple requested windows map to the same retained snapshot inside a run, the temporal comparison is aliased and cannot be treated as a real stability test."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    sweep_root = args.sweep_root.resolve()
    analysis_dir = args.analysis_dir.resolve() if args.analysis_dir else (sweep_root / "analysis_snapshot_feasibility").resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    offsets = [float(item) for item in parse_csv_list(args.offsets)]
    metrics = read_rate_metrics(sweep_root / "rate_sweep_metrics.csv")

    detail_rows: list[dict[str, float | int | str | bool]] = []
    summary_rows: list[dict[str, float | int | str | bool]] = []

    for case_dir in sorted((sweep_root / "cases").glob("*")):
        run_dir = case_dir / "output"
        if not (run_dir / "quench_log.dat").exists():
            continue
        label = case_dir.name
        meta = metrics[label]
        tc_cross = float(meta["tc_crossing_time_s"])
        log_data, _ = qv.load_quench_log(str(run_dir))
        it = np.atleast_1d(log_data["iteration"]).astype(float)
        t = np.atleast_1d(log_data["time_s"]).astype(float)

        aliases: dict[str, list[float]] = {}
        for offset in offsets:
            target_time_s = tc_cross + float(offset)
            field_path = Path(qv._select_snapshot_by_time(str(run_dir), target_time_s, it, t))
            selected_iter = extract_iter(field_path)
            selected_time_s = float(qv._nearest_from_log(it, t, selected_iter)) if selected_iter >= 0 else float("nan")
            actual_after_tc_s = selected_time_s - tc_cross if np.isfinite(selected_time_s) else float("nan")
            delta_s = actual_after_tc_s - float(offset) if np.isfinite(actual_after_tc_s) else float("nan")
            aliases.setdefault(field_path.name, []).append(float(offset))
            detail_rows.append(
                {
                    "run": label,
                    "rate_K_per_s": float(meta["rate_K_per_s"]),
                    "t_ramp_s": float(meta["t_ramp_s"]),
                    "requested_after_Tc_s": float(offset),
                    "requested_after_Tc_ns": float(offset) * 1e9,
                    "selected_file": field_path.name,
                    "selected_iter": int(selected_iter),
                    "selected_time_s": float(selected_time_s),
                    "actual_after_Tc_s": float(actual_after_tc_s),
                    "actual_after_Tc_ns": float(actual_after_tc_s) * 1e9 if np.isfinite(actual_after_tc_s) else float("nan"),
                    "delta_s": float(delta_s),
                    "delta_ns": float(delta_s) * 1e9 if np.isfinite(delta_s) else float("nan"),
                }
            )

        alias_groups = "; ".join(
            f"{file_name}: {', '.join(f'{offset * 1e9:.0f}' for offset in group)} ns"
            for file_name, group in aliases.items()
        )
        summary_rows.append(
            {
                "run": label,
                "requested_offsets_ns": ", ".join(f"{offset * 1e9:.0f}" for offset in offsets),
                "distinct_file_count": int(len(aliases)),
                "all_offsets_distinct": bool(len(aliases) == len(offsets)),
                "alias_groups": alias_groups,
            }
        )

    detail_path = analysis_dir / "snapshot_window_feasibility_detail.csv"
    summary_path = analysis_dir / "snapshot_window_feasibility_summary.csv"
    markdown_path = analysis_dir / "snapshot_window_feasibility_summary.md"
    write_csv(detail_path, detail_rows)
    write_csv(summary_path, summary_rows)
    write_markdown(markdown_path, summary_rows)

    blocked = any(not bool(row["all_offsets_distinct"]) for row in summary_rows)
    print(f"blocked={blocked}")
    print(f"detail_csv={detail_path}")
    print(f"summary_csv={summary_path}")
    print(f"summary_md={markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())