#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REGION_COLORS = {
    "skin": "#64748b",
    "focus": "#b91c1c",
    "bulk": "#0f766e",
}

REGION_MARKERS = {
    "skin": "o",
    "focus": "s",
    "bulk": "^",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact figure and table for the current confined shell-band result, "
            "showing the outer skin, common focus annulus, and deeper bulk interior directly."
        )
    )
    parser.add_argument("sweep_root", type=Path, help="Sweep root containing analysis_shell_depth outputs.")
    parser.add_argument(
        "--offsets",
        type=str,
        default="5.0e-8,6.0e-8",
        help="Comma-separated offsets after Tc in seconds.",
    )
    parser.add_argument(
        "--shell-analysis-dir",
        type=Path,
        default=None,
        help="Directory containing shell_depth_after_Tc_*.csv. Defaults to <sweep_root>/analysis_shell_depth.",
    )
    parser.add_argument(
        "--band-analysis-dir",
        type=Path,
        default=None,
        help="Directory containing band-decomposition outputs. Defaults to <shell-analysis-dir>/band_decomposition.",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=ROOT / "pics" / "qsr_shell_focus_summary_2026-04-11.png",
        help="PNG output path.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=ROOT / "pics" / "qsr_shell_focus_summary_2026-04-11.pdf",
        help="PDF output path.",
    )
    parser.add_argument(
        "--table-csv",
        type=Path,
        default=None,
        help="Optional CSV table output path. Defaults to <band-analysis-dir>/shell_focus_summary_table.csv.",
    )
    parser.add_argument(
        "--table-md",
        type=Path,
        default=None,
        help="Optional Markdown table output path. Defaults to <band-analysis-dir>/shell_focus_summary_table.md.",
    )
    return parser.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated value")
    return items


def parse_inf_float(raw: str) -> float:
    text = str(raw).strip().lower()
    if text in ("inf", "+inf", "infinity", "+infinity"):
        return float("inf")
    if text in ("-inf", "-infinity"):
        return float("-inf")
    return float(text)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_focus_band(common_summary_path: Path) -> dict[str, str]:
    rows = read_csv_rows(common_summary_path)
    for row in rows:
        if row.get("qualifies_focus_band") == "True":
            return row
    if rows:
        return rows[0]
    raise ValueError(f"No rows found in {common_summary_path}")


def group_shell_detail_by_run(detail_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_run: dict[str, list[dict[str, str]]] = {}
    for row in detail_rows:
        by_run.setdefault(row["run"], []).append(row)

    runs: list[dict[str, object]] = []
    for run_name, rows in by_run.items():
        rows_sorted = sorted(rows, key=lambda row: int(row["depth_bin_index"]))
        runs.append(
            {
                "run": run_name,
                "rate_K_per_s": float(rows_sorted[0]["rate_K_per_s"]),
                "t_ramp_s": float(rows_sorted[0]["t_ramp_s"]),
                "selected_avg_S": float(rows_sorted[0]["selected_avg_S"]),
                "total_used_plaquettes": float(rows_sorted[0]["total_used_plaquettes"]),
                "total_defect_count": float(rows_sorted[0]["total_defect_count"]),
                "total_density": float(rows_sorted[0]["total_density"]),
                "bins": rows_sorted,
            }
        )
    runs.sort(key=lambda row: float(row["rate_K_per_s"]))
    return runs


def aggregate_region(run: dict[str, object], start_index: int, end_index: int) -> dict[str, float]:
    bins = run["bins"]
    assert isinstance(bins, list)
    selected = bins[start_index : end_index + 1]
    used = float(sum(int(row["depth_bin_used_plaquettes"]) for row in selected))
    defect = float(sum(int(row["depth_bin_defect_count"]) for row in selected))
    total_used = float(run["total_used_plaquettes"])
    total_defect = float(run["total_defect_count"])
    total_density = float(run["total_density"])
    density = defect / used if used > 0.0 else float("nan")
    return {
        "used": used,
        "defect": defect,
        "density": density,
        "defect_share": defect / total_defect if total_defect > 0.0 else float("nan"),
        "support_share": used / total_used if total_used > 0.0 else float("nan"),
        "enrichment": density / total_density if density > 0.0 and total_density > 0.0 else float("nan"),
    }


def build_per_run_region_rows(
    runs: list[dict[str, object]],
    *,
    focus_start: int,
    focus_end: int,
    target_after_tc_s: float,
) -> list[dict[str, float | str | int]]:
    first_bins = runs[0]["bins"]
    assert isinstance(first_bins, list)
    max_index = len(first_bins) - 1
    region_specs: list[tuple[str, int, int]] = []
    if focus_start > 0:
        region_specs.append(("skin", 0, focus_start - 1))
    region_specs.append(("focus", focus_start, focus_end))
    if focus_end < max_index:
        region_specs.append(("bulk", focus_end + 1, max_index))

    out_rows: list[dict[str, float | str | int]] = []
    for run in runs:
        bins = run["bins"]
        assert isinstance(bins, list)
        for region_name, start_index, end_index in region_specs:
            agg = aggregate_region(run, start_index, end_index)
            lo = float(bins[start_index]["depth_bin_lo"])
            hi = parse_inf_float(bins[end_index]["depth_bin_hi"])
            label = f"[{lo:g},inf)" if np.isinf(hi) else f"[{lo:g},{hi:g})"
            out_rows.append(
                {
                    "target_after_Tc_s": float(target_after_tc_s),
                    "run": str(run["run"]),
                    "rate_K_per_s": float(run["rate_K_per_s"]),
                    "t_ramp_s": float(run["t_ramp_s"]),
                    "selected_avg_S": float(run["selected_avg_S"]),
                    "region": region_name,
                    "region_start_index": int(start_index),
                    "region_end_index": int(end_index),
                    "region_label": label,
                    "density": float(agg["density"]),
                    "defect_share": float(agg["defect_share"]),
                    "support_share": float(agg["support_share"]),
                    "enrichment": float(agg["enrichment"]),
                }
            )
    return out_rows


def load_region_summary(region_summary_path: Path) -> list[dict[str, str]]:
    return read_csv_rows(region_summary_path)


def load_moment_summary(moment_summary_path: Path) -> list[dict[str, str]]:
    return read_csv_rows(moment_summary_path)


def fmt(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def write_markdown_table(
    path: Path,
    *,
    focus_band_row: dict[str, str],
    region_summary_rows: list[dict[str, str]],
    moment_summary_rows: list[dict[str, str]],
) -> None:
    runner_rows = read_csv_rows(path.parent / "shell_band_common_summary.csv")[:3]
    lines: list[str] = []
    lines.append("# Shell Focus Summary")
    lines.append("")
    lines.append("## Chosen Focus Band")
    lines.append("")
    lines.append("| Focus band | Mean support | Mean density slope vs rate | Mean defect-share slope vs rate |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        "| "
        f"{focus_band_row['band_label']} | {fmt(float(focus_band_row['mean_support_fraction_across_offsets']))} | "
        f"{fmt(float(focus_band_row['mean_density_slope_vs_rate']))} | {fmt(float(focus_band_row['mean_defect_share_slope_vs_rate']))} |"
    )
    lines.append("")
    lines.append("## Top Common Bands")
    lines.append("")
    lines.append("| Band | Mean support | Mean density slope | Min density slope | Min corr | Min ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in runner_rows:
        lines.append(
            "| "
            f"{row['band_label']} | {fmt(float(row['mean_support_fraction_across_offsets']))} | "
            f"{fmt(float(row['mean_density_slope_vs_rate']))} | {fmt(float(row['min_density_slope_vs_rate']))} | "
            f"{fmt(float(row['min_density_corr_vs_rate']))} | {fmt(float(row['min_density_ratio_max_over_min']))} |"
        )
    lines.append("")
    lines.append("## Skin / Focus / Bulk")
    lines.append("")
    lines.append("| After Tc [ns] | Region | Depth range | Support | Density slope vs rate | Density corr | Density ratio | Defect-share slope vs rate | Defect-share corr |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in sorted(region_summary_rows, key=lambda item: (float(item["target_after_Tc_s"]), item["region"])):
        lines.append(
            "| "
            f"{float(row['target_after_Tc_s']) * 1e9:.0f} | {row['region']} | {row['region_label']} | "
            f"{fmt(float(row['mean_support_fraction']))} | {fmt(float(row['density_slope_vs_rate']))} | "
            f"{fmt(float(row['density_corr_vs_rate']))} | {fmt(float(row['density_ratio_max_over_min']))} | "
            f"{fmt(float(row['defect_share_slope_vs_rate']))} | {fmt(float(row['defect_share_corr_vs_rate']))} |"
        )
    lines.append("")
    lines.append("## Depth Moments")
    lines.append("")
    lines.append("| After Tc [ns] | Defect-weighted mean depth slope vs rate | Defect minus support mean depth slope vs rate |")
    lines.append("|---|---:|---:|")
    for row in sorted(moment_summary_rows, key=lambda item: float(item["target_after_Tc_s"])):
        lines.append(
            "| "
            f"{float(row['target_after_Tc_s']) * 1e9:.0f} | {fmt(float(row['defect_mean_depth_slope_vs_rate']))} | "
            f"{fmt(float(row['defect_minus_support_depth_slope_vs_rate']))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_table(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_region_matrix(
    region_summary_rows: list[dict[str, str]],
    *,
    offsets_sorted: list[float],
    regions: list[str],
    metric_name: str,
) -> np.ndarray:
    matrix = np.full((len(offsets_sorted), len(regions)), np.nan, dtype=float)
    for row_idx, offset in enumerate(offsets_sorted):
        subset = [row for row in region_summary_rows if abs(float(row["target_after_Tc_s"]) - offset) <= 1e-18]
        for col_idx, region in enumerate(regions):
            match = next((row for row in subset if row["region"] == region), None)
            if match is not None:
                matrix[row_idx, col_idx] = float(match[metric_name])
    return matrix


def annotate_heatmap(ax: plt.Axes, matrix: np.ndarray) -> None:
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if not np.isfinite(value):
                continue
            text_color = "white" if abs(value) > 0.22 else "#0f172a"
            ax.text(col_idx, row_idx, f"{value:+.3f}", ha="center", va="center", fontsize=9, color=text_color)


def plot_summary(
    *,
    png_path: Path,
    pdf_path: Path,
    focus_band_label: str,
    per_run_rows: list[dict[str, float | str | int]],
    region_summary_rows: list[dict[str, str]],
    moment_summary_rows: list[dict[str, str]],
) -> None:
    offsets_sorted = sorted({float(row["target_after_Tc_s"]) for row in per_run_rows})
    n_offsets = len(offsets_sorted)
    ncols = min(3, max(1, n_offsets))
    n_top_rows = int(math.ceil(n_offsets / ncols))

    fig = plt.figure(figsize=(5.4 * ncols, 3.8 * n_top_rows + 5.1))
    grid = fig.add_gridspec(n_top_rows + 1, ncols, height_ratios=[1.0] * n_top_rows + [0.92], hspace=0.42, wspace=0.28)

    for idx, offset in enumerate(offsets_sorted):
        ax = fig.add_subplot(grid[idx // ncols, idx % ncols])
        subset = [row for row in per_run_rows if abs(float(row["target_after_Tc_s"]) - offset) <= 1e-18]
        for region in ("skin", "focus", "bulk"):
            reg_rows = sorted([row for row in subset if row["region"] == region], key=lambda row: float(row["rate_K_per_s"]))
            if not reg_rows:
                continue
            rate = np.array([float(row["rate_K_per_s"]) for row in reg_rows], dtype=float)
            dens = np.array([float(row["density"]) for row in reg_rows], dtype=float)
            dens_rel = dens / dens[0] if dens[0] > 0.0 else dens
            ax.plot(
                rate,
                dens_rel,
                color=REGION_COLORS[region],
                marker=REGION_MARKERS[region],
                ms=4.8,
                lw=1.8,
                label=f"{region} {reg_rows[0]['region_label']} (x{dens_rel[-1]:.2f})",
            )
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.24)
        ax.set_title(f"After Tc = {offset * 1e9:.0f} ns")
        ax.set_xlabel("Cooling rate [K/s]")
        ax.set_ylabel("Density relative to slowest rate")
        ax.legend(loc="best", fontsize=8.3)

    for idx in range(n_offsets, n_top_rows * ncols):
        ax = fig.add_subplot(grid[idx // ncols, idx % ncols])
        ax.axis("off")

    bottom = grid[n_top_rows, :].subgridspec(1, 2, wspace=0.24)
    ax_density = fig.add_subplot(bottom[0, 0])
    ax_share = fig.add_subplot(bottom[0, 1])

    regions = ["skin", "focus", "bulk"]
    density_matrix = build_region_matrix(
        region_summary_rows,
        offsets_sorted=offsets_sorted,
        regions=regions,
        metric_name="density_slope_vs_rate",
    )
    share_matrix = build_region_matrix(
        region_summary_rows,
        offsets_sorted=offsets_sorted,
        regions=regions,
        metric_name="defect_share_slope_vs_rate",
    )

    density_norm = Normalize(
        vmin=float(np.nanmin(density_matrix)) if np.isfinite(density_matrix).any() else 0.0,
        vmax=float(np.nanmax(density_matrix)) if np.isfinite(density_matrix).any() else 1.0,
    )
    share_abs = float(np.nanmax(np.abs(share_matrix))) if np.isfinite(share_matrix).any() else 1.0
    share_norm = TwoSlopeNorm(vmin=-share_abs, vcenter=0.0, vmax=share_abs)

    im_density = ax_density.imshow(density_matrix, cmap="YlOrRd", norm=density_norm, aspect="auto")
    im_share = ax_share.imshow(share_matrix, cmap="RdBu_r", norm=share_norm, aspect="auto")

    for ax, title, matrix in (
        (ax_density, "Density slope vs rate by region", density_matrix),
        (ax_share, "Defect-share slope vs rate by region", share_matrix),
    ):
        ax.set_title(title)
        ax.set_xticks(np.arange(len(regions)), regions)
        ax.set_yticks(np.arange(len(offsets_sorted)), [f"{offset * 1e9:.0f} ns" for offset in offsets_sorted])
        annotate_heatmap(ax, matrix)
        ax.tick_params(axis="x", rotation=0)

    cbar_density = fig.colorbar(im_density, ax=ax_density, fraction=0.046, pad=0.03)
    cbar_density.set_label("Slope vs rate")
    cbar_share = fig.colorbar(im_share, ax=ax_share, fraction=0.046, pad=0.03)
    cbar_share.set_label("Slope vs rate")

    depth_lines = []
    for row in sorted(moment_summary_rows, key=lambda item: float(item["target_after_Tc_s"])):
        depth_lines.append(
            f"{float(row['target_after_Tc_s']) * 1e9:.0f} ns: defect-depth slope {float(row['defect_mean_depth_slope_vs_rate']):+.3f}, "
            f"defect-support slope {float(row['defect_minus_support_depth_slope_vs_rate']):+.3f}"
        )
    fig.text(
        0.06,
        0.012,
        "Current best confined bridge observable: full-z XY defect density restricted to the "
        f"{focus_band_label} sub-surface annulus. Bulk stays weakly positive in density but loses defect share with rate.\n"
        + "\n".join(depth_lines),
        ha="left",
        va="bottom",
        fontsize=9.4,
        wrap=True,
    )
    fig.suptitle("Confined shell-focus summary", fontsize=15)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.14)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    sweep_root = args.sweep_root.resolve()
    shell_analysis_dir = args.shell_analysis_dir.resolve() if args.shell_analysis_dir else (sweep_root / "analysis_shell_depth").resolve()
    band_analysis_dir = args.band_analysis_dir.resolve() if args.band_analysis_dir else (shell_analysis_dir / "band_decomposition").resolve()

    table_csv = args.table_csv.resolve() if args.table_csv else (band_analysis_dir / "shell_focus_summary_table.csv")
    table_md = args.table_md.resolve() if args.table_md else (band_analysis_dir / "shell_focus_summary_table.md")

    offsets = [float(item) for item in parse_csv_list(args.offsets)]
    focus_band_row = load_focus_band(band_analysis_dir / "shell_band_common_summary.csv")
    focus_start = int(focus_band_row["band_start_index"])
    focus_end = int(focus_band_row["band_end_index"])
    focus_band_label = str(focus_band_row["band_label"])

    per_run_rows: list[dict[str, float | str | int]] = []
    for offset in offsets:
        detail_path = shell_analysis_dir / f"shell_depth_after_Tc_{offset:.2e}.csv"
        detail_rows = read_csv_rows(detail_path)
        runs = group_shell_detail_by_run(detail_rows)
        per_run_rows.extend(
            build_per_run_region_rows(
                runs,
                focus_start=focus_start,
                focus_end=focus_end,
                target_after_tc_s=float(offset),
            )
        )

    region_summary_rows = load_region_summary(band_analysis_dir / "shell_focus_region_summary.csv")
    moment_summary_rows = load_moment_summary(band_analysis_dir / "shell_depth_moment_summary.csv")

    plot_summary(
        png_path=args.png.resolve(),
        pdf_path=args.pdf.resolve(),
        focus_band_label=focus_band_label,
        per_run_rows=per_run_rows,
        region_summary_rows=region_summary_rows,
        moment_summary_rows=moment_summary_rows,
    )

    table_rows: list[dict[str, float | str]] = []
    for row in region_summary_rows:
        table_rows.append(
            {
                "focus_band_label": focus_band_label,
                "target_after_Tc_ns": float(row["target_after_Tc_s"]) * 1e9,
                "region": str(row["region"]),
                "depth_range": str(row["region_label"]),
                "mean_support_fraction": float(row["mean_support_fraction"]),
                "density_slope_vs_rate": float(row["density_slope_vs_rate"]),
                "density_corr_vs_rate": float(row["density_corr_vs_rate"]),
                "density_ratio_max_over_min": float(row["density_ratio_max_over_min"]),
                "defect_share_slope_vs_rate": float(row["defect_share_slope_vs_rate"]),
                "defect_share_corr_vs_rate": float(row["defect_share_corr_vs_rate"]),
                "defect_share_ratio_max_over_min": float(row["defect_share_ratio_max_over_min"]),
                "enrichment_slope_vs_rate": float(row["enrichment_slope_vs_rate"]),
            }
        )
    write_csv_table(table_csv, table_rows)
    write_markdown_table(
        table_md,
        focus_band_row=focus_band_row,
        region_summary_rows=region_summary_rows,
        moment_summary_rows=moment_summary_rows,
    )

    print(f"focus_band={focus_band_label}")
    print(f"png={args.png.resolve()}")
    print(f"pdf={args.pdf.resolve()}")
    print(f"table_csv={table_csv}")
    print(f"table_md={table_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
