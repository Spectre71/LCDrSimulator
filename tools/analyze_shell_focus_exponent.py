#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REGION_COLORS = {
    "skin": "#64748b",
    "focus": "#b91c1c",
    "bulk": "#0f766e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the confined-bridge exponent from the validated shell-focus annulus by combining "
            "per-window log-log fits with a pooled common-slope model that allows different late-window amplitudes."
        )
    )
    parser.add_argument("sweep_root", type=Path, help="Sweep root containing analysis_shell_depth outputs.")
    parser.add_argument(
        "--offsets",
        type=str,
        default="4.5e-8,5.0e-8,5.5e-8,6.0e-8,6.5e-8",
        help="Comma-separated offsets after Tc in seconds.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Output directory for exponent CSVs and markdown. Defaults to <sweep_root>/analysis_shell_depth/focus_exponent.",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        help="Optional PNG figure path. Defaults to pics/<sweep_root_name>_focus_exponent.png.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Optional PDF figure path. Defaults to pics/<sweep_root_name>_focus_exponent.pdf.",
    )
    parser.add_argument(
        "--focus-start-index",
        type=int,
        default=None,
        help="Optional start shell-bin index for a fixed focus band override.",
    )
    parser.add_argument(
        "--focus-end-index",
        type=int,
        default=None,
        help="Optional end shell-bin index for a fixed focus band override.",
    )
    parser.add_argument(
        "--focus-band-label",
        type=str,
        default=None,
        help="Optional label for a fixed focus band override. If omitted, derive it from shell-bin edges.",
    )
    return parser.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated value")
    return items


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_focus_band(common_summary_path: Path) -> dict[str, str]:
    rows = read_csv_rows(common_summary_path)
    for row in rows:
        if row.get("qualifies_focus_band") == "True":
            return row
    if rows:
        return rows[0]
    raise ValueError(f"No rows found in {common_summary_path}")


def format_depth_value(value: float) -> str:
    if np.isinf(value):
        return "inf"
    if abs(value - round(value)) <= 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def format_band_label(lo: float, hi: float) -> str:
    return f"[{format_depth_value(lo)},{format_depth_value(hi)})"


def group_shell_detail_by_run(detail_rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in detail_rows:
        grouped.setdefault(row["run"], []).append(row)
    runs = [sorted(rows, key=lambda row: int(row["depth_bin_index"])) for rows in grouped.values()]
    runs.sort(key=lambda rows: float(rows[0]["rate_K_per_s"]))
    return runs


def aggregate_region(run_rows: list[dict[str, str]], start_index: int, end_index: int) -> dict[str, float]:
    selected = run_rows[start_index : end_index + 1]
    used = float(sum(int(row["depth_bin_used_plaquettes"]) for row in selected))
    defect = float(sum(int(row["depth_bin_defect_count"]) for row in selected))
    total_used = float(run_rows[0]["total_used_plaquettes"])
    total_defect = float(run_rows[0]["total_defect_count"])
    total_density = float(run_rows[0]["total_density"])
    density = defect / used if used > 0.0 else float("nan")
    return {
        "used": used,
        "defect": defect,
        "density": density,
        "defect_share": defect / total_defect if total_defect > 0.0 else float("nan"),
        "support_share": used / total_used if total_used > 0.0 else float("nan"),
        "enrichment": density / total_density if density > 0.0 and total_density > 0.0 else float("nan"),
    }


def fit_loglog_with_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if int(np.count_nonzero(mask)) < 2:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "corr": float("nan"),
            "rmse_log": float("nan"),
        }
    log_x = np.log(x[mask])
    log_y = np.log(y[mask])
    slope, intercept = np.polyfit(log_x, log_y, deg=1)
    pred = intercept + slope * log_x
    resid = log_y - pred
    rmse = float(np.sqrt(np.mean(resid * resid)))
    corr = float(np.corrcoef(log_x, log_y)[0, 1]) if log_x.size >= 2 else float("nan")
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "corr": corr,
        "rmse_log": rmse,
    }


def fit_common_log_slope_with_window_intercepts(
    x: np.ndarray,
    y: np.ndarray,
    window_ids: np.ndarray,
    *,
    ordered_window_ids: list[int],
) -> dict[str, float | dict[int, float]]:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if int(np.count_nonzero(mask)) < 3:
        return {
            "slope": float("nan"),
            "standard_error": float("nan"),
            "ci95_norm": float("nan"),
            "rmse_log": float("nan"),
            "intercepts": {},
            "n_points": 0,
            "dof": 0,
        }

    x_fit = np.log(x[mask])
    y_fit = np.log(y[mask])
    window_fit = window_ids[mask]
    dummies = np.zeros((x_fit.size, len(ordered_window_ids)), dtype=float)
    col_map = {window_id: col for col, window_id in enumerate(ordered_window_ids)}
    for row_idx, window_id in enumerate(window_fit):
        dummies[row_idx, col_map[int(window_id)]] = 1.0

    design = np.column_stack([dummies, x_fit])
    coef, *_ = np.linalg.lstsq(design, y_fit, rcond=None)
    pred = design @ coef
    resid = y_fit - pred
    dof = int(y_fit.size - design.shape[1])
    rmse = float(np.sqrt(np.mean(resid * resid)))
    if dof > 0:
        sigma2 = float((resid @ resid) / dof)
        cov = sigma2 * np.linalg.inv(design.T @ design)
        standard_error = float(np.sqrt(cov[-1, -1]))
    else:
        standard_error = float("nan")
    intercepts = {window_id: float(coef[col_map[window_id]]) for window_id in ordered_window_ids}
    slope = float(coef[-1])
    return {
        "slope": slope,
        "standard_error": standard_error,
        "ci95_norm": float(1.96 * standard_error) if np.isfinite(standard_error) else float("nan"),
        "rmse_log": rmse,
        "intercepts": intercepts,
        "n_points": int(y_fit.size),
        "dof": dof,
    }


def fmt(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def write_markdown_summary(
    path: Path,
    *,
    focus_band_label: str,
    detail_rows: list[dict[str, float | str]],
    window_rows: list[dict[str, float | str]],
    common_rows: list[dict[str, float | str]],
) -> None:
    focus_density = next(row for row in common_rows if row["region"] == "focus" and row["observable"] == "density")
    focus_share = next(row for row in common_rows if row["region"] == "focus" and row["observable"] == "defect_share")
    region_density_rows = [row for row in common_rows if row["observable"] == "density"]
    comparison_regions = [str(row["region"]) for row in region_density_rows if str(row["region"]) != "focus"]

    lines: list[str] = []
    lines.append("# Focus Exponent Summary")
    lines.append("")
    lines.append("## Recommended Exponent")
    lines.append("")
    lines.append("| Observable | Region | Preferred power law | Common exponent | 95% fit CI | Window std | Window half-range |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    lines.append(
        "| "
        f"density | {focus_band_label} | $\\rho \\propto \\tau_Q^{{-\\alpha}}$ | {fmt(float(focus_density['alpha_rate_common']))} | "
        f"{fmt(float(focus_density['alpha_rate_ci95_norm']))} | {fmt(float(focus_density['window_std_alpha_rate']))} | {fmt(float(focus_density['window_half_range_alpha_rate']))} |"
    )
    lines.append(
        "| "
        f"defect share | {focus_band_label} | $f \\propto \\tau_Q^{{-\\alpha}}$ | {fmt(float(focus_share['alpha_rate_common']))} | "
        f"{fmt(float(focus_share['alpha_rate_ci95_norm']))} | {fmt(float(focus_share['window_std_alpha_rate']))} | {fmt(float(focus_share['window_half_range_alpha_rate']))} |"
    )
    lines.append("")
    lines.append("## Per-window Focus Fits")
    lines.append("")
    lines.append("| After Tc [ns] | Focus density exponent $\\alpha$ | Focus density corr | Focus defect-share exponent | Focus defect-share corr |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in [r for r in window_rows if r["region"] == "focus"]:
        lines.append(
            "| "
            f"{float(row['offset_ns']):.0f} | {fmt(float(row['alpha_rate_density']))} | {fmt(float(row['corr_rate_density']))} | "
            f"{fmt(float(row['alpha_rate_defect_share']))} | {fmt(float(row['corr_rate_defect_share']))} |"
        )
    lines.append("")
    lines.append("## Region Comparison")
    lines.append("")
    lines.append("| Region | Common density exponent $\\alpha$ | 95% fit CI | Window half-range |")
    lines.append("|---|---:|---:|---:|")
    for row in region_density_rows:
        lines.append(
            "| "
            f"{row['region']} | {fmt(float(row['alpha_rate_common']))} | {fmt(float(row['alpha_rate_ci95_norm']))} | "
            f"{fmt(float(row['window_half_range_alpha_rate']))} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    interpretation = (
        f"The current best confined-bridge exponent comes from the validated focus band {focus_band_label}. "
        f"Across real 45-65 ns late windows, the density exponent stays near {fmt(float(focus_density['alpha_rate_common']), 3)}. "
    )
    if comparison_regions:
        interpretation += (
            f"Comparison regions available in this decomposition are {', '.join(comparison_regions)}; "
            "the exponent should still be quoted for the focus annulus rather than for the whole droplet interior."
        )
    else:
        interpretation += "This decomposition contains only the focus annulus, so the quoted exponent is the focus-band result by construction."
    lines.append(interpretation)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(
    *,
    png_path: Path,
    pdf_path: Path,
    sweep_name: str,
    focus_band_label: str,
    detail_rows: list[dict[str, float | str]],
    window_rows: list[dict[str, float | str]],
    common_rows: list[dict[str, float | str]],
) -> None:
    focus_density = next(row for row in common_rows if row["region"] == "focus" and row["observable"] == "density")
    focus_share = next(row for row in common_rows if row["region"] == "focus" and row["observable"] == "defect_share")
    region_density_rows = [row for row in common_rows if row["observable"] == "density"]
    comparison_regions = [str(row["region"]) for row in region_density_rows if str(row["region"]) != "focus"]
    offsets_ns = sorted({float(row["offset_ns"]) for row in detail_rows if row["region"] == "focus"})
    colors = plt.cm.plasma(np.linspace(0.12, 0.88, len(offsets_ns)))
    color_map = {offset: color for offset, color in zip(offsets_ns, colors, strict=True)}

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9.2,
        }
    )

    fig = plt.figure(figsize=(14.0, 9.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0], hspace=0.30, wspace=0.24)
    ax_main = fig.add_subplot(grid[:, 0])
    ax_window = fig.add_subplot(grid[0, 1])
    ax_regions = fig.add_subplot(grid[1, 1])

    for offset_ns in offsets_ns:
        subset = [row for row in detail_rows if row["region"] == "focus" and abs(float(row["offset_ns"]) - offset_ns) <= 1e-12]
        subset = sorted(subset, key=lambda row: float(row["tau_Q_s"]))
        tau_q = np.array([float(row["tau_Q_s"]) for row in subset], dtype=float)
        density = np.array([float(row["density"]) for row in subset], dtype=float)
        ax_main.loglog(
            tau_q,
            density,
            linestyle="none",
            marker="o",
            ms=6.0,
            color=color_map[offset_ns],
            label=f"{offset_ns:.0f} ns (alpha={next(r for r in window_rows if r['region'] == 'focus' and abs(float(r['offset_ns']) - offset_ns) <= 1e-12)['alpha_rate_density']:.3f})",
        )
        intercept_tau = float(next(r for r in common_rows if r["region"] == "focus" and r["observable"] == "density")[f"window_{int(offset_ns)}ns_intercept_tau"])
        xfit = np.geomspace(float(np.min(tau_q)), float(np.max(tau_q)), 100)
        yfit = np.exp(intercept_tau) * np.power(xfit, float(focus_density["slope_tauQ_common"]))
        ax_main.loglog(xfit, yfit, color=color_map[offset_ns], linewidth=1.6, alpha=0.85)

    ax_main.set_title(f"Focus-annulus log-log fits: {focus_band_label}")
    ax_main.set_xlabel("Quench time tau_Q [s]")
    ax_main.set_ylabel("Focus-band defect density")
    ax_main.grid(True, which="both", alpha=0.22)
    ax_main.legend(loc="lower left", frameon=False, ncol=2)
    ax_main.text(
        0.03,
        0.97,
        (
            "Preferred readout: rho_focus ~ tau_Q^(-alpha)" + "\n"
            f"alpha = {float(focus_density['alpha_rate_common']):.3f}"
            f" +/- {float(focus_density['window_half_range_alpha_rate']):.3f} (window half-range)" + "\n"
            f"95% fit CI ~= +/- {float(focus_density['alpha_rate_ci95_norm']):.3f}" + "\n"
            f"focus defect-share alpha = {float(focus_share['alpha_rate_common']):.3f}"
        ),
        transform=ax_main.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.93, "edgecolor": "#cbd5e1"},
    )

    focus_window_rows = [row for row in window_rows if row["region"] == "focus"]
    x_ns = np.array([float(row["offset_ns"]) for row in focus_window_rows], dtype=float)
    alpha_density = np.array([float(row["alpha_rate_density"]) for row in focus_window_rows], dtype=float)
    alpha_share = np.array([float(row["alpha_rate_defect_share"]) for row in focus_window_rows], dtype=float)
    ax_window.plot(x_ns, alpha_density, color="#b91c1c", marker="o", ms=5.5, lw=1.8, label="focus density alpha")
    ax_window.plot(x_ns, alpha_share, color="#1d4ed8", marker="s", ms=5.0, lw=1.6, label="focus defect-share alpha")
    ax_window.axhline(float(focus_density["alpha_rate_common"]), color="#7f1d1d", linestyle="--", lw=1.4)
    ax_window.fill_between(
        x_ns,
        float(focus_density["alpha_rate_common"]) - float(focus_density["window_half_range_alpha_rate"]),
        float(focus_density["alpha_rate_common"]) + float(focus_density["window_half_range_alpha_rate"]),
        color="#fecaca",
        alpha=0.55,
        label="focus density half-range",
    )
    ax_window.axhline(float(focus_share["alpha_rate_common"]), color="#1e3a8a", linestyle=":", lw=1.4)
    ax_window.set_title("Late-window exponent stability")
    ax_window.set_xlabel("Measurement offset after Tc [ns]")
    ax_window.set_ylabel("Positive exponent alpha in y ~ tau_Q^(-alpha)")
    ax_window.grid(True, alpha=0.22)
    ax_window.legend(loc="best", frameon=False)

    region_names = [str(row["region"]) for row in region_density_rows]
    alpha_region = np.array([float(row["alpha_rate_common"]) for row in region_density_rows], dtype=float)
    ci_region = np.array([float(row["alpha_rate_ci95_norm"]) for row in region_density_rows], dtype=float)
    bar_colors = [REGION_COLORS[name] for name in region_names]
    xpos = np.arange(len(region_names), dtype=float)
    ax_regions.bar(xpos, alpha_region, color=bar_colors, alpha=0.88)
    ax_regions.errorbar(xpos, alpha_region, yerr=ci_region, fmt="none", ecolor="#0f172a", elinewidth=1.2, capsize=4)
    ax_regions.set_xticks(xpos, region_names)
    ax_regions.set_ylabel("Common density exponent alpha")
    ax_regions.set_title("Common late-window density exponent by region")
    ax_regions.grid(True, axis="y", alpha=0.22)
    for x, row in zip(xpos, region_density_rows, strict=True):
        ax_regions.text(
            x,
            float(row["alpha_rate_common"]) + float(row["alpha_rate_ci95_norm"]) + 0.015,
            f"{float(row['alpha_rate_common']):.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle(f"Confined focus-band exponent: {sweep_name}", fontsize=16, y=0.98)
    footer = (
        f"Current best exponent readout: focus band {focus_band_label}. "
        f"The pooled late-window fit gives alpha ~= {float(focus_density['alpha_rate_common']):.3f} for defect density and alpha ~= {float(focus_share['alpha_rate_common']):.3f} for defect share. "
    )
    if comparison_regions:
        footer += (
            f"Comparison regions present here are {', '.join(comparison_regions)}. "
            "The confined-bridge exponent should still be quoted from the focus annulus rather than from the surrounding droplet support."
        )
    else:
        footer += "This decomposition contains only the focus annulus, so the reported exponent is the focus-band readout by construction."
    fig.text(
        0.06,
        0.02,
        footer,
        ha="left",
        va="bottom",
        fontsize=10.1,
        wrap=True,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.15, wspace=0.24, hspace=0.30)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    sweep_root = args.sweep_root.resolve()
    analysis_dir = args.analysis_dir.resolve() if args.analysis_dir else (sweep_root / "analysis_shell_depth" / "focus_exponent").resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    sweep_name = sweep_root.name
    png_path = args.png.resolve() if args.png else (ROOT / "pics" / f"{sweep_name}_focus_exponent.png")
    pdf_path = args.pdf.resolve() if args.pdf else (ROOT / "pics" / f"{sweep_name}_focus_exponent.pdf")

    offsets = [float(item) for item in parse_csv_list(args.offsets)]
    offset_ids = [int(round(offset * 1e9)) for offset in offsets]

    override_start = args.focus_start_index
    override_end = args.focus_end_index
    if (override_start is None) ^ (override_end is None):
        raise ValueError("Provide both --focus-start-index and --focus-end-index together for a fixed-band override")

    if override_start is None:
        focus_band = load_focus_band(sweep_root / "analysis_shell_depth" / "band_decomposition" / "shell_band_common_summary.csv")
        focus_start = int(focus_band["band_start_index"])
        focus_end = int(focus_band["band_end_index"])
        focus_label = str(focus_band["band_label"])
    else:
        focus_start = int(override_start)
        focus_end = int(override_end)
        focus_label = args.focus_band_label.strip() if args.focus_band_label else ""

    region_defs: dict[str, tuple[int, int]] | None = None
    detail_rows: list[dict[str, float | str]] = []
    grouped_points: dict[tuple[str, str], list[dict[str, float]]] = {}
    window_summary_rows: list[dict[str, float | str]] = []
    available_regions: list[str] = []

    for offset, offset_ns in zip(offsets, offset_ids, strict=True):
        detail_path = sweep_root / "analysis_shell_depth" / f"shell_depth_after_Tc_{offset:.2e}.csv"
        run_groups = group_shell_detail_by_run(read_csv_rows(detail_path))
        if region_defs is None:
            max_index = len(run_groups[0]) - 1
            if focus_start < 0 or focus_end < focus_start or focus_end > max_index:
                raise ValueError(
                    f"Requested focus band indices [{focus_start},{focus_end}] are invalid for max shell index {max_index}"
                )
            if not focus_label:
                lo = float(run_groups[0][focus_start]["depth_bin_lo"])
                hi = float(run_groups[0][focus_end]["depth_bin_hi"])
                focus_label = format_band_label(lo, hi)
            region_defs = {
                "skin": (0, focus_start - 1),
                "focus": (focus_start, focus_end),
                "bulk": (focus_end + 1, max_index),
            }
            available_regions = [name for name, (start_index, end_index) in region_defs.items() if start_index <= end_index]

        for run_rows in run_groups:
            run_name = str(run_rows[0]["run"])
            rate = float(run_rows[0]["rate_K_per_s"])
            tau_q = float(run_rows[0]["t_ramp_s"])
            for region_name, (start_index, end_index) in region_defs.items():
                if start_index > end_index:
                    continue
                agg = aggregate_region(run_rows, start_index, end_index)
                detail_row = {
                    "offset_s": float(offset),
                    "offset_ns": float(offset_ns),
                    "run": run_name,
                    "region": region_name,
                    "rate_K_per_s": rate,
                    "tau_Q_s": tau_q,
                    "density": float(agg["density"]),
                    "defect_share": float(agg["defect_share"]),
                    "support_share": float(agg["support_share"]),
                    "enrichment": float(agg["enrichment"]),
                }
                detail_rows.append(detail_row)
                for observable in ("density", "defect_share"):
                    grouped_points.setdefault((region_name, observable), []).append(
                        {
                            "offset_ns": float(offset_ns),
                            "window_id": int(offset_ns),
                            "rate_K_per_s": rate,
                            "tau_Q_s": tau_q,
                            "value": float(detail_row[observable]),
                        }
                    )

        for region_name in available_regions:
            subset = [row for row in detail_rows if row["region"] == region_name and abs(float(row["offset_ns"]) - float(offset_ns)) <= 1e-12]
            rate = np.array([float(row["rate_K_per_s"]) for row in subset], dtype=float)
            tau_q = np.array([float(row["tau_Q_s"]) for row in subset], dtype=float)
            dens = np.array([float(row["density"]) for row in subset], dtype=float)
            share = np.array([float(row["defect_share"]) for row in subset], dtype=float)
            fit_rate_d = fit_loglog_with_stats(rate, dens)
            fit_tau_d = fit_loglog_with_stats(tau_q, dens)
            fit_rate_s = fit_loglog_with_stats(rate, share)
            fit_tau_s = fit_loglog_with_stats(tau_q, share)
            window_summary_rows.append(
                {
                    "offset_s": float(offset),
                    "offset_ns": float(offset_ns),
                    "region": region_name,
                    "alpha_rate_density": float(fit_rate_d["slope"]),
                    "slope_tauQ_density": float(fit_tau_d["slope"]),
                    "corr_rate_density": float(fit_rate_d["corr"]),
                    "corr_tauQ_density": float(fit_tau_d["corr"]),
                    "rmse_log_density": float(fit_rate_d["rmse_log"]),
                    "alpha_rate_defect_share": float(fit_rate_s["slope"]),
                    "slope_tauQ_defect_share": float(fit_tau_s["slope"]),
                    "corr_rate_defect_share": float(fit_rate_s["corr"]),
                    "corr_tauQ_defect_share": float(fit_tau_s["corr"]),
                    "rmse_log_defect_share": float(fit_rate_s["rmse_log"]),
                }
            )

    common_rows: list[dict[str, float | str]] = []
    for region_name in available_regions:
        subset_density = list(grouped_points.get((region_name, "density"), []))
        if not subset_density:
            continue
        window_alpha_density = np.array(
            [float(row["alpha_rate_density"]) for row in window_summary_rows if row["region"] == region_name],
            dtype=float,
        )
        fit_density_rate = fit_common_log_slope_with_window_intercepts(
            np.array([float(row["rate_K_per_s"]) for row in subset_density], dtype=float),
            np.array([float(row["value"]) for row in subset_density], dtype=float),
            np.array([int(row["window_id"]) for row in subset_density], dtype=int),
            ordered_window_ids=offset_ids,
        )
        fit_density_tau = fit_common_log_slope_with_window_intercepts(
            np.array([float(row["tau_Q_s"]) for row in subset_density], dtype=float),
            np.array([float(row["value"]) for row in subset_density], dtype=float),
            np.array([int(row["window_id"]) for row in subset_density], dtype=int),
            ordered_window_ids=offset_ids,
        )
        row_density: dict[str, float | str] = {
            "region": region_name,
            "observable": "density",
            "alpha_rate_common": float(fit_density_rate["slope"]),
            "slope_tauQ_common": float(fit_density_tau["slope"]),
            "alpha_rate_standard_error": float(fit_density_rate["standard_error"]),
            "alpha_rate_ci95_norm": float(fit_density_rate["ci95_norm"]),
            "rmse_log_common": float(fit_density_rate["rmse_log"]),
            "n_points": int(fit_density_rate["n_points"]),
            "dof": int(fit_density_rate["dof"]),
            "window_mean_alpha_rate": float(np.mean(window_alpha_density)),
            "window_std_alpha_rate": float(np.std(window_alpha_density, ddof=1)),
            "window_min_alpha_rate": float(np.min(window_alpha_density)),
            "window_max_alpha_rate": float(np.max(window_alpha_density)),
            "window_half_range_alpha_rate": float((np.max(window_alpha_density) - np.min(window_alpha_density)) / 2.0),
        }
        for window_id in offset_ids:
            intercept_tau = float(fit_density_tau["intercepts"][window_id]) if window_id in fit_density_tau["intercepts"] else float("nan")
            row_density[f"window_{window_id}ns_intercept_tau"] = intercept_tau
        common_rows.append(row_density)

        if region_name != "focus":
            continue
        subset_share = list(grouped_points.get((region_name, "defect_share"), []))
        if not subset_share:
            continue
        window_alpha_share = np.array(
            [float(row["alpha_rate_defect_share"]) for row in window_summary_rows if row["region"] == region_name],
            dtype=float,
        )
        fit_share_rate = fit_common_log_slope_with_window_intercepts(
            np.array([float(row["rate_K_per_s"]) for row in subset_share], dtype=float),
            np.array([float(row["value"]) for row in subset_share], dtype=float),
            np.array([int(row["window_id"]) for row in subset_share], dtype=int),
            ordered_window_ids=offset_ids,
        )
        fit_share_tau = fit_common_log_slope_with_window_intercepts(
            np.array([float(row["tau_Q_s"]) for row in subset_share], dtype=float),
            np.array([float(row["value"]) for row in subset_share], dtype=float),
            np.array([int(row["window_id"]) for row in subset_share], dtype=int),
            ordered_window_ids=offset_ids,
        )
        row_share: dict[str, float | str] = {
            "region": region_name,
            "observable": "defect_share",
            "alpha_rate_common": float(fit_share_rate["slope"]),
            "slope_tauQ_common": float(fit_share_tau["slope"]),
            "alpha_rate_standard_error": float(fit_share_rate["standard_error"]),
            "alpha_rate_ci95_norm": float(fit_share_rate["ci95_norm"]),
            "rmse_log_common": float(fit_share_rate["rmse_log"]),
            "n_points": int(fit_share_rate["n_points"]),
            "dof": int(fit_share_rate["dof"]),
            "window_mean_alpha_rate": float(np.mean(window_alpha_share)),
            "window_std_alpha_rate": float(np.std(window_alpha_share, ddof=1)),
            "window_min_alpha_rate": float(np.min(window_alpha_share)),
            "window_max_alpha_rate": float(np.max(window_alpha_share)),
            "window_half_range_alpha_rate": float((np.max(window_alpha_share) - np.min(window_alpha_share)) / 2.0),
        }
        common_rows.append(row_share)

    detail_csv = analysis_dir / "focus_exponent_detail.csv"
    window_csv = analysis_dir / "focus_exponent_window_fits.csv"
    common_csv = analysis_dir / "focus_exponent_common_fit.csv"
    summary_md = analysis_dir / "focus_exponent_summary.md"
    write_csv(detail_csv, detail_rows)
    write_csv(window_csv, window_summary_rows)
    write_csv(common_csv, common_rows)
    write_markdown_summary(
        summary_md,
        focus_band_label=focus_label,
        detail_rows=detail_rows,
        window_rows=window_summary_rows,
        common_rows=common_rows,
    )
    plot_summary(
        png_path=png_path,
        pdf_path=pdf_path,
        sweep_name=sweep_name,
        focus_band_label=focus_label,
        detail_rows=detail_rows,
        window_rows=window_summary_rows,
        common_rows=common_rows,
    )

    focus_density = next(row for row in common_rows if row["region"] == "focus" and row["observable"] == "density")
    print(f"focus_band={focus_label}")
    print(f"alpha_rate_common={float(focus_density['alpha_rate_common']):.6g}")
    print(f"alpha_rate_ci95_norm={float(focus_density['alpha_rate_ci95_norm']):.6g}")
    print(f"window_half_range_alpha_rate={float(focus_density['window_half_range_alpha_rate']):.6g}")
    print(f"detail_csv={detail_csv}")
    print(f"window_csv={window_csv}")
    print(f"common_csv={common_csv}")
    print(f"summary_md={summary_md}")
    print(f"png={png_path}")
    print(f"pdf={pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
