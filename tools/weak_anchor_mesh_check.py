#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Geometry:
    Nx: int
    Ny: int
    Nz: int
    dx: float
    dy: float
    dz: float
    center_x: float
    center_y: float
    center_z: float
    radius_x: float
    radius_y: float
    radius_z: float
    min_spacing: float
    min_radius: float
    shell_half_thickness: float


@dataclass(frozen=True)
class RunMetrics:
    label: str
    config_path: str
    run_dir: str
    Nx: int
    Ny: int
    Nz: int
    dx: float
    dy: float
    dz: float
    final_iteration: int
    final_time_s: float
    final_free_energy: float
    final_radiality: float
    shell_order_mask_mean: float
    shell_order_mask_std: float
    shell_order_inner_mean: float
    shell_order_inner_std: float
    shell_mask_points: int
    shell_inner_points: int
    defect_density_midplane: float
    defect_plaquettes_used: int
    shell_voxels_solver: int
    shell_volume_solver_m3: float
    shell_surface_area_estimate_m2: float
    shell_nominal_thickness_m: float
    shell_effective_thickness_m: float
    shell_thickness_ratio_to_nominal: float
    weak_anchoring_scale_J_m3: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a two-grid weak-anchoring validation and compare final metrics.",
    )
    parser.add_argument(
        "configs",
        nargs="+",
        type=Path,
        help="Config files to run, ordered from coarse to fine.",
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
        default=Path("validation/weak_anchor_mesh"),
        help="Directory where per-run outputs and comparison CSVs are written.",
    )
    parser.add_argument(
        "--defect-s-threshold",
        type=float,
        default=0.1,
        help="S-threshold for the 2D mid-plane defect proxy.",
    )
    parser.add_argument(
        "--charge-cutoff",
        type=float,
        default=0.25,
        help="Charge magnitude cutoff for counting defect plaquettes.",
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


def make_geometry(cfg: dict[str, str]) -> Geometry:
    Nx = cfg_int(cfg, "Nx")
    Ny = cfg_int(cfg, "Ny")
    Nz = cfg_int(cfg, "Nz")
    dx = cfg_float(cfg, "dx")
    dy = cfg_float(cfg, "dy")
    dz = cfg_float(cfg, "dz")
    center_x = 0.5 * (float(Nx) - 1.0) * dx
    center_y = 0.5 * (float(Ny) - 1.0) * dy
    center_z = 0.5 * (float(Nz) - 1.0) * dz
    radius_index = 0.5 * (float(min(Nx, Ny, Nz)) - 1.0)
    radius_x = max(radius_index * dx, 0.5 * dx)
    radius_y = max(radius_index * dy, 0.5 * dy)
    radius_z = max(radius_index * dz, 0.5 * dz)
    min_spacing = min(dx, dy, dz)
    min_radius = min(radius_x, radius_y, radius_z)
    return Geometry(
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        dx=dx,
        dy=dy,
        dz=dz,
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
        radius_x=radius_x,
        radius_y=radius_y,
        radius_z=radius_z,
        min_spacing=min_spacing,
        min_radius=min_radius,
        shell_half_thickness=min_spacing,
    )


def signed_distance(geom: Geometry, i: int, j: int, k: int) -> float:
    x = i * geom.dx - geom.center_x
    y = j * geom.dy - geom.center_y
    z = k * geom.dz - geom.center_z
    center_radius = math.sqrt(x * x + y * y + z * z)
    scaled_x = x / geom.radius_x
    scaled_y = y / geom.radius_y
    scaled_z = z / geom.radius_z
    rho_sq = scaled_x * scaled_x + scaled_y * scaled_y + scaled_z * scaled_z
    if rho_sq <= 1e-30:
        return -geom.min_radius

    rho = math.sqrt(rho_sq)
    grad_x = x / (geom.radius_x * geom.radius_x)
    grad_y = y / (geom.radius_y * geom.radius_y)
    grad_z = z / (geom.radius_z * geom.radius_z)
    grad_norm = math.sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z) / rho
    if grad_norm > 1e-30:
        return (rho - 1.0) / grad_norm
    return center_radius - geom.min_radius


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def defect_density_2d_from_slice(
    S: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    *,
    S_threshold: float,
    charge_cutoff: float,
) -> tuple[float, int]:
    mask = (S > float(S_threshold)) & np.isfinite(S) & np.isfinite(nx) & np.isfinite(ny)
    theta = np.arctan2(ny, nx)
    psi = 2.0 * theta

    p00 = psi[:-1, :-1]
    p10 = psi[1:, :-1]
    p11 = psi[1:, 1:]
    p01 = psi[:-1, 1:]

    d1 = wrap_to_pi(p10 - p00)
    d2 = wrap_to_pi(p11 - p10)
    d3 = wrap_to_pi(p01 - p11)
    d4 = wrap_to_pi(p00 - p01)
    s_map = 0.5 * (d1 + d2 + d3 + d4) / (2.0 * np.pi)

    plaq_mask = mask[:-1, :-1] & mask[1:, :-1] & mask[1:, 1:] & mask[:-1, 1:]
    defects = np.abs(np.where(plaq_mask, s_map, 0.0)) > float(charge_cutoff)
    used = int(np.count_nonzero(plaq_mask))
    density = float(np.count_nonzero(defects)) / float(used) if used > 0 else 0.0
    return density, used


def mean_and_std(total: float, total_sq: float, count: int) -> tuple[float, float]:
    if count <= 0:
        return float("nan"), float("nan")
    mean = total / float(count)
    variance = max(total_sq / float(count) - mean * mean, 0.0)
    return mean, math.sqrt(variance)


def parse_energy_log(path: Path) -> tuple[int, float, float, float]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if getattr(data, "size", 0) == 0:
        raise ValueError(f"Energy log is empty: {path}")
    field_names = set(data.dtype.names or ())
    iterations = np.atleast_1d(data["iteration"]).astype(float)
    if "free_energy" in field_names:
        energy = np.atleast_1d(data["free_energy"]).astype(float)
    elif "total" in field_names:
        energy = np.atleast_1d(data["total"]).astype(float)
    else:
        raise ValueError(f"Energy log {path} is missing both 'free_energy' and 'total' columns")
    radiality = np.atleast_1d(data["radiality"]).astype(float)
    if "time" in field_names:
        time_s = np.atleast_1d(data["time"]).astype(float)
    elif "time_s" in field_names:
        time_s = np.atleast_1d(data["time_s"]).astype(float)
    else:
        raise ValueError(f"Energy log {path} is missing both 'time' and 'time_s' columns")
    return int(iterations[-1]), float(time_s[-1]), float(energy[-1]), float(radiality[-1])


def compute_field_metrics(
    field_path: Path,
    geom: Geometry,
    *,
    defect_s_threshold: float,
    charge_cutoff: float,
) -> tuple[float, float, float, float, int, int, float, int]:
    shell_sum = 0.0
    shell_sum_sq = 0.0
    shell_count = 0
    inner_sum = 0.0
    inner_sum_sq = 0.0
    inner_count = 0

    z_slice = geom.Nz // 2
    S_slice = np.zeros((geom.Nx, geom.Ny), dtype=float)
    nx_slice = np.zeros((geom.Nx, geom.Ny), dtype=float)
    ny_slice = np.zeros((geom.Nx, geom.Ny), dtype=float)

    with field_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line or line[0] == "#":
                continue
            parts = line.split()
            if len(parts) < 7:
                continue

            i = int(parts[0])
            j = int(parts[1])
            k = int(parts[2])
            S = float(parts[3])
            nx = float(parts[4])
            ny = float(parts[5])

            sd = signed_distance(geom, i, j, k)
            if abs(sd) <= geom.shell_half_thickness:
                shell_sum += S
                shell_sum_sq += S * S
                shell_count += 1
            if -geom.shell_half_thickness <= sd <= 0.0:
                inner_sum += S
                inner_sum_sq += S * S
                inner_count += 1
            if k == z_slice:
                S_slice[i, j] = S
                nx_slice[i, j] = nx
                ny_slice[i, j] = ny

    shell_mean, shell_std = mean_and_std(shell_sum, shell_sum_sq, shell_count)
    inner_mean, inner_std = mean_and_std(inner_sum, inner_sum_sq, inner_count)
    defect_density, used_plaquettes = defect_density_2d_from_slice(
        S_slice,
        nx_slice,
        ny_slice,
        S_threshold=defect_s_threshold,
        charge_cutoff=charge_cutoff,
    )
    return (
        shell_mean,
        shell_std,
        inner_mean,
        inner_std,
        shell_count,
        inner_count,
        defect_density,
        used_plaquettes,
    )


def extract_shell_diagnostics(stdout: str) -> dict[str, float]:
    patterns = {
        "shell_voxels_solver": r"Shell voxels\s*=\s*([0-9]+)",
        "shell_volume_solver_m3": r"Shell volume\s*=\s*([0-9.eE+-]+)",
        "shell_surface_area_estimate_m2": r"Estimated surface area\s*=\s*([0-9.eE+-]+)",
        "shell_nominal_thickness_m": r"Nominal shell thickness\s*=\s*([0-9.eE+-]+)",
        "shell_effective_thickness_m": r"Effective shell thickness\s*=\s*([0-9.eE+-]+)",
        "weak_anchoring_scale_J_m3": r"Weak anchoring scale W/.*=\s*([0-9.eE+-]+)",
    }
    values: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if not match:
            raise ValueError(f"Missing shell diagnostic '{key}' in solver output")
        number = match.group(1)
        values[key] = int(number) if key == "shell_voxels_solver" else float(number)
    return values


def clean_run_dir(run_dir: Path) -> None:
    if run_dir.exists():
        for target in (
            run_dir / "free_energy_vs_iteration.dat",
            run_dir / "energy_components_vs_iteration.dat",
            run_dir / "nematic_field_final.dat",
            run_dir / "Qtensor_output_final.dat",
            run_dir / "solver_stdout.log",
            run_dir / "solver_stderr.log",
        ):
            if target.exists():
                target.unlink()
        output_dir = run_dir / "output"
        if output_dir.exists():
            shutil.rmtree(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def run_case(
    binary: Path,
    config_path: Path,
    output_root: Path,
    *,
    defect_s_threshold: float,
    charge_cutoff: float,
) -> RunMetrics:
    cfg = parse_kv_config(config_path)
    geom = make_geometry(cfg)
    label = config_path.stem.replace("weak_anchor_mesh_", "")
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

    energy_path = run_dir / "free_energy_vs_iteration.dat"
    field_path = run_dir / "nematic_field_final.dat"
    if not energy_path.exists() or not field_path.exists():
        raise FileNotFoundError(f"Expected outputs not found in {run_dir}")

    final_iteration, final_time_s, final_free_energy, final_radiality = parse_energy_log(energy_path)
    shell_mean, shell_std, inner_mean, inner_std, shell_count, inner_count, defect_density, used_plaquettes = compute_field_metrics(
        field_path,
        geom,
        defect_s_threshold=defect_s_threshold,
        charge_cutoff=charge_cutoff,
    )
    shell_diag = extract_shell_diagnostics(result.stdout)

    return RunMetrics(
        label=label,
        config_path=str(config_path),
        run_dir=str(run_dir),
        Nx=geom.Nx,
        Ny=geom.Ny,
        Nz=geom.Nz,
        dx=geom.dx,
        dy=geom.dy,
        dz=geom.dz,
        final_iteration=final_iteration,
        final_time_s=final_time_s,
        final_free_energy=final_free_energy,
        final_radiality=final_radiality,
        shell_order_mask_mean=shell_mean,
        shell_order_mask_std=shell_std,
        shell_order_inner_mean=inner_mean,
        shell_order_inner_std=inner_std,
        shell_mask_points=shell_count,
        shell_inner_points=inner_count,
        defect_density_midplane=defect_density,
        defect_plaquettes_used=used_plaquettes,
        shell_voxels_solver=int(shell_diag["shell_voxels_solver"]),
        shell_volume_solver_m3=float(shell_diag["shell_volume_solver_m3"]),
        shell_surface_area_estimate_m2=float(shell_diag["shell_surface_area_estimate_m2"]),
        shell_nominal_thickness_m=float(shell_diag["shell_nominal_thickness_m"]),
        shell_effective_thickness_m=float(shell_diag["shell_effective_thickness_m"]),
        shell_thickness_ratio_to_nominal=float(shell_diag["shell_effective_thickness_m"]) / float(shell_diag["shell_nominal_thickness_m"]),
        weak_anchoring_scale_J_m3=float(shell_diag["weak_anchoring_scale_J_m3"]),
    )


def relative_difference(a: float, b: float, floor: float = 1e-30) -> float:
    scale = max(abs(a), abs(b), floor)
    return abs(a - b) / scale


def write_metrics_csv(path: Path, metrics: list[RunMetrics]) -> None:
    fieldnames = list(RunMetrics.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            writer.writerow(item.__dict__)


def write_comparison_csv(path: Path, metrics: list[RunMetrics]) -> None:
    reference = metrics[-1]
    fieldnames = [
        "reference_label",
        "test_label",
        "rel_free_energy",
        "abs_radiality_diff",
        "abs_shell_order_inner_diff",
        "abs_shell_order_mask_diff",
        "abs_defect_density_diff",
        "abs_shell_thickness_ratio_diff",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics[:-1]:
            writer.writerow(
                {
                    "reference_label": reference.label,
                    "test_label": item.label,
                    "rel_free_energy": relative_difference(item.final_free_energy, reference.final_free_energy),
                    "abs_radiality_diff": abs(item.final_radiality - reference.final_radiality),
                    "abs_shell_order_inner_diff": abs(item.shell_order_inner_mean - reference.shell_order_inner_mean),
                    "abs_shell_order_mask_diff": abs(item.shell_order_mask_mean - reference.shell_order_mask_mean),
                    "abs_defect_density_diff": abs(item.defect_density_midplane - reference.defect_density_midplane),
                    "abs_shell_thickness_ratio_diff": abs(
                        item.shell_thickness_ratio_to_nominal - reference.shell_thickness_ratio_to_nominal
                    ),
                }
            )


def print_summary(metrics: list[RunMetrics]) -> None:
    for item in metrics:
        print(
            f"[mesh-check] {item.label}: "
            f"F={item.final_free_energy:.8e}, "
            f"Rbar={item.final_radiality:.6f}, "
            f"Shell_in={item.shell_order_inner_mean:.6f}, "
            f"Defect2D={item.defect_density_midplane:.6f}, "
            f"delta_eff={item.shell_effective_thickness_m:.8e} m, "
            f"delta_ratio={item.shell_thickness_ratio_to_nominal:.6f}"
        )

    if len(metrics) > 1:
        reference = metrics[-1]
        for item in metrics[:-1]:
            print(
                f"[mesh-check] {item.label} vs {reference.label}: "
                f"rel_dF={relative_difference(item.final_free_energy, reference.final_free_energy):.6e}, "
                f"dRbar={abs(item.final_radiality - reference.final_radiality):.6e}, "
                f"dShell_in={abs(item.shell_order_inner_mean - reference.shell_order_inner_mean):.6e}, "
                f"dDefect2D={abs(item.defect_density_midplane - reference.defect_density_midplane):.6e}, "
                f"dDeltaRatio={abs(item.shell_thickness_ratio_to_nominal - reference.shell_thickness_ratio_to_nominal):.6e}"
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
    for cfg in configs:
        metrics.append(
            run_case(
                binary,
                cfg,
                output_root,
                defect_s_threshold=args.defect_s_threshold,
                charge_cutoff=args.charge_cutoff,
            )
        )

    write_metrics_csv(output_root / "mesh_metrics.csv", metrics)
    if len(metrics) > 1:
        write_comparison_csv(output_root / "mesh_comparison.csv", metrics)
    print_summary(metrics)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[mesh-check] ERROR: {exc}", file=sys.stderr)
        raise