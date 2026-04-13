#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import QSRvis as qv


SNAPSHOT_GLOBS = {
    "nematic": "nematic_field_iter_*.dat",
    "xy": "xy_field_iter_*.dat",
    "bulk_qtensor": "q_tensor_iter_*.dat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a GUI option-7-style quench GIF from confined nematic snapshots or "
            "converted periodic XY / bulk-LdG snapshots."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing snapshot files for one run.")
    parser.add_argument(
        "--kind",
        required=True,
        choices=tuple(SNAPSHOT_GLOBS.keys()),
        help="Snapshot format used by the selected run.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination GIF path.")
    parser.add_argument(
        "--slice-axis",
        default="z",
        choices=("x", "y", "z"),
        help="Slice axis passed to the GUI-style renderer.",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Slice index along the chosen axis. Defaults to the mid-plane.",
    )
    parser.add_argument("--duration", type=float, default=0.08, help="Frame duration in seconds.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Use every N-th snapshot when animating.")
    parser.add_argument(
        "--color-field",
        default="S",
        choices=("S", "nz", "n_perp"),
        help="Color field passed to the GUI-style renderer.",
    )
    parser.add_argument(
        "--interpolation",
        default="nearest",
        help="imshow interpolation mode passed to the GUI-style renderer.",
    )
    parser.add_argument(
        "--zoom-radius",
        type=int,
        default=None,
        help="Optional crop radius in lattice units around the domain center.",
    )
    parser.add_argument(
        "--arrows-per-axis",
        type=int,
        default=20,
        help="Target number of in-plane director arrows per axis. Set 0 to disable quiver.",
    )
    parser.add_argument(
        "--output-c",
        type=int,
        default=10,
        help="Console progress cadence passed to the GUI-style renderer.",
    )
    parser.add_argument(
        "--keep-converted-dir",
        type=Path,
        default=None,
        help="Optional directory for converted periodic snapshots. Defaults to a temporary directory.",
    )
    return parser.parse_args()


def extract_iter(path: Path) -> int:
    match = re.search(r"(\d+)", path.name)
    if not match:
        raise ValueError(f"Could not extract iteration number from {path}")
    return int(match.group(1))


def list_snapshots(input_dir: Path, kind: str) -> list[Path]:
    files = sorted(input_dir.glob(SNAPSHOT_GLOBS[kind]), key=extract_iter)
    if not files:
        raise FileNotFoundError(
            f"No snapshots matching {SNAPSHOT_GLOBS[kind]} found in {input_dir}"
        )
    return files


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def convert_xy_snapshot(src: Path, dst: Path) -> None:
    with src.open("r", encoding="utf-8", errors="replace") as fin, dst.open("w", encoding="utf-8") as fout:
        fout.write("# x y z S nx ny nz\n")
        for line in fin:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            x_str, y_str, z_str = parts[0], parts[1], parts[2]
            psi_re = float(parts[3])
            psi_im = float(parts[4])
            amp = float(parts[5])
            if amp > 1e-14 and math.isfinite(amp):
                nx = psi_re / amp
                ny = psi_im / amp
            else:
                nx = 1.0
                ny = 0.0
                amp = 0.0
            fout.write(f"{x_str} {y_str} {z_str} {amp:.16g} {nx:.16g} {ny:.16g} 0\n")


def convert_bulk_qtensor_snapshot(src: Path, dst: Path) -> None:
    with src.open("r", encoding="utf-8", errors="replace") as fin, dst.open("w", encoding="utf-8") as fout:
        fout.write("# x y z S nx ny nz\n")
        for line in fin:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 12:
                continue
            x_str, y_str, z_str = parts[0], parts[1], parts[2]
            s_val = float(parts[8])
            nx = float(parts[9])
            ny = float(parts[10])
            nz = float(parts[11])
            fout.write(f"{x_str} {y_str} {z_str} {s_val:.16g} {nx:.16g} {ny:.16g} {nz:.16g}\n")


def prepare_snapshot_dir(args: argparse.Namespace, files: list[Path]) -> tuple[Path, bool]:
    if args.kind == "nematic":
        return args.input_dir, False

    if args.keep_converted_dir is None:
        converted_dir = Path(tempfile.mkdtemp(prefix=f"option7_{args.kind}_"))
        cleanup = True
    else:
        converted_dir = args.keep_converted_dir.resolve()
        ensure_clean_dir(converted_dir)
        cleanup = False

    converter = convert_xy_snapshot if args.kind == "xy" else convert_bulk_qtensor_snapshot
    for src in files:
        iter_num = extract_iter(src)
        dst = converted_dir / f"nematic_field_iter_{iter_num:06d}.dat"
        converter(src, dst)
    return converted_dir, cleanup


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = list_snapshots(input_dir, args.kind)
    nx, ny, nz = qv.infer_grid_dims_from_nematic_field_file(str(files[0]))
    data_dir, cleanup = prepare_snapshot_dir(args, files)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames_dir = output_path.parent / f".frames_{output_path.stem}"
    try:
        qv.create_nematic_field_animation(
            data_dir=str(data_dir),
            output_gif=str(output_path),
            Nx=nx,
            Ny=ny,
            Nz=nz,
            slice_axis=args.slice_axis,
            slice_index=args.slice_index,
            frames_dir=str(frames_dir),
            duration=args.duration,
            frame_stride=max(1, args.frame_stride),
            color_field=args.color_field,
            interpol=args.interpolation,
            zoom_radius=args.zoom_radius,
            arrowColor="black",
            arrows_per_axis=args.arrows_per_axis,
            consistent_scale=True,
            output_c=args.output_c,
        )
    finally:
        if cleanup and data_dir.exists():
            shutil.rmtree(data_dir, ignore_errors=True)

    print(f"Saved GIF -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())