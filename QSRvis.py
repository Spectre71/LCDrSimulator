import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import imageio.v2 as imageio
import re
import io
from typing import Any


def prompt_pick_folder(
    *,
    prompt: str,
    default: str,
    pattern: str | None = None,
    must_exist: bool = True,
) -> str:
    """Lightweight "folder picker" for CLI.

    - If pattern is given, lists matching directories and lets user choose by index.
    - User may also paste a path.
    """
    candidates: list[str] = []
    if pattern:
        candidates = [p for p in glob.glob(pattern) if os.path.isdir(p)]
        candidates.sort()
        if candidates:
            print("Available folders:")
            for idx, p in enumerate(candidates):
                print(f"  [{idx}] {p}")

    raw = input(f"{prompt} [default: {default}]: ").strip()
    if not raw:
        chosen = default
    elif raw.isdigit() and candidates:
        idx = int(raw)
        idx = max(0, min(len(candidates) - 1, idx))
        chosen = candidates[idx]
    else:
        chosen = raw

    if must_exist and chosen and not os.path.exists(chosen):
        raise FileNotFoundError(f"Path not found: {chosen}")
    return chosen


def _parse_iter_window(s: str) -> tuple[int | None, int | None]:
    """Parse an iteration window string like '50000:70000', ':70000', '50000:' or '50000-70000'."""
    s = (s or '').strip()
    if not s:
        return None, None

    # Support '-' as a separator if ':' isn't used.
    if ':' not in s and '-' in s:
        parts = s.split('-', 1)
        s = f"{parts[0]}:{parts[1]}"

    if ':' in s:
        a, b = s.split(':', 1)
        a = a.strip()
        b = b.strip()
        it_min = int(float(a)) if a else None
        it_max = int(float(b)) if b else None
    else:
        it_min = int(float(s))
        it_max = None

    if it_min is not None and it_max is not None and it_min > it_max:
        it_min, it_max = it_max, it_min
    return it_min, it_max


def _apply_iter_window(it: np.ndarray, *arrays: np.ndarray, it_min: int | None, it_max: int | None):
    """Apply an iteration window to it and any parallel arrays."""
    it = np.asarray(it, dtype=float)
    m = np.ones_like(it, dtype=bool)
    if it_min is not None:
        m &= it >= float(it_min)
    if it_max is not None:
        m &= it <= float(it_max)
    out = [it[m]]
    for a in arrays:
        out.append(np.asarray(a)[m])
    return tuple(out)

# Optional deps for 3D analysis (installed in venv when needed).
# Keep them as Any so type-checkers don't get confused by conditional imports.
ndi: Any = None
skeletonize_3d: Any = None
try:  # pragma: no cover
    import scipy.ndimage as ndi  # type: ignore
except Exception:  # pragma: no cover
    pass
try:  # pragma: no cover
    # Newer scikit-image versions expose a 3D-capable `skeletonize` (M,N[,P]).
    from skimage.morphology import skeletonize as _skeletonize  # type: ignore
    skeletonize_3d = _skeletonize
except Exception:  # pragma: no cover
    pass

def load_qtensor_data(path, comments="#"):
    """Loads Qtensor output with columns: i j k Qxx Qxy Qxz Qyx Qyy Qyz Qzx Qzy Qzz."""
    return load_nematic_field_data(path, comments=comments)

def _compute_trQ2_trQ3_beta_from_rows(rows, trQ2_eps=1e-12):
    """Vectorized invariants and biaxiality from Qtensor rows.

    rows columns: i j k Qxx Qxy Qxz Qyx Qyy Qyz Qzx Qzy Qzz
    Uses symmetric components for invariants.
    """
    Qxx = rows[:, 3]
    Qxy = rows[:, 4]
    Qxz = rows[:, 5]
    Qyy = rows[:, 7]
    Qyz = rows[:, 8]
    Qzz = rows[:, 11]

    trQ2 = Qxx * Qxx + Qyy * Qyy + Qzz * Qzz + 2.0 * (Qxy * Qxy + Qxz * Qxz + Qyz * Qyz)
    trQ3 = (
        Qxx**3 + Qyy**3 + Qzz**3
        + 3.0 * Qxx * (Qxy * Qxy + Qxz * Qxz)
        + 3.0 * Qyy * (Qxy * Qxy + Qyz * Qyz)
        + 3.0 * Qzz * (Qxz * Qxz + Qyz * Qyz)
        + 6.0 * Qxy * Qxz * Qyz
    )

    beta = np.zeros_like(trQ2)
    valid = trQ2 > trQ2_eps
    beta[valid] = 1.0 - 6.0 * (trQ3[valid] * trQ3[valid]) / (trQ2[valid] * trQ2[valid] * trQ2[valid])
    beta = np.clip(beta, 0.0, 1.0)
    return trQ2, trQ3, beta


def biaxiality_report(
    qtensor_filename,
    Nx,
    Ny,
    Nz,
    z_slice=None,
    core_radius=6,
    trQ2_eps=1e-12,
    global_max=True,
    droplet_radius=None,
):
    """Print a biaxiality-based diagnostic report for a given z-slice.

    Uses the standard biaxiality parameter:
        beta = 1 - 6 * (tr(Q^3)^2) / (tr(Q^2)^3)
    where beta≈0 is uniaxial and beta→1 indicates strong biaxiality.

    Heuristic interpretation (LdG context):
    - True defect core often shows reduced tr(Q^2) and/or increased beta in a small central region.
    - "Escape" configurations can keep tr(Q^2) high while the director tilts out-of-plane.
    """
    if z_slice is None:
        z_slice = Nz // 2

    try:
        data = load_qtensor_data(qtensor_filename, comments="#")
    except IOError:
        print(f"Warning: Could not read file '{qtensor_filename}'.")
        return

    # Select the data for the desired z-slice
    slice_data = data[data[:, 2] == z_slice]
    if slice_data.shape[0] == 0:
        print(f"Warning: No data found for z_slice = {z_slice} in {qtensor_filename}.")
        return

    # Fill 2D fields from file (use symmetric components)
    Qxx = np.zeros((Nx, Ny))
    Qyy = np.zeros((Nx, Ny))
    Qzz = np.zeros((Nx, Ny))
    Qxy = np.zeros((Nx, Ny))
    Qxz = np.zeros((Nx, Ny))
    Qyz = np.zeros((Nx, Ny))

    for row in slice_data:
        i, j = int(row[0]), int(row[1])
        if i < Nx and j < Ny:
            Qxx[i, j] = row[3]
            Qxy[i, j] = row[4]
            Qxz[i, j] = row[5]
            Qyy[i, j] = row[7]
            Qyz[i, j] = row[8]
            Qzz[i, j] = row[11]

    # Invariants/biaxiality for this slice
    trQ2 = Qxx * Qxx + Qyy * Qyy + Qzz * Qzz + 2.0 * (Qxy * Qxy + Qxz * Qxz + Qyz * Qyz)
    trQ3 = (
        Qxx**3 + Qyy**3 + Qzz**3
        + 3.0 * Qxx * (Qxy * Qxy + Qxz * Qxz)
        + 3.0 * Qyy * (Qxy * Qxy + Qyz * Qyz)
        + 3.0 * Qzz * (Qxz * Qxz + Qyz * Qyz)
        + 6.0 * Qxy * Qxz * Qyz
    )

    beta = np.zeros_like(trQ2)
    valid = trQ2 > trQ2_eps
    beta[valid] = 1.0 - 6.0 * (trQ3[valid] * trQ3[valid]) / (trQ2[valid] * trQ2[valid] * trQ2[valid])
    beta = np.clip(beta, 0.0, 1.0)

    # Define a "nematic" mask and compute a bulk reference level
    nem_mask = trQ2 > (100.0 * trQ2_eps)
    if np.any(nem_mask):
        trQ2_bulk = float(np.percentile(trQ2[nem_mask], 95))
        beta_bulk = float(np.percentile(beta[nem_mask], 95))
    else:
        trQ2_bulk = 0.0
        beta_bulk = 0.0

    cx, cy = Nx // 2, Ny // 2
    yy, xx = np.meshgrid(np.arange(Ny), np.arange(Nx))
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    core = rr <= core_radius
    core_nem = core & nem_mask

    trQ2_center = float(trQ2[cx, cy])
    beta_center = float(beta[cx, cy])

    if np.any(core_nem):
        core_trQ2_min = float(np.min(trQ2[core_nem]))
        core_trQ2_mean = float(np.mean(trQ2[core_nem]))
        core_beta_max = float(np.max(beta[core_nem]))
        core_beta_mean = float(np.mean(beta[core_nem]))
    else:
        core_trQ2_min = 0.0
        core_trQ2_mean = 0.0
        core_beta_max = 0.0
        core_beta_mean = 0.0

    print(f"\n[BIAXIALITY | {os.path.basename(qtensor_filename)} | z={z_slice}] core_radius={core_radius}")
    print(f"  Center: tr(Q^2)={trQ2_center:.6g}, beta={beta_center:.6g}")
    print(f"  Core (nematic pts): tr(Q^2) min/mean={core_trQ2_min:.6g}/{core_trQ2_mean:.6g}, beta max/mean={core_beta_max:.6g}/{core_beta_mean:.6g}")
    print(f"  Bulk ref (slice): tr(Q^2)_p95={trQ2_bulk:.6g}, beta_p95={beta_bulk:.6g}")

    # Simple heuristic conclusion
    conclusion = "inconclusive"
    if trQ2_bulk > 0:
        drop = core_trQ2_min / trQ2_bulk
        if (core_trQ2_min > 0) and (drop < 0.2) and (core_beta_max > 0.5):
            conclusion = "likely LdG defect core (reduced order + biaxiality)"
        elif (core_trQ2_min > 0) and (drop < 0.2):
            conclusion = "possible defect core (order drops), check beta/3D structure"
        elif core_beta_max > 0.7:
            conclusion = "biaxial region present (may indicate defect core or strong distortion)"
        else:
            conclusion = "order stays high in core (consistent with escape/uniaxial core)"
    print(f"  Heuristic: {conclusion}\n")

    if global_max:
        # Global scan: find maximum beta in the whole droplet (3D).
        # This is often a robust way to localize defect cores (strong biaxiality pockets).
        try:
            all_rows = data
        except Exception:
            all_rows = None

        if all_rows is None or all_rows.size == 0:
            return

        i_all = all_rows[:, 0].astype(np.int32)
        j_all = all_rows[:, 1].astype(np.int32)
        k_all = all_rows[:, 2].astype(np.int32)

        trQ2_all, trQ3_all, beta_all = _compute_trQ2_trQ3_beta_from_rows(all_rows, trQ2_eps=trQ2_eps)

        cx3, cy3, cz3 = Nx / 2.0, Ny / 2.0, Nz / 2.0
        if droplet_radius is None:
            R = 0.5 * float(min(Nx, Ny, Nz))
        else:
            R = float(droplet_radius)

        dx = (i_all.astype(np.float64) - cx3)
        dy = (j_all.astype(np.float64) - cy3)
        dz = (k_all.astype(np.float64) - cz3)
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        inside = r < R

        # consider points inside droplet even if trQ2 small; beta is only defined where trQ2>eps.
        cand = inside & (trQ2_all > trQ2_eps)
        if not np.any(cand):
            print("[BIAXIALITY] Global scan: no valid points (trQ2 too small everywhere?).")
            return

        idxs = np.nonzero(cand)[0]
        local = idxs[np.argmax(beta_all[idxs])]
        bmax = float(beta_all[local])
        tr2 = float(trQ2_all[local])
        # For uniaxial Q: tr(Q^2) = (2/3) S^2  =>  S_mag ≈ sqrt(1.5*trQ2)
        Smag = float(np.sqrt(max(0.0, 1.5 * tr2)))
        print(f"[BIAXIALITY] Global beta_max={bmax:.6g} at (i,j,k)=({int(i_all[local])},{int(j_all[local])},{int(k_all[local])}), r={float(r[local]):.3g}")
        print(f"            tr(Q^2)={tr2:.6g}, |S|≈sqrt(1.5 trQ2)={Smag:.6g}")

def load_nematic_field_data(path, comments="#"):
    try:
        return np.loadtxt(path, comments=comments)
    except ValueError:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        text = re.sub(r"(?<=\d)[uU]([+-]\d+)", r"e\1", text)
        return np.loadtxt(io.StringIO(text), comments=comments)


def infer_grid_dims_from_nematic_field_file(path: str) -> tuple[int, int, int]:
    """Infer (Nx,Ny,Nz) from a nematic field dump file.

    Expected columns include integer i j k as the first 3 columns.
    Uses a streaming scan (no full load into memory).
    """
    max_i = -1
    max_j = -1
    max_k = -1
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                i = int(float(parts[0]))
                j = int(float(parts[1]))
                k = int(float(parts[2]))
            except ValueError:
                continue
            if i > max_i:
                max_i = i
            if j > max_j:
                max_j = j
            if k > max_k:
                max_k = k

    if max_i < 0 or max_j < 0 or max_k < 0:
        raise ValueError(f"Could not infer grid dims from: {path}")
    return max_i + 1, max_j + 1, max_k + 1

def plot_nematic_field_slice(
    filename,
    Nx,
    Ny,
    Nz,
    z_slice=None,
    slice_axis: str = 'z',
    output_path=None,
    arrowColor=None,
    zoom_radius=None,
    interpol='nearest',
    color_field='S',
    auto_vmax_percentile=99.0,
    vmin=None,
    vmax=None,
    print_stats=True,
    arrows_per_axis=None,
):
    """
    Loads pre-calculated nematic field data (S, n), plots a slice, and saves it.
    This function is much more direct as it doesn't need to calculate eigenvalues.
    """
    axis = (slice_axis or 'z').strip().lower()
    axis = axis[0] if axis else 'z'
    if axis not in ('x', 'y', 'z'):
        print(f"Warning: Unknown slice_axis='{slice_axis}'. Falling back to 'z'.")
        axis = 'z'

    # Backwards-compatible: keep the argument name z_slice, but interpret it as
    # the slice index along the chosen axis.
    if z_slice is None:
        if axis == 'z':
            z_slice = Nz // 2
        elif axis == 'y':
            z_slice = Ny // 2
        else:  # axis == 'x'
            z_slice = Nx // 2

    # Load the raw data from the simulation output.
    # Columns are: i, j, k, S, nx, ny, nz
    try:
        # Use comments='#' to ignore the header line in the data file
        data = load_nematic_field_data(filename, comments="#")
    except IOError:
        print(f"Warning: Could not read file '{filename}'. Skipping.")
        return

    # Select the data for the desired slice plane
    # Columns are: i, j, k, S, nx, ny, nz
    if axis == 'z':
        slice_data = data[data[:, 2] == z_slice]
        plane_u, plane_v = int(Nx), int(Ny)
        u_label, v_label = '$x$', '$y$'
        slice_label = 'z'
        # Vector components in the plane
        vec_u_name, vec_v_name = 'nx', 'ny'
    elif axis == 'y':
        slice_data = data[data[:, 1] == z_slice]
        plane_u, plane_v = int(Nx), int(Nz)
        u_label, v_label = '$x$', '$z$'
        slice_label = 'y'
        vec_u_name, vec_v_name = 'nx', 'nz'
    else:  # axis == 'x'
        slice_data = data[data[:, 0] == z_slice]
        plane_u, plane_v = int(Ny), int(Nz)
        u_label, v_label = '$y$', '$z$'
        slice_label = 'x'
        vec_u_name, vec_v_name = 'ny', 'nz'

    if slice_data.shape[0] == 0:
        print(f"Warning: No data found for {slice_label}_slice = {z_slice} in {filename}. Skipping.")
        return

    # Prepare 2D arrays for S, n (in the chosen plane)
    S = np.zeros((plane_u, plane_v))
    nx = np.zeros((plane_u, plane_v))
    ny = np.zeros((plane_u, plane_v))
    nz = np.zeros((plane_u, plane_v))

    # Directly populate arrays from the file data
    for row in slice_data:
        ii, jj, kk = int(row[0]), int(row[1]), int(row[2])
        if axis == 'z':
            u, v = ii, jj
        elif axis == 'y':
            u, v = ii, kk
        else:  # axis == 'x'
            u, v = jj, kk

        if 0 <= u < plane_u and 0 <= v < plane_v:
            S[u, v] = row[3]
            nx[u, v] = row[4]
            ny[u, v] = row[5]
            nz[u, v] = row[6]

    # Vector components to plot in the plane
    if vec_u_name == 'nx':
        vec_u = nx
    elif vec_u_name == 'ny':
        vec_u = ny
    else:
        vec_u = nz
    if vec_v_name == 'nx':
        vec_v = nx
    elif vec_v_name == 'ny':
        vec_v = ny
    else:
        vec_v = nz

    if print_stats:
        cu, cv = plane_u // 2, plane_v // 2
        if 0 <= cu < plane_u and 0 <= cv < plane_v:
            print(
                f"[{os.path.basename(filename)} | {slice_label}={z_slice}] "
                f"S_center={S[cu, cv]:.6g}, "
                f"S_min={np.min(S):.6g}, S_max={np.max(S):.6g}; "
                f"n_center=({nx[cu, cv]:.4g},{ny[cu, cv]:.4g},{nz[cu, cv]:.4g}), "
                f"|n_perp(view)|_center={np.sqrt(vec_u[cu, cv]**2 + vec_v[cu, cv]**2):.6g}"
            )

    # --- Handle zooming ---
    if zoom_radius is not None:
        center_u, center_v = plane_u // 2, plane_v // 2
        x_min = max(0, center_u - zoom_radius)
        x_max = min(plane_u, center_u + zoom_radius)
        y_min = max(0, center_v - zoom_radius)
        y_max = min(plane_v, center_v + zoom_radius)

        # Slice the arrays
        S_view = S[x_min:x_max, y_min:y_max]
        nx_view = nx[x_min:x_max, y_min:y_max]
        ny_view = ny[x_min:x_max, y_min:y_max]
        nz_view = nz[x_min:x_max, y_min:y_max]
        vec_u_view = vec_u[x_min:x_max, y_min:y_max]
        vec_v_view = vec_v[x_min:x_max, y_min:y_max]

        extent = (x_min, x_max, y_min, y_max)
        step = 1  # default: dense in zoomed view
    else:
        S_view = S
        nx_view = nx
        ny_view = ny
        nz_view = nz
        vec_u_view = vec_u
        vec_v_view = vec_v
        extent = (0, plane_u, 0, plane_v)

        # Default: dense (historical behavior)
        step = 1

    # --- Quiver density control ---
    # arrows_per_axis: target number of arrows along the longest axis.
    # - None: keep historical behavior
    # - 0: disable quiver
    if arrows_per_axis is not None:
        try:
            arrows_per_axis_int = int(arrows_per_axis)
        except Exception:
            arrows_per_axis_int = None

        if arrows_per_axis_int is not None:
            if arrows_per_axis_int <= 0:
                step = None  # sentinel: disable quiver
            else:
                width = int(extent[1] - extent[0])
                height = int(extent[3] - extent[2])
                max_dim = max(width, height)
                # ceil(max_dim / arrows_per_axis) keeps arrow count <= target
                step = max(1, int(np.ceil(max_dim / float(arrows_per_axis_int))))
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(9, 8))

    color_field_norm = (color_field or 'S').strip().lower()
    if color_field_norm in ('s', 'scalar', 'order', 'orderparameter'):
        color_field_norm = 's'
    elif color_field_norm in ('nz', 'n_z'):
        color_field_norm = 'nz'
    elif color_field_norm in ('nperp', 'n_perp', 'perp', 'nxy', 'n_xy'):
        color_field_norm = 'n_perp'
    else:
        print(f"Warning: Unknown color_field='{color_field}'. Falling back to 'S'.")
        color_field_norm = 's'

    if color_field_norm == 's':
        field = S_view
        cmap = 'viridis'
        label = '$S$'
        if vmin is None:
            vmin_use = 0.0
        else:
            vmin_use = float(vmin)
        if vmax is None:
            valid = field[np.isfinite(field)]
            if valid.size == 0:
                vmax_use = 1.0
            else:
                vmax_use = float(np.percentile(valid, auto_vmax_percentile))
                if vmax_use <= vmin_use:
                    vmax_use = float(np.max(valid))
                if vmax_use <= vmin_use:
                    vmax_use = vmin_use + 1e-6
        else:
            vmax_use = float(vmax)
    elif color_field_norm == 'nz':
        field = nz_view
        cmap = 'RdBu_r'
        label = '$n_z$'
        vmin_use = -1.0 if vmin is None else float(vmin)
        vmax_use = 1.0 if vmax is None else float(vmax)
    else:  # n_perp
        if axis == 'z':
            field = np.sqrt(nx_view * nx_view + ny_view * ny_view)
        elif axis == 'y':
            field = np.sqrt(nx_view * nx_view + nz_view * nz_view)
        else:  # axis == 'x'
            field = np.sqrt(ny_view * ny_view + nz_view * nz_view)
        cmap = 'magma'
        if axis == 'z':
            label = r'$|n_\perp(z)|=\sqrt{n_x^2+n_y^2}$'
        elif axis == 'y':
            label = r'$|n_\perp(y)|=\sqrt{n_x^2+n_z^2}$'
        else:
            label = r'$|n_\perp(x)|=\sqrt{n_y^2+n_z^2}$'
        vmin_use = 0.0 if vmin is None else float(vmin)
        vmax_use = 1.0 if vmax is None else float(vmax)

    im = ax.imshow(field.T, origin='lower', cmap=cmap, extent=extent, vmin=vmin_use, vmax=vmax_use, interpolation=interpol)
    fig.colorbar(im, ax=ax, label=label)
    
    if step is not None:
        # Create grid for quiver
        x_range = np.arange(extent[0], extent[1], step)
        y_range = np.arange(extent[2], extent[3], step)
        x_grid, y_grid = np.meshgrid(x_range, y_range)

        # Extract data for quiver at step intervals
        # Indices for slicing the *view* arrays
        ix = np.arange(0, S_view.shape[0], step)
        iy = np.arange(0, S_view.shape[1], step)

        # x_grid corresponds to columns (i), y_grid to rows (j)
        nx_plot = vec_u_view[np.ix_(ix, iy)].T
        ny_plot = vec_v_view[np.ix_(ix, iy)].T
        S_plot_mask = S_view[np.ix_(ix, iy)].T

        mask = S_plot_mask > 0.1

        if arrowColor is None:
            arrowColor = 'black'

        ax.quiver(
            x_grid[mask], y_grid[mask], nx_plot[mask], ny_plot[mask],
            color=arrowColor,
            scale=30 if zoom_radius is None else 15,
            headwidth=3,
            pivot='middle',
        )
    
    # Extract iteration number from filename for the title
    match = re.search(r'(\d+)', os.path.basename(filename))
    iter_str = match.group(1) if match else "Final"
    title = f"Nematično polje pri (${slice_label}={z_slice}$, Iter: {iter_str}, barva: {color_field_norm})"
    if zoom_radius: title += " [ZOOMED]"
        
    ax.set_title(title)
    ax.set_xlabel(u_label)
    ax.set_ylabel(v_label)
    ax.set_xlim(0, plane_u)
    ax.set_ylim(0, plane_v)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()

    if output_path:
        # if dir doesnt exist, make one
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
    else:
        # If no output path is given, show the plot interactively
        plt.show()
        
    plt.close(fig)

def _resolve_data_file(path_or_dir: str, filename: str) -> str:
    """Resolve a data file path from either a directory or a direct file path."""
    if not path_or_dir:
        path_or_dir = '.'
    if os.path.isdir(path_or_dir):
        return os.path.join(path_or_dir, filename)
    return path_or_dir


def plot_energy_VS_iter(path: str = '.', out_dir: str = 'pics', show: bool = True):
    """Plot free energy vs iteration.

    Primary source: `free_energy_vs_iteration.dat`.
    Fallback (for quench runs): `quench_log.dat/.csv` where we use `total` as F.

    `path` may be a directory or a direct file path.
    """
    data_path = _resolve_data_file(path, 'free_energy_vs_iteration.dat')

    use_quench_log = not os.path.exists(data_path)
    if use_quench_log:
        # Quench runs usually only have quench_log.*
        q, log_path = load_quench_log(path)
        it = np.atleast_1d(q['iteration']).astype(float)
        F = np.atleast_1d(q['total']).astype(float)
        radial = np.atleast_1d(q['radiality']).astype(float)
        time_s = np.atleast_1d(q['time_s']).astype(float)

        m = np.isfinite(it) & np.isfinite(F) & np.isfinite(radial) & np.isfinite(time_s)
        if not np.any(m):
            raise ValueError(f"Quench log contains no finite rows: {log_path}")
        it, F, radial, time_s = it[m], F[m], radial[m], time_s[m]
        tag = os.path.basename(os.path.normpath(os.path.dirname(log_path) or '.'))
        source_note = f"(from quench log: total)"
    else:
        # Format: iteration,free_energy,radiality,time
        data = np.genfromtxt(data_path, delimiter=',', names=True)
        it = np.asarray(data['iteration'], dtype=float)
        F = np.asarray(data['free_energy'], dtype=float)
        radial = np.asarray(data['radiality'], dtype=float)
        time_s = np.asarray(data['time'], dtype=float)
        tag = os.path.basename(os.path.normpath(os.path.dirname(data_path) or '.'))
        source_note = ""

    # Create subplots for energy, radiality, and time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Free Energy vs Iteration
    ax1.plot(it, F, marker='o', linestyle='-', color='blue')
    ax1.set_xlabel('$i$')
    ax1.set_ylabel('$F$ [J]', color='blue')
    ax1.set_title(f'$F(i)$ {source_note}'.strip())
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    
    # Plot radiality on secondary y-axis
    ax1_twin = ax1.twinx()
    ax1_twin.plot(it, radial, marker='s', linestyle='--', color='red', alpha=0.7)
    ax1_twin.set_ylabel(r'Radialnost $\overline{R}$', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    ax1_twin.set_ylim([0, 1.05])
    
    # Plot 2: Physical Time vs Iteration
    ax2.plot(it, time_s, marker='^', linestyle='-', color='green')
    ax2.set_xlabel('$i$')
    ax2.set_ylabel('$t$ [s]', color='green')
    ax2.set_title('$t(i)$')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'free_energy_vs_iteration_{tag}.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot -> {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

def energy_components():
    # Format: iteration,bulk,elastic,total,radiality,time
    data = np.genfromtxt('energy_components_vs_iteration.dat', delimiter=',', names=True)

    # Create subplots for energy components, radiality, and time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Energy Components
    ax1.plot(data['iteration'], data['bulk'], label='Bulk', linewidth=2)
    ax1.plot(data['iteration'], data['elastic'], label='Elastic', linewidth=2)
    ax1.plot(data['iteration'], data['total'], label='Total', linestyle='--', color='k', linewidth=2)
    ax1.set_xlabel('$i$')
    ax1.set_ylabel('$F$ [J]')
    ax1.set_title('$F_{komp}(t)$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Radiality and Time
    ax2_left = ax2
    ax2_left.plot(data['iteration'], data['radiality'], marker='o', linestyle='-', color='red', label=r'Radialnost $\overline{R}$')
    ax2_left.set_xlabel('$i$')
    ax2_left.set_ylabel(r'Radialnost $\overline{R}$', color='red')
    ax2_left.tick_params(axis='y', labelcolor='red')
    ax2_left.set_ylim([0, 1.05])
    ax2_left.grid(True, alpha=0.3)
    
    # Plot physical time on secondary y-axis
    ax2_right = ax2.twinx()
    ax2_right.plot(data['iteration'], data['time'], marker='s', linestyle='--', color='green', alpha=0.7, label='Fizični čas')
    ax2_right.set_ylabel('$t$ [s]', color='green')
    ax2_right.tick_params(axis='y', labelcolor='green')
    
    fig.tight_layout()
    if not os.path.exists('pics'):
        os.makedirs('pics')
    plt.savefig('pics/energy_components_vs_iteration.png', dpi=150)
    plt.show()

def create_nematic_field_animation(
    data_dir,
    output_gif,
    Nx,
    Ny,
    Nz,
    *,
    frames_dir="frames",
    duration=0.1,
    frame_stride=1,
    color_field='S',
    interpol='nearest',
    zoom_radius=None,
    arrowColor=None,
    arrows_per_axis=None,
    consistent_scale=True,
    output_c=None,
):
    """Create a GIF from nematic field snapshots in a directory.

    Uses files: {data_dir}/nematic_field_iter_*.dat
    """
    if frame_stride is None or frame_stride < 1:
        frame_stride = 1

    # Create a directory for temporary frames
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    else: # Clean up old frames from a previous run
        for f in glob.glob(os.path.join(frames_dir, '*.png')):
            os.remove(f)
    
    # Find all nematic field snapshot files and sort them numerically
    files = glob.glob(os.path.join(data_dir, 'nematic_field_iter_*.dat'))
    if not files:
        print(f"Error: No 'nematic_field_iter_*.dat' files found in '{data_dir}'.")
        return
        
    def _extract_iter_num(path):
        m = re.search(r'(\d+)', os.path.basename(path))
        return int(m.group(1)) if m else -1

    files.sort(key=_extract_iter_num)

    if frame_stride > 1:
        files = files[::frame_stride]

    # For smooth animations, keep color scaling constant across frames when possible.
    vmin, vmax = None, None
    cf = (color_field or 'S').strip().lower()
    if consistent_scale:
        if cf in ('s', 'scalar', 'order', 'orderparameter'):
            z_slice = Nz // 2
            vals = []
            for fp in files:
                try:
                    S_arr, _, _, _ = _load_nematic_slice_arrays(fp, Nx, Ny, Nz, z_slice)
                except Exception:
                    continue
                v = S_arr[np.isfinite(S_arr)]
                v = v[v > 0.1]
                if v.size:
                    vals.append(v)
            if vals:
                vv = np.concatenate(vals)
                vmin = 0.0
                vmax = float(np.percentile(vv, 99.0))
                if vmax <= vmin:
                    vmax = float(np.max(vv))
                if vmax <= vmin:
                    vmax = vmin + 1e-6
        elif cf in ('nz', 'n_z'):
            vmin, vmax = -1.0, 1.0
        elif cf in ('nperp', 'n_perp', 'perp', 'nxy', 'n_xy'):
            vmin, vmax = 0.0, 1.0

    print(f"Found {len(files)} nematic field snapshots. Generating frames...")

    frame_paths = []
    for i, file in enumerate(files):
        frame_path = os.path.join(frames_dir, f'frame_{i:04d}.png')
        if output_c is not None and i % output_c == 0:
            print(f"Generating frame {i+1}/{len(files)}: {frame_path}")
        plot_nematic_field_slice(
            file,
            Nx,
            Ny,
            Nz,
            output_path=frame_path,
            arrowColor=arrowColor,
            zoom_radius=zoom_radius,
            interpol=interpol,
            color_field=color_field,
            vmin=vmin,
            vmax=vmax,
            print_stats=False,
            arrows_per_axis=arrows_per_axis,
        )
        frame_paths.append(frame_path)

    # Create GIF from the generated frames
    print(f"\nStitching {len(frame_paths)} frames into {output_gif}...")
    os.makedirs(os.path.dirname(output_gif) or '.', exist_ok=True)
    writer: Any = imageio.get_writer(output_gif, mode='I', duration=float(duration))
    try:
        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)
    finally:
        writer.close()

    # Clean up temporary frames
    print("Cleaning up temporary frames...")
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.rmdir(frames_dir)

    print(f"\nAnimation saved to {output_gif}")

def plotS_F(path: str = 'output_temp_sweep', out_dir: str = 'pics', show: bool = True):
    """Plot average S(T) and final free energy F(T) from a temperature sweep summary.

    Primary source: output_temp_sweep/summary.dat.

    Fallback (for quench runs): one or more quench logs (output_quench*/quench_log.*).
    In fallback mode, we extract the *final* values per run:
      T_final = last finite T_K
      S_final = last finite avg_S
      F_final = last finite total

    `path` may be a directory containing summary.dat, a direct summary.dat path, a single
    quench run directory, or a directory containing multiple output_quench* runs.
    """
    summary_path = _resolve_data_file(path, 'summary.dat')
    if not os.path.exists(summary_path):
        # Quench fallback: build S(T) and F(T) from the full quench_log across steps.
        run_dirs: list[str] = []
        # If the user passed a glob pattern, expand it.
        if any(ch in (path or '') for ch in ('*', '?', '[')):
            run_dirs = [p for p in glob.glob(path) if os.path.isdir(p)]
        elif os.path.isdir(path):
            # Single run directory?
            try:
                lp = _resolve_quench_log_path(path)
                if os.path.exists(lp):
                    run_dirs = [path]
            except Exception:
                run_dirs = []
            # Or a parent directory containing multiple runs
            if not run_dirs:
                run_dirs = [p for p in glob.glob(os.path.join(path, 'output_quench*')) if os.path.isdir(p)]
        elif os.path.isfile(path):
            # Direct log file path
            run_dirs = [path]

        runs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        for rd in sorted(run_dirs):
            try:
                q, log_path = load_quench_log(rd)
                it = np.atleast_1d(q['iteration']).astype(float)
                Tq = np.atleast_1d(q['T_K']).astype(float)
                Sq = np.atleast_1d(q['avg_S']).astype(float)
                Fq = np.atleast_1d(q['total']).astype(float)
                m = np.isfinite(it) & np.isfinite(Tq) & np.isfinite(Sq) & np.isfinite(Fq)
                if np.count_nonzero(m) < 2:
                    continue

                it, Tq, Sq, Fq = it[m], Tq[m], Sq[m], Fq[m]
                order = np.argsort(it)
                Tq, Sq, Fq = Tq[order], Sq[order], Fq[order]
                label = os.path.basename(os.path.normpath(os.path.dirname(log_path) or rd))
                runs.append((label, Tq, Sq, Fq))
            except Exception:
                continue

        if not runs:
            print(f"Missing file: {summary_path}")
            print("Also no quench logs found for fallback (expected output_quench*/quench_log.*).")
            return

        os.makedirs(out_dir, exist_ok=True)
        tag = os.path.basename(os.path.normpath(path or '.'))
        show_legend = len(runs) <= 8

        plt.figure(figsize=(8, 5))
        for label, Tq, Sq, _ in runs:
            plt.plot(Tq, Sq, '-', lw=1.5, marker='o', ms=2.5, alpha=0.85, label=label)
        plt.xlabel('$T$ [K]')
        plt.ylabel('$S$')
        plt.title(r'$\langle S\rangle(T)$ from quench log')
        plt.grid(True)
        if show_legend:
            plt.legend()
        plt.tight_layout()
        out1 = os.path.join(out_dir, f'average_S_vs_T_quenchlog_{tag}.png')
        plt.savefig(out1)
        print(f"Saved plot -> {out1}")
        if show:
            plt.show()
        else:
            plt.close()

        plt.figure(figsize=(8, 5))
        for label, Tq, _, Fq in runs:
            plt.plot(Tq, Fq, '-', lw=1.5, marker='s', ms=2.5, alpha=0.85, label=label)
        plt.xlabel('$T$ [K]')
        plt.ylabel('$F$')
        plt.title('$F(T)$ from quench log (total energy)')
        plt.grid(True)
        if show_legend:
            plt.legend()
        plt.tight_layout()
        out2 = os.path.join(out_dir, f'free_energy_vs_T_quenchlog_{tag}.png')
        plt.savefig(out2)
        print(f"Saved plot -> {out2}")
        if show:
            plt.show()
        else:
            plt.close()
        return

    # Avoid numpy genfromtxt crashing on empty/header-only files (common if sweep is interrupted)
    with open(summary_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if len(lines) < 2:
        print(f"No sweep data to plot (file has {len(lines)} non-empty line(s)): {summary_path}")
        print("Re-run temp sweep or ensure at least one temperature finished and was logged.")
        return

    # Load data
    try:
        data = np.genfromtxt(summary_path, delimiter=',', names=True)
    except Exception as e:
        print(f"Failed to parse {summary_path}: {e}")
        print("If the sweep was interrupted mid-write, delete the file and re-run the sweep.")
        return

    if getattr(data, 'size', 0) == 0:
        print(f"No numeric rows found in {summary_path}")
        return

    # Ensure increasing temperature order
    try:
        order = np.argsort(data['temperature'])
        T = data['temperature'][order]
        S = data['average_S'][order]
        F = data['final_energy'][order]
    except Exception as e:
        print(f"summary.dat parsed but missing expected columns: {e}")
        print("Expected header: temperature,final_energy,average_S")
        return

    # Plot Average S vs Temperature
    plt.figure(figsize=(8, 5))
    plt.plot(T, S, 'o-', label='Average $S$')
    plt.xlabel('$T$ [K]')
    plt.ylabel('$S$')
    plt.title('$S(T)$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(os.path.normpath(os.path.dirname(summary_path) or '.'))
    out1 = os.path.join(out_dir, f'average_S_vs_T_{tag}.png')
    plt.savefig(out1)
    print(f"Saved plot -> {out1}")
    if show:
        plt.show()
    else:
        plt.close()

    # Plot Free Energy vs Temperature
    plt.figure(figsize=(8, 5))
    plt.plot(T, F, 's-', color='red', label='Free Energy')
    plt.xlabel('$T$ [K]')
    plt.ylabel('$F$')
    plt.title('$F(T)$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out2 = os.path.join(out_dir, f'free_energy_vs_T_{tag}.png')
    plt.savefig(out2)
    print(f"Saved plot -> {out2}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_quench_energy_vs_iteration(
    path: str = 'output_quench',
    out_dir: str = 'pics',
    show: bool = True,
    *,
    it_min: int | None = None,
    it_max: int | None = None,
):
    """Plot quench energy components vs iteration (separate, cleaner figure)."""
    data, log_path = load_quench_log(path)
    it = np.atleast_1d(data['iteration']).astype(float)
    bulk = np.atleast_1d(data['bulk']).astype(float)
    elastic = np.atleast_1d(data['elastic']).astype(float)
    total = np.atleast_1d(data['total']).astype(float)

    m = np.isfinite(it) & np.isfinite(bulk) & np.isfinite(elastic) & np.isfinite(total)
    if not np.any(m):
        raise ValueError(f"Quench log contains no finite energy rows: {log_path}")
    it, bulk, elastic, total = it[m], bulk[m], elastic[m], total[m]

    it, bulk, elastic, total = _apply_iter_window(it, bulk, elastic, total, it_min=it_min, it_max=it_max)
    if it.size < 2:
        raise ValueError("Selected iteration window has <2 points to plot.")

    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(os.path.normpath(os.path.dirname(log_path) or '.'))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(it, total, '-', lw=2, label='total')
    ax.plot(it, bulk, '--', lw=1.5, label='bulk')
    ax.plot(it, elastic, '--', lw=1.5, label='elastic')
    ax.set_xlabel('iteration')
    ax.set_ylabel('Energy')
    win = ''
    if it_min is not None or it_max is not None:
        a = str(it_min) if it_min is not None else ''
        b = str(it_max) if it_max is not None else ''
        win = f' [{a}:{b}]'
    ax.set_title(f'Quench energies vs iteration ({tag}){win}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    suffix = ''
    if it_min is not None or it_max is not None:
        a = str(it_min) if it_min is not None else ''
        b = str(it_max) if it_max is not None else ''
        suffix = f'_iter_{a}_{b}'
    out_path = os.path.join(out_dir, f'quench_energies_vs_iteration_{tag}{suffix}.png')
    fig.savefig(out_path, dpi=200)
    print(f"Saved quench energy plot -> {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def plot_quench_order_vs_iteration(
    path: str = 'output_quench',
    out_dir: str = 'pics',
    show: bool = True,
    *,
    it_min: int | None = None,
    it_max: int | None = None,
):
    """Plot quench average order parameter <S> vs iteration (separate, cleaner figure)."""
    data, log_path = load_quench_log(path)
    it = np.atleast_1d(data['iteration']).astype(float)
    avgS = np.atleast_1d(data['avg_S']).astype(float)

    m = np.isfinite(it) & np.isfinite(avgS)
    if not np.any(m):
        raise ValueError(f"Quench log contains no finite avg_S rows: {log_path}")
    it, avgS = it[m], avgS[m]

    it, avgS = _apply_iter_window(it, avgS, it_min=it_min, it_max=it_max)
    if it.size < 2:
        raise ValueError("Selected iteration window has <2 points to plot.")

    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(os.path.normpath(os.path.dirname(log_path) or '.'))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(it, avgS, '-', lw=2)
    ax.set_xlabel('iteration')
    ax.set_ylabel('<S> (droplet)')
    win = ''
    if it_min is not None or it_max is not None:
        a = str(it_min) if it_min is not None else ''
        b = str(it_max) if it_max is not None else ''
        win = f' [{a}:{b}]'
    ax.set_title(f'Average order parameter vs iteration ({tag}){win}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    suffix = ''
    if it_min is not None or it_max is not None:
        a = str(it_min) if it_min is not None else ''
        b = str(it_max) if it_max is not None else ''
        suffix = f'_iter_{a}_{b}'
    out_path = os.path.join(out_dir, f'quench_avgS_vs_iteration_{tag}{suffix}.png')
    fig.savefig(out_path, dpi=200)
    print(f"Saved quench <S> plot -> {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def plot_quench_energy_deltas(
    path: str = 'output_quench',
    out_dir: str = 'pics',
    show: bool = True,
    *,
    it_min: int | None = None,
    it_max: int | None = None,
    top_k: int = 6,
):
    """Plot per-log-step energy changes to detect short energetic events.

    Why useful:
    - Defect annihilation can be highly localized/fast, so the *absolute* total energy
      curve may look smooth while the *step-to-step change* (ΔE) shows sharp spikes.
    - The signal typically appears strongest in the elastic part.

    Produces a figure with:
    - Signed ΔE (symlog) vs iteration
    - |ΔE| (log) vs iteration (if possible)
    """
    data, log_path = load_quench_log(path)
    it = np.atleast_1d(data['iteration']).astype(float)
    bulk = np.atleast_1d(data['bulk']).astype(float)
    elastic = np.atleast_1d(data['elastic']).astype(float)
    total = np.atleast_1d(data['total']).astype(float)

    m = np.isfinite(it) & np.isfinite(bulk) & np.isfinite(elastic) & np.isfinite(total)
    if not np.any(m):
        raise ValueError(f"Quench log contains no finite energy rows: {log_path}")
    it, bulk, elastic, total = it[m], bulk[m], elastic[m], total[m]

    it, bulk, elastic, total = _apply_iter_window(it, bulk, elastic, total, it_min=it_min, it_max=it_max)
    if it.size < 3:
        raise ValueError("Selected iteration window has <3 points; need at least 2 deltas.")

    d_total = np.diff(total)
    d_bulk = np.diff(bulk)
    d_elastic = np.diff(elastic)
    it_d = it[1:]

    # Print candidate "events" (largest step-to-step changes) to help locate annihilation.
    # These are good first guesses to inspect snapshots around (iter-Δ, iter).
    def _print_top(label: str, itx: np.ndarray, dx: np.ndarray, k: int):
        k = int(k)
        if k <= 0:
            return
        a = np.abs(dx)
        m2 = np.isfinite(itx) & np.isfinite(a)
        if np.count_nonzero(m2) == 0:
            return
        itx2 = itx[m2]
        dx2 = dx[m2]
        a2 = a[m2]
        k2 = min(k, a2.size)
        idxs = np.argpartition(a2, -k2)[-k2:]
        idxs = idxs[np.argsort(a2[idxs])[::-1]]
        print(f"Top {k2} |Δ{label}| candidates (iteration, Δ{label}):")
        for j in idxs:
            print(f"  iter={int(itx2[j])}  Δ{label}={dx2[j]:.6g}")

    print("[ΔE detector] Largest step changes are good candidates for fast/local events (e.g., defect annihilation).")
    _print_top('elastic', it_d, d_elastic, top_k)
    _print_top('total', it_d, d_total, max(3, int(top_k // 2)))

    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(os.path.normpath(os.path.dirname(log_path) or '.'))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Signed deltas: symlog keeps sign while showing spikes
    ax1.plot(it_d, d_total, '-', lw=1.8, label='Δ total')
    ax1.plot(it_d, d_bulk, '--', lw=1.4, label='Δ bulk')
    ax1.plot(it_d, d_elastic, '--', lw=1.4, label='Δ elastic')
    ax1.axhline(0.0, color='k', lw=0.8, alpha=0.5)
    ax1.set_ylabel('ΔE')
    ax1.set_yscale('symlog', linthresh=max(1e-16, float(np.nanmedian(np.abs(d_total))) if np.isfinite(np.nanmedian(np.abs(d_total))) else 1e-12))
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc='best')

    # Absolute deltas: log scale if there are positive finite values
    a_total = np.abs(d_total)
    a_bulk = np.abs(d_bulk)
    a_elastic = np.abs(d_elastic)
    ax2.plot(it_d, a_total, '-', lw=1.8, label='|Δ total|')
    ax2.plot(it_d, a_bulk, '--', lw=1.4, label='|Δ bulk|')
    ax2.plot(it_d, a_elastic, '--', lw=1.4, label='|Δ elastic|')
    ax2.set_xlabel('iteration')
    ax2.set_ylabel('|ΔE|')
    ax2.grid(True, alpha=0.25)

    any_pos = np.any(np.isfinite(a_total) & (a_total > 0)) or np.any(np.isfinite(a_elastic) & (a_elastic > 0))
    if any_pos:
        ax2.set_yscale('log')
        # Set safe limits to avoid ticker overflow
        vals = np.concatenate([
            a_total[np.isfinite(a_total) & (a_total > 0)],
            a_bulk[np.isfinite(a_bulk) & (a_bulk > 0)],
            a_elastic[np.isfinite(a_elastic) & (a_elastic > 0)],
        ])
        if vals.size:
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
            if np.isfinite(vmin) and np.isfinite(vmax) and vmin > 0 and vmax > 0:
                ax2.set_ylim(vmin * 0.8, vmax * 1.2)

    win = ''
    if it_min is not None or it_max is not None:
        a = str(it_min) if it_min is not None else ''
        b = str(it_max) if it_max is not None else ''
        win = f' [{a}:{b}]'
    fig.suptitle(
        f"Defect fusion detection\n"
        f"Run: {tag}{win}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    suffix = ''
    if it_min is not None or it_max is not None:
        a = str(it_min) if it_min is not None else ''
        b = str(it_max) if it_max is not None else ''
        suffix = f'_iter_{a}_{b}'
    out_path = os.path.join(out_dir, f'quench_energy_deltas_{tag}{suffix}.png')
    fig.savefig(out_path, dpi=200)
    print(f"Saved quench energy-deltas plot -> {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def _resolve_quench_log_path(path: str) -> str:
    """Resolve a quench log file path.

    Accepts either a file path or a directory (e.g. 'output_quench').
    Prefers quench_log.dat, then quench_log.csv.
    """
    if not path:
        path = 'output_quench'

    if os.path.isdir(path):
        cand_dat = os.path.join(path, 'quench_log.dat')
        cand_csv = os.path.join(path, 'quench_log.csv')
        if os.path.exists(cand_dat):
            return cand_dat
        if os.path.exists(cand_csv):
            return cand_csv

    return path


def load_quench_log(path: str = 'output_quench'):
    """Load quench log produced by QSR_cuda/QSR_cpu.

    Expected header columns (comma-separated):
      iteration,time_s,T_K,bulk,elastic,total,radiality,avg_S
    """
    log_path = _resolve_quench_log_path(path)
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Quench log not found: {log_path}")

    data = np.genfromtxt(log_path, delimiter=',', names=True, dtype=float, encoding='utf-8')
    if data.size == 0:
        raise ValueError(f"Quench log is empty or unreadable: {log_path}")
    return data, log_path


def plot_quench_log(
    path: str = 'output_quench',
    out_dir: str = 'pics',
    show: bool = True,
    *,
    it_min: int | None = None,
    it_max: int | None = None,
):
    """Plot quench diagnostics: T(t), energy components, total energy, radiality, and <S>.

    Saves a summary figure to pics/quench_summary.png (and also returns (fig, axes)).
    """
    data, log_path = load_quench_log(path)

    # Support both scalar row and vector
    it = np.atleast_1d(data['iteration']).astype(float)
    t = np.atleast_1d(data['time_s']).astype(float)
    T = np.atleast_1d(data['T_K']).astype(float)
    bulk = np.atleast_1d(data['bulk']).astype(float)
    elastic = np.atleast_1d(data['elastic']).astype(float)
    total = np.atleast_1d(data['total']).astype(float)
    radial = np.atleast_1d(data['radiality']).astype(float)
    avgS = np.atleast_1d(data['avg_S']).astype(float)

    # Drop invalid rows early to avoid matplotlib inf/nan axis limits
    m = (
        np.isfinite(it)
        & np.isfinite(t)
        & np.isfinite(T)
        & np.isfinite(bulk)
        & np.isfinite(elastic)
        & np.isfinite(total)
        & np.isfinite(radial)
        & np.isfinite(avgS)
    )
    if not np.any(m):
        raise ValueError(f"Quench log contains no finite rows: {log_path}")
    it, t, T = it[m], t[m], T[m]
    bulk, elastic, total = bulk[m], elastic[m], total[m]
    radial, avgS = radial[m], avgS[m]

    it, t, T, bulk, elastic, total, radial, avgS = _apply_iter_window(
        it, t, T, bulk, elastic, total, radial, avgS, it_min=it_min, it_max=it_max
    )
    if it.size < 2:
        raise ValueError("Selected iteration window has <2 points to plot.")

    # Relative energy change between consecutive log points
    rel_dF = np.full_like(total, np.nan)
    if total.size >= 2:
        denom = np.where(np.abs(total[:-1]) > 0.0, total[:-1], np.nan)
        rel_dF[1:] = np.abs((total[1:] - total[:-1]) / denom)
        # Ensure finite, strictly-positive values for log-scale plotting
        rel_dF[~np.isfinite(rel_dF)] = np.nan
        rel_dF[rel_dF <= 0.0] = np.nan

    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axT, axE, axR, axS = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # T(t)
    axT.plot(t, T, '-', lw=2)
    axT.set_xlabel('time [s]')
    axT.set_ylabel('T [K]')
    axT.set_title('Temperature protocol')
    axT.grid(True, alpha=0.3)

    # Energies
    axE.plot(t, total, '-', lw=2, label='total')
    axE.plot(t, bulk, '--', lw=1.5, label='bulk')
    axE.plot(t, elastic, '--', lw=1.5, label='elastic')
    axE.set_xlabel('time [s]')
    axE.set_ylabel('Energy')
    axE.set_title('Energy vs time')
    axE.grid(True, alpha=0.3)
    axE.legend()

    # Radiality and rel dF
    axR.plot(t, radial, '-', lw=2, label='R̄ (radiality)')
    axR.set_xlabel('time [s]')
    axR.set_ylabel('R̄')
    axR.set_ylim(0.0, 1.01)
    axR.grid(True, alpha=0.3)
    axR.set_title('Radiality')
    axR.legend(loc='lower right')

    axR2 = axR.twinx()
    if np.any(np.isfinite(rel_dF)):
        axR2.plot(t, rel_dF, ':', lw=1.8, color='tab:red', label='|ΔF/F|')
        axR2.set_ylabel('|ΔF/F|')
        axR2.set_yscale('log')
        # Constrain y-limits to avoid ticker overflow for extreme/degenerate data
        y_min = float(np.nanmin(rel_dF))
        y_max = float(np.nanmax(rel_dF))
        if np.isfinite(y_min) and np.isfinite(y_max) and y_min > 0.0 and y_max > 0.0:
            axR2.set_ylim(y_min * 0.8, y_max * 1.2)
    else:
        # No valid positive data to show on a log scale
        axR2.set_ylabel('|ΔF/F| (n/a)')
        axR2.set_yticks([])
    axR2.grid(False)

    # Average S
    axS.plot(t, avgS, '-', lw=2)
    axS.set_xlabel('time [s]')
    axS.set_ylabel('<S> (droplet)')
    axS.set_title('Average order parameter')
    axS.grid(True, alpha=0.3)

    base = os.path.basename(log_path)
    win = ''
    if it_min is not None or it_max is not None:
        a = str(it_min) if it_min is not None else ''
        b = str(it_max) if it_max is not None else ''
        win = f"  [iter {a}:{b}]"
    fig.suptitle(f"Quench log: {base}{win}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    suffix = ''
    if it_min is not None or it_max is not None:
        a = str(it_min) if it_min is not None else ''
        b = str(it_max) if it_max is not None else ''
        suffix = f'_iter_{a}_{b}'
    out_path = os.path.join(out_dir, f'quench_summary{suffix}.png')
    fig.savefig(out_path, dpi=200)
    print(f"Saved quench summary plot -> {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes


def _parse_temperature_from_dir(d):
    # Accept paths like output_temp_sweep/T_300.000000/
    base = os.path.basename(os.path.normpath(d))
    m = re.match(r"T_([+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?)$", base)
    if not m:
        return float('nan')
    try:
        return float(m.group(1))
    except ValueError:
        return float('nan')


def _load_nematic_slice_arrays(path, Nx, Ny, Nz, z_slice):
    data = load_nematic_field_data(path, comments="#")
    slice_data = data[data[:, 2] == z_slice]
    if slice_data.shape[0] == 0:
        raise ValueError(f"No data found for z_slice={z_slice} in {path}")

    S = np.zeros((Nx, Ny), dtype=float)
    nx = np.zeros((Nx, Ny), dtype=float)
    ny = np.zeros((Nx, Ny), dtype=float)
    nz = np.zeros((Nx, Ny), dtype=float)
    for row in slice_data:
        i, j = int(row[0]), int(row[1])
        if 0 <= i < Nx and 0 <= j < Ny:
            S[i, j] = row[3]
            nx[i, j] = row[4]
            ny[i, j] = row[5]
            nz[i, j] = row[6]

    return S, nx, ny, nz


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def defect_density_2d_from_slice(
    S: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    *,
    S_threshold: float = 0.1,
    charge_cutoff: float = 0.25,
):
    """Compute a simple 2D nematic defect-density proxy on a slice.

    Uses the doubled-angle field psi = 2*theta, theta=atan2(ny,nx) to respect n~ -n.
    Computes the winding on each plaquette and counts |s|>charge_cutoff.

    Returns:
      density: defects per plaquette (lattice units)
      s_map: topological charge per plaquette (shape (Nx-1,Ny-1))

    Notes:
    - This is a 2D proxy (mid-plane slice). For full 3D line-defects, you'd need 3D analysis.
    """
    S = np.asarray(S)
    nx = np.asarray(nx)
    ny = np.asarray(ny)
    if S.shape != nx.shape or S.shape != ny.shape:
        raise ValueError("S,nx,ny must have the same shape")

    mask = (S > float(S_threshold)) & np.isfinite(S) & np.isfinite(nx) & np.isfinite(ny)
    theta = np.arctan2(ny, nx)
    psi = 2.0 * theta

    # Plaquette corners
    p00 = psi[:-1, :-1]
    p10 = psi[1:, :-1]
    p11 = psi[1:, 1:]
    p01 = psi[:-1, 1:]

    # Wrapped increments around plaquette
    d1 = _wrap_to_pi(p10 - p00)
    d2 = _wrap_to_pi(p11 - p10)
    d3 = _wrap_to_pi(p01 - p11)
    d4 = _wrap_to_pi(p00 - p01)
    dsum = d1 + d2 + d3 + d4

    w = dsum / (2.0 * np.pi)  # winding of psi
    s_map = 0.5 * w  # nematic charge

    plaq_mask = mask[:-1, :-1] & mask[1:, :-1] & mask[1:, 1:] & mask[:-1, 1:]
    s_map = np.where(plaq_mask, s_map, 0.0)

    defects = np.abs(s_map) > float(charge_cutoff)
    denom = int(np.count_nonzero(plaq_mask))
    density = (float(np.count_nonzero(defects)) / float(denom)) if denom > 0 else 0.0
    return density, s_map


def correlation_length_2d_from_slice(
    S: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    *,
    S_threshold: float = 0.1,
    max_r: int | None = None,
    target: float = np.e ** (-1.0),
):
    """Estimate a 2D correlation length xi from a nematic slice.

    Constructs complex order field q = exp(i*psi) with psi=2*atan2(ny,nx) and computes
    a masked autocorrelation via FFT. Returns xi in lattice units as first r where C(r)<target.
    """
    S = np.asarray(S)
    nx = np.asarray(nx)
    ny = np.asarray(ny)
    if S.shape != nx.shape or S.shape != ny.shape:
        raise ValueError("S,nx,ny must have the same shape")

    mask = (S > float(S_threshold)) & np.isfinite(S) & np.isfinite(nx) & np.isfinite(ny)
    if np.count_nonzero(mask) < 16:
        return float('nan'), np.array([]), np.array([])

    theta = np.arctan2(ny, nx)
    psi = 2.0 * theta
    q = np.exp(1j * psi) * mask

    # Mask-normalized autocorrelation using FFT
    Fq = np.fft.fft2(q)
    corr = np.fft.ifft2(Fq * np.conj(Fq)).real
    Fm = np.fft.fft2(mask.astype(float))
    norm = np.fft.ifft2(Fm * np.conj(Fm)).real

    with np.errstate(divide='ignore', invalid='ignore'):
        C = np.where(norm > 0, corr / norm, 0.0)

    # Shift zero-lag to center for radial averaging
    C = np.fft.fftshift(C)
    cx = C.shape[0] // 2
    cy = C.shape[1] // 2
    C0 = float(C[cx, cy])
    if not np.isfinite(C0) or C0 == 0.0:
        return float('nan'), np.array([]), np.array([])
    C = C / C0

    nxp, nyp = C.shape
    yy, xx = np.ogrid[:nxp, :nyp]
    rr = np.sqrt((yy - cx) ** 2 + (xx - cy) ** 2)
    r_int = rr.astype(np.int32)

    if max_r is None:
        max_r = int(min(nxp, nyp) // 2)
    max_r = int(max(2, min(max_r, r_int.max())))

    # radial average
    flat_r = r_int.ravel()
    flat_C = C.ravel()
    valid = (flat_r >= 0) & (flat_r <= max_r) & np.isfinite(flat_C)
    flat_r = flat_r[valid]
    flat_C = flat_C[valid]

    sums = np.bincount(flat_r, weights=flat_C, minlength=max_r + 1)
    counts = np.bincount(flat_r, minlength=max_r + 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        C_r = np.where(counts > 0, sums / counts, np.nan)
    r = np.arange(max_r + 1, dtype=float)

    # Find xi where correlation drops below target (sub-lattice via linear interpolation)
    xi = float('nan')
    targ = float(target)
    for ri in range(1, max_r + 1):
        if np.isfinite(C_r[ri]) and (C_r[ri] <= targ):
            # interpolate between (ri-1, C_r[ri-1]) and (ri, C_r[ri])
            c0 = float(C_r[ri - 1])
            c1 = float(C_r[ri])
            if np.isfinite(c0) and np.isfinite(c1) and (c1 != c0):
                frac = (targ - c0) / (c1 - c0)
                xi = float((ri - 1) + frac)
            else:
                xi = float(ri)
            break

    return xi, r, C_r


def _qtensor_from_Sn(
    S: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct a symmetric traceless Q-tensor from (S, n).

    Uses uniaxial form: Q_ij = S (n_i n_j - δ_ij/3).
    Invariant under n -> -n.
    """
    S = np.asarray(S)
    nx = np.asarray(nx)
    ny = np.asarray(ny)
    nz = np.asarray(nz)
    one_third = 1.0 / 3.0
    Qxx = S * (nx * nx - one_third)
    Qyy = S * (ny * ny - one_third)
    Qzz = S * (nz * nz - one_third)
    Qxy = S * (nx * ny)
    Qxz = S * (nx * nz)
    Qyz = S * (ny * nz)
    return Qxx, Qxy, Qxz, Qyy, Qyz, Qzz


def _radial_average_nd(C: np.ndarray, *, max_r: int | None = None):
    """Radial average of an nD correlation volume around its center."""
    C = np.asarray(C, dtype=float)
    if C.ndim < 2:
        return np.array([]), np.array([])

    center = tuple(s // 2 for s in C.shape)
    grids = np.ogrid[tuple(slice(0, s) for s in C.shape)]
    rr2: np.ndarray = np.zeros(C.shape, dtype=float)
    for ax, g in enumerate(grids):
        d = (g - center[ax]).astype(float)
        rr2 = rr2 + d * d
    rr = np.sqrt(rr2)
    r_int = rr.astype(np.int32)

    if max_r is None:
        max_r = int(min(C.shape) // 2)
    max_r = int(max(2, min(int(max_r), int(r_int.max()))))

    flat_r = r_int.ravel()
    flat_C = C.ravel()
    valid = (flat_r >= 0) & (flat_r <= max_r) & np.isfinite(flat_C)
    flat_r = flat_r[valid]
    flat_C = flat_C[valid]

    sums = np.bincount(flat_r, weights=flat_C, minlength=max_r + 1)
    counts = np.bincount(flat_r, minlength=max_r + 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        C_r = np.where(counts > 0, sums / counts, np.nan)
    r = np.arange(max_r + 1, dtype=float)
    return r, C_r


def correlation_length_3d_from_qtensor(
    Qxx: np.ndarray,
    Qxy: np.ndarray,
    Qxz: np.ndarray,
    Qyy: np.ndarray,
    Qyz: np.ndarray,
    Qzz: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    max_r: int | None = None,
    target: float = np.e ** (-1.0),
):
    """Estimate a 3D correlation length xi from a Q-tensor field.

    Computes a masked autocorrelation of the Frobenius inner product:
        C(r) ~ < Q_ij(x) Q_ij(x+r) >_mask / < Q_ij(x)^2 >
    using FFTs. Returns xi in lattice units as first r where C(r) < target.
    """
    Qxx = np.asarray(Qxx, dtype=float)
    Qxy = np.asarray(Qxy, dtype=float)
    Qxz = np.asarray(Qxz, dtype=float)
    Qyy = np.asarray(Qyy, dtype=float)
    Qyz = np.asarray(Qyz, dtype=float)
    Qzz = np.asarray(Qzz, dtype=float)
    if not (Qxx.shape == Qxy.shape == Qxz.shape == Qyy.shape == Qyz.shape == Qzz.shape):
        raise ValueError("All Q components must have the same shape")

    if mask is None:
        mask = np.isfinite(Qxx) & np.isfinite(Qyy) & np.isfinite(Qzz)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != Qxx.shape:
        raise ValueError("mask must have the same shape as Q components")
    if np.count_nonzero(mask) < 64:
        return float('nan'), np.array([]), np.array([])

    mask_f = mask.astype(float)

    # mask-normalized correlation via FFT
    Fm = np.fft.fftn(mask_f)
    norm = np.fft.ifftn(Fm * np.conj(Fm)).real

    corr_sum = 0.0
    for comp in (Qxx, Qyy, Qzz, Qxy, Qxz, Qyz):
        a = np.where(mask, comp, 0.0) * mask_f
        Fa = np.fft.fftn(a)
        corr_sum = corr_sum + np.fft.ifftn(Fa * np.conj(Fa)).real

    with np.errstate(divide='ignore', invalid='ignore'):
        C = np.where(norm > 0, corr_sum / norm, 0.0)

    C = np.fft.fftshift(C)
    center = tuple(s // 2 for s in C.shape)
    C0 = float(C[center])
    if not np.isfinite(C0) or C0 == 0.0:
        return float('nan'), np.array([]), np.array([])
    C = C / C0

    r, C_r = _radial_average_nd(C, max_r=max_r)
    if r.size == 0:
        return float('nan'), r, C_r

    xi = float('nan')
    targ = float(target)
    for ri in range(1, int(r.size)):
        if np.isfinite(C_r[ri]) and (C_r[ri] <= targ):
            c0 = float(C_r[ri - 1])
            c1 = float(C_r[ri])
            if np.isfinite(c0) and np.isfinite(c1) and (c1 != c0):
                frac = (targ - c0) / (c1 - c0)
                xi = float((ri - 1) + frac)
            else:
                xi = float(ri)
            break

    return xi, r, C_r


def load_nematic_field_volume(path: str, Nx: int, Ny: int, Nz: int):
    """Load full 3D arrays (S,nx,ny,nz) from nematic_field_*.dat."""
    data = load_nematic_field_data(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(f"Unexpected nematic field format in {path}")

    S = np.zeros((Nx, Ny, Nz), dtype=float)
    nx = np.zeros((Nx, Ny, Nz), dtype=float)
    ny = np.zeros((Nx, Ny, Nz), dtype=float)
    nz = np.zeros((Nx, Ny, Nz), dtype=float)

    # Vectorized fill (much faster than Python loop for ~1e6 rows)
    ii = data[:, 0].astype(np.int32, copy=False)
    jj = data[:, 1].astype(np.int32, copy=False)
    kk = data[:, 2].astype(np.int32, copy=False)
    m = (ii >= 0) & (ii < int(Nx)) & (jj >= 0) & (jj < int(Ny)) & (kk >= 0) & (kk < int(Nz))
    if np.any(m):
        i = ii[m]
        j = jj[m]
        k = kk[m]
        S[i, j, k] = data[m, 3]
        nx[i, j, k] = data[m, 4]
        ny[i, j, k] = data[m, 5]
        nz[i, j, k] = data[m, 6]
    return S, nx, ny, nz


def load_qtensor_volume(path: str, Nx: int, Ny: int, Nz: int):
    """Load full 3D arrays of Q components from Qtensor_output_*.dat."""
    data = load_qtensor_data(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 12:
        raise ValueError(f"Unexpected Qtensor format in {path}")

    Qxx = np.zeros((Nx, Ny, Nz), dtype=float)
    Qxy = np.zeros((Nx, Ny, Nz), dtype=float)
    Qxz = np.zeros((Nx, Ny, Nz), dtype=float)
    Qyy = np.zeros((Nx, Ny, Nz), dtype=float)
    Qyz = np.zeros((Nx, Ny, Nz), dtype=float)
    Qzz = np.zeros((Nx, Ny, Nz), dtype=float)

    # Vectorized fill
    ii = data[:, 0].astype(np.int32, copy=False)
    jj = data[:, 1].astype(np.int32, copy=False)
    kk = data[:, 2].astype(np.int32, copy=False)
    m = (ii >= 0) & (ii < int(Nx)) & (jj >= 0) & (jj < int(Ny)) & (kk >= 0) & (kk < int(Nz))
    if np.any(m):
        i = ii[m]
        j = jj[m]
        k = kk[m]
        Qxx[i, j, k] = data[m, 3]
        Qxy[i, j, k] = 0.5 * (data[m, 4] + data[m, 6])
        Qxz[i, j, k] = 0.5 * (data[m, 5] + data[m, 9])
        Qyy[i, j, k] = data[m, 7]
        Qyz[i, j, k] = 0.5 * (data[m, 8] + data[m, 10])
        Qzz[i, j, k] = data[m, 11]

    return Qxx, Qxy, Qxz, Qyy, Qyz, Qzz


def _infer_num_columns_from_text_file(path: str) -> int:
    """Infer number of whitespace-delimited columns from the first data row."""
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line:
                continue
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            return len(s.split())
    return 0


def correlation_length_3d_from_field_file(
    path: str,
    *,
    S_threshold: float = 0.1,
    max_r: int | None = None,
    target: float = np.e ** (-1.0),
):
    """Compute 3D xi from either nematic_field_*.dat or Qtensor_output_*.dat."""
    Nx, Ny, Nz = infer_grid_dims_from_nematic_field_file(path)

    ncol = int(_infer_num_columns_from_text_file(path))
    if ncol <= 0:
        raise ValueError(f"Could not infer columns for {path}")

    if ncol >= 12:
        Qxx, Qxy, Qxz, Qyy, Qyz, Qzz = load_qtensor_volume(path, Nx, Ny, Nz)
        # approximate S magnitude for masking
        trQ2 = Qxx * Qxx + Qyy * Qyy + Qzz * Qzz + 2.0 * (Qxy * Qxy + Qxz * Qxz + Qyz * Qyz)
        S_mag = np.sqrt(np.maximum(0.0, 1.5 * trQ2))
        mask = (S_mag > float(S_threshold)) & np.isfinite(S_mag)
        xi, r, C_r = correlation_length_3d_from_qtensor(
            Qxx, Qxy, Qxz, Qyy, Qyz, Qzz,
            mask=mask,
            max_r=max_r,
            target=target,
        )
        return xi, r, C_r, (Nx, Ny, Nz)

    if ncol >= 7:
        S, nx, ny, nz = load_nematic_field_volume(path, Nx, Ny, Nz)
        Qxx, Qxy, Qxz, Qyy, Qyz, Qzz = _qtensor_from_Sn(S, nx, ny, nz)
        mask = (S > float(S_threshold)) & np.isfinite(S)
        xi, r, C_r = correlation_length_3d_from_qtensor(
            Qxx, Qxy, Qxz, Qyy, Qyz, Qzz,
            mask=mask,
            max_r=max_r,
            target=target,
        )
        return xi, r, C_r, (Nx, Ny, Nz)

    raise ValueError(f"Unsupported field file format in {path} (ncol={ncol})")


def _largest_connected_component_3d(mask: np.ndarray) -> np.ndarray:
    if ndi is None:
        raise ImportError("scipy is required for 3D connected components (install scipy)")
    ndi_local = ndi
    assert ndi_local is not None
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("mask must be 3D")
    if np.count_nonzero(mask) == 0:
        return np.zeros_like(mask, dtype=bool)
    structure = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity
    lbl, n = ndi_local.label(mask, structure=structure)
    if n <= 1:
        return mask
    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    lab = int(np.argmax(counts))
    return lbl == lab


def _skeleton_length_from_bool(skel: np.ndarray) -> float:
    """Approximate skeleton length in lattice units using 26-neighbor edges."""
    skel = np.asarray(skel, dtype=bool)
    if skel.ndim != 3:
        raise ValueError("skel must be 3D")
    if np.count_nonzero(skel) == 0:
        return 0.0

    def _count_pairs(dx: int, dy: int, dz: int) -> int:
        # Count (a & shifted a) without wrap-around.
        xs0 = slice(max(0, dx), skel.shape[0] + min(0, dx))
        ys0 = slice(max(0, dy), skel.shape[1] + min(0, dy))
        zs0 = slice(max(0, dz), skel.shape[2] + min(0, dz))
        xs1 = slice(max(0, -dx), skel.shape[0] + min(0, -dx))
        ys1 = slice(max(0, -dy), skel.shape[1] + min(0, -dy))
        zs1 = slice(max(0, -dz), skel.shape[2] + min(0, -dz))
        a = skel[xs0, ys0, zs0]
        b = skel[xs1, ys1, zs1]
        return int(np.count_nonzero(a & b))

    length = 0.0
    # Use a half-neighborhood to avoid double counting.
    for dx in (0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                # half-space rule
                if dx == 0 and dy < 0:
                    continue
                if dx == 0 and dy == 0 and dz < 0:
                    continue
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                w = float(np.sqrt(dx * dx + dy * dy + dz * dz))
                if w <= 0:
                    continue
                length += w * float(_count_pairs(dx, dy, dz))
    return float(length)


def defect_line_metrics_3d_from_field_file(
    path: str,
    *,
    S_droplet: float = 0.1,
    S_core: float = 0.05,
    dilate_iters: int = 2,
    fill_holes: bool = False,
    core_erosion_iters: int = 0,
    min_core_voxels: int = 30,
    use_skeleton: bool = True,
):
    """Estimate 3D defect-line content from a field file.

    Current implementation is a pragmatic LdG-style proxy:
    - identify droplet as largest connected component of S > S_droplet
    - dilate droplet a bit to include adjacent low-S core voxels
    - define core candidates as (dilated droplet) & (S_mag < S_core)
    - optionally skeletonize core voxels and estimate total line length

    Works for:
      - nematic_field_*.dat (uses S directly)
      - Qtensor_output_*.dat (uses S_mag inferred from tr(Q^2))
    """
    if ndi is None:
        raise ImportError("scipy is required for 3D defect-line metrics (install scipy)")
    if use_skeleton and skeletonize_3d is None:
        raise ImportError("scikit-image is required for 3D skeletonization (install scikit-image)")
    ndi_local = ndi
    assert ndi_local is not None
    skel_fn = skeletonize_3d
    if use_skeleton:
        assert skel_fn is not None

    Nx, Ny, Nz = infer_grid_dims_from_nematic_field_file(path)
    ncol = int(_infer_num_columns_from_text_file(path))
    if ncol >= 12:
        Qxx, Qxy, Qxz, Qyy, Qyz, Qzz = load_qtensor_volume(path, Nx, Ny, Nz)
        trQ2 = Qxx * Qxx + Qyy * Qyy + Qzz * Qzz + 2.0 * (Qxy * Qxy + Qxz * Qxz + Qyz * Qyz)
        S_mag = np.sqrt(np.maximum(0.0, 1.5 * trQ2))
        S_use = S_mag
    elif ncol >= 7:
        S, _, _, _ = load_nematic_field_volume(path, Nx, Ny, Nz)
        S_use = S
    else:
        raise ValueError(f"Unsupported field file format in {path} (ncol={ncol})")

    S_use = np.asarray(S_use, dtype=float)
    finite = np.isfinite(S_use)

    droplet_seed = finite & (S_use > float(S_droplet))
    droplet = _largest_connected_component_3d(droplet_seed)
    if bool(fill_holes):
        # Important: defect cores (low-S) can appear as holes inside the droplet.
        # Filling holes lets us detect cores *inside* the droplet without relying on dilation,
        # and reduces the chance of capturing broad interface sheets.
        droplet = ndi_local.binary_fill_holes(droplet)
    droplet_vox = int(np.count_nonzero(droplet))
    if droplet_vox == 0:
        return {
            'Nx': Nx,
            'Ny': Ny,
            'Nz': Nz,
            'droplet_voxels': 0,
            'core_voxels': 0,
            'skeleton_voxels': 0,
            'line_length_lattice': 0.0,
            'line_density_per_voxel': float('nan'),
            'core_density_per_voxel': float('nan'),
        }

    # Define a safe interior region to avoid counting low-S interface voxels.
    interior = droplet
    if int(core_erosion_iters) > 0:
        structure = np.ones((3, 3, 3), dtype=bool)
        interior = ndi_local.binary_erosion(droplet, structure=structure, iterations=int(core_erosion_iters))

    # Optional dilation can help catch slightly offset cores, but large dilation tends to
    # include interface sheets. Keep it small in batch use (0-1).
    if int(dilate_iters) > 0:
        structure = np.ones((3, 3, 3), dtype=bool)
        region = ndi_local.binary_dilation(interior, structure=structure, iterations=int(dilate_iters))
    else:
        region = interior

    core = region & finite & (S_use < float(S_core))
    core_vox = int(np.count_nonzero(core))
    if core_vox == 0:
        return {
            'Nx': Nx,
            'Ny': Ny,
            'Nz': Nz,
            'droplet_voxels': droplet_vox,
            'core_voxels': 0,
            'skeleton_voxels': 0,
            'line_length_lattice': 0.0,
            'line_density_per_voxel': 0.0,
            'core_density_per_voxel': 0.0,
        }

    # Remove tiny noisy components
    structure = np.ones((3, 3, 3), dtype=bool)
    lbl, n = ndi_local.label(core, structure=structure)
    if n > 0:
        counts = np.bincount(lbl.ravel())
        counts[0] = 0
        keep_labels = np.zeros(int(counts.size), dtype=bool)
        keep_labels[counts >= int(min_core_voxels)] = True
        lbl_i = lbl.astype(np.int32, copy=False)
        core = keep_labels[lbl_i]
        core_vox = int(np.count_nonzero(core))

    if use_skeleton and core_vox > 0:
        skel = skel_fn(core, method='lee')
        skel = np.asarray(skel, dtype=bool)
        skel_vox = int(np.count_nonzero(skel))
        length = _skeleton_length_from_bool(skel)
    else:
        skel = np.zeros_like(core, dtype=bool)
        skel_vox = 0
        length = float('nan')

    density = float(length) / float(droplet_vox) if np.isfinite(length) and droplet_vox > 0 else float('nan')
    core_density = float(core_vox) / float(droplet_vox) if droplet_vox > 0 else float('nan')
    return {
        'Nx': Nx,
        'Ny': Ny,
        'Nz': Nz,
        'droplet_voxels': droplet_vox,
        'core_voxels': core_vox,
        'skeleton_voxels': skel_vox,
        'line_length_lattice': float(length) if np.isfinite(length) else float('nan'),
        'line_density_per_voxel': density,
        'core_density_per_voxel': core_density,
        'S_droplet': float(S_droplet),
        'S_core': float(S_core),
        'dilate_iters': int(dilate_iters),
        'min_core_voxels': int(min_core_voxels),
    }


def _nearest_from_log(iter_log: np.ndarray, values: np.ndarray, iter_target: int) -> float:
    """Nearest-neighbor lookup in a log sampled at iter_log."""
    iter_log = np.asarray(iter_log, dtype=float)
    values = np.asarray(values, dtype=float)
    if iter_log.size == 0:
        return float('nan')
    idx = int(np.searchsorted(iter_log, float(iter_target)))
    if idx <= 0:
        return float(values[0])
    if idx >= iter_log.size:
        return float(values[-1])
    # choose closer of idx-1, idx
    if abs(iter_log[idx] - iter_target) < abs(iter_log[idx - 1] - iter_target):
        return float(values[idx])
    return float(values[idx - 1])


def plot_quench_kz_metrics(
    path: str = 'output_quench',
    *,
    out_dir: str = 'pics',
    z_slice: int | None = None,
    frame_stride: int = 10,
    max_frames: int | None = 50,
    S_threshold: float = 0.1,
    show: bool = True,
):
    """Reconstruct KZ-style metrics from quench snapshots: defect density and correlation length.

    This uses 2D mid-plane slice analysis as a proxy:
    - defect density: plaquette winding count (|s|>0.25)
    - xi: first r where C(r) < 1/e
    """
    if not path:
        path = 'output_quench'

    data, log_path = load_quench_log(path)
    it_log = np.atleast_1d(data['iteration']).astype(float)
    t_log = np.atleast_1d(data['time_s']).astype(float)
    T_log = np.atleast_1d(data['T_K']).astype(float)

    data_dir = os.path.dirname(log_path) if os.path.isfile(log_path) else path
    files = glob.glob(os.path.join(data_dir, 'nematic_field_iter_*.dat'))
    if not files:
        names = set(getattr(getattr(data, 'dtype', None), 'names', []) or [])
        if 'defect_density_per_plaquette' in names:
            print(
                "[KZ metrics] No nematic_field_iter_*.dat snapshots found; using defect_density_per_plaquette from quench_log. "
                "(xi will use xi_grad_proxy if available; otherwise NaN.)"
            )
            ndef = np.atleast_1d(data['defect_density_per_plaquette']).astype(float)
            if 'xi_grad_proxy' in names:
                xi = np.atleast_1d(data['xi_grad_proxy']).astype(float)
            else:
                xi = np.full_like(ndef, np.nan, dtype=float)
            rows = np.column_stack((it_log, t_log, T_log, xi, ndef)).astype(float)

            os.makedirs(out_dir, exist_ok=True)
            tag = os.path.basename(os.path.normpath(data_dir)) or 'quench'
            csv_path = os.path.join(out_dir, f'kz_metrics_{tag}.csv')
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write('iteration,time_s,T_K,xi_lattice,defect_density_per_plaquette\n')
                for (ii, tt, TT, xii, nd) in rows:
                    f.write(f"{int(ii)},{tt:.8g},{TT:.8g},{xii:.8g},{nd:.8g}\n")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            ax1.plot(t_log, xi, 'o-', lw=1.5)
            ax1.set_ylabel(r'$\\xi$ (lattice units)')
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f'KZ proxies from {tag} (log-only defect density)')
            if 'xi_grad_proxy' in names:
                ax1.text(0.02, 0.9, 'xi = xi_grad_proxy (gradient proxy)', transform=ax1.transAxes)
            else:
                ax1.text(0.02, 0.9, 'xi unavailable (no snapshots)', transform=ax1.transAxes)

            ax2.plot(t_log, ndef, 'o-', color='tab:red', lw=1.5)
            ax2.set_xlabel('time [s]')
            ax2.set_ylabel('defect density (per plaquette)')
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            out_path = os.path.join(out_dir, f'kz_metrics_{tag}.png')
            fig.savefig(out_path, dpi=200)
            print(f"Saved KZ metrics -> {out_path}")
            print(f"Saved KZ metrics CSV -> {csv_path}")
            if show:
                plt.show()
            else:
                plt.close(fig)
            return rows, out_path, csv_path
        raise FileNotFoundError(f"No nematic_field_iter_*.dat snapshots found in {data_dir}")

    def _iter_num(p: str) -> int:
        m = re.search(r'(\d+)', os.path.basename(p))
        return int(m.group(1)) if m else -1

    files.sort(key=_iter_num)
    if frame_stride is None or frame_stride < 1:
        frame_stride = 1
    files = files[::frame_stride]
    if max_frames is not None and max_frames > 0:
        files = files[:max_frames]

    Nx, Ny, Nz = infer_grid_dims_from_nematic_field_file(files[0])
    if z_slice is None:
        z_slice = Nz // 2

    rows = []
    for fp in files:
        iter_i = _iter_num(fp)
        S2, nx2, ny2, _ = _load_nematic_slice_arrays(fp, Nx, Ny, Nz, z_slice)
        n_def, _ = defect_density_2d_from_slice(S2, nx2, ny2, S_threshold=S_threshold)
        xi, _, _ = correlation_length_2d_from_slice(S2, nx2, ny2, S_threshold=S_threshold)
        t_i = _nearest_from_log(it_log, t_log, iter_i)
        T_i = _nearest_from_log(it_log, T_log, iter_i)
        rows.append((iter_i, t_i, T_i, xi, n_def))

    rows = np.array(rows, dtype=float)
    iters = rows[:, 0]
    t = rows[:, 1]
    T = rows[:, 2]
    xi = rows[:, 3]
    ndef = rows[:, 4]

    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(os.path.normpath(data_dir)) or 'quench'
    csv_path = os.path.join(out_dir, f'kz_metrics_{tag}.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('iteration,time_s,T_K,xi_lattice,defect_density_per_plaquette\n')
        for (ii, tt, TT, xii, nd) in rows:
            f.write(f"{int(ii)},{tt:.8g},{TT:.8g},{xii:.8g},{nd:.8g}\n")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(t, xi, 'o-', lw=1.5)
    ax1.set_ylabel(r'$\xi$ (lattice units)')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'KZ proxies from {tag} (z={z_slice}, stride={frame_stride}, S>{S_threshold})')

    ax2.plot(t, ndef, 'o-', color='tab:red', lw=1.5)
    ax2.set_xlabel('time [s]')
    ax2.set_ylabel('defect density (per plaquette)')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f'kz_metrics_{tag}.png')
    fig.savefig(out_path, dpi=200)
    print(f"Saved KZ metrics -> {out_path}")
    print(f"Saved KZ metrics CSV -> {csv_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return rows, out_path, csv_path


def _infer_dt_from_log(iteration: np.ndarray, time_s: np.ndarray) -> float:
    """Infer dt from logged (iteration, time) pairs."""
    it = np.asarray(iteration, dtype=float)
    t = np.asarray(time_s, dtype=float)
    if it.size < 2 or t.size != it.size:
        return float('nan')
    di = np.diff(it)
    dt = np.diff(t)
    m = (di > 0) & np.isfinite(di) & np.isfinite(dt)
    if np.count_nonzero(m) == 0:
        return float('nan')
    vals = dt[m] / di[m]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return float('nan')
    return float(np.median(vals))


def _infer_trailing_int(name: str) -> int | None:
    m = re.search(r'(\d+)$', name or '')
    return int(m.group(1)) if m else None


def _resolve_run_dir_from_path(path: str) -> str:
    """Resolve a run directory from an input path.

    Accepts either a directory (e.g. output_quench_0) or a file path
    (e.g. output_quench_0/quench_log.dat).
    """
    if not path:
        return 'output_quench'
    if os.path.isdir(path):
        return path
    if os.path.isfile(path):
        d = os.path.dirname(path)
        return d if d else '.'
    return path


def _list_snapshot_files(run_dir: str) -> list[tuple[int, str]]:
    files = glob.glob(os.path.join(run_dir, 'nematic_field_iter_*.dat'))
    out: list[tuple[int, str]] = []
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        it = _infer_trailing_int(stem)
        if it is None:
            continue
        out.append((int(it), fp))
    out.sort(key=lambda x: x[0])
    return out


def prompt_plot_snapshot_slice_from_dir(
    in_path: str,
    *,
    out_dir: str = 'pics',
    default_Tc: float = 310.2,
    show: bool = True,
) -> None:
    """Interactive prompt to plot a slice from a selected quench snapshot.

    - Lists available `nematic_field_iter_*.dat` files.
    - Lets you choose by index/iteration or by time relative to Tc (if log exists).
    - Calls `plot_nematic_field_slice(...)` and saves a PNG in `out_dir`.
    """
    run_dir = _resolve_run_dir_from_path(in_path)
    snaps = _list_snapshot_files(run_dir)
    if not snaps:
        print(f"No nematic_field_iter_*.dat snapshots found in: {run_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)

    iters = [it for it, _ in snaps]
    print(f"Found {len(snaps)} snapshots in {run_dir}")
    print(f"Iteration range: {iters[0]} .. {iters[-1]}")
    if len(iters) >= 2:
        diffs = np.diff(np.asarray(iters, dtype=int))
        diffs = diffs[diffs > 0]
        if diffs.size:
            print(f"Estimated snapshot period (iters): ~{int(np.median(diffs))}")

    # Auto-detect grid from first snapshot
    Nx_use, Ny_use, Nz_use = infer_grid_dims_from_nematic_field_file(snaps[0][1])
    print(f"Detected grid: Nx={Nx_use}, Ny={Ny_use}, Nz={Nz_use} (from {os.path.basename(snaps[0][1])})")

    # Preview list (first/last few)
    n_show = min(8, len(snaps))
    print("Sample snapshots:")
    for idx in range(n_show):
        it, fp = snaps[idx]
        print(f"  [{idx}] iter={it}  file={os.path.basename(fp)}")
    if len(snaps) > n_show:
        print("  ...")
        base = max(0, len(snaps) - n_show)
        for rel, idx in enumerate(range(base, len(snaps))):
            it, fp = snaps[idx]
            print(f"  [{idx}] iter={it}  file={os.path.basename(fp)}")

    # Check if we can select by Tc/time
    have_log = False
    log_path: str = ''
    it_log: np.ndarray = np.empty(0, dtype=float)
    t_log: np.ndarray = np.empty(0, dtype=float)
    T_log: np.ndarray = np.empty(0, dtype=float)
    try:
        data, log_path = load_quench_log(run_dir)
        it_log = np.atleast_1d(data['iteration']).astype(float)
        t_log = np.atleast_1d(data['time_s']).astype(float)
        T_log = np.atleast_1d(data['T_K']).astype(float)
        have_log = True
    except Exception:
        have_log = False

    if have_log:
        default_method = 'a'
        print(f"Detected quench log: {os.path.basename(str(log_path))}")
    else:
        default_method = 'n'

    method = input(
        "Select snapshot by: (n) index, (i) iteration number, (a) after Tc+offset, (q) quit "
        f"[default: {default_method}]: "
    ).strip().lower()
    if not method:
        method = default_method
    if method == 'q':
        return

    chosen_fp = None
    chosen_iter = None
    if method == 'i':
        it_in = input("Iteration to plot (must match an available snapshot iter): ").strip()
        try:
            want_it = int(float(it_in))
        except ValueError:
            print("Invalid iteration.")
            return
        d = {it: fp for it, fp in snaps}
        if want_it not in d:
            print("That iteration is not available as a snapshot.")
            return
        chosen_iter = want_it
        chosen_fp = d[want_it]
    elif method == 'n':
        idx_in = input("Snapshot index to plot (see list above) [default: 0]: ").strip()
        try:
            idx = int(idx_in) if idx_in else 0
        except ValueError:
            idx = 0
        idx = max(0, min(len(snaps) - 1, idx))
        chosen_iter, chosen_fp = snaps[idx]
    elif method == 'a':
        if not have_log:
            print("No quench log found; cannot select by Tc/time.")
            return
        tc_in = input(f"Tc [K] [default: {default_Tc}]: ").strip()
        try:
            Tc = float(tc_in) if tc_in else float(default_Tc)
        except ValueError:
            Tc = float(default_Tc)
        off_in = input("Time offset after crossing Tc [s] [default: 0.0]: ").strip()
        try:
            after_Tc_s = float(off_in) if off_in else 0.0
        except ValueError:
            after_Tc_s = 0.0

        # Defensive casts: keep type checkers happy and make runtime robust.
        T_use = np.asarray(T_log, dtype=float)
        t_use = np.asarray(t_log, dtype=float)
        it_use = np.asarray(it_log, dtype=float)

        t_cross = _crossing_time_from_log(T_use, t_use, float(Tc))
        t_meas = float(t_cross) + float(after_Tc_s)
        chosen_fp = _select_snapshot_by_time(run_dir, t_meas, it_use, t_use)
        stem = os.path.splitext(os.path.basename(chosen_fp))[0]
        chosen_iter = _infer_trailing_int(stem)
        print(f"Selected snapshot closest to t = t_cross + offset = {t_meas:.6g} s -> {os.path.basename(chosen_fp)}")
    else:
        print("Unknown selection method.")
        return

    # Plot params
    ax_in = input("View axis / look along (x/y/z) [default: z]: ").strip().lower()
    if not ax_in:
        ax_in = 'z'
    axis = ax_in[0] if ax_in else 'z'
    if axis not in ('x', 'y', 'z'):
        print("Invalid axis; using 'z'.")
        axis = 'z'

    if axis == 'z':
        max_idx = Nz_use - 1
        default_idx = Nz_use // 2
        axis_label = 'z'
    elif axis == 'y':
        max_idx = Ny_use - 1
        default_idx = Ny_use // 2
        axis_label = 'y'
    else:
        max_idx = Nx_use - 1
        default_idx = Nx_use // 2
        axis_label = 'x'

    sl_in = input(f"{axis_label}-slice index (0..{max_idx}) [default: {default_idx}]: ").strip()
    try:
        slice_idx = int(sl_in) if sl_in else int(default_idx)
    except ValueError:
        slice_idx = int(default_idx)
    slice_idx = max(0, min(int(max_idx), int(slice_idx)))

    color_field = input("Choose color field (S, nz, n_perp) [default: S]: ").strip() or 'S'
    interpol = input("Choose interpolation method (nearest, bilinear, bicubic, spline16, spline36, sinc) [default: nearest]: ").strip() or 'nearest'
    do_zoom = input("Zoom into center? (y/n) [default: n]: ").strip().lower() == 'y'
    zoom_radius = 15 if do_zoom else None
    arrows_in = input("Number of arrows per axis (0=disable quiver) [default: 20]: ").strip()
    try:
        arrows_per_axis = int(arrows_in) if arrows_in else 20
    except ValueError:
        arrows_per_axis = 20

    tag = input("Optional name tag to append [default: none]: ").strip()
    tag = ("_" + re.sub(r'[^A-Za-z0-9_.-]+', '_', tag)) if tag else ''
    it_tag = f"{int(chosen_iter)}" if chosen_iter is not None else "unknown"
    out_path = os.path.join(out_dir, f"snap_slice_{color_field}_iter_{it_tag}_{axis_label}{int(slice_idx)}{tag}.png")

    plot_nematic_field_slice(
        filename=chosen_fp,
        Nx=Nx_use,
        Ny=Ny_use,
        Nz=Nz_use,
        z_slice=slice_idx,
        slice_axis=axis_label,
        output_path=out_path,
        arrowColor='black',
        zoom_radius=zoom_radius,
        interpol=interpol,
        color_field=color_field,
        print_stats=True,
        arrows_per_axis=arrows_per_axis,
    )
    print(f"Saved snapshot slice -> {out_path}")
    if show:
        plt.show()


def _estimate_ramp_from_log(iteration: np.ndarray, time_s: np.ndarray, T_K: np.ndarray, run_name: str = '', eps_T: float = 1e-6):
    """Estimate ramp duration and rate from a quench log.

    Returns:
      t_ramp: seconds spent changing T (0 for step/unknown)
      rate_abs: |dT/dt| during the ramp (K/s) (nan if not estimable)
      protocol: 'ramp' or 'step_or_unknown'

    Heuristic:
    - Find indices where |ΔT| > eps_T (temperature actually changes between samples).
    - Use first..last change as the ramp window and fit T(t) linearly there.
    - If run_name ends with digits (e.g. output_quench_1000), interpret as ramp_iters and
      compute t_ramp ≈ ramp_iters * dt (dt inferred from log). This helps when logFreq is coarse.
    """
    it = np.asarray(iteration, dtype=float)
    t = np.asarray(time_s, dtype=float)
    T = np.asarray(T_K, dtype=float)
    if t.size < 3 or T.size != t.size or it.size != t.size:
        return 0.0, float('nan'), 'step_or_unknown'

    dT = np.diff(T)
    changing = np.where(np.abs(dT) > float(eps_T))[0]
    if changing.size == 0:
        return 0.0, float('nan'), 'step_or_unknown'

    i0 = int(changing[0])
    i1 = int(changing[-1] + 1)
    if i1 <= i0:
        return 0.0, float('nan'), 'step_or_unknown'

    t0 = float(t[i0])
    t1 = float(t[i1])
    t_ramp = max(0.0, t1 - t0)
    if t_ramp <= 0.0:
        return 0.0, float('nan'), 'step_or_unknown'

    # Linear fit for rate (from the sampled ramp window)
    tt = t[i0:i1 + 1]
    TT = T[i0:i1 + 1]
    if tt.size < 2:
        return t_ramp, float('nan'), 'ramp'

    # Fit TT ≈ a + b*tt
    try:
        b = float(np.polyfit(tt, TT, 1)[0])
    except Exception:
        b = float('nan')
    rate_abs = abs(b) if np.isfinite(b) else float('nan')

    # If run name encodes ramp_iters, prefer t_ramp = ramp_iters * dt
    ramp_iters = _infer_trailing_int(run_name)
    dt_inf = _infer_dt_from_log(it, t)
    if ramp_iters is not None and np.isfinite(dt_inf) and dt_inf > 0:
        t_ramp_name = float(ramp_iters) * float(dt_inf)
        if t_ramp_name > 0:
            t_ramp = t_ramp_name
            dT = float(np.nanmax(T) - np.nanmin(T))
            if t_ramp > 0:
                rate_abs = abs(dT / t_ramp)

    return t_ramp, rate_abs, 'ramp'


def _choose_final_field_file(run_dir: str) -> str:
    """Pick a representative field file for KZ metrics."""
    cand = os.path.join(run_dir, 'nematic_field_final.dat')
    if os.path.exists(cand):
        return cand
    files = glob.glob(os.path.join(run_dir, 'nematic_field_iter_*.dat'))
    if not files:
        raise FileNotFoundError(f"No nematic field files found in {run_dir}")

    def _iter_num(p: str) -> int:
        m = re.search(r'(\d+)', os.path.basename(p))
        return int(m.group(1)) if m else -1

    files.sort(key=_iter_num)
    return files[-1]


def _select_snapshot_by_time(run_dir: str, desired_time_s: float, it_log: np.ndarray, t_log: np.ndarray) -> str:
    """Pick the snapshot file whose logged time is closest to desired_time_s."""
    files = glob.glob(os.path.join(run_dir, 'nematic_field_iter_*.dat'))
    if not files:
        cand = os.path.join(run_dir, 'nematic_field_final.dat')
        if os.path.exists(cand):
            return cand
        raise FileNotFoundError(f"No nematic_field_iter_*.dat or nematic_field_final.dat in {run_dir}")

    def _iter_num(p: str) -> int:
        m = re.search(r'(\d+)', os.path.basename(p))
        return int(m.group(1)) if m else -1

    best_fp = None
    best_dt = float('inf')
    for fp in files:
        it = _iter_num(fp)
        if it < 0:
            continue
        tt = _nearest_from_log(it_log, t_log, it)
        if not np.isfinite(tt):
            continue
        d = abs(float(tt) - float(desired_time_s))
        if d < best_dt:
            best_dt = d
            best_fp = fp

    if best_fp is None:
        # fallback
        cand = os.path.join(run_dir, 'nematic_field_final.dat')
        if os.path.exists(cand):
            return cand
        files.sort(key=_iter_num)
        return files[-1]
    return best_fp


def _choose_z_slices_for_avg(
    Nz: int,
    *,
    z_center: int | None,
    z_avg: int,
    z_min: int | None = None,
    z_max: int | None = None,
) -> list[int]:
    """Choose a list of z-slices to average over.

    - If z_avg <= 1: returns a single slice (center or mid-plane).
    - If z_center is None: picks z_avg slices evenly spaced across [z_min, z_max].
    - If z_center is given: picks z_avg slices evenly spaced across the full symmetric span
      around z_center that fits inside [z_min, z_max].
    """
    Nz = int(Nz)
    if Nz < 1:
        return [0]

    z_lo = 0 if z_min is None else int(z_min)
    z_hi = (Nz - 1) if z_max is None else int(z_max)
    z_lo = max(0, min(Nz - 1, z_lo))
    z_hi = max(0, min(Nz - 1, z_hi))
    if z_hi < z_lo:
        z_lo, z_hi = z_hi, z_lo

    if z_avg <= 1:
        z0 = (Nz // 2) if z_center is None else int(z_center)
        return [max(0, min(Nz - 1, z0))]

    if z_center is None:
        zz = np.linspace(z_lo, z_hi, int(z_avg))
    else:
        zc = max(0, min(Nz - 1, int(z_center)))
        span = min(zc - z_lo, z_hi - zc)
        zz = np.linspace(zc - span, zc + span, int(z_avg))

    z_int = [int(round(v)) for v in zz]
    # Deduplicate while preserving order
    out: list[int] = []
    seen = set()
    for z in z_int:
        z = max(0, min(Nz - 1, z))
        if z not in seen:
            out.append(z)
            seen.add(z)
    return out


def _crossing_time_from_log(T: np.ndarray, t: np.ndarray, Tc: float) -> float:
    """Estimate the time when temperature crosses from above Tc to <= Tc.

    Uses the first downward crossing (T[i] > Tc and T[i+1] <= Tc) with linear interpolation.
    If not found:
      - If max(T) <= Tc: returns first time.
      - If min(T)  > Tc: returns last time.
    """
    T = np.asarray(T, dtype=float)
    t = np.asarray(t, dtype=float)
    if T.size < 1 or t.size < 1:
        return float('nan')
    Tc = float(Tc)

    if np.nanmax(T) <= Tc:
        return float(t[0])
    if np.nanmin(T) > Tc:
        return float(t[-1])

    for i in range(int(T.size) - 1):
        Ti = float(T[i])
        Tj = float(T[i + 1])
        if not (np.isfinite(Ti) and np.isfinite(Tj)):
            continue
        if (Ti > Tc) and (Tj <= Tc):
            ti = float(t[i])
            tj = float(t[i + 1])
            dT = (Tj - Ti)
            if not (np.isfinite(ti) and np.isfinite(tj)):
                return ti if np.isfinite(ti) else tj
            if dT == 0.0:
                return tj
            alpha = (Tc - Ti) / dT
            alpha = max(0.0, min(1.0, float(alpha)))
            return ti + (tj - ti) * alpha

    return float(t[-1])


def _fit_powerlaw_loglog(xv, yv, *, x_min: float | None = None, x_max: float | None = None):
    """Fit y = prefactor * x^slope in log–log space.

    Optional x-range filtering can be applied via x_min/x_max (inclusive).
    """
    xv = np.asarray(xv, dtype=float)
    yv = np.asarray(yv, dtype=float)
    m = np.isfinite(xv) & np.isfinite(yv) & (xv > 0) & (yv > 0)
    if x_min is not None and np.isfinite(float(x_min)):
        m &= (xv >= float(x_min))
    if x_max is not None and np.isfinite(float(x_max)):
        m &= (xv <= float(x_max))
    if np.count_nonzero(m) < 3:
        return float('nan'), float('nan')
    a, b = np.polyfit(np.log(xv[m]), np.log(yv[m]), 1)
    return float(a), float(np.exp(b))


def _kz_metrics_over_z_slices(
    field_path: str,
    Nx: int,
    Ny: int,
    Nz: int,
    z_slices: list[int],
    *,
    S_threshold: float,
    min_valid_points: int = 64,
) -> tuple[float, float, int, float, float]:
    """Compute (xi_mean, defect_mean, n_used, xi_std, defect_std) across z_slices.

    Skips slices with too few valid (S>S_threshold) points or non-finite metrics.
    """
    xi_vals: list[float] = []
    def_vals: list[float] = []
    diag = []

    for z in z_slices:
        z_use = max(0, min(int(Nz) - 1, int(z)))
        S2, nx2, ny2, _ = _load_nematic_slice_arrays(field_path, Nx, Ny, Nz, z_use)
        n_valid = int(np.count_nonzero(np.isfinite(S2)))
        n_mask = int(np.count_nonzero((S2 > S_threshold) & np.isfinite(S2)))
        smax = float(np.nanmax(S2)) if n_valid else float('nan')
        diag.append((int(z_use), n_mask, n_valid, smax))
        if n_mask < int(min_valid_points):
            continue
        n_def, _ = defect_density_2d_from_slice(S2, nx2, ny2, S_threshold=S_threshold)
        xi, _, _ = correlation_length_2d_from_slice(S2, nx2, ny2, S_threshold=S_threshold)
        if np.isfinite(xi) and xi > 0:
            xi_vals.append(float(xi))
        if np.isfinite(n_def) and n_def >= 0:
            def_vals.append(float(n_def))

    n_used = min(len(xi_vals), len(def_vals))
    if n_used < 1:
        # Include a tiny diagnostic so it's clear this is usually a mask/threshold issue, not "file is empty".
        if diag:
            best = max(diag, key=lambda x: x[1])  # by n_mask
            zbest, nmask_best, nvalid_best, smax_best = best
            raise ValueError(
                f"No valid z-slices found for metrics in {os.path.basename(field_path)} (S_threshold={S_threshold}, "
                f"min_valid_points={int(min_valid_points)}). Best slice z={zbest}: mask={nmask_best}/{nvalid_best}, "
                f"Smax={smax_best:.3g}."
            )
        raise ValueError(
            f"No valid z-slices found for metrics in {os.path.basename(field_path)} (S_threshold={S_threshold}, "
            f"min_valid_points={int(min_valid_points)})."
        )

    xi_arr = np.array(xi_vals[:n_used], dtype=float)
    d_arr = np.array(def_vals[:n_used], dtype=float)
    return float(np.mean(xi_arr)), float(np.mean(d_arr)), int(n_used), float(np.std(xi_arr)), float(np.std(d_arr))


def aggregate_kz_scaling(
    parent_dir: str = '.',
    *,
    pattern: str = 'output_quench*',
    out_dir: str = 'pics',
    z_slice: int | None = None,
    z_avg: int = 1,
    z_margin_frac: float = 0.0,
    S_threshold: float = 0.1,
    prefer_log_defects: bool = True,
    allow_log_only: bool = False,
    prefer_log_xi_proxy: bool = True,
    x_axis: str = 't_ramp',
    fit_x_min: float | None = None,
    fit_x_max: float | None = None,
    measure: str = 'final',
    after_Tlow_s: float = 0.0,
    Tc: float | None = None,
    after_Tc_s: float = 0.0,
    show: bool = True,
    plot: bool = True,
    write_files: bool = True,
):
    """Aggregate KZ proxies across multiple quench runs.

        Scans subdirectories matching pattern that contain quench_log.dat, computes:
            - t_ramp and |dT/dt| from log
            - defect density from quench_log.dat if present (preferred)
            - xi from a field snapshot (2D mid-plane slice proxy)

        Notes:
            - If your runs were produced with on-the-fly defect logging enabled in QSR_cuda,
                the log columns `defect_density_per_plaquette` / `defect_plaquettes_used` are used.
            - Correlation length xi still requires snapshots unless you also log an xi proxy.
            - If allow_log_only=True, missing snapshots are tolerated and xi becomes NaN.

    x_axis:
      - 't_ramp' : seconds
      - 'rate'   : |dT/dt| (K/s)
    """
    if not parent_dir:
        parent_dir = '.'

    parent_dir = os.path.abspath(parent_dir)
    cand_dirs = [d for d in glob.glob(os.path.join(parent_dir, pattern)) if os.path.isdir(d)]
    cand_dirs.sort()
    run_dirs = []
    for d in cand_dirs:
        if os.path.exists(os.path.join(d, 'quench_log.dat')):
            run_dirs.append(d)

    if not run_dirs:
        raise FileNotFoundError(f"No quench runs found in {parent_dir} matching {pattern} (missing quench_log.dat)")

    measure_norm = (measure or 'final').strip().lower()
    if measure_norm in ('after_tlow', 'after_t_low', 'tlow', 'after_final_t'):
        offset = float(after_Tlow_s)
        offset_label = f"{offset:g}s".replace('.', 'p').replace('-', 'm')
        measure_label = f"after_Tlow_{offset_label}"
    elif measure_norm in ('after_tc', 'after_t_c', 'tc', 'after_transition'):
        offset = float(after_Tc_s)
        offset_label = f"{offset:g}s".replace('.', 'p').replace('-', 'm')
        Tc_label = f"{float(Tc) if Tc is not None else float('nan'):g}K".replace('.', 'p').replace('-', 'm')
        measure_label = f"after_Tc_{Tc_label}_{offset_label}"
    else:
        measure_label = 'final'

    rows = []
    for run_dir in run_dirs:
        data, log_path = load_quench_log(run_dir)
        it = np.atleast_1d(data['iteration']).astype(float)
        t = np.atleast_1d(data['time_s']).astype(float)
        T = np.atleast_1d(data['T_K']).astype(float)
        t_ramp, rate_abs, protocol = _estimate_ramp_from_log(it, t, T, run_name=os.path.basename(run_dir))

        def _nearest_log_index(t_log: np.ndarray, t_target: float) -> int:
            t_log = np.asarray(t_log, dtype=float)
            if t_log.size == 0:
                return 0
            if not np.isfinite(float(t_target)):
                return int(t_log.size - 1)
            d = np.abs(t_log - float(t_target))
            d[~np.isfinite(d)] = np.inf
            if not np.any(np.isfinite(d)):
                return int(t_log.size - 1)
            return int(np.argmin(d))

        def _try_defect_density_from_log(t_target: float) -> tuple[float, float, int] | None:
            if not bool(prefer_log_defects):
                return None
            names = set(getattr(data.dtype, 'names', []) or [])
            if 'defect_density_per_plaquette' not in names:
                return None
            dd = np.atleast_1d(data['defect_density_per_plaquette']).astype(float)
            used = None
            if 'defect_plaquettes_used' in names:
                used = np.atleast_1d(data['defect_plaquettes_used']).astype(float)
            idx = _nearest_log_index(t, float(t_target))
            if idx < 0 or idx >= dd.size:
                return None
            v = float(dd[idx])
            if not np.isfinite(v):
                return None
            used_i = 0
            if used is not None and idx < used.size:
                uu = float(used[idx])
                used_i = int(uu) if np.isfinite(uu) else 0
            # return (defect_density, time_at_row, used_plaquettes)
            t_row = float(np.atleast_1d(t)[idx]) if t.size else float('nan')
            return v, t_row, used_i

        def _try_xi_proxy_from_log(t_target: float) -> tuple[float, float, int] | None:
            if not bool(prefer_log_xi_proxy):
                return None
            names = set(getattr(data.dtype, 'names', []) or [])
            if 'xi_grad_proxy' not in names:
                return None
            xv = np.atleast_1d(data['xi_grad_proxy']).astype(float)
            used = None
            if 'xi_grad_edges_used' in names:
                used = np.atleast_1d(data['xi_grad_edges_used']).astype(float)
            idx = _nearest_log_index(t, float(t_target))
            if idx < 0 or idx >= xv.size:
                return None
            v = float(xv[idx])
            if not np.isfinite(v):
                return None
            used_i = 0
            if used is not None and idx < used.size:
                uu = float(used[idx])
                used_i = int(uu) if np.isfinite(uu) else 0
            t_row = float(np.atleast_1d(t)[idx]) if t.size else float('nan')
            return v, t_row, used_i

        # Choose which state to analyze
        t_meas = float(t[-1])
        if measure_norm in ('after_tlow', 'after_t_low', 'tlow', 'after_final_t'):
            # time when we reach final temperature (heuristic: first time within eps of min(T))
            Tmin = float(np.nanmin(T))
            epsT = 1e-9
            idxs = np.where(np.abs(T - Tmin) <= epsT)[0]
            if idxs.size == 0:
                t_reach = float(t[-1])
            else:
                t_reach = float(t[int(idxs[0])])
            t_meas = t_reach + float(after_Tlow_s)
            try:
                field_path = _select_snapshot_by_time(run_dir, t_meas, it, t)
            except FileNotFoundError:
                if not bool(allow_log_only):
                    raise
                field_path = ''
        elif measure_norm in ('after_tc', 'after_t_c', 'tc', 'after_transition'):
            if Tc is None or not np.isfinite(float(Tc)):
                raise ValueError("measure=after_Tc requires Tc to be set")
            t_cross = _crossing_time_from_log(T, t, float(Tc))
            t_meas = float(t_cross) + float(after_Tc_s)
            try:
                field_path = _select_snapshot_by_time(run_dir, t_meas, it, t)
            except FileNotFoundError:
                if not bool(allow_log_only):
                    raise
                field_path = ''
        else:
            try:
                field_path = _choose_final_field_file(run_dir)
            except FileNotFoundError:
                if not bool(allow_log_only):
                    raise
                field_path = ''

        log_def = _try_defect_density_from_log(t_meas)
        log_xi = _try_xi_proxy_from_log(t_meas)
        def _iter_num(p: str) -> int:
            m = re.search(r'(\d+)', os.path.basename(p))
            return int(m.group(1)) if m else -1

        def _compute_metrics_for_field(fp: str):
            Nx, Ny, Nz = infer_grid_dims_from_nematic_field_file(fp)
            z_center = None if z_slice is None else int(z_slice)

            z_margin = float(z_margin_frac)
            if not np.isfinite(z_margin):
                z_margin = 0.0
            z_margin = max(0.0, min(0.45, z_margin))
            if int(z_avg) > 1 and z_margin > 0.0:
                z_lo = int(round(z_margin * (Nz - 1)))
                z_hi = int(round((1.0 - z_margin) * (Nz - 1)))
            else:
                z_lo, z_hi = 0, Nz - 1

            z_slices = _choose_z_slices_for_avg(Nz, z_center=z_center, z_avg=int(z_avg), z_min=z_lo, z_max=z_hi)
            xi, n_def, n_used, xi_std, ndef_std = _kz_metrics_over_z_slices(
                fp,
                Nx,
                Ny,
                Nz,
                z_slices,
                S_threshold=S_threshold,
            )
            z_use = (Nz // 2) if z_center is None else max(0, min(Nz - 1, z_center))
            return Nx, Ny, Nz, xi, n_def, n_used, xi_std, ndef_std, z_use

        # For time-targeted measurements, Tc/Tlow can fall between sparse snapshot files. If the chosen snapshot
        # is still essentially isotropic (no S>S_threshold points), try the next snapshot(s).
        max_retries = 12
        tried = []
        files_sorted = None
        if measure_norm in ('after_tlow', 'after_t_low', 'tlow', 'after_final_t', 'after_tc', 'after_t_c', 'tc', 'after_transition'):
            files_sorted = glob.glob(os.path.join(run_dir, 'nematic_field_iter_*.dat'))
            files_sorted.sort(key=_iter_num)

        fp_try = field_path
        last_err: Exception | None = None
        # Initialize for static analyzers; these will be overwritten on success.
        Nx = Ny = Nz = 0
        xi = float('nan')
        n_def = float('nan')
        n_used = 0
        xi_std = float('nan')
        ndef_std = float('nan')
        z_use = 0
        if fp_try:
            for _attempt in range(max_retries + 1):
                try:
                    Nx, Ny, Nz, xi, n_def, n_used, xi_std, ndef_std, z_use = _compute_metrics_for_field(fp_try)
                    field_path = fp_try
                    break
                except ValueError as e:
                    last_err = e
                    tried.append(os.path.basename(fp_try))
                    msg = str(e)
                    if ('No valid z-slices found for metrics' not in msg) or (not files_sorted):
                        if not bool(allow_log_only):
                            raise
                        # In log-only mode, allow xi to be missing.
                        xi = float('nan')
                        n_def = float('nan')
                        n_used = 0
                        break
                    it_cur = _iter_num(fp_try)
                    # Advance to the next snapshot file by iteration
                    idx = -1
                    for j, fp in enumerate(files_sorted):
                        if _iter_num(fp) == it_cur:
                            idx = j
                            break
                    if idx < 0 or idx + 1 >= len(files_sorted):
                        break
                    fp_try = files_sorted[idx + 1]
            else:
                last_err = last_err or ValueError('No valid z-slices found for metrics')

        # Prefer defect density from quench_log if available (helps avoid huge snapshots).
        if log_def is not None:
            n_def = float(log_def[0])
            ndef_std = float('nan')

        # If snapshots are missing (or log-only is requested), use xi proxy from log if available.
        if (not np.isfinite(xi) or int(n_used) <= 0) and (log_xi is not None):
            xi = float(log_xi[0])
            xi_std = float('nan')
            n_used = max(int(n_used), 1)

        have_xi = bool(np.isfinite(xi) and int(n_used) > 0)
        have_def = bool(np.isfinite(n_def) and (n_def > 0.0 or n_def == 0.0))
        if not have_def:
            # Without defect density we can't do KZ scaling at all.
            if last_err is not None and 'No valid z-slices found for metrics' in str(last_err):
                tried_s = ','.join(tried[:6]) + ('...' if len(tried) > 6 else '')
                raise ValueError(f"{last_err} Tried snapshots: {tried_s}")
            if last_err is not None:
                raise last_err
            raise ValueError(f"Failed to compute defect density for run {os.path.basename(run_dir)}")
        if (not have_xi) and (not bool(allow_log_only)):
            if last_err is not None and 'No valid z-slices found for metrics' in str(last_err):
                tried_s = ','.join(tried[:6]) + ('...' if len(tried) > 6 else '')
                raise ValueError(f"{last_err} Tried snapshots: {tried_s}")
            if last_err is not None:
                raise last_err
            raise ValueError(f"Failed to compute xi for run {os.path.basename(run_dir)}")
        # A convenient "quench time" scale for ramp: tau_Q ~ t_ramp
        tau_Q = float(t_ramp)
        rows.append(
            (
                os.path.basename(run_dir),
                protocol,
                tau_Q,
                rate_abs,
                xi,
                n_def,
                z_use,
                S_threshold,
                int(z_avg),
                int(n_used),
                xi_std,
                ndef_std,
                os.path.basename(field_path) if field_path else '',
            )
        )

    out_path = None
    csv_path = None
    tag = os.path.basename(parent_dir) or 'runs'
    tag_full = f"{tag}_{measure_label}"
    if write_files:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f'kz_scaling_{tag_full}.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('run,protocol,t_ramp_s,rate_K_per_s,xi_lattice,defect_density,z_center,S_threshold,z_avg,z_used,xi_std,defect_std,field_file\n')
            for r in rows:
                f.write(
                    f"{r[0]},{r[1]},{r[2]:.8g},{r[3]:.8g},{r[4]:.8g},{r[5]:.8g},{int(r[6])},{r[7]:.8g},{int(r[8])},{int(r[9])},{r[10]:.8g},{r[11]:.8g},{r[12]}\n"
                )

    # Prepare arrays for plotting
    arr = np.array([[r[2], r[3], r[4], r[5]] for r in rows], dtype=float)
    t_ramp = arr[:, 0]
    rate = arr[:, 1]
    xi = arr[:, 2]
    ndef = arr[:, 3]

    if x_axis.strip().lower() == 'rate':
        x = rate
        x_label = r'$|dT/dt|$ [K/s]'
        x_name = 'rate'
    else:
        x = t_ramp
        x_label = r'$t_{ramp}$ [s]'
        x_name = 't_ramp'

    slope_xi, pref_xi = _fit_powerlaw_loglog(x, xi, x_min=fit_x_min, x_max=fit_x_max)
    slope_nd, pref_nd = _fit_powerlaw_loglog(x, ndef, x_min=fit_x_min, x_max=fit_x_max)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.loglog(x, xi, 'o', ms=7)
        ax1.set_ylabel(r'$\xi$ (lattice units)')
        ax1.grid(True, which='both', alpha=0.3)

        if np.isfinite(slope_xi) and np.isfinite(pref_xi):
            xs = np.linspace(np.nanmin(x[x > 0]), np.nanmax(x[x > 0]), 100)
            ax1.loglog(xs, pref_xi * xs ** slope_xi, '--', lw=1.5, label=f'fit: slope={slope_xi:.3g}')
            ax1.legend()

        ax2.loglog(x, ndef, 'o', ms=7, color='tab:red')
        ax2.set_xlabel(x_label)
        ax2.set_ylabel('defect density (per plaquette)')
        ax2.grid(True, which='both', alpha=0.3)

        if np.isfinite(slope_nd) and np.isfinite(pref_nd):
            xs = np.linspace(np.nanmin(x[x > 0]), np.nanmax(x[x > 0]), 100)
            ax2.loglog(xs, pref_nd * xs ** slope_nd, '--', lw=1.5, color='tab:red', label=f'fit: slope={slope_nd:.3g}')
            ax2.legend()

        fig.suptitle(
            f'KZ scaling (2D proxies) across {len(rows)} runs | measure={measure_label} | x={x_name} | S>{S_threshold} | z_avg={int(z_avg)}'
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        if write_files:
            out_path = os.path.join(out_dir, f'kz_scaling_{tag_full}_{x_name}.png')
            fig.savefig(out_path, dpi=220)
            print(f"Saved KZ scaling plot -> {out_path}")
            print(f"Saved KZ scaling CSV  -> {csv_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return rows, out_path, csv_path


def aggregate_kz_scaling_3d(
    parent_dir: str = '.',
    *,
    run_dirs: list[str] | None = None,
    pattern: str = 'output_quench*',
    out_dir: str = 'pics',
    x_axis: str = 't_ramp',
    fit_x_min: float | None = None,
    fit_x_max: float | None = None,
    measure: str = 'after_Tc',
    after_Tlow_s: float = 0.0,
    Tc: float | None = 310.2,
    after_Tc_s: float = 0.0,
    after_Tc_mode: str = 'fixed',
    after_Tc_frac_ramp: float = 0.1,
    avgS_target: float = 0.1,
    extra_after_target_s: float = 0.0,
    S_threshold_xi: float = 0.1,
    max_r: int | None = None,
    S_droplet: float = 0.1,
    S_core: float = 0.05,
    dilate_iters: int = 2,
    fill_holes: bool = False,
    core_erosion_iters: int = 0,
    min_core_voxels: int = 30,
    defect_proxy: str = 'skeleton',
    max_runs: int | None = None,
    show: bool = True,
    plot: bool = True,
    write_files: bool = True,
):
    """Aggregate 3D KZ-style metrics across multiple quench runs.

    Per run, selects a field snapshot (final / after_Tlow / after_Tc) and computes:
      - xi_3D from 3D Q-tensor correlation (masked by S > S_threshold_xi)
      - defect-line proxy as line length density from skeletonized low-S core

    x_axis:
      - 't_ramp' : seconds (tau_Q proxy)
      - 'rate'   : |dT/dt| (K/s)
    """
    if not parent_dir:
        parent_dir = '.'

    parent_dir = os.path.abspath(parent_dir)
    if run_dirs is None:
        cand_dirs = [d for d in glob.glob(os.path.join(parent_dir, pattern)) if os.path.isdir(d)]
        cand_dirs.sort()
        run_dirs = [d for d in cand_dirs if os.path.exists(os.path.join(d, 'quench_log.dat'))]
        if not run_dirs:
            raise FileNotFoundError(f"No quench runs found in {parent_dir} matching {pattern} (missing quench_log.dat)")
        if max_runs is not None and int(max_runs) > 0:
            run_dirs = run_dirs[: int(max_runs)]
    else:
        resolved = []
        for d in run_dirs:
            d_abs = d if os.path.isabs(d) else os.path.join(parent_dir, d)
            if not os.path.isdir(d_abs):
                raise FileNotFoundError(f"Run directory not found: {d_abs}")
            if not os.path.exists(os.path.join(d_abs, 'quench_log.dat')):
                raise FileNotFoundError(f"Missing quench_log.dat in run directory: {d_abs}")
            resolved.append(d_abs)
        run_dirs = resolved

    measure_norm = (measure or 'after_Tc').strip().lower()
    mode_norm = (after_Tc_mode or 'fixed').strip().lower()
    if measure_norm in ('after_tlow', 'after_t_low', 'tlow', 'after_final_t'):
        offset = float(after_Tlow_s)
        offset_label = f"{offset:g}s".replace('.', 'p').replace('-', 'm')
        measure_label = f"after_Tlow_{offset_label}"
    elif measure_norm in ('after_tc', 'after_t_c', 'tc', 'after_transition'):
        Tc_use = float(Tc) if Tc is not None and np.isfinite(float(Tc)) else float('nan')
        Tc_label = f"{Tc_use:g}K".replace('.', 'p').replace('-', 'm')
        if mode_norm in ('frac', 'fraction', 'frac_ramp', 'ramp_frac'):
            frac_label = f"{float(after_Tc_frac_ramp):g}".replace('.', 'p').replace('-', 'm')
            measure_label = f"after_Tc_{Tc_label}_frac_{frac_label}"
        elif mode_norm in ('avg_s', 'avgs', 's', 'order'):
            s_label = f"{float(avgS_target):g}".replace('.', 'p').replace('-', 'm')
            extra_label = f"{float(extra_after_target_s):g}s".replace('.', 'p').replace('-', 'm')
            measure_label = f"after_Tc_{Tc_label}_avgS_{s_label}_plus_{extra_label}"
        elif mode_norm in ('auto', 'best'):
            frac_label = f"{float(after_Tc_frac_ramp):g}".replace('.', 'p').replace('-', 'm')
            s_label = f"{float(avgS_target):g}".replace('.', 'p').replace('-', 'm')
            extra_label = f"{float(extra_after_target_s):g}s".replace('.', 'p').replace('-', 'm')
            measure_label = f"after_Tc_{Tc_label}_auto_frac_{frac_label}_avgS_{s_label}_plus_{extra_label}"
        else:
            offset = float(after_Tc_s)
            offset_label = f"{offset:g}s".replace('.', 'p').replace('-', 'm')
            measure_label = f"after_Tc_{Tc_label}_{offset_label}"
    else:
        measure_label = 'final'

    def _iter_num(p: str) -> int:
        m = re.search(r'(\d+)', os.path.basename(p))
        return int(m.group(1)) if m else -1

    defect_proxy_norm = (defect_proxy or 'skeleton').strip().lower()
    if defect_proxy_norm in ('core', 'core_density', 'corefrac', 'core_fraction', 'volume', 'vox'):
        use_skeleton = False
        defect_col_name = 'core_density_per_voxel'
        defect_label = 'core density proxy (core_vox/droplet_vox)'
    else:
        use_skeleton = True
        defect_col_name = 'line_density_per_voxel'
        defect_label = 'line density proxy (length/voxel)'

    def _compute_3d_metrics_for_field(fp: str):
        xi3, _, _, dims = correlation_length_3d_from_field_file(fp, S_threshold=float(S_threshold_xi), max_r=max_r)
        if not (np.isfinite(xi3) and float(xi3) > 0):
            raise ValueError(f"xi_3D is invalid for {os.path.basename(fp)} (xi_3D={xi3})")
        defect = defect_line_metrics_3d_from_field_file(
            fp,
            S_droplet=float(S_droplet),
            S_core=float(S_core),
            dilate_iters=int(dilate_iters),
            fill_holes=bool(fill_holes),
            core_erosion_iters=int(core_erosion_iters),
            min_core_voxels=int(min_core_voxels),
            use_skeleton=bool(use_skeleton),
        )
        droplet_vox = int(defect.get('droplet_voxels', 0) or 0)
        if droplet_vox <= 0:
            raise ValueError(f"droplet mask is empty for {os.path.basename(fp)} (S_droplet={S_droplet})")
        return float(xi3), defect, dims

    rows = []
    for run_dir in run_dirs:
        data, log_path = load_quench_log(run_dir)
        it = np.atleast_1d(data['iteration']).astype(float)
        t = np.atleast_1d(data['time_s']).astype(float)
        T = np.atleast_1d(data['T_K']).astype(float)
        names = data.dtype.names or ()
        avgS = np.atleast_1d(data['avg_S']).astype(float) if ('avg_S' in names) else np.full_like(t, np.nan)
        t_ramp, rate_abs, protocol = _estimate_ramp_from_log(it, t, T, run_name=os.path.basename(run_dir))

        # Choose which state to analyze
        if measure_norm in ('after_tlow', 'after_t_low', 'tlow', 'after_final_t'):
            Tmin = float(np.nanmin(T))
            epsT = 1e-9
            idxs = np.where(np.abs(T - Tmin) <= epsT)[0]
            t_reach = float(t[int(idxs[0])]) if idxs.size else float(t[-1])
            t_meas = t_reach + float(after_Tlow_s)
            field_path = _select_snapshot_by_time(run_dir, t_meas, it, t)
        elif measure_norm in ('after_tc', 'after_t_c', 'tc', 'after_transition'):
            if Tc is None or not np.isfinite(float(Tc)):
                raise ValueError("measure=after_Tc requires Tc to be set")
            t_cross = _crossing_time_from_log(T, t, float(Tc))
            # Choose offset using selected mode
            off_fixed = float(after_Tc_s)
            off_frac = float(after_Tc_frac_ramp) * float(t_ramp) if np.isfinite(float(t_ramp)) and float(t_ramp) > 0 else 0.0

            off_avgS = 0.0
            m = np.isfinite(t) & np.isfinite(avgS)
            m &= (t >= float(t_cross))
            idxs = np.where(m & (avgS >= float(avgS_target)))[0]
            if idxs.size:
                off_avgS = float(t[int(idxs[0])] - float(t_cross))

            if mode_norm in ('frac', 'fraction', 'frac_ramp', 'ramp_frac'):
                off = off_frac
            elif mode_norm in ('avg_s', 'avgs', 's', 'order'):
                off = off_avgS + float(extra_after_target_s)
            elif mode_norm in ('auto', 'best'):
                off = max(float(off_frac), float(off_avgS)) + float(extra_after_target_s)
            else:
                off = off_fixed

            t_meas = float(t_cross) + float(off)
            field_path = _select_snapshot_by_time(run_dir, t_meas, it, t)
        else:
            field_path = _choose_final_field_file(run_dir)

        # Retry forward snapshots if the selected one is still too isotropic / empty-mask
        max_retries = 12
        tried = []
        files_sorted = None
        if measure_norm in ('after_tlow', 'after_t_low', 'tlow', 'after_final_t', 'after_tc', 'after_t_c', 'tc', 'after_transition'):
            files_sorted = glob.glob(os.path.join(run_dir, 'nematic_field_iter_*.dat'))
            files_sorted.sort(key=_iter_num)

        fp_try = field_path
        last_err: Exception | None = None
        xi3 = float('nan')
        defect = {}
        dims = (0, 0, 0)
        for _attempt in range(max_retries + 1):
            try:
                xi3, defect, dims = _compute_3d_metrics_for_field(fp_try)
                field_path = fp_try
                break
            except Exception as e:
                last_err = e
                tried.append(os.path.basename(fp_try))
                if not files_sorted:
                    break
                it_cur = _iter_num(fp_try)
                idx = -1
                for j, fp in enumerate(files_sorted):
                    if _iter_num(fp) == it_cur:
                        idx = j
                        break
                if idx < 0 or idx + 1 >= len(files_sorted):
                    break
                fp_try = files_sorted[idx + 1]

        if not (np.isfinite(xi3) and xi3 > 0):
            tried_s = ','.join(tried[:6]) + ('...' if len(tried) > 6 else '')
            raise ValueError(
                f"Failed 3D metrics for run {os.path.basename(run_dir)}: {last_err}. Tried snapshots: {tried_s}"
            )

        # Selection diagnostics (useful to verify consistency)
        t_cross_row = float('nan')
        t_sel_row = float('nan')
        after_tc_actual = float('nan')
        iter_sel = -1
        if measure_norm in ('after_tc', 'after_t_c', 'tc', 'after_transition'):
            Tc_row = float(Tc) if (Tc is not None and np.isfinite(float(Tc))) else float('nan')
            t_cross_row = float(_crossing_time_from_log(T, t, Tc_row)) if np.isfinite(Tc_row) else float('nan')
            m_it = re.search(r'(\d+)', os.path.basename(field_path))
            iter_sel = int(m_it.group(1)) if m_it else -1
            t_sel_row = float(_nearest_from_log(it, t, iter_sel)) if iter_sel >= 0 else float('nan')
            after_tc_actual = float(t_sel_row - t_cross_row) if (np.isfinite(t_sel_row) and np.isfinite(t_cross_row)) else float('nan')

        rows.append(
            (
                os.path.basename(run_dir),
                protocol,
                float(t_ramp),
                float(rate_abs),
                float(xi3),
                float(defect.get('line_length_lattice', float('nan'))),
                float(defect.get(defect_col_name, float('nan'))),
                int(defect.get('Nx', dims[0])),
                int(defect.get('Ny', dims[1])),
                int(defect.get('Nz', dims[2])),
                float(S_threshold_xi),
                float(S_droplet),
                float(S_core),
                int(dilate_iters),
                int(min_core_voxels),
                os.path.basename(field_path),
                float(t_cross_row),
                float(t_sel_row),
                float(after_tc_actual),
                int(iter_sel),
            )
        )

    out_path = None
    csv_path = None
    tag = os.path.basename(parent_dir) or 'runs'
    tag_full = f"{tag}_{measure_label}"

    if write_files:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f'kz_scaling3d_{tag_full}.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(
                'run,protocol,t_ramp_s,rate_K_per_s,xi3d_lattice,line_length_lattice,line_density_per_voxel,'
                'Nx,Ny,Nz,S_threshold_xi,S_droplet,S_core,dilate_iters,min_core_voxels,field_file,'
                't_cross_s,t_selected_s,after_Tc_actual_s,iter_selected,defect_proxy\n'
            )
            for r in rows:
                f.write(
                    f"{r[0]},{r[1]},{r[2]:.8g},{r[3]:.8g},{r[4]:.8g},{r[5]:.8g},{r[6]:.8g},"
                    f"{int(r[7])},{int(r[8])},{int(r[9])},{r[10]:.8g},{r[11]:.8g},{r[12]:.8g},"
                    f"{int(r[13])},{int(r[14])},{r[15]},"
                    f"{r[16]:.8g},{r[17]:.8g},{r[18]:.8g},{int(r[19])},{defect_proxy_norm}\n"
                )

    # Prepare arrays for plotting
    arr = np.array([[r[2], r[3], r[4], r[6]] for r in rows], dtype=float)
    t_ramp = arr[:, 0]
    rate = arr[:, 1]
    xi3d = arr[:, 2]
    line_den = arr[:, 3]

    if x_axis.strip().lower() == 'rate':
        x = rate
        x_label = r'$|dT/dt|$ [K/s]'
        x_name = 'rate'
    else:
        x = t_ramp
        x_label = r'$t_{ramp}$ [s]'
        x_name = 't_ramp'

    slope_xi, pref_xi = _fit_powerlaw_loglog(x, xi3d, x_min=fit_x_min, x_max=fit_x_max)
    slope_ld, pref_ld = _fit_powerlaw_loglog(x, line_den, x_min=fit_x_min, x_max=fit_x_max)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.loglog(x, xi3d, 'o', ms=7)
        ax1.set_ylabel(r'$\xi_{3D}$ (lattice units)')
        ax1.grid(True, which='both', alpha=0.3)
        if np.isfinite(slope_xi) and np.isfinite(pref_xi):
            xs = np.linspace(np.nanmin(x[x > 0]), np.nanmax(x[x > 0]), 200)
            ax1.loglog(xs, pref_xi * xs ** slope_xi, '--', lw=1.5, label=f'fit: slope={slope_xi:.3g}')
            ax1.legend()

        ax2.loglog(x, line_den, 'o', ms=7, color='tab:purple')
        ax2.set_xlabel(x_label)
        ax2.set_ylabel(defect_label)
        ax2.grid(True, which='both', alpha=0.3)
        if np.isfinite(slope_ld) and np.isfinite(pref_ld):
            xs = np.linspace(np.nanmin(x[x > 0]), np.nanmax(x[x > 0]), 200)
            ax2.loglog(xs, pref_ld * xs ** slope_ld, '--', lw=1.5, color='tab:purple', label=f'fit: slope={slope_ld:.3g}')
            ax2.legend()

        fig.suptitle(
            f'KZ scaling (3D metrics) across {len(rows)} runs | measure={measure_label} | x={x_name} | '
            f'S_xi>{S_threshold_xi:g} | S_drop>{S_droplet:g} | S_core<{S_core:g}'
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        if write_files:
            out_path = os.path.join(out_dir, f'kz_scaling3d_{tag_full}_{x_name}.png')
            fig.savefig(out_path, dpi=220)
            print(f"Saved 3D KZ scaling plot -> {out_path}")
            print(f"Saved 3D KZ scaling CSV  -> {csv_path}")

        if np.isfinite(slope_xi) and np.isfinite(slope_ld) and x_name == 't_ramp':
            print(f"[consistency] slope(line_density)={slope_ld:.4g}; expected ~{-2.0*slope_xi:.4g} for line defects")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return rows, out_path, csv_path


def sweep_kz_slope_stability(
    parent_dir: str = '.',
    *,
    pattern: str = 'output_quench*',
    out_dir: str = 'pics',
    x_axis: str = 't_ramp',
    S_threshold: float = 0.02,
    z_slice: int | None = None,
    z_avg: int = 11,
    z_margin_frac: float = 0.2,
    Tc: float = 310.2,
    snapshotFreq_iters: int = 10000,
    offsets_in_snaps: list[int] | None = None,
    show: bool = True,
):
    """Sweep measurement offset after crossing Tc and report fitted slopes.

    Offsets are specified in *snapshot counts* (integers). They are converted to seconds using
    dt inferred from the quench_log (median Δt/Δiter).
    """
    if offsets_in_snaps is None:
        offsets_in_snaps = [0, 1, 2, 5, 10]
    snapshotFreq_iters = int(snapshotFreq_iters)
    if snapshotFreq_iters < 1:
        snapshotFreq_iters = 1

    parent_abs = os.path.abspath(parent_dir or '.')
    cand_dirs = [d for d in glob.glob(os.path.join(parent_abs, pattern)) if os.path.isdir(d)]
    cand_dirs.sort()
    run_dirs = [d for d in cand_dirs if os.path.exists(os.path.join(d, 'quench_log.dat'))]
    if not run_dirs:
        raise FileNotFoundError(f"No quench runs found in {parent_abs} matching {pattern} (missing quench_log.dat)")

    # Infer dt from the first available run
    data0, _ = load_quench_log(run_dirs[0])
    it0 = np.atleast_1d(data0['iteration']).astype(float)
    t0 = np.atleast_1d(data0['time_s']).astype(float)
    if it0.size < 2:
        raise ValueError("Not enough log points to infer dt")
    dit = np.diff(it0)
    dtm = np.diff(t0)
    m = np.isfinite(dit) & np.isfinite(dtm) & (dit > 0)
    if np.count_nonzero(m) < 1:
        raise ValueError("Cannot infer dt from log")
    dt_inf = float(np.median(dtm[m] / dit[m]))
    dt_snap = dt_inf * float(snapshotFreq_iters)
    if not (np.isfinite(dt_snap) and dt_snap > 0):
        raise ValueError("Inferred dt_snap is invalid")

    offsets_s = [int(k) * dt_snap for k in offsets_in_snaps]
    slopes = []
    for k, off_s in zip(offsets_in_snaps, offsets_s):
        rows, _, _ = aggregate_kz_scaling(
            parent_dir,
            pattern=pattern,
            out_dir=out_dir,
            z_slice=z_slice,
            z_avg=z_avg,
            z_margin_frac=z_margin_frac,
            S_threshold=S_threshold,
            x_axis=x_axis,
            measure='after_Tc',
            Tc=Tc,
            after_Tc_s=float(off_s),
            show=False,
            plot=False,
            write_files=False,
        )

        arr = np.array([[r[2], r[3], r[4], r[5]] for r in rows], dtype=float)
        t_ramp = arr[:, 0]
        rate = arr[:, 1]
        xi = arr[:, 2]
        ndef = arr[:, 3]
        x = rate if x_axis.strip().lower() == 'rate' else t_ramp

        slope_xi, _ = _fit_powerlaw_loglog(x, xi)
        slope_nd, _ = _fit_powerlaw_loglog(x, ndef)
        slopes.append((int(k), float(off_s), float(slope_xi), float(slope_nd)))

    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(parent_abs) or 'runs'
    csv_name = f'slope_stability_{tag}_after_Tc_{float(Tc):g}K.csv'.replace('.', 'p')
    csv_path = os.path.join(out_dir, csv_name)
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('offset_snaps,offset_s,slope_xi,slope_defect\n')
        for k, off_s, sx, sd in slopes:
            f.write(f"{k},{off_s:.8g},{sx:.8g},{sd:.8g}\n")

    ks = np.array([s[0] for s in slopes], dtype=float)
    sx = np.array([s[2] for s in slopes], dtype=float)
    sd = np.array([s[3] for s in slopes], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.8))
    ax.plot(ks, sx, 'o-', label=r'slope($\xi$)')
    ax.plot(ks, sd, 'o-', label='slope(defects)')
    ax.set_xlabel('offset after Tc (snapshots)')
    ax.set_ylabel('fitted log–log slope')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f'Slope stability vs offset | Tc={Tc:g}K | dt_snap≈{dt_snap:.3g}s | z_avg={int(z_avg)} | margin={z_margin_frac:g}')
    out_png = os.path.join(out_dir, f'slope_stability_{tag}_after_Tc.png')
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    print(f"Saved slope stability plot -> {out_png}")
    print(f"Saved slope stability CSV  -> {csv_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return slopes, out_png, csv_path

def animate_tempSweep(Vname=None, choice=None):
    # Find all temperature directories (sorted numerically by T)
    dirs = glob.glob('output_temp_sweep/T_*/')
    if not dirs:
        print("No directories found under output_temp_sweep/T_*/")
        return

    # Sorting numerically is important (lexicographic sorting can misorder temperatures)
    dirs.sort(key=_parse_temperature_from_dir)

    z_slice = Nz // 2
    color_field = input("Choose color field for temp sweep (S, nz, n_perp) [default: S]: ").strip()
    if not color_field:
        color_field = 'S'
    color_field_norm = color_field.strip().lower()
    if color_field_norm in ('s', 'scalar', 'order', 'orderparameter'):
        color_field_norm = 's'
    elif color_field_norm in ('nz', 'n_z'):
        color_field_norm = 'nz'
    elif color_field_norm in ('nperp', 'n_perp', 'perp', 'nxy', 'n_xy'):
        color_field_norm = 'n_perp'
    else:
        print(f"Warning: Unknown color_field='{color_field}', falling back to 'S'.")
        color_field_norm = 's'

    # For S: compute a global vmax (percentile) across the sweep so frames are comparable
    global_vmin, global_vmax = None, None
    if color_field_norm == 's':
        all_vals = []
        for d in dirs:
            try:
                S_arr, _, _, _ = _load_nematic_slice_arrays(os.path.join(d, 'nematic_field_final.dat'), Nx, Ny, Nz, z_slice)
            except Exception:
                continue
            vals = S_arr[np.isfinite(S_arr)]
            # Focus scaling on droplet interior (outside droplet tends to be zeros)
            vals = vals[vals > 0.1]
            if vals.size:
                all_vals.append(vals)
        if all_vals:
            all_vals = np.concatenate(all_vals)
            global_vmin = 0.0
            global_vmax = float(np.percentile(all_vals, 99.0))
            if global_vmax <= global_vmin:
                global_vmax = float(np.max(all_vals))
            if global_vmax <= global_vmin:
                global_vmax = global_vmin + 1e-6
        else:
            global_vmin, global_vmax = 0.0, 1.0
    elif color_field_norm == 'nz':
        global_vmin, global_vmax = -1.0, 1.0
    else:
        global_vmin, global_vmax = 0.0, 1.0
    
    # If saving only selected frames, parse indices once.
    # print thze number of possible indices
    print(f"Found {len(dirs)} temperature points (indices 0 to {len(dirs)-1}).")
    selected_idxs = set()
    if choice == 'f':
        sel = input("Specify frame indices to save (e.g., 0,2,5-7 or 'all'): ").strip()
        if not sel:
            print("No indices specified; nothing will be saved.")
            return

        selected_idxs = set()
        if sel.lower() == 'all':
            selected_idxs.update(range(len(dirs)))
        else:
            parts = [p.strip() for p in sel.split(',') if p.strip()]
            for part in parts:
                if '-' in part:
                    try:
                        a, b = part.split('-')
                        start = int(a)
                        end = int(b)
                        lo, hi = (start, end) if start <= end else (end, start)
                        selected_idxs.update(range(lo, hi + 1))
                    except ValueError:
                        print(f"Invalid range '{part}', skipping.")
                else:
                    try:
                        selected_idxs.add(int(part))
                    except ValueError:
                        print(f"Invalid index '{part}', skipping.")

        # Clamp to valid range
        selected_idxs = {i for i in selected_idxs if 0 <= i < len(dirs)}
        if not selected_idxs:
            print("No valid indices in selection; nothing will be saved.")
            return

        os.makedirs('pics', exist_ok=True)

    frame_paths = []
    for idx, d in enumerate(dirs):
        S, nx, ny, nz = _load_nematic_slice_arrays(os.path.join(d, 'nematic_field_final.dat'), Nx, Ny, Nz, z_slice)
        if color_field_norm == 's':
            field = S
            cmap = 'viridis'
            cbar_label = 'Scalar Order Parameter $S$'
        elif color_field_norm == 'nz':
            field = nz
            cmap = 'RdBu_r'
            cbar_label = '$n_z$'
        else:
            field = np.sqrt(nx * nx + ny * ny)
            cmap = 'magma'
            cbar_label = r'$|n_\perp|=\sqrt{n_x^2+n_y^2}$'
        
        fig, ax = plt.subplots(figsize=(9, 8))
        # Use field.T to match plot_nematic_field_slice orientation (x=i, y=j)
        im = ax.imshow(field.T, origin='lower', cmap=cmap, extent=(0, Nx, 0, Ny), vmin=global_vmin, vmax=global_vmax)
        fig.colorbar(im, ax=ax, label=cbar_label)

        # Masking to hide directors in isotropic regions
        mask = S > 0.1
        step = max(Nx // 20, 1)
        x_grid, y_grid = np.meshgrid(np.arange(0, Nx, step), np.arange(0, Ny, step))
        
        # Apply the mask to the director components and coordinates
        x_coords = x_grid.T[mask[::step, ::step]]
        y_coords = y_grid.T[mask[::step, ::step]]
        nx_plot = nx[::step, ::step][mask[::step, ::step]]
        ny_plot = ny[::step, ::step][mask[::step, ::step]]

        if x_coords.size > 0:
            ax.quiver(x_coords, y_coords, nx_plot, ny_plot,
                    color='black', scale=30, headwidth=3, pivot='middle')

        temp_val = _parse_temperature_from_dir(d)
        temp_str = f"{temp_val:g}" if np.isfinite(temp_val) else os.path.basename(os.path.normpath(d))
        ax.set_title(f'Temperature: {temp_str} K')
        ax.set_xlabel('$x$ grid index')
        ax.set_ylabel('$y$ grid index')
        ax.set_xlim(0, Nx)
        ax.set_ylim(0, Ny)
        ax.set_aspect('equal', adjustable='box')

        if choice == 'g':
            frame_path = f'output_temp_sweep/frame_{idx:04d}.png'
            plt.savefig(frame_path)
            plt.close(fig)
            frame_paths.append(frame_path)
        elif choice == 'f':
            if idx not in selected_idxs:
                plt.close(fig)
                continue
            # Include both index and temperature in the filename
            out = f'pics/temp_sweep_frame_{idx:04d}_T_{temp_str}.png'
            if Vname:
                out = f'pics/temp_sweep_frame_{idx:04d}_T_{temp_str}_{Vname}.png'
            plt.savefig(out)
            plt.close(fig)
            print(f"Saved frame {idx} -> {out}")
        else:
            plt.close(fig)

    if choice == 'g':
        os.makedirs('pics', exist_ok=True)
        gif_path = f'pics/temp_sweep_animation_{Vname}.gif' if Vname else 'pics/temp_sweep_animation.gif'
        writer: Any = imageio.get_writer(gif_path, mode='I', duration=0.5)
        try:
            for fp in frame_paths:
                image = imageio.imread(fp)
                writer.append_data(image)
        finally:
            writer.close()

        # Clean up temporary frames
        for fp in frame_paths:
            try:
                os.remove(fp)
            except OSError:
                pass

        print(f"Animation saved to {gif_path}")

# Use imageio or ffmpeg to make an animation from the frames
# ----------------------------------------------------------------------- MAIN SCRIPT ---------------------------------------------------------------------------|
if __name__ == '__main__':
    # --- Configuration ---
    # These should match the parameters used in your C++ simulation
    Nx, Ny, Nz = 100, 100, 100 
    z_slice_to_plot = Nz // 2

    do_biax = input("Compute biaxiality report from Qtensor_output_final.dat? (y/n): ").lower() == 'y'
    if do_biax:
        biaxiality_report('Qtensor_output_final.dat', Nx=Nx, Ny=Ny, Nz=Nz, z_slice=z_slice_to_plot)

    # create pics directory if it doesn't exist
    if not os.path.exists('pics'):
        os.makedirs('pics')
    i = input("Enter the number of the plot you want to create:\n"
              "0: Plot final state from pre-calculated Nematic Field data\n"
              "1: Create animation from Nematic Field data\n"
              "2: Plot Free Energy vs Iteration\n"
              "3: Plot Energy Components vs Iteration\n"
              "4: Plot Average S and Free Energy vs Temperature\n"
              "5: Animate Temperature Sweep\n"
              "6: Plot Quench Log (T, energies, radiality, <S>)\n"
              "7: Create Quench animation (GIF)\n"
              "8: Plot KZ metrics from quench (xi + defect density)\n"
              "9: Aggregate KZ scaling across runs (log-log fit)\n"
              "10: Sweep KZ slope stability vs offset after Tc\n"
              "11: 3D metrics from a snapshot (xi_3D + defect-line proxy)\n"
              "12: Aggregate KZ scaling using 3D metrics (xi_3D + defect-line proxy)\n"
              "Enter your choice (0-12): ").strip()
    while not i.isdigit() or int(i) < 0 or int(i) > 12:
        print("Invalid input. Please enter a number between 0 and 12.")
        i = input("Please enter a number between 0 and 12: ").strip()
    i = int(i)
# ---------------------------------------------------------------------- PART 1 ----------------------------------------------------------------------------|
    if i == 0:
        # --- Plotting Individual Final States ---
        print("Plotting final state from pre-calculated Nematic Field data...")
        ax_in = input("View axis / look along (x/y/z) [default: z]: ").strip().lower()
        if not ax_in:
            ax_in = 'z'
        axis = ax_in[0] if ax_in else 'z'
        if axis not in ('x', 'y', 'z'):
            print("Invalid axis; using 'z'.")
            axis = 'z'

        if axis == 'z':
            max_idx = Nz - 1
            default_idx = Nz // 2
            axis_label = 'z'
        elif axis == 'y':
            max_idx = Ny - 1
            default_idx = Ny // 2
            axis_label = 'y'
        else:
            max_idx = Nx - 1
            default_idx = Nx // 2
            axis_label = 'x'

        sl_in = input(f"{axis_label}-slice index (0..{max_idx}) [default: {default_idx}]: ").strip()
        try:
            slice_idx = int(sl_in) if sl_in else int(default_idx)
        except ValueError:
            slice_idx = int(default_idx)
        slice_idx = max(0, min(int(max_idx), int(slice_idx)))

        do_zoom = input("Zoom into center? (y/n): ").lower() == 'y'
        radius = 15 if do_zoom else None # +/- 15 units = 30x30 window
        interpol=input("Choose interpolation method (nearest, bilinear, bicubic, spline16, spline36, sinc) [default: nearest]: ").strip()
        if not interpol: # if empty, use default
            interpol='nearest'
        color_field = input("Choose color field (S, nz, n_perp) [default: S]: ").strip()
        if not color_field:
            color_field = 'S'
        custom_name = input("Add custom name to append to file-name for multi-sumulation scenario: ").strip()
        if not custom_name:
            custom_name = ''
        plot_nematic_field_slice(
            filename='nematic_field_final.dat',
            Nx=Nx, Ny=Ny, Nz=Nz,
            z_slice=slice_idx,
            slice_axis=axis_label,
            output_path=f'pics/{color_field}_final_state_{custom_name}.png',
            arrowColor='black' if not do_zoom else 'black',
            zoom_radius=radius,
            interpol=interpol,
            color_field=color_field,
            print_stats=True,
            )
    elif i == 1:
        # --- Creating Animation ---
        inpt = input("\nPlease specify filename for the animation gif: ").strip()
        if not inpt:
            inpt = 'nematic_field_evolution'
        if not inpt.lower().endswith('.gif'):
            out_gif = f'pics/{inpt}.gif'
        else:
            out_gif = os.path.join('pics', inpt)
        print("\nCreating animation from Nematic Field data...")
        try:
            create_nematic_field_animation(
                data_dir='output',
                output_gif=out_gif,
                Nx=Nx, Ny=Ny, Nz=Nz)
        except Exception as e:
            print(f"Error creating animation: {e}")
    elif i == 2:
        # --- Plotting Free Energy vs Iteration ---
        print("\nPlotting Free Energy vs Iteration...")
        try:
            in_path = prompt_pick_folder(
                prompt="Folder/file for energy-vs-iteration (free_energy_vs_iteration.dat or output_quench*/quench_log.*)",
                default='.',
                pattern='*',
                must_exist=True,
            )
            plot_energy_VS_iter(in_path, out_dir='pics', show=True)
        except Exception as e:
            print(f"Error plotting free energy vs iteration: {e}")
# ---------------------------------------------------------------------- PART 2 -----------------------------------------------------------------------------|
    elif i == 3:
        # --- Plotting Energy Components vs Iteration ---
        print("\nPlotting Energy Components vs Iteration...")
        energy_components()
# ---------------------------------------------------------------------- PART 3 -----------------------------------------------------------------------------|
    elif i == 4:
        # --- Plotting S vs Temperature and Free Energy ---
        print("\nPlotting Average S and Free Energy vs Temperature...")
        try:
            in_path = prompt_pick_folder(
                prompt="Folder/file for S(T),F(T) (output_temp_sweep/summary.dat OR output_quench*/quench_log.*)",
                default='output_temp_sweep',
                pattern='output_*',
                must_exist=True,
            )
            plotS_F(in_path, out_dir='pics', show=True)
        except Exception as e:
            print(f"Error plotting sweep summary: {e}")
    elif i == 5:
        # --- Animate Temperature Sweep ---
        # Ask until we get a valid choice
        while True:
            ch = input("Want to create a GIF, or save specific frames? (g/f): ").strip().lower()
            if ch in ('g', 'f'):
                break
            print("Invalid choice. Please enter 'g' for GIF or 'f' for frames.")

        V_name = input("Specify name to append to end of file-names for multi-simulation scenario (press Enter for none): ").strip()

        if ch == 'f':
            print("\nSaving Selected Frames from Temperature Sweep...")
        elif ch == 'g':
            print("\nAnimating Temperature Sweep...")
        animate_tempSweep(Vname=V_name, choice=ch)
# ---------------------------------------------------------------------- PART 4 --------------------------------------------------------------------------------|
    elif i == 6:
        # --- Plot Quench Log ---
        print("\nPlotting quench log...")
        try:
            mode = input(
                "Choose quench plot:\n"
                "  (s) summary vs time\n"
                "  (2) energies vs iteration\n"
                "  (4) <S> vs iteration\n"
                "  (d) energy deltas (ΔE per log step) — useful for defect-annihilation analysis: spikes often show up in Δelastic even when total looks smooth\n"
                "  (a) all\n"
                "[default: s]: "
            ).strip().lower()
            if not mode:
                mode = 's'

            in_path = prompt_pick_folder(
                prompt="Quench run folder (or quench_log.* path)",
                default='output_quench',
                pattern='output_quench*',
                must_exist=True,
            )

            win_in = input("Iteration window min:max (blank=all, examples: 50000:70000, :70000, 50000:) [default: all]: ").strip()
            it_min, it_max = _parse_iter_window(win_in)

            if mode in ('s', 'a'):
                plot_quench_log(in_path, out_dir='pics', show=True, it_min=it_min, it_max=it_max)
            if mode in ('2', 'a'):
                plot_quench_energy_vs_iteration(in_path, out_dir='pics', show=True, it_min=it_min, it_max=it_max)
            if mode in ('4', 'a'):
                plot_quench_order_vs_iteration(in_path, out_dir='pics', show=True, it_min=it_min, it_max=it_max)
            if mode in ('d', 'a'):
                plot_quench_energy_deltas(in_path, out_dir='pics', show=True, it_min=it_min, it_max=it_max)
        except Exception as e:
            print(f"Error plotting quench log: {e}")
    elif i == 7:
        # --- Create Quench Animation ---
        print("\nCreating quench animation (GIF) from output_quench/nematic_field_iter_*.dat...")
        data_dir = input("Data directory [default: output_quench]: ").strip()
        if not data_dir:
            data_dir = 'output_quench'

        # Auto-detect grid size from first snapshot (helps if Nx,Ny,Nz differ from defaults)
        Nx_use, Ny_use, Nz_use = Nx, Ny, Nz
        try:
            snap_files = glob.glob(os.path.join(data_dir, 'nematic_field_iter_*.dat'))
            if snap_files:
                def _snap_iter_num(p: str) -> int:
                    m = re.search(r'(\d+)', os.path.basename(p))
                    return int(m.group(1)) if m else -1
                snap_files.sort(key=_snap_iter_num)
                Nx_use, Ny_use, Nz_use = infer_grid_dims_from_nematic_field_file(snap_files[0])
                print(f"Detected grid: Nx={Nx_use}, Ny={Ny_use}, Nz={Nz_use} (from {os.path.basename(snap_files[0])})")
        except Exception as e:
            print(f"Warning: could not auto-detect grid dims ({e}); using defaults Nx={Nx}, Ny={Ny}, Nz={Nz}.")

        inpt = input("Please specify filename for the quench animation gif [default: quench_evolution]: ").strip()
        if not inpt:
            inpt = 'quench_evolution'
        if not inpt.lower().endswith('.gif'):
            out_gif = f'pics/{inpt}.gif'
        else:
            out_gif = os.path.join('pics', inpt)

        color_field = input("Choose color field (S, nz, n_perp) [default: S]: ").strip()
        if not color_field:
            color_field = 'S'
        interpol = input("Choose interpolation method (nearest, bilinear, bicubic, spline16, spline36, sinc) [default: nearest]: ").strip()
        if not interpol:
            interpol = 'nearest'
        do_zoom = input("Zoom into center? (y/n): ").lower() == 'y'
        zoom_radius = 15 if do_zoom else None
        dur = input("Frame duration in seconds [default: 0.08]: ").strip()
        try:
            duration = float(dur) if dur else 0.08
        except ValueError:
            duration = 0.08
        stride_in = input("Use every N-th snapshot (frame stride) [default: 1]: ").strip()
        try:
            stride = int(stride_in) if stride_in else 1
        except ValueError:
            stride = 1

        arrows_in = input("Number of arrows per axis (0=disable quiver) [default: 20]: ").strip()
        try:
            arrows_per_axis = int(arrows_in) if arrows_in else 20
        except ValueError:
            arrows_per_axis = 20

        try:
            output_c=input("Per how many itterations do you want output in the console? (default: 10): ").strip()
            if not output_c:
                output_c = 10
            else:
                output_c = int(output_c)
            create_nematic_field_animation(
                data_dir=data_dir,
                output_gif=out_gif,
                Nx=Nx_use, Ny=Ny_use, Nz=Nz_use,
                frames_dir='frames_quench',
                duration=duration,
                frame_stride=stride,
                color_field=color_field,
                interpol=interpol,
                zoom_radius=zoom_radius,
                arrowColor='black',
                arrows_per_axis=arrows_per_axis,
                consistent_scale=True,
                output_c=output_c,
            )
        except Exception as e:
            print(f"Error creating quench animation: {e}")
    elif i == 8:
        kzin = input("\nPlot a single slice from quench snapshots, or compute KZ metrics? (snap/kz_metrics) [default: kz_metrics]: ").strip().lower()
        while kzin not in ('snap', 'kz_metrics', ''):
            print("Invalid input. Please enter 'snap' or 'kz_metrics'.")
            kzin = input("Please enter 'snap' or 'kz_metrics': ").strip().lower()
        if kzin in ('kz_metrics', ''):
            print("\nComputing KZ metrics (2D proxies) from quench snapshots...")
            in_path = input("Path to quench directory or quench_log.dat [default: output_quench]: ").strip()
            if not in_path:
                in_path = 'output_quench'
            stride_in = input("Use every N-th snapshot (frame stride) [default: 10]: ").strip()
            try:
                stride = int(stride_in) if stride_in else 10
            except ValueError:
                stride = 10
            max_in = input("Max frames to analyze (blank=no limit) [default: 50]: ").strip()
            try:
                max_frames = int(max_in) if max_in else 50
            except ValueError:
                max_frames = 50
            sth_in = input("S threshold for droplet mask [default: 0.1]: ").strip()
            try:
                sthr = float(sth_in) if sth_in else 0.1
            except ValueError:
                sthr = 0.1
            try:
                plot_quench_kz_metrics(in_path, out_dir='pics', frame_stride=stride, max_frames=max_frames, S_threshold=sthr, show=True)
            except Exception as e:
                print(f"Error computing KZ metrics: {e}")
        elif kzin == 'snap':
            in_path = input("Path to quench directory or quench_log.dat [default: output_quench]: ").strip()
            if not in_path:
                in_path = 'output_quench'
            try:
                do_snap = input("Plot a z-slice from one of the available nematic_field_iter_*.dat snapshots? (y/n): ").strip().lower()
                if do_snap == 'y':
                    prompt_plot_snapshot_slice_from_dir(in_path, out_dir='pics', show=True)
            except Exception as e:
                print(f"Error plotting snapshot slice: {e}")
# ----------------------------------------------------------------------- PART 5 ------------------------------------------------------------------------------|
    elif i == 9:
        print("\nAggregating KZ scaling across multiple quench runs...")
        parent = input("Parent directory to scan [default: .]: ").strip() or '.'
        pattern = input("Folder pattern [default: output_quench*]: ").strip() or 'output_quench*'
        x_axis = input("X-axis (t_ramp or rate) [default: t_ramp]: ").strip() or 't_ramp'
        measure = input("Measure state (final / after_Tlow / after_Tc) [default: final]: ").strip() or 'final'
        after_Tlow_s = 0.0
        Tc = None
        after_Tc_s = 0.0
        if measure.strip().lower() in ('after_tlow', 'after_t_low', 'tlow', 'after_final_t'):
            off_in = input("Time offset after reaching T_low in seconds [default: 0.0]: ").strip()
            try:
                after_Tlow_s = float(off_in) if off_in else 0.0
            except ValueError:
                after_Tlow_s = 0.0
        elif measure.strip().lower() in ('after_tc', 'after_t_c', 'tc', 'after_transition'):
            tc_in = input("Transition temperature Tc in Kelvin [default: 310.2]: ").strip()
            try:
                Tc = float(tc_in) if tc_in else 310.2
            except ValueError:
                Tc = 310.2
            off_in = input("Time offset after crossing Tc in seconds [default: 0.0]: ").strip()
            try:
                after_Tc_s = float(off_in) if off_in else 0.0
            except ValueError:
                after_Tc_s = 0.0
        sth_in = input("S threshold for droplet mask [default: 0.1]: ").strip()
        try:
            sthr = float(sth_in) if sth_in else 0.1
        except ValueError:
            sthr = 0.1
        z_in = input("z-slice (blank=mid-plane): ").strip()
        z_slice = None
        if z_in:
            try:
                z_slice = int(z_in)
            except ValueError:
                z_slice = None

        zavg_in = input("Average over N z-slices (1=off) [default: 1]: ").strip()
        try:
            z_avg = int(zavg_in) if zavg_in else 1
        except ValueError:
            z_avg = 1
        if z_avg < 1:
            z_avg = 1

        fit_x_min = None
        fit_x_max = None
        if x_axis.strip().lower() == 'rate':
            fx1 = input("Fit window min |dT/dt| [K/s] (blank=all): ").strip()
            fx2 = input("Fit window max |dT/dt| [K/s] (blank=all): ").strip()
        else:
            fx1 = input("Fit window min t_ramp [s] (blank=all): ").strip()
            fx2 = input("Fit window max t_ramp [s] (blank=all): ").strip()
        try:
            fit_x_min = float(fx1) if fx1 else None
        except ValueError:
            fit_x_min = None
        try:
            fit_x_max = float(fx2) if fx2 else None
        except ValueError:
            fit_x_max = None

        z_margin_frac = 0.0
        if z_avg > 1:
            zm_in = input("Exclude outer z band fraction (0..0.45) [default: 0.2]: ").strip()
            try:
                z_margin_frac = float(zm_in) if zm_in else 0.2
            except ValueError:
                z_margin_frac = 0.2
            z_margin_frac = max(0.0, min(0.45, z_margin_frac))
        try:
            aggregate_kz_scaling(
                parent,
                pattern=pattern,
                out_dir='pics',
                z_slice=z_slice,
                z_avg=z_avg,
                z_margin_frac=z_margin_frac,
                S_threshold=sthr,
                x_axis=x_axis,
                fit_x_min=fit_x_min,
                fit_x_max=fit_x_max,
                measure=measure,
                after_Tlow_s=after_Tlow_s,
                Tc=Tc,
                after_Tc_s=after_Tc_s,
                show=True,
            )
        except Exception as e:
            print(f"Error aggregating KZ scaling: {e}")
    elif i == 10:
        print("\nSweeping KZ slope stability vs offset after Tc...")
        parent = input("Parent directory to scan [default: .]: ").strip() or '.'
        pattern = input("Folder pattern [default: output_quench*]: ").strip() or 'output_quench*'
        x_axis = input("X-axis (t_ramp or rate) [default: t_ramp]: ").strip() or 't_ramp'
        tc_in = input("Transition temperature Tc in Kelvin [default: 310.2]: ").strip()
        try:
            Tc = float(tc_in) if tc_in else 310.2
        except ValueError:
            Tc = 310.2
        sf_in = input("Snapshot frequency in iterations [default: 10000]: ").strip()
        try:
            snapshotFreq_iters = int(float(sf_in)) if sf_in else 10000
        except ValueError:
            snapshotFreq_iters = 10000
        sth_in = input("S threshold for droplet mask [default: 0.02]: ").strip()
        try:
            sthr = float(sth_in) if sth_in else 0.02
        except ValueError:
            sthr = 0.02
        z_in = input("z-slice (blank=mid-plane): ").strip()
        z_slice = None
        if z_in:
            try:
                z_slice = int(z_in)
            except ValueError:
                z_slice = None
        zavg_in = input("Average over N z-slices [default: 11]: ").strip()
        try:
            z_avg = int(zavg_in) if zavg_in else 11
        except ValueError:
            z_avg = 11
        if z_avg < 1:
            z_avg = 1
        zm_in = input("Exclude outer z band fraction (0..0.45) [default: 0.2]: ").strip()
        try:
            z_margin_frac = float(zm_in) if zm_in else 0.2
        except ValueError:
            z_margin_frac = 0.2
        z_margin_frac = max(0.0, min(0.45, z_margin_frac))

        off_in = input("Offsets after Tc in snapshots (comma-separated) [default: 0,1,2,5,10]: ").strip()
        offsets_in_snaps = None
        if off_in:
            try:
                offsets_in_snaps = [int(s.strip()) for s in off_in.split(',') if s.strip()]
            except ValueError:
                offsets_in_snaps = None

        try:
            sweep_kz_slope_stability(
                parent,
                pattern=pattern,
                out_dir='pics',
                x_axis=x_axis,
                S_threshold=sthr,
                z_slice=z_slice,
                z_avg=z_avg,
                z_margin_frac=z_margin_frac,
                Tc=Tc,
                snapshotFreq_iters=snapshotFreq_iters,
                offsets_in_snaps=offsets_in_snaps,
                show=True,
            )
        except Exception as e:
            print(f"Error sweeping slope stability: {e}")
    elif i == 11:
        print("\n3D metrics from a snapshot (xi_3D + defect-line proxy)...")
        in_path = input("Path to field file OR run directory OR quench_log.dat [default: output_quench]: ").strip()
        if not in_path:
            in_path = 'output_quench'

        run_dir = _resolve_run_dir_from_path(in_path)
        field_fp = None

        if os.path.isdir(run_dir):
            # Choose a field file from the directory
            final_fp = os.path.join(run_dir, 'nematic_field_final.dat')
            snaps = _list_snapshot_files(run_dir)
            have_final = os.path.exists(final_fp)
            if have_final:
                default_pick = 'f'
            elif snaps:
                default_pick = 'n'
            else:
                default_pick = 'q'

            # Try to allow time-based selection if log exists
            have_log = False
            it_log = np.empty(0, dtype=float)
            t_log = np.empty(0, dtype=float)
            T_log = np.empty(0, dtype=float)
            try:
                data, log_path = load_quench_log(run_dir)
                it_log = np.atleast_1d(data['iteration']).astype(float)
                t_log = np.atleast_1d(data['time_s']).astype(float)
                T_log = np.atleast_1d(data['T_K']).astype(float)
                have_log = True
                print(f"Detected quench log: {os.path.basename(str(log_path))}")
            except Exception:
                have_log = False

            prompt = "Select field file: (f) final, (n) snapshot index"
            if have_log and snaps:
                prompt += ", (a) after Tc+offset"
                if default_pick == 'n':
                    default_pick = 'a'
            prompt += f", (q) quit [default: {default_pick}]: "
            pick = input(prompt).strip().lower() or default_pick
            if pick == 'q':
                field_fp = None
            elif pick == 'f' and have_final:
                field_fp = final_fp
            elif pick == 'a':
                if not (have_log and snaps):
                    print("Tc-based selection requires a quench log and snapshots.")
                    field_fp = None
                else:
                    tc_in = input("Tc [K] [default: 310.2]: ").strip()
                    try:
                        Tc = float(tc_in) if tc_in else 310.2
                    except ValueError:
                        Tc = 310.2
                    off_in = input("Time offset after crossing Tc [s] [default: 0.0]: ").strip()
                    try:
                        after_Tc_s = float(off_in) if off_in else 0.0
                    except ValueError:
                        after_Tc_s = 0.0
                    t_cross = _crossing_time_from_log(T_log, t_log, float(Tc))
                    t_meas = float(t_cross) + float(after_Tc_s)
                    field_fp = _select_snapshot_by_time(run_dir, t_meas, it_log, t_log)
                    print(f"Selected snapshot closest to t_cross+offset={t_meas:.6g}s -> {os.path.basename(field_fp)}")
            else:
                if not snaps:
                    print("No nematic_field_iter_*.dat snapshots found.")
                    field_fp = final_fp if have_final else None
                else:
                    idx_in = input(f"Snapshot index [0..{len(snaps)-1}] [default: {len(snaps)-1}]: ").strip()
                    try:
                        idx = int(idx_in) if idx_in else (len(snaps) - 1)
                    except ValueError:
                        idx = len(snaps) - 1
                    idx = max(0, min(len(snaps) - 1, idx))
                    _, field_fp = snaps[idx]
        else:
            field_fp = in_path

        if not field_fp or not os.path.exists(field_fp):
            print("No valid field file selected.")
        else:
            base = os.path.splitext(os.path.basename(field_fp))[0]

            do_xi3 = input("Compute 3D correlation length xi_3D (Q-tensor correlation)? (y/n) [default: y]: ").strip().lower()
            if not do_xi3:
                do_xi3 = 'y'
            if do_xi3 == 'y':
                sth_in = input("S threshold for droplet mask (used for masking correlation) [default: 0.1]: ").strip()
                try:
                    sthr = float(sth_in) if sth_in else 0.1
                except ValueError:
                    sthr = 0.1
                mr_in = input("Max r to consider (blank=auto): ").strip()
                try:
                    max_r = int(float(mr_in)) if mr_in else None
                except ValueError:
                    max_r = None

                try:
                    xi3, r, C_r, dims = correlation_length_3d_from_field_file(field_fp, S_threshold=sthr, max_r=max_r)
                    print(f"[xi_3D] {os.path.basename(field_fp)} dims={dims} -> xi_3D ≈ {xi3:.6g} (lattice units)")

                    os.makedirs('pics', exist_ok=True)
                    csv_path = os.path.join('pics', f'xi3d_corr_{base}.csv')
                    with open(csv_path, 'w', encoding='utf-8') as f:
                        f.write('r_lattice,C_r\n')
                        for rr, cc in zip(r, C_r):
                            if np.isfinite(rr) and np.isfinite(cc):
                                f.write(f"{float(rr):.8g},{float(cc):.8g}\n")
                    print(f"Saved 3D correlation CSV -> {csv_path}")

                    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.6))
                    ax.plot(r, C_r, 'o-', ms=3)
                    ax.axhline(np.e ** (-1.0), color='k', lw=1, ls='--', alpha=0.6, label=r'$1/e$')
                    if np.isfinite(xi3):
                        ax.axvline(xi3, color='tab:red', lw=1.5, ls='--', alpha=0.8, label=fr'$\xi_{{3D}}\approx{xi3:.3g}$')
                    ax.set_xlabel('r (lattice units)')
                    ax.set_ylabel('C(r)')
                    ax.set_title(f'3D Q-correlation | {base} | S>{sthr:g}')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    fig.tight_layout()
                    out_png = os.path.join('pics', f'xi3d_corr_{base}.png')
                    fig.savefig(out_png, dpi=220)
                    plt.show()
                except Exception as e:
                    print(f"Error computing xi_3D: {e}")

            do_def3 = input("Compute 3D defect-line proxy (skeletonized low-S core)? (y/n) [default: y]: ").strip().lower()
            if not do_def3:
                do_def3 = 'y'
            if do_def3 == 'y':
                sth_in = input("S threshold for droplet identification [default: 0.1]: ").strip()
                try:
                    S_d = float(sth_in) if sth_in else 0.1
                except ValueError:
                    S_d = 0.1
                sc_in = input("S threshold for core voxels (S < S_core) [default: 0.05]: ").strip()
                try:
                    S_c = float(sc_in) if sc_in else 0.05
                except ValueError:
                    S_c = 0.05
                di_in = input("Dilate droplet by N voxels to include nearby core [default: 2]: ").strip()
                try:
                    dil = int(float(di_in)) if di_in else 2
                except ValueError:
                    dil = 2
                mv_in = input("Min core component size (voxels) [default: 30]: ").strip()
                try:
                    minv = int(float(mv_in)) if mv_in else 30
                except ValueError:
                    minv = 30

                try:
                    out = defect_line_metrics_3d_from_field_file(
                        field_fp,
                        S_droplet=S_d,
                        S_core=S_c,
                        dilate_iters=dil,
                        min_core_voxels=minv,
                        use_skeleton=True,
                    )
                    print(
                        f"[defect_3D] droplet_vox={out.get('droplet_voxels')} core_vox={out.get('core_voxels')} "
                        f"skel_vox={out.get('skeleton_voxels')} length≈{out.get('line_length_lattice'):.6g} "
                        f"density≈{out.get('line_density_per_voxel'):.6g}"
                    )

                    os.makedirs('pics', exist_ok=True)
                    csv_path = os.path.join('pics', f'defect3d_{base}.csv')
                    with open(csv_path, 'w', encoding='utf-8') as f:
                        for k, v in out.items():
                            f.write(f"{k},{v}\n")
                    print(f"Saved 3D defect-line CSV -> {csv_path}")
                except Exception as e:
                    print(f"Error computing 3D defect-line proxy: {e}")
    elif i == 12:
        print("\nAggregating KZ scaling using 3D metrics (xi_3D + defect-line proxy) across runs...")
        parent = input("Parent directory to scan [default: .]: ").strip() or '.'
        pattern = input("Folder pattern [default: output_quench*]: ").strip() or 'output_quench*'
        x_axis = input("X-axis (t_ramp or rate) [default: t_ramp]: ").strip() or 't_ramp'
        measure = input("Measure state (final / after_Tlow / after_Tc) [default: after_Tc]: ").strip() or 'after_Tc'

        after_Tlow_s = 0.0
        Tc = None
        after_Tc_s = 0.0
        after_Tc_mode = 'fixed'
        after_Tc_frac_ramp = 0.1
        avgS_target = 0.1
        extra_after_target_s = 0.0
        if measure.strip().lower() in ('after_tlow', 'after_t_low', 'tlow', 'after_final_t'):
            off_in = input("Time offset after reaching T_low in seconds [default: 0.0]: ").strip()
            try:
                after_Tlow_s = float(off_in) if off_in else 0.0
            except ValueError:
                after_Tlow_s = 0.0
        elif measure.strip().lower() in ('after_tc', 'after_t_c', 'tc', 'after_transition'):
            tc_in = input("Transition temperature Tc in Kelvin [default: 310.2]: ").strip()
            try:
                Tc = float(tc_in) if tc_in else 310.2
            except ValueError:
                Tc = 310.2
            mode_in = input("After-Tc selection mode: fixed / frac / avgS / auto [default: auto]: ").strip().lower()
            after_Tc_mode = mode_in if mode_in else 'auto'
            if after_Tc_mode in ('fixed', 'f'):
                off_in = input("Time offset after crossing Tc in seconds [default: 0.0]: ").strip()
                try:
                    after_Tc_s = float(off_in) if off_in else 0.0
                except ValueError:
                    after_Tc_s = 0.0
            elif after_Tc_mode in ('frac', 'fraction', 'frac_ramp', 'ramp_frac', 'r'):
                fr_in = input("Offset after Tc as fraction of ramp time t_ramp [default: 0.1]: ").strip()
                try:
                    after_Tc_frac_ramp = float(fr_in) if fr_in else 0.1
                except ValueError:
                    after_Tc_frac_ramp = 0.1
            elif after_Tc_mode in ('avg_s', 'avgs', 's', 'order'):
                s_in = input("Measure when avg_S first exceeds threshold [default: 0.1]: ").strip()
                try:
                    avgS_target = float(s_in) if s_in else 0.1
                except ValueError:
                    avgS_target = 0.1
                ex_in = input("Extra time after reaching avg_S threshold [s] [default: 0.0]: ").strip()
                try:
                    extra_after_target_s = float(ex_in) if ex_in else 0.0
                except ValueError:
                    extra_after_target_s = 0.0
            else:
                # auto: max(frac*t_ramp, time to reach avg_S threshold) + extra
                fr_in = input("AUTO: fraction of ramp time t_ramp [default: 0.1]: ").strip()
                try:
                    after_Tc_frac_ramp = float(fr_in) if fr_in else 0.1
                except ValueError:
                    after_Tc_frac_ramp = 0.1
                s_in = input("AUTO: avg_S threshold [default: 0.1]: ").strip()
                try:
                    avgS_target = float(s_in) if s_in else 0.1
                except ValueError:
                    avgS_target = 0.1
                ex_in = input("AUTO: extra time after threshold [s] [default: 0.0]: ").strip()
                try:
                    extra_after_target_s = float(ex_in) if ex_in else 0.0
                except ValueError:
                    extra_after_target_s = 0.0

        # xi_3D params
        sth_in = input("S threshold for xi_3D masking (S > S_threshold_xi) [default: 0.1]: ").strip()
        try:
            S_threshold_xi = float(sth_in) if sth_in else 0.1
        except ValueError:
            S_threshold_xi = 0.1
        mr_in = input("Max r for xi_3D correlation (blank=auto) [default: blank]: ").strip()
        try:
            max_r = int(float(mr_in)) if mr_in else None
        except ValueError:
            max_r = None

        # defect proxy params
        dp_in = input("Defect proxy (skeleton or core_density) [default: skeleton]: ").strip().lower()
        defect_proxy = dp_in if dp_in else 'skeleton'

        sd_in = input("S threshold for droplet identification (S > S_droplet) [default: 0.1]: ").strip()
        try:
            S_droplet = float(sd_in) if sd_in else 0.1
        except ValueError:
            S_droplet = 0.1
        sc_in = input("S threshold for core voxels (S < S_core) [default: 0.05]: ").strip()
        try:
            S_core = float(sc_in) if sc_in else 0.05
        except ValueError:
            S_core = 0.05
        di_in = input("Dilate droplet by N voxels [default: 2]: ").strip()
        try:
            dilate_iters = int(float(di_in)) if di_in else 2
        except ValueError:
            dilate_iters = 2

        fh_in = input("Fill holes in droplet mask (helps include internal low-S cores) (y/n) [default: n]: ").strip().lower()
        fill_holes = True if fh_in in ('y', 'yes', '1', 'true', 't') else False

        ero_in = input("Erode droplet by N voxels before core detection (avoid interface sheets) [default: 0]: ").strip()
        try:
            core_erosion_iters = int(float(ero_in)) if ero_in else 0
        except ValueError:
            core_erosion_iters = 0
        mv_in = input("Min core component size (voxels) [default: 30]: ").strip()
        try:
            min_core_voxels = int(float(mv_in)) if mv_in else 30
        except ValueError:
            min_core_voxels = 30

        max_runs = None
        mruns_in = input("Max number of runs to process (blank=all): ").strip()
        try:
            max_runs = int(float(mruns_in)) if mruns_in else None
        except ValueError:
            max_runs = None

        fit_x_min = None
        fit_x_max = None
        if x_axis.strip().lower() == 'rate':
            fx1 = input("Fit window min |dT/dt| [K/s] (blank=all): ").strip()
            fx2 = input("Fit window max |dT/dt| [K/s] (blank=all): ").strip()
        else:
            fx1 = input("Fit window min t_ramp [s] (blank=all): ").strip()
            fx2 = input("Fit window max t_ramp [s] (blank=all): ").strip()
        try:
            fit_x_min = float(fx1) if fx1 else None
        except ValueError:
            fit_x_min = None
        try:
            fit_x_max = float(fx2) if fx2 else None
        except ValueError:
            fit_x_max = None

        try:
            aggregate_kz_scaling_3d(
                parent,
                pattern=pattern,
                out_dir='pics',
                x_axis=x_axis,
                fit_x_min=fit_x_min,
                fit_x_max=fit_x_max,
                measure=measure,
                after_Tlow_s=after_Tlow_s,
                Tc=Tc,
                after_Tc_s=after_Tc_s,
                after_Tc_mode=after_Tc_mode,
                after_Tc_frac_ramp=after_Tc_frac_ramp,
                avgS_target=avgS_target,
                extra_after_target_s=extra_after_target_s,
                S_threshold_xi=S_threshold_xi,
                max_r=max_r,
                S_droplet=S_droplet,
                S_core=S_core,
                dilate_iters=dilate_iters,
                fill_holes=fill_holes,
                core_erosion_iters=core_erosion_iters,
                min_core_voxels=min_core_voxels,
                defect_proxy=defect_proxy,
                max_runs=max_runs,
                show=True,
            )
        except Exception as e:
            print(f"Error aggregating 3D KZ scaling: {e}")
    # ---------------------------------------------------------------------- FIN --------------------------------------------------------------------------------|