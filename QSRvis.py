import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import imageio.v2 as imageio
import re
import io
from typing import Any

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
    if z_slice is None:
        z_slice = Nz // 2

    # Load the raw data from the simulation output.
    # Columns are: i, j, k, S, nx, ny, nz
    try:
        # Use comments='#' to ignore the header line in the data file
        data = load_nematic_field_data(filename, comments="#")
    except IOError:
        print(f"Warning: Could not read file '{filename}'. Skipping.")
        return

    # Select the data for the desired z-slice
    slice_data = data[data[:, 2] == z_slice]

    if slice_data.shape[0] == 0:
        print(f"Warning: No data found for z_slice = {z_slice} in {filename}. Skipping.")
        return

    # Prepare 2D arrays for S, n
    S = np.zeros((Nx, Ny))
    nx = np.zeros((Nx, Ny))
    ny = np.zeros((Nx, Ny))
    nz = np.zeros((Nx, Ny))

    # Directly populate arrays from the file data
    for row in slice_data:
        i, j = int(row[0]), int(row[1])
        if i < Nx and j < Ny:
            S[i, j] = row[3]
            nx[i, j] = row[4]
            ny[i, j] = row[5]
            nz[i, j] = row[6]

    if print_stats:
        ci, cj = Nx // 2, Ny // 2
        if 0 <= ci < Nx and 0 <= cj < Ny:
            print(
                f"[{os.path.basename(filename)} | z={z_slice}] "
                f"S_center={S[ci, cj]:.6g}, "
                f"S_min={np.min(S):.6g}, S_max={np.max(S):.6g}; "
                f"n_center=({nx[ci, cj]:.4g},{ny[ci, cj]:.4g},{nz[ci, cj]:.4g}), "
                f"|n_perp|_center={np.sqrt(nx[ci, cj]**2 + ny[ci, cj]**2):.6g}"
            )

    # --- Handle zooming ---
    if zoom_radius is not None:
        center_x, center_y = Nx // 2, Ny // 2
        x_min = max(0, center_x - zoom_radius)
        x_max = min(Nx, center_x + zoom_radius)
        y_min = max(0, center_y - zoom_radius)
        y_max = min(Ny, center_y + zoom_radius)

        # Slice the arrays
        S_view = S[x_min:x_max, y_min:y_max]
        nx_view = nx[x_min:x_max, y_min:y_max]
        ny_view = ny[x_min:x_max, y_min:y_max]
        nz_view = nz[x_min:x_max, y_min:y_max]

        extent = (x_min, x_max, y_min, y_max)
        step = 1  # default: dense in zoomed view
    else:
        S_view = S
        nx_view = nx
        ny_view = ny
        nz_view = nz
        extent = (0, Nx, 0, Ny)

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
        field = np.sqrt(nx_view * nx_view + ny_view * ny_view)
        cmap = 'magma'
        label = r'$|n_\perp|=\sqrt{n_x^2+n_y^2}$'
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
        nx_plot = nx_view[np.ix_(ix, iy)].T
        ny_plot = ny_view[np.ix_(ix, iy)].T
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
    title = f"Nematično polje pri ($z={z_slice}$, Iter: {iter_str}, barva: {color_field_norm})"
    if zoom_radius: title += " [ZOOMED]"
        
    ax.set_title(title)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(0, Nx)
    ax.set_ylim(0, Ny)
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

def plot_energy_VS_iter():
    # Load the data, skipping the header
    # Format: iteration,free_energy,radiality,time
    data = np.genfromtxt('free_energy_vs_iteration.dat', delimiter=',', names=True)

    # Create subplots for energy, radiality, and time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Free Energy vs Iteration
    ax1.plot(data['iteration'], data['free_energy'], marker='o', linestyle='-', color='blue')
    ax1.set_xlabel('$i$')
    ax1.set_ylabel('$F$ [J]', color='blue')
    ax1.set_title('$F(i)$')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    
    # Plot radiality on secondary y-axis
    ax1_twin = ax1.twinx()
    ax1_twin.plot(data['iteration'], data['radiality'], marker='s', linestyle='--', color='red', alpha=0.7)
    ax1_twin.set_ylabel(r'Radialnost $\overline{R}$', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    ax1_twin.set_ylim([0, 1.05])
    
    # Plot 2: Physical Time vs Iteration
    ax2.plot(data['iteration'], data['time'], marker='^', linestyle='-', color='green')
    ax2.set_xlabel('$i$')
    ax2.set_ylabel('$t$ [s]', color='green')
    ax2.set_title('$t(i)$')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    if not os.path.exists('pics'):
        os.makedirs('pics')
    plt.savefig('pics/free_energy_vs_iteration.png', dpi=150)
    plt.show()

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
        print(f"  - Plotting {file} -> {frame_path}")
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

def plotS_F():

    summary_path = 'output_temp_sweep/summary.dat'
    if not os.path.exists(summary_path):
        print(f"Missing file: {summary_path}")
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
    if not os.path.exists('pics'):
        os.makedirs('pics')
    plt.savefig('pics/average_S_vs_T.png')
    plt.show()

    # Plot Free Energy vs Temperature
    plt.figure(figsize=(8, 5))
    plt.plot(T, F, 's-', color='red', label='Free Energy')
    plt.xlabel('$T$ [K]')
    plt.ylabel('$F$')
    plt.title('$F(T)$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pics/free_energy_vs_T.png')
    plt.show()


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


def plot_quench_log(path: str = 'output_quench', out_dir: str = 'pics', show: bool = True):
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

    # Relative energy change between consecutive log points
    rel_dF = np.full_like(total, np.nan)
    if total.size >= 2:
        denom = np.where(np.abs(total[:-1]) > 0.0, total[:-1], np.nan)
        rel_dF[1:] = np.abs((total[1:] - total[:-1]) / denom)

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
    axR2.plot(t, rel_dF, ':', lw=1.8, color='tab:red', label='|ΔF/F|')
    axR2.set_ylabel('|ΔF/F|')
    axR2.set_yscale('log')
    axR2.grid(False)

    # Average S
    axS.plot(t, avgS, '-', lw=2)
    axS.set_xlabel('time [s]')
    axS.set_ylabel('<S> (droplet)')
    axS.set_title('Average order parameter')
    axS.grid(True, alpha=0.3)

    base = os.path.basename(log_path)
    fig.suptitle(f"Quench log: {base}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = os.path.join(out_dir, 'quench_summary.png')
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

    # Find xi where correlation drops below target
    xi = float('nan')
    for ri in range(1, max_r + 1):
        if np.isfinite(C_r[ri]) and (C_r[ri] <= float(target)):
            xi = float(ri)
            break

    return xi, r, C_r


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
              "Enter your choice (0-8): ").strip()
    while not i.isdigit() or int(i) < 0 or int(i) > 8:
        print("Invalid input. Please enter a number between 0 and 8.")
        i = input("Please enter a number between 0 and 8: ").strip()
    i = int(i)
    # ---------------------------------------------------------------------- PART 1 ----------------------------------------------------------------------------|
    if i == 0:
        # --- Plotting Individual Final States ---
        print("Plotting final state from pre-calculated Nematic Field data...")
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
            z_slice=z_slice_to_plot,
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
        plot_energy_VS_iter()
    # ---------------------------------------------------------------------- PART 2 -----------------------------------------------------------------------------|
    elif i == 3:
        # --- Plotting Energy Components vs Iteration ---
        print("\nPlotting Energy Components vs Iteration...")
        energy_components()
    # ---------------------------------------------------------------------- PART 3 -----------------------------------------------------------------------------|
    elif i == 4:
        # --- Plotting S vs Temperature and Free Energy ---
        print("\nPlotting Average S and Free Energy vs Temperature...")
        plotS_F()
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
        in_path = input("Path to quench log file or directory [default: output_quench]: ").strip()
        if not in_path:
            in_path = 'output_quench'
        try:
            plot_quench_log(in_path, out_dir='pics', show=True)
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
            )
        except Exception as e:
            print(f"Error creating quench animation: {e}")
    elif i == 8:
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
    # ---------------------------------------------------------------------- FIN --------------------------------------------------------------------------------|

# WHAT I NEED TO DO:
# - Run simulation for different external field strengths (1e6-1e7)(conpare order parameter evolutions (energy minimization))
# - combine final energy vs iteration for different external field strengths (to see the influence on energy minimization)
# - plot energy contributions for one external energy strength (e.g. 1e7)
# - Run temperature sweep simulations with different external field strengths

# EXPLAIN TO COLLEAGUES:
# - Parameters are set to immitate 5CB (semi-successfully)
# - simulation allows for material prototyping and testing of different nematic field configurations
# - due to time limitations, i couldn't run separate simualtions, testing all parameter changes, and is why i approximated 5CB.
# - most important params: kappa, gamma, A, B, C, T, T*, alpha