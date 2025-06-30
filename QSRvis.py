import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import imageio
import re

def plot_nematic_field_slice(filename, Nx, Ny, Nz, z_slice=None, output_path=None):
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
        data = np.loadtxt(filename, comments='#')
    except IOError:
        print(f"Warning: Could not read file '{filename}'. Skipping.")
        return

    # Select the data for the desired z-slice
    slice_data = data[data[:, 2] == z_slice]

    if slice_data.shape[0] == 0:
        print(f"Warning: No data found for z_slice = {z_slice} in {filename}. Skipping.")
        return

    # Prepare 2D arrays for S, nx, and ny
    S = np.zeros((Nx, Ny))
    nx = np.zeros((Nx, Ny))
    ny = np.zeros((Nx, Ny))

    # Directly populate arrays from the file data
    for row in slice_data:
        i, j = int(row[0]), int(row[1])
        if i < Nx and j < Ny:
            S[i, j] = row[3]
            nx[i, j] = row[4]
            ny[i, j] = row[5]
            # nz (row[6]) is not needed for the 2D quiver plot

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(S.T, origin='lower', cmap='viridis', extent=[0, Nx, 0, Ny], vmin=0, vmax=0.6)
    fig.colorbar(im, ax=ax, label='Scalar Order Parameter $S$')
    
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
                  color='white', scale=30, headwidth=3, pivot='middle')
    
    # Extract iteration number from filename for the title
    match = re.search(r'(\d+)', os.path.basename(filename))
    if match:
        iteration = match.group(1)
        title = f"Nematic Field at z={z_slice} (Iteration: {iteration})"
    else:
        # Handle filenames like 'nematic_field_final.dat'
        title = f"Nematic Field at z={z_slice} ({os.path.basename(filename)})"
        
    ax.set_title(title)
    ax.set_xlabel('$x$ grid index')
    ax.set_ylabel('$y$ grid index')
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
    data = np.genfromtxt('free_energy_vs_iteration.dat', delimiter=',', names=True)

    plt.figure(figsize=(8, 5))
    plt.plot(data['iteration'], data['free_energy'], marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Free Energy')
    plt.title('Free Energy vs Iteration')
    plt.grid(True)
    plt.tight_layout()
    if not os.path.exists('pics'):
        os.makedirs('pics')
    plt.savefig('pics/free_energy_vs_iteration.png')
    plt.show()

def energy_components():
    data = np.genfromtxt('energy_components_vs_iteration.dat', delimiter=',', names=True)

    plt.figure(figsize=(8, 5))
    plt.plot(data['iteration'], data['bulk'], label='Bulk')
    plt.plot(data['iteration'], data['elastic'], label='Elastic')
    plt.plot(data['iteration'], data['field'], label='Field')
    plt.plot(data['iteration'], data['total'], label='Total', linestyle='--', color='k')
    plt.xlabel('Iteration')
    plt.ylabel('Energy [$J$]')
    plt.title('Energy Component Breakdown vs Iteration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if not os.path.exists('pics'):
        os.makedirs('pics')
    plt.savefig('pics/energy_components_vs_iteration.png')
    plt.show()

def create_nematic_field_animation(data_dir, output_gif, Nx, Ny, Nz):
    """
    Finds all nematic field snapshots, plots them, and creates a GIF.
    """
    # Create a directory for temporary frames
    frames_dir = "frames"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    else: # Clean up old frames from a previous run
        for f in glob.glob(os.path.join(frames_dir, '*.png')):
            os.remove(f)

    # Find all nematic field snapshot files and sort them numerically
    files = glob.glob(os.path.join(data_dir, 'nematic_field_iter_*.dat'))
    if not files:
        print("Error: No 'nematic_field_iter_*.dat' files found in the 'output' directory.")
        return
        
    files.sort(key=lambda f: int(re.search(r'(\d+)', f).group(1)))

    print(f"Found {len(files)} nematic field snapshots. Generating frames...")

    frame_paths = []
    for i, file in enumerate(files):
        frame_path = os.path.join(frames_dir, f'frame_{i:04d}.png')
        print(f"  - Plotting {file} -> {frame_path}")
        plot_nematic_field_slice(file, Nx, Ny, Nz, output_path=frame_path)
        frame_paths.append(frame_path)

    # Create GIF from the generated frames
    print(f"\nStitching {len(frame_paths)} frames into {output_gif}...")
    with imageio.get_writer(output_gif, mode='I', duration=0.1) as writer:
        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)

    # Clean up temporary frames
    print("Cleaning up temporary frames...")
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.rmdir(frames_dir)

    print(f"\nAnimation saved to {output_gif}")

def plotS_F():

    # Load data
    data = np.genfromtxt('output_temp_sweep/summary.dat', delimiter=',', names=True)

    # Normalize and flip sign of average_S
    S = data['average_S']
    S_norm = S / np.max(np.abs(S))

    # Plot Average S vs Temperature
    plt.figure(figsize=(8, 5))
    plt.plot(data['temperature'], S_norm, 'o-', label='Average S')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Average Order Parameter S')
    plt.title('Nematic Order Parameter vs Temperature')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if not os.path.exists('pics'):
        os.makedirs('pics')
    plt.savefig('pics/average_S_vs_T.png')
    plt.show()

    # Plot Free Energy vs Temperature
    plt.figure(figsize=(8, 5))
    plt.plot(data['temperature'], data['final_energy'], 's-', color='red', label='Free Energy')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Final Free Energy')
    plt.title('Free Energy vs Temperature')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pics/free_energy_vs_T.png')
    plt.show()

def animate_tempSweep():
    # Find all temperature directories
    dirs = sorted(glob.glob('output_temp_sweep/T_*/'))

    frame_paths = []
    for idx, d in enumerate(dirs):
        data = np.loadtxt(d + 'nematic_field_final.dat', comments='#')
        # data columns: i, j, k, S, nx, ny, nz
        # For a 2D slice (e.g., k=Nz//2):
        slice_data = data[data[:,2] == Nz//2]
        S = slice_data[:,3].reshape(Nx, Ny)
        nx = slice_data[:,4].reshape(Nx, Ny)
        ny = slice_data[:,5].reshape(Nx, Ny)
        plt.figure(figsize=(8, 8))
        plt.imshow(S, origin='lower', cmap='viridis', vmin=0, vmax=0.6)
        # x,y= np.meshgrid(np.arange(Nx), np.arange(Ny))
        # plt.quiver(
        #     x[::5, ::5], y[::5, ::5], 
        #     nx[::5, ::5], ny[::5, ::5], 
        #     color='white', scale=30, headwidth=3, pivot='middle'
        # )
        # plt.quiver(nx, ny, color='white', scale=30, headwidth=3, pivot='middle')
        temp_str = d.strip('/').split('_')[-1]
        plt.title(f'Temperature: {temp_str}')
        frame_path = f'output_temp_sweep/frame_{idx:04d}.png'
        plt.savefig(frame_path)
        plt.close()
        frame_paths.append(frame_path)

    # Create GIF from the frames
    if not os.path.exists('pics'):
        os.makedirs('pics')
    gif_path = 'pics/temp_sweep_animation.gif'
    with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)

    # Optionally, clean up frame files
    for frame_path in frame_paths:
        os.remove(frame_path)

    print(f"Animation saved to {gif_path}")

# Use imageio or ffmpeg to make an animation from the frames
if __name__ == '__main__':
    # --- Configuration ---
    # These should match the parameters used in your C++ simulation
    Nx, Ny, Nz = 100, 100, 100 
    z_slice_to_plot = Nz // 2
    i = input("Enter the number of the plot you want to create:\n"
              "0: Plot final state from pre-calculated Nematic Field data\n"
              "1: Create animation from Nematic Field data\n"
              "2: Plot Free Energy vs Iteration\n"
              "3: Plot Energy Components vs Iteration\n"
              "4: Plot Average S and Free Energy vs Temperature\n"
              "5: Animate Temperature Sweep\n"
              "Enter your choice (0-5): ").strip()
    while not i.isdigit() or int(i) < 0 or int(i) > 5:
        print("Invalid input. Please enter a number between 0 and 5.")
        i = input("Please enter a number between 0 and 5: ").strip()
    i = int(i)
    # ---------------------------------------------------------------------- PART 1 ----------------------------------------------------------------------------|
    if i == 0:
        # --- Plotting Individual Final States ---
        print("Plotting final state from pre-calculated Nematic Field data...")
        plot_nematic_field_slice(filename='nematic_field_final.dat', Nx=Nx, Ny=Ny, Nz=Nz, z_slice=z_slice_to_plot, output_path='pics/final_state.png')
    elif i == 1:
        # --- Creating Animation ---
        print("\nCreating animation from Nematic Field data...")
        create_nematic_field_animation(data_dir='output', output_gif='pics/nematic_field_evolution.gif', Nx=Nx, Ny=Ny, Nz=Nz)
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
        print("\nAnimating Temperature Sweep...")
        animate_tempSweep()
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
# - most importants params: kappa, gamma, A, B, C, T, T*, alpha