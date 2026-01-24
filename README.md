# QSR - Landau-De Gennes Q-Tensor Solver

## Overview

**QSR** is a high-performance numerical solver designed to simulate the equilibrium structure of a **5CB Liquid Crystal droplet** in a radial (hedgehog) configuration. It employs the **Landau-De Gennes Q-tensor theory** to minimize the system's free energy using the finite difference method on a 3D grid.

The project provides two parallelized implementations:
- **CPU Version:** Utilizes **OpenMP** for multi-threading.
- **GPU Version:** Utilizes **CUDA** for massive parallelism on NVIDIA GPUs.

## Features

- **Physical Model:** Solves for the traceless, symmetric Q-tensor (5 independent components).
- **Energy Minimization:** Includes contributions from:
  - **Bulk Energy:** Landau-de Gennes expansion (A, B, C coefficients).
  - **Elastic Energy:** Gradient terms with anisotropic elastic constants (L1, L2, L3).
  - **Surface Anchoring:** Strong (radial) anchoring and weak anchoring penalties.
  - **External Fields:** (Structure ready for electric/magnetic field integration).
- **Observables:** Calculates the scalar order parameter ($S$), nematic director ($\mathbf{n}$), total free energy, and a "radiality" metric to track convergence.
- **Performance:** optimized Laplacian calculations and memory layouts for both CPU and GPU architectures.

## Prerequisites

- **Linux** or **Windows** environment.
- **C++ Compiler:** Supporting C++17 (for CUDA) or C++23 (standard).
- **CUDA Toolkit:** Required for compiling the GPU version (tested with `nvcc`).
- **OpenMP:** Required for the CPU version.

## Building and Running

### Linux (CLI)

#### 1. CPU Version
Compiling with `g++` (requires OpenMP):
```bash
g++ -O3 -std=c++23 -fopenmp -o QSR_cpu QSR.cpp
```
Run:
```bash
./QSR_cpu
```

#### 2. GPU (CUDA) Version
Compiling with `nvcc`:
```bash
# Adjust -arch=sm_XX to match your GPU's compute capability (e.g., sm_75 for Turing)
nvcc -O3 -arch=sm_75 -std=c++17 -o QSR_cuda QSR.cu
```
Run:
```bash
./QSR_cuda
```

### Windows (Visual Studio)

1. Open `QSR.sln` in Visual Studio.
2. Select the desired configuration (e.g., **Release** / **x64**).
3. Build the solution (Ctrl+Shift+B).
4. Run the executable from the debugger or command line.

## Project Structure

- **Core Source:**
  - `QSR.h`, `QSR.cpp`: Main C++ implementation (Host/CPU).
  - `QSR.cuh`, `QSR.cu`: CUDA implementation (Device/GPU).
- **Output:**
  - `output/`: Directory where `.dat` simulation results are saved.
  - `pics/`: Placeholder for generated visualizations.
- **Configuration:**
  - Physics constants (Dimensionless parameters, Elastic constants) are defined in the `DimensionalParams` struct within the header files.

## Output Files

The simulation generates data files (typically ignored by git) containing:
- **Free Energy History:** Evolution of free energy over iterations.
- **Nematic Field:** Final director field $\mathbf{n}$ and order parameter $S$.
- **Q-Tensor:** Raw Q-tensor components for the final state.

## Future Work (TODO)

- Implement **FDTDR** (Finite-Difference Time-Domain Radial) for light propagation analysis.
- Develop **QSB** solver for Bipolar configurations (droplets with boojums).
- Train **CNN** models to predict LC parameters from optical images.
