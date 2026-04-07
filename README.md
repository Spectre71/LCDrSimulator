# QSR - Landau-De Gennes Q-Tensor Solver

## Overview

QSR is a Landau-de Gennes Q-tensor solver for confined liquid-crystal droplets, with the active implementation in the CUDA backend. The current research workflow is centered on fixed-`dt` quench and Kibble-Zurek studies, supported by reduced validation harnesses, a Tkinter launcher, and post-processing utilities.

The repository contains two solver paths:

- `QSR.cu` / `QSR.cuh`: primary CUDA implementation used for current work.
- `QSR.cpp` / `QSR.h`: CPU/OpenMP reference implementation.

## Current Solver Scope

- Traceless, symmetric Q-tensor with 5 independent components.
- Landau-de Gennes bulk free energy plus anisotropic elastic terms (`L1`, `L2`, `L3`) or a Frank-mapped path.
- Strong and weak surface anchoring, with explicit shell-order control through `boundary_order_mode` and `boundary_S`.
- Shared physical-space droplet geometry used across initialization, masks, anchoring, observables, and guards.
- Simulation modes for single-temperature relaxation, temperature sweeps, and fixed-`dt` quenches.
- Observable logging for total energy, energy components, radiality, average `S`, max `S`, 2D defect density, and 2D xi-gradient proxy.

## Build

### Linux

CUDA build:

```bash
nvcc -O3 -arch=sm_75 -std=c++17 -o QSR_cuda QSR.cu
```

CPU reference build:

```bash
g++ -O3 -std=c++23 -fopenmp -o QSR_cpu QSR.cpp
```

Adjust `sm_XX` to your local GPU architecture.

### Windows

Open `QSR.sln` in Visual Studio, select the desired configuration, and build the solution.

## Running the CUDA Solver

The recommended workflow is config-driven:

```bash
./QSR_cuda --config configs/q_100_K123.cfg
```

Config files use `key = value` syntax with optional `#` comments. Omitted keys fall back to backend defaults.

### Important Config Semantics

- `sim_mode = 1`: single-temperature run.
- `sim_mode = 2`: temperature sweep.
- `sim_mode = 3`: quench / Kibble-Zurek workflow.
- `random_seed`: fixes stochastic initialization so validation and coarse/fine comparisons are reproducible.
- `boundary_order_mode = equilibrium`: uses the equilibrium shell order `max(S_eq(T), 0)`.
- `boundary_order_mode = custom`: uses explicit `boundary_S` instead.
- `S0`: initialization amplitude only; it is no longer the shell-order control.
- `enable_adaptive_dt`: legacy quench key only. In quench mode it is ignored on purpose; fixed `dt` is the contract.
- `enable_early_stop`: now relies on energy plus defect-density stability. Radiality remains diagnostic only.
- `defect_density_abs_eps`: absolute tolerance for the defect-density stopping channel.

## GUI and Plotting

Launch the GUI with:

```bash
python3 GUI.py
```

Install the Python plotting dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

`GUI.py` is a config/launcher frontend. `QSRvis.py` is the plotting and post-processing utility used by the GUI and from the command line. The plotting code accepts both the legacy single-temperature iteration schema (`free_energy,time`) and the current schema (`total,time` plus optional `anchoring`).

## Output Files

Typical outputs include:

- `free_energy_vs_iteration.dat`: single-temperature submode 1 iteration log.
- `energy_components_vs_iteration.dat`: single-temperature submode 2 iteration log.
- `quench_log.dat`: quench log with time, temperature, energy components, and observables.
- `Qtensor_output_*.dat`: saved Q-tensor snapshots or final states.
- `run_config.cfg`: config snapshot preserved alongside GUI-launched runs.

Output locations depend on the workflow:

- single-temperature runs typically write under `output/`
- sweeps under `output_temp_sweep/`
- quenches under the configured `out_dir`

## Validation and Analysis Utilities

Reduced validation configs live under `configs/validation/`.

Current analysis harnesses:

- Weak-anchoring mesh check:

```bash
venv/bin/python tools/weak_anchor_mesh_check.py \
  configs/validation/weak_anchor_mesh_coarse.cfg \
  configs/validation/weak_anchor_mesh_fine.cfg \
  --binary ./QSR_cuda \
  --output-root validation/weak_anchor_mesh
```

- Fixed-`dt` quench protocol-convergence check:

```bash
venv/bin/python tools/quench_protocol_convergence_check.py \
  configs/validation/quench_protocol_convergence_coarse.cfg \
  configs/validation/quench_protocol_convergence_fine.cfg \
  --binary ./QSR_cuda \
  --output-root validation/quench_protocol_convergence
```

## Production Kibble-Zurek Workflow

For production Kibble-Zurek analysis, do not rely on a single long quench. Run a matched coarse/fine pair and compare them with the protocol-convergence harness.

Start by copying `configs/q_100_K123.cfg` into two production configs, for example:

- `configs/q_100_K123_prod_coarse.cfg`
- `configs/q_100_K123_prod_fine.cfg`

### Keep Constant Between the Two Production Runs

- droplet geometry: `Nx`, `Ny`, `Nz`, `dx`, `dy`, `dz`
- material model: `a`, `b`, `c`, `T_star`, or the Frank constants / mapping choice used by the config
- anchoring and boundary setup: `W`, `gamma`, `boundary_order_mode`, and `boundary_S` when custom
- initialization and noise model: `init_mode`, `S0`, `noise_amplitude`, `random_seed`
- quench temperatures and protocol family: `T_high`, `T_low`, `Tc_KZ`, `protocol`
- solver path choices: `use_semi_implicit`, `L_stab`, `jacobi_iters`, limiter and guard toggles
- logged observables and slices: defect / xi logging toggles and their slice thresholds

### Change for the Fine Run

- halve `dt`
- double `pre_equil_iters`
- double `ramp_iters`
- double `total_iters`
- if `kz_stop_early = true`, also double `kzExtraIters`
- if you use `kzSnapshotFreq` and want the same physical snapshot spacing, double it as well

`logFreq` does not control the physics. For analysis it is acceptable to keep it unchanged so the fine run logs more densely in physical time.

### What to Send Back for Analysis

After both production runs finish, I need one of the following:

- the two output directories containing `quench_log.dat`
- or the two configs plus the harness output directory from:

```bash
venv/bin/python tools/quench_protocol_convergence_check.py \
  configs/q_100_K123_prod_coarse.cfg \
  configs/q_100_K123_prod_fine.cfg \
  --binary ./QSR_cuda \
  --output-root validation/quench_protocol_convergence_prod
```

If you run through the GUI instead of the CLI, just make sure the two runs use distinct `out_dir` values and keep the same coarse/fine refinement rule above.

## Repository Layout

- `QSR.cu`, `QSR.cuh`: primary CUDA solver.
- `QSR.cpp`, `QSR.h`: CPU reference solver.
- `GUI.py`: config builder and launcher for the CUDA backend.
- `QSRvis.py`: plots and post-processing.
- `configs/`: reusable run configurations.
- `tools/`: validation and analysis harnesses.
- `validation/`: generated validation outputs.
- `literature/`: local physics references used to verify model changes.
