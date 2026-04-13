# QSR

QSR is a Landau-de Gennes Q-tensor solver for confined liquid-crystal droplets. The active implementation is the CUDA backend in `QSR.cu` and `QSR.cuh`; the CPU/OpenMP solver in `QSR.cpp` and `QSR.h` is retained as a reference path.

The repository also includes two benchmark solvers used to validate Kibble-Zurek workflows outside confinement:

- `KZM_prooving_ground.cu` and `KZM_prooving_ground.cuh`: periodic 3D XY benchmark.
- `KZM_bulk_ldg.cu` and `KZM_bulk_ldg.cuh`: periodic bulk Landau-de Gennes benchmark.

## Features

- Symmetric, traceless Q-tensor with five independent components.
- Landau-de Gennes bulk free energy with anisotropic elastic terms (`L1`, `L2`, `L3`) or Frank-mapped coefficients.
- Strong and weak surface anchoring with explicit shell-order control through `boundary_order_mode` and `boundary_S`.
- Shared droplet geometry in physical space across initialization, masks, anchoring, observables, and stability guards.
- Single-temperature, temperature-sweep, and fixed-`dt` quench workflows.
- Validation and post-processing utilities for protocol convergence, transient analysis, shell-depth localization, and exponent extraction.

## Repository Layout

- `QSR.cu`, `QSR.cuh`: primary CUDA solver.
- `QSR.cpp`, `QSR.h`: CPU/OpenMP reference solver.
- `KZM_prooving_ground.cu`, `KZM_prooving_ground.cuh`: periodic XY benchmark solver.
- `KZM_bulk_ldg.cu`, `KZM_bulk_ldg.cuh`: periodic bulk-LdG benchmark solver.
- `GUI.py`: Tkinter launcher and config editor.
- `QSRvis.py`: plotting and post-processing library.
- `configs/`: reusable solver and validation configurations.
- `tools/`: analysis, benchmarking, and sweep utilities.
- `validation/`: reduced validation outputs, analysis tables, and dated notes.
- `pics/`: generated figures.
- `literature/`: local physics references.

## Build

### Linux

CUDA solver:

```bash
nvcc -O3 -arch=sm_75 -std=c++17 -o exe/QSR_cuda QSR.cu
```

CPU reference solver:

```bash
g++ -O3 -std=c++23 -fopenmp -o exe/QSR_cpu QSR.cpp
```

Periodic XY benchmark:

```bash
nvcc -O3 -arch=sm_75 -std=c++17 -o exe/KZM_prooving_ground_cuda KZM_prooving_ground.cu
```

Periodic bulk-LdG benchmark:

```bash
nvcc -O3 -arch=sm_75 -std=c++17 -o exe/KZM_bulk_ldg_cuda KZM_bulk_ldg.cu
```

Replace `sm_75` with the architecture of the target GPU.

### Windows

Open `QSR.sln` in Visual Studio and build the desired configuration.

## Python Environment

Create a virtual environment and install the analysis dependencies:

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
```

Launch the GUI with:

```bash
venv/bin/python GUI.py
```

The GUI now exposes a dedicated `Tools` tab for the shared post-processing utilities. It lets you pick a registered tool, inspect its purpose and argument template, substitute `{source}` and `{out_dir}` placeholders, and preview generated text or PNG outputs without leaving the launcher.

## Running the Solver

The solver is intended to be driven by configuration files:

```bash
./exe/QSR_cuda --config <config.cfg>
```

Configuration files use `key = value` syntax with optional `#` comments. Example configurations are provided under `configs/` and `configs/validation/`.

### Important Configuration Semantics

- `sim_mode = 1`: single-temperature relaxation.
- `sim_mode = 2`: temperature sweep.
- `sim_mode = 3`: fixed-`dt` quench / Kibble-Zurek workflow.
- `random_seed`: fixes stochastic initialization for reproducible comparisons.
- `boundary_order_mode = equilibrium`: uses the equilibrium shell order `max(S_eq(T), 0)`.
- `boundary_order_mode = custom`: uses explicit `boundary_S`.
- `S0`: initialization amplitude only; it is not the shell-order control.
- `enable_adaptive_dt`: legacy quench key. In quench mode it is ignored by design.
- `enable_early_stop`: uses energy plus defect-density stability. Radiality remains diagnostic.
- `defect_density_abs_eps`: absolute tolerance for the defect-density stopping channel.

## Output Files

Typical outputs include:

- `quench_log.dat`: time, temperature, energies, and observables for quench runs.
- `free_energy_vs_iteration.dat`: single-temperature iteration log.
- `energy_components_vs_iteration.dat`: component-wise single-temperature log.
- `Qtensor_output_*.dat`: saved Q-tensor states or snapshots.
- `nematic_field_iter_*.dat`, `Qtensor_output_iter_*.dat`, `xy_field_iter_*.dat`, `q_tensor_iter_*.dat`: optional raw snapshot payloads.
- `run_config.cfg`: preserved configuration snapshot for a completed run.

Output locations depend on the workflow:

- single-temperature runs usually write under `output/`
- temperature sweeps under `output_temp_sweep/`
- quenches under the configured `out_dir`

## Validation Data Policy

The repository retains durable analysis products and removes bulky transient payloads once they have been reduced. The retained artifacts are:

- generated configs and sweep plans
- `quench_log.dat` and reduced metrics tables
- analysis CSV, Markdown, and figure outputs
- dated validation notes under `validation/`

Raw snapshot iteration files and raw final field dumps are treated as reproducible intermediates. When a result needs additional temporal or spatial resolution, the corresponding run should be reproduced with a targeted configuration rather than relying on archived multi-gigabyte field trees.

## Core Utilities

The main analysis entry points are:

- `QSRvis.py` option `13`: interactive wrapper around the registered `tools/` post-processing suite.

- `tools/quench_rate_sweep.py`: launch parameterized quench-rate ladders.
- `tools/check_quench_protocol_convergence.py`: run matched coarse/fine protocol checks.
- `tools/analyze_confined_final_state.py`: summarize final-state confined 3D defect metrics.
- `tools/analyze_midplane_slab.py`: integrate the 2D defect metric across a centered slab.
- `tools/analyze_shell_depth.py`: bin defects by inward shell depth and build shell-exclusion scans.
- `tools/analyze_shell_band_decomposition.py`: choose a common shell-focus annulus from shell-depth tables.
- `tools/plot_shell_focus_summary.py`: render skin/focus/bulk summaries from shell-band analysis.
- `tools/analyze_shell_focus_exponent.py`: extract pooled late-window exponents for the selected shell band.
- `tools/plot_xy_kzm_benchmark_figure.py`: render the periodic XY benchmark figure from existing analysis artifacts.
- `tools/plot_bulk_ldg_benchmark_figure.py`: render the periodic bulk-LdG benchmark figure from existing analysis artifacts.
- `tools/plot_confined_transient_bridge_summary.py`: render the confined transient bridge summary from existing cohort analyses.

## Typical Workflows

### Protocol Convergence

```bash
venv/bin/python tools/check_quench_protocol_convergence.py \
  <coarse.cfg> \
  <fine.cfg> \
  --binary ./exe/QSR_cuda \
  --output-root validation/protocol_convergence
```

For quench studies, the fine configuration should preserve the physical protocol by halving `dt` and doubling the relevant iteration counts.

### Quench-Rate Sweep

```bash
venv/bin/python tools/quench_rate_sweep.py run \
  <base.cfg> \
  --binary ./exe/QSR_cuda \
  --output-root validation/confined_rate_sweep \
  --ramp-values 25,50,100,200,400 \
  --keep-going
```

For large transient sweeps that only need a fixed set of late windows, add `--retain-offsets` so completed cases are pruned immediately after reduction.

### Confined Localization and Exponent Extraction

```bash
venv/bin/python tools/analyze_shell_depth.py \
  <sweep_root> \
  --offsets 4.5e-8,5.0e-8,5.5e-8,6.0e-8,6.5e-8

venv/bin/python tools/analyze_shell_band_decomposition.py \
  <sweep_root> \
  --offsets 4.5e-8,5.0e-8,5.5e-8,6.0e-8,6.5e-8

venv/bin/python tools/plot_shell_focus_summary.py \
  <sweep_root> \
  --offsets 4.5e-8,5.0e-8,5.5e-8,6.0e-8,6.5e-8

venv/bin/python tools/analyze_shell_focus_exponent.py \
  <sweep_root> \
  --offsets 4.5e-8,5.0e-8,5.5e-8,6.0e-8,6.5e-8
```

The same sweep root can also be interrogated with `tools/analyze_confined_final_state.py` or `tools/analyze_midplane_slab.py` when alternative observables are required.

### Benchmark Figures

Periodic XY benchmark:

```bash
venv/bin/python tools/plot_xy_kzm_benchmark_figure.py
```

Periodic bulk-LdG benchmark:

```bash
venv/bin/python tools/plot_bulk_ldg_benchmark_figure.py
```

Confined transient bridge summary:

```bash
venv/bin/python tools/plot_confined_transient_bridge_summary.py
```

## Benchmark Ladder

The validation program is organized as a three-step ladder:

1. periodic XY benchmark for a textbook continuous-transition reference
2. periodic bulk Landau-de Gennes benchmark for the unconfined nematic bridge
3. confined droplet sweeps for boundary-conditioned localization and scaling analysis

This ordering keeps the confined interpretation grounded in branches where the protocol and readouts can be validated without confinement or anchoring.

## Notes

- The confined solver stores five independent Q-tensor components and enforces tracelessness through `Qzz = -(Qxx + Qyy)`.
- Fixed `dt` is the required quench contract for Kibble-Zurek work in this repository.
- Physics-facing changes should be checked against the references in `literature/` before they are promoted into the active workflows.
