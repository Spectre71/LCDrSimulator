# LCDrSimulator

C++ applets that allow you to simulate different LC droplets based on their physical parameters

# Future
- add Bipolar configuration to Q-solver,
- make proper FDTD method for light propagation,
- pack into single program for versatility.
- update readme with full userManual.

# What it can do and what it cannot do
## Can:
- effectively simulates a relaxation process of the free energy in a radially configured LC droplet (hedgehog config.) with strong radial ancgoring at the boundary (Solves Landau de Gennes Q-Tensor equations and energy minimization).
## Can't:
- simulate bipolar configuration,
- simulate light propagation through such an anisotropic field such as that of an LC droplet.

# Dependencies
- Standard C++ libraries

# Compilation
```bash
g++ -O3 -std=c++23 -fopenmp -o QSR QSR.cpp
```
