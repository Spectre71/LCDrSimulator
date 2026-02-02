#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <string>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <iomanip>

// ---------- Constants ----------
#ifndef PI
#define PI 3.14159265358979323846
#endif

// -------- Q-Tensor Parameters (traceless and symmetric - 5 independent components) ----------
struct QTensor {
    double Qxx = 0, Qxy = 0, Qxz = 0, Qyy = 0, Qyz = 0;

    // Operator overloads for vector-like operations
    __host__ __device__ QTensor operator+(const QTensor& other) const {
        return {Qxx + other.Qxx, Qxy + other.Qxy, Qxz + other.Qxz, Qyy + other.Qyy, Qyz + other.Qyz};
    }
    __host__ __device__ QTensor operator-(const QTensor& other) const {
        return {Qxx - other.Qxx, Qxy - other.Qxy, Qxz - other.Qxz, Qyy - other.Qyy, Qyz - other.Qyz};
    }
    __host__ __device__ QTensor operator*(double scalar) const {
        return {Qxx * scalar, Qxy * scalar, Qxz * scalar, Qyy * scalar, Qyz * scalar};
    }
    // Compound assignment operators
    __host__ __device__ QTensor& operator+=(const QTensor& other) {
        Qxx += other.Qxx; Qxy += other.Qxy; Qxz += other.Qxz; Qyy += other.Qyy; Qyz += other.Qyz;
        return *this;
    }
    __host__ __device__ QTensor& operator-=(const QTensor& other) {
        Qxx -= other.Qxx; Qxy -= other.Qxy; Qxz -= other.Qxz; Qyy -= other.Qyy; Qyz -= other.Qyz;
        return *this;
    }
};

struct FullQTensor {
    double Qxx = 0, Qxy = 0, Qxz = 0;
    double Qyx = 0, Qyy = 0, Qyz = 0;
    double Qzx = 0, Qzy = 0, Qzz = 0;

    // Constructor from a 5-component Qtensor
    __host__ __device__ FullQTensor(const QTensor& q_5comp) {
        Qxx = q_5comp.Qxx;
        Qxy = q_5comp.Qxy;
        Qxz = q_5comp.Qxz;
        Qyy = q_5comp.Qyy;
        Qyz = q_5comp.Qyz;
        
        // symmetry
        Qyx = Qxy;
        Qzx = Qxz;
        Qzy = Qyz;

        // enforce tracelessness
        Qzz = -Qxx - Qyy;
    }
    // Default constructor
    __host__ __device__ FullQTensor() = default;
};

// Struct to hold the calculated nematic director and order parameter
struct NematicField {
    double S = 0;
    double nx = 0, ny = 0, nz = 0;
};

struct EnergyComponents {
    double bulk = 0.0;
    double elastic = 0.0;
    double field = 0.0;
    __host__ __device__ double total() const { return bulk + elastic + field; }
};

struct DimensionalParams {
    double a;
    double b;
    double c;
    double T;
    double T_star;
    // Elastic anisotropy (LdG gradient terms)
    double L1 = 0.0;
    double L2 = 0.0;
    double L3 = 0.0;
    // Weak anchoring strength (surface penalty)
    double W = 0.0;
};

// Kernel declarations
__global__ void computeLaplacianKernel(const QTensor* Q, QTensor* laplacianQ, int Nx, int Ny, int Nz, double dx, double dy, double dz);
__global__ void computeChemicalPotentialKernel(const QTensor* Q, QTensor* mu, const QTensor* laplacianQ, int Nx, int Ny, int Nz, double dx, double dy, double dz, DimensionalParams params, double kappa, int modechoice, double* Dcol_x, double* Dcol_y, double* Dcol_z);
// L3-only contribution to the chemical potential (adds to mu in-place).
// Split out from computeChemicalPotentialKernel to avoid excessive register pressure when L3 is unused.
__global__ void computeChemicalPotentialL3Kernel(const QTensor* Q, QTensor* mu, int Nx, int Ny, int Nz,
                                                double dx, double dy, double dz, DimensionalParams params,
                                                const double* Dcol_x, const double* Dcol_y, const double* Dcol_z);
// Weak anchoring W is interpreted as a surface energy density (J/m^2).
// Internally we convert it to a volumetric penalty via W_eff = W / shell_thickness.
__global__ void applyWeakAnchoringPenaltyKernel(QTensor* mu, const QTensor* Q, int Nx, int Ny, int Nz,
                                               const bool* is_shell, double S_shell, double W, double shell_thickness);
// Optional numerical limiter: if q_norm_cap>0, clamp Frobenius norm of Q to q_norm_cap to prevent runaway S blow-ups.
// For uniaxial Q=S(nn-I/3), ||Q||_F = sqrt(2/3)*|S|.
__global__ void updateQTensorKernel(QTensor* Q, const QTensor* mu, int Nx, int Ny, int Nz, double dt, const bool* is_shell, double gamma, double W, double q_norm_cap);

// Stabilized semi-implicit stepper: treat the isotropic Laplacian-like elastic term implicitly.
// For mu = mu_bulk + mu_aniso - L_stab*Laplace(Q), the IMEX update is:
//   (I - alpha*Laplace) Q^{n+1} = Q^n - dt*(1/gamma)*(mu_bulk + mu_aniso)
// with alpha = dt*(1/gamma)*L_stab.
// Since (mu_bulk + mu_aniso) = mu + L_stab*Laplace(Q), the RHS becomes:
//   RHS = Q^n - dt*(1/gamma)*mu(Q^n) - alpha*Laplace(Q^n).
// This damps high-k modes and is robust for stiff elasticity.
__global__ void computeRhsSemiImplicitKernel(const QTensor* Q, const QTensor* mu, const QTensor* laplacianQ, QTensor* rhs,
                                            int Nx, int Ny, int Nz, double dt, double gamma, double L_stab,
                                            const bool* is_shell, double W);
__global__ void jacobiHelmholtzStepKernel(const QTensor* rhs, const QTensor* Q_old, QTensor* Q_new,
                                         int Nx, int Ny, int Nz, double alpha, double dx, double dy, double dz,
                                         const bool* is_shell, double W);
__global__ void applyBoundaryConditionsKernel(QTensor* Q, int Nx, int Ny, int Nz, const bool* is_shell, double S0, double W);
__global__ void computeEnergyKernel(const QTensor* Q, double* bulk_energy, double* elastic_energy, int Nx, int Ny, int Nz, double dx, double dy, double dz, DimensionalParams params, double kappa, int modechoice);
__global__ void computeNematicFieldKernel(const QTensor* Q, NematicField* nf, int Nx, int Ny, int Nz);
__global__ void computeRadialityKernel(const QTensor* Q, double* radiality_vals, int* count_vals, int Nx, int Ny, int Nz);

// Device helper functions
__host__ __device__ NematicField calculateNematicFieldDevice(const QTensor& q_5comp);

// -------- NOTES ---------
/*
nvcc -O3 -arch=sm_75 -std=c++17 -o QSR_cuda QSR.cu
*/

// -------- TESTS ---------
/*
QUENCH: TESTING QUENCH RATE
Select initialization mode: [0] random, [1] radial, [2] isotropic+noise (default: 1): 2

--- Spatial Parameters ---
Enter grid size Nx [100]: 
Enter grid size Ny [100]: 
Enter grid size Nz [100]: 
Enter dx (m) [1e-08]: 
Enter dy (m) [1e-08]: 
Enter dz (m) [1e-08]: 

--- Landau-de Gennes Material Parameters ---
Enter a (J/m^3/K) [44000]: 
Enter b (J/m^3) [1.413e+06]: 
Enter c (J/m^3) [1.153e+06]: 
Enter T (K) [300]: 312
Enter T* (K) [A=0, isotropic spinodal] [308]: 
Enter Bulk Energy convention (1=std, 2=ravnik) [1]: 

Bulk convention: std
Computed S_c (heuristic) = 0.612749, S_eq(T=312K) = 0
Bulk transition estimates (uniaxial, homogeneous):
  T* (A=0, isotropic spinodal) = 308 K
  T_NI (coexistence, global min) ≈ 310.186 K
  T_spin,N (nematic disappears)  ≈ 310.46 K
Enter initial S0 [0.5]: 

--- Elastic and Dynamic Parameters ---
Enter kappa (J/m) [6.5e-12]: 
Use Frank-to-LdG mapping with K1=K3 ≠ K2? (y/n) [n]: y
Enter K1=K3 [6.5e-12]: 
Enter K2 [4e-12]: 8e-12
Enter S_ref for mapping (default: S_eq(T)) [0.5]: 
Mapped L1=1.6e-11, L2=-6e-12, L3=0 (Q=S(nn-I/3), S_ref=0.5, Kappa set to 0)
Enable correlation-length guard (abort if xi is under-resolved)? (y/n) [y]: n

--- Correlation Length (xi) Estimate ---
Using L_eff=1.6e-11 J/m (L1), L2=-6e-12
Bulk convention = std
Bulk stiffness (linear) |A_lin| = |A| = |3 a (T-T*)| = 528000 J/m^3
Bulk curvature at S_eq: unavailable (likely isotropic / above T*)
xi_lin ≈ sqrt(L_eff/|A_lin|) = 5.50482e-09 m
xi_used = 5.50482e-09 m,  xi/min(dx,dy,dz) = 0.550482
Suggested resolution: dx ≲ xi/3 => dx ≲ 1.83494e-09 m
Droplet radius R ≈ 5e-07 m,  R/xi ≈ 90.8295
Enter weak anchoring W (J/m^2) [0]: 
Enter gamma (Pa·s) [0.1]: 
Enter iterations [100000]: 2000000
Enter print freq [200]: 1000
Enter tolerance [0.01]: 
Enter radiality convergence eps RbEps (relative ΔR̄) [0.01]: 

Select simulation mode: [1] Single Temp, [2] Temp Range, [3] Quench (time-dependent T):  [1]: 3
Output directory for quench results [default: output_quench]: 
Directory 'output_quench' exists. Delete it and start fresh? (y/n) [y]:  
Quench protocol (1=step, 2=ramp) [2]: 
T_high (K) [312]: 
T_low (K) [307]: 305
Pre-equilibration iterations at T_high [0]: 10000
----------------------------------------------
Ramp iterations (>=1) [1000]: 100 # WHAT WE ARE CURRENTLY TESTING FOR
----------------------------------------------
Total iterations [2000000]: 
Log/print freq [1000]: 
Snapshot freq to output directory (0=off) [1000]: 
Noise amplitude for isotropic init [0.001]: 

Maximum stable dt = 5.20833e-08 s
Enter time step dt (s) [5.20833e-08]: 1e-9
Enable early-stop once converged at final T? (rel. ΔF/F + rel. ΔR̄/R̄) (y/n) [n]: y
Radiality threshold (Rbar, 0=disable) [0.998]: 0.995
Do you want output in the console every 1000 iterations? (y/n): n
#----------------------------------------------------------------------------------------#
QUENCH: TESTING ELASTIC CONSTANTS
Select initialization mode: [0] random, [1] radial, [2] isotropic+noise (default: 1): 2

--- Spatial Parameters ---
Enter grid size Nx [100]: 
Enter grid size Ny [100]: 
Enter grid size Nz [100]: 
Enter dx (m) [1e-08]: 1e-9
Enter dy (m) [1e-08]: 1e-9
Enter dz (m) [1e-08]: 1e-9

--- Landau-de Gennes Material Parameters ---
Enter a (J/m^3/K) [44000]: 
Enter b (J/m^3) [1.413e+06]: 
Enter c (J/m^3) [1.153e+06]: 
Enter T (K) [300]: 312
Enter T* (K) [A=0, isotropic spinodal] [308]: 
Enter Bulk Energy convention (1=std, 2=ravnik) [1]: 

Bulk convention: std
Computed S_c (heuristic) = 0.612749, S_eq(T=312K) = 0
Bulk transition estimates (uniaxial, homogeneous):
  T* (A=0, isotropic spinodal) = 308 K
  T_NI (coexistence, global min) ≈ 310.186 K
  T_spin,N (nematic disappears)  ≈ 310.46 K
Enter initial S0 [0.5]: 

--- Elastic and Dynamic Parameters ---
Generally:  Twist (K2) < Splay (K1) < Bend (K3)
Enter kappa (J/m) [6.5e-12]: 
Use Frank-to-LdG mapping with K1, K2, K3? (y/n) [n]: y
Enter K1 [6.5e-12]: 
Enter K2 [4e-12]: 6.5e-12
Enter K3 [8e-12]: 6.5e-12
Enter S_ref for mapping (default: S_eq(T)) [0.5]: 
Mapped L1=1.3e-11, L2=0, L3=0 (Q=S(nn-I/3), S_ref=0.5, Kappa set to 0)
  Check K1_back=6.5e-12, K2_back=6.5e-12, K3_back=6.5e-12
  ΔK: (K1_back-K1)=0, (K2_back-K2)=0, (K3_back-K3)=0
Enable correlation-length guard (abort if xi is under-resolved)? (y/n) [y]: 

--- Correlation Length (xi) Estimate ---
Using L_eff=1.3e-11 J/m (L1)
Bulk convention = std
Bulk stiffness (linear) |A_lin| = |A| = |3 a (T-T*)| = 528000 J/m^3
Bulk curvature at S_eq: unavailable (likely isotropic / above T*)
xi_lin ≈ sqrt(L_eff/|A_lin|) = 4.96198e-09 m
xi_used = 4.96198e-09 m,  xi/min(dx,dy,dz) = 4.96198
Suggested resolution: dx ≲ xi/3 => dx ≲ 1.65399e-09 m
Droplet radius R ≈ 5e-08 m,  R/xi ≈ 10.0766
Enter weak anchoring W (J/m^2) [0]: 
Enter gamma (Pa·s) [0.1]: 
Enter iterations [100000]: 1000000
Enter print freq [200]: 
Enter tolerance [0.01]: 
Enter radiality convergence eps RbEps (relative ΔR̄) [0.01]: 
Debug: enable CUDA error checks at log points? (syncs GPU; slower) (y/n) [n]: 
Debug: print max|mu| and max|ΔQ| at log points? (copies arrays; slower) (y/n) [n]: 

Select simulation mode: [1] Single Temp, [2] Temp Range, [3] Quench (time-dependent T):  [1]: 3
Output directory for quench results [default: output_quench]: 
Directory 'output_quench' exists. Delete it and start fresh? (y/n) [y]: 
Quench protocol (1=step, 2=ramp) [2]: 
T_high (K) [312]: 
T_low (K) [307]: 305
Pre-equilibration iterations at T_high [0]: 1000
Ramp iterations (>=1) [1000]: 
Total iterations [1000000]: 
Log/print freq [200]: 

Snapshot intent: [0] Off (final only), [1] GIF (regular snapshots), [2] KZ (snapshots around Tc)
Select snapshot mode [2]: 1
GIF snapshot frequency in iterations (0=off) [10000]: 1000
Noise amplitude for isotropic init [0.001]: 
Quench: use semi-implicit elastic stabilizer? (Helmholtz solve via Jacobi) (y/n) [y]: 
Semi-implicit: L_stab (use |L1| or |kappa|) [1.3e-11]: 
Semi-implicit: Jacobi iterations per step [25]: 

Maximum stable dt estimates:
  dt_diff (elastic) ≈ 6.41026e-10 s
  dt_diff_total      ≈ 6.41026e-10 s (before semi-implicit)
  dt_bulk (bulk)    ≈ 1.65631e-10 s
  dt_max            = 1.65631e-10 s
Enter time step dt (s) [1.65631e-10]: 
Enable instability guard? (abort/adjust if S blows up or Nyquist checkerboard grows) (y/n) [y]: 
Guard: adapt dt downward when instability is detected? (no rollback) (y/n) [y]: n
Guard: abort if max S exceeds [2]: 
Guard: abort if checkerboard rel-amplitude exceeds [0.1]: 
Optional: clamp |Q| (caps S) to prevent blow-up? (numerical limiter) (y/n) [n]: 
Enable early-stop once converged at final T? (rel. ΔF/F + rel. ΔR̄/R̄) (y/n) [n]: 
Do you want output in the console every 200 iterations? (y/n): n
Quench: log 2D KZ defect proxy to quench_log? (mid-plane winding) (y/n) [y]: 
Quench: z_slice for 2D proxy (0..Nz-1) [50]: 
Quench: S_threshold for 2D proxy [0.1]: 
Quench: charge_cutoff |s|> [0.25]: 
Quench: log 2D xi gradient proxy to quench_log? (cheap, snapshot-free) (y/n) [y]: 
Quench: z_slice for xi proxy (0..Nz-1) [50]: 
Quench: S_threshold for xi proxy [0.1]: 
*/