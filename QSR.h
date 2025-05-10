#pragma once

// This header file contains the function prototypes and includes necessary libraries for the Landau de Gennes Q-Tensor solver (QSR) for a radial (hedgehog) 5CB Liquid Crystal droplet with strong surface anchoring int the nematic phase (uniaxial).
// The solver uses the finite difference method to solve the Q-tensor equations in a spherical coordinate system.
// The code is designed to be modular and easy to read, with clear function names and comments explaining the purpose of each function.

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdio> 
#include <cstdlib>
#include <omp.h>

// ---------- Constants ----------
const double PI = 3.14159265358979323846; // Value of pi
const double kB = 1.380649e-23; // Boltzmann constant in J/K
const double epsilon = 8.854187817e-12; // Permittivity of free space in F/m
const double mu0 = 1.256637061e-6; // Permeability of free space in H/m
const double theta0 = 0.0; // Initial value of the polar angle in radians
const double phi0 = 0.0; // Initial value of the azimuthal angle in radians
const double T = 300.0; // Temperature in Kelvin
const double kappa = 4.0e-11; // Elastic constant - one coonst. approx (N)
const double A = -0.172e6; // Landau-de Gennes coefficients (J/m^3), negative in nematic phase
const double B = -2.12e6;
const double C = 1.73e6;
const double S0 = 0.5; // Initial value of the order parameter
const double dt = 1e-16; // Time step for the simulation
const double eps = 1e-9; // Small value for numerical stability
const double tolerance = 1e-6; // Tolerance for convergence
const double maxTime = 1.0; // Maximum simulation time - redundant, may be used elsewhere.
const double maxIterations = 100000; // Maximum number of iterations for convergence

// ---------- Grid Parameters ----------
struct Grid {
	double x, y, z; // Coordinates of the grid point
	double dx, dy, dz; // Grid spacing in x, y, and z directions
};
const double Lx = 1.0e-6, Ly = 1.0e-6, Lz = 1.0e-6; // Length of the simulation box in x, y, and z directions
const int Nx = 100, Ny = 100, Nz = 100; // Number of grid points in x, y, and z directions
//const double nx = 0, ny = 0, nz = 0; // Director field components
const double dx = Lx / Nx, dy = Ly / Ny, dz = Lz / Nz; // Grid spacing in x, y, and z directions

// -------- Q-Tensor Parameters (traceless and symmetric - 5 independent components) ---------- Q: which five are independent? A: Qxx, Qxy, Qxz, Qyy, Qyz)
struct QTensor {
	double Qxx = 0, Qxy = 0, Qxz = 0;
	double Qyx = 0, Qyy = 0, Qyz = 0;
	double Qzx = 0, Qzy = 0, Qzz = 0;
};

// ----------- Functions -----------

// Function to initialize the grid
void initializeGrid(int Nx, int Ny, int Nz, double dx, double dy, double dz);

// Function to initialize the Q-tensor field
// Q has to be sized to Nx*Ny*Nz
// mapping from (q0,theta,phi) -> Q_ij should be uniaxial form
// a uniaxial director n=(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)) with scalar order parameter S should give Q_ij = S*(n_i*n_j-1/3*delta_ij)
void initializeQTensor(std::vector<std::vector<std::vector<QTensor>>>& Q, double q0, double theta0, double phi0);

// Function to compute the Laplacian of the Q-tensor field
// 3D Laplacian: nabla^2Q_ij=(Q_[i+1]-2Q_i+Q_[i-1])/((dx)^2)+(Q_[j+1]-2Q_j+Q_[j-1])/((dy)^2)+(Q_[k+1]-2Q_k+Q_[k-1])/((dz)^2)
// either use ghost points, or reapply BCs after each step
void computeLaplacian(const std::vector<std::vector<std::vector<QTensor>>>& Q, std::vector<std::vector<std::vector<QTensor>>>& laplacianQ);

// Function to compute the free energy density
// f_bulk=A/2*Tr(Q^2)-B/3*Tr(Q^3)+C/4*[Tr(Q^2)]^2
// note: freeEnergyDensity += A*Qxx*Qxx+B*Qxy*Qxy+C*Qxz*Qxz -> use somewhere
// need to compute Tr(Q^2) and Tr(Q^3) from all components
double computeFreeEnergyDensity(const std::vector<std::vector<std::vector<QTensor>>>& Q);

// Function to compute the chemical potential
// Molecular field (chemical potential) is: H_ij=-dF/dQ_ij=-df_bulk/dQ_ij+kappa\nabla^2Q_ij
// need both the bulk - derivative(with A, B, C and traces) and the elastic term
void computeChemicalPotential(const std::vector<std::vector<std::vector<QTensor>>>& Q, std::vector<std::vector<std::vector<QTensor>>>& mu);

// Function to update the Q-tensor field using the finite difference method
// typically the evolution is: dQ_ij/dt=-Gamma*H_ij, Gamma is the rotational viscosity
// probably want to track the max |\Delta Q| to check convergence
void updateQTensor(std::vector<std::vector<std::vector<QTensor>>>& Q, const std::vector<std::vector<std::vector<QTensor>>>& mu, double dt);

// Function to apply boundary conditions
// Strong surface anchoring: Q_ij=S_0(n_i*n_j-1/3*\delta_ij), n_i=x_i/r
void applyBoundaryConditions(std::vector<std::vector<std::vector<QTensor>>>& Q);

// Function to save the Q-tensor field to a file
void saveQTensorToFile(const std::vector<std::vector<std::vector<QTensor>>>& Q, const std::string& filename);

// -------- RECENT CHANGES --------
/*
- changed dt 1e-12 -> 1e-16
- changed maxIterations 10000 -> 100000
*/

// --------- TODO ---------
/*
- implement FDTDR (Finnite-Difference Time-Domain Radial) method for light propagation in radial droplet
- develop a QSB (bipolar) for bipolar (2 boojums at poles) LC droplet
- implement FDTDB (Finite-Differnece Time-Domain Bipolar) method for light propagation in bipolar droplet
*/