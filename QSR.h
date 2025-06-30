#pragma once

// This header file contains the function prototypes and includes necessary libraries for the Landau-De Gennes Q-Tensor solver (QSR) for a radial (hedgehog) 5CB Liquid Crystal droplet with strong surface anchoring int the nematic phase (uniaxial).
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
#include <limits>
#include <sstream>
#include <fstream>
#include <random>
#include <filesystem>

// ---------- Constants ----------
const double PI = 3.14159265358979323846; // Value of pi

// -------- Q-Tensor Parameters (traceless and symmetric - 5 independent components) ---------- Q: which five are independent? A: Qxx, Qxy, Qxz, Qyy, Qyz)
struct QTensor {
	double Qxx = 0, Qxy = 0, Qxz = 0, Qyy = 0, Qyz = 0;
};

struct FullQTensor {
	double Qxx = 0, Qxy = 0, Qxz = 0;
	double Qyx = 0, Qyy = 0, Qyz = 0;
	double Qzx = 0, Qzy = 0, Qzz = 0;

	// Constructor from a 5-component Qtensor
	FullQTensor(const QTensor& q_5comp) {
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
	FullQTensor() = default;
};

// Struct to hold the calculated nematic director and order parameter
struct NematicField {
	double S = 0;
	double nx = 0, ny = 0, nz = 0;
};
NematicField calculateNematicField(const QTensor& q_5comp);

struct EnergyComponents {
	double bulk = 0.0;
	double elastic = 0.0;
	double field = 0.0;
	double total() const { return bulk + elastic + field; }
};
EnergyComponents computeEnergyComponents(const std::vector<QTensor>& Q, int Nx, int Ny, int Nz, double dx, double dy, double dz, double A,
	double B, double C, double kappa, double K_field, double S0);
// ----------- Functions -----------

// Function to initialize the Q-tensor field
// Q has to be sized to Nx*Ny*Nz
// mapping from (q0,theta,phi) -> Q_ij should be uniaxial form
// a uniaxial director n=(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)) with scalar order parameter S should give Q_ij = S*(n_i*n_j-1/3*delta_ij)
void initializeQTensor(std::vector<QTensor>& Q, int Nx, int Ny, int Nz, double q0, int mode);

// Function to compute the Laplacian of the Q-tensor field
// 3D Laplacian: nabla^2Q_ij=(Q_[i+1]-2Q_i+Q_[i-1])/((dx)^2)+(Q_[j+1]-2Q_j+Q_[j-1])/((dy)^2)+(Q_[k+1]-2Q_k+Q_[k-1])/((dz)^2)
// either use ghost points, or reapply BCs after each step
void computeLaplacian(const std::vector<QTensor>& Q, std::vector<QTensor>& laplacianQ, int Nx, int Ny, int Nz, double dx, double dy, double dz);

// Function to compute the free energy density
// f_bulk=A/2*Tr(Q^2)-B/3*Tr(Q^3)+C/4*[Tr(Q^2)]^2
// note: freeEnergyDensity += A*Qxx*Qxx+B*Qxy*Qxy+C*Qxz*Qxz -> use somewhere
// need to compute Tr(Q^2) and Tr(Q^3) from all components
double computeTotalFreeEnergy(const std::vector<QTensor>& Q, int Nx, int Ny, int Nz, double dx,
	double dy, double dz, double A, double B, double C, double kappa, double K_field, double S0);

// Function to compute the chemical potential
// Molecular field (chemical potential) is: H_ij=-dF/dQ_ij=-df_bulk/dQ_ij+kappa\nabla^2Q_ij
// need both the bulk - derivative(with A, B, C and traces) and the elastic term
void computeChemicalPotential(const std::vector<QTensor>& Q, std::vector<QTensor>& mu,
	const std::vector<QTensor>& laplacianQ, int Nx, int Ny, int Nz, double dx, double dy,
	double dz, double A, double B, double C, double kappa, double K_field, double S0);

// Function to update the Q-tensor field using the finite difference method
// typically the evolution is: dQ_ij/dt=-Gamma*H_ij, Gamma is the rotational viscosity
// probably want to track the max |\Delta Q| to check convergence
void updateQTensor(std::vector<QTensor>& Q, const std::vector<QTensor>& mu, int Nx, int Ny, int Nz,
	double dt, const std::vector<bool>& is_shell, double gamma);

// Function to apply boundary conditions
// Strong surface anchoring: Q_ij=S_0(n_i*n_j-1/3*\delta_ij), n_i=x_i/r
void applyBoundaryConditions(std::vector<QTensor>& Q, int Nx, int Ny, int Nz, const std::vector<bool>& is_shell, double S0);

// Function to save the Q-tensor field to a file
void saveQTensorToFile(const std::vector<QTensor>& Q, int Nx, int Ny, int Nz, const std::string& filename);

// helper for user input
template<typename T>
T prompt_with_default(const std::string& prompt, T default_value);

// Function to save the nematic field (director and order parameter) from a Q-tensor
void saveNematicFieldToFile(const std::vector<NematicField>& nematicField, int Nx, int Ny, int Nz, const std::string& filename);

// Function to calculate the average scalar order parameter S in the droplet
double calculateAverageS(const std::vector<QTensor>& Q, int Nx, int Ny, int Nz);

// -------- RECENT CHANGES --------
/*
- reworked the mode selector
- compilation with g++: g++ -O3 -std=c++23 -fopenmp -o QSR QSR.cpp
- changed sign in f_bulk term B (-2.12->2.12)e6
- updated T_star_default phase transition temperature from 307.0 to 308.0
*/

// --------- TODO ---------
/*
- implement FDTDR (Finnite-Difference Time-Domain Radial) method for light propagation in radial droplet
- develop a QSB (bipolar) for bipolar (2 boojums at poles) LC droplet
- implement FDTDB (Finite-Difference Time-Domain Bipolar) method for light propagation in bipolar droplet
- develop CNN to predict LC parameters from images
*/

// --------- DEBUG NOTES ---------
/*
*/