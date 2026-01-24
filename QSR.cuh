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
__global__ void applyWeakAnchoringPenaltyKernel(QTensor* mu, const QTensor* Q, int Nx, int Ny, int Nz, const bool* is_shell, double S_shell, double W);
__global__ void updateQTensorKernel(QTensor* Q, const QTensor* mu, int Nx, int Ny, int Nz, double dt, const bool* is_shell, double gamma, double W);
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
