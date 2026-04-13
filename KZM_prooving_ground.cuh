#pragma once

#include <cuda_runtime.h>

#ifndef KZM_PG_PI
#define KZM_PG_PI 3.14159265358979323846
#endif

struct XYField {
    double real = 0.0;
    double imag = 0.0;

    __host__ __device__ XYField operator+(const XYField& other) const {
        return {real + other.real, imag + other.imag};
    }

    __host__ __device__ XYField operator-(const XYField& other) const {
        return {real - other.real, imag - other.imag};
    }

    __host__ __device__ XYField operator*(double scalar) const {
        return {real * scalar, imag * scalar};
    }
};

struct ProvingGroundParams {
    int Nx = 64;
    int Ny = 64;
    int Nz = 64;

    double dx = 1.0;
    double dy = 1.0;
    double dz = 1.0;

    double mobility = 1.0;
    double alpha = 1.0;
    double beta = 1.0;
    double kappa = 1.0;

    double Tc = 1.0;
    double Tc_KZ = 1.0;
    double T_high = 1.2;
    double T_low = 0.8;

    int pre_equil_iters = 200;
    int ramp_iters = 300;
    int total_iters = 1000;

    double dt = 0.02;
    double init_amplitude = 1e-3;
    double noise_strength = 0.02;
    double defect_amp_threshold = 0.05;
    long long random_seed = 12345;

    int logFreq = 10;
    int snapshot_mode = 0;
    int snapshotFreq = 100;
};

__global__ void updateXYFieldKernel(
    const XYField* current,
    XYField* next,
    ProvingGroundParams params,
    double reduced_mass,
    double noise_sigma,
    unsigned long long base_seed,
    int iteration
);