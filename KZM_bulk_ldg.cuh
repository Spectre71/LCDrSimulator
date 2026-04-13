#pragma once

#include <cuda_runtime.h>

#ifndef KZM_BULK_PI
#define KZM_BULK_PI 3.14159265358979323846
#endif

struct QTensor {
    double Qxx = 0.0;
    double Qxy = 0.0;
    double Qxz = 0.0;
    double Qyy = 0.0;
    double Qyz = 0.0;

    __host__ __device__ QTensor operator+(const QTensor& other) const {
        return {Qxx + other.Qxx, Qxy + other.Qxy, Qxz + other.Qxz, Qyy + other.Qyy, Qyz + other.Qyz};
    }

    __host__ __device__ QTensor operator-(const QTensor& other) const {
        return {Qxx - other.Qxx, Qxy - other.Qxy, Qxz - other.Qxz, Qyy - other.Qyy, Qyz - other.Qyz};
    }

    __host__ __device__ QTensor operator*(double scalar) const {
        return {Qxx * scalar, Qxy * scalar, Qxz * scalar, Qyy * scalar, Qyz * scalar};
    }
};

struct FullQTensor {
    double Qxx = 0.0, Qxy = 0.0, Qxz = 0.0;
    double Qyx = 0.0, Qyy = 0.0, Qyz = 0.0;
    double Qzx = 0.0, Qzy = 0.0, Qzz = 0.0;

    __host__ __device__ explicit FullQTensor(const QTensor& q) {
        Qxx = q.Qxx;
        Qxy = q.Qxy;
        Qxz = q.Qxz;
        Qyx = q.Qxy;
        Qyy = q.Qyy;
        Qyz = q.Qyz;
        Qzx = q.Qxz;
        Qzy = q.Qyz;
        Qzz = -(q.Qxx + q.Qyy);
    }

    __host__ __device__ FullQTensor() = default;
};

struct NematicField {
    double S = 0.0;
    double nx = 1.0;
    double ny = 0.0;
    double nz = 0.0;
};

struct BulkLdgParams {
    int Nx = 64;
    int Ny = 64;
    int Nz = 64;

    double dx = 1e-9;
    double dy = 1e-9;
    double dz = 1e-9;

    double a = 0.044e6;
    double b = 1.413e6;
    double c = 1.153e6;
    double T_star = 308.0;
    int bulk_modechoice = 1;

    double kappa = 6.5e-12;
    double gamma = 0.01;

    double T_high = 315.0;
    double T_low = 290.0;
    double Tc_KZ = 310.2;

    int pre_equil_iters = 200;
    int ramp_iters = 300;
    int total_iters = 1200;

    double dt = 1e-11;
    double init_noise_amplitude = 1e-4;
    double noise_strength = 0.0;
    double defects_S_threshold = 0.1;
    double defects_charge_cutoff = 0.25;
    double q_norm_cap = 0.0;
    long long random_seed = 12345;

    int logFreq = 10;
    int snapshot_mode = 0;
    int snapshotFreq = 100;
};

__global__ void updateBulkLdgKernel(
    const QTensor* current,
    QTensor* next,
    BulkLdgParams params,
    double A,
    double B,
    double C,
    double noise_sigma,
    unsigned long long base_seed,
    int iteration
);