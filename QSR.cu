#include "QSR.cuh"
#include <cctype>
#include <cstdlib>
#include <limits>
#include <tuple>
#include <type_traits>

namespace fs = std::filesystem;

static void cuda_check_or_die(const char* where) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] Kernel launch error after " << where << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] Sync error after " << where << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

static double qtensor_frobenius_norm(const QTensor& q5) {
    FullQTensor q(q5);
    const double trQ2 = q.Qxx*q.Qxx + q.Qyy*q.Qyy + q.Qzz*q.Qzz
        + 2.0*(q.Qxy*q.Qxy + q.Qxz*q.Qxz + q.Qyz*q.Qyz);
    return std::sqrt(std::max(0.0, trQ2));
}

// ------------------------------------------------------------------
// Device Helper Functions
// ------------------------------------------------------------------

__host__ __device__ NematicField calculateNematicFieldDevice(const QTensor& q_5comp) {
    FullQTensor q(q_5comp);

    // Isotropic case
    if (std::abs(q.Qxx) < 1e-9 && std::abs(q.Qxy) < 1e-9 && std::abs(q.Qxz) < 1e-9 &&
        std::abs(q.Qyy) < 1e-9 && std::abs(q.Qyz) < 1e-9) {
        return { 0.0, 0.0, 0.0, 1.0 }; 
    }

    // Power Iteration for dominant eigenvector
    double nx = q.Qxx + q.Qxy + q.Qxz; 
    double ny = q.Qxy + q.Qyy + q.Qyz;
    double nz = q.Qxz + q.Qyz - q.Qxx - q.Qyy; 
    double init_norm = std::sqrt(nx*nx + ny*ny + nz*nz);
    
    if (init_norm > 1e-12) {
        nx /= init_norm; ny /= init_norm; nz /= init_norm;
    } else {
        nx = 0.0; ny = 0.0; nz = 1.0; 
    }
    
    // IMPORTANT: plain power iteration converges to the eigenvector with largest |eigenvalue|.
    // For symmetric traceless Q this can select the most negative eigenvalue in oblate/biaxial regions,
    // producing negative S and a wrong director. We therefore shift: Q' = Q + alpha*I with alpha > -lambda_min.
    // Eigenvectors are unchanged, but all eigenvalues become positive, so power iteration converges to lambda_max.
    double row1 = std::abs(q.Qxx) + std::abs(q.Qxy) + std::abs(q.Qxz);
    double row2 = std::abs(q.Qyx) + std::abs(q.Qyy) + std::abs(q.Qyz);
    double row3 = std::abs(q.Qzx) + std::abs(q.Qzy) + std::abs(q.Qzz);
    double alpha = fmax(row1, fmax(row2, row3)) + 1e-12;

    for (int i = 0; i < 15; ++i) { 
        double q_nx = q.Qxx * nx + q.Qxy * ny + q.Qxz * nz + alpha * nx;
        double q_ny = q.Qyx * nx + q.Qyy * ny + q.Qyz * nz + alpha * ny;
        double q_nz = q.Qzx * nx + q.Qzy * ny + q.Qzz * nz + alpha * nz;

        double norm = std::sqrt(q_nx * q_nx + q_ny * q_ny + q_nz * q_nz);
        if (norm < 1e-13) break; 
        nx = q_nx / norm;
        ny = q_ny / norm;
        nz = q_nz / norm;
    }

    // Rayleigh quotient
    double lambda_max = (nx * (q.Qxx * nx + q.Qxy * ny + q.Qxz * nz) +
        ny * (q.Qyx * nx + q.Qyy * ny + q.Qyz * nz) +
        nz * (q.Qzx * nx + q.Qzy * ny + q.Qzz * nz));

    double S = 1.5 * lambda_max;
    // Gauge fixing
    if (nz < -1e-10) { nx = -nx; ny = -ny; nz = -nz; }

    return { S, nx, ny, nz };
}

__device__ inline double get_deriv_device(const double* data, int idx, int stride, int max_idx, int curr_i, double inv_2d, double inv_d) {
    if (curr_i > 0 && curr_i < max_idx - 1) {
        return (data[idx + stride] - data[idx - stride]) * inv_2d;
    } else if (curr_i == 0) {
        return (data[idx + stride] - data[idx]) * inv_d; 
    } else {
        return (data[idx] - data[idx - stride]) * inv_d; 
    }
}

__device__ inline double get_Q_deriv_device(const QTensor* Q, int idx, int stride, int max_idx, int curr_i, double inv_2d, double inv_d, int comp_offset) {
    // comp_offset: 0=Qxx, 1=Qxy, 2=Qxz, 3=Qyy, 4=Qyz
    auto get_val = [&](int i) { 
        const QTensor& q = Q[i];
        if(comp_offset==0) return q.Qxx;
        if(comp_offset==1) return q.Qxy;
        if(comp_offset==2) return q.Qxz;
        if(comp_offset==3) return q.Qyy;
        return q.Qyz;
    };
    
    if (curr_i > 0 && curr_i < max_idx - 1) {
        return (get_val(idx + stride) - get_val(idx - stride)) * inv_2d;
    } else if (curr_i == 0) {
        return (get_val(idx + stride) - get_val(idx)) * inv_d;
    } else {
        return (get_val(idx) - get_val(idx - stride)) * inv_d;
    }
}

// ------------------------------------------------------------------
// Kernels
// ------------------------------------------------------------------

__global__ void computeLaplacianKernel(const QTensor* Q, QTensor* laplacianQ, int Nx, int Ny, int Nz, double dx, double dy, double dz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= Nx || j >= Ny || k >= Nz) return;

    int idx = k + Nz * (j + Ny * i);
    const QTensor& q_center = Q[idx];
    
    double inv_dx2 = 1.0 / (dx * dx);
    double inv_dy2 = 1.0 / (dy * dy);
    double inv_dz2 = 1.0 / (dz * dz);

    QTensor d2x, d2y, d2z;

    // X-derivative
    if (i > 0 && i < Nx - 1) {
        d2x = (Q[k + Nz * (j + Ny * (i + 1))] - q_center * 2.0 + Q[k + Nz * (j + Ny * (i - 1))]) * inv_dx2;
    } else if (i == 0) {
        d2x = (Q[k + Nz * (j + Ny * (i + 2))] - Q[k + Nz * (j + Ny * (i + 1))] * 2.0 + q_center) * inv_dx2;
    } else { // i == Nx - 1
        d2x = (q_center - Q[k + Nz * (j + Ny * (i - 1))] * 2.0 + Q[k + Nz * (j + Ny * (i - 2))]) * inv_dx2;
    }

    // Y-derivative
    if (j > 0 && j < Ny - 1) {
        d2y = (Q[k + Nz * ((j + 1) + Ny * i)] - q_center * 2.0 + Q[k + Nz * ((j - 1) + Ny * i)]) * inv_dy2;
    } else if (j == 0) {
        d2y = (Q[k + Nz * ((j + 2) + Ny * i)] - Q[k + Nz * ((j + 1) + Ny * i)] * 2.0 + q_center) * inv_dy2;
    } else { // j == Ny - 1
        d2y = (q_center - Q[k + Nz * ((j - 1) + Ny * i)] * 2.0 + Q[k + Nz * ((j - 2) + Ny * i)]) * inv_dy2;
    }

    // Z-derivative
    if (k > 0 && k < Nz - 1) {
        d2z = (Q[(k + 1) + Nz * (j + Ny * i)] - q_center * 2.0 + Q[(k - 1) + Nz * (j + Ny * i)]) * inv_dz2;
    } else if (k == 0) {
        d2z = (Q[(k + 2) + Nz * (j + Ny * i)] - Q[(k + 1) + Nz * (j + Ny * i)] * 2.0 + q_center) * inv_dz2;
    } else { // k == Nz - 1
        d2z = (q_center - Q[(k - 1) + Nz * (j + Ny * i)] * 2.0 + Q[(k - 2) + Nz * (j + Ny * i)]) * inv_dz2;
    }

    laplacianQ[idx] = d2x + d2y + d2z;
}

__global__ void computeDivergenceKernel(const QTensor* Q, double* Dcol_x, double* Dcol_y, double* Dcol_z, int Nx, int Ny, int Nz, double dx, double dy, double dz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= Nx || j >= Ny || k >= Nz) return;

    int idx = k + Nz * (j + Ny * i);
    
    double inv_2dx = 1.0 / (2.0 * dx);
    double inv_2dy = 1.0 / (2.0 * dy);
    double inv_2dz = 1.0 / (2.0 * dz);
    double inv_dx = 1.0 / dx;
    double inv_dy = 1.0 / dy;
    double inv_dz = 1.0 / dz;

    // Dcol_x
    double dQxx_dx = get_Q_deriv_device(Q, idx, Ny*Nz, Nx, i, inv_2dx, inv_dx, 0); 
    double dQxy_dy = get_Q_deriv_device(Q, idx, Nz, Ny, j, inv_2dy, inv_dy, 1);    
    double dQxz_dz = get_Q_deriv_device(Q, idx, 1, Nz, k, inv_2dz, inv_dz, 2);     
    Dcol_x[idx] = dQxx_dx + dQxy_dy + dQxz_dz;

    // Dcol_y
    double dQyx_dx = get_Q_deriv_device(Q, idx, Ny*Nz, Nx, i, inv_2dx, inv_dx, 1); 
    double dQyy_dy = get_Q_deriv_device(Q, idx, Nz, Ny, j, inv_2dy, inv_dy, 3);    
    double dQyz_dz = get_Q_deriv_device(Q, idx, 1, Nz, k, inv_2dz, inv_dz, 4);     
    Dcol_y[idx] = dQyx_dx + dQyy_dy + dQyz_dz;

    // Dcol_z
    double dQzx_dx = get_Q_deriv_device(Q, idx, Ny*Nz, Nx, i, inv_2dx, inv_dx, 2); 
    double dQzy_dy = get_Q_deriv_device(Q, idx, Nz, Ny, j, inv_2dy, inv_dy, 4);    
    
    double Qzz_c = -(Q[idx].Qxx + Q[idx].Qyy);
    double dQzz_dz;
    if (k > 0 && k < Nz - 1) {
            double Qzz_p = -(Q[idx + 1].Qxx + Q[idx + 1].Qyy);
            double Qzz_m = -(Q[idx - 1].Qxx + Q[idx - 1].Qyy);
            dQzz_dz = (Qzz_p - Qzz_m) * inv_2dz;
    } else if (k == 0) {
            double Qzz_p = -(Q[idx + 1].Qxx + Q[idx + 1].Qyy);
            dQzz_dz = (Qzz_p - Qzz_c) * inv_dz;
    } else {
            double Qzz_m = -(Q[idx - 1].Qxx + Q[idx - 1].Qyy);
            dQzz_dz = (Qzz_c - Qzz_m) * inv_dz;
    }
    Dcol_z[idx] = dQzx_dx + dQzy_dy + dQzz_dz;
}

__global__ void computeChemicalPotentialKernel(const QTensor* Q, QTensor* mu, const QTensor* laplacianQ, int Nx, int Ny, int Nz, double dx, double dy, double dz, DimensionalParams params, double kappa, int modechoice, double* Dcol_x, double* Dcol_y, double* Dcol_z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= Nx || j >= Ny || k >= Nz) return;

    int idx = k + Nz * (j + Ny * i);
    FullQTensor q(Q[idx]);
    const QTensor& lap_q = laplacianQ[idx];
    QTensor h; // Local variable for mu

    double dT, coeff_Q, coeff_Q2, coeff_Q3;
    if(modechoice==1){
        dT = params.T - params.T_star;
        coeff_Q = 3.0 * params.a * dT;
        coeff_Q2 = (9.0 * params.b) / 2.0; 
        coeff_Q3 = (9.0 * params.c) / 2.0; 
    }
    else { // mode 2
        dT = params.T - params.T_star;
        coeff_Q = 3.0 * params.a * dT;
        coeff_Q2 = (27.0 * params.b) / 2.0; 
        coeff_Q3 = (9.0 * params.c);        
    }

    // Bulk
    double trQ2 = q.Qxx * q.Qxx + q.Qyy * q.Qyy + q.Qzz * q.Qzz
        + 2 * (q.Qxy * q.Qxy + q.Qxz * q.Qxz + q.Qyz * q.Qyz);
    FullQTensor q2;
    q2.Qxx = q.Qxx * q.Qxx + q.Qxy * q.Qyx + q.Qxz * q.Qzx;
    q2.Qxy = q.Qxx * q.Qxy + q.Qxy * q.Qyy + q.Qxz * q.Qzy;
    q2.Qxz = q.Qxx * q.Qxz + q.Qxy * q.Qyz + q.Qxz * q.Qzz;
    q2.Qyy = q.Qyx * q.Qxy + q.Qyy * q.Qyy + q.Qyz * q.Qzy;
    q2.Qyz = q.Qyx * q.Qxz + q.Qyy * q.Qyz + q.Qyz * q.Qzz;
    q2.Qzz = q.Qzx * q.Qxz + q.Qzy * q.Qyz + q.Qzz * q.Qzz;

    double h_xx = coeff_Q * q.Qxx - coeff_Q2 * (q2.Qxx - trQ2 / 3.0) + coeff_Q3 * trQ2 * q.Qxx;
    double h_yy = coeff_Q * q.Qyy - coeff_Q2 * (q2.Qyy - trQ2 / 3.0) + coeff_Q3 * trQ2 * q.Qyy;
    double h_zz = coeff_Q * q.Qzz - coeff_Q2 * (q2.Qzz - trQ2 / 3.0) + coeff_Q3 * trQ2 * q.Qzz;
    double h_xy = coeff_Q * q.Qxy - coeff_Q2 * q2.Qxy + coeff_Q3 * trQ2 * q.Qxy;
    double h_xz = coeff_Q * q.Qxz - coeff_Q2 * q2.Qxz + coeff_Q3 * trQ2 * q.Qxz;
    double h_yz = coeff_Q * q.Qyz - coeff_Q2 * q2.Qyz + coeff_Q3 * trQ2 * q.Qyz;

    // Elastic
    if (params.L1 == 0.0 && params.L2 == 0.0 && params.L3 == 0.0) {
        h_xx -= kappa * lap_q.Qxx;
        h_yy -= kappa * lap_q.Qyy;
        h_zz -= kappa * (-lap_q.Qxx - lap_q.Qyy);
        h_xy -= kappa * lap_q.Qxy;
        h_xz -= kappa * lap_q.Qxz;
        h_yz -= kappa * lap_q.Qyz;
    }

    if (params.L1 != 0.0) {
        h_xx -= params.L1 * lap_q.Qxx;
        h_yy -= params.L1 * lap_q.Qyy;
        h_zz -= params.L1 * (-lap_q.Qxx - lap_q.Qyy);
        h_xy -= params.L1 * lap_q.Qxy;
        h_xz -= params.L1 * lap_q.Qxz;
        h_yz -= params.L1 * lap_q.Qyz;
    }

    if (params.L2 != 0.0) {
        double inv_2dx = 1.0 / (2.0 * dx);
        double inv_2dy = 1.0 / (2.0 * dy);
        double inv_2dz = 1.0 / (2.0 * dz);
        double inv_dx = 1.0 / dx;
        double inv_dy = 1.0 / dy;
        double inv_dz = 1.0 / dz;

        double dDx_dx = get_deriv_device(Dcol_x, idx, Ny*Nz, Nx, i, inv_2dx, inv_dx);
        double dDy_dy = get_deriv_device(Dcol_y, idx, Nz, Ny, j, inv_2dy, inv_dy);
        double dDz_dz = get_deriv_device(Dcol_z, idx, 1, Nz, k, inv_2dz, inv_dz);
        double dDx_dy = get_deriv_device(Dcol_x, idx, Nz, Ny, j, inv_2dy, inv_dy);
        double dDy_dx = get_deriv_device(Dcol_y, idx, Ny*Nz, Nx, i, inv_2dx, inv_dx);
        double dDx_dz = get_deriv_device(Dcol_x, idx, 1, Nz, k, inv_2dz, inv_dz);
        double dDz_dx = get_deriv_device(Dcol_z, idx, Ny*Nz, Nx, i, inv_2dx, inv_dx);
        double dDy_dz = get_deriv_device(Dcol_y, idx, 1, Nz, k, inv_2dz, inv_dz);
        double dDz_dy = get_deriv_device(Dcol_z, idx, Nz, Ny, j, inv_2dy, inv_dy);

        h_xx -= params.L2 * dDx_dx;
        h_yy -= params.L2 * dDy_dy;
        h_zz -= params.L2 * dDz_dz;
        h_xy -= params.L2 * (dDy_dx + dDx_dy);
        h_xz -= params.L2 * (dDz_dx + dDx_dz);
        h_yz -= params.L2 * (dDz_dy + dDy_dz);
    }

    // Project onto traceless
    double trace_h = h_xx + h_yy + h_zz;
    double corr = trace_h / 3.0;
    h.Qxx = h_xx - corr;
    h.Qyy = h_yy - corr;
    h.Qxy = h_xy;
    h.Qxz = h_xz;
    h.Qyz = h_yz;

    mu[idx] = h;
}

__global__ void computeChemicalPotentialL3Kernel(const QTensor* Q, QTensor* mu, int Nx, int Ny, int Nz,
                                                double dx, double dy, double dz, DimensionalParams params,
                                                const double* Dcol_x, const double* Dcol_y, const double* Dcol_z) {
    if (params.L3 == 0.0) return;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= Nx || j >= Ny || k >= Nz) return;
    if (!(i > 0 && i < Nx - 1 && j > 0 && j < Ny - 1 && k > 0 && k < Nz - 1)) return;

    const int idx = k + Nz * (j + Ny * i);
    FullQTensor q(Q[idx]);

    // Start from existing mu (already includes bulk + L1/L2/kappa and was projected traceless).
    QTensor h0 = mu[idx];
    double h_xx = h0.Qxx;
    double h_yy = h0.Qyy;
    double h_zz = -(h_xx + h_yy);
    double h_xy = h0.Qxy;
    double h_xz = h0.Qxz;
    double h_yz = h0.Qyz;

    const int sx = Ny * Nz;
    const int sy = Nz;
    const int sz = 1;

    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);
    const double inv_2dz = 1.0 / (2.0 * dz);

    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;
    const double inv_dz = 1.0 / dz;

    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);
    const double inv_4dxdy = 1.0 / (4.0 * dx * dy);
    const double inv_4dxdz = 1.0 / (4.0 * dx * dz);
    const double inv_4dydz = 1.0 / (4.0 * dy * dz);

    auto get_comp = [&](int id, int comp) -> double {
        const QTensor& qq = Q[id];
        if (comp == 0) return qq.Qxx;
        if (comp == 1) return qq.Qxy;
        if (comp == 2) return qq.Qxz;
        if (comp == 3) return qq.Qyy;
        if (comp == 4) return qq.Qyz;
        return -(qq.Qxx + qq.Qyy); // Qzz
    };

    // First derivatives of the independent components
    const double dQxx_dx = get_Q_deriv_device(Q, idx, sx, Nx, i, inv_2dx, inv_dx, 0);
    const double dQxx_dy = get_Q_deriv_device(Q, idx, sy, Ny, j, inv_2dy, inv_dy, 0);
    const double dQxx_dz = get_Q_deriv_device(Q, idx, sz, Nz, k, inv_2dz, inv_dz, 0);

    const double dQyy_dx = get_Q_deriv_device(Q, idx, sx, Nx, i, inv_2dx, inv_dx, 3);
    const double dQyy_dy = get_Q_deriv_device(Q, idx, sy, Ny, j, inv_2dy, inv_dy, 3);
    const double dQyy_dz = get_Q_deriv_device(Q, idx, sz, Nz, k, inv_2dz, inv_dz, 3);

    const double dQxy_dx = get_Q_deriv_device(Q, idx, sx, Nx, i, inv_2dx, inv_dx, 1);
    const double dQxy_dy = get_Q_deriv_device(Q, idx, sy, Ny, j, inv_2dy, inv_dy, 1);
    const double dQxy_dz = get_Q_deriv_device(Q, idx, sz, Nz, k, inv_2dz, inv_dz, 1);

    const double dQxz_dx = get_Q_deriv_device(Q, idx, sx, Nx, i, inv_2dx, inv_dx, 2);
    const double dQxz_dy = get_Q_deriv_device(Q, idx, sy, Ny, j, inv_2dy, inv_dy, 2);
    const double dQxz_dz = get_Q_deriv_device(Q, idx, sz, Nz, k, inv_2dz, inv_dz, 2);

    const double dQyz_dx = get_Q_deriv_device(Q, idx, sx, Nx, i, inv_2dx, inv_dx, 4);
    const double dQyz_dy = get_Q_deriv_device(Q, idx, sy, Ny, j, inv_2dy, inv_dy, 4);
    const double dQyz_dz = get_Q_deriv_device(Q, idx, sz, Nz, k, inv_2dz, inv_dz, 4);

    const double dQzz_dx = -(dQxx_dx + dQyy_dx);
    const double dQzz_dy = -(dQxx_dy + dQyy_dy);
    const double dQzz_dz = -(dQxx_dz + dQyy_dz);

    // M_ij = (∂i Qkl)(∂j Qkl)
    const double M_xx = dQxx_dx*dQxx_dx + dQyy_dx*dQyy_dx + dQzz_dx*dQzz_dx
                      + 2.0*(dQxy_dx*dQxy_dx + dQxz_dx*dQxz_dx + dQyz_dx*dQyz_dx);
    const double M_yy = dQxx_dy*dQxx_dy + dQyy_dy*dQyy_dy + dQzz_dy*dQzz_dy
                      + 2.0*(dQxy_dy*dQxy_dy + dQxz_dy*dQxz_dy + dQyz_dy*dQyz_dy);
    const double M_zz = dQxx_dz*dQxx_dz + dQyy_dz*dQyy_dz + dQzz_dz*dQzz_dz
                      + 2.0*(dQxy_dz*dQxy_dz + dQxz_dz*dQxz_dz + dQyz_dz*dQyz_dz);

    const double M_xy = dQxx_dx*dQxx_dy + dQyy_dx*dQyy_dy + dQzz_dx*dQzz_dy
                      + 2.0*(dQxy_dx*dQxy_dy + dQxz_dx*dQxz_dy + dQyz_dx*dQyz_dy);
    const double M_xz = dQxx_dx*dQxx_dz + dQyy_dx*dQyy_dz + dQzz_dx*dQzz_dz
                      + 2.0*(dQxy_dx*dQxy_dz + dQxz_dx*dQxz_dz + dQyz_dx*dQyz_dz);
    const double M_yz = dQxx_dy*dQxx_dz + dQyy_dy*dQyy_dz + dQzz_dy*dQzz_dz
                      + 2.0*(dQxy_dy*dQxy_dz + dQxz_dy*dQxz_dz + dQyz_dy*dQyz_dz);

    // Add (L3/2) M term
    h_xx += 0.5 * params.L3 * M_xx;
    h_yy += 0.5 * params.L3 * M_yy;
    h_zz += 0.5 * params.L3 * M_zz;
    h_xy += 0.5 * params.L3 * M_xy;
    h_xz += 0.5 * params.L3 * M_xz;
    h_yz += 0.5 * params.L3 * M_yz;

    // Helper: compute B(Qcomp) = ∂i( Qij ∂j Qcomp )
    auto compute_B_for_comp = [&](int comp, double dQ_dx, double dQ_dy, double dQ_dz) -> double {
        const double Qc = get_comp(idx, comp);

        const double d2_xx = (get_comp(idx + sx, comp) - 2.0*Qc + get_comp(idx - sx, comp)) * inv_dx2;
        const double d2_yy = (get_comp(idx + sy, comp) - 2.0*Qc + get_comp(idx - sy, comp)) * inv_dy2;
        const double d2_zz = (get_comp(idx + sz, comp) - 2.0*Qc + get_comp(idx - sz, comp)) * inv_dz2;

        const double d2_xy = (get_comp(idx + sx + sy, comp) - get_comp(idx + sx - sy, comp)
                            - get_comp(idx - sx + sy, comp) + get_comp(idx - sx - sy, comp)) * inv_4dxdy;
        const double d2_xz = (get_comp(idx + sx + sz, comp) - get_comp(idx + sx - sz, comp)
                            - get_comp(idx - sx + sz, comp) + get_comp(idx - sx - sz, comp)) * inv_4dxdz;
        const double d2_yz = (get_comp(idx + sy + sz, comp) - get_comp(idx + sy - sz, comp)
                            - get_comp(idx - sy + sz, comp) + get_comp(idx - sy - sz, comp)) * inv_4dydz;

        const double Q_contract = q.Qxx * d2_xx + q.Qyy * d2_yy + q.Qzz * d2_zz
                                + 2.0 * (q.Qxy * d2_xy + q.Qxz * d2_xz + q.Qyz * d2_yz);

        return Dcol_x[idx] * dQ_dx + Dcol_y[idx] * dQ_dy + Dcol_z[idx] * dQ_dz + Q_contract;
    };

    const double B_xx = compute_B_for_comp(0, dQxx_dx, dQxx_dy, dQxx_dz);
    const double B_yy = compute_B_for_comp(3, dQyy_dx, dQyy_dy, dQyy_dz);
    const double B_zz = -(B_xx + B_yy);
    const double B_xy = compute_B_for_comp(1, dQxy_dx, dQxy_dy, dQxy_dz);
    const double B_xz = compute_B_for_comp(2, dQxz_dx, dQxz_dy, dQxz_dz);
    const double B_yz = compute_B_for_comp(4, dQyz_dx, dQyz_dy, dQyz_dz);

    // Subtract L3 * B term
    h_xx -= params.L3 * B_xx;
    h_yy -= params.L3 * B_yy;
    h_zz -= params.L3 * B_zz;
    h_xy -= params.L3 * B_xy;
    h_xz -= params.L3 * B_xz;
    h_yz -= params.L3 * B_yz;

    // Project onto traceless
    const double trace_h = h_xx + h_yy + h_zz;
    const double corr = trace_h / 3.0;
    QTensor hout;
    hout.Qxx = h_xx - corr;
    hout.Qyy = h_yy - corr;
    hout.Qxy = h_xy;
    hout.Qxz = h_xz;
    hout.Qyz = h_yz;
    mu[idx] = hout;
}

__global__ void applyWeakAnchoringPenaltyKernel(QTensor* mu, const QTensor* Q, int Nx, int Ny, int Nz,
                                               const bool* is_shell, double S_shell, double W, double shell_thickness) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= Nx || j >= Ny || k >= Nz) return;
    int idx = k + Nz * (j + Ny * i);

    if (!is_shell[idx]) return;
    if (W == 0.0) return;
    if (!(shell_thickness > 0.0)) return;

    double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0;
    double x = i - cx, y = j - cy, z = k - cz;
    double r = sqrt(x*x + y*y + z*z);
    if (r < 1e-9) return;
    
    double nx = x/r, ny = y/r, nz = z/r;
    QTensor Q0;
    Q0.Qxx = S_shell * (nx*nx - 1.0/3.0);
    Q0.Qxy = S_shell * (nx*ny);
    Q0.Qxz = S_shell * (nx*nz);
    Q0.Qyy = S_shell * (ny*ny - 1.0/3.0);
    Q0.Qyz = S_shell * (ny*nz);
    
    const QTensor& q = Q[idx];
    QTensor& h = mu[idx];

    // Convert surface anchoring strength (J/m^2) to volumetric penalty (J/m^3): W_eff = W / δ.
    const double W_eff = W / shell_thickness;

    h.Qxx += W_eff * (q.Qxx - Q0.Qxx);
    h.Qxy += W_eff * (q.Qxy - Q0.Qxy);
    h.Qxz += W_eff * (q.Qxz - Q0.Qxz);
    h.Qyy += W_eff * (q.Qyy - Q0.Qyy);
    h.Qyz += W_eff * (q.Qyz - Q0.Qyz);
}

__global__ void computeAnchoringEnergyKernel(const QTensor* Q, const bool* is_shell, double* anch_energy,
                                            int Nx, int Ny, int Nz, double dx, double dy, double dz,
                                            double S_shell, double W, double shell_thickness) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= Nx || j >= Ny || k >= Nz) return;
    int idx = k + Nz * (j + Ny * i);

    if (W == 0.0 || !is_shell[idx] || !(shell_thickness > 0.0)) {
        anch_energy[idx] = 0.0;
        return;
    }

    double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0;
    double x = i - cx, y = j - cy, z = k - cz;
    double r = sqrt(x * x + y * y + z * z);
    if (r < 1e-9) {
        anch_energy[idx] = 0.0;
        return;
    }

    double nx = x / r, ny = y / r, nz = z / r;

    // Target Q0 = S_shell (n⊗n - I/3)
    const double Q0_xx = S_shell * (nx * nx - 1.0 / 3.0);
    const double Q0_xy = S_shell * (nx * ny);
    const double Q0_xz = S_shell * (nx * nz);
    const double Q0_yy = S_shell * (ny * ny - 1.0 / 3.0);
    const double Q0_yz = S_shell * (ny * nz);
    const double Q0_zz = -(Q0_xx + Q0_yy);

    FullQTensor q(Q[idx]);
    const double dxx = q.Qxx - Q0_xx;
    const double dxy = q.Qxy - Q0_xy;
    const double dxz = q.Qxz - Q0_xz;
    const double dyy = q.Qyy - Q0_yy;
    const double dyz = q.Qyz - Q0_yz;
    const double dzz = q.Qzz - Q0_zz;

    // Frobenius norm squared for symmetric tensor: dQij dQij
    const double diff2 = dxx * dxx + dyy * dyy + dzz * dzz + 2.0 * (dxy * dxy + dxz * dxz + dyz * dyz);

    // Consistent with applyWeakAnchoringPenaltyKernel (mu += (W/δ)*(Q-Q0)):
    // use volumetric density f = (W/(2δ)) * ||Q - Q0||^2 so F approximates a surface integral.
    anch_energy[idx] = 0.5 * (W / shell_thickness) * diff2 * (dx * dy * dz);
}

__global__ void updateQTensorKernel(QTensor* Q, const QTensor* mu, int Nx, int Ny, int Nz, double dt, const bool* is_shell, double gamma, double W) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= Nx || j >= Ny || k >= Nz) return;
    int idx = k + Nz * (j + Ny * i);

    // Skip shell for strong anchoring
    if (W == 0.0 && is_shell[idx]) return;

    double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0;
    double R = min(min((double)Nx, (double)Ny), (double)Nz) / 2.0;
    double x = i - cx, y = j - cy, z = k - cz;
    double r = sqrt(x * x + y * y + z * z);
    
    if (W == 0.0) { if (r > R) return; } 
    else { if (r > R + 1.5) return; }

    double mobility = 1.0 / gamma;
    const QTensor& current_mu = mu[idx];
    QTensor& current_Q = Q[idx];

    current_Q.Qxx -= mobility * current_mu.Qxx * dt;
    current_Q.Qxy -= mobility * current_mu.Qxy * dt;
    current_Q.Qxz -= mobility * current_mu.Qxz * dt;
    current_Q.Qyy -= mobility * current_mu.Qyy * dt;
    current_Q.Qyz -= mobility * current_mu.Qyz * dt;
}

__global__ void applyBoundaryConditionsKernel(QTensor* Q, int Nx, int Ny, int Nz, const bool* is_shell, double S0, double W) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= Nx || j >= Ny || k >= Nz) return;
    int idx = k + Nz * (j + Ny * i);

    if (W > 0.0) return;

    if (is_shell[idx]) {
        double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0;
        double x = i - cx, y = j - cy, z = k - cz;
        double r = sqrt(x * x + y * y + z * z);
        if (r > 1e-9) {
            double nx = x / r, ny = y / r, nz = z / r;
            QTensor& q = Q[idx];
            q.Qxx = S0 * (nx * nx - 1.0 / 3.0);
            q.Qxy = S0 * (nx * ny);
            q.Qxz = S0 * (nx * nz);
            q.Qyy = S0 * (ny * ny - 1.0 / 3.0);
            q.Qyz = S0 * (ny * nz);
        }
    }
}

__global__ void computeEnergyKernel(const QTensor* Q, double* bulk_energy, double* elastic_energy, int Nx, int Ny, int Nz, double dx, double dy, double dz, DimensionalParams params, double kappa, int modechoice) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= Nx-1 || j >= Ny-1 || k >= Nz-1 || i < 1 || j < 1 || k < 1) {
        // Boundary points don't contribute to energy sum in this simplified integration
        int idx = k + Nz * (j + Ny * i);
        if (idx < Nx*Ny*Nz) {
            bulk_energy[idx] = 0.0;
            elastic_energy[idx] = 0.0;
        }
        return;
    }

    int idx = k + Nz * (j + Ny * i);
    FullQTensor q(Q[idx]);

    double dT, coeff_Q2, coeff_Q3, coeff_Q4;
    if(modechoice==1){
        dT = params.T - params.T_star;
        coeff_Q2 = (3.0 * params.a * dT) / 2.0;
        coeff_Q3 = (3.0 * params.b) / 2.0;
        coeff_Q4 = (9.0 * params.c) / 8.0;
    }
    else {
        dT = params.T - params.T_star;
        coeff_Q2 = (3.0 * params.a * dT) / 2.0;
        coeff_Q3 = (9.0 * params.b) / 2.0;
        coeff_Q4 = (9.0 * params.c) / 4.0;
    }

    // Bulk
    double trQ2 = q.Qxx * q.Qxx + q.Qyy * q.Qyy + q.Qzz * q.Qzz
        + 2 * (q.Qxy * q.Qxy + q.Qxz * q.Qxz + q.Qyz * q.Qyz);
    FullQTensor q2;
    q2.Qxx = q.Qxx * q.Qxx + q.Qxy * q.Qyx + q.Qxz * q.Qzx;
    q2.Qxy = q.Qxx * q.Qxy + q.Qxy * q.Qyy + q.Qxz * q.Qzy;
    q2.Qxz = q.Qxx * q.Qxz + q.Qxy * q.Qyz + q.Qxz * q.Qzz;
    q2.Qyx = q.Qyx * q.Qxx + q.Qyy * q.Qyx + q.Qyz * q.Qzx;
    q2.Qyy = q.Qyx * q.Qxy + q.Qyy * q.Qyy + q.Qyz * q.Qzy;
    q2.Qyz = q.Qyx * q.Qxz + q.Qyy * q.Qyz + q.Qyz * q.Qzz;
    q2.Qzx = q.Qzx * q.Qxx + q.Qzy * q.Qyx + q.Qzz * q.Qzx;
    q2.Qzy = q.Qzx * q.Qxy + q.Qzy * q.Qyy + q.Qzz * q.Qzy;
    q2.Qzz = q.Qzx * q.Qxz + q.Qzy * q.Qyz + q.Qzz * q.Qzz;
    double trQ3 = q.Qxx * q2.Qxx + q.Qxy * q2.Qyx + q.Qxz * q2.Qzx +
                  q.Qyx * q2.Qxy + q.Qyy * q2.Qyy + q.Qyz * q2.Qzy +
                  q.Qzx * q2.Qxz + q.Qzy * q2.Qyz + q.Qzz * q2.Qzz;
    
    bulk_energy[idx] = (coeff_Q2 * trQ2 - coeff_Q3 * trQ3 + coeff_Q4 * trQ2 * trQ2) * (dx * dy * dz);

    // Elastic
    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);
    const double inv_2dz = 1.0 / (2.0 * dz);
    
    auto get_val = [&](int id, int comp) {
        const QTensor& qq = Q[id];
        if(comp==0) return qq.Qxx;
        if(comp==1) return qq.Qxy;
        if(comp==2) return qq.Qxz;
        if(comp==3) return qq.Qyy;
        return qq.Qyz;
    };

    double f_elastic = 0.0;
    
    // Calculate gradients
    double dQ[5][3]; // [component][xyz]
    for(int c=0; c<5; ++c) {
        dQ[c][0] = (get_val(k + Nz * (j + Ny * (i + 1)), c) - get_val(k + Nz * (j + Ny * (i - 1)), c)) * inv_2dx;
        dQ[c][1] = (get_val(k + Nz * ((j + 1) + Ny * i), c) - get_val(k + Nz * ((j - 1) + Ny * i), c)) * inv_2dy;
        dQ[c][2] = (get_val((k + 1) + Nz * (j + Ny * i), c) - get_val((k - 1) + Nz * (j + Ny * i), c)) * inv_2dz;
    }
    
    double dQzz[3];
    dQzz[0] = -(dQ[0][0] + dQ[3][0]);
    dQzz[1] = -(dQ[0][1] + dQ[3][1]);
    dQzz[2] = -(dQ[0][2] + dQ[3][2]);

    auto sq = [&](double x, double y, double z){ return x*x + y*y + z*z; };
    double grad_Q_sq = sq(dQ[0][0], dQ[0][1], dQ[0][2]) + sq(dQ[3][0], dQ[3][1], dQ[3][2]) + sq(dQzz[0], dQzz[1], dQzz[2])
                     + 2.0*(sq(dQ[1][0], dQ[1][1], dQ[1][2]) + sq(dQ[2][0], dQ[2][1], dQ[2][2]) + sq(dQ[4][0], dQ[4][1], dQ[4][2]));

    if (params.L1 == 0.0 && params.L2 == 0.0 && params.L3 == 0.0) {
        f_elastic = 0.5 * kappa * grad_Q_sq;
    } else {
        double fel_L1 = 0.5 * params.L1 * grad_Q_sq;
        double divQ_x = dQ[0][0] + dQ[1][0] + dQ[2][0]; // Qxx_x + Qxy_x + Qxz_x ? No, divQ_x = dQxx/dx + dQxy/dy + dQxz/dz
        // Wait, the indices in dQ are [component][direction]. 
        // dQ[0] is Qxx. dQ[0][0] is dQxx/dx.
        divQ_x = dQ[0][0] + dQ[1][1] + dQ[2][2];
        double divQ_y = dQ[1][0] + dQ[3][1] + dQ[4][2]; // dQyx/dx + dQyy/dy + dQyz/dz
        double divQ_z = dQ[2][0] + dQ[4][1] + dQzz[2];  // dQzx/dx + dQzy/dy + dQzz/dz
        
        double fel_L2 = 0.5 * params.L2 * (divQ_x*divQ_x + divQ_y*divQ_y + divQ_z*divQ_z);

        double fel_L3 = 0.0;
        if (params.L3 != 0.0) {
            // L3 term: (L3/2) Qij (∂i Qkl)(∂j Qkl)
            // First build the symmetric matrix M_ij = (∂i Qkl)(∂j Qkl).
            const double M_xx = dQ[0][0]*dQ[0][0] + dQ[3][0]*dQ[3][0] + dQzz[0]*dQzz[0]
                              + 2.0*(dQ[1][0]*dQ[1][0] + dQ[2][0]*dQ[2][0] + dQ[4][0]*dQ[4][0]);
            const double M_yy = dQ[0][1]*dQ[0][1] + dQ[3][1]*dQ[3][1] + dQzz[1]*dQzz[1]
                              + 2.0*(dQ[1][1]*dQ[1][1] + dQ[2][1]*dQ[2][1] + dQ[4][1]*dQ[4][1]);
            const double M_zz = dQ[0][2]*dQ[0][2] + dQ[3][2]*dQ[3][2] + dQzz[2]*dQzz[2]
                              + 2.0*(dQ[1][2]*dQ[1][2] + dQ[2][2]*dQ[2][2] + dQ[4][2]*dQ[4][2]);

            const double M_xy = dQ[0][0]*dQ[0][1] + dQ[3][0]*dQ[3][1] + dQzz[0]*dQzz[1]
                              + 2.0*(dQ[1][0]*dQ[1][1] + dQ[2][0]*dQ[2][1] + dQ[4][0]*dQ[4][1]);
            const double M_xz = dQ[0][0]*dQ[0][2] + dQ[3][0]*dQ[3][2] + dQzz[0]*dQzz[2]
                              + 2.0*(dQ[1][0]*dQ[1][2] + dQ[2][0]*dQ[2][2] + dQ[4][0]*dQ[4][2]);
            const double M_yz = dQ[0][1]*dQ[0][2] + dQ[3][1]*dQ[3][2] + dQzz[1]*dQzz[2]
                              + 2.0*(dQ[1][1]*dQ[1][2] + dQ[2][1]*dQ[2][2] + dQ[4][1]*dQ[4][2]);

            fel_L3 = 0.5 * params.L3 * (q.Qxx * M_xx + q.Qyy * M_yy + q.Qzz * M_zz
                                     + 2.0*(q.Qxy * M_xy + q.Qxz * M_xz + q.Qyz * M_yz));
        }

        f_elastic = fel_L1 + fel_L2 + fel_L3;
    }

    elastic_energy[idx] = f_elastic * (dx * dy * dz);
}

__global__ void computeRadialityKernel(const QTensor* Q, double* radiality_vals, int* count_vals, int Nx, int Ny, int Nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= Nx || j >= Ny || k >= Nz) return;
    int idx = k + Nz * (j + Ny * i);

    double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0;
    double R = min(min((double)Nx, (double)Ny), (double)Nz) / 2.0;
    double x = i - cx, y = j - cy, z = k - cz;
    double r = sqrt(x*x + y*y + z*z);

    if (r < R && r > 2.0) {
        NematicField nf = calculateNematicFieldDevice(Q[idx]);
        double nx_r = x/r, ny_r = y/r, nz_r = z/r;
        double dot_product = abs(nf.nx*nx_r + nf.ny*ny_r + nf.nz*nz_r);
        radiality_vals[idx] = dot_product;
        count_vals[idx] = 1;
    } else {
        radiality_vals[idx] = 0.0;
        count_vals[idx] = 0;
    }
}

__global__ void computeNematicFieldKernel(const QTensor* Q, NematicField* nf, int Nx, int Ny, int Nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= Nx || j >= Ny || k >= Nz) return;
    int idx = k + Nz * (j + Ny * i);
    nf[idx] = calculateNematicFieldDevice(Q[idx]);
}

// ------------------------------------------------------------------
// Host Helper Functions
// ------------------------------------------------------------------

template<typename T>
T prompt_with_default(const std::string& prompt, T default_value) {
    std::string line;
    std::cout << prompt << " [" << default_value << "]: ";
    std::getline(std::cin, line);
    if (line.empty()) return default_value;

    // NOTE: for integral types, operator>> would parse "1e6" as 1 (stops at 'e').
    // We instead parse via floating-point and round when the target is integral.
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
        try {
            size_t idx = 0;
            long double v = std::stold(line, &idx);
            while (idx < line.size() && std::isspace(static_cast<unsigned char>(line[idx]))) {
                ++idx;
            }
            if (idx != line.size()) return default_value;

            long double vr = std::llround(v);
            const long double lo = static_cast<long double>(std::numeric_limits<T>::min());
            const long double hi = static_cast<long double>(std::numeric_limits<T>::max());
            if (vr < lo || vr > hi) return default_value;
            return static_cast<T>(vr);
        } catch (...) {
            return default_value;
        }
    } else {
        std::istringstream iss(line);
        T value;
        if (!(iss >> value)) return default_value;
        return value;
    }
}

static bool prompt_yes_no(const std::string& prompt, bool default_yes) {
    std::cout << prompt << " (y/n) [" << (default_yes ? "y" : "n") << "]: ";
    std::string line;
    std::getline(std::cin, line);
    if (line.empty()) return default_yes;
    char c = line[0];
    if (c == 'y' || c == 'Y') return true;
    if (c == 'n' || c == 'N') return false;
    return default_yes;
}

static double correlation_length_from_L_and_A(double L, double absA) {
    if (L <= 0.0 || absA <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    return std::sqrt(L / absA);
}

static void bulk_ABC_from_convention(double a, double b, double c, double T, double T_star, int modechoice,
                                    double& A, double& B, double& C) {
    const double dT = T - T_star;
    // These match the bulk molecular field used in computeChemicalPotentialKernel via:
    //   H = A Q - B (Q^2 - I tr(Q^2)/3) + C tr(Q^2) Q
    // which corresponds to a bulk free energy:
    //   f = (A/2) tr(Q^2) - (B/3) tr(Q^3) + (C/4) (tr(Q^2))^2
    A = 3.0 * a * dT;
    if (modechoice == 1) {
        B = (9.0 * b) / 2.0;
        C = (9.0 * c) / 2.0;
    } else {
        B = (27.0 * b) / 2.0;
        C = (9.0 * c);
    }
}

static double bulk_fS_uniaxial(double A, double B, double C, double S) {
    // For Q = S (n⊗n - I/3), invariants give:
    //   tr(Q^2) = 2 S^2 / 3
    //   tr(Q^3) = 2 S^3 / 9
    // With f = (A/2) tr(Q^2) - (B/3) tr(Q^3) + (C/4) (tr(Q^2))^2,
    // this reduces to:
    //   f(S) = (A/3) S^2 - (2B/27) S^3 + (C/9) S^4
    const double S2 = S * S;
    return (A / 3.0) * S2 - (2.0 * B / 27.0) * (S2 * S) + (C / 9.0) * (S2 * S2);
}

static double S_eq_uniaxial_from_ABC(double A, double B, double C) {
    // Equilibrium scalar order parameter for the bulk Landau-de Gennes model.
    // For Q = S (n⊗n - I/3), stationary points satisfy:
    //   2 C S^2 - B S + 3 A = 0
    // A first-order I-N transition generally occurs at A = B^2/(27 C) (>0),
    // so S_eq must allow A>0 and decide via free-energy comparison.
    if (!(C > 0.0)) return 0.0;
    const double disc = B * B - 24.0 * A * C;
    if (!(disc > 0.0)) return 0.0;

    const double sqrt_disc = std::sqrt(disc);
    const double S1 = (B - sqrt_disc) / (4.0 * C);
    const double S2 = (B + sqrt_disc) / (4.0 * C);

    double best_S = 0.0;
    double best_f = 0.0; // f(0)=0
    auto consider = [&](double S) {
        if (!(S > 0.0)) return;
        const double fS = bulk_fS_uniaxial(A, B, C, S);
        if (fS < best_f) {
            best_f = fS;
            best_S = S;
        }
    };
    consider(S1);
    consider(S2);
    return best_S;
}

static double bulk_A_for_transition_NI(double B, double C) {
    // Coexistence (global-min) I-N transition for the uniaxial reduction:
    // A_NI = B^2/(27 C)
    if (!(C > 0.0)) return std::numeric_limits<double>::quiet_NaN();
    return (B * B) / (27.0 * C);
}

static double bulk_A_for_spinodal_nematic(double B, double C) {
    // Upper spinodal where the nematic stationary points disappear:
    // A_spin,N = B^2/(24 C)
    if (!(C > 0.0)) return std::numeric_limits<double>::quiet_NaN();
    return (B * B) / (24.0 * C);
}

static double T_from_A(double A, double a, double T_star) {
    // A = 3 a (T - T*)
    if (!(a > 0.0)) return std::numeric_limits<double>::quiet_NaN();
    return T_star + A / (3.0 * a);
}

// Prints an estimate of the LdG correlation length xi (core size ~ O(xi)) and optionally aborts
// if xi is under-resolved by the spatial discretization.
static bool correlation_length_guard(
    int Nx, int Ny, int Nz,
    double dx, double dy, double dz,
    double a, double b, double c,
    double T, double T_star,
    int modechoice,
    double kappa, double L1, double L2,
    bool guard_enabled,
    double min_ratio_to_pass = 2.0
) {
    const double min_d = std::min({dx, dy, dz});
    const double dT = T - T_star;

    // Bulk coefficients actually used by the selected convention.
    double A = 0.0, B = 0.0, C = 0.0;
    bulk_ABC_from_convention(a, b, c, T, T_star, modechoice, A, B, C);

    // Linearized bulk stiffness around Q≈0: H ≈ A Q
    const double absA_lin = std::abs(A);

    // Nematic-well curvature (heuristic stiffness scale) computed from the same bulk convention.
    const double S_eq = S_eq_uniaxial_from_ABC(A, B, C);
    // For Q = S(nn-I/3), the scalar bulk energy is:
    //   f(S) = (A/3) S^2 - (2B/27) S^3 + (C/9) S^4
    // so f''(S) = 2A/3 - (4B/9) S + (4C/3) S^2
    const double fpp_S = (S_eq > 0.0) ? (2.0 * A / 3.0 - (4.0 * B / 9.0) * S_eq + (4.0 * C / 3.0) * S_eq * S_eq)
                                      : std::numeric_limits<double>::quiet_NaN();
    const double absA_nem = (std::isfinite(fpp_S) ? std::abs(fpp_S) : std::numeric_limits<double>::quiet_NaN());

    // Choose an elastic scale for xi. L1 is the primary gradient stiffness when using L1/L2;
    // otherwise kappa is used.
    const double L_eff = (L1 > 0.0 ? L1 : kappa);

    const double xi_lin = correlation_length_from_L_and_A(L_eff, absA_lin);
    const double xi_nem = correlation_length_from_L_and_A(L_eff, absA_nem);
    const double xi_core = (std::isfinite(xi_nem) ? xi_nem : xi_lin);
    const double ratio = (std::isfinite(xi_core) && min_d > 0.0) ? (xi_core / min_d) : 0.0;
    const double R_phys = 0.5 * std::min({(double)Nx, (double)Ny, (double)Nz}) * min_d;

    std::cout << "\n--- Correlation Length (xi) Estimate ---\n";
    std::cout << "Using L_eff=" << L_eff << " J/m (" << (L1 > 0.0 ? "L1" : "kappa") << ")";
    if (L2 != 0.0) std::cout << ", L2=" << L2;
    std::cout << "\n";
    std::cout << "Bulk convention = " << (modechoice == 1 ? "std" : "ravnik") << "\n";
    std::cout << "Bulk stiffness (linear) |A_lin| = |A| = |3 a (T-T*)| = " << absA_lin << " J/m^3\n";
    if (std::isfinite(fpp_S)) {
        std::cout << "Bulk curvature at S_eq: S_eq=" << S_eq << ", |f''(S_eq)|=" << absA_nem << " (J/m^3, heuristic)\n";
    } else {
        std::cout << "Bulk curvature at S_eq: unavailable (likely isotropic / above T*)\n";
    }
    if (std::isfinite(xi_lin)) std::cout << "xi_lin ≈ sqrt(L_eff/|A_lin|) = " << xi_lin << " m\n";
    if (std::isfinite(xi_nem)) std::cout << "xi_nem ≈ sqrt(L_eff/|f''(S_eq)|) = " << xi_nem << " m\n";
    if (std::isfinite(xi_core)) {
        std::cout << "xi_used = " << xi_core << " m,  xi/min(dx,dy,dz) = " << ratio << "\n";
        std::cout << "Suggested resolution: dx \u2272 xi/3 => dx \u2272 " << (xi_core / 3.0) << " m\n";
        std::cout << "Droplet radius R \u2248 " << R_phys << " m,  R/xi \u2248 " << (R_phys / xi_core) << "\n";
    } else {
        std::cout << "xi estimate unavailable (check L_eff>0 and that T is not exactly T*)\n";
    }

    if (!guard_enabled) return true;

    // Under-resolution heuristic: if xi is less than ~2 grid spacings, the core is not reliably resolved.
    if (std::isfinite(xi_core) && ratio < min_ratio_to_pass) {
        std::cout << "\n[WARN] xi is under-resolved by the grid (xi/dx \u2248 " << ratio
                  << " < " << min_ratio_to_pass << ").\n";
        std::cout << "       Expect mesh-dependent cores and more frequent escaped/uniaxial solutions.\n";
        bool proceed = prompt_yes_no("Proceed anyway?", false);
        return proceed;
    }

    return true;
}

void saveNematicFieldToFile(const std::vector<NematicField>& nematicField, int Nx, int Ny, int Nz, const std::string& filename) {
    FILE* f = fopen(filename.c_str(), "w");
    if (!f) return;
    fprintf(f, "# i j k S nx ny nz\n");
    auto idx = [&](int i, int j, int k) { return k + Nz * (j + Ny * i); };
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                const NematicField& nf = nematicField[idx(i, j, k)];
                fprintf(f, "%d %d %d %.6e %.6e %.6e %.6e\n", i, j, k, nf.S, nf.nx, nf.ny, nf.nz);
            }
        }
    }
    fclose(f);
}

void saveQTensorToFile(const std::vector<QTensor>& Q, int Nx, int Ny, int Nz, const std::string& filename) {
    FILE* f = fopen(filename.c_str(), "w");
    if (!f) return;
    auto idx = [&](int i, int j, int k) { return k + Nz * (j + Ny * i); };
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                FullQTensor q(Q[idx(i, j, k)]);
                fprintf(f, "%d %d %d %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
                    i, j, k, q.Qxx, q.Qxy, q.Qxz, q.Qyx, q.Qyy, q.Qyz, q.Qzx, q.Qzy, q.Qzz);
            }
        }
    }
    fclose(f);
}

void initializeQTensor(std::vector<QTensor>& Q, int Nx, int Ny, int Nz, double q0, int mode) {
    double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0; 
    double R = std::min({ (double)Nx, (double)Ny, (double)Nz }) / 2.0;
    auto idx = [&](int i, int j, int k) {return k + Nz * (j + Ny * i); };
    
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double>dist_theta(0.0, PI);
    std::uniform_real_distribution<double> dist_phi(0.0, 2.0 * PI);

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                double x = i - cx, y = j - cy, z = k - cz;
                double r = std::sqrt(x * x + y * y + z * z);
                double r_min = 1.0; 

                if (r < R) {
                    double S = q0;
                    if (r < r_min) S *= (r / r_min); 

                    QTensor& q = Q[idx(i, j, k)]; 
                    if (mode == 0) { // Random
                        double theta = dist_theta(gen);
                        double phi = dist_phi(gen);
                        double nx = std::sin(theta) * std::cos(phi);
                        double ny = std::sin(theta) * std::sin(phi);
                        double nz = std::cos(theta);
                        q.Qxx = S * (nx * nx - 1.0 / 3.0);
                        q.Qxy = S * (nx * ny);
                        q.Qxz = S * (nx * nz);
                        q.Qyy = S * (ny * ny - 1.0 / 3.0);
                        q.Qyz = S * (ny * nz);
                    }
                    else { // Radial
                        double nx, ny, nz;
                        if (r < 1e-6) { nx=1; ny=0; nz=0; } 
                        else { nx=x/r; ny=y/r; nz=z/r; }
                        q.Qxx = S * (nx * nx - 1.0 / 3.0);
                        q.Qxy = S * (nx * ny);
                        q.Qxz = S * (nx * nz);
                        q.Qyy = S * (ny * ny - 1.0 / 3.0);
                        q.Qyz = S * (ny * nz);
                    }
                } else {
                    Q[idx(i,j,k)] = {0,0,0,0,0};
                }
            }
        }
    }
}

static void initializeIsotropicWithNoise(std::vector<QTensor>& Q, int Nx, int Ny, int Nz, double noise_amplitude) {
    Q.assign((size_t)Nx * (size_t)Ny * (size_t)Nz, {0, 0, 0, 0, 0});
    if (noise_amplitude <= 0.0) return;

    double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0;
    double R = std::min({ (double)Nx, (double)Ny, (double)Nz }) / 2.0;
    auto idx = [&](int i, int j, int k) { return (size_t)k + (size_t)Nz * ((size_t)j + (size_t)Ny * (size_t)i); };

    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> dist(0.0, noise_amplitude);

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                double x = i - cx, y = j - cy, z = k - cz;
                double r = std::sqrt(x * x + y * y + z * z);
                if (r < R) {
                    QTensor& q = Q[idx(i, j, k)];
                    q.Qxx = dist(gen);
                    q.Qxy = dist(gen);
                    q.Qxz = dist(gen);
                    q.Qyy = dist(gen);
                    q.Qyz = dist(gen);
                }
            }
        }
    }
}

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------

int main() {
    bool run_again = false;
    do {
        // Default values
        int Nx_default = 100, Ny_default = 100, Nz_default = 100, maxIterations_default = 100000;
        double dx_default = 1e-8, dy_default = 1e-8, dz_default = 1e-8;
        double a_default = 0.044e6, b_default = 1.413e6, c_default = 1.153e6;      
        double T_default = 300.0, T_star_default = 308.0;   
        double kappa_default = 6.5e-12, gamma_default = 0.1;      
        int modechoice_default=1;      

        std::cout << "\033[1mQ-tensor evolution simulation (CUDA Accelerated)\033[0m\n" << std::endl;
        std::cout << "Press {\033[1;33mEnter\033[0m} to use [\033[1;34mdefault\033[0m] values.\n" << std::endl;

        std::cout << "Select initialization mode: [0] random, [1] radial, [2] isotropic+noise (default: 1): ";
        std::string mode_input; std::getline(std::cin, mode_input);
        int mode = 1;
        if (mode_input == "0") mode = 0;
        else if (mode_input == "2") mode = 2;

        std::cout << "\n--- Spatial Parameters ---" << std::endl;
        int Nx = prompt_with_default("Enter grid size Nx", Nx_default);
        int Ny = prompt_with_default("Enter grid size Ny", Ny_default);
        int Nz = prompt_with_default("Enter grid size Nz", Nz_default);
        double dx = prompt_with_default("Enter dx (m)", dx_default);
        double dy = prompt_with_default("Enter dy (m)", dy_default);
        double dz = prompt_with_default("Enter dz (m)", dz_default);
        
        std::cout << "\n--- Landau-de Gennes Material Parameters ---" << std::endl;
        double a = prompt_with_default("Enter a (J/m^3/K)", a_default);
        double b = prompt_with_default("Enter b (J/m^3)", b_default);
        double c = prompt_with_default("Enter c (J/m^3)", c_default);
        double T = prompt_with_default("Enter T (K)", T_default);
        double T_star = prompt_with_default("Enter T* (K) [A=0, isotropic spinodal]", T_star_default);
        int modechoice = prompt_with_default("Enter Bulk Energy convention (1=std, 2=ravnik)", modechoice_default);

        double A = 0.0, B = 0.0, C = 0.0;
        bulk_ABC_from_convention(a, b, c, T, T_star, modechoice, A, B, C);
        double S_eq = S_eq_uniaxial_from_ABC(A, B, C);
        double S_c = b / (2.0 * c); // heuristic reference only; depends on convention/normalization

          // Report where the bulk model predicts the first-order I-N transition.
          const double A_NI = bulk_A_for_transition_NI(B, C);
          const double A_spinN = bulk_A_for_spinodal_nematic(B, C);
          const double T_NI = T_from_A(A_NI, a, T_star);
          const double T_spinN = T_from_A(A_spinN, a, T_star);

        std::cout << "\nBulk convention: " << (modechoice == 1 ? "std" : "ravnik")
              << "\nComputed S_c (heuristic) = " << S_c
              << ", S_eq(T=" << T << "K) = " << S_eq << std::endl;
          if (std::isfinite(T_NI) && std::isfinite(T_spinN)) {
            std::cout << "Bulk transition estimates (uniaxial, homogeneous):\n"
                    << "  T* (A=0, isotropic spinodal) = " << T_star << " K\n"
                    << "  T_NI (coexistence, global min) ≈ " << T_NI << " K\n"
                    << "  T_spin,N (nematic disappears)  ≈ " << T_spinN << " K\n";
          }
        double S0 = prompt_with_default("Enter initial S0", S_eq > 0 ? S_eq : 0.5);
        
        std::cout << "\n--- Elastic and Dynamic Parameters ---" << std::endl;
        // color yellow text: \033[1;33m
        std::cout << "Generally: \033[1;33m Twist (K2) < Splay (K1) < Bend (K3)\033[0m" << std::endl;
        double kappa = prompt_with_default("Enter kappa (J/m)", kappa_default);
        std::cout << "Use Frank-to-LdG mapping with K1, K2, K3? (y/n) [n]: ";
        std::string use_frank_map_in; std::getline(std::cin, use_frank_map_in);
        bool use_frank_map = (!use_frank_map_in.empty() && (use_frank_map_in[0]=='y' || use_frank_map_in[0]=='Y'));
        double L1 = 0.0, L2 = 0.0, L3 = 0.0;
        double S_ref_used = std::numeric_limits<double>::quiet_NaN();
        if (use_frank_map) {
            double K1 = prompt_with_default("Enter K1", 6.5e-12);
            double K2 = prompt_with_default("Enter K2", 4.0e-12);
            double K3 = prompt_with_default("Enter K3", K1);
            // IMPORTANT: mapping depends on the amplitude convention for Q via S_ref.
            // Default to S_eq(T) computed from the selected bulk convention (std/ravnik),
            // since that keeps Frank->LdG consistent when switching conventions.
            double S_ref_default = (S_eq > 0.0) ? S_eq : ((S0 > 0.0) ? S0 : (b / (2.0 * c)));
            double S_ref = prompt_with_default("Enter S_ref for mapping (default: S_eq(T))", S_ref_default);
            if (S_ref <= 1e-12) S_ref = 0.5;
            S_ref_used = S_ref;

            // Frank -> LdG mapping for the elastic energy used in this code:
            //   f_el = (L1/2) (∂k Qij)(∂k Qij)
            //        + (L2/2) (∂j Qij)(∂k Qik)
            //        + (L3/2) Qij (∂i Qkl)(∂j Qkl)
            // and the uniaxial, constant-S ansatz used throughout this codebase:
            //   Q = S (n⊗n - I/3)
            // Under these conventions, matching to Frank (K1,K2,K3) gives:
            //   L2 = (K1 - K2) / S^2
            //   L3 = (9/4) (K3 - K1) / S^3
            //   L1 = (K1 + K2 - K3) / (2 S^2)
            const double S2 = S_ref * S_ref;
            const double S3 = S2 * S_ref;
            L2 = (K1 - K2) / S2;
            L3 = (9.0 / 4.0) * (K3 - K1) / S3;
            L1 = (K1 + K2 - K3) / (2.0 * S2);
            kappa = 0.0;
            std::cout << "Mapped L1=" << L1 << ", L2=" << L2 << ", L3=" << L3
                      << " (Q=S(nn-I/3), S_ref=" << S_ref << ", Kappa set to 0)" << std::endl;

            // Self-check: reconstruct Frank constants implied by (L1,L2,L3) at the same S_ref.
            // Inverting the mapping yields:
            //   K1 = 2 S^2 L1 + S^2 L2 + (4/9) S^3 L3
            //   K2 = 2 S^2 L1           + (4/9) S^3 L3
            //   K3 = 2 S^2 L1 + S^2 L2 + (8/9) S^3 L3
            const double K1_back = 2.0 * S2 * L1 + S2 * L2 + (4.0 / 9.0) * S3 * L3;
            const double K2_back = 2.0 * S2 * L1 + (4.0 / 9.0) * S3 * L3;
            const double K3_back = 2.0 * S2 * L1 + S2 * L2 + (8.0 / 9.0) * S3 * L3;
            std::cout << "  Check K1_back=" << K1_back << ", K2_back=" << K2_back << ", K3_back=" << K3_back << std::endl;
            std::cout << "  ΔK: (K1_back-K1)=" << (K1_back - K1)
                      << ", (K2_back-K2)=" << (K2_back - K2)
                      << ", (K3_back-K3)=" << (K3_back - K3) << std::endl;

            // Stability notes:
            // - For L3=0, a necessary boundedness condition is L1>0 and (L1+L2)>0.
            // - For L3!=0, boundedness becomes Q-dependent (since the term is cubic in Q and gradients).
            //   We still warn if the L3=0 necessary condition is violated, since it is a strong indicator
            //   of pathological parameter choices.
            if (!(L1 > 0.0) || !((L1 + L2) > 0.0)) {
                std::cout << "\n[WARNING] Potentially unstable elastic parameters (L1=" << L1
                          << ", L2=" << L2 << ", L1+L2=" << (L1 + L2) << ").\n"
                          << "          For L3=0, require L1>0 and L1+L2>0. With L3!=0 this check is not sufficient\n"
                          << "          but violations often lead to blow-up (unphysical |Q| growth).\n";
                bool proceed = prompt_yes_no("Proceed anyway?", false);
                if (!proceed) {
                    std::cout << "Aborting this run; choose different K1/K2/K3 or do not use Frank mapping." << std::endl;
                    std::cout << "Would you like to run another simulation? (y/n): ";
                    std::string again; std::getline(std::cin, again);
                    run_again = (!again.empty() && (again[0] == 'y' || again[0] == 'Y'));
                    continue;
                }
            }
        } else {
            L1 = prompt_with_default("Enter L1", 0.0);
            L2 = prompt_with_default("Enter L2", 0.0);
            L3 = prompt_with_default("Enter L3", 0.0);
            if (L1 != 0 || L2 != 0 || L3 != 0) kappa = 0.0;

            if ((L1 != 0.0 || L2 != 0.0) && (!(L1 > 0.0) || !((L1 + L2) > 0.0))) {
                std::cout << "\n[ERROR] Unstable elastic parameters for this model (L1=" << L1
                          << ", L2=" << L2 << ", L1+L2=" << (L1 + L2) << ").\n"
                          << "        For the implemented L1/L2 form with L3=0, require L1>0 and L1+L2>0.\n";
                bool proceed = prompt_yes_no("Proceed anyway?", false);
                if (!proceed) {
                    std::cout << "Aborting this run; choose stable L1/L2 or use kappa instead." << std::endl;
                    std::cout << "Would you like to run another simulation? (y/n): ";
                    std::string again; std::getline(std::cin, again);
                    run_again = (!again.empty() && (again[0] == 'y' || again[0] == 'Y'));
                    continue;
                }
            }
        }

        bool xi_guard_enabled = prompt_yes_no("Enable correlation-length guard (abort if xi is under-resolved)?", true);
        if (!correlation_length_guard(Nx, Ny, Nz, dx, dy, dz, a, b, c, T, T_star, modechoice, kappa, L1, L2, xi_guard_enabled)) {
            std::cout << "\nAborting this run due to correlation-length guard." << std::endl;
            std::cout << "Tip: decrease dx/dy/dz or move T closer to T* to increase xi." << std::endl;
            std::cout << "Would you like to run another simulation? (y/n): ";
            std::string again; std::getline(std::cin, again);
            run_again = (!again.empty() && (again[0] == 'y' || again[0] == 'Y'));
            continue;
        }

        double W = prompt_with_default("Enter weak anchoring W (J/m^2)", 0.0);
        double gamma = prompt_with_default("Enter gamma (Pa·s)", gamma_default);
        int maxIterations = prompt_with_default("Enter iterations", maxIterations_default);
        int printFreq = prompt_with_default("Enter print freq", 200);
        double tolerance = prompt_with_default("Enter tolerance", 1e-2);
        // Relative convergence threshold for radiality changes between consecutive samples.
        // Example: 1e-2 means "< 1% change".
        double RbEps = prompt_with_default("Enter radiality convergence eps RbEps (relative ΔR̄)", 1e-2);

        // Debugging aids (off by default because they can slow down runs).
        const bool debug_cuda_checks = prompt_yes_no(
            "Debug: enable CUDA error checks at log points? (syncs GPU; slower)",
            false
        );
        const bool debug_dynamics = prompt_yes_no(
            "Debug: print max|mu| and max|ΔQ| at log points? (copies arrays; slower)",
            false
        );

        // Host Memory
        size_t num_elements = Nx * Ny * Nz;
        size_t size_Q = num_elements * sizeof(QTensor);
        size_t size_bool = num_elements * sizeof(bool);
        size_t size_double = num_elements * sizeof(double);
        size_t size_int = num_elements * sizeof(int);

        std::vector<QTensor> h_Q(num_elements);
        std::vector<unsigned char> h_is_shell(num_elements, 0);

        std::vector<QTensor> prev_Q_debug;
        bool have_prev_Q_debug = false;
        
        // Initialize Shell
        double R_geom = std::min({ (double)Nx, (double)Ny, (double)Nz }) / 2.0;
        double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0;
        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                for (int k = 0; k < Nz; ++k) {
                    double r = std::sqrt(std::pow(i - cx, 2) + std::pow(j - cy, 2) + std::pow(k - cz, 2));
                    if (std::abs(r - R_geom) < 1.0) {
                        h_is_shell[k + Nz * (j + Ny * i)] = true;
                    }
                }
            }
        }

        // Device Memory
        QTensor *d_Q, *d_laplacianQ, *d_mu;
        bool *d_is_shell;
        double *d_Dcol_x, *d_Dcol_y, *d_Dcol_z;
        double *d_bulk_energy, *d_elastic_energy;
        double *d_anch_energy;
        double *d_radiality_vals;
        int *d_count_vals;

        cudaMalloc(&d_Q, size_Q);
        cudaMalloc(&d_laplacianQ, size_Q);
        cudaMalloc(&d_mu, size_Q);
        cudaMalloc(&d_is_shell, size_bool);
        cudaMalloc(&d_Dcol_x, size_double);
        cudaMalloc(&d_Dcol_y, size_double);
        cudaMalloc(&d_Dcol_z, size_double);
        cudaMalloc(&d_bulk_energy, size_double);
        cudaMalloc(&d_elastic_energy, size_double);
        cudaMalloc(&d_anch_energy, size_double);
        cudaMalloc(&d_radiality_vals, size_double);
        cudaMalloc(&d_count_vals, size_int);

        // Copy static data
        cudaMemcpy(d_is_shell, h_is_shell.data(), size_bool, cudaMemcpyHostToDevice);

        // CUDA Grid
        dim3 threadsPerBlock(8, 8, 8);
        dim3 numBlocks((Nx + 7) / 8, (Ny + 7) / 8, (Nz + 7) / 8);

        std::cout << "\nSelect simulation mode: [1] Single Temp, [2] Temp Range, [3] Quench (time-dependent T): ";
        int sim_mode = prompt_with_default("", 1);

        auto compute_avg_S_droplet = [&](const std::vector<NematicField>& nf) -> double {
            double total_S = 0.0;
            int count = 0;
            double cx_l = Nx / 2.0, cy_l = Ny / 2.0, cz_l = Nz / 2.0;
            double R_l = std::min({ (double)Nx, (double)Ny, (double)Nz }) / 2.0;
            // Exclude a thin layer near the shell to avoid anchoring-dominated averages.
            const double shell_exclude = 2.0;
            for (int i = 0; i < Nx; ++i) {
                for (int j = 0; j < Ny; ++j) {
                    for (int k = 0; k < Nz; ++k) {
                        double r = std::sqrt(std::pow(i - cx_l, 2) + std::pow(j - cy_l, 2) + std::pow(k - cz_l, 2));
                        if (r < (R_l - shell_exclude)) {
                            size_t id = (size_t)k + (size_t)Nz * ((size_t)j + (size_t)Ny * (size_t)i);
                            total_S += nf[id].S;
                            ++count;
                        }
                    }
                }
            }
            return (count > 0) ? (total_S / count) : 0.0;
        };

        auto reduce_energy = [&](DimensionalParams p, double S_shell) -> std::tuple<double, double, double> {
            computeEnergyKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_bulk_energy, d_elastic_energy, Nx, Ny, Nz, dx, dy, dz, p, kappa, modechoice);
            const double shell_thickness = std::min({dx, dy, dz});
            computeAnchoringEnergyKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_is_shell, d_anch_energy, Nx, Ny, Nz, dx, dy, dz, S_shell, p.W, shell_thickness);

            std::vector<double> h_bulk(num_elements), h_elastic(num_elements), h_anch(num_elements);
            cudaMemcpy(h_bulk.data(), d_bulk_energy, size_double, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_elastic.data(), d_elastic_energy, size_double, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_anch.data(), d_anch_energy, size_double, cudaMemcpyDeviceToHost);

            double bulk_sum = 0.0;
            double elastic_sum = 0.0;
            double anch_sum = 0.0;
            for (size_t ii = 0; ii < num_elements; ++ii) {
                bulk_sum += h_bulk[ii];
                elastic_sum += h_elastic[ii];
                anch_sum += h_anch[ii];
            }
            return {bulk_sum, elastic_sum, anch_sum};
        };

        if (sim_mode == 3) {
            std::cout << "Output directory for quench results [default: output_quench]: ";
            std::string out_dir_name;
            std::getline(std::cin, out_dir_name);
            if (out_dir_name.empty()) out_dir_name = "output_quench";

            fs::path out_dir = out_dir_name;
            if (fs::exists(out_dir)) {
                bool overwrite = prompt_yes_no(
                    ("Directory '" + out_dir.string() + "' exists. Delete it and start fresh?").c_str(),
                    true
                );
                if (overwrite) {
                    fs::remove_all(out_dir);
                }
            }
            fs::create_directories(out_dir);

            int protocol = prompt_with_default("Quench protocol (1=step, 2=ramp)", 2);
            double T_high = prompt_with_default("T_high (K)", T);
            double T_low = prompt_with_default("T_low (K)", T_high - 5.0);
            int pre_equil_iters = prompt_with_default("Pre-equilibration iterations at T_high", 0);
            int ramp_iters = 0;
            if (protocol == 2) {
                ramp_iters = prompt_with_default("Ramp iterations (>=1)", 1000);
                if (ramp_iters < 1) ramp_iters = 1;
            }
            int total_iters = prompt_with_default("Total iterations", maxIterations);
            if (total_iters < 1) total_iters = 1;
            int logFreq = prompt_with_default("Log/print freq", printFreq);
            if (logFreq < 1) logFreq = 1;

            std::cout << "\nSnapshot intent: [0] Off (final only), [1] GIF (regular snapshots), [2] KZ (snapshots around Tc)" << std::endl;
            int snapshot_mode = prompt_with_default("Select snapshot mode", 2);
            if (snapshot_mode < 0) snapshot_mode = 0;
            if (snapshot_mode > 2) snapshot_mode = 2;

            // GIF mode: save every snapshotFreq iterations.
            int snapshotFreq = 0;
            if (snapshot_mode == 1) {
                snapshotFreq = prompt_with_default("GIF snapshot frequency in iterations (0=off)", 10000);
                if (snapshotFreq < 0) snapshotFreq = 0;
            }

            // KZ mode: save snapshots only near the transition. This keeps disk usage low while still
            // letting Python choose an offset after Tc (or do slope-stability sweeps).
            double Tc_KZ = std::numeric_limits<double>::quiet_NaN();
            double Tc_window_K = 0.0;
            int kzSnapshotFreq = 0;
            int kzItersAfterStep = 0;
            bool kz_stop_early = false;
            int kzExtraIters = 0;
            if (snapshot_mode == 2) {
                std::cout << "\n[KZ MODE] Snapshots will be saved around Tc so you can measure at a chosen offset." << std::endl;
                std::cout << "         Use a relatively dense snapshot frequency for flexibility (e.g. 1k–10k iters)." << std::endl;
                Tc_KZ = prompt_with_default("Tc for KZ snapshots (K)", 310.2);
                if (protocol == 2) {
                    Tc_window_K = prompt_with_default("Temperature window half-width around Tc (K)", 0.5);
                    if (Tc_window_K < 0.0) Tc_window_K = -Tc_window_K;
                    kzSnapshotFreq = prompt_with_default("KZ snapshot frequency in iterations", 1000);
                    if (kzSnapshotFreq < 1) kzSnapshotFreq = 1;
                } else {
                    std::cout << "\n[NOTE] Step quench has no Tc ramp. We'll snapshot for a fixed number of iterations after the step." << std::endl;
                    kzItersAfterStep = prompt_with_default("Iterations after the step to record KZ snapshots", 200000);
                    if (kzItersAfterStep < 1) kzItersAfterStep = 1;
                    kzSnapshotFreq = prompt_with_default("KZ snapshot frequency in iterations", 1000);
                    if (kzSnapshotFreq < 1) kzSnapshotFreq = 1;
                }

                kz_stop_early = prompt_yes_no(
                    "In KZ mode, stop simulation once recording window is finished? (saves final state at stop)",
                    true
                );
                if (kz_stop_early) {
                    kzExtraIters = prompt_with_default("Extra iterations after leaving window (for coarsening/offset flexibility)", 0);
                    if (kzExtraIters < 0) kzExtraIters = 0;
                }
            }

            double noise_amplitude = 0.0;
            if (mode == 2) {
                noise_amplitude = prompt_with_default("Noise amplitude for isotropic init", 1e-3);
                initializeIsotropicWithNoise(h_Q, Nx, Ny, Nz, noise_amplitude);
            } else {
                initializeQTensor(h_Q, Nx, Ny, Nz, S0, mode);
            }
            cudaMemcpy(d_Q, h_Q.data(), size_Q, cudaMemcpyHostToDevice);

            double min_dx = std::min({ dx, dy, dz });
            // dt_max estimate: use an effective elastic scale even if L1<=0 or L3 dominates.
            double S_scale = std::isfinite(S_ref_used) ? std::abs(S_ref_used) : std::max({std::abs(S0), std::abs(S_eq), 0.5});
            if (!(S_scale > 0.0)) S_scale = 0.5;
            double L_eff = 0.0;
            if (L1 != 0.0 || L2 != 0.0 || L3 != 0.0) {
                L_eff = std::max({std::abs(L1), std::abs(L1 + L2), std::abs(L3) * S_scale});
            } else {
                L_eff = std::abs(kappa);
            }
            if (!(L_eff > 0.0)) L_eff = 1e-12;
            double D = L_eff / gamma; if (D == 0) D = 1e-12;
            const double dt_diff = 0.5 * (min_dx * min_dx) / (6.0 * D);

            // Bulk stiffness can be much larger than elastic diffusion (especially deep in nematic),
            // so explicit Euler typically needs a dt cap based on |A|,|B|,|C|.
            auto bulk_rate_at_T = [&](double Tval) -> double {
                double A_t = 0.0, B_t = 0.0, C_t = 0.0;
                bulk_ABC_from_convention(a, b, c, Tval, T_star, modechoice, A_t, B_t, C_t);
                const double S2 = S_scale * S_scale;
                return std::abs(A_t) + std::abs(B_t) * S_scale + std::abs(C_t) * S2;
            };
            const double bulk_rate = std::max(bulk_rate_at_T(T_high), bulk_rate_at_T(T_low));
            const double dt_bulk = (bulk_rate > 0.0) ? (0.1 * gamma / bulk_rate) : std::numeric_limits<double>::infinity();

            double dt_max = std::min(dt_diff, dt_bulk);
            if (!(dt_max > 0.0) || !std::isfinite(dt_max)) dt_max = dt_diff;

            std::cout << "\nMaximum stable dt estimates:\n"
                      << "  dt_diff (elastic) ≈ " << dt_diff << " s\n"
                      << "  dt_bulk (bulk)    ≈ " << dt_bulk << " s\n"
                      << "  dt_max            = " << dt_max << " s" << std::endl;
            double dt = prompt_with_default("Enter time step dt (s)", dt_max);
            if (dt > dt_max) dt = dt_max;

            // Optional convergence/early-stop guard (useful when you only care about final aligned state).
            // For a quench/ramp, we only allow early-stop once the protocol has reached the final temperature.
            // Criterion: relative energy change + relative radiality change between consecutive samples,
            // optionally combined with an absolute radiality threshold.
            const bool enable_early_stop = prompt_yes_no(
                "Enable early-stop once converged at final T? (rel. ΔF/F + rel. ΔR̄/R̄)",
                false
            );
            const double radiality_threshold = enable_early_stop
                ? prompt_with_default("Radiality threshold (Rbar, 0=disable)", 0.998)
                : 0.0;

            char user_choice = 'y';
            std::cout << "Do you want output in the console every " << logFreq << " iterations? (y/n): ";
            std::string user_choice_line;
            std::getline(std::cin, user_choice_line);
            if (!user_choice_line.empty()) user_choice = user_choice_line[0];

            std::ofstream quench_log((out_dir / "quench_log.dat").string());
            quench_log << "iteration,time_s,T_K,bulk,elastic,anchoring,total,radiality,avg_S\n";

            auto compute_T_current = [&](int iter) -> double {
                if (iter < pre_equil_iters) return T_high;
                if (protocol == 1) return T_low;
                const int t = iter - pre_equil_iters;
                if (t <= 0) return T_high;
                if (t >= ramp_iters) return T_low;
                const double alpha = (double)t / (double)ramp_iters;
                return T_high + (T_low - T_high) * alpha;
            };

            auto has_reached_final_T = [&](int iter) -> bool {
                if (iter < pre_equil_iters) return false;
                if (protocol == 1) return true; // step: after pre-equil we are at T_low
                return (iter >= pre_equil_iters + ramp_iters);
            };

            auto compute_S_shell = [&](double Tcur) -> double {
                double A_c = 0.0, B_c = 0.0, C_c = 0.0;
                bulk_ABC_from_convention(a, b, c, Tcur, T_star, modechoice, A_c, B_c, C_c);
                double S_eq_cur = S_eq_uniaxial_from_ABC(A_c, B_c, C_c);
                return (S_eq_cur > 0.0) ? S_eq_cur : 0.0;
            };

            double physical_time = 0.0;
            double prev_F_for_stop = std::numeric_limits<double>::quiet_NaN();
            double prev_R_for_stop = std::numeric_limits<double>::quiet_NaN();
            bool kz_entered_window = false;
            int kz_exit_iter = -1;
            for (int iter = 0; iter < total_iters; ++iter) {
                const double T_current = compute_T_current(iter);
                DimensionalParams params_q = {a, b, c, T_current, T_star, L1, L2, L3, W};
                const double S_shell = compute_S_shell(T_current);

                computeLaplacianKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_laplacianQ, Nx, Ny, Nz, dx, dy, dz);
                if (L2 != 0.0 || L3 != 0.0) computeDivergenceKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_Dcol_x, d_Dcol_y, d_Dcol_z, Nx, Ny, Nz, dx, dy, dz);
                computeChemicalPotentialKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_mu, d_laplacianQ, Nx, Ny, Nz, dx, dy, dz, params_q, kappa, modechoice, d_Dcol_x, d_Dcol_y, d_Dcol_z);
                if (L3 != 0.0) {
                    computeChemicalPotentialL3Kernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_mu, Nx, Ny, Nz, dx, dy, dz, params_q, d_Dcol_x, d_Dcol_y, d_Dcol_z);
                }
                const double shell_thickness = std::min({dx, dy, dz});
                applyWeakAnchoringPenaltyKernel<<<numBlocks, threadsPerBlock>>>(d_mu, d_Q, Nx, Ny, Nz, d_is_shell, S_shell, W, shell_thickness);
                updateQTensorKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_mu, Nx, Ny, Nz, dt, d_is_shell, gamma, W);
                applyBoundaryConditionsKernel<<<numBlocks, threadsPerBlock>>>(d_Q, Nx, Ny, Nz, d_is_shell, S_shell, W);

                physical_time += dt;

                const bool do_log = (iter % logFreq == 0) || (iter == total_iters - 1);

                bool in_kz_window = false;
                bool do_snap = false;
                if (snapshot_mode == 1) {
                    // GIF: regular snapshots.
                    do_snap = (snapshotFreq > 0) && ((iter % snapshotFreq == 0) || (iter == total_iters - 1));
                } else if (snapshot_mode == 2) {
                    // KZ: snapshots only around Tc (ramp) or after step (step quench).
                    if (protocol == 2) {
                        if (std::isfinite(Tc_KZ) && std::isfinite(Tc_window_K) && Tc_window_K > 0.0) {
                            in_kz_window = (std::abs(T_current - Tc_KZ) <= Tc_window_K);
                        } else {
                            // If window is degenerate, fall back to saving around the crossing point only.
                            in_kz_window = std::isfinite(Tc_KZ) ? (std::abs(T_current - Tc_KZ) <= 1e-12) : false;
                        }
                    } else {
                        // Step quench: snapshot early-time ordering after the step.
                        const int step_iter = pre_equil_iters;
                        in_kz_window = (iter >= step_iter) && (iter <= step_iter + kzItersAfterStep);
                    }

                    // Optional: keep saving snapshots for a short period after leaving the window.
                    bool in_post = false;
                    if (kz_stop_early && kzExtraIters > 0 && kz_exit_iter >= 0) {
                        in_post = (iter > kz_exit_iter) && (iter <= kz_exit_iter + kzExtraIters);
                    }
                    do_snap = (kzSnapshotFreq > 0) && ((iter % kzSnapshotFreq) == 0) && (in_kz_window || in_post);
                }

                if (do_log || do_snap) {
                    computeRadialityKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_radiality_vals, d_count_vals, Nx, Ny, Nz);
                    auto [bulk_sum, elastic_sum, anch_sum] = reduce_energy(params_q, S_shell);
                    double total_F = bulk_sum + elastic_sum + anch_sum;

                    std::vector<double> h_rad(num_elements);
                    std::vector<int> h_count(num_elements);
                    cudaMemcpy(h_rad.data(), d_radiality_vals, size_double, cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_count.data(), d_count_vals, size_int, cudaMemcpyDeviceToHost);
                    double total_rad = 0.0;
                    int rad_count = 0;
                    for (size_t ii = 0; ii < num_elements; ++ii) {
                        total_rad += h_rad[ii];
                        rad_count += h_count[ii];
                    }
                    double avg_rad = (rad_count > 0) ? total_rad / rad_count : 0.0;

                    NematicField* d_nf;
                    cudaMalloc(&d_nf, num_elements * sizeof(NematicField));
                    computeNematicFieldKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_nf, Nx, Ny, Nz);
                    std::vector<NematicField> h_nf(num_elements);
                    cudaMemcpy(h_nf.data(), d_nf, num_elements * sizeof(NematicField), cudaMemcpyDeviceToHost);
                    cudaFree(d_nf);
                    double avg_S = compute_avg_S_droplet(h_nf);

                    if (do_log) {
                        quench_log << iter << "," << physical_time << "," << T_current << "," << bulk_sum << "," << elastic_sum
                                  << "," << anch_sum << "," << total_F << "," << avg_rad << "," << avg_S << "\n";
                        quench_log.flush();
                        if (user_choice == 'y' || user_choice == 'Y') {
                            std::cout << "Iter " << iter << "  t=" << physical_time << " s  T=" << T_current
                                      << " K  F=" << total_F << " (bulk=" << bulk_sum << ", el=" << elastic_sum
                                      << ", anch=" << anch_sum << ")  Rbar=" << avg_rad << "  <S>=" << avg_S << std::endl;
                        }

                        if (enable_early_stop && has_reached_final_T(iter)) {
                            bool energy_converged = false;
                            bool radiality_converged = false;

                            if (std::isfinite(prev_F_for_stop) && prev_F_for_stop != 0.0) {
                                const double rel_dF = std::abs((total_F - prev_F_for_stop) / prev_F_for_stop);
                                energy_converged = rel_dF < tolerance;
                            }
                            if (std::isfinite(prev_R_for_stop) && prev_R_for_stop != 0.0) {
                                const double rel_dR = std::abs((avg_rad - prev_R_for_stop) / prev_R_for_stop);
                                radiality_converged = rel_dR < RbEps;
                            }

                            const bool radially_aligned = (radiality_threshold <= 0.0) ? true : (avg_rad > radiality_threshold);
                            if (energy_converged && radiality_converged && radially_aligned) {
                                std::cout << "\n=== Early-stop: converged at final T ===\n";
                                break;
                            }

                            // Update references for the *next* convergence check.
                            prev_F_for_stop = total_F;
                            prev_R_for_stop = avg_rad;
                        }
                    }

                    if (do_snap) {
                        saveNematicFieldToFile(h_nf, Nx, Ny, Nz, (out_dir / ("nematic_field_iter_" + std::to_string(iter) + ".dat")).string());
                    }
                }

                // KZ speed-up: if we only care about the transition window, end the run once the
                // recording window has finished (plus optional extra iterations).
                if (snapshot_mode == 2 && kz_stop_early) {
                    if (protocol == 2) {
                        if (in_kz_window) {
                            kz_entered_window = true;
                        }
                        // Stop after we've entered the window and then cooled below Tc - ΔT.
                        const bool below_window = (std::isfinite(Tc_KZ) && std::isfinite(Tc_window_K)) ? (T_current < (Tc_KZ - Tc_window_K)) : false;
                        if (kz_entered_window && !in_kz_window && below_window) {
                            if (kz_exit_iter < 0) kz_exit_iter = iter;
                            if (iter >= kz_exit_iter + kzExtraIters) {
                                std::cout << "\n=== KZ mode: stopping after leaving Tc window ===\n";
                                break;
                            }
                        }
                    } else {
                        const int step_iter = pre_equil_iters;
                        const int stop_iter = step_iter + kzItersAfterStep + kzExtraIters;
                        if (iter >= stop_iter) {
                            std::cout << "\n=== KZ mode: stopping after post-step recording window ===\n";
                            break;
                        }
                    }
                }
            }

            // Final save
            cudaMemcpy(h_Q.data(), d_Q, size_Q, cudaMemcpyDeviceToHost);
            NematicField* d_nf;
            cudaMalloc(&d_nf, num_elements * sizeof(NematicField));
            computeNematicFieldKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_nf, Nx, Ny, Nz);
            std::vector<NematicField> finalNF(num_elements);
            cudaMemcpy(finalNF.data(), d_nf, num_elements * sizeof(NematicField), cudaMemcpyDeviceToHost);
            cudaFree(d_nf);
            saveNematicFieldToFile(finalNF, Nx, Ny, Nz, (out_dir / "nematic_field_final.dat").string());
            saveQTensorToFile(h_Q, Nx, Ny, Nz, (out_dir / "Qtensor_output_final.dat").string());
            quench_log.close();

        } else if (sim_mode == 2) {
            if (fs::exists("output_temp_sweep")) fs::remove_all("output_temp_sweep");
            fs::create_directory("output_temp_sweep");
            std::ofstream sweep_log("output_temp_sweep/summary.dat");
            sweep_log << "temperature,final_energy_including_anchoring,average_S\n";

            double T_start = prompt_with_default("Start T", 295.0);
            double T_end = prompt_with_default("End T", 315.0);
            double T_step = prompt_with_default("Step T", 1.0);

            if (T_step == 0.0) {
                std::cout << "Step T cannot be 0. Aborting sweep." << std::endl;
                sweep_log.close();
                return 0;
            }
            // If user provides a step sign inconsistent with the sweep direction, auto-fix it.
            if ((T_end - T_start) * T_step < 0.0) {
                T_step = -T_step;
            }

            // Give a heads-up if the requested sweep range is unlikely to cross the bulk transition.
            {
                double A0 = 0.0, B0 = 0.0, C0 = 0.0;
                bulk_ABC_from_convention(a, b, c, 0.0, 0.0, modechoice, A0, B0, C0);
                const double A_NI = bulk_A_for_transition_NI(B0, C0);
                const double A_spinN = bulk_A_for_spinodal_nematic(B0, C0);
                const double T_NI = T_from_A(A_NI, a, T_star);
                const double T_spinN = T_from_A(A_spinN, a, T_star);
                if (std::isfinite(T_NI) && (std::max(T_start, T_end) < T_NI)) {
                    std::cout << "\n[NOTE] Sweep max(T) < T_NI (" << T_NI << " K)."
                              << " You should not expect a bulk I→N transition within this range." << std::endl;
                }
                if (std::isfinite(T_spinN) && (std::min(T_start, T_end) > T_spinN)) {
                    std::cout << "\n[NOTE] Sweep min(T) > T_spin,N (" << T_spinN << " K)."
                              << " Bulk nematic minimum does not exist; expect isotropic." << std::endl;
                }
            }

            if (mode == 2) {
                double noise_amplitude = prompt_with_default("Noise amplitude for isotropic init", 1e-3);
                initializeIsotropicWithNoise(h_Q, Nx, Ny, Nz, noise_amplitude);
            } else {
                initializeQTensor(h_Q, Nx, Ny, Nz, S0, mode);
            }
            cudaMemcpy(d_Q, h_Q.data(), size_Q, cudaMemcpyHostToDevice);

            double min_dx = std::min({ dx, dy, dz });
            // dt_max estimate: use an effective elastic scale even if L1<=0 or L3 dominates.
            double S_scale = std::isfinite(S_ref_used) ? std::abs(S_ref_used) : std::max({std::abs(S0), std::abs(S_eq), 0.5});
            if (!(S_scale > 0.0)) S_scale = 0.5;
            double L_eff = 0.0;
            if (L1 != 0.0 || L2 != 0.0 || L3 != 0.0) {
                L_eff = std::max({std::abs(L1), std::abs(L1 + L2), std::abs(L3) * S_scale});
            } else {
                L_eff = std::abs(kappa);
            }
            if (!(L_eff > 0.0)) L_eff = 1e-12;
            double D = L_eff / gamma; if(D==0) D=1e-12;
            const double dt_diff = 0.5 * (min_dx * min_dx) / (6.0 * D);

            auto bulk_rate_at_T = [&](double Tval) -> double {
                double A_dt = 0.0, B_dt = 0.0, C_dt = 0.0;
                bulk_ABC_from_convention(a, b, c, Tval, T_star, modechoice, A_dt, B_dt, C_dt);
                return std::abs(A_dt) + std::abs(B_dt) * S_scale + std::abs(C_dt) * S_scale * S_scale;
            };
            const double bulk_rate = std::max(bulk_rate_at_T(T_start), bulk_rate_at_T(T_end));
            const double dt_bulk = (bulk_rate > 0.0)
                ? (0.1 * gamma / bulk_rate)
                : std::numeric_limits<double>::infinity();

            double dt_max = std::min(dt_diff, dt_bulk);
            if (!(dt_max > 0.0) || !std::isfinite(dt_max)) dt_max = dt_diff;

            std::cout << "\nMaximum stable dt estimates:\n"
                      << "  dt_diff (elastic) ≈ " << dt_diff << " s\n"
                      << "  dt_bulk (bulk)    ≈ " << dt_bulk << " s\n"
                      << "  dt_max            = " << dt_max << " s" << std::endl;
            double dt = prompt_with_default("Enter time step dt (s)", dt_max);
            if (dt > dt_max) dt = dt_max;

            // Simple physical-time floor to reduce "premature convergence" in the sweep.
            double R_phys = (std::min({(double)Nx, (double)Ny, (double)Nz}) / 2.0) * min_dx;
            double tau_align = (R_phys * R_phys) / D;
            double min_alignment_time = 0.25 * tau_align;
            std::cout << "Estimated alignment time τ_align ≈ " << tau_align << " s"
                      << ", enforcing minimum sweep time ≈ " << min_alignment_time << " s" << std::endl;

            auto sweep_done = [&](double Tval) {
                const double eps = 1e-12;
                return (T_step > 0.0) ? (Tval > T_end + eps) : (Tval < T_end - eps);
            };

            for (double T_current = T_start; !sweep_done(T_current); T_current += T_step) {
                std::cout << "\n--- Running simulation for T = " << T_current << " K ---\n"; 
                DimensionalParams params_temp = {a, b, c, T_current, T_star, L1, L2, L3, W};

                double A_s = 0.0, B_s = 0.0, C_s = 0.0;
                bulk_ABC_from_convention(a, b, c, T_current, T_star, modechoice, A_s, B_s, C_s);
                double S_eq_sweep = S_eq_uniaxial_from_ABC(A_s, B_s, C_s);
                double S_shell = S_eq_sweep;

                std::string temp_dir = "output_temp_sweep/T_" + std::to_string(T_current);
                fs::create_directory(temp_dir);

                double prev_F = std::numeric_limits<double>::max();
                double physical_time = 0.0;
                for (int iter = 0; iter < maxIterations; ++iter) {
                    computeLaplacianKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_laplacianQ, Nx, Ny, Nz, dx, dy, dz);
                    if (L2 != 0.0 || L3 != 0.0) computeDivergenceKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_Dcol_x, d_Dcol_y, d_Dcol_z, Nx, Ny, Nz, dx, dy, dz);
                    computeChemicalPotentialKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_mu, d_laplacianQ, Nx, Ny, Nz, dx, dy, dz, params_temp, kappa, modechoice, d_Dcol_x, d_Dcol_y, d_Dcol_z);
                    if (L3 != 0.0) {
                        computeChemicalPotentialL3Kernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_mu, Nx, Ny, Nz, dx, dy, dz, params_temp, d_Dcol_x, d_Dcol_y, d_Dcol_z);
                    }
                    const double shell_thickness = std::min({dx, dy, dz});
                    applyWeakAnchoringPenaltyKernel<<<numBlocks, threadsPerBlock>>>(d_mu, d_Q, Nx, Ny, Nz, d_is_shell, S_shell, W, shell_thickness);
                    updateQTensorKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_mu, Nx, Ny, Nz, dt, d_is_shell, gamma, W);
                    applyBoundaryConditionsKernel<<<numBlocks, threadsPerBlock>>>(d_Q, Nx, Ny, Nz, d_is_shell, S_shell, W);

                    physical_time += dt;
                    
                    if (iter > 0 && iter % printFreq == 0) {
                        auto [bulk_sum, elastic_sum, anch_sum] = reduce_energy(params_temp, S_shell);
                        double total_F = bulk_sum + elastic_sum + anch_sum;
                        std::cout << "  Iter " << iter << ", Free Energy: " << total_F << std::endl;
                        if (physical_time >= min_alignment_time) {
                            if (prev_F != 0.0 && (std::abs((total_F - prev_F) / prev_F) < tolerance || total_F == 0.0)) {
                                std::cout << "  Convergence reached at iteration " << iter << std::endl;
                                break;
                            }
                        }
                        prev_F = total_F;
                    }
                }

                cudaMemcpy(h_Q.data(), d_Q, size_Q, cudaMemcpyDeviceToHost);
                NematicField* d_nf;
                cudaMalloc(&d_nf, num_elements * sizeof(NematicField));
                computeNematicFieldKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_nf, Nx, Ny, Nz);
                std::vector<NematicField> finalNF(num_elements);
                cudaMemcpy(finalNF.data(), d_nf, num_elements * sizeof(NematicField), cudaMemcpyDeviceToHost);
                cudaFree(d_nf);

                auto [bulk_final, elastic_final, anch_final] = reduce_energy(params_temp, S_shell);
                double final_F = bulk_final + elastic_final + anch_final;
                double avg_S = compute_avg_S_droplet(finalNF);
                sweep_log << T_current << "," << final_F << "," << avg_S << "\n";
                sweep_log.flush();

                saveNematicFieldToFile(finalNF, Nx, Ny, Nz, temp_dir + "/nematic_field_final.dat");
                saveQTensorToFile(h_Q, Nx, Ny, Nz, temp_dir + "/Qtensor_output_final.dat");
                std::cout << "  Saved final state for T = " << T_current << " K. Average S = " << avg_S << std::endl;
            }
            sweep_log.close();

        } else {
            // Single Temp Mode
            int submode_default = 1;
            std::cout << "Select submode: [1] full energy, [2] energy components (default: 1): ";
            std::string submode_input;
            std::getline(std::cin, submode_input);
            int submode = submode_default;
            if (!submode_input.empty()) {
                try { submode = std::stoi(submode_input); }
                catch (...) { submode = submode_default; }
            }
            if (submode != 1 && submode != 2) submode = submode_default;
            std::cout << "Submode selected: " << (submode == 1 ? "full energy" : "energy components") << std::endl;

            if (fs::exists("output")) { for (const auto& entry : fs::directory_iterator("output")) fs::remove(entry.path()); }
            fs::create_directory("output");

            DimensionalParams params = {a, b, c, T, T_star, L1, L2, L3, W};
            double min_dx = std::min({ dx, dy, dz });
            // dt_max estimate: use an effective elastic scale even if L1<=0 or L3 dominates.
            double S_scale = std::isfinite(S_ref_used) ? std::abs(S_ref_used) : std::max({std::abs(S0), std::abs(S_eq), 0.5});
            if (!(S_scale > 0.0)) S_scale = 0.5;
            double L_eff = 0.0;
            if (L1 != 0.0 || L2 != 0.0 || L3 != 0.0) {
                L_eff = std::max({std::abs(L1), std::abs(L1 + L2), std::abs(L3) * S_scale});
            } else {
                L_eff = std::abs(kappa);
            }
            if (!(L_eff > 0.0)) L_eff = 1e-12;
            double D = L_eff / gamma; if(D==0) D=1e-12;
            const double dt_diff = 0.5 * (min_dx * min_dx) / (6.0 * D);

            // Bulk-driven dt cap (single temperature)
            double A_dt = 0.0, B_dt = 0.0, C_dt = 0.0;
            bulk_ABC_from_convention(a, b, c, T, T_star, modechoice, A_dt, B_dt, C_dt);
            const double bulk_rate = std::abs(A_dt) + std::abs(B_dt) * S_scale + std::abs(C_dt) * S_scale * S_scale;
            const double dt_bulk = (bulk_rate > 0.0) ? (0.1 * gamma / bulk_rate) : std::numeric_limits<double>::infinity();

            double dt_max = std::min(dt_diff, dt_bulk);
            if (!(dt_max > 0.0) || !std::isfinite(dt_max)) dt_max = dt_diff;

            double R_phys = (std::min({(double)Nx, (double)Ny, (double)Nz}) / 2.0) * min_dx;
            double tau_align = (R_phys * R_phys) / D;
            std::cout << "\n--- Time Scale Analysis ---" << std::endl;
            std::cout << "Diffusion coefficient D = " << D << " m²/s" << std::endl;
            std::cout << "Droplet radius R ≈ " << R_phys << " m" << std::endl;
            std::cout << "Estimated alignment time τ_align ≈ " << tau_align << " s" << std::endl;
            std::cout << "Maximum stable dt estimates:" << std::endl;
            std::cout << "  dt_diff (elastic) ≈ " << dt_diff << " s" << std::endl;
            std::cout << "  dt_bulk (bulk)    ≈ " << dt_bulk << " s" << std::endl;
            std::cout << "  dt_max            = " << dt_max << " s" << std::endl;

            double dt = prompt_with_default("Enter time step dt (s)", dt_max);
            if (dt > dt_max) dt = dt_max;

            if (mode == 2) {
                double noise_amplitude = prompt_with_default("Noise amplitude for isotropic init", 1e-3);
                initializeIsotropicWithNoise(h_Q, Nx, Ny, Nz, noise_amplitude);
            } else {
                initializeQTensor(h_Q, Nx, Ny, Nz, S0, mode);
            }
            cudaMemcpy(d_Q, h_Q.data(), size_Q, cudaMemcpyHostToDevice);

            // Boundary amplitude used for anchoring in single-temp mode.
            // If the bulk equilibrium is isotropic at this T, set S_shell=0 so the shell does not
            // artificially enforce nematic order above the I-N transition.
            const double S_shell_single = (S_eq > 0.0) ? S0 : 0.0;

            char user_choice = 'y';
            std::cout << "Do you want output in the console every " << printFreq << " iterations? (y/n): ";
            std::string user_choice_line;
            std::getline(std::cin, user_choice_line);
            if (!user_choice_line.empty()) user_choice = user_choice_line[0];

            double prev_F = std::numeric_limits<double>::max();
            double prev_R = std::numeric_limits<double>::quiet_NaN();
            double physical_time = 0.0;
            double radiality_threshold = 0.998;
            double min_alignment_time = 0.5 * tau_align;

            if (submode == 1) {
                std::ofstream energy_log("free_energy_vs_iteration.dat");
                energy_log << "iteration,free_energy,radiality,time\n";

                for (int iter = 0; iter < maxIterations; ++iter) {
                    physical_time += dt;
                    const bool check_now = debug_cuda_checks && (iter == 0 || (iter % printFreq) == 0);
                    computeLaplacianKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_laplacianQ, Nx, Ny, Nz, dx, dy, dz);
                    if (check_now) cuda_check_or_die("computeLaplacianKernel");
                    if (L2 != 0.0 || L3 != 0.0) computeDivergenceKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_Dcol_x, d_Dcol_y, d_Dcol_z, Nx, Ny, Nz, dx, dy, dz);
                    if (check_now && (L2 != 0.0 || L3 != 0.0)) cuda_check_or_die("computeDivergenceKernel");
                    computeChemicalPotentialKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_mu, d_laplacianQ, Nx, Ny, Nz, dx, dy, dz, params, kappa, modechoice, d_Dcol_x, d_Dcol_y, d_Dcol_z);
                    if (check_now) cuda_check_or_die("computeChemicalPotentialKernel");
                    if (L3 != 0.0) {
                        computeChemicalPotentialL3Kernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_mu, Nx, Ny, Nz, dx, dy, dz, params, d_Dcol_x, d_Dcol_y, d_Dcol_z);
                        if (check_now) cuda_check_or_die("computeChemicalPotentialL3Kernel");
                    }
                    const double shell_thickness = std::min({dx, dy, dz});
                    applyWeakAnchoringPenaltyKernel<<<numBlocks, threadsPerBlock>>>(d_mu, d_Q, Nx, Ny, Nz, d_is_shell, S_shell_single, W, shell_thickness);
                    if (check_now) cuda_check_or_die("applyWeakAnchoringPenaltyKernel");
                    updateQTensorKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_mu, Nx, Ny, Nz, dt, d_is_shell, gamma, W);
                    if (check_now) cuda_check_or_die("updateQTensorKernel");
                    applyBoundaryConditionsKernel<<<numBlocks, threadsPerBlock>>>(d_Q, Nx, Ny, Nz, d_is_shell, S_shell_single, W);
                    if (check_now) cuda_check_or_die("applyBoundaryConditionsKernel");

                    if (iter % printFreq == 0) {
                        computeRadialityKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_radiality_vals, d_count_vals, Nx, Ny, Nz);
                        if (check_now) cuda_check_or_die("computeRadialityKernel");
                        auto [bulk_sum, elastic_sum, anch_sum] = reduce_energy(params, S_shell_single);
                        double total_F = bulk_sum + elastic_sum + anch_sum;

                        std::vector<double> h_rad(num_elements);
                        std::vector<int> h_count(num_elements);
                        cudaMemcpy(h_rad.data(), d_radiality_vals, size_double, cudaMemcpyDeviceToHost);
                        cudaMemcpy(h_count.data(), d_count_vals, size_int, cudaMemcpyDeviceToHost);
                        double total_rad = 0.0;
                        int rad_count = 0;
                        for (size_t ii = 0; ii < num_elements; ++ii) {
                            total_rad += h_rad[ii];
                            rad_count += h_count[ii];
                        }
                        double avg_rad = (rad_count > 0) ? total_rad / rad_count : 0.0;

                        energy_log << iter << "," << total_F << "," << avg_rad << "," << physical_time << "\n";
                        if (user_choice == 'y' || user_choice == 'Y') {
                            std::cout << "Iter " << iter << "  t=" << physical_time << " s" << "  F=" << total_F
                                      << " (bulk=" << bulk_sum << ", el=" << elastic_sum << ", anch=" << anch_sum
                                      << ")  R̄=" << avg_rad << std::endl;
                        }

                        if (debug_dynamics) {
                            std::vector<QTensor> h_mu(num_elements);
                            cudaError_t e1 = cudaMemcpy(h_mu.data(), d_mu, size_Q, cudaMemcpyDeviceToHost);
                            cudaError_t e2 = cudaMemcpy(h_Q.data(), d_Q, size_Q, cudaMemcpyDeviceToHost);
                            if (e1 != cudaSuccess || e2 != cudaSuccess) {
                                std::cerr << "[CUDA] cudaMemcpy failed in debug_dynamics: "
                                          << cudaGetErrorString(e1 != cudaSuccess ? e1 : e2) << std::endl;
                                std::exit(1);
                            }

                            double max_mu = 0.0;
                            double max_Q = 0.0;
                            double max_dQ = 0.0;
                            for (size_t ii = 0; ii < num_elements; ++ii) {
                                max_mu = std::max(max_mu, qtensor_frobenius_norm(h_mu[ii]));
                                max_Q = std::max(max_Q, qtensor_frobenius_norm(h_Q[ii]));
                                if (have_prev_Q_debug) {
                                    QTensor dq = h_Q[ii] - prev_Q_debug[ii];
                                    max_dQ = std::max(max_dQ, qtensor_frobenius_norm(dq));
                                }
                            }
                            if (!have_prev_Q_debug) {
                                prev_Q_debug = h_Q;
                                have_prev_Q_debug = true;
                            } else {
                                prev_Q_debug = h_Q;
                            }

                            std::cout << "  [debug] max|mu|=" << max_mu
                                      << "  max|Q|=" << max_Q
                                      << "  max|ΔQ|=" << max_dQ << std::endl;
                        }

                        NematicField* d_nf;
                        cudaMalloc(&d_nf, num_elements * sizeof(NematicField));
                        computeNematicFieldKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_nf, Nx, Ny, Nz);
                        if (check_now) cuda_check_or_die("computeNematicFieldKernel");
                        std::vector<NematicField> h_nf(num_elements);
                        cudaMemcpy(h_nf.data(), d_nf, num_elements * sizeof(NematicField), cudaMemcpyDeviceToHost);
                        cudaFree(d_nf);
                        saveNematicFieldToFile(h_nf, Nx, Ny, Nz, "output/nematic_field_iter_" + std::to_string(iter) + ".dat");

                        if (iter > 100 && prev_F != 0.0) {
                            double energy_change = std::abs((total_F - prev_F) / prev_F);
                            bool energy_converged = energy_change < tolerance;
                            bool radially_aligned = avg_rad > radiality_threshold;
                            bool time_sufficient = physical_time > min_alignment_time;
                            bool radiality_converged = false;
                            if (std::isfinite(prev_R) && prev_R != 0.0) {
                                double rad_change = std::abs((avg_rad - prev_R) / prev_R);
                                radiality_converged = rad_change < RbEps;
                            }
                            if (energy_converged && radiality_converged && radially_aligned && time_sufficient) {
                                std::cout << "\n=== Convergence Achieved ===";
                                break;
                            }
                        }
                        prev_F = total_F;
                        prev_R = avg_rad;
                    }
                }
                energy_log.close();
            } else {
                std::ofstream energy_components_log("energy_components_vs_iteration.dat");
                energy_components_log << "iteration,bulk,elastic,anchoring,total,radiality,time\n";

                for (int iter = 0; iter < maxIterations; ++iter) {
                    physical_time += dt;
                    const bool check_now = debug_cuda_checks && (iter == 0 || (iter % printFreq) == 0);
                    computeLaplacianKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_laplacianQ, Nx, Ny, Nz, dx, dy, dz);
                    if (check_now) cuda_check_or_die("computeLaplacianKernel");
                    if (L2 != 0.0 || L3 != 0.0) computeDivergenceKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_Dcol_x, d_Dcol_y, d_Dcol_z, Nx, Ny, Nz, dx, dy, dz);
                    if (check_now && (L2 != 0.0 || L3 != 0.0)) cuda_check_or_die("computeDivergenceKernel");
                    computeChemicalPotentialKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_mu, d_laplacianQ, Nx, Ny, Nz, dx, dy, dz, params, kappa, modechoice, d_Dcol_x, d_Dcol_y, d_Dcol_z);
                    if (check_now) cuda_check_or_die("computeChemicalPotentialKernel");
                    if (L3 != 0.0) {
                        computeChemicalPotentialL3Kernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_mu, Nx, Ny, Nz, dx, dy, dz, params, d_Dcol_x, d_Dcol_y, d_Dcol_z);
                        if (check_now) cuda_check_or_die("computeChemicalPotentialL3Kernel");
                    }
                    const double shell_thickness = std::min({dx, dy, dz});
                    applyWeakAnchoringPenaltyKernel<<<numBlocks, threadsPerBlock>>>(d_mu, d_Q, Nx, Ny, Nz, d_is_shell, S_shell_single, W, shell_thickness);
                    if (check_now) cuda_check_or_die("applyWeakAnchoringPenaltyKernel");
                    updateQTensorKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_mu, Nx, Ny, Nz, dt, d_is_shell, gamma, W);
                    if (check_now) cuda_check_or_die("updateQTensorKernel");
                    applyBoundaryConditionsKernel<<<numBlocks, threadsPerBlock>>>(d_Q, Nx, Ny, Nz, d_is_shell, S_shell_single, W);
                    if (check_now) cuda_check_or_die("applyBoundaryConditionsKernel");

                    if (iter % printFreq == 0) {
                        computeRadialityKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_radiality_vals, d_count_vals, Nx, Ny, Nz);
                        if (check_now) cuda_check_or_die("computeRadialityKernel");
                        auto [bulk_sum, elastic_sum, anch_sum] = reduce_energy(params, S_shell_single);
                        double total_F = bulk_sum + elastic_sum + anch_sum;

                        std::vector<double> h_rad(num_elements);
                        std::vector<int> h_count(num_elements);
                        cudaMemcpy(h_rad.data(), d_radiality_vals, size_double, cudaMemcpyDeviceToHost);
                        cudaMemcpy(h_count.data(), d_count_vals, size_int, cudaMemcpyDeviceToHost);
                        double total_rad = 0.0;
                        int rad_count = 0;
                        for (size_t ii = 0; ii < num_elements; ++ii) {
                            total_rad += h_rad[ii];
                            rad_count += h_count[ii];
                        }
                        double avg_rad = (rad_count > 0) ? total_rad / rad_count : 0.0;

                        energy_components_log << iter << "," << bulk_sum << "," << elastic_sum << "," << anch_sum
                                             << "," << total_F << "," << avg_rad << "," << physical_time << "\n";
                        if (user_choice == 'y' || user_choice == 'Y') {
                            std::cout << "Iter " << iter << "  t=" << physical_time << " s" << "  F=" << total_F
                                      << " (bulk=" << bulk_sum << ", el=" << elastic_sum << ", anch=" << anch_sum
                                      << ")  R̄=" << avg_rad << std::endl;
                        }

                        if (debug_dynamics) {
                            std::vector<QTensor> h_mu(num_elements);
                            cudaError_t e1 = cudaMemcpy(h_mu.data(), d_mu, size_Q, cudaMemcpyDeviceToHost);
                            cudaError_t e2 = cudaMemcpy(h_Q.data(), d_Q, size_Q, cudaMemcpyDeviceToHost);
                            if (e1 != cudaSuccess || e2 != cudaSuccess) {
                                std::cerr << "[CUDA] cudaMemcpy failed in debug_dynamics: "
                                          << cudaGetErrorString(e1 != cudaSuccess ? e1 : e2) << std::endl;
                                std::exit(1);
                            }

                            double max_mu = 0.0;
                            double max_Q = 0.0;
                            double max_dQ = 0.0;
                            for (size_t ii = 0; ii < num_elements; ++ii) {
                                max_mu = std::max(max_mu, qtensor_frobenius_norm(h_mu[ii]));
                                max_Q = std::max(max_Q, qtensor_frobenius_norm(h_Q[ii]));
                                if (have_prev_Q_debug) {
                                    QTensor dq = h_Q[ii] - prev_Q_debug[ii];
                                    max_dQ = std::max(max_dQ, qtensor_frobenius_norm(dq));
                                }
                            }
                            if (!have_prev_Q_debug) {
                                prev_Q_debug = h_Q;
                                have_prev_Q_debug = true;
                            } else {
                                prev_Q_debug = h_Q;
                            }

                            std::cout << "  [debug] max|mu|=" << max_mu
                                      << "  max|Q|=" << max_Q
                                      << "  max|ΔQ|=" << max_dQ << std::endl;
                        }

                        NematicField* d_nf;
                        cudaMalloc(&d_nf, num_elements * sizeof(NematicField));
                        computeNematicFieldKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_nf, Nx, Ny, Nz);
                        if (check_now) cuda_check_or_die("computeNematicFieldKernel");
                        std::vector<NematicField> h_nf(num_elements);
                        cudaMemcpy(h_nf.data(), d_nf, num_elements * sizeof(NematicField), cudaMemcpyDeviceToHost);
                        cudaFree(d_nf);
                        saveNematicFieldToFile(h_nf, Nx, Ny, Nz, "output/nematic_field_iter_" + std::to_string(iter) + ".dat");

                        if (iter > 100 && prev_F != 0.0) {
                            double energy_change = std::abs((total_F - prev_F) / prev_F);
                            bool energy_converged = energy_change < tolerance;
                            bool radiality_converged = false;
                            if (std::isfinite(prev_R) && prev_R != 0.0) {
                                double rad_change = std::abs((avg_rad - prev_R) / prev_R);
                                radiality_converged = rad_change < RbEps;
                            }
                            if (energy_converged && radiality_converged && avg_rad > radiality_threshold && physical_time > min_alignment_time) {
                                std::cout << "\n=== Convergence Achieved ===";
                                break;
                            }
                        }
                        prev_F = total_F;
                        prev_R = avg_rad;
                    }
                }
                energy_components_log.close();
            }
            
            // Final Save
            cudaMemcpy(h_Q.data(), d_Q, size_Q, cudaMemcpyDeviceToHost);
            NematicField* d_nf;
            cudaMalloc(&d_nf, num_elements * sizeof(NematicField));
            computeNematicFieldKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_nf, Nx, Ny, Nz);
            std::vector<NematicField> finalNF(num_elements);
            cudaMemcpy(finalNF.data(), d_nf, num_elements * sizeof(NematicField), cudaMemcpyDeviceToHost);
            cudaFree(d_nf);
            saveNematicFieldToFile(finalNF, Nx, Ny, Nz, "nematic_field_final.dat");
            saveQTensorToFile(h_Q, Nx, Ny, Nz, "Qtensor_output_final.dat");
        }

        // Cleanup
        cudaFree(d_Q); cudaFree(d_laplacianQ); cudaFree(d_mu); cudaFree(d_is_shell);
        cudaFree(d_Dcol_x); cudaFree(d_Dcol_y); cudaFree(d_Dcol_z);
        cudaFree(d_bulk_energy); cudaFree(d_elastic_energy); cudaFree(d_anch_energy);
        cudaFree(d_radiality_vals); cudaFree(d_count_vals);

        std::cout << "\nSimulation finished." << std::endl;
        std::cout << "Would you like to run another simulation? (y/n): ";
        std::string again; std::getline(std::cin, again);
        run_again = (!again.empty() && (again[0] == 'y' || again[0] == 'Y'));

    } while (run_again);

    std::cout << "Exiting. Press enter to close." << std::endl;
    std::cin.get();
    return 0;
}