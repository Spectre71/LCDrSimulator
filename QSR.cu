#include "QSR.cuh"
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <limits>
#include <tuple>
#include <type_traits>
#include <unordered_map>

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

__host__ __device__ __forceinline__ double qtensor_frobenius_norm(const QTensor& q5) {
    FullQTensor q(q5);
    const double trQ2 = q.Qxx*q.Qxx + q.Qyy*q.Qyy + q.Qzz*q.Qzz
        + 2.0*(q.Qxy*q.Qxy + q.Qxz*q.Qxz + q.Qyz*q.Qyz);
    const double x = (trQ2 > 0.0) ? trQ2 : 0.0;
    return sqrt(x);
}

struct DropletGeometry {
    double dx = 1.0;
    double dy = 1.0;
    double dz = 1.0;
    double center_x = 0.0;
    double center_y = 0.0;
    double center_z = 0.0;
    double radius_x = 1.0;
    double radius_y = 1.0;
    double radius_z = 1.0;
    double min_spacing = 1.0;
    double min_radius = 1.0;
    double shell_half_thickness = 1.0;
    double core_radius = 1.0;
};

struct DropletGeometryPoint {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double center_radius = 0.0;
    double signed_distance = 0.0;
};

__constant__ DropletGeometry d_droplet_geometry;

__host__ __device__ __forceinline__ QTensor make_uniaxial_qtensor(double S, double nx, double ny, double nz) {
    return {
        S * (nx * nx - 1.0 / 3.0),
        S * (nx * ny),
        S * (nx * nz),
        S * (ny * ny - 1.0 / 3.0),
        S * (ny * nz)
    };
}

__host__ __device__ inline DropletGeometryPoint sampleDropletGeometry(const DropletGeometry& geom, int i, int j, int k) {
    DropletGeometryPoint point;
    point.x = i * geom.dx - geom.center_x;
    point.y = j * geom.dy - geom.center_y;
    point.z = k * geom.dz - geom.center_z;
    point.center_radius = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);

    const double scaled_x = point.x / geom.radius_x;
    const double scaled_y = point.y / geom.radius_y;
    const double scaled_z = point.z / geom.radius_z;
    const double rho_sq = scaled_x * scaled_x + scaled_y * scaled_y + scaled_z * scaled_z;
    if (rho_sq <= 1e-30) {
        point.signed_distance = -geom.min_radius;
        return point;
    }

    const double rho = sqrt(rho_sq);
    const double grad_x = point.x / (geom.radius_x * geom.radius_x);
    const double grad_y = point.y / (geom.radius_y * geom.radius_y);
    const double grad_z = point.z / (geom.radius_z * geom.radius_z);
    const double grad_norm = sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z) / rho;

    if (grad_norm > 1e-30) {
        point.signed_distance = (rho - 1.0) / grad_norm;
    } else {
        point.signed_distance = point.center_radius - geom.min_radius;
    }
    return point;
}

__host__ __device__ inline bool dropletContainsPoint(const DropletGeometryPoint& point) {
    return point.signed_distance <= 0.0;
}

__host__ __device__ inline bool dropletIsInShell(const DropletGeometry& geom, const DropletGeometryPoint& point) {
    return fabs(point.signed_distance) <= geom.shell_half_thickness;
}

__host__ __device__ inline bool dropletShouldEvolve(const DropletGeometry& geom, const DropletGeometryPoint& point, double W) {
    const double outer_band = (W > 0.0) ? geom.shell_half_thickness : 0.0;
    return point.signed_distance <= outer_band;
}

__host__ __device__ inline double dropletNominalShellThickness(const DropletGeometry& geom) {
    return 2.0 * geom.shell_half_thickness;
}

__host__ __device__ inline bool getCenterDirection(const DropletGeometryPoint& point, double& nx, double& ny, double& nz) {
    if (point.center_radius <= 1e-30) {
        nx = 1.0;
        ny = 0.0;
        nz = 0.0;
        return false;
    }
    nx = point.x / point.center_radius;
    ny = point.y / point.center_radius;
    nz = point.z / point.center_radius;
    return true;
}

__host__ __device__ inline bool getSurfaceNormal(const DropletGeometry& geom, const DropletGeometryPoint& point, double& nx, double& ny, double& nz) {
    const double grad_x = point.x / (geom.radius_x * geom.radius_x);
    const double grad_y = point.y / (geom.radius_y * geom.radius_y);
    const double grad_z = point.z / (geom.radius_z * geom.radius_z);
    const double grad_norm = sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);
    if (grad_norm <= 1e-30) {
        return getCenterDirection(point, nx, ny, nz);
    }
    nx = grad_x / grad_norm;
    ny = grad_y / grad_norm;
    nz = grad_z / grad_norm;
    return true;
}

static DropletGeometry makeDropletGeometry(int Nx, int Ny, int Nz, double dx, double dy, double dz) {
    DropletGeometry geom;
    geom.dx = dx;
    geom.dy = dy;
    geom.dz = dz;
    geom.center_x = 0.5 * (static_cast<double>(Nx) - 1.0) * dx;
    geom.center_y = 0.5 * (static_cast<double>(Ny) - 1.0) * dy;
    geom.center_z = 0.5 * (static_cast<double>(Nz) - 1.0) * dz;

    const double radius_index = 0.5 * (static_cast<double>(std::min({Nx, Ny, Nz})) - 1.0);
    geom.radius_x = std::max(radius_index * dx, 0.5 * dx);
    geom.radius_y = std::max(radius_index * dy, 0.5 * dy);
    geom.radius_z = std::max(radius_index * dz, 0.5 * dz);
    geom.min_spacing = std::min({dx, dy, dz});
    geom.min_radius = std::min({geom.radius_x, geom.radius_y, geom.radius_z});
    geom.shell_half_thickness = geom.min_spacing;
    geom.core_radius = geom.min_spacing;
    return geom;
}

static double estimateEllipsoidSurfaceArea(const DropletGeometry& geom) {
    constexpr double p = 1.6075;
    const double ap = std::pow(geom.radius_x, p);
    const double bp = std::pow(geom.radius_y, p);
    const double cp = std::pow(geom.radius_z, p);
    const double mean_term = (ap * bp + ap * cp + bp * cp) / 3.0;
    if (!(mean_term > 0.0)) return std::numeric_limits<double>::quiet_NaN();
    return 4.0 * PI * std::pow(mean_term, 1.0 / p);
}

static double estimateShellThicknessFromMask(const DropletGeometry& geom, size_t shell_point_count) {
    const double nominal_shell_thickness = dropletNominalShellThickness(geom);
    if (shell_point_count == 0) return nominal_shell_thickness;

    const double shell_volume = static_cast<double>(shell_point_count) * geom.dx * geom.dy * geom.dz;
    const double surface_area = estimateEllipsoidSurfaceArea(geom);
    if (!(surface_area > 0.0)) return nominal_shell_thickness;

    const double shell_thickness = shell_volume / surface_area;
    if (!std::isfinite(shell_thickness) || !(shell_thickness > 0.0)) return nominal_shell_thickness;
    return shell_thickness;
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

    const DropletGeometryPoint point = sampleDropletGeometry(d_droplet_geometry, i, j, k);
    double nx = 1.0, ny = 0.0, nz = 0.0;
    getSurfaceNormal(d_droplet_geometry, point, nx, ny, nz);
    const QTensor Q0 = make_uniaxial_qtensor(S_shell, nx, ny, nz);
    
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

    const DropletGeometryPoint point = sampleDropletGeometry(d_droplet_geometry, i, j, k);
    double nx = 1.0, ny = 0.0, nz = 0.0;
    getSurfaceNormal(d_droplet_geometry, point, nx, ny, nz);

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

__global__ void updateQTensorKernel(QTensor* Q, const QTensor* mu, int Nx, int Ny, int Nz, double dt, const bool* is_shell, double gamma, double W, double q_norm_cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= Nx || j >= Ny || k >= Nz) return;
    int idx = k + Nz * (j + Ny * i);

    // Skip shell for strong anchoring
    if (W == 0.0 && is_shell[idx]) return;

    const DropletGeometryPoint point = sampleDropletGeometry(d_droplet_geometry, i, j, k);
    if (!dropletShouldEvolve(d_droplet_geometry, point, W)) return;

    double mobility = 1.0 / gamma;
    const QTensor& current_mu = mu[idx];
    QTensor& current_Q = Q[idx];

    current_Q.Qxx -= mobility * current_mu.Qxx * dt;
    current_Q.Qxy -= mobility * current_mu.Qxy * dt;
    current_Q.Qxz -= mobility * current_mu.Qxz * dt;
    current_Q.Qyy -= mobility * current_mu.Qyy * dt;
    current_Q.Qyz -= mobility * current_mu.Qyz * dt;

    if (q_norm_cap > 0.0) {
        const double qnorm = qtensor_frobenius_norm(current_Q);
        if (qnorm > q_norm_cap && qnorm > 0.0) {
            const double s = q_norm_cap / qnorm;
            current_Q.Qxx *= s;
            current_Q.Qxy *= s;
            current_Q.Qxz *= s;
            current_Q.Qyy *= s;
            current_Q.Qyz *= s;
        }
    }
}

__global__ void computeRhsSemiImplicitKernel(
    const QTensor* Q,
    const QTensor* mu,
    const QTensor* laplacianQ,
    QTensor* rhs,
    int Nx, int Ny, int Nz,
    double dt,
    double gamma,
    double L_stab,
    const bool* is_shell,
    double W
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= Nx || j >= Ny || k >= Nz) return;
    const int idx = k + Nz * (j + Ny * i);

    // Keep boundary / shell fixed
    if (W == 0.0 && is_shell[idx]) {
        rhs[idx] = Q[idx];
        return;
    }

    const DropletGeometryPoint point = sampleDropletGeometry(d_droplet_geometry, i, j, k);
    if (!dropletShouldEvolve(d_droplet_geometry, point, W)) {
        rhs[idx] = Q[idx];
        return;
    }

    const double mobility = 1.0 / gamma;
    const double alpha = dt * mobility * L_stab;

    const QTensor q = Q[idx];
    const QTensor h = mu[idx];
    const QTensor lap = laplacianQ[idx];

    // For mu = mu_bulk + mu_aniso - L_stab*Laplace(Q),
    // we want (I - alpha*Laplace) Q^{n+1} = Q^n - dt*m*(mu_bulk + mu_aniso)
    // where alpha = dt*m*L_stab.
    // Since (mu_bulk + mu_aniso) = mu + L_stab*Laplace(Q),
    // RHS = Q^n - dt*m*mu(Q^n) - alpha*Laplace(Q^n).
    QTensor out;
    out.Qxx = q.Qxx - (dt * mobility) * h.Qxx - alpha * lap.Qxx;
    out.Qxy = q.Qxy - (dt * mobility) * h.Qxy - alpha * lap.Qxy;
    out.Qxz = q.Qxz - (dt * mobility) * h.Qxz - alpha * lap.Qxz;
    out.Qyy = q.Qyy - (dt * mobility) * h.Qyy - alpha * lap.Qyy;
    out.Qyz = q.Qyz - (dt * mobility) * h.Qyz - alpha * lap.Qyz;
    rhs[idx] = out;
}

__global__ void jacobiHelmholtzStepKernel(
    const QTensor* rhs,
    const QTensor* Q_old,
    QTensor* Q_new,
    int Nx, int Ny, int Nz,
    double alpha,
    double dx, double dy, double dz,
    const bool* is_shell,
    double W
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= Nx || j >= Ny || k >= Nz) return;
    const int idx = k + Nz * (j + Ny * i);

    // Keep boundaries fixed for simplicity and stability.
    if (i == 0 || j == 0 || k == 0 || i == Nx - 1 || j == Ny - 1 || k == Nz - 1) {
        Q_new[idx] = Q_old[idx];
        return;
    }
    if (W == 0.0 && is_shell[idx]) {
        Q_new[idx] = Q_old[idx];
        return;
    }

    const DropletGeometryPoint point = sampleDropletGeometry(d_droplet_geometry, i, j, k);
    if (!dropletShouldEvolve(d_droplet_geometry, point, W)) {
        Q_new[idx] = Q_old[idx];
        return;
    }

    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);

    const double denom = 1.0 + 2.0 * alpha * (inv_dx2 + inv_dy2 + inv_dz2);
    if (!(denom > 0.0)) {
        Q_new[idx] = Q_old[idx];
        return;
    }

    const int sx = Ny * Nz;
    const int sy = Nz;
    const int sz = 1;

    const QTensor qxp = Q_old[idx + sx];
    const QTensor qxm = Q_old[idx - sx];
    const QTensor qyp = Q_old[idx + sy];
    const QTensor qym = Q_old[idx - sy];
    const QTensor qzp = Q_old[idx + sz];
    const QTensor qzm = Q_old[idx - sz];

    const QTensor b = rhs[idx];

    QTensor out;
    out.Qxx = (b.Qxx + alpha * (inv_dx2 * (qxp.Qxx + qxm.Qxx) + inv_dy2 * (qyp.Qxx + qym.Qxx) + inv_dz2 * (qzp.Qxx + qzm.Qxx))) / denom;
    out.Qxy = (b.Qxy + alpha * (inv_dx2 * (qxp.Qxy + qxm.Qxy) + inv_dy2 * (qyp.Qxy + qym.Qxy) + inv_dz2 * (qzp.Qxy + qzm.Qxy))) / denom;
    out.Qxz = (b.Qxz + alpha * (inv_dx2 * (qxp.Qxz + qxm.Qxz) + inv_dy2 * (qyp.Qxz + qym.Qxz) + inv_dz2 * (qzp.Qxz + qzm.Qxz))) / denom;
    out.Qyy = (b.Qyy + alpha * (inv_dx2 * (qxp.Qyy + qxm.Qyy) + inv_dy2 * (qyp.Qyy + qym.Qyy) + inv_dz2 * (qzp.Qyy + qzm.Qyy))) / denom;
    out.Qyz = (b.Qyz + alpha * (inv_dx2 * (qxp.Qyz + qxm.Qyz) + inv_dy2 * (qyp.Qyz + qym.Qyz) + inv_dz2 * (qzp.Qyz + qzm.Qyz))) / denom;
    Q_new[idx] = out;
}

__global__ void applyBoundaryConditionsKernel(QTensor* Q, int Nx, int Ny, int Nz, const bool* is_shell, double S_shell, double W) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= Nx || j >= Ny || k >= Nz) return;
    int idx = k + Nz * (j + Ny * i);

    if (W > 0.0) return;

    if (is_shell[idx]) {
        const DropletGeometryPoint point = sampleDropletGeometry(d_droplet_geometry, i, j, k);
        double nx = 1.0, ny = 0.0, nz = 0.0;
        getSurfaceNormal(d_droplet_geometry, point, nx, ny, nz);
        Q[idx] = make_uniaxial_qtensor(S_shell, nx, ny, nz);
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

    const DropletGeometryPoint point = sampleDropletGeometry(d_droplet_geometry, i, j, k);

    if (dropletContainsPoint(point) && point.center_radius > 2.0 * d_droplet_geometry.min_spacing) {
        NematicField nf = calculateNematicFieldDevice(Q[idx]);
        double nx_r = 1.0, ny_r = 0.0, nz_r = 0.0;
        getCenterDirection(point, nx_r, ny_r, nz_r);
        double dot_product = fabs(nf.nx * nx_r + nf.ny * ny_r + nf.nz * nz_r);
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

__device__ inline double wrap_to_pi_device(double x) {
    // Wrap to (-pi, pi]
    constexpr double kPi = 3.141592653589793238462643383279502884;
    constexpr double kTwoPi = 2.0 * kPi;
    x = x + kPi;
    x -= floor(x / kTwoPi) * kTwoPi;
    x = x - kPi;
    return x;
}

// 2D correlation-length proxy on a z-slice using local gradients of psi=2*atan2(ny,nx).
// We estimate a length scale from the mean-squared wrapped phase differences:
//   mean_sq = < (dpsi)^2 > over nearest-neighbor edges (masked by S>S_threshold)
//   xi_grad_proxy ~ 1/sqrt(mean_sq)
// This is a cheap proxy for scaling; it does not equal the FFT-based xi used in QSRvis.py.
__global__ void computeXiGradEdgesXKernel(
    const QTensor* Q,
    int Nx,
    int Ny,
    int Nz,
    int z_slice,
    double S_threshold,
    double* sum_sq,
    unsigned int* edges_used
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Nx - 1 || j >= Ny) return;
    if (z_slice < 0 || z_slice >= Nz) return;

    const int k = z_slice;
    const int id0 = k + Nz * (j + Ny * i);
    const int id1 = k + Nz * (j + Ny * (i + 1));

    NematicField nf0 = calculateNematicFieldDevice(Q[id0]);
    NematicField nf1 = calculateNematicFieldDevice(Q[id1]);
    const bool m0 = isfinite(nf0.S) && isfinite(nf0.nx) && isfinite(nf0.ny) && (nf0.S > S_threshold);
    const bool m1 = isfinite(nf1.S) && isfinite(nf1.nx) && isfinite(nf1.ny) && (nf1.S > S_threshold);
    if (!(m0 && m1)) return;

    const double p0 = 2.0 * atan2(nf0.ny, nf0.nx);
    const double p1 = 2.0 * atan2(nf1.ny, nf1.nx);
    const double d = wrap_to_pi_device(p1 - p0);

    atomicAdd(sum_sq, d * d);
    atomicAdd(edges_used, 1u);
}

__global__ void computeXiGradEdgesYKernel(
    const QTensor* Q,
    int Nx,
    int Ny,
    int Nz,
    int z_slice,
    double S_threshold,
    double* sum_sq,
    unsigned int* edges_used
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Nx || j >= Ny - 1) return;
    if (z_slice < 0 || z_slice >= Nz) return;

    const int k = z_slice;
    const int id0 = k + Nz * (j + Ny * i);
    const int id1 = k + Nz * ((j + 1) + Ny * i);

    NematicField nf0 = calculateNematicFieldDevice(Q[id0]);
    NematicField nf1 = calculateNematicFieldDevice(Q[id1]);
    const bool m0 = isfinite(nf0.S) && isfinite(nf0.nx) && isfinite(nf0.ny) && (nf0.S > S_threshold);
    const bool m1 = isfinite(nf1.S) && isfinite(nf1.nx) && isfinite(nf1.ny) && (nf1.S > S_threshold);
    if (!(m0 && m1)) return;

    const double p0 = 2.0 * atan2(nf0.ny, nf0.nx);
    const double p1 = 2.0 * atan2(nf1.ny, nf1.nx);
    const double d = wrap_to_pi_device(p1 - p0);

    atomicAdd(sum_sq, d * d);
    atomicAdd(edges_used, 1u);
}

// 2D nematic defect density proxy on a z-slice.
// Matches QSRvis.py:defect_density_2d_from_slice:
// - psi = 2*theta where theta=atan2(ny,nx)
// - winding around plaquettes, s = 0.5*w
// - count plaquettes with |s| > charge_cutoff
// Returns counts: defects_count, plaq_used_count.
__global__ void computeDefectDensity2DProxyKernel(
    const QTensor* Q,
    int Nx,
    int Ny,
    int Nz,
    int z_slice,
    double S_threshold,
    double charge_cutoff,
    unsigned int* defects_count,
    unsigned int* plaq_used_count
) {
    constexpr double kPi = 3.141592653589793238462643383279502884;
    constexpr double kInv2Pi = 1.0 / (2.0 * kPi);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= Nx - 1 || j >= Ny - 1) return;
    if (z_slice < 0 || z_slice >= Nz) return;

    auto idx3 = [&](int ii, int jj, int kk) {
        return kk + Nz * (jj + Ny * ii);
    };

    const int k = z_slice;

    // corners: (i,j), (i+1,j), (i+1,j+1), (i,j+1)
    const int id00 = idx3(i, j, k);
    const int id10 = idx3(i + 1, j, k);
    const int id11 = idx3(i + 1, j + 1, k);
    const int id01 = idx3(i, j + 1, k);

    NematicField nf00 = calculateNematicFieldDevice(Q[id00]);
    NematicField nf10 = calculateNematicFieldDevice(Q[id10]);
    NematicField nf11 = calculateNematicFieldDevice(Q[id11]);
    NematicField nf01 = calculateNematicFieldDevice(Q[id01]);

    const bool m00 = isfinite(nf00.S) && isfinite(nf00.nx) && isfinite(nf00.ny) && (nf00.S > S_threshold);
    const bool m10 = isfinite(nf10.S) && isfinite(nf10.nx) && isfinite(nf10.ny) && (nf10.S > S_threshold);
    const bool m11 = isfinite(nf11.S) && isfinite(nf11.nx) && isfinite(nf11.ny) && (nf11.S > S_threshold);
    const bool m01 = isfinite(nf01.S) && isfinite(nf01.nx) && isfinite(nf01.ny) && (nf01.S > S_threshold);

    if (!(m00 && m10 && m11 && m01)) return;

    // psi = 2*theta with theta=atan2(ny,nx)
    const double p00 = 2.0 * atan2(nf00.ny, nf00.nx);
    const double p10 = 2.0 * atan2(nf10.ny, nf10.nx);
    const double p11 = 2.0 * atan2(nf11.ny, nf11.nx);
    const double p01 = 2.0 * atan2(nf01.ny, nf01.nx);

    const double d1 = wrap_to_pi_device(p10 - p00);
    const double d2 = wrap_to_pi_device(p11 - p10);
    const double d3 = wrap_to_pi_device(p01 - p11);
    const double d4 = wrap_to_pi_device(p00 - p01);
    const double dsum = d1 + d2 + d3 + d4;

    const double w = dsum * kInv2Pi;
    const double s = 0.5 * w;

    atomicAdd(plaq_used_count, 1u);
    if (fabs(s) > charge_cutoff) {
        atomicAdd(defects_count, 1u);
    }
}

// ------------------------------------------------------------------
// Host Helper Functions
// ------------------------------------------------------------------

static inline std::string trim_copy(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

static inline std::string lower_copy(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static std::unordered_map<std::string, std::string> parse_kv_config_file(const std::string& path) {
    std::unordered_map<std::string, std::string> cfg;
    if (path.empty()) return cfg;
    std::ifstream f(path);
    if (!f) {
        std::cerr << "[Config] Could not open config file: " << path << std::endl;
        return cfg;
    }
    std::string line;
    while (std::getline(f, line)) {
        // strip comments
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim_copy(line);
        if (line.empty()) continue;

        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim_copy(line.substr(0, eq));
        std::string val = trim_copy(line.substr(eq + 1));
        if (key.empty()) continue;
        cfg[key] = val;
    }
    return cfg;
}

template<typename T>
static T parse_value_or_default(const std::string& s, T default_value) {
    if (s.empty()) return default_value;

    if constexpr (std::is_same_v<T, bool>) {
        std::string v;
        v.reserve(s.size());
        for (char c : s) v.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        v = trim_copy(v);
        if (v == "1" || v == "true" || v == "yes" || v == "y" || v == "on") return true;
        if (v == "0" || v == "false" || v == "no" || v == "n" || v == "off") return false;
        return default_value;
    } else if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
        try {
            size_t idx = 0;
            long double v = std::stold(s, &idx);
            while (idx < s.size() && std::isspace(static_cast<unsigned char>(s[idx]))) ++idx;
            if (idx != s.size()) return default_value;
            long double vr = std::llround(v);
            const long double lo = static_cast<long double>(std::numeric_limits<T>::min());
            const long double hi = static_cast<long double>(std::numeric_limits<T>::max());
            if (vr < lo || vr > hi) return default_value;
            return static_cast<T>(vr);
        } catch (...) {
            return default_value;
        }
    } else {
        std::istringstream iss(s);
        T out;
        if (!(iss >> out)) return default_value;
        return out;
    }
}

static std::string cfg_get_string(
    const std::unordered_map<std::string, std::string>& cfg,
    const std::string& key,
    const std::string& default_value
) {
    auto it = cfg.find(key);
    if (it == cfg.end()) return default_value;
    return it->second;
}

template<typename T>
static T cfg_get_value(
    const std::unordered_map<std::string, std::string>& cfg,
    const std::string& key,
    T default_value
) {
    auto it = cfg.find(key);
    if (it == cfg.end()) return default_value;
    return parse_value_or_default<T>(it->second, default_value);
}

// Forward declarations (needed because cfg_or_prompt* uses these)
template<typename T>
T prompt_with_default(const std::string& prompt, T default_value);

static bool prompt_yes_no(const std::string& prompt, bool default_yes);

template<typename T>
static T cfg_or_prompt(
    const std::unordered_map<std::string, std::string>& cfg,
    const std::string& key,
    const std::string& prompt,
    T default_value,
    bool interactive
) {
    if (!interactive) {
        return cfg_get_value<T>(cfg, key, default_value);
    }

    // interactive: let config override the displayed default (handy for tweaking)
    const T def = cfg_get_value<T>(cfg, key, default_value);
    return prompt_with_default<T>(prompt, def);
}

static bool cfg_or_prompt_yes_no(
    const std::unordered_map<std::string, std::string>& cfg,
    const std::string& key,
    const std::string& prompt,
    bool default_yes,
    bool interactive
) {
    if (!interactive) {
        return cfg_get_value<bool>(cfg, key, default_yes);
    }
    const bool def = cfg_get_value<bool>(cfg, key, default_yes);
    return prompt_yes_no(prompt, def);
}

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

enum class BoundaryOrderMode {
    Equilibrium,
    Custom
};

static BoundaryOrderMode boundary_order_mode_from_string(const std::string& raw_mode) {
    const std::string mode = lower_copy(trim_copy(raw_mode));
    if (mode == "custom" || mode == "fixed") return BoundaryOrderMode::Custom;
    return BoundaryOrderMode::Equilibrium;
}

static const char* boundary_order_mode_name(BoundaryOrderMode mode) {
    switch (mode) {
        case BoundaryOrderMode::Custom:
            return "custom";
        case BoundaryOrderMode::Equilibrium:
        default:
            return "equilibrium";
    }
}

static double equilibrium_shell_order_from_temperature(
    double a,
    double b,
    double c,
    double T,
    double T_star,
    int modechoice
) {
    double A = 0.0, B = 0.0, C = 0.0;
    bulk_ABC_from_convention(a, b, c, T, T_star, modechoice, A, B, C);
    const double S_eq = S_eq_uniaxial_from_ABC(A, B, C);
    return (S_eq > 0.0) ? S_eq : 0.0;
}

static double resolve_boundary_shell_order(
    BoundaryOrderMode mode,
    double custom_shell_order,
    double a,
    double b,
    double c,
    double T,
    double T_star,
    int modechoice
) {
    if (mode == BoundaryOrderMode::Custom) return custom_shell_order;
    return equilibrium_shell_order_from_temperature(a, b, c, T, T_star, modechoice);
}

struct DtEstimateRequest {
    double dx = 1.0;
    double dy = 1.0;
    double dz = 1.0;
    double gamma = 1.0;
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double T_star = 0.0;
    int modechoice = 1;
    double kappa = 0.0;
    double L1 = 0.0;
    double L2 = 0.0;
    double L3 = 0.0;
    double S_ref_used = std::numeric_limits<double>::quiet_NaN();
    double S0 = 0.0;
    double T_a = 0.0;
    double T_b = 0.0;
    bool use_semi_implicit = false;
    double L_stab = 0.0;
    int jacobi_iters = 0;
};

struct DtEstimateResult {
    double min_dx = 0.0;
    double S_scale = 1.0;
    double L_eff_total = 0.0;
    double D_total = 0.0;
    double dt_diff_total = std::numeric_limits<double>::infinity();
    double L_eff_explicit = 0.0;
    double dt_diff = std::numeric_limits<double>::infinity();
    double bulk_rate = 0.0;
    double dt_bulk = std::numeric_limits<double>::infinity();
    double dt_max = std::numeric_limits<double>::infinity();
    bool using_LdG_elastic = false;
    bool anisotropic_penalty_applied = false;
    bool semi_implicit_active = false;
};

static DtEstimateResult estimate_stable_dt_window(const DtEstimateRequest& req) {
    DtEstimateResult out;
    out.min_dx = std::min({req.dx, req.dy, req.dz});

    out.S_scale = std::isfinite(req.S_ref_used)
        ? std::abs(req.S_ref_used)
        : std::max({std::abs(req.S0), 0.5});
    const double S_eq_a = std::abs(equilibrium_shell_order_from_temperature(req.a, req.b, req.c, req.T_a, req.T_star, req.modechoice));
    const double S_eq_b = std::abs(equilibrium_shell_order_from_temperature(req.a, req.b, req.c, req.T_b, req.T_star, req.modechoice));
    if (std::isfinite(S_eq_a) && S_eq_a > 0.0) out.S_scale = std::max(out.S_scale, S_eq_a);
    if (std::isfinite(S_eq_b) && S_eq_b > 0.0) out.S_scale = std::max(out.S_scale, S_eq_b);
    out.S_scale = std::max(out.S_scale, 1.0);

    out.using_LdG_elastic = (req.L1 != 0.0 || req.L2 != 0.0 || req.L3 != 0.0);
    out.anisotropic_penalty_applied = (req.L2 != 0.0 || req.L3 != 0.0);

    if (out.using_LdG_elastic) {
        out.L_eff_total = std::max({
            std::abs(req.L1),
            std::abs(req.L2),
            std::abs(req.L1 + req.L2),
            std::abs(req.L1 + 2.0 * req.L2),
            std::abs(req.L3) * out.S_scale
        });
    } else {
        out.L_eff_total = std::abs(req.kappa);
    }
    if (!(out.L_eff_total > 0.0)) out.L_eff_total = 1e-12;
    out.D_total = out.L_eff_total / req.gamma;
    if (!(out.D_total > 0.0) || !std::isfinite(out.D_total)) out.D_total = 1e-12;

    out.dt_diff_total = 0.5 * (out.min_dx * out.min_dx) / (6.0 * out.D_total);
    if (out.anisotropic_penalty_applied) out.dt_diff_total *= 0.25;

    out.L_eff_explicit = out.L_eff_total;
    out.dt_diff = out.dt_diff_total;
    if (req.use_semi_implicit && req.L_stab > 0.0 && req.jacobi_iters > 0) {
        const double L_base = out.using_LdG_elastic ? req.L1 : req.kappa;
        const double L_stab_signed = (L_base >= 0.0) ? req.L_stab : -req.L_stab;
        const double L_rem = std::abs(L_base - L_stab_signed);
        if (out.using_LdG_elastic) {
            out.L_eff_explicit = std::max({
                L_rem,
                std::abs(req.L2),
                std::abs(req.L1 + req.L2),
                std::abs(req.L1 + 2.0 * req.L2),
                std::abs(req.L3) * out.S_scale
            });
        } else {
            out.L_eff_explicit = L_rem;
        }

        if (out.L_eff_explicit > 0.0) {
            const double D_explicit = out.L_eff_explicit / req.gamma;
            out.dt_diff = 0.5 * (out.min_dx * out.min_dx) / (6.0 * D_explicit);
            if (out.anisotropic_penalty_applied) out.dt_diff *= 0.25;
        } else {
            out.dt_diff = std::numeric_limits<double>::infinity();
        }
        out.semi_implicit_active = true;
    }

    auto bulk_rate_at_T = [&](double Tval) {
        double A = 0.0, B = 0.0, C = 0.0;
        bulk_ABC_from_convention(req.a, req.b, req.c, Tval, req.T_star, req.modechoice, A, B, C);
        const double S2 = out.S_scale * out.S_scale;
        return std::abs(A) + std::abs(B) * out.S_scale + std::abs(C) * S2;
    };
    out.bulk_rate = std::max(bulk_rate_at_T(req.T_a), bulk_rate_at_T(req.T_b));
    out.dt_bulk = (out.bulk_rate > 0.0)
        ? (0.02 * req.gamma / out.bulk_rate)
        : std::numeric_limits<double>::infinity();

    out.dt_max = std::min(out.dt_diff, out.dt_bulk);
    if (!(out.dt_max > 0.0) || !std::isfinite(out.dt_max)) out.dt_max = out.dt_diff;
    return out;
}

static void print_dt_estimate_summary(const DtEstimateResult& estimate) {
    std::cout << "\nMaximum stable dt estimates:\n"
              << "  dt_diff (elastic) ~= " << estimate.dt_diff << " s\n"
              << "  dt_diff_total     ~= " << estimate.dt_diff_total << " s"
              << (estimate.semi_implicit_active ? " (before semi-implicit)" : "") << "\n"
              << "  dt_bulk (bulk)    ~= " << estimate.dt_bulk << " s\n"
              << "  dt_max            = " << estimate.dt_max << " s" << std::endl;
}

struct CorrelationLengthElasticBand {
    bool using_ldg = false;
    bool used_magnitude_fallback = false;
    double base_min = std::numeric_limits<double>::quiet_NaN();
    double base_max = std::numeric_limits<double>::quiet_NaN();
    double nematic_min = std::numeric_limits<double>::quiet_NaN();
    double nematic_max = std::numeric_limits<double>::quiet_NaN();
    double l3_envelope = 0.0;
};

static void update_positive_band(double value, bool& have_band, double& band_min, double& band_max) {
    if (!std::isfinite(value) || !(value > 0.0)) return;
    if (!have_band) {
        band_min = value;
        band_max = value;
        have_band = true;
        return;
    }
    band_min = std::min(band_min, value);
    band_max = std::max(band_max, value);
}

static CorrelationLengthElasticBand estimate_correlation_length_elastic_band(
    double kappa,
    double L1,
    double L2,
    double L3,
    double S_eq
) {
    CorrelationLengthElasticBand band;
    band.using_ldg = (L1 != 0.0 || L2 != 0.0 || L3 != 0.0);

    if (!band.using_ldg) {
        const double L_iso = std::abs(kappa);
        band.base_min = L_iso;
        band.base_max = L_iso;
        band.nematic_min = L_iso;
        band.nematic_max = L_iso;
        return band;
    }

    bool have_band = false;
    double band_min = 0.0;
    double band_max = 0.0;
    update_positive_band(L1, have_band, band_min, band_max);
    update_positive_band(L1 + L2, have_band, band_min, band_max);
    update_positive_band(L1 + 2.0 * L2, have_band, band_min, band_max);

    if (!have_band) {
        const double fallback = std::max({std::abs(L1), std::abs(L1 + L2), std::abs(L1 + 2.0 * L2), 1e-12});
        band.base_min = fallback;
        band.base_max = fallback;
        band.nematic_min = fallback;
        band.nematic_max = fallback;
        band.used_magnitude_fallback = true;
        return band;
    }

    band.base_min = band_min;
    band.base_max = band_max;
    band.l3_envelope = (S_eq > 0.0 && std::isfinite(S_eq)) ? (std::abs(L3) * S_eq) : 0.0;
    band.nematic_min = band.base_min;
    band.nematic_max = band.base_max;
    if (band.l3_envelope > 0.0) {
        band.nematic_min = std::max(band.base_min - band.l3_envelope, 1e-12);
        band.nematic_max = band.base_max + band.l3_envelope;
    }
    return band;
}

// Prints an estimate of the LdG correlation length xi (core size ~ O(xi)) and optionally aborts
// if xi is under-resolved by the spatial discretization.
static bool correlation_length_guard(
    const DropletGeometry& droplet_geometry,
    double a, double b, double c,
    double T, double T_star,
    int modechoice,
    double kappa, double L1, double L2, double L3,
    bool guard_enabled,
    bool interactive,
    double min_ratio_to_pass = 2.0
) {
    const double min_d = droplet_geometry.min_spacing;
    const double max_d = std::max({droplet_geometry.dx, droplet_geometry.dy, droplet_geometry.dz});

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

    const CorrelationLengthElasticBand elastic_band = estimate_correlation_length_elastic_band(kappa, L1, L2, L3, S_eq);
    const double xi_lin_min = correlation_length_from_L_and_A(elastic_band.base_min, absA_lin);
    const double xi_lin_max = correlation_length_from_L_and_A(elastic_band.base_max, absA_lin);
    const double xi_nem_min = correlation_length_from_L_and_A(elastic_band.nematic_min, absA_nem);
    const double xi_nem_max = correlation_length_from_L_and_A(elastic_band.nematic_max, absA_nem);
    const double xi_guard = std::isfinite(xi_nem_min) ? xi_nem_min : xi_lin_min;
    const double xi_conf = std::isfinite(xi_nem_max)
        ? xi_nem_max
        : (std::isfinite(xi_lin_max) ? xi_lin_max : xi_guard);

    const double ratio_x = (std::isfinite(xi_guard) && droplet_geometry.dx > 0.0) ? (xi_guard / droplet_geometry.dx) : 0.0;
    const double ratio_y = (std::isfinite(xi_guard) && droplet_geometry.dy > 0.0) ? (xi_guard / droplet_geometry.dy) : 0.0;
    const double ratio_z = (std::isfinite(xi_guard) && droplet_geometry.dz > 0.0) ? (xi_guard / droplet_geometry.dz) : 0.0;
    const double worst_spacing_ratio = std::min({ratio_x, ratio_y, ratio_z});
    const double best_spacing_ratio = (std::isfinite(xi_guard) && min_d > 0.0) ? (xi_guard / min_d) : 0.0;
    const double radius_ratio_x = (std::isfinite(xi_conf) && xi_conf > 0.0) ? (droplet_geometry.radius_x / xi_conf) : 0.0;
    const double radius_ratio_y = (std::isfinite(xi_conf) && xi_conf > 0.0) ? (droplet_geometry.radius_y / xi_conf) : 0.0;
    const double radius_ratio_z = (std::isfinite(xi_conf) && xi_conf > 0.0) ? (droplet_geometry.radius_z / xi_conf) : 0.0;
    const double min_radius_ratio = std::min({radius_ratio_x, radius_ratio_y, radius_ratio_z});

    std::cout << "\n--- Correlation Length (xi) Estimate ---\n";
    if (elastic_band.using_ldg) {
        std::cout << "Elastic stiffness band for xi (LdG): base=[" << elastic_band.base_min << ", " << elastic_band.base_max << "] J/m";
        if (elastic_band.l3_envelope > 0.0) {
            std::cout << ", nematic=[" << elastic_band.nematic_min << ", " << elastic_band.nematic_max << "] J/m"
                      << " using +- |L3| S_eq = " << elastic_band.l3_envelope << " J/m";
        }
        std::cout << "\n";
        if (elastic_band.used_magnitude_fallback) {
            std::cout << "[xi] Elastic band fallback used because no positive LdG stiffness combination was available.\n";
        }
    } else {
        std::cout << "Elastic stiffness for xi: " << elastic_band.base_min << " J/m (kappa)\n";
    }
    std::cout << "Bulk convention = " << (modechoice == 1 ? "std" : "ravnik") << "\n";
    std::cout << "Bulk stiffness (linear) |A_lin| = |A| = |3 a (T-T*)| = " << absA_lin << " J/m^3\n";
    if (std::isfinite(fpp_S)) {
        std::cout << "Bulk curvature at S_eq: S_eq=" << S_eq << ", |f''(S_eq)|=" << absA_nem << " (J/m^3, heuristic)\n";
    } else {
        std::cout << "Bulk curvature at S_eq: unavailable (likely isotropic / above T*)\n";
    }
    if (std::isfinite(xi_lin_min)) {
        std::cout << "xi_lin band ≈ [" << xi_lin_min << ", " << xi_lin_max << "] m\n";
    }
    if (std::isfinite(xi_nem_min)) {
        std::cout << "xi_nem band ≈ [" << xi_nem_min << ", " << xi_nem_max << "] m\n";
    }
    if (std::isfinite(xi_guard)) {
        std::cout << "xi_guard = " << xi_guard
                  << " m, xi/dx = " << ratio_x
                  << ", xi/dy = " << ratio_y
                  << ", xi/dz = " << ratio_z
                  << " (worst axis ratio = " << worst_spacing_ratio << ")\n";
        std::cout << "Best-axis ratio xi/min(dx,dy,dz) = " << best_spacing_ratio
                  << ", worst-axis spacing = " << max_d << " m\n";
        std::cout << "Suggested coarsest spacing target: max(dx,dy,dz) <= xi_guard/3 => " << (xi_guard / 3.0) << " m\n";
        std::cout << "Droplet semi-axes: Rx=" << droplet_geometry.radius_x
                  << " m, Ry=" << droplet_geometry.radius_y
                  << " m, Rz=" << droplet_geometry.radius_z << " m\n";
        std::cout << "Geometry confinement ratios using xi_conf = " << xi_conf
                  << ": Rx/xi=" << radius_ratio_x
                  << ", Ry/xi=" << radius_ratio_y
                  << ", Rz/xi=" << radius_ratio_z
                  << " (min = " << min_radius_ratio << ")\n";
    } else {
        std::cout << "xi estimate unavailable (check elastic stiffness and that T is not exactly T*)\n";
    }

    if (!guard_enabled) return true;

    // Under-resolution heuristic: use the coarsest physical spacing, not the smallest one,
    // because an anisotropic grid only resolves the core as well as its worst axis.
    if (std::isfinite(xi_guard) && worst_spacing_ratio < min_ratio_to_pass) {
        std::cout << "\n[WARN] xi is under-resolved on the coarsest grid axis (worst xi/dh ≈ " << worst_spacing_ratio
                  << " < " << min_ratio_to_pass << ").\n";
        std::cout << "       Expect mesh-dependent cores, axis-biased defect structure, and more frequent escaped/uniaxial solutions.\n";
        if (!interactive) {
            std::cout << "       Non-interactive run: aborting instead of prompting.\n";
            return false;
        }
        return prompt_yes_no("Proceed anyway?", false);
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

static std::mt19937 makeHostRng(long long random_seed) {
    if (random_seed >= 0) {
        return std::mt19937(static_cast<std::mt19937::result_type>(random_seed));
    }
    return std::mt19937(std::random_device{}());
}

void initializeQTensor(std::vector<QTensor>& Q, int Nx, int Ny, int Nz, double q0, int mode, const DropletGeometry& droplet_geometry, long long random_seed) {
    Q.assign((size_t)Nx * (size_t)Ny * (size_t)Nz, {0, 0, 0, 0, 0});
    auto idx = [&](int i, int j, int k) { return (size_t)k + (size_t)Nz * ((size_t)j + (size_t)Ny * (size_t)i); };

    std::mt19937 gen = makeHostRng(random_seed);
    std::uniform_real_distribution<double> dist_theta(0.0, PI);
    std::uniform_real_distribution<double> dist_phi(0.0, 2.0 * PI);

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                const DropletGeometryPoint point = sampleDropletGeometry(droplet_geometry, i, j, k);
                const double r_min = droplet_geometry.core_radius;

                if (dropletContainsPoint(point)) {
                    double S = q0;
                    if (point.center_radius < r_min) S *= (point.center_radius / r_min);

                    QTensor& q = Q[idx(i, j, k)];
                    if (mode == 0) {
                        const double theta = dist_theta(gen);
                        const double phi = dist_phi(gen);
                        const double nx = std::sin(theta) * std::cos(phi);
                        const double ny = std::sin(theta) * std::sin(phi);
                        const double nz = std::cos(theta);
                        q = make_uniaxial_qtensor(S, nx, ny, nz);
                    } else {
                        double nx = 1.0;
                        double ny = 0.0;
                        double nz = 0.0;
                        getCenterDirection(point, nx, ny, nz);
                        q = make_uniaxial_qtensor(S, nx, ny, nz);
                    }
                }
            }
        }
    }
}

static void initializeIsotropicWithNoise(std::vector<QTensor>& Q, int Nx, int Ny, int Nz, double noise_amplitude, const DropletGeometry& droplet_geometry, long long random_seed) {
    Q.assign((size_t)Nx * (size_t)Ny * (size_t)Nz, {0, 0, 0, 0, 0});
    if (noise_amplitude <= 0.0) return;

    auto idx = [&](int i, int j, int k) { return (size_t)k + (size_t)Nz * ((size_t)j + (size_t)Ny * (size_t)i); };

    std::mt19937 gen = makeHostRng(random_seed);
    std::normal_distribution<double> dist(0.0, noise_amplitude);

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                const DropletGeometryPoint point = sampleDropletGeometry(droplet_geometry, i, j, k);
                if (dropletContainsPoint(point)) {
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

static void print_cli_help() {
    std::cout
        << "QSR_cuda options:\n"
        << "  --config <path>       Run non-interactively using key=value config file\n"
        << "  --interactive         When used with --config, still prompt (config values become defaults)\n"
        << "  --help                Show this help\n";
}

int main(int argc, char** argv) {
    std::string config_path;
    bool interactive = true;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i] ? argv[i] : "";
        if (arg == "--help" || arg == "-h") {
            print_cli_help();
            return 0;
        }
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
            interactive = false;
            continue;
        }
        if (arg == "--interactive") {
            interactive = true;
            continue;
        }
    }

    const auto cfg = parse_kv_config_file(config_path);
    const bool allow_run_again_loop = interactive; // if config is provided, default to a single run

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
        if (interactive) {
            std::cout << "Press {\033[1;33mEnter\033[0m} to use [\033[1;34mdefault\033[0m] values.\n" << std::endl;
        } else {
            std::cout << "[Config] Running non-interactively" << (config_path.empty() ? " (no config path?)" : (" with " + config_path)) << "\n";
        }

        int mode = cfg_or_prompt<int>(cfg, "init_mode", "Select initialization mode: [0] random, [1] radial, [2] isotropic+noise", 1, interactive);
        if (mode < 0) mode = 0;
        if (mode > 2) mode = 2;
        long long random_seed = cfg_or_prompt<long long>(
            cfg,
            "random_seed",
            "Random seed for stochastic initialization (-1 = random_device)",
            -1,
            interactive
        );

        std::cout << "\n--- Spatial Parameters ---" << std::endl;
        int Nx = cfg_or_prompt<int>(cfg, "Nx", "Enter grid size Nx", Nx_default, interactive);
        int Ny = cfg_or_prompt<int>(cfg, "Ny", "Enter grid size Ny", Ny_default, interactive);
        int Nz = cfg_or_prompt<int>(cfg, "Nz", "Enter grid size Nz", Nz_default, interactive);
        double dx = cfg_or_prompt<double>(cfg, "dx", "Enter dx (m)", dx_default, interactive);
        double dy = cfg_or_prompt<double>(cfg, "dy", "Enter dy (m)", dy_default, interactive);
        double dz = cfg_or_prompt<double>(cfg, "dz", "Enter dz (m)", dz_default, interactive);
        
        std::cout << "\n--- Landau-de Gennes Material Parameters ---" << std::endl;
        double a = cfg_or_prompt<double>(cfg, "a", "Enter a (J/m^3/K)", a_default, interactive);
        double b = cfg_or_prompt<double>(cfg, "b", "Enter b (J/m^3)", b_default, interactive);
        double c = cfg_or_prompt<double>(cfg, "c", "Enter c (J/m^3)", c_default, interactive);
        double T = cfg_or_prompt<double>(cfg, "T", "Enter T (K)", T_default, interactive);
        double T_star = cfg_or_prompt<double>(cfg, "T_star", "Enter T* (K) [A=0, isotropic spinodal]", T_star_default, interactive);
        int modechoice = cfg_or_prompt<int>(cfg, "bulk_modechoice", "Enter Bulk Energy convention (1=std, 2=ravnik)", modechoice_default, interactive);

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
        double S0 = cfg_or_prompt<double>(cfg, "S0", "Enter initial S0 (initial condition only)", S_eq > 0 ? S_eq : 0.5, interactive);
        
        std::cout << "\n--- Elastic and Dynamic Parameters ---" << std::endl;
        // color yellow text: \033[1;33m
        std::cout << "Generally: \033[1;33m Twist (K2) < Splay (K1) < Bend (K3)\033[0m" << std::endl;
        double kappa = cfg_or_prompt<double>(cfg, "kappa", "Enter kappa (J/m)", kappa_default, interactive);
        bool use_frank_map = cfg_or_prompt_yes_no(cfg, "use_frank_map", "Use Frank-to-LdG mapping with K1, K2, K3?", false, interactive);
        double L1 = 0.0, L2 = 0.0, L3 = 0.0;
        double S_ref_used = std::numeric_limits<double>::quiet_NaN();
        if (use_frank_map) {
            double K1 = cfg_or_prompt<double>(cfg, "K1", "Enter K1", 6.5e-12, interactive);
            double K2 = cfg_or_prompt<double>(cfg, "K2", "Enter K2", 4.0e-12, interactive);
            double K3 = cfg_or_prompt<double>(cfg, "K3", "Enter K3", 8e-12, interactive);
            // IMPORTANT: mapping depends on the amplitude convention for Q via S_ref.
            // Default to S_eq(T) computed from the selected bulk convention (std/ravnik),
            // since that keeps Frank->LdG consistent when switching conventions.
            double S_ref_default = (S_eq > 0.0) ? S_eq : ((S0 > 0.0) ? S0 : (b / (2.0 * c)));
            double S_ref = cfg_or_prompt<double>(cfg, "S_ref", "Enter S_ref for mapping (default: S_eq(T))", S_ref_default, interactive);
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
                bool proceed = cfg_or_prompt_yes_no(cfg, "proceed_unstable_elastic", "Proceed anyway?", false, interactive);
                if (!proceed) {
                    std::cout << "Aborting this run; choose different K1/K2/K3 or do not use Frank mapping." << std::endl;
                    if (!allow_run_again_loop) return 1;
                    std::cout << "Would you like to run another simulation? (y/n): ";
                    std::string again; std::getline(std::cin, again);
                    run_again = (!again.empty() && (again[0] == 'y' || again[0] == 'Y'));
                    continue;
                }
            }
        } else {
            L1 = cfg_or_prompt<double>(cfg, "L1", "Enter L1", 0.0, interactive);
            L2 = cfg_or_prompt<double>(cfg, "L2", "Enter L2", 0.0, interactive);
            L3 = cfg_or_prompt<double>(cfg, "L3", "Enter L3", 0.0, interactive);
            if (L1 != 0 || L2 != 0 || L3 != 0) kappa = 0.0;

            if ((L1 != 0.0 || L2 != 0.0) && (!(L1 > 0.0) || !((L1 + L2) > 0.0))) {
                std::cout << "\n[ERROR] Unstable elastic parameters for this model (L1=" << L1
                          << ", L2=" << L2 << ", L1+L2=" << (L1 + L2) << ").\n"
                          << "        For the implemented L1/L2 form with L3=0, require L1>0 and L1+L2>0.\n";
                bool proceed = cfg_or_prompt_yes_no(cfg, "proceed_unstable_elastic", "Proceed anyway?", false, interactive);
                if (!proceed) {
                    std::cout << "Aborting this run; choose stable L1/L2 or use kappa instead." << std::endl;
                    if (!allow_run_again_loop) return 1;
                    std::cout << "Would you like to run another simulation? (y/n): ";
                    std::string again; std::getline(std::cin, again);
                    run_again = (!again.empty() && (again[0] == 'y' || again[0] == 'Y'));
                    continue;
                }
            }
        }

        const DropletGeometry droplet_geometry = makeDropletGeometry(Nx, Ny, Nz, dx, dy, dz);

        bool xi_guard_enabled = cfg_or_prompt_yes_no(cfg, "xi_guard_enabled", "Enable correlation-length guard (abort if xi is under-resolved)?", true, interactive);
        if (!correlation_length_guard(droplet_geometry, a, b, c, T, T_star, modechoice, kappa, L1, L2, L3, xi_guard_enabled, interactive)) {
            std::cout << "\nAborting this run due to correlation-length guard." << std::endl;
            std::cout << "Tip: decrease the coarsest spacing, enlarge the smallest droplet semi-axis, or move T closer to T* to increase xi." << std::endl;
            if (!allow_run_again_loop) return 1;
            std::cout << "Would you like to run another simulation? (y/n): ";
            std::string again; std::getline(std::cin, again);
            run_again = (!again.empty() && (again[0] == 'y' || again[0] == 'Y'));
            continue;
        }

        double W = cfg_or_prompt<double>(cfg, "W", "Enter weak anchoring W (J/m^2)", 0.0, interactive);
        double gamma = cfg_or_prompt<double>(cfg, "gamma", "Enter gamma (Pa·s)", gamma_default, interactive);
        int maxIterations = cfg_or_prompt<int>(cfg, "iterations", "Enter iterations", maxIterations_default, interactive);
        int printFreq = cfg_or_prompt<int>(cfg, "print_freq", "Enter print freq", 200, interactive);
        double tolerance = cfg_or_prompt<double>(cfg, "tolerance", "Enter tolerance", 1e-2, interactive);
        // Absolute convergence threshold for the 2D defect-density proxy between consecutive samples.
        double defectDensityAbsEps = cfg_or_prompt<double>(
            cfg,
            "defect_density_abs_eps",
            "Enter 2D defect-density convergence eps (absolute Δn_def)",
            1e-4,
            interactive
        );

        // Debugging aids (off by default because they can slow down runs).
        const bool debug_cuda_checks = cfg_or_prompt_yes_no(
            cfg,
            "debug_cuda_checks",
            "Debug: enable CUDA error checks at log points? (syncs GPU; slower)",
            false
        , interactive);
        const bool debug_dynamics = cfg_or_prompt_yes_no(
            cfg,
            "debug_dynamics",
            "Debug: print max|mu| and max|ΔQ| at log points? (copies arrays; slower)",
            false
        , interactive);

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

        cudaError_t geom_copy_err = cudaMemcpyToSymbol(d_droplet_geometry, &droplet_geometry, sizeof(DropletGeometry));
        if (geom_copy_err != cudaSuccess) {
            std::cerr << "[CUDA] Failed to upload droplet geometry: " << cudaGetErrorString(geom_copy_err) << std::endl;
            return 1;
        }
        
        // Initialize Shell
        size_t shell_point_count = 0;
        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                for (int k = 0; k < Nz; ++k) {
                    const DropletGeometryPoint point = sampleDropletGeometry(droplet_geometry, i, j, k);
                    if (dropletIsInShell(droplet_geometry, point)) {
                        h_is_shell[k + Nz * (j + Ny * i)] = true;
                        ++shell_point_count;
                    }
                }
            }
        }

        const double shell_nominal_thickness = dropletNominalShellThickness(droplet_geometry);
        const double shell_surface_area_estimate = estimateEllipsoidSurfaceArea(droplet_geometry);
        const double shell_volume = static_cast<double>(shell_point_count) * dx * dy * dz;
        const double shell_effective_thickness = estimateShellThicknessFromMask(droplet_geometry, shell_point_count);

        if (W > 0.0) {
            std::cout << "\n--- Weak Anchoring Shell ---" << std::endl;
            std::cout << "Shell voxels              = " << shell_point_count << std::endl;
            std::cout << "Shell volume              = " << shell_volume << " m^3" << std::endl;
            std::cout << "Estimated surface area    = " << shell_surface_area_estimate << " m^2" << std::endl;
            std::cout << "Nominal shell thickness   = " << shell_nominal_thickness << " m" << std::endl;
            std::cout << "Effective shell thickness = " << shell_effective_thickness << " m" << std::endl;
            std::cout << "Weak anchoring scale W/δ  = " << (W / shell_effective_thickness) << " J/m^3" << std::endl;
        }

        // Device Memory
        QTensor *d_Q, *d_Q_alt, *d_laplacianQ, *d_mu, *d_rhs;
        bool *d_is_shell;
        double *d_Dcol_x, *d_Dcol_y, *d_Dcol_z;
        double *d_bulk_energy, *d_elastic_energy;
        double *d_anch_energy;
        double *d_radiality_vals;
        int *d_count_vals;
        unsigned int *d_defects_2d_count;
        unsigned int *d_defects_2d_plaq_used;
        double *d_xi_grad_sum;
        unsigned int *d_xi_grad_edges_used;

        cudaMalloc(&d_Q, size_Q);
        cudaMalloc(&d_Q_alt, size_Q);
        cudaMalloc(&d_laplacianQ, size_Q);
        cudaMalloc(&d_mu, size_Q);
        cudaMalloc(&d_rhs, size_Q);
        cudaMalloc(&d_is_shell, size_bool);
        cudaMalloc(&d_Dcol_x, size_double);
        cudaMalloc(&d_Dcol_y, size_double);
        cudaMalloc(&d_Dcol_z, size_double);
        cudaMalloc(&d_bulk_energy, size_double);
        cudaMalloc(&d_elastic_energy, size_double);
        cudaMalloc(&d_anch_energy, size_double);
        cudaMalloc(&d_radiality_vals, size_double);
        cudaMalloc(&d_count_vals, size_int);
        cudaMalloc(&d_defects_2d_count, sizeof(unsigned int));
        cudaMalloc(&d_defects_2d_plaq_used, sizeof(unsigned int));
        cudaMalloc(&d_xi_grad_sum, sizeof(double));
        cudaMalloc(&d_xi_grad_edges_used, sizeof(unsigned int));

        // Copy static data
        cudaMemcpy(d_is_shell, h_is_shell.data(), size_bool, cudaMemcpyHostToDevice);

        // CUDA Grid
        dim3 threadsPerBlock(8, 8, 8);
        dim3 numBlocks((Nx + 7) / 8, (Ny + 7) / 8, (Nz + 7) / 8);
        dim3 threads2D(16, 16, 1);
        dim3 blocks2D((Nx - 1 + threads2D.x - 1) / threads2D.x, (Ny - 1 + threads2D.y - 1) / threads2D.y, 1);
        dim3 blocksX((Nx - 1 + threads2D.x - 1) / threads2D.x, (Ny + threads2D.y - 1) / threads2D.y, 1);
        dim3 blocksY((Nx + threads2D.x - 1) / threads2D.x, (Ny - 1 + threads2D.y - 1) / threads2D.y, 1);

        std::cout << "\nSelect simulation mode: [1] Single Temp, [2] Temp Range, [3] Quench (time-dependent T): ";
        int sim_mode = cfg_or_prompt<int>(cfg, "sim_mode", "", 1, interactive);

        const std::string boundary_order_mode_raw = cfg_or_prompt<std::string>(
            cfg,
            "boundary_order_mode",
            "Boundary order policy [equilibrium/custom]",
            "equilibrium",
            interactive
        );
        const std::string boundary_order_mode_norm = lower_copy(trim_copy(boundary_order_mode_raw));
        const BoundaryOrderMode boundary_order_mode = boundary_order_mode_from_string(boundary_order_mode_norm);
        if (!boundary_order_mode_norm.empty() &&
            boundary_order_mode_norm != "equilibrium" &&
            boundary_order_mode_norm != "custom" &&
            boundary_order_mode_norm != "fixed") {
            std::cout << "[Boundary] Unrecognized boundary_order_mode='" << boundary_order_mode_raw
                      << "'; using equilibrium." << std::endl;
        }

        double boundary_S_custom = 0.0;
        if (boundary_order_mode == BoundaryOrderMode::Custom) {
            const double boundary_S_default = equilibrium_shell_order_from_temperature(a, b, c, T, T_star, modechoice);
            boundary_S_custom = cfg_or_prompt<double>(
                cfg,
                "boundary_S",
                "Boundary shell order S_shell for custom policy",
                boundary_S_default,
                interactive
            );
            if (!std::isfinite(boundary_S_custom)) boundary_S_custom = boundary_S_default;
        }

        auto compute_S_shell = [&](double T_shell) {
            return resolve_boundary_shell_order(
                boundary_order_mode,
                boundary_S_custom,
                a,
                b,
                c,
                T_shell,
                T_star,
                modechoice
            );
        };

        if (boundary_order_mode == BoundaryOrderMode::Equilibrium) {
            std::cout << "[Boundary] Shell order policy: equilibrium S_shell = max(S_eq(T), 0)." << std::endl;
        } else {
            std::cout << "[Boundary] Shell order policy: custom S_shell = " << boundary_S_custom
                      << " (independent of init S0 and temperature)." << std::endl;
        }

        auto compute_avg_S_droplet = [&](const std::vector<NematicField>& nf) -> double {
            double total_S = 0.0;
            int count = 0;
            // Exclude a thin layer near the shell to avoid anchoring-dominated averages.
            const double shell_exclude = 2.0 * droplet_geometry.min_spacing;
            for (int i = 0; i < Nx; ++i) {
                for (int j = 0; j < Ny; ++j) {
                    for (int k = 0; k < Nz; ++k) {
                        const DropletGeometryPoint point = sampleDropletGeometry(droplet_geometry, i, j, k);
                        if (point.signed_distance < -shell_exclude) {
                            size_t id = (size_t)k + (size_t)Nz * ((size_t)j + (size_t)Ny * (size_t)i);
                            total_S += nf[id].S;
                            ++count;
                        }
                    }
                }
            }
            return (count > 0) ? (total_S / count) : 0.0;
        };

        auto compute_max_S_droplet = [&](const std::vector<NematicField>& nf) -> double {
            double max_S = 0.0;
            const double shell_exclude = 2.0 * droplet_geometry.min_spacing;
            for (int i = 0; i < Nx; ++i) {
                for (int j = 0; j < Ny; ++j) {
                    for (int k = 0; k < Nz; ++k) {
                        const DropletGeometryPoint point = sampleDropletGeometry(droplet_geometry, i, j, k);
                        if (point.signed_distance < -shell_exclude) {
                            const size_t id = (size_t)k + (size_t)Nz * ((size_t)j + (size_t)Ny * (size_t)i);
                            const double s = nf[id].S;
                            if (std::isfinite(s)) max_S = std::max(max_S, s);
                        }
                    }
                }
            }
            return max_S;
        };

        auto reduce_energy = [&](DimensionalParams p, double S_shell) -> std::tuple<double, double, double> {
            computeEnergyKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_bulk_energy, d_elastic_energy, Nx, Ny, Nz, dx, dy, dz, p, kappa, modechoice);
            computeAnchoringEnergyKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_is_shell, d_anch_energy, Nx, Ny, Nz, dx, dy, dz, S_shell, p.W, shell_effective_thickness);

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

                    auto compute_defect_density_2d = [&](int z_slice, double S_threshold, double charge_cutoff, double& density, unsigned int& plaq_used) {
                        cudaMemset(d_defects_2d_count, 0, sizeof(unsigned int));
                        cudaMemset(d_defects_2d_plaq_used, 0, sizeof(unsigned int));
                        computeDefectDensity2DProxyKernel<<<blocks2D, threads2D>>>(
                            d_Q,
                            Nx,
                            Ny,
                            Nz,
                            z_slice,
                            S_threshold,
                            charge_cutoff,
                            d_defects_2d_count,
                            d_defects_2d_plaq_used
                        );
                        if (debug_cuda_checks) cuda_check_or_die("computeDefectDensity2DProxyKernel");

                        unsigned int defect_count = 0u;
                        plaq_used = 0u;
                        cudaMemcpy(&defect_count, d_defects_2d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                        cudaMemcpy(&plaq_used, d_defects_2d_plaq_used, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                        density = (plaq_used > 0u) ? (double(defect_count) / double(plaq_used)) : std::numeric_limits<double>::quiet_NaN();
                    };

                    auto defect_density_converged = [&](double prev_density, double current_density, unsigned int plaq_used) {
                        if (!(plaq_used > 0u)) return false;
                        if (!std::isfinite(prev_density) || !std::isfinite(current_density)) return false;
                        return std::abs(current_density - prev_density) < defectDensityAbsEps;
                    };

                    auto compute_xi_grad_proxy_2d = [&](int z_slice, double S_threshold, double& xi_grad_proxy, unsigned int& xi_grad_edges_used) {
                        xi_grad_proxy = std::numeric_limits<double>::quiet_NaN();
                        xi_grad_edges_used = 0u;

                        cudaMemset(d_xi_grad_sum, 0, sizeof(double));
                        cudaMemset(d_xi_grad_edges_used, 0, sizeof(unsigned int));
                        computeXiGradEdgesXKernel<<<blocksX, threads2D>>>(
                            d_Q,
                            Nx,
                            Ny,
                            Nz,
                            z_slice,
                            S_threshold,
                            d_xi_grad_sum,
                            d_xi_grad_edges_used
                        );
                        computeXiGradEdgesYKernel<<<blocksY, threads2D>>>(
                            d_Q,
                            Nx,
                            Ny,
                            Nz,
                            z_slice,
                            S_threshold,
                            d_xi_grad_sum,
                            d_xi_grad_edges_used
                        );
                        if (debug_cuda_checks) cuda_check_or_die("computeXiGradEdges kernels");

                        double sum_sq = 0.0;
                        cudaMemcpy(&sum_sq, d_xi_grad_sum, sizeof(double), cudaMemcpyDeviceToHost);
                        cudaMemcpy(&xi_grad_edges_used, d_xi_grad_edges_used, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                        if (xi_grad_edges_used > 0u) {
                            const double mean_sq = sum_sq / double(xi_grad_edges_used);
                            const double eps = 1e-30;
                            const double inv = (mean_sq > 0.0) ? (1.0 / sqrt(mean_sq + eps)) : (1.0 / sqrt(eps));
                            const double xi_cap = 0.5 * double(std::min(Nx, Ny));
                            xi_grad_proxy = std::min(inv, xi_cap);
                        }
                    };

        if (sim_mode == 3) {
            std::string out_dir_name;
            if (!interactive) {
                out_dir_name = cfg_get_string(cfg, "out_dir", "output_quench");
            } else {
                std::cout << "Output directory for quench results [default: output_quench]: ";
                std::getline(std::cin, out_dir_name);
                if (out_dir_name.empty()) out_dir_name = cfg_get_string(cfg, "out_dir", "output_quench");
            }

            fs::path out_dir = out_dir_name;
            if (fs::exists(out_dir)) {
                bool overwrite = cfg_or_prompt_yes_no(
                    cfg,
                    "overwrite_out_dir",
                    ("Directory '" + out_dir.string() + "' exists. Delete it and start fresh?").c_str(),
                    true,
                    interactive
                );
                if (overwrite) {
                    fs::remove_all(out_dir);
                }
            }
            fs::create_directories(out_dir);

            int protocol = cfg_or_prompt<int>(cfg, "protocol", "Quench protocol (1=step, 2=ramp)", 2, interactive);
            double T_high = cfg_or_prompt<double>(cfg, "T_high", "T_high (K)", T, interactive);
            double T_low = cfg_or_prompt<double>(cfg, "T_low", "T_low (K)", T_high - 5.0, interactive);
            int pre_equil_iters = cfg_or_prompt<int>(cfg, "pre_equil_iters", "Pre-equilibration iterations at T_high", 0, interactive);
            int ramp_iters = 0;
            if (protocol == 2) {
                ramp_iters = cfg_or_prompt<int>(cfg, "ramp_iters", "Ramp iterations (>=1)", 1000, interactive);
                if (ramp_iters < 1) ramp_iters = 1;
            }
            int total_iters = cfg_or_prompt<int>(cfg, "total_iters", "Total iterations", maxIterations, interactive);
            if (total_iters < 1) total_iters = 1;
            int logFreq = cfg_or_prompt<int>(cfg, "logFreq", "Log/print freq", printFreq, interactive);
            if (logFreq < 1) logFreq = 1;

            std::cout << "\nSnapshot intent: [0] Off (final only), [1] GIF (regular snapshots), [2] KZ (snapshots around Tc)" << std::endl;
            int snapshot_mode = cfg_or_prompt<int>(cfg, "snapshot_mode", "Select snapshot mode", 2, interactive);
            if (snapshot_mode < 0) snapshot_mode = 0;
            if (snapshot_mode > 2) snapshot_mode = 2;

            // GIF mode: save every snapshotFreq iterations.
            int snapshotFreq = 0;
            if (snapshot_mode == 1) {
                snapshotFreq = cfg_or_prompt<int>(cfg, "snapshotFreq", "GIF snapshot frequency in iterations (0=off)", 10000, interactive);
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
            bool save_qtensor_snapshots = false;
            if (snapshot_mode == 2) {
                std::cout << "\n[KZ MODE] Snapshots will be saved around Tc so you can measure at a chosen offset." << std::endl;
                std::cout << "         Use a relatively dense snapshot frequency for flexibility (e.g. 1k–10k iters)." << std::endl;
                Tc_KZ = cfg_or_prompt<double>(cfg, "Tc_KZ", "Tc for KZ snapshots (K)", 310.2, interactive);
                if (protocol == 2) {
                    Tc_window_K = cfg_or_prompt<double>(cfg, "Tc_window_K", "Temperature window half-width around Tc (K)", 0.5, interactive);
                    if (Tc_window_K < 0.0) Tc_window_K = -Tc_window_K;
                    kzSnapshotFreq = cfg_or_prompt<int>(cfg, "kzSnapshotFreq", "KZ snapshot frequency in iterations", 1000, interactive);
                    if (kzSnapshotFreq < 1) kzSnapshotFreq = 1;
                } else {
                    std::cout << "\n[NOTE] Step quench has no Tc ramp. We'll snapshot for a fixed number of iterations after the step." << std::endl;
                    kzItersAfterStep = cfg_or_prompt<int>(cfg, "kzItersAfterStep", "Iterations after the step to record KZ snapshots", 200000, interactive);
                    if (kzItersAfterStep < 1) kzItersAfterStep = 1;
                    kzSnapshotFreq = cfg_or_prompt<int>(cfg, "kzSnapshotFreq", "KZ snapshot frequency in iterations", 1000, interactive);
                    if (kzSnapshotFreq < 1) kzSnapshotFreq = 1;
                }

                kz_stop_early = cfg_or_prompt_yes_no(
                    cfg,
                    "kz_stop_early",
                    "In KZ mode, stop simulation once recording window is finished? (saves final state at stop)",
                    true,
                    interactive
                );
                if (kz_stop_early) {
                    kzExtraIters = cfg_or_prompt<int>(cfg, "kzExtraIters", "Extra iterations after leaving window (for coarsening/offset flexibility)", 0, interactive);
                    if (kzExtraIters < 0) kzExtraIters = 0;
                }
                save_qtensor_snapshots = cfg_or_prompt_yes_no(
                    cfg,
                    "save_qtensor_snapshots",
                    "Also save Qtensor_output_iter_*.dat snapshots in KZ mode? (needed for biaxial-core analysis)",
                    false,
                    interactive
                );
            }

            double noise_amplitude = 0.0;
            if (mode == 2) {
                noise_amplitude = cfg_or_prompt<double>(cfg, "noise_amplitude", "Noise amplitude for isotropic init", 1e-3, interactive);
                initializeIsotropicWithNoise(h_Q, Nx, Ny, Nz, noise_amplitude, droplet_geometry, random_seed);
            } else {
                initializeQTensor(h_Q, Nx, Ny, Nz, S0, mode, droplet_geometry, random_seed);
            }
            cudaMemcpy(d_Q, h_Q.data(), size_Q, cudaMemcpyHostToDevice);

            // Semi-implicit stabilization for stiff elastic Laplacian term:
            // backward-Euler on +L_stab*Laplace(Q)/gamma, explicit on everything else.
            // NOTE: this only stabilizes the isotropic Laplacian-like piece; L2/L3 contributions remain explicit.
            const bool use_semi_implicit = cfg_or_prompt_yes_no(
                cfg,
                "use_semi_implicit",
                "Quench: use semi-implicit elastic stabilizer? (Helmholtz solve via Jacobi)",
                true,
                interactive
            );

            // Choose the coefficient that multiplies the isotropic Laplacian in the chemical potential.
            // - In LdG (L1/L2/L3) mode, that's L1.
            // - In one-constant mode, that's kappa.
            const bool using_LdG_elastic = (L1 != 0.0 || L2 != 0.0 || L3 != 0.0);
            const double L_base = using_LdG_elastic ? L1 : kappa;
            const double L_base_abs = std::abs(L_base);

            double L_stab = 0.0;
            int jacobi_iters = 0;
            if (use_semi_implicit) {
                const double L_stab_default = (L_base_abs > 0.0) ? L_base_abs : 0.0;
                L_stab = cfg_or_prompt<double>(cfg, "L_stab", "Semi-implicit: L_stab (use |L1| or |kappa|)", L_stab_default, interactive);
                if (!std::isfinite(L_stab) || L_stab < 0.0) L_stab = 0.0;
                if (L_base_abs > 0.0 && L_stab > L_base_abs) {
                    std::cout << "[SemiImplicit] Clamping L_stab from " << L_stab << " to |L_base|=" << L_base_abs
                              << " to avoid anti-diffusive explicit remainder." << std::endl;
                    L_stab = L_base_abs;
                }
                jacobi_iters = cfg_or_prompt<int>(cfg, "jacobi_iters", "Semi-implicit: Jacobi iterations per step", 25, interactive);
                if (jacobi_iters < 0) jacobi_iters = 0;
            }

            DtEstimateRequest dt_request;
            dt_request.dx = dx;
            dt_request.dy = dy;
            dt_request.dz = dz;
            dt_request.gamma = gamma;
            dt_request.a = a;
            dt_request.b = b;
            dt_request.c = c;
            dt_request.T_star = T_star;
            dt_request.modechoice = modechoice;
            dt_request.kappa = kappa;
            dt_request.L1 = L1;
            dt_request.L2 = L2;
            dt_request.L3 = L3;
            dt_request.S_ref_used = S_ref_used;
            dt_request.S0 = S0;
            dt_request.T_a = T_high;
            dt_request.T_b = T_low;
            dt_request.use_semi_implicit = use_semi_implicit;
            dt_request.L_stab = L_stab;
            dt_request.jacobi_iters = jacobi_iters;
            const DtEstimateResult dt_estimate = estimate_stable_dt_window(dt_request);
            print_dt_estimate_summary(dt_estimate);

            double dt = cfg_or_prompt<double>(cfg, "dt", "Enter time step dt (s)", dt_estimate.dt_max, interactive);
            if (dt > dt_estimate.dt_max) dt = dt_estimate.dt_max;

            // Optional numerical guards to detect runaway-S / Nyquist checkerboard.
            // IMPORTANT: quench/KZ runs keep dt fixed because the temperature protocol is iteration-driven.
            const bool enable_instability_guard = cfg_or_prompt_yes_no(
                cfg,
                "enable_instability_guard",
                "Enable instability guard? (abort if S blows up or Nyquist checkerboard grows)",
                true,
                interactive
            );
            const auto adaptive_dt_cfg_it = cfg.find("enable_adaptive_dt");
            if (adaptive_dt_cfg_it != cfg.end()) {
                std::cout << "[Protocol] Ignoring legacy enable_adaptive_dt=" << adaptive_dt_cfg_it->second
                          << " in quench mode. Kibble-Zurek runs use fixed dt." << std::endl;
            }
            std::cout << "[Protocol] Quench/KZ runs enforce fixed dt = " << dt
                      << " s. Instability handling may abort the run, but it will not change dt." << std::endl;
            const double S_abort = enable_instability_guard
                ? cfg_or_prompt<double>(cfg, "S_abort", "Guard: abort if max S exceeds", 2.0, interactive)
                : 0.0;
            const double checker_rel_abort = enable_instability_guard
                ? cfg_or_prompt<double>(cfg, "checker_rel_abort", "Guard: abort if checkerboard rel-amplitude exceeds", 0.10, interactive)
                : 0.0;

            const bool enable_q_limiter = cfg_or_prompt_yes_no(
                cfg,
                "enable_q_limiter",
                "Optional: clamp |Q| (caps S) to prevent blow-up? (numerical limiter)",
                false,
                interactive
            );
            const double S_cap = enable_q_limiter
                ? cfg_or_prompt<double>(cfg, "S_cap", "Limiter: S_cap (approx physical max)", 1.2, interactive)
                : 0.0;
            const double q_norm_cap = (enable_q_limiter && S_cap > 0.0) ? (S_cap * std::sqrt(2.0 / 3.0)) : 0.0;

            // Optional convergence/early-stop guard (useful when you only care about final aligned state).
            // For a quench/ramp, we only allow early-stop once the protocol has reached the final temperature.
            // Criterion: relative energy change + absolute change in a topology-aware 2D defect proxy.
            // Radiality remains a diagnostic observable only.
            const bool enable_early_stop = cfg_or_prompt_yes_no(
                cfg,
                "enable_early_stop",
                "Enable early-stop once converged at final T? (rel. ΔF/F + abs. Δdefect2D)",
                false,
                interactive
            );
            const double radiality_threshold = enable_early_stop
                ? cfg_or_prompt<double>(cfg, "radiality_threshold", "Diagnostic-only radiality threshold (Rbar, 0=disable)", 0.998, interactive)
                : 0.0;

            const bool console_output = cfg_or_prompt_yes_no(
                cfg,
                "console_output",
                ("Do you want output in the console every " + std::to_string(logFreq) + " iterations?").c_str(),
                true,
                interactive
            );

            const char user_choice = console_output ? 'y' : 'n';

            const bool log_defects_2d = cfg_or_prompt_yes_no(
                cfg,
                "log_defects_2d",
                "Quench: log 2D KZ defect proxy to quench_log? (mid-plane winding)",
                true,
                interactive
            );
            const bool track_defects_2d = log_defects_2d || enable_early_stop;
            int defects_z_slice = Nz / 2;
            double defects_S_threshold = 0.1;
            double defects_charge_cutoff = 0.25;
            if (track_defects_2d) {
                defects_z_slice = cfg_or_prompt<int>(cfg, "defects_z_slice", "Quench: z_slice for 2D proxy (0..Nz-1)", defects_z_slice, interactive);
                if (defects_z_slice < 0) defects_z_slice = 0;
                if (defects_z_slice >= Nz) defects_z_slice = Nz - 1;
                defects_S_threshold = cfg_or_prompt<double>(cfg, "defects_S_threshold", "Quench: S_threshold for 2D proxy", defects_S_threshold, interactive);
                defects_charge_cutoff = cfg_or_prompt<double>(cfg, "defects_charge_cutoff", "Quench: charge_cutoff |s|>", defects_charge_cutoff, interactive);
            }
            if (enable_early_stop) {
                std::cout << "[Convergence] Quench early-stop uses energy + 2D defect proxy stability; radiality is diagnostic only." << std::endl;
            }

            const bool log_xi_grad_2d = cfg_or_prompt_yes_no(
                cfg,
                "log_xi_grad_2d",
                "Quench: log 2D xi gradient proxy to quench_log? (cheap, snapshot-free)",
                true,
                interactive
            );
            int xi_z_slice = defects_z_slice;
            double xi_S_threshold = defects_S_threshold;
            if (log_xi_grad_2d) {
                xi_z_slice = cfg_or_prompt<int>(cfg, "xi_z_slice", "Quench: z_slice for xi proxy (0..Nz-1)", xi_z_slice, interactive);
                if (xi_z_slice < 0) xi_z_slice = 0;
                if (xi_z_slice >= Nz) xi_z_slice = Nz - 1;
                xi_S_threshold = cfg_or_prompt<double>(cfg, "xi_S_threshold", "Quench: S_threshold for xi proxy", xi_S_threshold, interactive);
            }

            std::ofstream quench_log((out_dir / "quench_log.dat").string());
            quench_log << "iteration,time_s,dt_s,T_K,bulk,elastic,anchoring,total,radiality,avg_S,max_S,checker_rel,defect_density_per_plaquette,defect_plaquettes_used,xi_grad_proxy,xi_grad_edges_used\n";

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

            if (boundary_order_mode == BoundaryOrderMode::Equilibrium) {
                std::cout << "[Boundary] Quench shell order follows S_eq(T): S_shell(T_high)=" << compute_S_shell(T_high)
                          << ", S_shell(T_low)=" << compute_S_shell(T_low) << std::endl;
            } else {
                std::cout << "[Boundary] Quench shell order fixed at S_shell=" << boundary_S_custom << std::endl;
            }

            double physical_time = 0.0;
            double prev_F_for_stop = std::numeric_limits<double>::quiet_NaN();
            double prev_defect_for_stop = std::numeric_limits<double>::quiet_NaN();
            bool kz_entered_window = false;
            int kz_exit_iter = -1;
            bool quench_aborted_due_to_instability = false;
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
                applyWeakAnchoringPenaltyKernel<<<numBlocks, threadsPerBlock>>>(d_mu, d_Q, Nx, Ny, Nz, d_is_shell, S_shell, W, shell_effective_thickness);

                if (use_semi_implicit && L_stab > 0.0 && jacobi_iters > 0) {
                    computeRhsSemiImplicitKernel<<<numBlocks, threadsPerBlock>>>(
                        d_Q, d_mu, d_laplacianQ, d_rhs,
                        Nx, Ny, Nz,
                        dt, gamma, L_stab,
                        d_is_shell, W
                    );

                    const double alpha = (dt / gamma) * L_stab;
                    QTensor* q_old = d_Q;
                    QTensor* q_new = d_Q_alt;
                    for (int it = 0; it < jacobi_iters; ++it) {
                        jacobiHelmholtzStepKernel<<<numBlocks, threadsPerBlock>>>(
                            d_rhs,
                            q_old,
                            q_new,
                            Nx, Ny, Nz,
                            alpha,
                            dx, dy, dz,
                            d_is_shell,
                            W
                        );
                        QTensor* tmp = q_old;
                        q_old = q_new;
                        q_new = tmp;
                    }
                    // Keep d_Q as the latest solution buffer.
                    d_Q = q_old;
                    d_Q_alt = q_new;
                } else {
                    updateQTensorKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_mu, Nx, Ny, Nz, dt, d_is_shell, gamma, W, q_norm_cap);
                }
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

                    double defect_density_2d = std::numeric_limits<double>::quiet_NaN();
                    unsigned int defect_plaq_used = 0u;
                    if (do_log && track_defects_2d) {
                        compute_defect_density_2d(defects_z_slice, defects_S_threshold, defects_charge_cutoff, defect_density_2d, defect_plaq_used);
                    }

                    double xi_grad_proxy = std::numeric_limits<double>::quiet_NaN();
                    unsigned int xi_grad_edges_used = 0u;
                    if (do_log && log_xi_grad_2d) {
                        compute_xi_grad_proxy_2d(xi_z_slice, xi_S_threshold, xi_grad_proxy, xi_grad_edges_used);
                    }

                    NematicField* d_nf;
                    cudaMalloc(&d_nf, num_elements * sizeof(NematicField));
                    computeNematicFieldKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_nf, Nx, Ny, Nz);
                    std::vector<NematicField> h_nf(num_elements);
                    cudaMemcpy(h_nf.data(), d_nf, num_elements * sizeof(NematicField), cudaMemcpyDeviceToHost);
                    cudaFree(d_nf);
                    double avg_S = compute_avg_S_droplet(h_nf);

                    // Diagnostics for numerical instability:
                    // - max_S inside droplet interior
                    // - checkerboard (Nyquist) relative amplitude on a representative z-slice
                    double max_S = 0.0;
                    double checker_rel = std::numeric_limits<double>::quiet_NaN();
                    {
                        const double shell_exclude = 2.0 * droplet_geometry.min_spacing;

                        // max_S in interior
                        for (int i = 0; i < Nx; ++i) {
                            for (int j = 0; j < Ny; ++j) {
                                for (int k = 0; k < Nz; ++k) {
                                    const DropletGeometryPoint point = sampleDropletGeometry(droplet_geometry, i, j, k);
                                    if (point.signed_distance < -shell_exclude) {
                                        const size_t id = (size_t)k + (size_t)Nz * ((size_t)j + (size_t)Ny * (size_t)i);
                                        const double s = h_nf[id].S;
                                        if (std::isfinite(s)) max_S = std::max(max_S, s);
                                    }
                                }
                            }
                        }

                        // checkerboard (Nyquist) metric on a chosen z slice.
                        // IMPORTANT: early in the quench, only a small fraction of points have S > S_thr,
                        // which can make even/odd means noisy and trigger false positives. We therefore
                        // only evaluate this once the droplet has meaningfully ordered.
                        const double maxS_for_checker = 0.5;
                        if (max_S >= maxS_for_checker) {
                            const int z_check = defects_z_slice;
                            double even_sum = 0.0, odd_sum = 0.0, all_sum = 0.0;
                            int even_n = 0, odd_n = 0, all_n = 0;
                            const double S_thr = 0.01;
                            for (int i = 0; i < Nx; ++i) {
                                for (int j = 0; j < Ny; ++j) {
                                    const int k = z_check;
                                    const DropletGeometryPoint point = sampleDropletGeometry(droplet_geometry, i, j, k);
                                    if (point.signed_distance >= -shell_exclude) continue;
                                    const size_t id = (size_t)k + (size_t)Nz * ((size_t)j + (size_t)Ny * (size_t)i);
                                    const double s = h_nf[id].S;
                                    if (!std::isfinite(s) || s <= S_thr) continue;
                                    all_sum += s; all_n++;
                                    if (((i + j) & 1) == 0) { even_sum += s; even_n++; }
                                    else { odd_sum += s; odd_n++; }
                                }
                            }
                            if (even_n > 10 && odd_n > 10 && all_n > 20 && all_sum != 0.0) {
                                // Parity projection: 0 means no odd-even bias.
                                checker_rel = std::abs(even_sum - odd_sum) / std::abs(all_sum);
                            }
                        }
                    }

                    if (do_log) {
                        quench_log << iter << "," << physical_time << "," << dt << "," << T_current << "," << bulk_sum << "," << elastic_sum
                                  << "," << anch_sum << "," << total_F << "," << avg_rad << "," << avg_S
                                  << "," << max_S << "," << checker_rel
                                  << "," << defect_density_2d << "," << defect_plaq_used
                                  << "," << xi_grad_proxy << "," << xi_grad_edges_used << "\n";
                        quench_log.flush();
                        if (user_choice == 'y' || user_choice == 'Y') {
                            std::cout << "Iter " << iter << "  t=" << physical_time << " s  T=" << T_current
                                      << " K  F=" << total_F << " (bulk=" << bulk_sum << ", el=" << elastic_sum
                                      << ", anch=" << anch_sum << ")  Rbar=" << avg_rad << "  <S>=" << avg_S
                                      << "  maxS=" << max_S << "  checker=" << checker_rel << std::endl;
                        }

                        if (enable_instability_guard) {
                            const bool s_bad = (!std::isfinite(max_S)) || (S_abort > 0.0 && max_S > S_abort);
                            const bool cb_bad = std::isfinite(checker_rel) && (checker_rel_abort > 0.0) && (checker_rel > checker_rel_abort);
                            if (s_bad || cb_bad) {
                                std::cerr << "\n[InstabilityGuard] Detected runaway/aliasing at iter " << iter
                                          << ": max_S=" << max_S << ", checker_rel=" << checker_rel
                                          << ". dt=" << dt << " s" << std::endl;
                                std::cerr << "[InstabilityGuard] Quench/KZ contract requires fixed dt."
                                          << " Aborting instead of changing dt."
                                          << " Try smaller dt, stronger stabilization, or milder elastic settings." << std::endl;
                                quench_aborted_due_to_instability = true;
                                break;
                            }
                        }

                        if (enable_early_stop && has_reached_final_T(iter)) {
                            bool energy_converged = false;
                            bool defect_converged = false;

                            if (std::isfinite(prev_F_for_stop) && prev_F_for_stop != 0.0) {
                                const double rel_dF = std::abs((total_F - prev_F_for_stop) / prev_F_for_stop);
                                energy_converged = rel_dF < tolerance;
                            }
                            defect_converged = defect_density_converged(prev_defect_for_stop, defect_density_2d, defect_plaq_used);

                            const bool radiality_threshold_met = (radiality_threshold <= 0.0) ? true : (avg_rad > radiality_threshold);
                            if (energy_converged && defect_converged) {
                                std::cout << "\n=== Early-stop: converged at final T (energy + defect proxy stable; Rbar="
                                          << avg_rad << ", defect2D=" << defect_density_2d
                                          << ", radiality_threshold_met=" << (radiality_threshold_met ? "yes" : "no")
                                          << ") ===\n";
                                break;
                            }

                            // Update references for the *next* convergence check.
                            prev_F_for_stop = total_F;
                            prev_defect_for_stop = (defect_plaq_used > 0u) ? defect_density_2d : std::numeric_limits<double>::quiet_NaN();
                        }
                    }

                    if (do_snap) {
                        saveNematicFieldToFile(h_nf, Nx, Ny, Nz, (out_dir / ("nematic_field_iter_" + std::to_string(iter) + ".dat")).string());
                        if (save_qtensor_snapshots) {
                            cudaMemcpy(h_Q.data(), d_Q, size_Q, cudaMemcpyDeviceToHost);
                            saveQTensorToFile(h_Q, Nx, Ny, Nz, (out_dir / ("Qtensor_output_iter_" + std::to_string(iter) + ".dat")).string());
                        }
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

            if (quench_aborted_due_to_instability) {
                quench_log.close();
                std::cerr << "[Quench] Run aborted by the fixed-dt instability guard."
                          << " Final state was not saved." << std::endl;
                return 1;
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

            double T_start = cfg_or_prompt<double>(cfg, "T_start", "Start T", 295.0, interactive);
            double T_end = cfg_or_prompt<double>(cfg, "T_end", "End T", 315.0, interactive);
            double T_step = cfg_or_prompt<double>(cfg, "T_step", "Step T", 1.0, interactive);

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
                double noise_amplitude = cfg_or_prompt<double>(cfg, "noise_amplitude", "Noise amplitude for isotropic init", 1e-3, interactive);
                initializeIsotropicWithNoise(h_Q, Nx, Ny, Nz, noise_amplitude, droplet_geometry, random_seed);
            } else {
                initializeQTensor(h_Q, Nx, Ny, Nz, S0, mode, droplet_geometry, random_seed);
            }
            cudaMemcpy(d_Q, h_Q.data(), size_Q, cudaMemcpyHostToDevice);

            DtEstimateRequest dt_request;
            dt_request.dx = dx;
            dt_request.dy = dy;
            dt_request.dz = dz;
            dt_request.gamma = gamma;
            dt_request.a = a;
            dt_request.b = b;
            dt_request.c = c;
            dt_request.T_star = T_star;
            dt_request.modechoice = modechoice;
            dt_request.kappa = kappa;
            dt_request.L1 = L1;
            dt_request.L2 = L2;
            dt_request.L3 = L3;
            dt_request.S_ref_used = S_ref_used;
            dt_request.S0 = S0;
            dt_request.T_a = T_start;
            dt_request.T_b = T_end;
            const DtEstimateResult dt_estimate = estimate_stable_dt_window(dt_request);
            print_dt_estimate_summary(dt_estimate);

            double dt = cfg_or_prompt<double>(cfg, "dt", "Enter time step dt (s)", dt_estimate.dt_max, interactive);
            if (dt > dt_estimate.dt_max) dt = dt_estimate.dt_max;

            // Simple physical-time floor to reduce "premature convergence" in the sweep.
            double R_phys = droplet_geometry.min_radius;
            double tau_align = (R_phys * R_phys) / dt_estimate.D_total;
            double min_alignment_time = 0.25 * tau_align;
            std::cout << "Estimated alignment time τ_align ≈ " << tau_align << " s"
                      << ", enforcing minimum sweep time ≈ " << min_alignment_time << " s" << std::endl;

            auto sweep_done = [&](double Tval) {
                const double eps = 1e-12;
                return (T_step > 0.0) ? (Tval > T_end + eps) : (Tval < T_end - eps);
            };

            for (double T_current = T_start; !sweep_done(T_current); T_current += T_step) {
                const double S_shell = compute_S_shell(T_current);
                std::cout << "\n--- Running simulation for T = " << T_current << " K"
                          << " with S_shell = " << S_shell
                          << " (" << boundary_order_mode_name(boundary_order_mode) << ") ---\n";
                DimensionalParams params_temp = {a, b, c, T_current, T_star, L1, L2, L3, W};

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
                    applyWeakAnchoringPenaltyKernel<<<numBlocks, threadsPerBlock>>>(d_mu, d_Q, Nx, Ny, Nz, d_is_shell, S_shell, W, shell_effective_thickness);
                    updateQTensorKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_mu, Nx, Ny, Nz, dt, d_is_shell, gamma, W, 0.0);
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
            int submode = cfg_or_prompt<int>(cfg, "submode", "Select submode: [1] full energy, [2] energy components", 1, interactive);
            if (submode != 1 && submode != 2) submode = 1;
            std::cout << "Submode selected: " << (submode == 1 ? "full energy" : "energy components") << std::endl;

            if (fs::exists("output")) { for (const auto& entry : fs::directory_iterator("output")) fs::remove(entry.path()); }
            fs::create_directory("output");

            DimensionalParams params = {a, b, c, T, T_star, L1, L2, L3, W};

            DtEstimateRequest dt_request;
            dt_request.dx = dx;
            dt_request.dy = dy;
            dt_request.dz = dz;
            dt_request.gamma = gamma;
            dt_request.a = a;
            dt_request.b = b;
            dt_request.c = c;
            dt_request.T_star = T_star;
            dt_request.modechoice = modechoice;
            dt_request.kappa = kappa;
            dt_request.L1 = L1;
            dt_request.L2 = L2;
            dt_request.L3 = L3;
            dt_request.S_ref_used = S_ref_used;
            dt_request.S0 = S0;
            dt_request.T_a = T;
            dt_request.T_b = T;
            const DtEstimateResult dt_estimate = estimate_stable_dt_window(dt_request);

            double R_phys = droplet_geometry.min_radius;
            double tau_align = (R_phys * R_phys) / dt_estimate.D_total;
            std::cout << "\n--- Time Scale Analysis ---" << std::endl;
            std::cout << "Diffusion coefficient D = " << dt_estimate.D_total << " m²/s" << std::endl;
            std::cout << "Droplet radius R ≈ " << R_phys << " m" << std::endl;
            std::cout << "Estimated alignment time τ_align ≈ " << tau_align << " s" << std::endl;
            print_dt_estimate_summary(dt_estimate);

            double dt = cfg_or_prompt<double>(cfg, "dt", "Enter time step dt (s)", dt_estimate.dt_max, interactive);
            if (dt > dt_estimate.dt_max) dt = dt_estimate.dt_max;

            if (mode == 2) {
                double noise_amplitude = cfg_or_prompt<double>(cfg, "noise_amplitude", "Noise amplitude for isotropic init", 1e-3, interactive);
                initializeIsotropicWithNoise(h_Q, Nx, Ny, Nz, noise_amplitude, droplet_geometry, random_seed);
            } else {
                initializeQTensor(h_Q, Nx, Ny, Nz, S0, mode, droplet_geometry, random_seed);
            }
            cudaMemcpy(d_Q, h_Q.data(), size_Q, cudaMemcpyHostToDevice);

            const double S_shell_single = compute_S_shell(T);
            std::cout << "[Boundary] Single-temp shell order at T=" << T << " K: S_shell=" << S_shell_single
                      << " (" << boundary_order_mode_name(boundary_order_mode) << " policy)." << std::endl;

            const bool console_output_single = cfg_or_prompt_yes_no(
                cfg,
                "console_output",
                ("Do you want output in the console every " + std::to_string(printFreq) + " iterations?").c_str(),
                true,
                interactive
            );
            char user_choice = console_output_single ? 'y' : 'n';

            double prev_F = std::numeric_limits<double>::max();
            double prev_defect = std::numeric_limits<double>::quiet_NaN();
            double physical_time = 0.0;
            double radiality_threshold = 0.998;
            double min_alignment_time = 0.5 * tau_align;
            int defects_z_slice_single = cfg_or_prompt<int>(cfg, "defects_z_slice", "Single-temp: z_slice for 2D defect proxy (0..Nz-1)", Nz / 2, interactive);
            if (defects_z_slice_single < 0) defects_z_slice_single = 0;
            if (defects_z_slice_single >= Nz) defects_z_slice_single = Nz - 1;
            double defects_S_threshold_single = cfg_or_prompt<double>(cfg, "defects_S_threshold", "Single-temp: S_threshold for 2D defect proxy", 0.1, interactive);
            double defects_charge_cutoff_single = cfg_or_prompt<double>(cfg, "defects_charge_cutoff", "Single-temp: charge_cutoff |s|>", 0.25, interactive);
            const bool log_xi_grad_2d_single = cfg_or_prompt_yes_no(
                cfg,
                "log_xi_grad_2d",
                "Single-temp: log 2D xi gradient proxy to iteration log?",
                true,
                interactive
            );
            int xi_z_slice_single = defects_z_slice_single;
            double xi_S_threshold_single = defects_S_threshold_single;
            if (log_xi_grad_2d_single) {
                xi_z_slice_single = cfg_or_prompt<int>(cfg, "xi_z_slice", "Single-temp: z_slice for xi proxy (0..Nz-1)", xi_z_slice_single, interactive);
                if (xi_z_slice_single < 0) xi_z_slice_single = 0;
                if (xi_z_slice_single >= Nz) xi_z_slice_single = Nz - 1;
                xi_S_threshold_single = cfg_or_prompt<double>(cfg, "xi_S_threshold", "Single-temp: S_threshold for xi proxy", xi_S_threshold_single, interactive);
            }
            std::cout << "[Convergence] Single-temp stopping uses energy + 2D defect proxy stability; radiality is diagnostic only." << std::endl;

            if (submode == 1) {
                std::ofstream energy_log("free_energy_vs_iteration.dat");
                energy_log << "iteration,bulk,elastic,anchoring,total,radiality,time,avg_S,max_S,defect_density_per_plaquette,defect_plaquettes_used,xi_grad_proxy,xi_grad_edges_used\n";

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
                    applyWeakAnchoringPenaltyKernel<<<numBlocks, threadsPerBlock>>>(d_mu, d_Q, Nx, Ny, Nz, d_is_shell, S_shell_single, W, shell_effective_thickness);
                    if (check_now) cuda_check_or_die("applyWeakAnchoringPenaltyKernel");
                    updateQTensorKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_mu, Nx, Ny, Nz, dt, d_is_shell, gamma, W, 0.0);
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
                        double defect_density_2d = std::numeric_limits<double>::quiet_NaN();
                        unsigned int defect_plaq_used = 0u;
                        compute_defect_density_2d(defects_z_slice_single, defects_S_threshold_single, defects_charge_cutoff_single, defect_density_2d, defect_plaq_used);

                        double xi_grad_proxy = std::numeric_limits<double>::quiet_NaN();
                        unsigned int xi_grad_edges_used = 0u;
                        if (log_xi_grad_2d_single) {
                            compute_xi_grad_proxy_2d(xi_z_slice_single, xi_S_threshold_single, xi_grad_proxy, xi_grad_edges_used);
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
                        const double avg_S = compute_avg_S_droplet(h_nf);
                        const double max_S = compute_max_S_droplet(h_nf);
                        energy_log << iter << "," << bulk_sum << "," << elastic_sum << "," << anch_sum
                                   << "," << total_F << "," << avg_rad << "," << physical_time
                                   << "," << avg_S << "," << max_S
                                   << "," << defect_density_2d << "," << defect_plaq_used
                                   << "," << xi_grad_proxy << "," << xi_grad_edges_used << "\n";
                        if (user_choice == 'y' || user_choice == 'Y') {
                            std::cout << "Iter " << iter << "  t=" << physical_time << " s" << "  F=" << total_F
                                      << " (bulk=" << bulk_sum << ", el=" << elastic_sum << ", anch=" << anch_sum
                                      << ")  R̄=" << avg_rad << "  <S>=" << avg_S << "  maxS=" << max_S
                                      << "  defect2D=" << defect_density_2d << "  xiGrad=" << xi_grad_proxy << std::endl;
                        }
                        saveNematicFieldToFile(h_nf, Nx, Ny, Nz, "output/nematic_field_iter_" + std::to_string(iter) + ".dat");

                        if (iter > 100 && prev_F != 0.0) {
                            double energy_change = std::abs((total_F - prev_F) / prev_F);
                            bool energy_converged = energy_change < tolerance;
                            bool time_sufficient = physical_time > min_alignment_time;
                            bool defect_converged = defect_density_converged(prev_defect, defect_density_2d, defect_plaq_used);
                            const bool radiality_threshold_met = avg_rad > radiality_threshold;
                            if (energy_converged && defect_converged && time_sufficient) {
                                std::cout << "\n=== Convergence Achieved (energy + defect proxy stable; Rbar="
                                          << avg_rad << ", defect2D=" << defect_density_2d
                                          << ", radiality_threshold_met=" << (radiality_threshold_met ? "yes" : "no")
                                          << ") ===";
                                break;
                            }
                        }
                        prev_F = total_F;
                        prev_defect = (defect_plaq_used > 0u) ? defect_density_2d : std::numeric_limits<double>::quiet_NaN();
                    }
                }
                energy_log.close();
            } else {
                std::ofstream energy_components_log("energy_components_vs_iteration.dat");
                energy_components_log << "iteration,bulk,elastic,anchoring,total,radiality,time,avg_S,max_S,defect_density_per_plaquette,defect_plaquettes_used,xi_grad_proxy,xi_grad_edges_used\n";

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
                    applyWeakAnchoringPenaltyKernel<<<numBlocks, threadsPerBlock>>>(d_mu, d_Q, Nx, Ny, Nz, d_is_shell, S_shell_single, W, shell_effective_thickness);
                    if (check_now) cuda_check_or_die("applyWeakAnchoringPenaltyKernel");
                    updateQTensorKernel<<<numBlocks, threadsPerBlock>>>(d_Q, d_mu, Nx, Ny, Nz, dt, d_is_shell, gamma, W, 0.0);
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
                        double defect_density_2d = std::numeric_limits<double>::quiet_NaN();
                        unsigned int defect_plaq_used = 0u;
                        compute_defect_density_2d(defects_z_slice_single, defects_S_threshold_single, defects_charge_cutoff_single, defect_density_2d, defect_plaq_used);

                        double xi_grad_proxy = std::numeric_limits<double>::quiet_NaN();
                        unsigned int xi_grad_edges_used = 0u;
                        if (log_xi_grad_2d_single) {
                            compute_xi_grad_proxy_2d(xi_z_slice_single, xi_S_threshold_single, xi_grad_proxy, xi_grad_edges_used);
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
                        const double avg_S = compute_avg_S_droplet(h_nf);
                        const double max_S = compute_max_S_droplet(h_nf);
                        energy_components_log << iter << "," << bulk_sum << "," << elastic_sum << "," << anch_sum
                                             << "," << total_F << "," << avg_rad << "," << physical_time
                                             << "," << avg_S << "," << max_S
                                             << "," << defect_density_2d << "," << defect_plaq_used
                                             << "," << xi_grad_proxy << "," << xi_grad_edges_used << "\n";
                        if (user_choice == 'y' || user_choice == 'Y') {
                            std::cout << "Iter " << iter << "  t=" << physical_time << " s" << "  F=" << total_F
                                      << " (bulk=" << bulk_sum << ", el=" << elastic_sum << ", anch=" << anch_sum
                                      << ")  R̄=" << avg_rad << "  <S>=" << avg_S << "  maxS=" << max_S
                                      << "  defect2D=" << defect_density_2d << "  xiGrad=" << xi_grad_proxy << std::endl;
                        }
                        saveNematicFieldToFile(h_nf, Nx, Ny, Nz, "output/nematic_field_iter_" + std::to_string(iter) + ".dat");

                        if (iter > 100 && prev_F != 0.0) {
                            double energy_change = std::abs((total_F - prev_F) / prev_F);
                            bool energy_converged = energy_change < tolerance;
                            bool defect_converged = defect_density_converged(prev_defect, defect_density_2d, defect_plaq_used);
                            const bool radiality_threshold_met = avg_rad > radiality_threshold;
                            if (energy_converged && defect_converged && physical_time > min_alignment_time) {
                                std::cout << "\n=== Convergence Achieved (energy + defect proxy stable; Rbar="
                                          << avg_rad << ", defect2D=" << defect_density_2d
                                          << ", radiality_threshold_met=" << (radiality_threshold_met ? "yes" : "no")
                                          << ") ===";
                                break;
                            }
                        }
                        prev_F = total_F;
                        prev_defect = (defect_plaq_used > 0u) ? defect_density_2d : std::numeric_limits<double>::quiet_NaN();
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
        cudaFree(d_Q); cudaFree(d_Q_alt); cudaFree(d_rhs);
        cudaFree(d_laplacianQ); cudaFree(d_mu); cudaFree(d_is_shell);
        cudaFree(d_Dcol_x); cudaFree(d_Dcol_y); cudaFree(d_Dcol_z);
        cudaFree(d_bulk_energy); cudaFree(d_elastic_energy); cudaFree(d_anch_energy);
        cudaFree(d_radiality_vals); cudaFree(d_count_vals);
        cudaFree(d_defects_2d_count); cudaFree(d_defects_2d_plaq_used);
        cudaFree(d_xi_grad_sum); cudaFree(d_xi_grad_edges_used);

        std::cout << "\nSimulation finished." << std::endl;
        if (allow_run_again_loop) {
            std::cout << "Would you like to run another simulation? (y/n): ";
            std::string again; std::getline(std::cin, again);
            run_again = (!again.empty() && (again[0] == 'y' || again[0] == 'Y'));
        } else {
            run_again = false;
        }

    } while (run_again);

    if (interactive) {
        std::cout << "Exiting. Press enter to close." << std::endl;
        std::cin.get();
    }
    return 0;
}