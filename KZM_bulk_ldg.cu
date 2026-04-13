#include "KZM_bulk_ldg.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            std::ostringstream oss__;                                                    \
            oss__ << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "      \
                  << cudaGetErrorString(err__);                                          \
            throw std::runtime_error(oss__.str());                                       \
        }                                                                                \
    } while (0)

struct BulkLdgObservables {
    double bulk = std::numeric_limits<double>::quiet_NaN();
    double elastic = std::numeric_limits<double>::quiet_NaN();
    double total = std::numeric_limits<double>::quiet_NaN();
    double avg_S = std::numeric_limits<double>::quiet_NaN();
    double max_S = std::numeric_limits<double>::quiet_NaN();
    double defect_density_per_plaquette = std::numeric_limits<double>::quiet_NaN();
    double defect_line_density = std::numeric_limits<double>::quiet_NaN();
    double xi_grad_proxy = std::numeric_limits<double>::quiet_NaN();
    long long defect_plaquettes_used = 0;
    long long nonzero_defect_plaquettes = 0;
};

static inline std::string trim_copy(const std::string& s) {
    size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) {
        ++begin;
    }
    size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(begin, end - begin);
}

template <typename T>
static T parse_value_or_default(const std::string& raw, T fallback) {
    const std::string value = trim_copy(raw);
    if (value.empty()) {
        return fallback;
    }

    if constexpr (std::is_same_v<T, bool>) {
        std::string lower;
        lower.reserve(value.size());
        for (char c : value) {
            lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }
        if (lower == "1" || lower == "true" || lower == "yes" || lower == "y" || lower == "on") {
            return true;
        }
        if (lower == "0" || lower == "false" || lower == "no" || lower == "n" || lower == "off") {
            return false;
        }
        return fallback;
    } else if constexpr (std::is_integral_v<T>) {
        try {
            size_t idx = 0;
            long double numeric = std::stold(value, &idx);
            while (idx < value.size() && std::isspace(static_cast<unsigned char>(value[idx]))) {
                ++idx;
            }
            if (idx != value.size()) {
                return fallback;
            }
            const long double rounded = std::llround(numeric);
            if (rounded < static_cast<long double>(std::numeric_limits<T>::min()) ||
                rounded > static_cast<long double>(std::numeric_limits<T>::max())) {
                return fallback;
            }
            return static_cast<T>(rounded);
        } catch (...) {
            return fallback;
        }
    } else {
        std::istringstream iss(value);
        T parsed = fallback;
        if (!(iss >> parsed)) {
            return fallback;
        }
        return parsed;
    }
}

static std::unordered_map<std::string, std::string> parse_kv_config_file(const std::string& path) {
    std::unordered_map<std::string, std::string> cfg;
    if (path.empty()) {
        return cfg;
    }

    std::ifstream handle(path);
    if (!handle) {
        throw std::runtime_error("Could not open config file: " + path);
    }

    std::string line;
    while (std::getline(handle, line)) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) {
            line = line.substr(0, hash);
        }
        line = trim_copy(line);
        if (line.empty()) {
            continue;
        }
        const size_t eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        std::string key = trim_copy(line.substr(0, eq));
        std::string value = trim_copy(line.substr(eq + 1));
        if (!key.empty()) {
            cfg[key] = value;
        }
    }
    return cfg;
}

template <typename T>
static T cfg_get_value(const std::unordered_map<std::string, std::string>& cfg, const std::string& key, T fallback) {
    auto it = cfg.find(key);
    if (it == cfg.end()) {
        return fallback;
    }
    return parse_value_or_default<T>(it->second, fallback);
}

static std::string cfg_get_string(
    const std::unordered_map<std::string, std::string>& cfg,
    const std::string& key,
    const std::string& fallback
) {
    auto it = cfg.find(key);
    if (it == cfg.end()) {
        return fallback;
    }
    return trim_copy(it->second);
}

static inline size_t flatten_host(int x, int y, int z, int Nx, int Ny, int Nz) {
    (void)Nz;
    return static_cast<size_t>(x) + static_cast<size_t>(Nx) * (static_cast<size_t>(y) + static_cast<size_t>(Ny) * static_cast<size_t>(z));
}

__host__ __device__ static inline size_t flatten_device(int x, int y, int z, int Nx, int Ny, int Nz) {
    (void)Nz;
    return static_cast<size_t>(x) + static_cast<size_t>(Nx) * (static_cast<size_t>(y) + static_cast<size_t>(Ny) * static_cast<size_t>(z));
}

__device__ static inline unsigned long long splitmix64_next(unsigned long long& state) {
    state += 0x9E3779B97F4A7C15ull;
    unsigned long long z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

__device__ static inline double uniform01(unsigned long long& state) {
    const unsigned long long bits = splitmix64_next(state);
    return (static_cast<double>(bits >> 11) + 1.0) * 0x1.0p-53;
}

__device__ static inline void normal_pair(unsigned long long& state, double& g0, double& g1) {
    const double u1 = fmax(uniform01(state), 1e-14);
    const double u2 = uniform01(state);
    const double radius = sqrt(-2.0 * log(u1));
    const double angle = 2.0 * KZM_BULK_PI * u2;
    g0 = radius * cos(angle);
    g1 = radius * sin(angle);
}

__host__ __device__ static inline double qtensor_frobenius_norm(const QTensor& q5) {
    const FullQTensor q(q5);
    const double tr_q2 = q.Qxx * q.Qxx + q.Qyy * q.Qyy + q.Qzz * q.Qzz
        + 2.0 * (q.Qxy * q.Qxy + q.Qxz * q.Qxz + q.Qyz * q.Qyz);
    return sqrt(fmax(tr_q2, 0.0));
}

__host__ __device__ static NematicField calculateNematicFieldDevice(const QTensor& q_5comp) {
    FullQTensor q(q_5comp);

    if (fabs(q.Qxx) < 1e-12 && fabs(q.Qxy) < 1e-12 && fabs(q.Qxz) < 1e-12 &&
        fabs(q.Qyy) < 1e-12 && fabs(q.Qyz) < 1e-12) {
        return {0.0, 1.0, 0.0, 0.0};
    }

    double nx = q.Qxx + q.Qxy + q.Qxz;
    double ny = q.Qxy + q.Qyy + q.Qyz;
    double nz = q.Qxz + q.Qyz - q.Qxx - q.Qyy;
    double init_norm = sqrt(nx * nx + ny * ny + nz * nz);
    if (init_norm > 1e-12) {
        nx /= init_norm;
        ny /= init_norm;
        nz /= init_norm;
    } else {
        nx = 0.0;
        ny = 0.0;
        nz = 1.0;
    }

    double row1 = fabs(q.Qxx) + fabs(q.Qxy) + fabs(q.Qxz);
    double row2 = fabs(q.Qyx) + fabs(q.Qyy) + fabs(q.Qyz);
    double row3 = fabs(q.Qzx) + fabs(q.Qzy) + fabs(q.Qzz);
    double alpha = fmax(row1, fmax(row2, row3)) + 1e-12;

    for (int iter = 0; iter < 15; ++iter) {
        const double qnx = q.Qxx * nx + q.Qxy * ny + q.Qxz * nz + alpha * nx;
        const double qny = q.Qyx * nx + q.Qyy * ny + q.Qyz * nz + alpha * ny;
        const double qnz = q.Qzx * nx + q.Qzy * ny + q.Qzz * nz + alpha * nz;
        const double norm = sqrt(qnx * qnx + qny * qny + qnz * qnz);
        if (norm < 1e-13) {
            break;
        }
        nx = qnx / norm;
        ny = qny / norm;
        nz = qnz / norm;
    }

    const double lambda_max =
        nx * (q.Qxx * nx + q.Qxy * ny + q.Qxz * nz) +
        ny * (q.Qyx * nx + q.Qyy * ny + q.Qyz * nz) +
        nz * (q.Qzx * nx + q.Qzy * ny + q.Qzz * nz);

    double S = 1.5 * lambda_max;
    if (nz < -1e-10) {
        nx = -nx;
        ny = -ny;
        nz = -nz;
    }
    return {S, nx, ny, nz};
}

static void bulk_ABC_from_convention(double a, double b, double c, double T, double T_star, int modechoice,
                                    double& A, double& B, double& C) {
    const double dT = T - T_star;
    A = 3.0 * a * dT;
    if (modechoice == 1) {
        B = (9.0 * b) / 2.0;
        C = (9.0 * c) / 2.0;
    } else {
        B = (27.0 * b) / 2.0;
        C = 9.0 * c;
    }
}

static double bulk_fS_uniaxial(double A, double B, double C, double S) {
    const double S2 = S * S;
    return (A / 3.0) * S2 - (2.0 * B / 27.0) * (S2 * S) + (C / 9.0) * (S2 * S2);
}

static double S_eq_uniaxial_from_ABC(double A, double B, double C) {
    if (!(C > 0.0)) {
        return 0.0;
    }
    const double disc = B * B - 24.0 * A * C;
    if (!(disc > 0.0)) {
        return 0.0;
    }

    const double sqrt_disc = std::sqrt(disc);
    const double S1 = (B - sqrt_disc) / (4.0 * C);
    const double S2 = (B + sqrt_disc) / (4.0 * C);

    double best_S = 0.0;
    double best_f = 0.0;
    auto consider = [&](double S) {
        if (!(S > 0.0)) {
            return;
        }
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
    if (!(C > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return (B * B) / (27.0 * C);
}

static double T_from_A(double A, double a, double T_star) {
    if (!(a > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return T_star + A / (3.0 * a);
}

static double compute_temperature(const BulkLdgParams& params, int iteration) {
    if (iteration < params.pre_equil_iters) {
        return params.T_high;
    }
    if (iteration >= params.pre_equil_iters + params.ramp_iters) {
        return params.T_low;
    }
    if (params.ramp_iters <= 0) {
        return params.T_low;
    }
    const double frac = static_cast<double>(iteration - params.pre_equil_iters) / static_cast<double>(params.ramp_iters);
    return params.T_high + frac * (params.T_low - params.T_high);
}

static double estimate_stable_dt(const BulkLdgParams& params) {
    double A_high = 0.0, B_high = 0.0, C_high = 0.0;
    double A_low = 0.0, B_low = 0.0, C_low = 0.0;
    bulk_ABC_from_convention(params.a, params.b, params.c, params.T_high, params.T_star, params.bulk_modechoice, A_high, B_high, C_high);
    bulk_ABC_from_convention(params.a, params.b, params.c, params.T_low, params.T_star, params.bulk_modechoice, A_low, B_low, C_low);

    const double S_eq_low = S_eq_uniaxial_from_ABC(A_low, B_low, C_low);
    const double tr_q2_eq = 2.0 * S_eq_low * S_eq_low / 3.0;
    const double nonlinear_scale = std::abs(B_low) * std::abs(S_eq_low) + std::abs(C_low) * tr_q2_eq;
    const double inv_dx2 = 1.0 / (params.dx * params.dx);
    const double inv_dy2 = 1.0 / (params.dy * params.dy);
    const double inv_dz2 = 1.0 / (params.dz * params.dz);
    const double laplacian_scale = 2.0 * params.kappa * (inv_dx2 + inv_dy2 + inv_dz2);
    const double local_scale = std::max(std::abs(A_high), std::abs(A_low)) + nonlinear_scale;
    const double stiffness = (1.0 / params.gamma) * (laplacian_scale + local_scale);
    if (!(stiffness > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return 0.2 / stiffness;
}

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
) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    const size_t total_sites = static_cast<size_t>(params.Nx) * static_cast<size_t>(params.Ny) * static_cast<size_t>(params.Nz);
    if (idx >= total_sites) {
        return;
    }

    const int x = static_cast<int>(idx % static_cast<size_t>(params.Nx));
    const size_t yz = idx / static_cast<size_t>(params.Nx);
    const int y = static_cast<int>(yz % static_cast<size_t>(params.Ny));
    const int z = static_cast<int>(yz / static_cast<size_t>(params.Ny));

    const int xp = (x + 1) % params.Nx;
    const int xm = (x + params.Nx - 1) % params.Nx;
    const int yp = (y + 1) % params.Ny;
    const int ym = (y + params.Ny - 1) % params.Ny;
    const int zp = (z + 1) % params.Nz;
    const int zm = (z + params.Nz - 1) % params.Nz;

    const QTensor center = current[idx];
    const QTensor plus_x = current[flatten_device(xp, y, z, params.Nx, params.Ny, params.Nz)];
    const QTensor minus_x = current[flatten_device(xm, y, z, params.Nx, params.Ny, params.Nz)];
    const QTensor plus_y = current[flatten_device(x, yp, z, params.Nx, params.Ny, params.Nz)];
    const QTensor minus_y = current[flatten_device(x, ym, z, params.Nx, params.Ny, params.Nz)];
    const QTensor plus_z = current[flatten_device(x, y, zp, params.Nx, params.Ny, params.Nz)];
    const QTensor minus_z = current[flatten_device(x, y, zm, params.Nx, params.Ny, params.Nz)];

    const double inv_dx2 = 1.0 / (params.dx * params.dx);
    const double inv_dy2 = 1.0 / (params.dy * params.dy);
    const double inv_dz2 = 1.0 / (params.dz * params.dz);

    QTensor lap;
    lap.Qxx = (plus_x.Qxx - 2.0 * center.Qxx + minus_x.Qxx) * inv_dx2 +
              (plus_y.Qxx - 2.0 * center.Qxx + minus_y.Qxx) * inv_dy2 +
              (plus_z.Qxx - 2.0 * center.Qxx + minus_z.Qxx) * inv_dz2;
    lap.Qxy = (plus_x.Qxy - 2.0 * center.Qxy + minus_x.Qxy) * inv_dx2 +
              (plus_y.Qxy - 2.0 * center.Qxy + minus_y.Qxy) * inv_dy2 +
              (plus_z.Qxy - 2.0 * center.Qxy + minus_z.Qxy) * inv_dz2;
    lap.Qxz = (plus_x.Qxz - 2.0 * center.Qxz + minus_x.Qxz) * inv_dx2 +
              (plus_y.Qxz - 2.0 * center.Qxz + minus_y.Qxz) * inv_dy2 +
              (plus_z.Qxz - 2.0 * center.Qxz + minus_z.Qxz) * inv_dz2;
    lap.Qyy = (plus_x.Qyy - 2.0 * center.Qyy + minus_x.Qyy) * inv_dx2 +
              (plus_y.Qyy - 2.0 * center.Qyy + minus_y.Qyy) * inv_dy2 +
              (plus_z.Qyy - 2.0 * center.Qyy + minus_z.Qyy) * inv_dz2;
    lap.Qyz = (plus_x.Qyz - 2.0 * center.Qyz + minus_x.Qyz) * inv_dx2 +
              (plus_y.Qyz - 2.0 * center.Qyz + minus_y.Qyz) * inv_dy2 +
              (plus_z.Qyz - 2.0 * center.Qyz + minus_z.Qyz) * inv_dz2;

    const FullQTensor q(center);
    FullQTensor q2;
    q2.Qxx = q.Qxx * q.Qxx + q.Qxy * q.Qyx + q.Qxz * q.Qzx;
    q2.Qxy = q.Qxx * q.Qxy + q.Qxy * q.Qyy + q.Qxz * q.Qzy;
    q2.Qxz = q.Qxx * q.Qxz + q.Qxy * q.Qyz + q.Qxz * q.Qzz;
    q2.Qyy = q.Qyx * q.Qxy + q.Qyy * q.Qyy + q.Qyz * q.Qzy;
    q2.Qyz = q.Qyx * q.Qxz + q.Qyy * q.Qyz + q.Qyz * q.Qzz;
    q2.Qzz = q.Qzx * q.Qxz + q.Qzy * q.Qyz + q.Qzz * q.Qzz;

    const double trQ2 = q.Qxx * q.Qxx + q.Qyy * q.Qyy + q.Qzz * q.Qzz
        + 2.0 * (q.Qxy * q.Qxy + q.Qxz * q.Qxz + q.Qyz * q.Qyz);

    double h_xx = A * q.Qxx - B * (q2.Qxx - trQ2 / 3.0) + C * trQ2 * q.Qxx - params.kappa * lap.Qxx;
    double h_yy = A * q.Qyy - B * (q2.Qyy - trQ2 / 3.0) + C * trQ2 * q.Qyy - params.kappa * lap.Qyy;
    double h_zz = A * q.Qzz - B * (q2.Qzz - trQ2 / 3.0) + C * trQ2 * q.Qzz + params.kappa * (lap.Qxx + lap.Qyy);
    double h_xy = A * q.Qxy - B * q2.Qxy + C * trQ2 * q.Qxy - params.kappa * lap.Qxy;
    double h_xz = A * q.Qxz - B * q2.Qxz + C * trQ2 * q.Qxz - params.kappa * lap.Qxz;
    double h_yz = A * q.Qyz - B * q2.Qyz + C * trQ2 * q.Qyz - params.kappa * lap.Qyz;

    const double trace_h = h_xx + h_yy + h_zz;
    const double corr = trace_h / 3.0;
    h_xx -= corr;
    h_yy -= corr;

    const double mobility = 1.0 / params.gamma;
    QTensor updated;
    updated.Qxx = center.Qxx - mobility * params.dt * h_xx;
    updated.Qxy = center.Qxy - mobility * params.dt * h_xy;
    updated.Qxz = center.Qxz - mobility * params.dt * h_xz;
    updated.Qyy = center.Qyy - mobility * params.dt * h_yy;
    updated.Qyz = center.Qyz - mobility * params.dt * h_yz;

    if (noise_sigma > 0.0) {
        unsigned long long state = base_seed;
        state ^= 0x9E3779B97F4A7C15ull * (idx + 1ull);
        state ^= 0xBF58476D1CE4E5B9ull * static_cast<unsigned long long>(iteration + 1);
        double g0 = 0.0, g1 = 0.0, g2 = 0.0, g3 = 0.0, g4 = 0.0, g5 = 0.0;
        normal_pair(state, g0, g1);
        normal_pair(state, g2, g3);
        normal_pair(state, g4, g5);
        updated.Qxx += noise_sigma * g0;
        updated.Qxy += noise_sigma * g1;
        updated.Qxz += noise_sigma * g2;
        updated.Qyy += noise_sigma * g3;
        updated.Qyz += noise_sigma * g4;
    }

    if (params.q_norm_cap > 0.0) {
        const double qnorm = qtensor_frobenius_norm(updated);
        if (qnorm > params.q_norm_cap && qnorm > 0.0) {
            const double s = params.q_norm_cap / qnorm;
            updated.Qxx *= s;
            updated.Qxy *= s;
            updated.Qxz *= s;
            updated.Qyy *= s;
            updated.Qyz *= s;
        }
    }

    next[idx] = updated;
}

static double wrap_to_pi(double delta) {
    while (delta <= -KZM_BULK_PI) {
        delta += 2.0 * KZM_BULK_PI;
    }
    while (delta > KZM_BULK_PI) {
        delta -= 2.0 * KZM_BULK_PI;
    }
    return delta;
}

static bool project_to_plane(const NematicField& nf, int plane_id, double& u, double& v) {
    if (plane_id == 0) {
        u = nf.nx;
        v = nf.ny;
    } else if (plane_id == 1) {
        u = nf.ny;
        v = nf.nz;
    } else {
        u = nf.nx;
        v = nf.nz;
    }
    const double norm = std::sqrt(u * u + v * v);
    if (!(norm > 1e-8)) {
        return false;
    }
    u /= norm;
    v /= norm;
    return true;
}

static bool plaquette_charge(
    const NematicField& nf00,
    const NematicField& nf10,
    const NematicField& nf11,
    const NematicField& nf01,
    int plane_id,
    double S_threshold,
    double charge_cutoff,
    double& abs_charge
) {
    abs_charge = std::numeric_limits<double>::quiet_NaN();
    if (!(std::isfinite(nf00.S) && std::isfinite(nf10.S) && std::isfinite(nf11.S) && std::isfinite(nf01.S))) {
        return false;
    }
    if (!(nf00.S > S_threshold && nf10.S > S_threshold && nf11.S > S_threshold && nf01.S > S_threshold)) {
        return false;
    }

    double u00 = 0.0, v00 = 0.0, u10 = 0.0, v10 = 0.0, u11 = 0.0, v11 = 0.0, u01 = 0.0, v01 = 0.0;
    if (!project_to_plane(nf00, plane_id, u00, v00) ||
        !project_to_plane(nf10, plane_id, u10, v10) ||
        !project_to_plane(nf11, plane_id, u11, v11) ||
        !project_to_plane(nf01, plane_id, u01, v01)) {
        return false;
    }

    const double p00 = 2.0 * std::atan2(v00, u00);
    const double p10 = 2.0 * std::atan2(v10, u10);
    const double p11 = 2.0 * std::atan2(v11, u11);
    const double p01 = 2.0 * std::atan2(v01, u01);

    const double dsum =
        wrap_to_pi(p10 - p00) +
        wrap_to_pi(p11 - p10) +
        wrap_to_pi(p01 - p11) +
        wrap_to_pi(p00 - p01);
    const double w = dsum / (2.0 * KZM_BULK_PI);
    const double s = 0.5 * w;
    abs_charge = std::fabs(s);
    return abs_charge > charge_cutoff;
}

static BulkLdgObservables compute_observables(
    const std::vector<QTensor>& field,
    const BulkLdgParams& params,
    double A,
    double B,
    double C
) {
    BulkLdgObservables obs;

    const double cell_volume = params.dx * params.dy * params.dz;
    const double total_volume = cell_volume * static_cast<double>(params.Nx) * static_cast<double>(params.Ny) * static_cast<double>(params.Nz);
    const double inv_dx = 1.0 / params.dx;
    const double inv_dy = 1.0 / params.dy;
    const double inv_dz = 1.0 / params.dz;
    const double site_count = static_cast<double>(params.Nx) * static_cast<double>(params.Ny) * static_cast<double>(params.Nz);

    double bulk_sum = 0.0;
    double elastic_sum = 0.0;
    double qnorm_sq_integral = 0.0;
    double grad_sq_integral = 0.0;
    double S_sum = 0.0;
    double S_max = 0.0;
    std::vector<NematicField> nf(field.size());

    for (int z = 0; z < params.Nz; ++z) {
        const int zp = (z + 1) % params.Nz;
        for (int y = 0; y < params.Ny; ++y) {
            const int yp = (y + 1) % params.Ny;
            for (int x = 0; x < params.Nx; ++x) {
                const int xp = (x + 1) % params.Nx;
                const size_t idx = flatten_host(x, y, z, params.Nx, params.Ny, params.Nz);
                const QTensor& center = field[idx];
                const QTensor& plus_x = field[flatten_host(xp, y, z, params.Nx, params.Ny, params.Nz)];
                const QTensor& plus_y = field[flatten_host(x, yp, z, params.Nx, params.Ny, params.Nz)];
                const QTensor& plus_z = field[flatten_host(x, y, zp, params.Nx, params.Ny, params.Nz)];

                const FullQTensor q(center);
                FullQTensor q2;
                q2.Qxx = q.Qxx * q.Qxx + q.Qxy * q.Qyx + q.Qxz * q.Qzx;
                q2.Qxy = q.Qxx * q.Qxy + q.Qxy * q.Qyy + q.Qxz * q.Qzy;
                q2.Qxz = q.Qxx * q.Qxz + q.Qxy * q.Qyz + q.Qxz * q.Qzz;
                q2.Qyy = q.Qyx * q.Qxy + q.Qyy * q.Qyy + q.Qyz * q.Qzy;
                q2.Qyz = q.Qyx * q.Qxz + q.Qyy * q.Qyz + q.Qyz * q.Qzz;
                q2.Qzz = q.Qzx * q.Qxz + q.Qzy * q.Qyz + q.Qzz * q.Qzz;

                const double trQ2 = q.Qxx * q.Qxx + q.Qyy * q.Qyy + q.Qzz * q.Qzz
                    + 2.0 * (q.Qxy * q.Qxy + q.Qxz * q.Qxz + q.Qyz * q.Qyz);
                const double trQ3 =
                    q.Qxx * q2.Qxx + q.Qxy * q2.Qxy + q.Qxz * q2.Qxz +
                    q.Qyx * q2.Qxy + q.Qyy * q2.Qyy + q.Qyz * q2.Qyz +
                    q.Qzx * q2.Qxz + q.Qzy * q2.Qyz + q.Qzz * q2.Qzz;
                bulk_sum += ((A / 2.0) * trQ2 - (B / 3.0) * trQ3 + (C / 4.0) * trQ2 * trQ2) * cell_volume;

                const double dQxx_dx = (plus_x.Qxx - center.Qxx) * inv_dx;
                const double dQxx_dy = (plus_y.Qxx - center.Qxx) * inv_dy;
                const double dQxx_dz = (plus_z.Qxx - center.Qxx) * inv_dz;
                const double dQyy_dx = (plus_x.Qyy - center.Qyy) * inv_dx;
                const double dQyy_dy = (plus_y.Qyy - center.Qyy) * inv_dy;
                const double dQyy_dz = (plus_z.Qyy - center.Qyy) * inv_dz;
                const double dQxy_dx = (plus_x.Qxy - center.Qxy) * inv_dx;
                const double dQxy_dy = (plus_y.Qxy - center.Qxy) * inv_dy;
                const double dQxy_dz = (plus_z.Qxy - center.Qxy) * inv_dz;
                const double dQxz_dx = (plus_x.Qxz - center.Qxz) * inv_dx;
                const double dQxz_dy = (plus_y.Qxz - center.Qxz) * inv_dy;
                const double dQxz_dz = (plus_z.Qxz - center.Qxz) * inv_dz;
                const double dQyz_dx = (plus_x.Qyz - center.Qyz) * inv_dx;
                const double dQyz_dy = (plus_y.Qyz - center.Qyz) * inv_dy;
                const double dQyz_dz = (plus_z.Qyz - center.Qyz) * inv_dz;
                const double dQzz_dx = -(dQxx_dx + dQyy_dx);
                const double dQzz_dy = -(dQxx_dy + dQyy_dy);
                const double dQzz_dz = -(dQxx_dz + dQyy_dz);

                const double grad_sq =
                    dQxx_dx * dQxx_dx + dQxx_dy * dQxx_dy + dQxx_dz * dQxx_dz +
                    dQyy_dx * dQyy_dx + dQyy_dy * dQyy_dy + dQyy_dz * dQyy_dz +
                    dQzz_dx * dQzz_dx + dQzz_dy * dQzz_dy + dQzz_dz * dQzz_dz +
                    2.0 * (
                        dQxy_dx * dQxy_dx + dQxy_dy * dQxy_dy + dQxy_dz * dQxy_dz +
                        dQxz_dx * dQxz_dx + dQxz_dy * dQxz_dy + dQxz_dz * dQxz_dz +
                        dQyz_dx * dQyz_dx + dQyz_dy * dQyz_dy + dQyz_dz * dQyz_dz
                    );
                grad_sq_integral += grad_sq * cell_volume;
                elastic_sum += 0.5 * params.kappa * grad_sq * cell_volume;
                qnorm_sq_integral += trQ2 * cell_volume;

                nf[idx] = calculateNematicFieldDevice(center);
                S_sum += nf[idx].S;
                S_max = std::max(S_max, nf[idx].S);
            }
        }
    }

    long long defects_count = 0;
    long long plaquettes_used = 0;
    double line_length = 0.0;

    for (int z = 0; z < params.Nz; ++z) {
        const int zp = (z + 1) % params.Nz;
        for (int y = 0; y < params.Ny; ++y) {
            const int yp = (y + 1) % params.Ny;
            for (int x = 0; x < params.Nx; ++x) {
                const int xp = (x + 1) % params.Nx;

                const NematicField& nf000 = nf[flatten_host(x, y, z, params.Nx, params.Ny, params.Nz)];
                const NematicField& nf100 = nf[flatten_host(xp, y, z, params.Nx, params.Ny, params.Nz)];
                const NematicField& nf110 = nf[flatten_host(xp, yp, z, params.Nx, params.Ny, params.Nz)];
                const NematicField& nf010 = nf[flatten_host(x, yp, z, params.Nx, params.Ny, params.Nz)];
                const NematicField& nf001 = nf[flatten_host(x, y, zp, params.Nx, params.Ny, params.Nz)];
                const NematicField& nf101 = nf[flatten_host(xp, y, zp, params.Nx, params.Ny, params.Nz)];
                const NematicField& nf011 = nf[flatten_host(x, yp, zp, params.Nx, params.Ny, params.Nz)];

                double abs_charge = 0.0;
                const bool xy_has_defect = plaquette_charge(
                    nf000, nf100, nf110, nf010, 0,
                    params.defects_S_threshold, params.defects_charge_cutoff, abs_charge
                );
                if (std::isfinite(abs_charge)) {
                    ++plaquettes_used;
                    if (xy_has_defect) {
                        ++defects_count;
                        line_length += params.dz;
                    }
                }

                abs_charge = 0.0;
                const bool yz_has_defect = plaquette_charge(
                    nf000, nf010, nf011, nf001, 1,
                    params.defects_S_threshold, params.defects_charge_cutoff, abs_charge
                );
                if (std::isfinite(abs_charge)) {
                    ++plaquettes_used;
                    if (yz_has_defect) {
                        ++defects_count;
                        line_length += params.dx;
                    }
                }

                abs_charge = 0.0;
                const bool xz_has_defect = plaquette_charge(
                    nf000, nf100, nf101, nf001, 2,
                    params.defects_S_threshold, params.defects_charge_cutoff, abs_charge
                );
                if (std::isfinite(abs_charge)) {
                    ++plaquettes_used;
                    if (xz_has_defect) {
                        ++defects_count;
                        line_length += params.dy;
                    }
                }
            }
        }
    }

    obs.bulk = bulk_sum / total_volume;
    obs.elastic = elastic_sum / total_volume;
    obs.total = obs.bulk + obs.elastic;
    obs.avg_S = S_sum / site_count;
    obs.max_S = S_max;
    obs.defect_plaquettes_used = plaquettes_used;
    obs.nonzero_defect_plaquettes = defects_count;
    obs.defect_density_per_plaquette = (plaquettes_used > 0)
        ? static_cast<double>(defects_count) / static_cast<double>(plaquettes_used)
        : std::numeric_limits<double>::quiet_NaN();
    obs.defect_line_density = (total_volume > 0.0)
        ? line_length / total_volume
        : std::numeric_limits<double>::quiet_NaN();
    if (grad_sq_integral > 0.0 && qnorm_sq_integral > 0.0) {
        obs.xi_grad_proxy = std::sqrt(qnorm_sq_integral / grad_sq_integral);
    }
    return obs;
}

static void save_qtensor_field(const fs::path& path, const std::vector<QTensor>& field, const BulkLdgParams& params) {
    std::ofstream handle(path);
    handle << std::setprecision(16);
    handle << "# x y z Qxx Qxy Qxz Qyy Qyz S nx ny nz\n";
    for (int z = 0; z < params.Nz; ++z) {
        for (int y = 0; y < params.Ny; ++y) {
            for (int x = 0; x < params.Nx; ++x) {
                const QTensor& q = field[flatten_host(x, y, z, params.Nx, params.Ny, params.Nz)];
                const NematicField nf = calculateNematicFieldDevice(q);
                handle << x << ' ' << y << ' ' << z << ' '
                       << q.Qxx << ' ' << q.Qxy << ' ' << q.Qxz << ' ' << q.Qyy << ' ' << q.Qyz << ' '
                       << nf.S << ' ' << nf.nx << ' ' << nf.ny << ' ' << nf.nz << '\n';
            }
        }
    }
}

static void write_summary(
    const fs::path& path,
    const BulkLdgParams& params,
    double dt_limit,
    double T_NI,
    double S_eq_low,
    const BulkLdgObservables& final_observables
) {
    std::ofstream handle(path);
    handle << std::setprecision(16);
    handle << "Nx=" << params.Nx << '\n';
    handle << "Ny=" << params.Ny << '\n';
    handle << "Nz=" << params.Nz << '\n';
    handle << "dt=" << params.dt << '\n';
    handle << "dt_limit_estimate=" << dt_limit << '\n';
    handle << "T_high=" << params.T_high << '\n';
    handle << "T_low=" << params.T_low << '\n';
    handle << "Tc_KZ=" << params.Tc_KZ << '\n';
    handle << "T_NI_estimate=" << T_NI << '\n';
    handle << "S_eq_low=" << S_eq_low << '\n';
    handle << "bulk_modechoice=" << params.bulk_modechoice << '\n';
    handle << "defects_S_threshold=" << params.defects_S_threshold << '\n';
    handle << "defects_charge_cutoff=" << params.defects_charge_cutoff << '\n';
    handle << "pre_equil_iters=" << params.pre_equil_iters << '\n';
    handle << "ramp_iters=" << params.ramp_iters << '\n';
    handle << "total_iters=" << params.total_iters << '\n';
    handle << "final_avg_S=" << final_observables.avg_S << '\n';
    handle << "final_max_S=" << final_observables.max_S << '\n';
    handle << "final_defect_density_per_plaquette=" << final_observables.defect_density_per_plaquette << '\n';
    handle << "final_defect_line_density=" << final_observables.defect_line_density << '\n';
    handle << "final_xi_grad_proxy=" << final_observables.xi_grad_proxy << '\n';
}

static void print_cli_help() {
    std::cout
        << "KZM_bulk_ldg_cuda options:\n"
        << "  --config <path>       Run non-interactively using key=value config file\n"
        << "  --help                Show this help\n";
}

int main(int argc, char** argv) {
    try {
        std::string config_path;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i] ? argv[i] : "";
            if (arg == "--help" || arg == "-h") {
                print_cli_help();
                return 0;
            }
            if (arg == "--config" && i + 1 < argc) {
                config_path = argv[++i];
                continue;
            }
        }

        const auto cfg = parse_kv_config_file(config_path);
        BulkLdgParams params;
        params.Nx = cfg_get_value<int>(cfg, "Nx", params.Nx);
        params.Ny = cfg_get_value<int>(cfg, "Ny", params.Ny);
        params.Nz = cfg_get_value<int>(cfg, "Nz", params.Nz);
        params.dx = cfg_get_value<double>(cfg, "dx", params.dx);
        params.dy = cfg_get_value<double>(cfg, "dy", params.dy);
        params.dz = cfg_get_value<double>(cfg, "dz", params.dz);
        params.a = cfg_get_value<double>(cfg, "a", params.a);
        params.b = cfg_get_value<double>(cfg, "b", params.b);
        params.c = cfg_get_value<double>(cfg, "c", params.c);
        params.T_star = cfg_get_value<double>(cfg, "T_star", params.T_star);
        params.bulk_modechoice = cfg_get_value<int>(cfg, "bulk_modechoice", params.bulk_modechoice);
        params.kappa = cfg_get_value<double>(cfg, "kappa", params.kappa);
        params.gamma = cfg_get_value<double>(cfg, "gamma", params.gamma);
        params.T_high = cfg_get_value<double>(cfg, "T_high", params.T_high);
        params.T_low = cfg_get_value<double>(cfg, "T_low", params.T_low);
        params.pre_equil_iters = cfg_get_value<int>(cfg, "pre_equil_iters", params.pre_equil_iters);
        params.ramp_iters = cfg_get_value<int>(cfg, "ramp_iters", params.ramp_iters);
        params.total_iters = cfg_get_value<int>(cfg, "total_iters", params.total_iters);
        params.dt = cfg_get_value<double>(cfg, "dt", params.dt);
        params.init_noise_amplitude = cfg_get_value<double>(cfg, "init_noise_amplitude", params.init_noise_amplitude);
        params.noise_strength = cfg_get_value<double>(cfg, "noise_strength", params.noise_strength);
        params.defects_S_threshold = cfg_get_value<double>(cfg, "defects_S_threshold", params.defects_S_threshold);
        params.defects_charge_cutoff = cfg_get_value<double>(cfg, "defects_charge_cutoff", params.defects_charge_cutoff);
        params.q_norm_cap = cfg_get_value<double>(cfg, "q_norm_cap", params.q_norm_cap);
        params.random_seed = cfg_get_value<long long>(cfg, "random_seed", params.random_seed);
        params.logFreq = std::max(1, cfg_get_value<int>(cfg, "logFreq", params.logFreq));
        params.snapshot_mode = std::max(0, cfg_get_value<int>(cfg, "snapshot_mode", params.snapshot_mode));
        params.snapshotFreq = std::max(1, cfg_get_value<int>(cfg, "snapshotFreq", params.snapshotFreq));

        double A_ni = 0.0, B_ni = 0.0, C_ni = 0.0;
        bulk_ABC_from_convention(params.a, params.b, params.c, params.T_star, params.T_star, params.bulk_modechoice, A_ni, B_ni, C_ni);
        const double T_NI = T_from_A(bulk_A_for_transition_NI(B_ni, C_ni), params.a, params.T_star);
        params.Tc_KZ = cfg_get_value<double>(cfg, "Tc_KZ", std::isfinite(T_NI) ? T_NI : params.Tc_KZ);

        if (params.Nx <= 1 || params.Ny <= 1 || params.Nz <= 1) {
            throw std::runtime_error("Nx, Ny, and Nz must all be greater than 1 for the periodic bulk LdG branch");
        }
        if (params.dx <= 0.0 || params.dy <= 0.0 || params.dz <= 0.0) {
            throw std::runtime_error("dx, dy, and dz must be positive");
        }
        if (params.dt <= 0.0) {
            throw std::runtime_error("dt must be positive");
        }
        if (!(params.gamma > 0.0)) {
            throw std::runtime_error("gamma must be positive");
        }
        if (!(params.kappa >= 0.0)) {
            throw std::runtime_error("kappa must be non-negative");
        }
        if (params.total_iters < params.pre_equil_iters + params.ramp_iters) {
            throw std::runtime_error("total_iters must be at least pre_equil_iters + ramp_iters");
        }

        const std::string out_dir_name = cfg_get_string(cfg, "out_dir", "output_kzm_bulk_ldg");
        const bool overwrite_out_dir = cfg_get_value<bool>(cfg, "overwrite_out_dir", true);
        const fs::path out_dir = fs::absolute(out_dir_name);
        if (fs::exists(out_dir)) {
            if (!overwrite_out_dir) {
                throw std::runtime_error("Output directory exists and overwrite_out_dir=false: " + out_dir.string());
            }
            fs::remove_all(out_dir);
        }
        fs::create_directories(out_dir);

        double A_low = 0.0, B_low = 0.0, C_low = 0.0;
        bulk_ABC_from_convention(params.a, params.b, params.c, params.T_low, params.T_star, params.bulk_modechoice, A_low, B_low, C_low);
        const double S_eq_low = S_eq_uniaxial_from_ABC(A_low, B_low, C_low);
        const double dt_limit = estimate_stable_dt(params);

        std::cout << "KZM bulk intermediate branch (3D periodic bulk Landau-de Gennes)\n";
        std::cout << "Grid: " << params.Nx << "x" << params.Ny << "x" << params.Nz << '\n';
        std::cout << "T_high=" << params.T_high << ", T_low=" << params.T_low << ", Tc_KZ=" << params.Tc_KZ;
        if (std::isfinite(T_NI)) {
            std::cout << " (T_NI estimate ~ " << T_NI << ")";
        }
        std::cout << '\n';
        std::cout << "dt=" << params.dt;
        if (std::isfinite(dt_limit)) {
            std::cout << " (rough explicit limit ~ " << dt_limit << ")";
            if (params.dt > dt_limit) {
                std::cout << " [warning: dt exceeds the rough explicit limit]";
            }
        }
        std::cout << '\n';

        unsigned long long seed = 0ull;
        if (params.random_seed < 0) {
            std::random_device rd;
            seed = (static_cast<unsigned long long>(rd()) << 32) ^ static_cast<unsigned long long>(rd());
        } else {
            seed = static_cast<unsigned long long>(params.random_seed);
        }
        std::cout << "random_seed=" << seed << '\n';

        const size_t site_count = static_cast<size_t>(params.Nx) * static_cast<size_t>(params.Ny) * static_cast<size_t>(params.Nz);
        std::vector<QTensor> host_field(site_count);
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> gaussian(0.0, 1.0);
        for (QTensor& q : host_field) {
            q.Qxx = params.init_noise_amplitude * gaussian(rng);
            q.Qxy = params.init_noise_amplitude * gaussian(rng);
            q.Qxz = params.init_noise_amplitude * gaussian(rng);
            q.Qyy = params.init_noise_amplitude * gaussian(rng);
            q.Qyz = params.init_noise_amplitude * gaussian(rng);
        }

        QTensor* d_current = nullptr;
        QTensor* d_next = nullptr;
        CUDA_CHECK(cudaMalloc(&d_current, site_count * sizeof(QTensor)));
        CUDA_CHECK(cudaMalloc(&d_next, site_count * sizeof(QTensor)));
        CUDA_CHECK(cudaMemcpy(d_current, host_field.data(), site_count * sizeof(QTensor), cudaMemcpyHostToDevice));

        std::ofstream quench_log(out_dir / "quench_log.dat");
        quench_log << std::setprecision(16);
        quench_log
            << "iteration,time_s,T_K,dt_s,bulk,elastic,anchoring,total,radiality,avg_S,max_S,"
            << "defect_density_per_plaquette,defect_plaquettes_used,xi_grad_proxy,defect_line_density\n";

        const int block_size = 256;
        const int grid_size = static_cast<int>((site_count + static_cast<size_t>(block_size) - 1) / static_cast<size_t>(block_size));
        BulkLdgObservables final_observables;

        auto should_log = [&](int iter) {
            return iter == 0 || iter == params.total_iters || (iter % params.logFreq == 0);
        };
        auto should_snapshot = [&](int iter) {
            return params.snapshot_mode > 0 && (iter == params.total_iters || iter % params.snapshotFreq == 0);
        };

        for (int iter = 0; iter <= params.total_iters; ++iter) {
            const double temperature = compute_temperature(params, iter);
            double A = 0.0, B = 0.0, C = 0.0;
            bulk_ABC_from_convention(params.a, params.b, params.c, temperature, params.T_star, params.bulk_modechoice, A, B, C);

            if (should_log(iter) || should_snapshot(iter)) {
                CUDA_CHECK(cudaMemcpy(host_field.data(), d_current, site_count * sizeof(QTensor), cudaMemcpyDeviceToHost));
                final_observables = compute_observables(host_field, params, A, B, C);
                if (should_log(iter)) {
                    quench_log
                        << iter << ','
                        << params.dt * static_cast<double>(iter) << ','
                        << temperature << ','
                        << params.dt << ','
                        << final_observables.bulk << ','
                        << final_observables.elastic << ','
                        << 0.0 << ','
                        << final_observables.total << ','
                        << std::numeric_limits<double>::quiet_NaN() << ','
                        << final_observables.avg_S << ','
                        << final_observables.max_S << ','
                        << final_observables.defect_density_per_plaquette << ','
                        << final_observables.defect_plaquettes_used << ','
                        << final_observables.xi_grad_proxy << ','
                        << final_observables.defect_line_density << '\n';
                }
                if (should_snapshot(iter)) {
                    std::ostringstream name;
                    name << "q_tensor_iter_" << std::setw(6) << std::setfill('0') << iter << ".dat";
                    save_qtensor_field(out_dir / name.str(), host_field, params);
                }
            }

            if (iter == params.total_iters) {
                break;
            }

            const double noise_sigma = (params.noise_strength > 0.0)
                ? std::sqrt(2.0 * (1.0 / params.gamma) * params.noise_strength * params.dt)
                : 0.0;
            updateBulkLdgKernel<<<grid_size, block_size>>>(
                d_current,
                d_next,
                params,
                A,
                B,
                C,
                noise_sigma,
                seed,
                iter
            );
            CUDA_CHECK(cudaGetLastError());
            std::swap(d_current, d_next);
        }

        CUDA_CHECK(cudaMemcpy(host_field.data(), d_current, site_count * sizeof(QTensor), cudaMemcpyDeviceToHost));
        save_qtensor_field(out_dir / "q_tensor_final.dat", host_field, params);
        write_summary(out_dir / "run_summary.txt", params, dt_limit, T_NI, S_eq_low, final_observables);

        CUDA_CHECK(cudaFree(d_current));
        CUDA_CHECK(cudaFree(d_next));

        std::cout << "Final avg_S=" << final_observables.avg_S
                  << ", defect_line_density=" << final_observables.defect_line_density
                  << ", xi_grad_proxy=" << final_observables.xi_grad_proxy << '\n';
        std::cout << "Output directory: " << out_dir << '\n';
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "KZM_bulk_ldg_cuda ERROR: " << exc.what() << std::endl;
        return 1;
    }
}