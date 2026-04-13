#include "KZM_prooving_ground.cuh"

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
            oss__ << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "         \
                  << cudaGetErrorString(err__);                                          \
            throw std::runtime_error(oss__.str());                                       \
        }                                                                                \
    } while (0)

struct XYObservables {
    double bulk = std::numeric_limits<double>::quiet_NaN();
    double elastic = std::numeric_limits<double>::quiet_NaN();
    double total = std::numeric_limits<double>::quiet_NaN();
    double avg_amp = std::numeric_limits<double>::quiet_NaN();
    double max_amp = std::numeric_limits<double>::quiet_NaN();
    double magnetization_abs = std::numeric_limits<double>::quiet_NaN();
    double defect_density_per_plaquette = std::numeric_limits<double>::quiet_NaN();
    double vortex_line_density = std::numeric_limits<double>::quiet_NaN();
    double xi_grad_proxy = std::numeric_limits<double>::quiet_NaN();
    long long defect_plaquettes_used = 0;
    long long nonzero_vortex_plaquettes = 0;
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
    return static_cast<size_t>(x) + static_cast<size_t>(Nx) * (static_cast<size_t>(y) + static_cast<size_t>(Ny) * static_cast<size_t>(z));
}

__host__ __device__ static inline size_t flatten_device(int x, int y, int z, int Nx, int Ny, int Nz) {
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
    const double angle = 2.0 * KZM_PG_PI * u2;
    g0 = radius * cos(angle);
    g1 = radius * sin(angle);
}

__global__ void updateXYFieldKernel(
    const XYField* current,
    XYField* next,
    ProvingGroundParams params,
    double reduced_mass,
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

    const XYField center = current[idx];
    const XYField plus_x = current[flatten_device(xp, y, z, params.Nx, params.Ny, params.Nz)];
    const XYField minus_x = current[flatten_device(xm, y, z, params.Nx, params.Ny, params.Nz)];
    const XYField plus_y = current[flatten_device(x, yp, z, params.Nx, params.Ny, params.Nz)];
    const XYField minus_y = current[flatten_device(x, ym, z, params.Nx, params.Ny, params.Nz)];
    const XYField plus_z = current[flatten_device(x, y, zp, params.Nx, params.Ny, params.Nz)];
    const XYField minus_z = current[flatten_device(x, y, zm, params.Nx, params.Ny, params.Nz)];

    const double inv_dx2 = 1.0 / (params.dx * params.dx);
    const double inv_dy2 = 1.0 / (params.dy * params.dy);
    const double inv_dz2 = 1.0 / (params.dz * params.dz);

    const double lap_real =
        (plus_x.real - 2.0 * center.real + minus_x.real) * inv_dx2 +
        (plus_y.real - 2.0 * center.real + minus_y.real) * inv_dy2 +
        (plus_z.real - 2.0 * center.real + minus_z.real) * inv_dz2;
    const double lap_imag =
        (plus_x.imag - 2.0 * center.imag + minus_x.imag) * inv_dx2 +
        (plus_y.imag - 2.0 * center.imag + minus_y.imag) * inv_dy2 +
        (plus_z.imag - 2.0 * center.imag + minus_z.imag) * inv_dz2;

    const double amp2 = center.real * center.real + center.imag * center.imag;
    const double drift_real = params.mobility * (params.kappa * lap_real - reduced_mass * center.real - params.beta * amp2 * center.real);
    const double drift_imag = params.mobility * (params.kappa * lap_imag - reduced_mass * center.imag - params.beta * amp2 * center.imag);

    double noise_real = 0.0;
    double noise_imag = 0.0;
    if (noise_sigma > 0.0) {
        unsigned long long state = base_seed;
        state ^= (0x9E3779B97F4A7C15ull * (idx + 1ull));
        state ^= (0xBF58476D1CE4E5B9ull * static_cast<unsigned long long>(iteration + 1));
        normal_pair(state, noise_real, noise_imag);
        noise_real *= noise_sigma;
        noise_imag *= noise_sigma;
    }

    next[idx].real = center.real + params.dt * drift_real + noise_real;
    next[idx].imag = center.imag + params.dt * drift_imag + noise_imag;
}

static double wrapped_phase_difference(double delta) {
    while (delta <= -KZM_PG_PI) {
        delta += 2.0 * KZM_PG_PI;
    }
    while (delta > KZM_PG_PI) {
        delta -= 2.0 * KZM_PG_PI;
    }
    return delta;
}

static double field_phase(const XYField& value) {
    return std::atan2(value.imag, value.real);
}

static double compute_temperature(const ProvingGroundParams& params, int iteration) {
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

static double reduced_mass_from_temperature(const ProvingGroundParams& params, double temperature) {
    return params.alpha * (temperature - params.Tc) / params.Tc;
}

static double estimate_stable_dt(const ProvingGroundParams& params) {
    const double inv_dx2 = 1.0 / (params.dx * params.dx);
    const double inv_dy2 = 1.0 / (params.dy * params.dy);
    const double inv_dz2 = 1.0 / (params.dz * params.dz);
    const double laplacian_scale = 2.0 * params.kappa * (inv_dx2 + inv_dy2 + inv_dz2);
    const double r_high = std::abs(reduced_mass_from_temperature(params, params.T_high));
    const double r_low = reduced_mass_from_temperature(params, params.T_low);
    const double amp_eq_low = (params.beta > 0.0 && r_low < 0.0) ? std::sqrt(-r_low / params.beta) : 0.0;
    const double nonlinear_scale = 3.0 * params.beta * amp_eq_low * amp_eq_low;
    const double local_scale = std::max(r_high, std::abs(r_low)) + nonlinear_scale;
    const double stiffness = params.mobility * (laplacian_scale + local_scale);
    if (stiffness <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return 0.2 / stiffness;
}

static XYObservables compute_observables(
    const std::vector<XYField>& field,
    const ProvingGroundParams& params,
    double reduced_mass
) {
    XYObservables obs;

    const double cell_volume = params.dx * params.dy * params.dz;
    const double total_volume = cell_volume * static_cast<double>(params.Nx) * static_cast<double>(params.Ny) * static_cast<double>(params.Nz);
    const double inv_dx = 1.0 / params.dx;
    const double inv_dy = 1.0 / params.dy;
    const double inv_dz = 1.0 / params.dz;

    double bulk_sum = 0.0;
    double elastic_sum = 0.0;
    double grad_sq_integral = 0.0;
    double psi_sq_integral = 0.0;
    double amp_sum = 0.0;
    double max_amp = 0.0;
    double magnetization_real = 0.0;
    double magnetization_imag = 0.0;

    for (int z = 0; z < params.Nz; ++z) {
        const int zp = (z + 1) % params.Nz;
        for (int y = 0; y < params.Ny; ++y) {
            const int yp = (y + 1) % params.Ny;
            for (int x = 0; x < params.Nx; ++x) {
                const int xp = (x + 1) % params.Nx;
                const XYField& center = field[flatten_host(x, y, z, params.Nx, params.Ny, params.Nz)];
                const XYField& plus_x = field[flatten_host(xp, y, z, params.Nx, params.Ny, params.Nz)];
                const XYField& plus_y = field[flatten_host(x, yp, z, params.Nx, params.Ny, params.Nz)];
                const XYField& plus_z = field[flatten_host(x, y, zp, params.Nx, params.Ny, params.Nz)];

                const double amp2 = center.real * center.real + center.imag * center.imag;
                const double amp = std::sqrt(amp2);

                amp_sum += amp;
                max_amp = std::max(max_amp, amp);
                magnetization_real += center.real;
                magnetization_imag += center.imag;
                psi_sq_integral += amp2 * cell_volume;

                bulk_sum += (0.5 * reduced_mass * amp2 + 0.25 * params.beta * amp2 * amp2) * cell_volume;

                const double diff_x_real = (plus_x.real - center.real) * inv_dx;
                const double diff_x_imag = (plus_x.imag - center.imag) * inv_dx;
                const double diff_y_real = (plus_y.real - center.real) * inv_dy;
                const double diff_y_imag = (plus_y.imag - center.imag) * inv_dy;
                const double diff_z_real = (plus_z.real - center.real) * inv_dz;
                const double diff_z_imag = (plus_z.imag - center.imag) * inv_dz;

                const double grad_sq =
                    diff_x_real * diff_x_real + diff_x_imag * diff_x_imag +
                    diff_y_real * diff_y_real + diff_y_imag * diff_y_imag +
                    diff_z_real * diff_z_real + diff_z_imag * diff_z_imag;
                grad_sq_integral += grad_sq * cell_volume;
                elastic_sum += 0.5 * params.kappa * grad_sq * cell_volume;
            }
        }
    }

    long long count_xy = 0;
    long long count_yz = 0;
    long long count_xz = 0;

    for (int z = 0; z < params.Nz; ++z) {
        const int zp = (z + 1) % params.Nz;
        for (int y = 0; y < params.Ny; ++y) {
            const int yp = (y + 1) % params.Ny;
            for (int x = 0; x < params.Nx; ++x) {
                const int xp = (x + 1) % params.Nx;

                const XYField& f000 = field[flatten_host(x, y, z, params.Nx, params.Ny, params.Nz)];
                const XYField& f100 = field[flatten_host(xp, y, z, params.Nx, params.Ny, params.Nz)];
                const XYField& f110 = field[flatten_host(xp, yp, z, params.Nx, params.Ny, params.Nz)];
                const XYField& f010 = field[flatten_host(x, yp, z, params.Nx, params.Ny, params.Nz)];
                const XYField& f001 = field[flatten_host(x, y, zp, params.Nx, params.Ny, params.Nz)];
                const XYField& f101 = field[flatten_host(xp, y, zp, params.Nx, params.Ny, params.Nz)];
                const XYField& f011 = field[flatten_host(x, yp, zp, params.Nx, params.Ny, params.Nz)];

                const double a000 = std::sqrt(f000.real * f000.real + f000.imag * f000.imag);
                const double a100 = std::sqrt(f100.real * f100.real + f100.imag * f100.imag);
                const double a110 = std::sqrt(f110.real * f110.real + f110.imag * f110.imag);
                const double a010 = std::sqrt(f010.real * f010.real + f010.imag * f010.imag);
                const double a001 = std::sqrt(f001.real * f001.real + f001.imag * f001.imag);
                const double a101 = std::sqrt(f101.real * f101.real + f101.imag * f101.imag);
                const double a011 = std::sqrt(f011.real * f011.real + f011.imag * f011.imag);

                if (
                    a000 >= params.defect_amp_threshold &&
                    a100 >= params.defect_amp_threshold &&
                    a110 >= params.defect_amp_threshold &&
                    a010 >= params.defect_amp_threshold
                ) {
                    const double p000 = field_phase(f000);
                    const double p100 = field_phase(f100);
                    const double p110 = field_phase(f110);
                    const double p010 = field_phase(f010);
                    const double winding_xy =
                        wrapped_phase_difference(p100 - p000) +
                        wrapped_phase_difference(p110 - p100) +
                        wrapped_phase_difference(p010 - p110) +
                        wrapped_phase_difference(p000 - p010);
                    count_xy += std::llabs(std::llround(winding_xy / (2.0 * KZM_PG_PI)));
                }

                if (
                    a000 >= params.defect_amp_threshold &&
                    a010 >= params.defect_amp_threshold &&
                    a011 >= params.defect_amp_threshold &&
                    a001 >= params.defect_amp_threshold
                ) {
                    const double p000 = field_phase(f000);
                    const double p010 = field_phase(f010);
                    const double p011 = field_phase(f011);
                    const double p001 = field_phase(f001);
                    const double winding_yz =
                        wrapped_phase_difference(p010 - p000) +
                        wrapped_phase_difference(p011 - p010) +
                        wrapped_phase_difference(p001 - p011) +
                        wrapped_phase_difference(p000 - p001);
                    count_yz += std::llabs(std::llround(winding_yz / (2.0 * KZM_PG_PI)));
                }

                if (
                    a001 >= params.defect_amp_threshold &&
                    a101 >= params.defect_amp_threshold &&
                    a100 >= params.defect_amp_threshold &&
                    a000 >= params.defect_amp_threshold
                ) {
                    const double p001 = field_phase(f001);
                    const double p101 = field_phase(f101);
                    const double p100 = field_phase(f100);
                    const double p000 = field_phase(f000);
                    const double winding_xz =
                        wrapped_phase_difference(p101 - p001) +
                        wrapped_phase_difference(p100 - p101) +
                        wrapped_phase_difference(p000 - p100) +
                        wrapped_phase_difference(p001 - p000);
                    count_xz += std::llabs(std::llround(winding_xz / (2.0 * KZM_PG_PI)));
                }
            }
        }
    }

    long long total_plaquettes = 0;
    for (int z = 0; z < params.Nz; ++z) {
        const int zp = (z + 1) % params.Nz;
        for (int y = 0; y < params.Ny; ++y) {
            const int yp = (y + 1) % params.Ny;
            for (int x = 0; x < params.Nx; ++x) {
                const int xp = (x + 1) % params.Nx;
                const XYField& f000 = field[flatten_host(x, y, z, params.Nx, params.Ny, params.Nz)];
                const XYField& f100 = field[flatten_host(xp, y, z, params.Nx, params.Ny, params.Nz)];
                const XYField& f110 = field[flatten_host(xp, yp, z, params.Nx, params.Ny, params.Nz)];
                const XYField& f010 = field[flatten_host(x, yp, z, params.Nx, params.Ny, params.Nz)];
                const XYField& f001 = field[flatten_host(x, y, zp, params.Nx, params.Ny, params.Nz)];
                const XYField& f101 = field[flatten_host(xp, y, zp, params.Nx, params.Ny, params.Nz)];
                const XYField& f011 = field[flatten_host(x, yp, zp, params.Nx, params.Ny, params.Nz)];

                const double a000 = std::sqrt(f000.real * f000.real + f000.imag * f000.imag);
                const double a100 = std::sqrt(f100.real * f100.real + f100.imag * f100.imag);
                const double a110 = std::sqrt(f110.real * f110.real + f110.imag * f110.imag);
                const double a010 = std::sqrt(f010.real * f010.real + f010.imag * f010.imag);
                const double a001 = std::sqrt(f001.real * f001.real + f001.imag * f001.imag);
                const double a101 = std::sqrt(f101.real * f101.real + f101.imag * f101.imag);
                const double a011 = std::sqrt(f011.real * f011.real + f011.imag * f011.imag);

                if (a000 >= params.defect_amp_threshold && a100 >= params.defect_amp_threshold && a110 >= params.defect_amp_threshold && a010 >= params.defect_amp_threshold) {
                    ++total_plaquettes;
                }
                if (a000 >= params.defect_amp_threshold && a010 >= params.defect_amp_threshold && a011 >= params.defect_amp_threshold && a001 >= params.defect_amp_threshold) {
                    ++total_plaquettes;
                }
                if (a001 >= params.defect_amp_threshold && a101 >= params.defect_amp_threshold && a100 >= params.defect_amp_threshold && a000 >= params.defect_amp_threshold) {
                    ++total_plaquettes;
                }
            }
        }
    }
    const long long total_nonzero = count_xy + count_yz + count_xz;
    const double vortex_line_length =
        static_cast<double>(count_xy) * params.dz +
        static_cast<double>(count_yz) * params.dx +
        static_cast<double>(count_xz) * params.dy;

    double sum0_real = magnetization_real;
    double sum0_imag = magnetization_imag;

    const double site_count = static_cast<double>(params.Nx) * static_cast<double>(params.Ny) * static_cast<double>(params.Nz);
    obs.bulk = bulk_sum / total_volume;
    obs.elastic = elastic_sum / total_volume;
    obs.total = obs.bulk + obs.elastic;
    obs.avg_amp = amp_sum / site_count;
    obs.max_amp = max_amp;
    obs.magnetization_abs = std::sqrt(sum0_real * sum0_real + sum0_imag * sum0_imag) / site_count;
    obs.defect_plaquettes_used = total_plaquettes;
    obs.nonzero_vortex_plaquettes = total_nonzero;
    obs.defect_density_per_plaquette = (total_plaquettes > 0) ? static_cast<double>(total_nonzero) / static_cast<double>(total_plaquettes) : std::numeric_limits<double>::quiet_NaN();
    obs.vortex_line_density = (total_volume > 0.0) ? vortex_line_length / total_volume : std::numeric_limits<double>::quiet_NaN();
    if (grad_sq_integral > 0.0 && psi_sq_integral > 0.0) {
        obs.xi_grad_proxy = std::sqrt(psi_sq_integral / grad_sq_integral);
    }
    return obs;
}

static void save_xy_field(const fs::path& path, const std::vector<XYField>& field, const ProvingGroundParams& params) {
    std::ofstream handle(path);
    handle << std::setprecision(16);
    handle << "# x y z psi_real psi_imag abs_psi phase\n";
    for (int z = 0; z < params.Nz; ++z) {
        for (int y = 0; y < params.Ny; ++y) {
            for (int x = 0; x < params.Nx; ++x) {
                const XYField& value = field[flatten_host(x, y, z, params.Nx, params.Ny, params.Nz)];
                const double amp = std::sqrt(value.real * value.real + value.imag * value.imag);
                handle << x << ' ' << y << ' ' << z << ' '
                       << value.real << ' ' << value.imag << ' '
                       << amp << ' ' << std::atan2(value.imag, value.real) << '\n';
            }
        }
    }
}

static void write_summary(
    const fs::path& path,
    const ProvingGroundParams& params,
    double dt_limit,
    const XYObservables& final_observables
) {
    std::ofstream handle(path);
    handle << std::setprecision(16);
    handle << "Nx=" << params.Nx << '\n';
    handle << "Ny=" << params.Ny << '\n';
    handle << "Nz=" << params.Nz << '\n';
    handle << "dt=" << params.dt << '\n';
    handle << "dt_limit_estimate=" << dt_limit << '\n';
    handle << "Tc=" << params.Tc << '\n';
    handle << "Tc_KZ=" << params.Tc_KZ << '\n';
    handle << "T_high=" << params.T_high << '\n';
    handle << "T_low=" << params.T_low << '\n';
    handle << "defect_amp_threshold=" << params.defect_amp_threshold << '\n';
    handle << "pre_equil_iters=" << params.pre_equil_iters << '\n';
    handle << "ramp_iters=" << params.ramp_iters << '\n';
    handle << "total_iters=" << params.total_iters << '\n';
    handle << "final_avg_amp=" << final_observables.avg_amp << '\n';
    handle << "final_max_amp=" << final_observables.max_amp << '\n';
    handle << "final_magnetization_abs=" << final_observables.magnetization_abs << '\n';
    handle << "final_vortex_plaquette_density=" << final_observables.defect_density_per_plaquette << '\n';
    handle << "final_vortex_line_density=" << final_observables.vortex_line_density << '\n';
    handle << "final_xi_grad_proxy=" << final_observables.xi_grad_proxy << '\n';
}

static void print_cli_help() {
    std::cout
        << "KZM_prooving_ground_cuda options:\n"
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
        ProvingGroundParams params;
        params.Nx = cfg_get_value<int>(cfg, "Nx", params.Nx);
        params.Ny = cfg_get_value<int>(cfg, "Ny", params.Ny);
        params.Nz = cfg_get_value<int>(cfg, "Nz", params.Nz);
        params.dx = cfg_get_value<double>(cfg, "dx", params.dx);
        params.dy = cfg_get_value<double>(cfg, "dy", params.dy);
        params.dz = cfg_get_value<double>(cfg, "dz", params.dz);
        params.mobility = cfg_get_value<double>(cfg, "mobility", params.mobility);
        params.alpha = cfg_get_value<double>(cfg, "alpha", params.alpha);
        params.beta = cfg_get_value<double>(cfg, "beta", params.beta);
        params.kappa = cfg_get_value<double>(cfg, "kappa", params.kappa);
        params.Tc = cfg_get_value<double>(cfg, "Tc", params.Tc);
        params.Tc_KZ = cfg_get_value<double>(cfg, "Tc_KZ", params.Tc);
        params.T_high = cfg_get_value<double>(cfg, "T_high", params.T_high);
        params.T_low = cfg_get_value<double>(cfg, "T_low", params.T_low);
        params.pre_equil_iters = cfg_get_value<int>(cfg, "pre_equil_iters", params.pre_equil_iters);
        params.ramp_iters = cfg_get_value<int>(cfg, "ramp_iters", params.ramp_iters);
        params.total_iters = cfg_get_value<int>(cfg, "total_iters", params.total_iters);
        params.dt = cfg_get_value<double>(cfg, "dt", params.dt);
        params.init_amplitude = cfg_get_value<double>(cfg, "init_amplitude", params.init_amplitude);
        params.noise_strength = cfg_get_value<double>(cfg, "noise_strength", params.noise_strength);
        params.defect_amp_threshold = cfg_get_value<double>(cfg, "defect_amp_threshold", params.defect_amp_threshold);
        params.random_seed = cfg_get_value<long long>(cfg, "random_seed", params.random_seed);
        params.logFreq = std::max(1, cfg_get_value<int>(cfg, "logFreq", params.logFreq));
        params.snapshot_mode = std::max(0, cfg_get_value<int>(cfg, "snapshot_mode", params.snapshot_mode));
        params.snapshotFreq = std::max(1, cfg_get_value<int>(cfg, "snapshotFreq", params.snapshotFreq));

        if (params.Nx <= 1 || params.Ny <= 1 || params.Nz <= 1) {
            throw std::runtime_error("Nx, Ny, and Nz must all be greater than 1 for the periodic 3D proving ground");
        }
        if (params.dx <= 0.0 || params.dy <= 0.0 || params.dz <= 0.0) {
            throw std::runtime_error("dx, dy, and dz must be positive");
        }
        if (params.dt <= 0.0) {
            throw std::runtime_error("dt must be positive");
        }
        if (params.beta <= 0.0) {
            throw std::runtime_error("beta must be positive for a stable XY quartic potential");
        }
        if (params.total_iters < params.pre_equil_iters + params.ramp_iters) {
            throw std::runtime_error("total_iters must be at least pre_equil_iters + ramp_iters");
        }

        const std::string out_dir_name = cfg_get_string(cfg, "out_dir", "output_kzm_prooving_ground");
        const bool overwrite_out_dir = cfg_get_value<bool>(cfg, "overwrite_out_dir", true);
        const fs::path out_dir = fs::absolute(out_dir_name);

        if (fs::exists(out_dir)) {
            if (!overwrite_out_dir) {
                throw std::runtime_error("Output directory exists and overwrite_out_dir=false: " + out_dir.string());
            }
            fs::remove_all(out_dir);
        }
        fs::create_directories(out_dir);

        const double dt_limit = estimate_stable_dt(params);
        std::cout << "KZM proving ground (3D periodic XY, Model-A TDGL)\n";
        std::cout << "Grid: " << params.Nx << "x" << params.Ny << "x" << params.Nz << '\n';
        std::cout << "T_high=" << params.T_high << ", T_low=" << params.T_low << ", Tc=" << params.Tc << '\n';
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
        std::vector<XYField> host_field(site_count);
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> gaussian(0.0, 1.0);
        for (XYField& value : host_field) {
            value.real = params.init_amplitude * gaussian(rng);
            value.imag = params.init_amplitude * gaussian(rng);
        }

        XYField* d_current = nullptr;
        XYField* d_next = nullptr;
        CUDA_CHECK(cudaMalloc(&d_current, site_count * sizeof(XYField)));
        CUDA_CHECK(cudaMalloc(&d_next, site_count * sizeof(XYField)));
        CUDA_CHECK(cudaMemcpy(d_current, host_field.data(), site_count * sizeof(XYField), cudaMemcpyHostToDevice));

        std::ofstream quench_log(out_dir / "quench_log.dat");
        quench_log << std::setprecision(16);
        quench_log
            << "iteration,time_s,T_K,dt_s,bulk,elastic,anchoring,total,radiality,avg_S,max_S,"
            << "defect_density_per_plaquette,defect_plaquettes_used,xi_grad_proxy,vortex_line_density,magnetization_abs\n";

        const int block_size = 256;
        const int grid_size = static_cast<int>((site_count + static_cast<size_t>(block_size) - 1) / static_cast<size_t>(block_size));
        XYObservables final_observables;

        auto should_log = [&](int iter) {
            return iter == 0 || iter == params.total_iters || (iter % params.logFreq == 0);
        };
        auto should_snapshot = [&](int iter) {
            return params.snapshot_mode > 0 && (iter == params.total_iters || iter % params.snapshotFreq == 0);
        };

        for (int iter = 0; iter <= params.total_iters; ++iter) {
            const double temperature = compute_temperature(params, iter);
            const double reduced_mass = reduced_mass_from_temperature(params, temperature);

            if (should_log(iter) || should_snapshot(iter)) {
                CUDA_CHECK(cudaMemcpy(host_field.data(), d_current, site_count * sizeof(XYField), cudaMemcpyDeviceToHost));
                const XYObservables obs = compute_observables(host_field, params, reduced_mass);
                final_observables = obs;

                if (should_log(iter)) {
                    quench_log
                        << iter << ','
                        << params.dt * static_cast<double>(iter) << ','
                        << temperature << ','
                        << params.dt << ','
                        << obs.bulk << ','
                        << obs.elastic << ','
                        << 0.0 << ','
                        << obs.total << ','
                        << std::numeric_limits<double>::quiet_NaN() << ','
                        << obs.avg_amp << ','
                        << obs.max_amp << ','
                        << obs.defect_density_per_plaquette << ','
                        << obs.defect_plaquettes_used << ','
                        << obs.xi_grad_proxy << ','
                        << obs.vortex_line_density << ','
                        << obs.magnetization_abs << '\n';
                }

                if (should_snapshot(iter)) {
                    std::ostringstream name;
                    name << "xy_field_iter_" << std::setw(6) << std::setfill('0') << iter << ".dat";
                    save_xy_field(out_dir / name.str(), host_field, params);
                }
            }

            if (iter == params.total_iters) {
                break;
            }

            const double noise_sigma = (params.noise_strength > 0.0)
                ? std::sqrt(2.0 * params.mobility * params.noise_strength * params.dt)
                : 0.0;
            updateXYFieldKernel<<<grid_size, block_size>>>(
                d_current,
                d_next,
                params,
                reduced_mass,
                noise_sigma,
                seed,
                iter
            );
            CUDA_CHECK(cudaGetLastError());
            std::swap(d_current, d_next);
        }

        CUDA_CHECK(cudaMemcpy(host_field.data(), d_current, site_count * sizeof(XYField), cudaMemcpyDeviceToHost));
        save_xy_field(out_dir / "xy_field_final.dat", host_field, params);
        write_summary(out_dir / "run_summary.txt", params, dt_limit, final_observables);

        CUDA_CHECK(cudaFree(d_current));
        CUDA_CHECK(cudaFree(d_next));

        std::cout << "Final avg_amp=" << final_observables.avg_amp
                  << ", vortex_line_density=" << final_observables.vortex_line_density
                  << ", xi_grad_proxy=" << final_observables.xi_grad_proxy << '\n';
        std::cout << "Output directory: " << out_dir << '\n';
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "KZM_prooving_ground_cuda ERROR: " << exc.what() << std::endl;
        return 1;
    }
}