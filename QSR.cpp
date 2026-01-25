#include "QSR.h"

namespace fs = std::filesystem;
#pragma warning(disable: 4996) 

static double computeTransitionTemperature(double a, double b, double c, double T_star, int modechoice);
static void addIsotropicNoise(std::vector<QTensor>& Q, int Nx, int Ny, int Nz, double amplitude);

// Calculates the nematic field (director and order parameter) from a Q-tensor.
NematicField calculateNematicField(const QTensor& q_5comp) {
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
    // This can select the most negative eigenvalue in oblate/biaxial regions, producing negative S.
    // Shift: Q' = Q + alpha*I with alpha > -lambda_min; eigenvectors unchanged and all eigenvalues positive.
    double row1 = std::abs(q.Qxx) + std::abs(q.Qxy) + std::abs(q.Qxz);
    double row2 = std::abs(q.Qyx) + std::abs(q.Qyy) + std::abs(q.Qyz);
    double row3 = std::abs(q.Qzx) + std::abs(q.Qzy) + std::abs(q.Qzz);
    double alpha = std::max(row1, std::max(row2, row3)) + 1e-12;

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

// Compute local S (Simplified for Energy Calc if needed, though mostly redundant now)
double computeLocalS(const QTensor& q_5comp) {
    return calculateNematicField(q_5comp).S;
}

// Helper for robust derivatives
inline double get_deriv(const std::vector<double>& data, int idx, int stride, int max_idx, int curr_i, double inv_2d, double inv_d) {
    if (curr_i > 0 && curr_i < max_idx - 1) {
        return (data[idx + stride] - data[idx - stride]) * inv_2d;
    } else if (curr_i == 0) {
        return (data[idx + stride] - data[idx]) * inv_d; 
    } else {
        return (data[idx] - data[idx - stride]) * inv_d; 
    }
}

// Helper for Q-tensor component derivative
inline double get_Q_deriv(const std::vector<QTensor>& Q, int idx, int stride, int max_idx, int curr_i, double inv_2d, double inv_d, int comp_offset) {
    auto get_val = [&](int i) { return ((const double*)&Q[i])[comp_offset]; };
    if (curr_i > 0 && curr_i < max_idx - 1) {
        return (get_val(idx + stride) - get_val(idx - stride)) * inv_2d;
    } else if (curr_i == 0) {
        return (get_val(idx + stride) - get_val(idx)) * inv_d;
    } else {
        return (get_val(idx) - get_val(idx - stride)) * inv_d;
    }
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
    
    #pragma omp parallel
    {
        std::mt19937 gen(std::random_device{}() + omp_get_thread_num());
        std::uniform_real_distribution<double>dist_theta(0.0, PI);
        std::uniform_real_distribution<double> dist_phi(0.0, 2.0 * PI);

        #pragma omp for collapse(3)
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
}

// Standard 7-point Laplacian with boundary handling, optimized by separating interior and boundary calculations.
void computeLaplacian(const std::vector<QTensor>& Q, std::vector<QTensor>& laplacianQ, int Nx, int Ny, int Nz, double dx, double dy, double dz) {
    // Inverse squared steps
    double inv_dx2 = 1.0 / (dx * dx);
    double inv_dy2 = 1.0 / (dy * dy);
    double inv_dz2 = 1.0 / (dz * dz);
    auto idx = [&](int i, int j, int k) { return k + Nz * (j + Ny * i); };

    // --- 1. Interior Points ---
    // This loop is branch-free, making it ideal for vectorization.
    #pragma omp parallel for collapse(3)
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int k = 1; k < Nz - 1; ++k) {
                const int center_idx = idx(i, j, k);
                const QTensor& q_center = Q[center_idx];

                // Central difference for all three axes
                QTensor d2x = (Q[idx(i + 1, j, k)] - q_center * 2.0 + Q[idx(i - 1, j, k)]) * inv_dx2;
                QTensor d2y = (Q[idx(i, j + 1, k)] - q_center * 2.0 + Q[idx(i, j - 1, k)]) * inv_dy2;
                QTensor d2z = (Q[idx(i, j, k + 1)] - q_center * 2.0 + Q[idx(i, j, k - 1)]) * inv_dz2;

                laplacianQ[center_idx] = d2x + d2y + d2z;
            }
        }
    }

    // --- 2. Boundary Points ---
    // These loops handle the faces of the 3D grid, where the finite difference stencil changes.
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < Ny; ++j) {
        for (int k = 0; k < Nz; ++k) {
            // X-boundaries (i=0 and i=Nx-1)
            for (int i : {0, Nx - 1}) {
                const int center_idx = idx(i, j, k);
                const QTensor& q_center = Q[center_idx];
                QTensor d2x, d2y, d2z;

                // Y-derivative
                if (j > 0 && j < Ny - 1) d2y = (Q[idx(i, j + 1, k)] - q_center * 2.0 + Q[idx(i, j - 1, k)]) * inv_dy2;
                else if (j == 0)         d2y = (Q[idx(i, j + 2, k)] - Q[idx(i, j + 1, k)] * 2.0 + q_center) * inv_dy2;
                else /* j == Ny-1 */   d2y = (q_center - Q[idx(i, j - 1, k)] * 2.0 + Q[idx(i, j - 2, k)]) * inv_dy2;

                // Z-derivative
                if (k > 0 && k < Nz - 1) d2z = (Q[idx(i, j, k + 1)] - q_center * 2.0 + Q[idx(i, j, k - 1)]) * inv_dz2;
                else if (k == 0)         d2z = (Q[idx(i, j, k + 2)] - Q[idx(i, j, k + 1)] * 2.0 + q_center) * inv_dz2;
                else /* k == Nz-1 */   d2z = (q_center - Q[idx(i, j, k - 1)] * 2.0 + Q[idx(i, j, k - 2)]) * inv_dz2;
                
                // X-derivative (boundary condition)
                if (i == 0)           d2x = (Q[idx(i + 2, j, k)] - Q[idx(i + 1, j, k)] * 2.0 + q_center) * inv_dx2;
                else /* i == Nx-1 */ d2x = (q_center - Q[idx(i - 1, j, k)] * 2.0 + Q[idx(i - 2, j, k)]) * inv_dx2;

                laplacianQ[center_idx] = d2x + d2y + d2z;
            }
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < Nx-1; ++i) { // Skip corners already done in X-boundary loops
        for (int k = 0; k < Nz; ++k) {
            // Y-boundaries (j=0 and j=Ny-1)
            for (int j : {0, Ny - 1}) {
                const int center_idx = idx(i, j, k);
                const QTensor& q_center = Q[center_idx];
                QTensor d2x, d2y, d2z;

                // X-derivative (is always interior here)
                d2x = (Q[idx(i + 1, j, k)] - q_center * 2.0 + Q[idx(i - 1, j, k)]) * inv_dx2;

                // Z-derivative
                if (k > 0 && k < Nz - 1) d2z = (Q[idx(i, j, k + 1)] - q_center * 2.0 + Q[idx(i, j, k - 1)]) * inv_dz2;
                else if (k == 0)         d2z = (Q[idx(i, j, k + 2)] - Q[idx(i, j, k + 1)] * 2.0 + q_center) * inv_dz2;
                else /* k == Nz-1 */   d2z = (q_center - Q[idx(i, j, k - 1)] * 2.0 + Q[idx(i, j, k - 2)]) * inv_dz2;

                // Y-derivative (boundary condition)
                if (j == 0)           d2y = (Q[idx(i, j + 2, k)] - Q[idx(i, j + 1, k)] * 2.0 + q_center) * inv_dy2;
                else /* j == Ny-1 */ d2y = (q_center - Q[idx(i, j - 1, k)] * 2.0 + Q[idx(i, j - 2, k)]) * inv_dy2;

                laplacianQ[center_idx] = d2x + d2y + d2z;
            }
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < Nx - 1; ++i) { // Skip corners already done
        for (int j = 1; j < Ny - 1; ++j) {
            // Z-boundaries (k=0 and k=Nz-1)
            for (int k : {0, Nz - 1}) {
                const int center_idx = idx(i, j, k);
                const QTensor& q_center = Q[center_idx];
                QTensor d2x, d2y, d2z;
                
                // X- and Y-derivatives are always interior here
                d2x = (Q[idx(i + 1, j, k)] - q_center * 2.0 + Q[idx(i - 1, j, k)]) * inv_dx2;
                d2y = (Q[idx(i, j + 1, k)] - q_center * 2.0 + Q[idx(i, j - 1, k)]) * inv_dy2;

                // Z-derivative (boundary condition)
                if (k == 0)           d2z = (Q[idx(i, j, k + 2)] - Q[idx(i, j, k + 1)] * 2.0 + q_center) * inv_dz2;
                else /* k == Nz-1 */ d2z = (q_center - Q[idx(i, j, k - 1)] * 2.0 + Q[idx(i, j, k - 2)]) * inv_dz2;
                
                laplacianQ[center_idx] = d2x + d2y + d2z;
            }
        }
    }
}

double computeTotalFreeEnergy(const std::vector<QTensor>& Q, int Nx, int Ny, int Nz, double dx, double dy, double dz,
    const DimensionalParams& params, double kappa, int modechoice) {
    double total_energy = 0.0;
	auto idx = [&](int i, int j, int k) { return k + Nz * (j + Ny * i); };
	
    double dT, coeff_Q2, coeff_Q3, coeff_Q4;
    if(modechoice==1){
        dT = params.T - params.T_star;
        coeff_Q2 = (3.0 * params.a * dT) / 2.0; // 1.5 * a * dT
        coeff_Q3 = (3.0 * params.b) / 2.0; // 1.5 * b ( = 1/3 * Force_coeff)
        coeff_Q4 = (9.0 * params.c) / 8.0; // 1.125 * c ( = 1/4 * Force_coeff)
    }
    else if (modechoice==2) {
        // ravnik-zumer strict
        dT = params.T - params.T_star;
        coeff_Q2 = (3.0 * params.a * dT) / 2.0; // 1.5 * a * dT
        coeff_Q3 = (9.0 * params.b) / 2.0;      // 4.5 * b
        coeff_Q4 = (9.0 * params.c) / 4.0;      // 2.25 * c
    }
    else{
        std::cerr << "Warning: Unknown modechoice for free energy calculation. Defaulting to mode 1." << std::endl;
        modechoice = 1;
    }

    // Check if we are using anisotropic elastic constants
    bool using_anisotropic = (params.L1 != 0.0 || params.L2 != 0.0);

    #pragma omp parallel for collapse(3) reduction(+:total_energy)
    for (int i = 1; i < Nx-1; ++i) {
        for (int j = 1; j < Ny-1; ++j) {
            for (int k = 1; k < Nz-1; ++k) {
                FullQTensor q(Q[idx(i, j, k)]);

				// --- Bulk Free Energy ---
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
                double f_bulk = coeff_Q2 * trQ2 - coeff_Q3 * trQ3 + coeff_Q4 * trQ2 * trQ2;

				// --- Elastic Energy ---
                const double inv_2dx = 1.0 / (2.0 * dx);
                const double inv_2dy = 1.0 / (2.0 * dy);
                const double inv_2dz = 1.0 / (2.0 * dz);
                const QTensor& q_ip1 = Q[idx(i + 1, j, k)];
                const QTensor& q_im1 = Q[idx(i - 1, j, k)];
                const QTensor& q_jp1 = Q[idx(i, j + 1, k)];
                const QTensor& q_jm1 = Q[idx(i, j - 1, k)];
                const QTensor& q_kp1 = Q[idx(i, j, k + 1)];
                const QTensor& q_km1 = Q[idx(i, j, k - 1)];
                
                auto d_comp = [&](auto get){
                    double dxv = (get(q_ip1) - get(q_im1)) * inv_2dx;
                    double dyv = (get(q_jp1) - get(q_jm1)) * inv_2dy;
                    double dzv = (get(q_kp1) - get(q_km1)) * inv_2dz;
                    return std::array<double,3>{dxv,dyv,dzv};
                };
                auto dQxx = d_comp([&](const QTensor& qq){return qq.Qxx;});
                auto dQxy = d_comp([&](const QTensor& qq){return qq.Qxy;});
                auto dQxz = d_comp([&](const QTensor& qq){return qq.Qxz;});
                auto dQyy = d_comp([&](const QTensor& qq){return qq.Qyy;});
                auto dQyz = d_comp([&](const QTensor& qq){return qq.Qyz;});
                std::array<double,3> dQzz{ - (dQxx[0] + dQyy[0]), - (dQxx[1] + dQyy[1]), - (dQxx[2] + dQyy[2]) };
                auto sq = [&](const std::array<double,3>& v){return v[0]*v[0]+v[1]*v[1]+v[2]*v[2];};

                double f_elastic = 0.0;
                double grad_Q_sq = sq(dQxx) + sq(dQyy) + sq(dQzz) + 2.0*(sq(dQxy) + sq(dQxz) + sq(dQyz));

                if (!using_anisotropic) {
                    // One-Constant approximation (Kappa)
                    f_elastic = 0.5 * kappa * grad_Q_sq;
                } else {
                    // L1 Term
                    double fel_L1 = 0.5 * params.L1 * grad_Q_sq;
                    // L2 Term
                    auto get_div = [&](int row) { 
                        double val = 0;
                         if(row==0) val += dQxx[0]; else if(row==1) val += dQxy[0]; else val += dQxz[0];
                         if(row==0) val += dQxy[1]; else if(row==1) val += dQyy[1]; else val += dQyz[1];
                         if(row==0) val += dQxz[2]; else if(row==1) val += dQyz[2]; else val += dQzz[2];
                        return val;
                    };
                    double divQ_x = get_div(0);
                    double divQ_y = get_div(1);
                    double divQ_z = get_div(2);
                    double fel_L2 = 0.5 * params.L2 * (divQ_x*divQ_x + divQ_y*divQ_y + divQ_z*divQ_z);
                    
                    f_elastic = fel_L1 + fel_L2;
                }
                total_energy += (f_bulk + f_elastic) * (dx * dy * dz);
            }
        }
    }
    return total_energy;
}

EnergyComponents computeEnergyComponents(const std::vector<QTensor>& Q, int Nx, int Ny, int Nz, double dx, double dy, double dz,
    const DimensionalParams& params, double kappa, int modechoice) {

    // 1. local scalars for openMP reduction (struct members not allowed in reduction)
    double sum_bulk = 0.0;
    double sum_elastic = 0.0;
    
    //EnergyComponents ec; -> replaced by local scalars
    auto idx = [&](int i, int j, int k) { return k + Nz * (j + Ny * i); };

    double dT, coeff_Q2, coeff_Q3, coeff_Q4;

    if(modechoice==1){
        dT = params.T - params.T_star;
        coeff_Q2 = (3.0 * params.a * dT) / 2.0;
        coeff_Q3 = (3.0 * params.b) / 2.0;
        coeff_Q4 = (9.0 * params.c) / 8.0;
    }
    else if (modechoice==2) {
        // mode 2
        dT = params.T - params.T_star;
        coeff_Q2 = (3.0 * params.a * dT) / 2.0; // 1.5 * a * dT
        coeff_Q3 = (9.0 * params.b) / 2.0;      // 4.5 * b
        coeff_Q4 = (9.0 * params.c) / 4.0;      // 2.25 * c
    }
    else{
        std::cerr << "Warning: Unknown modechoice for free energy calculation. Defaulting to mode 1." << std::endl;
        modechoice = 1;
    }
    // Check if we are using anisotropic elastic constants
    bool using_anisotropic = (params.L1 != 0.0 || params.L2 != 0.0);
    
    // Pre-calculate inverse steps
    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);
    const double inv_2dz = 1.0 / (2.0 * dz);
    
    // Use reduction to sum safely in parallel
    #pragma omp parallel for collapse(3) reduction(+:sum_bulk, sum_elastic)
    for (int i = 1; i < Nx-1; ++i) {
        for (int j = 1; j < Ny-1; ++j) {
            for (int k = 1; k < Nz-1; ++k) {
                int id = k + Nz * (j + Ny * i);
                FullQTensor q(Q[id]);

                // --- Bulk Free Energy ---
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
                double f_bulk = coeff_Q2 * trQ2 - coeff_Q3 * trQ3 + coeff_Q4 * trQ2 * trQ2;

                // accumulate to scalar
                sum_bulk += f_bulk * (dx * dy * dz);

                // --- Elastic Energy ---
                const double inv_2dx = 1.0 / (2.0 * dx);
                const double inv_2dy = 1.0 / (2.0 * dy);
                const double inv_2dz = 1.0 / (2.0 * dz);
                const QTensor& q_ip1 = Q[idx(i + 1, j, k)];
                const QTensor& q_im1 = Q[idx(i - 1, j, k)];
                const QTensor& q_jp1 = Q[idx(i, j + 1, k)];
                const QTensor& q_jm1 = Q[idx(i, j - 1, k)];
                const QTensor& q_kp1 = Q[idx(i, j, k + 1)];
                const QTensor& q_km1 = Q[idx(i, j, k - 1)];
                
                auto d_comp = [&](auto get){
                    double dxv = (get(q_ip1) - get(q_im1)) * inv_2dx;
                    double dyv = (get(q_jp1) - get(q_jm1)) * inv_2dy;
                    double dzv = (get(q_kp1) - get(q_km1)) * inv_2dz;
                    return std::array<double,3>{dxv,dyv,dzv};
                };
                auto dQxx = d_comp([&](const QTensor& qq){return qq.Qxx;});
                auto dQxy = d_comp([&](const QTensor& qq){return qq.Qxy;});
                auto dQxz = d_comp([&](const QTensor& qq){return qq.Qxz;});
                auto dQyy = d_comp([&](const QTensor& qq){return qq.Qyy;});
                auto dQyz = d_comp([&](const QTensor& qq){return qq.Qyz;});
                std::array<double,3> dQzz{ - (dQxx[0] + dQyy[0]), - (dQxx[1] + dQyy[1]), - (dQxx[2] + dQyy[2]) };
                auto sq = [&](const std::array<double,3>& v){return v[0]*v[0]+v[1]*v[1]+v[2]*v[2];};

                double f_elastic = 0.0;
                double grad_Q_sq = sq(dQxx) + sq(dQyy) + sq(dQzz) + 2.0*(sq(dQxy) + sq(dQxz) + sq(dQyz));

                if (!using_anisotropic) {
                    // One-Constant approximation (Kappa)
                    f_elastic = 0.5 * kappa * grad_Q_sq;
                } else {
                    // L1 Term
                    double fel_L1 = 0.5 * params.L1 * grad_Q_sq;
                    // L2 Term
                    auto get_div = [&](int row) { 
                        double val = 0;
                         if(row==0) val += dQxx[0]; else if(row==1) val += dQxy[0]; else val += dQxz[0];
                         if(row==0) val += dQxy[1]; else if(row==1) val += dQyy[1]; else val += dQyz[1];
                         if(row==0) val += dQxz[2]; else if(row==1) val += dQyz[2]; else val += dQzz[2];
                        return val;
                    };
                    double divQ_x = get_div(0);
                    double divQ_y = get_div(1);
                    double divQ_z = get_div(2);

                    double fel_L2 = 0.5 * params.L2 * (divQ_x*divQ_x + divQ_y*divQ_y + divQ_z*divQ_z);
                    
                    f_elastic = fel_L1 + fel_L2;
                }

                // accumulate to scalar
                sum_elastic += f_elastic * (dx * dy * dz);
            }
        }
    }
    EnergyComponents ec;
    ec.bulk = sum_bulk;
    ec.elastic = sum_elastic;
    return ec;
}

void computeChemicalPotential(const std::vector<QTensor>& Q, std::vector<QTensor>& mu,
    const std::vector<QTensor>& laplacianQ, int Nx, int Ny, int Nz, double dx, double dy, double dz,
    const DimensionalParams& params, double kappa, int modechoice) {
	auto idx = [&](int i, int j, int k) { return k + Nz * (j + Ny * i); };
	
    double dT, coeff_Q, coeff_Q2, coeff_Q3;
    if(modechoice==1){
        dT = params.T - params.T_star;
        coeff_Q = 3.0 * params.a * dT;
        coeff_Q2 = (9.0 * params.b) / 2.0; // 4.5 * b (Derivative of 3/2 b Q^2)
        coeff_Q3 = (9.0 * params.c) / 2.0; // 4.5 * c (Derivative of 9/8 c Q^3)
    }
    else if (modechoice==2) {
        // ravnik-zumer strict
        dT = params.T - params.T_star;
        coeff_Q = 3.0 * params.a * dT;
        coeff_Q2 = (27.0 * params.b) / 2.0; // 13.5 * b (Derivative of 9/2 b Q^3)
        coeff_Q3 = (9.0 * params.c);        // 9.0 * c  (Derivative of 9/4 c Q^4)
    }
    else{
        std::cerr << "Warning: Unknown modechoice for chemical potential calculation. Defaulting to mode 1." << std::endl;
        modechoice = 1;
    }

    // Precompute divergence of columns D_j = ∂k Q_kj ONLY if L2 is active
    bool use_L2 = (params.L2 != 0.0);
    std::vector<double> Dcol_x, Dcol_y, Dcol_z;
    
    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);
    const double inv_2dz = 1.0 / (2.0 * dz);
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;
    const double inv_dz = 1.0 / dz;

    if (use_L2) {
        Dcol_x.resize(Nx*Ny*Nz); Dcol_y.resize(Nx*Ny*Nz); Dcol_z.resize(Nx*Ny*Nz);
        #pragma omp parallel for collapse(3)
        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                for (int k = 0; k < Nz; ++k) {
                    int id = idx(i,j,k);
                    double dQxx_dx = get_Q_deriv(Q, id, Ny*Nz, Nx, i, inv_2dx, inv_dx, 0); 
                    double dQxy_dy = get_Q_deriv(Q, id, Nz, Ny, j, inv_2dy, inv_dy, 1);    
                    double dQxz_dz = get_Q_deriv(Q, id, 1, Nz, k, inv_2dz, inv_dz, 2);     
                    Dcol_x[id] = dQxx_dx + dQxy_dy + dQxz_dz;

                    double dQyx_dx = get_Q_deriv(Q, id, Ny*Nz, Nx, i, inv_2dx, inv_dx, 1); 
                    double dQyy_dy = get_Q_deriv(Q, id, Nz, Ny, j, inv_2dy, inv_dy, 3);    
                    double dQyz_dz = get_Q_deriv(Q, id, 1, Nz, k, inv_2dz, inv_dz, 4);     
                    Dcol_y[id] = dQyx_dx + dQyy_dy + dQyz_dz;

                    double dQzx_dx = get_Q_deriv(Q, id, Ny*Nz, Nx, i, inv_2dx, inv_dx, 2); 
                    double dQzy_dy = get_Q_deriv(Q, id, Nz, Ny, j, inv_2dy, inv_dy, 4);    
                    double Qzz_c = -(Q[id].Qxx + Q[id].Qyy);
                    double dQzz_dz;
                    if (k > 0 && k < Nz - 1) {
                         double Qzz_p = -(Q[id + 1].Qxx + Q[id + 1].Qyy);
                         double Qzz_m = -(Q[id - 1].Qxx + Q[id - 1].Qyy);
                         dQzz_dz = (Qzz_p - Qzz_m) * inv_2dz;
                    } else if (k == 0) {
                         double Qzz_p = -(Q[id + 1].Qxx + Q[id + 1].Qyy);
                         dQzz_dz = (Qzz_p - Qzz_c) * inv_dz;
                    } else {
                         double Qzz_m = -(Q[id - 1].Qxx + Q[id - 1].Qyy);
                         dQzz_dz = (Qzz_c - Qzz_m) * inv_dz;
                    }
                    Dcol_z[id] = dQzx_dx + dQzy_dy + dQzz_dz;
                }
            }
        }
    }

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                int id = idx(i, j, k);
                FullQTensor q(Q[id]);
                const QTensor& lap_q = laplacianQ[id];
                QTensor& h = mu[id];

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

                // Legacy Kappa Term (Only if L1==0 && L2==0)
                if (params.L1 == 0.0 && params.L2 == 0.0) {
                    h_xx -= kappa * lap_q.Qxx;
                    h_yy -= kappa * lap_q.Qyy;
                    h_zz -= kappa * (-lap_q.Qxx - lap_q.Qyy);
                    h_xy -= kappa * lap_q.Qxy;
                    h_xz -= kappa * lap_q.Qxz;
                    h_yz -= kappa * lap_q.Qyz;
                }

                // L1 Term
                if (params.L1 != 0.0) {
                    h_xx -= params.L1 * lap_q.Qxx;
                    h_yy -= params.L1 * lap_q.Qyy;
                    h_zz -= params.L1 * (-lap_q.Qxx - lap_q.Qyy);
                    h_xy -= params.L1 * lap_q.Qxy;
                    h_xz -= params.L1 * lap_q.Qxz;
                    h_yz -= params.L1 * lap_q.Qyz;
                }

                // L2 Term
                if (use_L2) {
                    double dDx_dx = get_deriv(Dcol_x, id, Ny*Nz, Nx, i, inv_2dx, inv_dx);
                    double dDy_dy = get_deriv(Dcol_y, id, Nz, Ny, j, inv_2dy, inv_dy);
                    double dDz_dz = get_deriv(Dcol_z, id, 1, Nz, k, inv_2dz, inv_dz);
                    double dDx_dy = get_deriv(Dcol_x, id, Nz, Ny, j, inv_2dy, inv_dy);
                    double dDy_dx = get_deriv(Dcol_y, id, Ny*Nz, Nx, i, inv_2dx, inv_dx);
                    double dDx_dz = get_deriv(Dcol_x, id, 1, Nz, k, inv_2dz, inv_dz);
                    double dDz_dx = get_deriv(Dcol_z, id, Ny*Nz, Nx, i, inv_2dx, inv_dx);
                    double dDy_dz = get_deriv(Dcol_y, id, 1, Nz, k, inv_2dz, inv_dz);
                    double dDz_dy = get_deriv(Dcol_z, id, Nz, Ny, j, inv_2dy, inv_dy);

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
            }
        }
    }
}

void applyWeakAnchoringPenalty(std::vector<QTensor>& mu, const std::vector<QTensor>& Q,
    int Nx, int Ny, int Nz, const std::vector<bool>& is_shell, double S_shell, double W) {
    if (W == 0.0) return;
    double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0;
    auto idx = [&](int i, int j, int k) { return k + Nz * (j + Ny * i); };

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (!is_shell[idx(i,j,k)]) continue;
                double x = i - cx, y = j - cy, z = k - cz;
                double r = std::sqrt(x*x + y*y + z*z);
                if (r < 1e-9) continue;
                double nx = x/r, ny = y/r, nz = z/r;
                QTensor Q0;
                Q0.Qxx = S_shell * (nx*nx - 1.0/3.0);
                Q0.Qxy = S_shell * (nx*ny);
                Q0.Qxz = S_shell * (nx*nz);
                Q0.Qyy = S_shell * (ny*ny - 1.0/3.0);
                Q0.Qyz = S_shell * (ny*nz);
                const QTensor& q = Q[idx(i,j,k)];
                QTensor& h = mu[idx(i,j,k)];
                h.Qxx += W * (q.Qxx - Q0.Qxx);
                h.Qxy += W * (q.Qxy - Q0.Qxy);
                h.Qxz += W * (q.Qxz - Q0.Qxz);
                h.Qyy += W * (q.Qyy - Q0.Qyy);
                h.Qyz += W * (q.Qyz - Q0.Qyz);
            }
        }
    }
}

void updateQTensor(std::vector<QTensor>& Q, const std::vector<QTensor>& mu, int Nx, int Ny, int Nz,
    double dt, const std::vector<bool>& is_shell, double gamma, double W) {
	auto idx = [&](int i, int j, int k) { return k + Nz * (j + Ny * i); };
    double R = std::min({ (double)Nx, (double)Ny, (double)Nz }) / 2.0;
    double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0;
    double mobility = 1.0 / gamma;

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                // Skip shell for strong anchoring
                if (W == 0.0 && is_shell[idx(i, j, k)]) continue;

                double x = i - cx, y = j - cy, z = k - cz;
				double r = std::sqrt(x * x + y * y + z * z);
                if (W == 0.0) { if (r > R) continue; } 
                else { if (r > R + 1.5) continue; }

				const QTensor& current_mu = mu[idx(i, j, k)];
				QTensor& current_Q = Q[idx(i, j, k)];

                current_Q.Qxx -= mobility * current_mu.Qxx * dt;
                current_Q.Qxy -= mobility * current_mu.Qxy * dt;
                current_Q.Qxz -= mobility * current_mu.Qxz * dt;
                current_Q.Qyy -= mobility * current_mu.Qyy * dt;
                current_Q.Qyz -= mobility * current_mu.Qyz * dt;
            }
        }
    }
}

void applyBoundaryConditions(std::vector<QTensor>& Q, int Nx, int Ny, int Nz, const std::vector<bool>& is_shell, double S0, double W) {
    if (W > 0.0) return;
    double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0;
    auto idx = [&](int i, int j, int k) { return k + Nz * (j + Ny * i); };

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (is_shell[idx(i, j, k)]) {
                    double x = i - cx, y = j - cy, z = k - cz;
                    double r = std::sqrt(x * x + y * y + z * z);
                    if (r > 1e-9) {
                        double nx = x / r, ny = y / r, nz = z / r;
                        QTensor& q = Q[idx(i, j, k)];
                        q.Qxx = S0 * (nx * nx - 1.0 / 3.0);
                        q.Qxy = S0 * (nx * ny);
                        q.Qxz = S0 * (nx * nz);
                        q.Qyy = S0 * (ny * ny - 1.0 / 3.0);
                        q.Qyz = S0 * (ny * nz);
                    }
                }
            }
        }
    }
}

template<typename T>
T prompt_with_default(const std::string& prompt, T default_value) {
    std::string line;
    std::cout << prompt << " [" << default_value << "]: ";
    std::getline(std::cin, line);
    if (line.empty()) return default_value;
    std::istringstream iss(line);
    T value;
    if (!(iss >> value)) return default_value;
    return value;
}

double calculateAverageS(const std::vector<QTensor>& Q, int Nx, int Ny, int Nz) {
    double total_S = 0.0;
    int count = 0;
    auto idx = [&](int i, int j, int k) { return k + Nz * (j + Ny * i); };
    double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0;
    double R = std::min({ (double)Nx, (double)Ny, (double)Nz }) / 2.0;

    #pragma omp parallel for collapse(3) reduction(+:total_S, count)
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                double r = std::sqrt(std::pow(i - cx, 2) + std::pow(j - cy, 2) + std::pow(k - cz, 2));
                if (r < R) { 
                    NematicField nf = calculateNematicField(Q[idx(i, j, k)]);
                    total_S += nf.S;
                    count++;
                }
            }
        }
    }
    return (count > 0) ? (total_S / count) : 0.0;
}

double computeRadiality(const std::vector<QTensor>& Q, int Nx, int Ny, int Nz) {
    auto idx = [&](int i, int j, int k) { return k + Nz * (j + Ny * i); };
    double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0;
    double R = std::min({ (double)Nx, (double)Ny, (double)Nz }) / 2.0;
    double radiality_sum = 0.0;
    int count = 0;
    #pragma omp parallel for collapse(3) reduction(+:radiality_sum, count)
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                double x = i - cx, y = j - cy, z = k - cz;
                double r = std::sqrt(x*x + y*y + z*z);
                if (r < R && r > 2.0) {
                    NematicField nf = calculateNematicField(Q[idx(i,j,k)]);
                    double nx_r = x/r, ny_r = y/r, nz_r = z/r;
                    double dot_product = std::abs(nf.nx*nx_r + nf.ny*ny_r + nf.nz*nz_r);
                    radiality_sum += dot_product;
                    count++;
                }
            }
        }
    }
    return (count > 0) ? (radiality_sum / count) : 0.0;
}

int main() {
    bool run_again = false;
    do {
        // Default values for grid and simulation
        int Nx_default = 100, Ny_default = 100, Nz_default = 100, maxIterations_default = 100000;
        double dx_default = 1e-8, dy_default = 1e-8, dz_default = 1e-8;
        
        // Default Landau-de Gennes parameters for 5CB
        double a_default = 0.044e6;     
        double b_default = 1.413e6;      
        double c_default = 1.153e6;      
        double T_default = 300.0;        
        double T_star_default = 308.0;   
        
        double kappa_default = 6.5e-12;  
        double gamma_default = 0.1;      
        double dt_default = 1e-16;       
        int mode_default = 1;      
        int modechoice_default=1;      

        std::cout << "\033[1mQ-tensor evolution simulation for a 5CB liquid crystal droplet\033[0m\n" << std::endl;
        std::cout << "Press {\033[1;33mEnter\033[0m} to use [\033[1;34mdefault\033[0m] values, or input your own.\n" << std::endl;

        std::cout << "Select initialization mode for Q-tensor field: [0] random, [1] radial (default: 1): ";
        std::string mode_input;
        std::getline(std::cin, mode_input);
        int mode = mode_default;
        if (!mode_input.empty()) {
            try { mode = std::stoi(mode_input); }
            catch (...) { mode = mode_default; }
        }
        if (mode != 0 && mode != 1) mode = mode_default;
        std::cout << "Initialization mode: " << (mode == 0 ? "random" : "radial") << std::endl;

        std::cout << "\n--- Spatial Parameters ---" << std::endl;
        int Nx = prompt_with_default("Enter grid size Nx", Nx_default);
        int Ny = prompt_with_default("Enter grid size Ny", Ny_default);
        int Nz = prompt_with_default("Enter grid size Nz", Nz_default);
        double dx = prompt_with_default("Enter grid spacing dx (m)", dx_default);
        double dy = prompt_with_default("Enter grid spacing dy (m)", dy_default);
        double dz = prompt_with_default("Enter grid spacing dz (m)", dz_default);
        
        std::cout << "\n--- Landau-de Gennes Material Parameters ---" << std::endl;
        double a = prompt_with_default("Enter parameter a (J/m^3/K)", a_default);
        double b = prompt_with_default("Enter parameter b (J/m^3)", b_default);
        double c = prompt_with_default("Enter parameter c (J/m^3)", c_default);
        double T = prompt_with_default("Enter temperature T (K)", T_default);
        double T_star = prompt_with_default("Enter T* (K)", T_star_default);
        int modechoice = prompt_with_default("Enter Bulk Energy convention choice (1=standard, 2=ravnik-zumer)", modechoice_default);
        
        double S_c = b / (2.0 * c);
        double dT = T - T_star;
        double discriminant = 9.0 * b * b - 32.0 * a * c * dT;
        double S_eq = 0.0;
        if (discriminant > 0 && dT < 0) {
            S_eq = (3.0 * b + std::sqrt(discriminant)) / (8.0 * c);
        }
        std::cout << "\nComputed S_c = " << S_c << ", S_eq(T=" << T << "K) = " << S_eq << std::endl;
        
        double S0 = prompt_with_default("Enter initial order parameter S0", S_eq > 0 ? S_eq : 0.5);
        
        std::cout << "\n--- Elastic and Dynamic Parameters ---" << std::endl;
        double kappa = prompt_with_default("Enter elastic constant kappa (J/m)", kappa_default);
        std::cout << "Use Frank-to-LdG mapping with K1=K3 ≠ K2? (y/n) [n]: ";
        std::string use_frank_map_in;
        std::getline(std::cin, use_frank_map_in);
        bool use_frank_map = (!use_frank_map_in.empty() && (use_frank_map_in[0]=='y' || use_frank_map_in[0]=='Y'));
        double L1 = 0.0, L2 = 0.0, L3 = 0.0;
        if (use_frank_map) {
            double K1 = prompt_with_default("Enter K1 = K3 (J/m)", 6.5e-12);
            double K2 = prompt_with_default("Enter K2 (J/m)", 4.0e-12);
            double S_ref = (S0 > 0 ? S0 : (b / (2.0 * c)));
            if (S_ref <= 1e-12) S_ref = 0.5; 
            
            // Simple mapping used in literature:
            L1 = (2.0 * K2) / (9.0 * S_ref * S_ref);
            L2 = (2.0 * K1) / (9.0 * S_ref * S_ref) - L1; // derived from K1 = 9/2 S^2 (L1+L2)
            
            // PHYSICS FIX 1: If using L-constants, zero out isotropic kappa to prevent double counting
            kappa = 0.0; 
            L3 = 0.0; // PHYSICS FIX 2: Force L3 to 0 for stability
            std::cout << "Mapped L1=" << L1 << ", L2=" << L2 << ", L3=" << L3 << " (Kappa set to 0)" << std::endl;
        } else {
            L1 = prompt_with_default("Enter L1 (J/m)", 0.0);
            L2 = prompt_with_default("Enter L2 (J/m)", 0.0);
            L3 = prompt_with_default("Enter L3 (J/m)", 0.0);
            if (L1 != 0 || L2 != 0) kappa = 0.0; // Auto-disable kappa if user manually sets L terms
        }
        double W = prompt_with_default("Enter weak anchoring W (J/m^2)", 0.0);
        
        double gamma = prompt_with_default("Enter rotational viscosity gamma (Pa·s)", gamma_default);
        int maxIterations = prompt_with_default("Enter number of iterations", maxIterations_default);
        int printFreq = prompt_with_default("Enter print frequency (iterations)", 200);
        double tolerance = prompt_with_default("Enter convergence tolerance (relative change)", 1e-2);

        std::vector<QTensor> Q(Nx * Ny * Nz);
        std::vector<QTensor> laplacianQ(Nx * Ny * Nz);
        std::vector<QTensor> mu(Nx * Ny * Nz);
        std::vector<bool> is_shell(Nx * Ny * Nz, false);

        auto idx = [&](int i, int j, int k) { return k + Nz * (j + Ny * i); };
        double R = std::min({ (double)Nx, (double)Ny, (double)Nz }) / 2.0;
        double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0;
        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                for (int k = 0; k < Nz; ++k) {
                    double r = std::sqrt(std::pow(i - cx, 2) + std::pow(j - cy, 2) + std::pow(k - cz, 2));
                    if (std::abs(r - R) < 1.0) {
                        is_shell[idx(i, j, k)] = true;
                    }
                }
            }
        }

        std::cout << "\nSelect simulation mode:\n";
        std::cout << "  [1] Single Temperature\n";
        std::cout << "  [2] Temperature Range\n";
        std::cout << "  [3] Quench Dynamics (time-dependent T)\n";
        int sim_mode = prompt_with_default("Enter mode", 1);

        if (sim_mode == 3) {
            if (fs::exists("output_quench")) {
                fs::remove_all("output_quench");
            }
            fs::create_directory("output_quench");

            // Estimate a coexistence temperature for reference (depends on bulk convention).
            double T_trans = computeTransitionTemperature(a, b, c, T_star, modechoice);
            std::cout << "\nEstimated coexistence temperature T_NI ≈ " << T_trans << " K (based on modechoice=" << modechoice << ")\n";
            std::cout << "(Note: T* is the spinodal/supercooling reference in this model.)\n";

            std::cout << "\nQuench protocol:\n";
            std::cout << "  [1] Step quench (T_high -> T_low instantly)\n";
            std::cout << "  [2] Linear ramp (T_high -> T_low over N_ramp steps)\n";
            int qprot = prompt_with_default("Enter protocol", 2);
            if (qprot != 1 && qprot != 2) qprot = 2;

            double T_high = prompt_with_default("Enter high temperature T_high (K)", T_trans + 2.0);
            double T_low = prompt_with_default("Enter low temperature T_low (K)", T_star - 2.0);
            int pre_equil_iters = prompt_with_default("Enter pre-equilibration steps at T_high", 0);
            int ramp_iters = (qprot == 2) ? prompt_with_default("Enter ramp steps N_ramp", 2000) : 0;
            int total_iters = prompt_with_default("Enter total steps", maxIterations);
            int snapshotFreq = prompt_with_default("Enter snapshot frequency (iterations)", printFreq);

            // Choose initialization specific for quench dynamics
            std::cout << "\nInitialize from isotropic state (Q=0) + small noise? (y/n) [y]: ";
            std::string iso_in;
            std::getline(std::cin, iso_in);
            bool iso_init = iso_in.empty() || iso_in[0] == 'y' || iso_in[0] == 'Y';
            double noise_amp = 0.0;
            if (iso_init) {
                noise_amp = prompt_with_default("Noise amplitude for initial Q components", 1e-4);
                initializeQTensor(Q, Nx, Ny, Nz, 0.0, 0);
                addIsotropicNoise(Q, Nx, Ny, Nz, noise_amp);
            } else {
                initializeQTensor(Q, Nx, Ny, Nz, S0, mode);
            }

            double min_dx = std::min({ dx, dy, dz });
            double D = (L1 > 0 ? L1 : kappa) / gamma;
            if (D == 0.0) D = 1e-12;
            double dt_max = 0.5 * (min_dx * min_dx) / (6.0 * D);
            std::cout << "\nMaximum stable dt ≈ " << dt_max << " s" << std::endl;
            double dt = prompt_with_default("Enter time step dt (s)", dt_max);
            if (dt > dt_max) dt = dt_max;

            std::ofstream quench_log("output_quench/quench_log.dat");
            quench_log << "iteration,time_s,T_K,bulk,elastic,field,total,avg_S,radiality\n";

            DimensionalParams params_q;
            params_q.a = a; params_q.b = b; params_q.c = c;
            params_q.T = T_high; params_q.T_star = T_star;
            params_q.L1 = L1; params_q.L2 = L2; params_q.L3 = L3;
            params_q.W = W;

            double physical_time = 0.0;
            for (int iter = 0; iter < total_iters; ++iter) {
                physical_time += dt;

                // Temperature protocol T(iter)
                double T_current = T_high;
                if (iter < pre_equil_iters) {
                    T_current = T_high;
                } else if (qprot == 1) {
                    T_current = T_low;
                } else {
                    int t0 = iter - pre_equil_iters;
                    if (t0 <= 0) {
                        T_current = T_high;
                    } else if (t0 >= ramp_iters) {
                        T_current = T_low;
                    } else {
                        double s = (double)t0 / (double)ramp_iters;
                        T_current = T_high + (T_low - T_high) * s;
                    }
                }

                params_q.T = T_current;

                // Shell order parameter: switch on when below T* (consistent with current usage)
                double S_shell = (T_current > T_star) ? 0.0 : S0;

                computeLaplacian(Q, laplacianQ, Nx, Ny, Nz, dx, dy, dz);
                computeChemicalPotential(Q, mu, laplacianQ, Nx, Ny, Nz, dx, dy, dz, params_q, kappa, modechoice);
                applyWeakAnchoringPenalty(mu, Q, Nx, Ny, Nz, is_shell, S_shell, params_q.W);
                updateQTensor(Q, mu, Nx, Ny, Nz, dt, is_shell, gamma, params_q.W);
                applyBoundaryConditions(Q, Nx, Ny, Nz, is_shell, S_shell, params_q.W);

                if (iter % printFreq == 0) {
                    EnergyComponents ec = computeEnergyComponents(Q, Nx, Ny, Nz, dx, dy, dz, params_q, kappa, modechoice);
                    double avg_S = calculateAverageS(Q, Nx, Ny, Nz);
                    double radiality = computeRadiality(Q, Nx, Ny, Nz);
                    quench_log << iter << "," << physical_time << "," << T_current << ","
                              << ec.bulk << "," << ec.elastic << "," << ec.field << "," << ec.total() << ","
                              << avg_S << "," << radiality << "\n";
                    std::cout << "Iter " << iter << "  t=" << physical_time << " s  T=" << T_current
                              << " K  F=" << ec.total() << "  <S>=" << avg_S << "  R̄=" << radiality << std::endl;
                }

                if (snapshotFreq > 0 && iter % snapshotFreq == 0) {
                    std::string nematic_filename = "output_quench/nematic_field_iter_" + std::to_string(iter) + ".dat";
                    std::vector<NematicField> nematicField(Nx * Ny * Nz);
                    #pragma omp parallel for collapse(3)
                    for (int i = 0; i < Nx; ++i) {
                        for (int j = 0; j < Ny; ++j) {
                            for (int k = 0; k < Nz; ++k) {
                                nematicField[idx(i, j, k)] = calculateNematicField(Q[idx(i, j, k)]);
                            }
                        }
                    }
                    saveNematicFieldToFile(nematicField, Nx, Ny, Nz, nematic_filename);
                }
            }

            quench_log.close();
            saveQTensorToFile(Q, Nx, Ny, Nz, "output_quench/Qtensor_output_final.dat");
            std::cout << "\nQuench simulation finished. Logs in output_quench/quench_log.dat" << std::endl;
        }
        else if (sim_mode == 2) {
            if (fs::exists("output_temp_sweep")) {
                fs::remove_all("output_temp_sweep");
            }
            fs::create_directory("output_temp_sweep");
            std::ofstream sweep_log("output_temp_sweep/summary.dat");
            sweep_log << "temperature,final_energy,average_S\n";

            double T_start = prompt_with_default("Enter start temperature T (K)", 295.0);
            double T_end = prompt_with_default("Enter end temperature T (K)", 315.0);
            double T_step = prompt_with_default("Enter temperature step (K)", 1.0);

            initializeQTensor(Q, Nx, Ny, Nz, S0, mode);

            double min_dx = std::min({ dx, dy, dz });
            double D = (L1 > 0 ? L1 : kappa) / gamma; // Use L1 if kappa is 0
            if (D==0) D = 1e-12; // safety
            double dt_max = 0.5 * (min_dx * min_dx) / (6.0 * D);
            std::cout << "\nMaximum stable dt ≈ " << dt_max << " s" << std::endl;
            double dt = prompt_with_default("Enter time step dt (s)", dt_max);
            if (dt > dt_max) dt = dt_max;

            for (double T_current = T_start; T_current <= T_end; T_current += T_step) {
                std::cout << "\n--- Running simulation for T = " << T_current << " K ---\n"; 
                
                DimensionalParams params_temp;
                params_temp.a = a; params_temp.b = b; params_temp.c = c;
                params_temp.T = T_current; params_temp.T_star = T_star;
                params_temp.L1 = L1; params_temp.L2 = L2; params_temp.L3 = L3;
                params_temp.W = W;

                double dT_sweep = T_current - T_star;
                double disc_sweep = 9.0*b*b - 32.0*a*c*dT_sweep;
                double S_eq_sweep = (disc_sweep > 0 && dT_sweep < 0) ? (3.0*b + std::sqrt(disc_sweep))/(8.0*c) : 0.0;
                double S_shell = S_eq_sweep;
                
                std::string temp_dir = "output_temp_sweep/T_" + std::to_string(T_current);
                fs::create_directory(temp_dir);

                bool converged = false;
                double prev_F = std::numeric_limits<double>::max();
                for (int iter = 0; iter < maxIterations; ++iter) {
                    computeLaplacian(Q, laplacianQ, Nx, Ny, Nz, dx, dy, dz);
                    computeChemicalPotential(Q, mu, laplacianQ, Nx, Ny, Nz, dx, dy, dz, params_temp, kappa, modechoice);
                    applyWeakAnchoringPenalty(mu, Q, Nx, Ny, Nz, is_shell, S_shell, params_temp.W);
                    updateQTensor(Q, mu, Nx, Ny, Nz, dt, is_shell, gamma, params_temp.W);
                    applyBoundaryConditions(Q, Nx, Ny, Nz, is_shell, S_shell, params_temp.W);

                    if (iter > 0 && iter % printFreq == 0) {
                        double F = computeTotalFreeEnergy(Q, Nx, Ny, Nz, dx, dy, dz, params_temp, kappa, modechoice);
                        std::cout << "  Iter " << iter << ", Free Energy: " << F << std::endl;
                        if (std::abs((F - prev_F) / prev_F) < tolerance || F == 0.0) {
                            std::cout << "  Convergence reached at iteration " << iter << std::endl;
                            converged = true;
                            break;
                        }
                        prev_F = F;
                    }
                }

                double final_F = computeTotalFreeEnergy(Q, Nx, Ny, Nz, dx, dy, dz, params_temp, kappa, modechoice);
                double avg_S = calculateAverageS(Q, Nx, Ny, Nz);
                sweep_log << T_current << "," << final_F << "," << avg_S << "\n";

                std::vector<NematicField> finalNematicField(Nx * Ny * Nz);
                #pragma omp parallel for collapse(3)
                for (int i = 0; i < Nx; ++i) {
                    for (int j = 0; j < Ny; ++j) {
                        for (int k = 0; k < Nz; ++k) {
                            finalNematicField[idx(i, j, k)] = calculateNematicField(Q[idx(i, j, k)]);
                        }
                    }
                }
                saveNematicFieldToFile(finalNematicField, Nx, Ny, Nz, temp_dir + "/nematic_field_final.dat");
                saveQTensorToFile(Q, Nx, Ny, Nz, temp_dir + "/Qtensor_output_final.dat");
                std::cout << "  Saved final state for T = " << T << " K. Average S = " << avg_S << std::endl;
            }
            sweep_log.close();
        }
        else {
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
            std::cout << "\n";

            if (fs::exists("output")) {
                for (const auto& entry : fs::directory_iterator("output")) {
                    fs::remove(entry.path());
                }
            }
            fs::create_directory("output");

            DimensionalParams params;
            params.a = a; params.b = b; params.c = c;
            params.T = T; params.T_star = T_star;
            params.L1 = L1; params.L2 = L2; params.L3 = L3;
            params.W = W;
            
            std::cout << "\nUsing parameters: a=" << a << ", b=" << b << ", c=" << c << std::endl;
            std::cout << "Temperature: T=" << T << " K, T*=" << T_star << " K" << std::endl;
            
            double min_dx = std::min({ dx, dy, dz });
            double D = (L1 > 0 ? L1 : kappa) / gamma; 
            if(D==0) D = 1e-12;
            double dt_max = 0.5 * (min_dx * min_dx) / (6.0 * D);
            
            double R_phys = (std::min({(double)Nx, (double)Ny, (double)Nz}) / 2.0) * min_dx;
            double tau_align = (R_phys * R_phys) / D;
            
            std::cout << "\n--- Time Scale Analysis ---" << std::endl;
            std::cout << "Diffusion coefficient D = " << D << " m²/s" << std::endl;
            std::cout << "Droplet radius R ≈ " << R_phys << " m" << std::endl;
            std::cout << "Estimated alignment time τ_align ≈ " << tau_align << " s" << std::endl;
            std::cout << "Maximum stable dt ≈ " << dt_max << " s" << std::endl;
            
            double dt = prompt_with_default("Enter time step dt (s)", dt_max);
            if (dt > dt_max) {
                std::cout << "Warning: dt exceeds stability limit. Setting to dt_max = " << dt_max << std::endl;
                dt = dt_max;
            }

            initializeQTensor(Q, Nx, Ny, Nz, S0, mode);
            if (submode == 1) {
                std::ofstream energy_log("free_energy_vs_iteration.dat");
                energy_log << "iteration,free_energy,radiality,time\n";

                bool converged = false;
                double prev_F = std::numeric_limits<double>::max();
                double physical_time = 0.0;
                double radiality_threshold = 0.998;
                double min_alignment_time = 0.5 * tau_align;  
                
                char user_choice;
                std::cout << "Do you want output in the console every " << printFreq << " iterations? (y/n): ";
                std::cin >> user_choice;

                for (int iter = 0; iter < maxIterations; ++iter) {
                    physical_time += dt;
                    computeLaplacian(Q, laplacianQ, Nx, Ny, Nz, dx, dy, dz);
                    computeChemicalPotential(Q, mu, laplacianQ, Nx, Ny, Nz, dx, dy, dz, params, kappa, modechoice);
                    double S_shell = (T > T_star) ? 0.0 : S0;
                    applyWeakAnchoringPenalty(mu, Q, Nx, Ny, Nz, is_shell, S_shell, params.W);
                    updateQTensor(Q, mu, Nx, Ny, Nz, dt, is_shell, gamma, params.W);
                    applyBoundaryConditions(Q, Nx, Ny, Nz, is_shell, S_shell, params.W);
                    
                    if (iter % printFreq == 0) {
                        double F = computeTotalFreeEnergy(Q, Nx, Ny, Nz, dx, dy, dz, params, kappa, modechoice);
                        double radiality = computeRadiality(Q, Nx, Ny, Nz);
                        energy_log << iter << "," << F << "," << radiality << "," << physical_time << "\n";
                        if (user_choice == 'y' || user_choice == 'Y'){
                        std::cout << "Iter " << iter 
                                  << "  t=" << physical_time << " s"
                                  << "  F=" << F 
                                  << "  R̄=" << radiality << std::endl;}

                        std::string nematic_filename = "output/nematic_field_iter_" + std::to_string(iter) + ".dat";
                        std::vector<NematicField> nematicField(Nx * Ny * Nz);
                        #pragma omp parallel for collapse(3)
                        for (int i = 0; i < Nx; ++i) {
                            for (int j = 0; j < Ny; ++j) {
                                for (int k = 0; k < Nz; ++k) {
                                    nematicField[idx(i, j, k)] = calculateNematicField(Q[idx(i, j, k)]);
                                }
                            }
                        }
                        saveNematicFieldToFile(nematicField, Nx, Ny, Nz, nematic_filename);

                        if (iter > 100 && prev_F != 0.0) {
                            double energy_change = std::abs((F - prev_F) / prev_F);
                            bool energy_converged = energy_change < tolerance;
                            bool radially_aligned = radiality > radiality_threshold;
                            bool time_sufficient = physical_time > min_alignment_time;
                            if (energy_converged && radially_aligned && time_sufficient) {
                                std::cout << "\n=== Convergence Achieved ===";
                                converged = true;
                                break;
                            }
                        }
                        prev_F = F;
                    }
                }
                energy_log.close();
            }
            else if (submode == 2) {
                std::ofstream energy_components_log("energy_components_vs_iteration.dat");
                energy_components_log << "iteration,bulk,elastic,total,radiality,time\n";

                bool converged = false;
                double prev_F = std::numeric_limits<double>::max();
                double physical_time = 0.0;
                double radiality_threshold = 0.999; 
                double min_alignment_time = 0.5 * tau_align;
                
                char user_choice;
                std::cout << "Do you want output in the console every " << printFreq << " iterations? (y/n): ";
                std::cin >> user_choice;

                for (int iter = 0; iter < maxIterations; ++iter) {
                    physical_time += dt;
                    computeLaplacian(Q, laplacianQ, Nx, Ny, Nz, dx, dy, dz);
                    computeChemicalPotential(Q, mu, laplacianQ, Nx, Ny, Nz, dx, dy, dz, params, kappa, modechoice);
                    double S_shell = (T > T_star) ? 0.0 : S0;
                    applyWeakAnchoringPenalty(mu, Q, Nx, Ny, Nz, is_shell, S_shell, params.W);
                    updateQTensor(Q, mu, Nx, Ny, Nz, dt, is_shell, gamma, params.W);
                    applyBoundaryConditions(Q, Nx, Ny, Nz, is_shell, S_shell, params.W);
                    
                    if (iter % printFreq == 0) {
                        EnergyComponents ec = computeEnergyComponents(Q, Nx, Ny, Nz, dx, dy, dz, params, kappa, modechoice);
                        double radiality = computeRadiality(Q, Nx, Ny, Nz);
                        energy_components_log << iter << "," << ec.bulk << "," << ec.elastic << "," << ec.total() 
                                             << "," << radiality << "," << physical_time << "\n";
                        if (user_choice == 'y' || user_choice == 'Y'){
                        std::cout << "Iter " << iter
                                  << "  t=" << physical_time << " s"
                                  << "  F=" << ec.total()
                                  << "  R̄=" << radiality << std::endl;}

                        std::string nematic_filename = "output/nematic_field_iter_" + std::to_string(iter) + ".dat";
                        std::vector<NematicField> nematicField(Nx * Ny * Nz);
                        #pragma omp parallel for collapse(3)
                        for (int i = 0; i < Nx; ++i) {
                            for (int j = 0; j < Ny; ++j) {
                                for (int k = 0; k < Nz; ++k) {
                                    nematicField[idx(i, j, k)] = calculateNematicField(Q[idx(i, j, k)]);
                                }
                            }
                        }
                        saveNematicFieldToFile(nematicField, Nx, Ny, Nz, nematic_filename);

                        if (iter > 100 && prev_F != 0.0) {
                            double energy_change = std::abs((ec.total() - prev_F) / prev_F);
                            bool energy_converged = energy_change < tolerance;
                            if (energy_converged && radiality > radiality_threshold && physical_time > min_alignment_time) {
                                std::cout << "\n=== Convergence Achieved ===";
                                converged = true;
                                break;
                            }
                        }
                        prev_F = ec.total();
                    }
                }
                energy_components_log.close();
            }

            std::vector<NematicField> finalNematicField(Nx * Ny * Nz);
            #pragma omp parallel for collapse(3)
            for (int i = 0; i < Nx; ++i) {
                for (int j = 0; j < Ny; ++j) {
                    for (int k = 0; k < Nz; ++k) {
                        finalNematicField[idx(i, j, k)] = calculateNematicField(Q[idx(i, j, k)]);
                    }
                }
            }
            saveNematicFieldToFile(finalNematicField, Nx, Ny, Nz, "nematic_field_final.dat");
            saveQTensorToFile(Q, Nx, Ny, Nz, "Qtensor_output_final.dat");
        }
        std::cout << "\nSimulation finished." << std::endl;
        std::cout << "Would you like to run another simulation? (y/n): ";
        std::string again;
        std::getline(std::cin, again);
        run_again = (!again.empty() && (again[0] == 'y' || again[0] == 'Y'));

    } while (run_again);

    std::cout << "Exiting. Press enter to close." << std::endl;
    std::cin.get();
    return 0;
}

static double computeTransitionTemperature(double a, double b, double c, double T_star, int modechoice) {
    // Uses the uniaxial scalar reduction of your bulk convention.
    // modechoice==1 -> f(S)=a dT S^2 - (b/3) S^3 + (c/2) S^4  => dT_NI = b^2/(18 a c)
    // modechoice==2 -> f(S)=a dT S^2 - b S^3 + c S^4          => dT_NI = b^2/(4 a c)
    if (a == 0.0 || c == 0.0) return T_star;
    if (modechoice == 2) {
        return T_star + (b * b) / (4.0 * a * c);
    }
    return T_star + (b * b) / (18.0 * a * c);
}

static void addIsotropicNoise(std::vector<QTensor>& Q, int Nx, int Ny, int Nz, double amplitude) {
    if (amplitude <= 0.0) return;
    double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0;
    double R = std::min({ (double)Nx, (double)Ny, (double)Nz }) / 2.0;
    auto idx = [&](int i, int j, int k) { return k + Nz * (j + Ny * i); };

    #pragma omp parallel
    {
        std::mt19937 gen(std::random_device{}() + 1337u * (unsigned)omp_get_thread_num());
        std::normal_distribution<double> dist(0.0, amplitude);

        #pragma omp for collapse(3)
        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                for (int k = 0; k < Nz; ++k) {
                    double x = i - cx, y = j - cy, z = k - cz;
                    double r = std::sqrt(x * x + y * y + z * z);
                    if (r >= R) continue;

                    QTensor& q = Q[idx(i, j, k)];
                    q.Qxx += dist(gen);
                    q.Qxy += dist(gen);
                    q.Qxz += dist(gen);
                    q.Qyy += dist(gen);
                    q.Qyz += dist(gen);
                }
            }
        }
    }
}