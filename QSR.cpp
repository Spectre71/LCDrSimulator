#include "QSR.h"
namespace fs = std::filesystem;
#pragma warning(disable: 4996) // Disable deprecation warnings for fopen

// Calculates the nematic field (director and order parameter) from a Q-tensor.
// Uses the Power Iteration method to find the dominant eigenvector (the director).
NematicField calculateNematicField(const QTensor& q_5comp) {
    FullQTensor q(q_5comp);

    // Handle the isotropic case (Q is all zeros)
    if (std::abs(q.Qxx) < 1e-12 && std::abs(q.Qxy) < 1e-12 && std::abs(q.Qxz) < 1e-12 &&
        std::abs(q.Qyy) < 1e-12 && std::abs(q.Qyz) < 1e-12) {
        return { 0.0, 1.0, 0.0, 0.0 }; // S=0, director is undefined (return default)
    }

    // Power Iteration to find the dominant eigenvector (director n)
    double nx = 1.0, ny = 0.0, nz = 0.0; // Initial guess for the director
    for (int i = 0; i < 10; ++i) { // 10 iterations are typically sufficient for convergence
        double q_nx = q.Qxx * nx + q.Qxy * ny + q.Qxz * nz;
        double q_ny = q.Qyx * nx + q.Qyy * ny + q.Qyz * nz;
        double q_nz = q.Qzx * nx + q.Qzy * ny + q.Qzz * nz;

        double norm = std::sqrt(q_nx * q_nx + q_ny * q_ny + q_nz * q_nz);
        if (norm < 1e-12) break; // Avoid division by zero
        nx = q_nx / norm;
        ny = q_ny / norm;
        nz = q_nz / norm;
    }

    // The largest eigenvalue (lambda_max) can be found using the Rayleigh quotient
    double lambda_max = (nx * (q.Qxx * nx + q.Qxy * ny + q.Qxz * nz) +
        ny * (q.Qyx * nx + q.Qyy * ny + q.Qyz * nz) +
        nz * (q.Qzx * nx + q.Qzy * ny + q.Qzz * nz));

    // The scalar order parameter S = 1.5 * lambda_max
    double S = 1.5 * lambda_max;

    // Ensure consistent director orientation (gauge fixing)
    if (nx < 0) {
        nx = -nx; ny = -ny; nz = -nz;
    }

    return { S, nx, ny, nz };
}

// Saves the calculated nematic field data to a file.
void saveNematicFieldToFile(const std::vector<NematicField>& nematicField, int Nx, int Ny, int Nz, const std::string& filename) {
    FILE* f = fopen(filename.c_str(), "w");
    if (!f) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    fprintf(f, "# i j k S nx ny nz\n"); // Add a header for clarity
    auto idx = [&](int i, int j, int k) { return k + Nz * (j + Ny * i); };

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                const NematicField& nf = nematicField[idx(i, j, k)];
                fprintf(f, "%d %d %d %.6e %.6e %.6e %.6e\n",
                    i, j, k, nf.S, nf.nx, nf.ny, nf.nz);
            }
        }
    }
    fclose(f);
}

void initializeQTensor(std::vector<QTensor>& Q, int Nx, int Ny, int Nz, double q0, int mode) {
    double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0; // Center of droplet
    double R = std::min({ (double)Nx, (double)Ny, (double)Nz }) / 2.0;
    auto idx = [&](int i, int j, int k) {return k + Nz * (j + Ny * i); };
    #pragma omp parallel
    {
        // Random number generators for theta (0, pi) and phi (0, 2*pi) - thread safe
        std::mt19937 gen(std::random_device{}() + omp_get_thread_num());
        std::uniform_real_distribution<double>dist_theta(0.0, PI);
        std::uniform_real_distribution<double> dist_phi(0.0, 2.0 * PI);

        #pragma omp for collapse(3)
        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                for (int k = 0; k < Nz; ++k) {

                    // Position relative to center
                    double x = i - cx, y = j - cy, z = k - cz;
                    double r = std::sqrt(x * x + y * y + z * z);

                    if (r < R && r > 1e-8) {
                        double S = q0;
                        QTensor& q = Q[idx(i, j, k)]; // Q_ij = S * (n_i \otimes n_j - (1/3) \delta_ij)

                        // Generate a random orientation (unit vec) using spherical coordinates
                        if (mode == 0) {
                            double theta = dist_theta(gen); // Polar angle
                            double phi = dist_phi(gen); // Azimuthal angle
                            double nx = std::sin(theta) * std::cos(phi);
                            double ny = std::sin(theta) * std::sin(phi);
                            double nz = std::cos(theta);

                            q.Qxx = S * (nx * nx - 1.0 / 3.0);
                            q.Qxy = S * (nx * ny);
                            q.Qxz = S * (nx * nz);
                            q.Qyy = S * (ny * ny - 1.0 / 3.0);
                            q.Qyz = S * (ny * nz);
                        }
                        //Radial mode: director is radial, pointing outward
                        else if (mode == 1) {
                            double nx = x / r;
                            double ny = y / r;
                            double nz = z / r;

                            q.Qxx = S * (nx * nx - 1.0 / 3.0);
                            q.Qxy = S * (nx * ny);
                            q.Qxz = S * (nx * nz);
                            q.Qyy = S * (ny * ny - 1.0 / 3.0);
                            q.Qyz = S * (ny * nz);
                        }
                    }
                    else {
                        // At the center, director is undefined; Q set to zero
                        Q[idx(i, j, k)] = QTensor(); // zero init
                    }
                }
            }
        }
    }
}

void computeLaplacian(const std::vector<QTensor>& Q, std::vector<QTensor>& laplacianQ, int Nx, int Ny, int Nz, double dx, double dy, double dz) {
    double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
	auto idx = [&](int i, int j, int k) { return k + Nz * (j + Ny * i); };

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                // Check if the point is an interior point where the stencil is valid
                if (i > 0 && i < Nx - 1 && j > 0 && j < Ny - 1 && k > 0 && k < Nz - 1) {
                    QTensor& lap = laplacianQ[idx(i, j, k)];
                    const QTensor& q_center = Q[idx(i, j, k)];

                    #define LAPLACE(comp) \
                    (Q[idx(i+1,j,k)].comp-2*q_center.comp+Q[idx(i-1,j,k)].comp) / dx2 + \
                    (Q[idx(i,j+1,k)].comp-2*q_center.comp+Q[idx(i,j-1,k)].comp) / dy2 + \
                    (Q[idx(i,j,k+1)].comp-2*q_center.comp+Q[idx(i,j,k-1)].comp) / dz2

                    lap.Qxx = LAPLACE(Qxx);
                    lap.Qxy = LAPLACE(Qxy);
                    lap.Qxz = LAPLACE(Qxz);
                    lap.Qyy = LAPLACE(Qyy);
                    lap.Qyz = LAPLACE(Qyz);
                }
                else {
                    // For points on the simulation box boundary, set Laplacian to zero.
                    // This implies a zero-flux boundary condition for the simulation box.
                    laplacianQ[idx(i, j, k)] = QTensor();
                }
            }
        }
    }
}

double computeTotalFreeEnergy(const std::vector<QTensor>& Q, int Nx, int Ny, int Nz, double dx, double dy, double dz, double A,
    double B, double C, double kappa, double K_field, double S0) {
    double total_energy = 0.0;
	auto idx = [&](int i, int j, int k) { return k + Nz * (j + Ny * i); };
    double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0;

    #pragma omp parallel for collapse(3) reduction(+:total_energy)
    for (int i = 1; i < Nx-1; ++i) {
        for (int j = 1; j < Ny-1; ++j) {
            for (int k = 1; k < Nz-1; ++k) {
                FullQTensor q(Q[idx(i, j, k)]);

				//Bulk free energy density
                double trQ2 = q.Qxx * q.Qxx + q.Qyy * q.Qyy + q.Qzz * q.Qzz
                    + 2 * (q.Qxy * q.Qxy + q.Qxz * q.Qxz + q.Qyz * q.Qyz);

                // Calculate Q^2 for Tr(Q^3)
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

                // Calculate Tr(Q^3) = Tr(Q * Q^2)
                double trQ3 = q.Qxx * q2.Qxx + q.Qxy * q2.Qyx + q.Qxz * q2.Qzx +
                              q.Qyx * q2.Qxy + q.Qyy * q2.Qyy + q.Qyz * q2.Qzy +
                              q.Qzx * q2.Qxz + q.Qzy * q2.Qyz + q.Qzz * q2.Qzz;

                double f_bulk = (0.5 * A * trQ2) - (B / 3.0 * trQ3) + (0.25 * C * trQ2 * trQ2);

				// Elastic free energy density
                double grad_Q_sq = 0;
                const double inv_2dx = 1.0 / (2.0 * dx);
                const double inv_2dy = 1.0 / (2.0 * dy);
                const double inv_2dz = 1.0 / (2.0 * dz);

                // Pre-fetch neighbor Q-tensors
                const QTensor& q_ip1 = Q[idx(i + 1, j, k)];
                const QTensor& q_im1 = Q[idx(i - 1, j, k)];
                const QTensor& q_jp1 = Q[idx(i, j + 1, k)];
                const QTensor& q_jm1 = Q[idx(i, j - 1, k)];
                const QTensor& q_kp1 = Q[idx(i, j, k + 1)];
                const QTensor& q_km1 = Q[idx(i, j, k - 1)];

                // Macro to calculate gradient squared for one component
                #define CALC_GRAD_SQ(comp) \
                    (std::pow((q_ip1.comp - q_im1.comp) * inv_2dx, 2) + \
                     std::pow((q_jp1.comp - q_jm1.comp) * inv_2dy, 2) + \
                     std::pow((q_kp1.comp - q_km1.comp) * inv_2dz, 2))

                // Sum of squares for independent components
                grad_Q_sq += CALC_GRAD_SQ(Qxx);
                grad_Q_sq += CALC_GRAD_SQ(Qyy);

                //off - diagonal components, multiplied by 2 for symmetry
                grad_Q_sq += 2 * CALC_GRAD_SQ(Qxy);
                grad_Q_sq += 2 * CALC_GRAD_SQ(Qxz);
                grad_Q_sq += 2 * CALC_GRAD_SQ(Qyz);

                // Calculate for dependent Qzz component
                double dQzz_dx = (-(q_ip1.Qxx + q_ip1.Qyy) + (q_im1.Qxx + q_im1.Qyy)) * inv_2dx;
                double dQzz_dy = (-(q_jp1.Qxx + q_jp1.Qyy) + (q_jm1.Qxx + q_jm1.Qyy)) * inv_2dy;
                double dQzz_dz = (-(q_kp1.Qxx + q_kp1.Qyy) + (q_km1.Qxx + q_km1.Qyy)) * inv_2dz;
                grad_Q_sq += dQzz_dx * dQzz_dx + dQzz_dy * dQzz_dy + dQzz_dz * dQzz_dz;
                
				double f_elastic = 0.5 * kappa * grad_Q_sq;

                // field field anchoring energy
				double f_field = 0.0;
                if (K_field > 1e-9) {
                    double x = (i - cx) * dx;
					double y = (j - cy) * dy;
					double z = (k - cz) * dz;
					double r = std::sqrt(x * x + y * y + z * z);

                    if (r > 1e-9) {
						double nx_r = x / r;
						double ny_r = y / r;
						double nz_r = z / r;

                        FullQTensor q_target;
                        q_target.Qxx = S0 * (nx_r * nx_r - 1.0 / 3.0);
                        q_target.Qxy = S0 * (nx_r * ny_r);
                        q_target.Qxz = S0 * (nx_r * nz_r);
                        q_target.Qyy = S0 * (ny_r * ny_r - 1.0 / 3.0);
                        q_target.Qyz = S0 * (ny_r * nz_r);
                        q_target.Qyx = q_target.Qxy;
                        q_target.Qzx = q_target.Qxz;
                        q_target.Qzy = q_target.Qyz;
                        q_target.Qzz = -q_target.Qxx - q_target.Qyy;

                        double q_dot_q_target=
                            q.Qxx * q_target.Qxx + q.Qyy * q_target.Qyy + q.Qzz * q_target.Qzz +
                            2 * (q.Qxy * q_target.Qxy + q.Qxz * q_target.Qxz + q.Qyz * q_target.Qyz);

                        f_field = - K_field * q_dot_q_target;
                    }
                }

                total_energy += (f_bulk + f_elastic + f_field) * (dx * dy * dz); // Volume element
            }
        }
    }
    return total_energy;
}

void computeChemicalPotential(const std::vector<QTensor>& Q, std::vector<QTensor>& mu,
    const std::vector<QTensor>& laplacianQ, int Nx, int Ny, int Nz, double dx, double dy, double dz,
    double A, double B, double C, double kappa, double K_field, double S0) {
	auto idx = [&](int i, int j, int k) { return k + Nz * (j + Ny * i); };
    double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0;

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
				FullQTensor q(Q[idx(i, j, k)]);
				const QTensor& lap_q = laplacianQ[idx(i, j, k)];
				QTensor& h = mu[idx(i, j, k)]; // h is mum which is the gradient dF/dQ

                // Bulk derivatives (simplified for uniaxial Q)
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

                h.Qxx = A * q.Qxx - B * (q2.Qxx - trQ2 / 3.0) + C * trQ2 * q.Qxx - kappa * lap_q.Qxx;
                h.Qxy = A * q.Qxy - B * q2.Qxy + C * trQ2 * q.Qxy - kappa * lap_q.Qxy;
                h.Qxz = A * q.Qxz - B * q2.Qxz + C * trQ2 * q.Qxz - kappa * lap_q.Qxz;
                h.Qyy = A * q.Qyy - B * (q2.Qyy - trQ2 / 3.0) + C * trQ2 * q.Qyy - kappa * lap_q.Qyy;
                h.Qyz = A * q.Qyz - B * q2.Qyz + C * trQ2 * q.Qyz - kappa * lap_q.Qyz;

                // field field contribution to the chemical potential
                if (K_field > 1e-9) {
                    double x = (i - cx) * dx;
                    double y = (j - cy) * dy;
                    double z = (k - cz) * dz;
                    double r = std::sqrt(x * x + y * y + z * z);

                    if (r > 1e-9) {
                        double nx_r = x / r;
                        double ny_r = y / r;
                        double nz_r = z / r;

                        QTensor q_target;
                        q_target.Qxx = S0 * (nx_r * nx_r - 1.0 / 3.0);
                        q_target.Qxy = S0 * (nx_r * ny_r);
                        q_target.Qxz = S0 * (nx_r * nz_r);
                        q_target.Qyy = S0 * (ny_r * ny_r - 1.0 / 3.0);
                        q_target.Qyz = S0 * (ny_r * nz_r);

                        h.Qxx -= K_field * q_target.Qxx;
                        h.Qxy -= K_field * q_target.Qxy;
                        h.Qxz -= K_field * q_target.Qxz;
                        h.Qyy -= K_field * q_target.Qyy;
                        h.Qyz -= K_field * q_target.Qyz;
                    }
                }

            }
        }
    }
}

void updateQTensor(std::vector<QTensor>& Q, const std::vector<QTensor>& mu, int Nx, int Ny, int Nz,
    double dt, const std::vector<bool>& is_shell, double gamma) {
	auto idx = [&](int i, int j, int k) { return k + Nz * (j + Ny * i); };
    double R = std::min({ (double)Nx, (double)Ny, (double)Nz }) / 2.0;
    double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0;

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                // Skip updates for shell points
                if(is_shell[idx(i, j, k)]) {
                    continue;
				}

                /*Only update points within the droplet radius.
                This prevents the simulation from "leaking" into the vacuum.*/
                double x = i - cx;
				double y = j - cy;
				double z = k - cz;
				double r = std::sqrt(x * x + y * y + z * z);
                if (r > R) {
                    continue; // Skip points outside the droplet
				}
				const QTensor& current_mu = mu[idx(i, j, k)];
				QTensor& current_Q = Q[idx(i, j, k)];
                
                current_Q.Qxx -= gamma * current_mu.Qxx * dt;
                current_Q.Qxy -= gamma * current_mu.Qxy * dt;
                current_Q.Qxz -= gamma * current_mu.Qxz * dt;
                current_Q.Qyy -= gamma * current_mu.Qyy * dt;
                current_Q.Qyz -= gamma * current_mu.Qyz * dt;
            }
        }
    }
}

void applyBoundaryConditions(std::vector<QTensor>& Q, int Nx, int Ny, int Nz, const std::vector<bool>& is_shell, double S0) {
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

void saveQTensorToFile(const std::vector<QTensor>& Q, int Nx, int Ny, int Nz, const std::string& filename) {
    FILE* f = fopen(filename.c_str(), "w");
    if (!f) {
		std::cerr << "Error opening file for writing: " << filename << std::endl;
		return;
    }
	auto idx = [&](int i, int j, int k) { return k + Nz * (j + Ny * i); };

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                FullQTensor q(Q[idx(i, j, k)]);
                fprintf(f, "%d %d %d %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
                    i, j, k,
                    q.Qxx, q.Qxy, q.Qxz, q.Qyx, q.Qyy, q.Qyz, q.Qzx, q.Qzy, q.Qzz);
            }
        }
    }
    fclose(f);
}

template<typename T>
T prompt_with_default(const std::string& prompt, T default_value) {
    std::string line;
    std::cout << prompt << " [" << default_value << "]: ";
    std::getline(std::cin, line);
    if (line.empty()) {
        return default_value;
    }
    std::istringstream iss(line);
    T value;
    if (!(iss >> value)) {
        return default_value;
    }
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
                if (r < R) { // Only average inside the droplet
                    NematicField nf = calculateNematicField(Q[idx(i, j, k)]);
                    total_S += nf.S;
                    count++;
                }
            }
        }
    }
    return (count > 0) ? (total_S / count) : 0.0;
}

EnergyComponents computeEnergyComponents(const std::vector<QTensor>& Q, int Nx, int Ny, int Nz, double dx, double dy, double dz, double A,
    double B, double C, double kappa, double K_field, double S0) {
    EnergyComponents ec;
    auto idx = [&](int i, int j, int k) { return k + Nz * (j + Ny * i); };
    double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0;
    double bulk = 0.0, elastic = 0.0, field = 0.0;

    #pragma omp parallel for collapse(3) reduction(+:bulk,elastic,field)
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int k = 1; k < Nz - 1; ++k) {
                FullQTensor q(Q[idx(i, j, k)]);
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

                double f_bulk = (0.5 * A * trQ2) - (B / 3.0 * trQ3) + (0.25 * C * trQ2 * trQ2);

                double grad_Q_sq = 0;
                const double inv_2dx = 1.0 / (2.0 * dx);
                const double inv_2dy = 1.0 / (2.0 * dy);
                const double inv_2dz = 1.0 / (2.0 * dz);

                const QTensor& q_ip1 = Q[idx(i + 1, j, k)];
                const QTensor& q_im1 = Q[idx(i - 1, j, k)];
                const QTensor& q_jp1 = Q[idx(i, j + 1, k)];
                const QTensor& q_jm1 = Q[idx(i, j - 1, k)];
                const QTensor& q_kp1 = Q[idx(i, j, k + 1)];
                const QTensor& q_km1 = Q[idx(i, j, k - 1)];

                #define CALC_GRAD_SQ(comp) \
                    (std::pow((q_ip1.comp - q_im1.comp) * inv_2dx, 2) + \
                     std::pow((q_jp1.comp - q_jm1.comp) * inv_2dy, 2) + \
                     std::pow((q_kp1.comp - q_km1.comp) * inv_2dz, 2))

                grad_Q_sq += CALC_GRAD_SQ(Qxx);
                grad_Q_sq += CALC_GRAD_SQ(Qyy);
                grad_Q_sq += 2 * CALC_GRAD_SQ(Qxy);
                grad_Q_sq += 2 * CALC_GRAD_SQ(Qxz);
                grad_Q_sq += 2 * CALC_GRAD_SQ(Qyz);

                double dQzz_dx = (-(q_ip1.Qxx + q_ip1.Qyy) + (q_im1.Qxx + q_im1.Qyy)) * inv_2dx;
                double dQzz_dy = (-(q_jp1.Qxx + q_jp1.Qyy) + (q_jm1.Qxx + q_jm1.Qyy)) * inv_2dy;
                double dQzz_dz = (-(q_kp1.Qxx + q_kp1.Qyy) + (q_km1.Qxx + q_km1.Qyy)) * inv_2dz;
                grad_Q_sq += dQzz_dx * dQzz_dx + dQzz_dy * dQzz_dy + dQzz_dz * dQzz_dz;

                double f_elastic = 0.5 * kappa * grad_Q_sq;

                double f_field = 0.0;
                if (K_field > 1e-9) {
                    double x = (i - cx) * dx;
                    double y = (j - cy) * dy;
                    double z = (k - cz) * dz;
                    double r = std::sqrt(x * x + y * y + z * z);

                    if (r > 1e-9) {
                        double nx_r = x / r;
                        double ny_r = y / r;
                        double nz_r = z / r;

                        FullQTensor q_target;
                        q_target.Qxx = S0 * (nx_r * nx_r - 1.0 / 3.0);
                        q_target.Qxy = S0 * (nx_r * ny_r);
                        q_target.Qxz = S0 * (nx_r * nz_r);
                        q_target.Qyy = S0 * (ny_r * ny_r - 1.0 / 3.0);
                        q_target.Qyz = S0 * (ny_r * nz_r);
                        q_target.Qyx = q_target.Qxy;
                        q_target.Qzx = q_target.Qxz;
                        q_target.Qzy = q_target.Qyz;
                        q_target.Qzz = -q_target.Qxx - q_target.Qyy;

                        double q_dot_q_target =
                            q.Qxx * q_target.Qxx + q.Qyy * q_target.Qyy + q.Qzz * q_target.Qzz +
                            2 * (q.Qxy * q_target.Qxy + q.Qxz * q_target.Qxz + q.Qyz * q_target.Qyz);

                        f_field = -K_field * q_dot_q_target;
                    }
                }

                double dV = dx * dy * dz;
                bulk += f_bulk * dV;
                elastic += f_elastic * dV;
                field += f_field * dV;
            }
        }
    }
    ec.bulk = bulk;
    ec.elastic = elastic;
    ec.field = field;
    return ec;
}

int main() {
    bool run_again = false;
    do {
        // Default values
        int Nx_default = 100, Ny_default = 100, Nz_default = 100, maxIterations_default = 100000;
        double dx_default = 1e-8, dy_default = 1e-8, dz_default = 1e-8;
        double S0_default = 0.5;
        double A_default = -0.172e6, B_default = 2.12e6, C_default = 1.73e6; // A_default is at phase transition \sim 308 K
        double kappa_default = 6.5e-12, gamma_default = 0.1, dt_default = 1e-16;
        double T_default = 298.0, alpha_default = 0.172e6, T_star_default = 308.0;
        double K_field_default = 1e6; // field coupling constant
        int mode_default = 1; // 0 = random, 1 = radial

        // Prompt user, using defaults if input is empty
        std::cout << "\033[1mQ-tensor evolution simulation for a 5CB liquid crystal droplet\033[0m\n" << std::endl;
        std::cout << "Press {\033[1;33mEnter\033[0m} to use [\033[1;34mdefault\033[0m] values, or input your own.\n" << std::endl;

        std::cout << "Select initialization mode for Q-tensor field: [0] random, [1] radial (default: 1): ";
        std::string mode_input;
        std::getline(std::cin, mode_input);
        int mode = mode_default;
        if (!mode_input.empty()) {
            try {
                mode = std::stoi(mode_input);
            }
            catch (...) {
                mode = mode_default;
            }
        }
        if (mode != 0 && mode != 1)mode = mode_default;
        std::cout << "Initialization mode: " << (mode == 0 ? "random" : "radial") << std::endl;
        std::cout << "\n";

        int Nx = prompt_with_default("Enter grid size Nx", Nx_default);
        int Ny = prompt_with_default("Enter grid size Ny", Ny_default);
        int Nz = prompt_with_default("Enter grid size Nz", Nz_default);
        double dx = prompt_with_default("Enter grid spacing dx", dx_default);
        double dy = prompt_with_default("Enter grid spacing dy", dy_default);
        double dz = prompt_with_default("Enter grid spacing dz", dz_default);
        double S0 = prompt_with_default("Enter Q-tensor parameter S0", S0_default);
        double alpha = prompt_with_default("Enter alpha (J/m^3/K)", alpha_default);
        double T_star = prompt_with_default("Enter T* (K)", T_star_default);
        double K_field = prompt_with_default("Enter field field coupling constant K (J/m^3)", K_field_default);
        double B = prompt_with_default("Enter Landau-De Gennes parameter B", B_default);
        double C = prompt_with_default("Enter Landau-De Gennes parameter C", C_default);
        double kappa = prompt_with_default("Enter elastic constant kappa", kappa_default);
        double gamma = prompt_with_default("Enter rotational viscosity gamma", gamma_default); // Rotational viscosity (Pa.s) for 5CB at some temp.
        int maxIterations = prompt_with_default("Enter number of iterations", maxIterations_default);
        int printFreq = prompt_with_default("Enter print frequency (iterations)", 200);
        double tolerance = prompt_with_default("Enter convergence tolerance (relative change)", 1e-2);

        // vecs to allocate Q-tensor field to
        std::vector<QTensor> Q(Nx * Ny * Nz);
        std::vector<QTensor> laplacianQ(Nx * Ny * Nz);
        std::vector<QTensor> mu(Nx * Ny * Nz);
        std::vector<bool> is_shell(Nx * Ny * Nz, false);

        // setup shell for boundary conditions
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

        // --- Simulation Mode Selection ---
        std::cout << "\nSelect simulation mode:\n";
        std::cout << "  [1] Single Temperature\n";
        std::cout << "  [2] Temperature Range\n";
        int sim_mode = prompt_with_default("Enter mode", 1);

        if (sim_mode == 2) {
            // --- Temperature Range Mode ---
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

            for (double T = T_start; T <= T_end; T += T_step) {
                std::cout << "\n--- Running simulation for T = " << T << " K ---\n";
                double A = alpha * (T - T_star);
                std::string temp_dir = "output_temp_sweep/T_" + std::to_string(T);
                fs::create_directory(temp_dir);

                double min_dx = std::min({ dx, dy, dz });
                double D = kappa / gamma;
                double dt = 0.95 * (min_dx * min_dx) / (6.0 * D);

                bool converged = false;
                double prev_F = std::numeric_limits<double>::max();
                for (int iter = 0; iter < maxIterations; ++iter) {
                    computeLaplacian(Q, laplacianQ, Nx, Ny, Nz, dx, dy, dz);
                    computeChemicalPotential(Q, mu, laplacianQ, Nx, Ny, Nz, dx, dy, dz, A, B, C, kappa, K_field, S0);
                    updateQTensor(Q, mu, Nx, Ny, Nz, dt, is_shell, gamma);
                    applyBoundaryConditions(Q, Nx, Ny, Nz, is_shell, S0);

                    if (iter > 0 && iter % printFreq == 0) {
                        double F = computeTotalFreeEnergy(Q, Nx, Ny, Nz, dx, dy, dz, A, B, C, kappa, K_field, S0);
                        std::cout << "  Iter " << iter << ", Free Energy: " << F << std::endl;
                        if (std::abs((F - prev_F) / prev_F) < tolerance) {
                            std::cout << "  Convergence reached at iteration " << iter << std::endl;
                            converged = true;
                            break;
                        }
                        prev_F = F;
                    }
                }

                double final_F = computeTotalFreeEnergy(Q, Nx, Ny, Nz, dx, dy, dz, A, B, C, kappa, K_field, S0);
                double avg_S = calculateAverageS(Q, Nx, Ny, Nz);
                sweep_log << T << "," << final_F << "," << avg_S << "\n";

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
            // --- Single Temperature Mode ---
            int submode_default = 1; // Default to full energy calculation
            std::cout << "Select submode: [1] full energy, [2] energy components (default: 1): ";
            std::string submode_input;
            std::getline(std::cin, submode_input);
            int submode = submode_default;
            if (!submode_input.empty()) {
                try {
                    submode = std::stoi(submode_input);
                }
                catch (...) {
                    submode = submode_default;
                }
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

            double T = prompt_with_default("Enter temperature T (K)", T_default);
            double A;
            std::cout << "\nUse default A (" << A_default << ") [d], or computed from T, alpha and T* [c]? (d/c): ";
            std::string choice;
            std::getline(std::cin, choice);
            if (choice == "c" || choice == "C") {
                A = alpha * (T - T_star);
                std::cout << "Computed A = " << A << std::endl;
            }
            else if (choice == "d" || choice == "D") {
                A = A_default;
                std::cout << "Using default A = " << A << std::endl;
            }
            else {
                std::cout << "Invalid choice. Using default A = " << A_default << std::endl;
                A = A_default;
            }
            std::cout << "Using A = " << A << std::endl;

            double min_dx = std::min({ dx, dy, dz });
            double D = kappa / gamma;
            double dt_max = 0.95 * (min_dx * min_dx) / (6.0 * D);
            std::cout << "\nThe maximum stable time step dt is: " << dt_max << " seconds." << std::endl;
            double dt = prompt_with_default("Enter time step dt", dt_default);
            if (dt > dt_max) {
                std::cout << "Warning: dt is too large. Setting to max stable value: " << dt_max << std::endl;
                dt = dt_max;
            }

            initializeQTensor(Q, Nx, Ny, Nz, S0, mode);
            if (submode == 1) {
                std::ofstream energy_log("free_energy_vs_iteration.dat");
                energy_log << "iteration,free_energy\n";

                bool converged = false;
                double prev_F = std::numeric_limits<double>::max();
                for (int iter = 0; iter < maxIterations; ++iter) {
                    computeLaplacian(Q, laplacianQ, Nx, Ny, Nz, dx, dy, dz);
                    computeChemicalPotential(Q, mu, laplacianQ, Nx, Ny, Nz, dx, dy, dz, A, B, C, kappa, K_field, S0);
                    updateQTensor(Q, mu, Nx, Ny, Nz, dt, is_shell, gamma);
                    applyBoundaryConditions(Q, Nx, Ny, Nz, is_shell, S0);

                    if (iter % printFreq == 0) {
                        double F = computeTotalFreeEnergy(Q, Nx, Ny, Nz, dx, dy, dz, A, B, C, kappa, K_field, S0);
                        energy_log << iter << "," << F << "\n";
                        std::cout << "Iteration " << iter << ", Free Energy: " << F << std::endl;

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
                            if (std::abs((F - prev_F) / prev_F) < tolerance) {
                                std::cout << "Convergence reached at iteration " << iter << std::endl;
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
                energy_components_log << "iteration,bulk,elastic,field,total\n";

                bool converged = false;
                double prev_F = std::numeric_limits<double>::max();
                for (int iter = 0; iter < maxIterations; ++iter) {
                    computeLaplacian(Q, laplacianQ, Nx, Ny, Nz, dx, dy, dz);
                    computeChemicalPotential(Q, mu, laplacianQ, Nx, Ny, Nz, dx, dy, dz, A, B, C, kappa, K_field, S0);
                    updateQTensor(Q, mu, Nx, Ny, Nz, dt, is_shell, gamma);
                    applyBoundaryConditions(Q, Nx, Ny, Nz, is_shell, S0);

                    if (iter % printFreq == 0) {
                        EnergyComponents ec = computeEnergyComponents(Q, Nx, Ny, Nz, dx, dy, dz, A, B, C, kappa, K_field, S0);
                        energy_components_log << iter << "," << ec.bulk << "," << ec.elastic << "," << ec.field << "," << ec.total() << "\n";
                        std::cout << "Iteration " << iter << ", Free Energy: " << ec.total() << std::endl;

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
                            if (std::abs((ec.total() - prev_F) / prev_F) < tolerance) {
                                std::cout << "Convergence reached at iteration " << iter << std::endl;
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

    }while (run_again);

    std::cout << "Exiting. Press enter to close." << std::endl;
    std::cin.get();
    return 0;
}