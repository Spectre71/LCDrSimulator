#include "QSR.h"
#pragma warning(disable: 4996) // Disable deprecation warnings for fopen
void initializeGrid(int Nx, int Ny, int Nz, double dx, double dy, double dz) {
    // For now, nothing to do since grid is implicit in Q array indices.
    // You could allocate a grid of Grid structs here if needed.
}

void initializeQTensor(std::vector<std::vector<std::vector<QTensor>>>& Q, double q0, double theta0, double phi0) {
    int Nx = Q.size();
    int Ny = Q[0].size();
    int Nz = Q[0][0].size();

    double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0; // Center of droplet
    double R = std::min({ Nx, Ny, Nz }) / 2.0;

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                // Position relative to center
                double x = i - cx;
                double y = j - cy;
                double z = k - cz;
                double r = std::sqrt(x * x + y * y + z * z);

                QTensor q;
                if (r<R && r > 1e-8) {
                    double nx = x / r;
                    double ny = y / r;
                    double nz = z / r;
                    double S = q0;

                    // Q_ij = S * (n_i n_j - (1/3) δ_ij)
                    q.Qxx = S * (nx * nx - 1.0 / 3.0);
                    q.Qxy = S * (nx * ny);
                    q.Qxz = S * (nx * nz);
                    q.Qyx = q.Qxy;
                    q.Qyy = S * (ny * ny - 1.0 / 3.0);
                    q.Qyz = S * (ny * nz);
                    q.Qzx = q.Qxz;
                    q.Qzy = q.Qyz;
                    q.Qzz = S * (nz * nz - 1.0 / 3.0);
                }
                else {
                    // At the center, director is undefined; set Q to zero
                    q = { 0,0,0,0,0,0,0,0,0 };
                }
                Q[i][j][k] = q;
            }
        }
    }
}

void computeLaplacian(const std::vector<std::vector<std::vector<QTensor>>>& Q, std::vector<std::vector<std::vector<QTensor>>>& laplacianQ) {
    int Nx = Q.size();
    int Ny = Q[0].size();
    int Nz = Q[0][0].size();

    double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    #pragma omp parallel for collapse(3)
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int k = 1; k < Nz - 1; ++k) {
                QTensor lap;
                #define LAPLACE(comp) \
                (Q[i+1][j][k].comp - 2*Q[i][j][k].comp + Q[i-1][j][k].comp)/dx2 + \
                (Q[i][j+1][k].comp - 2*Q[i][j][k].comp + Q[i][j-1][k].comp)/dy2 + \
                (Q[i][j][k+1].comp - 2*Q[i][j][k].comp + Q[i][j][k-1].comp)/dz2

                lap.Qxx = LAPLACE(Qxx);
                lap.Qxy = LAPLACE(Qxy);
                lap.Qxz = LAPLACE(Qxz);
                lap.Qyx = LAPLACE(Qyx);
                lap.Qyy = LAPLACE(Qyy);
                lap.Qyz = LAPLACE(Qyz);
                lap.Qzx = LAPLACE(Qzx);
                lap.Qzy = LAPLACE(Qzy);
                lap.Qzz = LAPLACE(Qzz);

                laplacianQ[i][j][k] = lap;
            }
        }
    }
}

double computeFreeEnergyDensity(const std::vector<std::vector<std::vector<QTensor>>>& Q) {
    int Nx = Q.size();
    int Ny = Q[0].size();
    int Nz = Q[0][0].size();
    double total = 0.0;
    #pragma omp parallel for collapse(3) reduction(+:total)
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                const QTensor& q = Q[i][j][k];
                // Tr(Q^2)
                double trQ2 = q.Qxx * q.Qxx + q.Qyy * q.Qyy + q.Qzz * q.Qzz
                    + 2 * (q.Qxy * q.Qxy + q.Qxz * q.Qxz + q.Qyz * q.Qyz);
                // Tr(Q^3)
                double trQ3 = 3.0 * (q.Qxy * q.Qyz * q.Qzx + q.Qxz * q.Qxy * q.Qyz + q.Qyz * q.Qxz * q.Qxy)
                    + q.Qxx * q.Qyy * q.Qzz
                    - q.Qxx * q.Qyz * q.Qyz
                    - q.Qyy * q.Qxz * q.Qxz
                    - q.Qzz * q.Qxy * q.Qxy;
                double f = 0.5 * A * trQ2 - (1.0 / 3.0) * B * trQ3 + 0.25 * C * trQ2 * trQ2;
				total += f * (dx * dy * dz); // Volume element
            }
        }
    }
    return total;
}

void zeroBoundary(std::vector<std::vector<std::vector<QTensor>>>& arr) {
    int Nx = arr.size(), Ny = arr[0].size(), Nz = arr[0][0].size();
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            for (int k = 0; k < Nz; ++k)
                if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1 || k == 0 || k == Nz - 1)
                    arr[i][j][k] = QTensor(); // default zero
}

void computeChemicalPotential(const std::vector<std::vector<std::vector<QTensor>>>& Q, std::vector<std::vector<std::vector<QTensor>>>& mu, std::vector<std::vector<std::vector<QTensor>>>& laplacianQ) {
    int Nx = Q.size();
    int Ny = Q[0].size();
    int Nz = Q[0][0].size();

    zeroBoundary(mu);
    zeroBoundary(laplacianQ);

    computeLaplacian(Q, laplacianQ);
    #pragma omp parallel for collapse(3)
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int k = 1; k < Nz - 1; ++k) {
                const QTensor& q = Q[i][j][k];
                QTensor h;
                // Bulk derivatives (simplified for uniaxial Q)
                double trQ2 = q.Qxx * q.Qxx + q.Qyy * q.Qyy + q.Qzz * q.Qzz
                    + 2 * (q.Qxy * q.Qxy + q.Qxz * q.Qxz + q.Qyz * q.Qyz);

                h.Qxx = -A * q.Qxx + B * (q.Qxx * q.Qxx + q.Qxy * q.Qxy + q.Qxz * q.Qxz) - C * trQ2 * q.Qxx + kappa * laplacianQ[i][j][k].Qxx;
                h.Qxy = -A * q.Qxy + B * (q.Qxx * q.Qxy + q.Qxy * q.Qyy + q.Qxz * q.Qyz) - C * trQ2 * q.Qxy + kappa * laplacianQ[i][j][k].Qxy;
                h.Qxz = -A * q.Qxz + B * (q.Qxx * q.Qxz + q.Qxy * q.Qyz + q.Qxz * q.Qzz) - C * trQ2 * q.Qxz + kappa * laplacianQ[i][j][k].Qxz;
                h.Qyx = h.Qxy;
                h.Qyy = -A * q.Qyy + B * (q.Qxy * q.Qxy + q.Qyy * q.Qyy + q.Qyz * q.Qyz) - C * trQ2 * q.Qyy + kappa * laplacianQ[i][j][k].Qyy;
                h.Qyz = -A * q.Qyz + B * (q.Qxy * q.Qxz + q.Qyy * q.Qyz + q.Qyz * q.Qzz) - C * trQ2 * q.Qyz + kappa * laplacianQ[i][j][k].Qyz;
                h.Qzx = h.Qxz;
                h.Qzy = h.Qyz;
                h.Qzz = -A * q.Qzz + B * (q.Qxz * q.Qxz + q.Qyz * q.Qyz + q.Qzz * q.Qzz) - C * trQ2 * q.Qzz + kappa * laplacianQ[i][j][k].Qzz;

                mu[i][j][k] = h;
            }
        }
    }
}

const double Gamma = 1.0; // Rotational viscosity

void updateQTensor(std::vector<std::vector<std::vector<QTensor>>>& Q, const std::vector<std::vector<std::vector<QTensor>>>& mu, double dt,
    const std::vector<std::vector<std::vector<bool>>>& is_shell) {
    int Nx = Q.size(), Ny = Q[0].size(), Nz = Q[0][0].size();
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (std::isnan(Q[i][j][k].Qxx) || std::isnan(Q[i][j][k].Qyy) || std::isnan(Q[i][j][k].Qzz)) {
                    std::cout << "NaN in Q at " << i << " " << j << " " << k << std::endl;
                }
                Q[i][j][k].Qxx += -Gamma * mu[i][j][k].Qxx * dt;
                Q[i][j][k].Qxy += -Gamma * mu[i][j][k].Qxy * dt;
                Q[i][j][k].Qxz += -Gamma * mu[i][j][k].Qxz * dt;
                Q[i][j][k].Qyx += -Gamma * mu[i][j][k].Qyx * dt;
                Q[i][j][k].Qyy += -Gamma * mu[i][j][k].Qyy * dt;
                Q[i][j][k].Qyz += -Gamma * mu[i][j][k].Qyz * dt;
                Q[i][j][k].Qzx += -Gamma * mu[i][j][k].Qzx * dt;
                Q[i][j][k].Qzy += -Gamma * mu[i][j][k].Qzy * dt;
                Q[i][j][k].Qzz += -Gamma * mu[i][j][k].Qzz * dt;
            }
        }
    }
}

void applyBoundaryConditions(std::vector<std::vector<std::vector<QTensor>>>& Q,
    const std::vector<std::vector<std::vector<bool>>>& is_shell) {
    int Nx = Q.size(), Ny = Q[0].size(), Nz = Q[0][0].size();
    double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0;
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (is_shell[i][j][k]) {
                    double x = i - cx, y = j - cy, z = k - cz;
                    double r = std::sqrt(x * x + y * y + z * z);
                    double nx = x / r, ny = y / r, nz = z / r;
                    double S = S0;
                    QTensor q;
                    q.Qxx = S * (nx * nx - 1.0 / 3.0);
                    q.Qxy = S * (nx * ny);
                    q.Qxz = S * (nx * nz);
                    q.Qyx = q.Qxy;
                    q.Qyy = S * (ny * ny - 1.0 / 3.0);
                    q.Qyz = S * (ny * nz);
                    q.Qzx = q.Qxz;
                    q.Qzy = q.Qyz;
                    q.Qzz = S * (nz * nz - 1.0 / 3.0);
                    Q[i][j][k] = q;
                }
            }
        }
    }
}

void saveQTensorToFile(const std::vector<std::vector<std::vector<QTensor>>>& Q, const std::string& filename) {
    FILE* f = fopen(filename.c_str(), "w");
    if (!f) {
		std::cerr << "Error opening file for writing: " << filename << std::endl;
		return;
    }

    int Nx = Q.size();
    int Ny = Q[0].size();
    int Nz = Q[0][0].size();

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                const QTensor& q = Q[i][j][k];
                fprintf(f, "%d %d %d %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
                    i, j, k,
                    q.Qxx, q.Qxy, q.Qxz, q.Qyx, q.Qyy, q.Qyz, q.Qzx, q.Qzy, q.Qzz);
            }
        }
    }
    fclose(f);
}

int main() {
    // Allocate Q-tensor field
    std::vector<std::vector<std::vector<QTensor>>> Q(Nx, std::vector<std::vector<QTensor>>(Ny, std::vector<QTensor>(Nz)));
    std::vector<std::vector<std::vector<QTensor>>> laplacianQ(Nx, std::vector<std::vector<QTensor>>(Ny, std::vector<QTensor>(Nz)));
    std::vector<std::vector<std::vector<QTensor>>> mu(Nx, std::vector<std::vector<QTensor>>(Ny, std::vector<QTensor>(Nz)));

    initializeGrid(Nx, Ny, Nz, dx, dy, dz);
    initializeQTensor(Q, S0, theta0, phi0);

    std::vector<std::vector<std::vector<bool>>> is_shell(Nx, std::vector<std::vector<bool>>(Ny, std::vector<bool>(Nz, false)));
    double cx = Nx / 2.0, cy = Ny / 2.0, cz = Nz / 2.0;
    double R = std::min({ Nx,Ny,Nz }) / 2.0;
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                double x = i - cx, y = j - cy, z = k - cz;
                double r = std::sqrt(x * x + y * y + z * z);
                if (std::abs(r - R) < 0.5 && r > 1e-8) {
                    is_shell[i][j][k] = true;
                }
            }
        }
    }

    for (int iter = 0; iter < maxIterations; ++iter) {
        computeChemicalPotential(Q, mu, laplacianQ);
        updateQTensor(Q, mu, dt, is_shell);
        applyBoundaryConditions(Q, is_shell);

        if (iter % 100 == 0) {
            double F = computeFreeEnergyDensity(Q);
			std::cout << "Iteration " << iter << ", Free Energy: " << F << std::endl;
        }
    }

    saveQTensorToFile(Q, "Qtensor_output.dat");
	std::cout << "Press enter to exit" << std::endl;
	std::cin.get();
    return 0;
}
