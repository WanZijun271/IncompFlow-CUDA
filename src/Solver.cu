#include "Solver.h"
#include "config.h"
#include "constants.h"
#include "kernels.cuh"
#include <cmath>
#include <fstream>
#include <iostream>
#include <filesystem>

using namespace std;

Solver::Solver(scalar u, scalar v, scalar w, scalar p, scalar temp) {

    _u.assign(nx * ny * nz, u);
    _v.assign(nx * ny * nz, v);
    _w.assign(nx * ny * nz, w);

    _p.assign(nx * ny * nz, p);

    _temp.assign(nx * ny * nz, temp);     // initialize temperature field

    // initialize coefficient
    // _coef.assign(nx * ny * nz * (2 + 2 * dim), 0);
    // calcCoef();    // calculate coefficient
}

void Solver::solve() {

    size_t fieldSize = nx * ny * nz * sizeof(scalar);
    size_t coefSize = nx * ny * nz * (2+2*dim) * sizeof(scalar);

    scalar *u_dev, *v_dev, *w_dev, *p_dev, *temp_dev;
    cudaMalloc(&u_dev, fieldSize);
    cudaMalloc(&v_dev, fieldSize);
    cudaMalloc(&w_dev, fieldSize);
    cudaMalloc(&p_dev, fieldSize);
    cudaMalloc(&temp_dev, fieldSize);

    cudaMemcpy(u_dev, _u.data(), fieldSize, cudaMemcpyHostToDevice);
    cudaMemcpy(v_dev, _v.data(), fieldSize, cudaMemcpyHostToDevice);
    cudaMemcpy(w_dev, _w.data(), fieldSize, cudaMemcpyHostToDevice);
    cudaMemcpy(p_dev, _p.data(), fieldSize, cudaMemcpyHostToDevice);
    cudaMemcpy(temp_dev, _temp.data(), fieldSize, cudaMemcpyHostToDevice);

    size_t ufSize = (nx+1) * ny * nz * sizeof(scalar);
    size_t vfSize = nx * (ny+1) * nz * sizeof(scalar);
    size_t wfSize = nx * ny * (nz+1) * sizeof(scalar);

    scalar *uf_dev, *vf_dev, *wf_dev;
    cudaMalloc(&uf_dev, ufSize);
    cudaMalloc(&vf_dev, vfSize);
    cudaMalloc(&wf_dev, wfSize);

    cudaMemset(uf_dev, 0.0, ufSize);
    cudaMemset(vf_dev, 0.0, vfSize);
    cudaMemset(wf_dev, 0.0, wfSize);

    for (int it = 0; it < niter; ++it) {

        if (it == 0) {
            initVelOnFace(u_dev, v_dev, w_dev, uf_dev, vf_dev, wf_dev);
        }

    }

    cudaFree(u_dev);
    cudaFree(v_dev);
    cudaFree(w_dev);
    cudaFree(p_dev);
    cudaFree(temp_dev);

    cudaFree(uf_dev);
    cudaFree(vf_dev);
    cudaFree(wf_dev);
}

/* void Solver::pointJacobiSolver() {
    pointJacobiIterate(_temp, _coef);
}
 */
/* void Solver::GaussSeidelSolver() {
    GaussSeidelIterate(_temp, _coef);
} */

void Solver::writeVTK(const string &filename) const {

    filesystem::path filepath = filesystem::path("output") / filesystem::path(filename);
    filesystem::path parent = filepath.parent_path();

    if (!parent.empty() && !filesystem::exists(parent)) {
        filesystem::create_directories(parent);
    }

    ofstream file(filepath);

    if (!file.is_open()) {
        cerr << "无法打开文件: " << filepath << std::endl;
    }

    // write header
	file << "# vtk DataFile Version 3.0" << endl;
    file << "flash 3d grid and solution" << endl;
    file << "ASCII" << endl;
    file << "DATASET RECTILINEAR_GRID" << endl;

    // write mesh information
    file << "DIMENSIONS " << nx + 1 << " " << ny + 1 << " " << nz + 1 << endl;
    file << "X_COORDINATES " << nx + 1 << " float" << endl;
    for (int i = 0; i <= nx; ++i) {
        file << xmin + (scalar)i * dx << " ";
    }
    file << endl;
    file << "Y_COORDINATES " << ny + 1 << " float" << endl;
    for (int i = 0; i <= ny; ++i) {
        file << ymin + (scalar)i * dy << " ";
    }
    file << endl;
    file << "Z_COORDINATES " << nz + 1 << " float" << endl;
    for (int i = 0; i <= nz; ++i) {
        file << zmin + (scalar)i * dz << " ";
    }
    file << endl;

    // write cell data
    int ncell = nx * ny * nz;
    file << "CELL_DATA " << ncell << endl;

    file << "FIELD FieldData 1" << endl;

    // write temperature data
    file << "temperature 1 " << ncell << " float" << endl;
    for (const scalar& temp : _temp) {
        file << temp << " ";
    }
    file << endl;

    file.close();
}

/* void Solver::calcCoef() {

    scalar areaE = dy * dz;
    scalar areaW = dy * dz;
    scalar areaN = dz * dx;
    scalar areaS = dz * dx;
    scalar areaT = dx * dy;
    scalar areaB = dx * dy;

    scalar aE = -thermalConductivity * areaE / dx;
    scalar aW = -thermalConductivity * areaW / dx;
    scalar aN = -thermalConductivity * areaN / dy;
    scalar aS = -thermalConductivity * areaS / dy;
    scalar aT = -thermalConductivity * areaT / dz;
    scalar aB = -thermalConductivity * areaB / dz;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {

                int id;
                if (dim == 2) {
                    id = i * 6 + j * nx * 6;
                } else if (dim == 3) {
                    id = i * 8 + j * nx * 8 + k * nx * ny * 8;
                }

                _coef[id+id_aE] += aE;
                _coef[id+id_aW] += aW;
                _coef[id+id_aN] += aN;
                _coef[id+id_aS] += aS;
                _coef[id+id_aP] += -(aE + aW + aN + aS);
                if (dim == 3) {
                    _coef[id+id_aT] += aT;
                    _coef[id+id_aB] += aB;
                    _coef[id+id_aP] += -(aT + aB);
                }

                // set boundary conditios
                if (i == nx - 1) { // east
                    if (TempBCs::type[east] == 0) { // "Dirichlet" type
                        _coef[id+id_aP] += -aE;
                        _coef[id+id_b] += -2 * aE * TempBCs::val[east];
                        _coef[id+id_aE] = 0.0;
                    } else if (TempBCs::type[east] == 1) { // "Neumann" type
                        _coef[id+id_aP] += aE;
                        _coef[id+id_b] += -TempBCs::val[east] * areaE;
                        _coef[id+id_aE] = 0.0;
                    }
                }
                if (i == 0) { // west
                    if (TempBCs::type[west] == 0) { // "Dirichlet" type
                        _coef[id+id_aP] += -aW;
                        _coef[id+id_b] += -2 * aW * TempBCs::val[west];
                        _coef[id+id_aW] = 0.0;
                    } else if (TempBCs::type[west] == 1) { // "Neumann" type
                        _coef[id+id_aP] += aW;
                        _coef[id+id_b] += -TempBCs::val[west] * areaW;
                        _coef[id+id_aW] = 0.0;
                    }
                }
                if (j == ny - 1) { // north
                    if (TempBCs::type[north] == 0) { // "Dirichlet" type
                        _coef[id+id_aP] += -aN;
                        _coef[id+id_b] += -2 * aN * TempBCs::val[north];
                        _coef[id+id_aN] = 0.0;
                    } else if (TempBCs::type[north] == 1) { // "Neumann" type
                        _coef[id+id_aP] += aN;
                        _coef[id+id_b] += -TempBCs::val[north] * areaN;
                        _coef[id+id_aN] = 0.0;
                    }
                }
                if (j == 0) { // south
                    if (TempBCs::type[south] == 0) { // "Dirichlet" type
                        _coef[id+id_aP] += -aS;
                        _coef[id+id_b] += -2 * aS * TempBCs::val[south];
                        _coef[id+id_aS] = 0.0;
                    } else if (TempBCs::type[south] == 1) { // "Neumann" type
                        _coef[id+id_aP] += aS;
                        _coef[id+id_b] += -TempBCs::val[south] * areaS;
                        _coef[id+id_aS] = 0.0;
                    }
                }
                if (dim == 3 && k == nz - 1) { // top
                    if (TempBCs::type[top] == 0) { // "Dirichlet" type
                        _coef[id+id_aP] += -aT;
                        _coef[id+id_b] += -2 * aT * TempBCs::val[top];
                        _coef[id+id_aT] = 0.0;
                    } else if (TempBCs::type[top] == 1) { // "Neumann" type
                        _coef[id+id_aP] += aT;
                        _coef[id+id_b] += -TempBCs::val[top] * areaT;
                        _coef[id+id_aT] = 0.0;
                    }
                }
                if (dim == 3 && k == 0) { // bottom
                    if (TempBCs::type[bottom] == 0) { // "Dirichlet" type
                        _coef[id+id_aP] += -aB;
                        _coef[id+id_b] += -2 * aB * TempBCs::val[bottom];
                        _coef[id+id_aB] = 0.0;
                    } else if (TempBCs::type[bottom] == 1) { // "Neumann" type
                        _coef[id+id_aP] += aB;
                        _coef[id+id_b] += -TempBCs::val[bottom] * areaB;
                        _coef[id+id_aB] = 0.0;
                    }
                }
            }
        }
    }
} */