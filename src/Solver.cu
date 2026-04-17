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

    size_t coefSize = nx * ny * nz * (1+2*dim) * sizeof(scalar);

    scalar *uCoef_dev, *vCoef_dev, *wCoef_dev, *pCoef_dev;
    cudaMalloc(&uCoef_dev, coefSize);
    cudaMalloc(&vCoef_dev, coefSize);
    cudaMalloc(&wCoef_dev, coefSize);
    cudaMalloc(&pCoef_dev, coefSize);

    cudaMemset(uCoef_dev, 0.0, coefSize);
    cudaMemset(vCoef_dev, 0.0, coefSize);
    cudaMemset(wCoef_dev, 0.0, coefSize);
    cudaMemset(pCoef_dev, 0.0, coefSize);

    size_t srcTermSize = nx * ny * nz * sizeof(scalar);

    scalar *uSrcTerm_dev, *vSrcTerm_dev, *wSrcTerm_dev, *pSrcTerm_dev;
    cudaMalloc(&uSrcTerm_dev, srcTermSize);
    cudaMalloc(&vSrcTerm_dev, srcTermSize);
    cudaMalloc(&wSrcTerm_dev, srcTermSize);
    cudaMalloc(&pSrcTerm_dev, srcTermSize);

    cudaMemset(uSrcTerm_dev, 0.0, srcTermSize);
    cudaMemset(vSrcTerm_dev, 0.0, srcTermSize);
    cudaMemset(wSrcTerm_dev, 0.0, srcTermSize);
    cudaMemset(pSrcTerm_dev, 0.0, srcTermSize);

    for (int it = 0; it < numOuterIter; ++it) {

        if (it == 0) {
            initFaceVel(u_dev, v_dev, w_dev, uf_dev, vf_dev, wf_dev); // initalize the velocity on the faces
        }
        applyBCsToFaceVel(uf_dev, vf_dev, wf_dev, u_dev, v_dev, w_dev);

        calcMomLinkCoef(uCoef_dev, uf_dev, vf_dev, wf_dev);
        cudaMemcpy(vCoef_dev, uCoef_dev, coefSize, cudaMemcpyDeviceToDevice);
        if (dim == 3) {
            cudaMemcpy(wCoef_dev, uCoef_dev, coefSize, cudaMemcpyDeviceToDevice);
        }

        calcMomSrcTerm(uSrcTerm_dev, vSrcTerm_dev, wSrcTerm_dev, p_dev);

        applyBCsToMomEq(uCoef_dev, uSrcTerm_dev, vCoef_dev, vSrcTerm_dev, wCoef_dev, wSrcTerm_dev);

        pointJacobiIterate(u_dev, fieldSize, uCoef_dev, uSrcTerm_dev);
        pointJacobiIterate(v_dev, fieldSize, vCoef_dev, vSrcTerm_dev);
        if (dim == 3) {
            pointJacobiIterate(w_dev, fieldSize, wCoef_dev, wSrcTerm_dev);
        }

        RhieChowInterpolate(uf_dev, vf_dev, wf_dev, u_dev, v_dev, w_dev, uCoef_dev, vCoef_dev, wCoef_dev, p_dev);
        applyBCsToFaceVel(uf_dev, vf_dev, wf_dev, u_dev, v_dev, w_dev);
    }

    cudaMemcpy(_u.data(), u_dev, fieldSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(_v.data(), v_dev, fieldSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(_w.data(), w_dev, fieldSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(_p.data(), p_dev, fieldSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(_temp.data(), temp_dev, fieldSize, cudaMemcpyDeviceToHost);

    cudaFree(u_dev);
    cudaFree(v_dev);
    cudaFree(w_dev);
    cudaFree(p_dev);
    cudaFree(temp_dev);

    cudaFree(uf_dev);
    cudaFree(vf_dev);
    cudaFree(wf_dev);

    cudaFree(uCoef_dev);
    cudaFree(vCoef_dev);
    cudaFree(wCoef_dev);
    cudaFree(pCoef_dev);

    cudaFree(uSrcTerm_dev);
    cudaFree(vSrcTerm_dev);
    cudaFree(wSrcTerm_dev);
    cudaFree(pSrcTerm_dev);
}

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