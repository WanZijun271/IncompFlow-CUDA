#include "Solver.h"
#include "config.h"
#include "constants.h"
#include "kernels.cuh"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <filesystem>

using namespace std;

Solver::Solver(scalar u, scalar v, scalar w, scalar p, scalar temp) {

    _u.assign(nx * ny * nz, u);
    _v.assign(nx * ny * nz, v);
    _w.assign(nx * ny * nz, w);

    _p.assign(nx * ny * nz, p);

    _temp.assign(nx * ny * nz, temp);
}

void Solver::solve() {

    size_t fieldSize = nx * ny * nz * sizeof(scalar);

    scalar *u_dev, *v_dev, *w_dev, *p_dev, *temp_dev, *pCorr_dev;
    cudaMalloc(&u_dev, fieldSize);
    cudaMalloc(&v_dev, fieldSize);
    cudaMalloc(&w_dev, fieldSize);
    cudaMalloc(&p_dev, fieldSize);
    cudaMalloc(&temp_dev, fieldSize);
    cudaMalloc(&pCorr_dev, fieldSize);

    cudaMemcpy(u_dev, _u.data(), fieldSize, cudaMemcpyHostToDevice);
    cudaMemcpy(v_dev, _v.data(), fieldSize, cudaMemcpyHostToDevice);
    cudaMemcpy(w_dev, _w.data(), fieldSize, cudaMemcpyHostToDevice);
    cudaMemcpy(p_dev, _p.data(), fieldSize, cudaMemcpyHostToDevice);
    cudaMemcpy(temp_dev, _temp.data(), fieldSize, cudaMemcpyHostToDevice);
    cudaMemset(pCorr_dev, 0, fieldSize);

    size_t ufSize = (nx+1) * ny * nz * sizeof(scalar);
    size_t vfSize = nx * (ny+1) * nz * sizeof(scalar);
    size_t wfSize = nx * ny * (nz+1) * sizeof(scalar);

    scalar *uf_dev, *vf_dev, *wf_dev;
    cudaMalloc(&uf_dev, ufSize);
    cudaMalloc(&vf_dev, vfSize);
    cudaMalloc(&wf_dev, wfSize);

    cudaMemset(uf_dev, 0, ufSize);
    cudaMemset(vf_dev, 0, vfSize);
    cudaMemset(wf_dev, 0, wfSize);

    size_t coefSize = nx * ny * nz * (1+2*dim) * sizeof(scalar);

    scalar *uCoef_dev, *vCoef_dev, *wCoef_dev, *pCorrCoef_dev;
    cudaMalloc(&uCoef_dev, coefSize);
    cudaMalloc(&vCoef_dev, coefSize);
    cudaMalloc(&wCoef_dev, coefSize);
    cudaMalloc(&pCorrCoef_dev, coefSize);

    cudaMemset(uCoef_dev, 0, coefSize);
    cudaMemset(vCoef_dev, 0, coefSize);
    cudaMemset(wCoef_dev, 0, coefSize);
    cudaMemset(pCorrCoef_dev, 0, coefSize);

    size_t srcTermSize = nx * ny * nz * sizeof(scalar);

    scalar *uSrcTerm_dev, *vSrcTerm_dev, *wSrcTerm_dev, *pCorrSrcTerm_dev;
    cudaMalloc(&uSrcTerm_dev, srcTermSize);
    cudaMalloc(&vSrcTerm_dev, srcTermSize);
    cudaMalloc(&wSrcTerm_dev, srcTermSize);
    cudaMalloc(&pCorrSrcTerm_dev, srcTermSize);

    cudaMemset(uSrcTerm_dev, 0, srcTermSize);
    cudaMemset(vSrcTerm_dev, 0, srcTermSize);
    cudaMemset(wSrcTerm_dev, 0, srcTermSize);
    cudaMemset(pCorrSrcTerm_dev, 0, srcTermSize);

    scalar *uNorm_dev, *vNorm_dev, *wNorm_dev, *ufNorm_dev, *vfNorm_dev, *wfNorm_dev, *pNorm_dev;
    cudaMalloc(&uNorm_dev, sizeof(scalar));
    cudaMalloc(&vNorm_dev, sizeof(scalar));
    cudaMalloc(&wNorm_dev, sizeof(scalar));
    cudaMalloc(&ufNorm_dev, sizeof(scalar));
    cudaMalloc(&vfNorm_dev, sizeof(scalar));
    cudaMalloc(&wfNorm_dev, sizeof(scalar));
    cudaMalloc(&pNorm_dev, sizeof(scalar));

    scalar maxPNorm = -1e20, maxUNorm = -1e20, maxVNorm = -1e20, maxWNorm = -1e20, maxUfNorm = -1e20, maxVfNorm = -1e20, maxWfNorm = -1e20;

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

        pointJacobiIterate(u_dev, fieldSize, uCoef_dev, uSrcTerm_dev, nIter_u, relax_u, tol_u);
        pointJacobiIterate(v_dev, fieldSize, vCoef_dev, vSrcTerm_dev, nIter_v, relax_v, tol_v);
        if (dim == 3) {
            pointJacobiIterate(w_dev, fieldSize, wCoef_dev, wSrcTerm_dev, nIter_w, relax_w, tol_w);
        }

        RhieChowInterpolate(uf_dev, vf_dev, wf_dev, u_dev, v_dev, w_dev, uCoef_dev, vCoef_dev, wCoef_dev, p_dev);
        applyBCsToFaceVel(uf_dev, vf_dev, wf_dev, u_dev, v_dev, w_dev);

        calcPresCorrLinkCoef(pCorrCoef_dev, uCoef_dev, vCoef_dev, wCoef_dev);

        calcPresCorrSrcTerm(pCorrSrcTerm_dev, uf_dev, vf_dev, wf_dev);

        pointJacobiIterate(pCorr_dev, fieldSize, pCorrCoef_dev, pCorrSrcTerm_dev, nIter_p, relax_p, tol_p);

        updateField(u_dev, v_dev, w_dev, uNorm_dev, vNorm_dev, wNorm_dev, uf_dev, vf_dev, wf_dev, ufNorm_dev, vfNorm_dev, wfNorm_dev
            , p_dev, pNorm_dev, uCoef_dev, vCoef_dev, wCoef_dev, pCorr_dev);
        
        scalar pNorm = 0, uNorm = 0, vNorm = 0, wNorm = 0, ufNorm = 0, vfNorm = 0, wfNorm = 0;

        cudaStream_t stream[2*dim+1];
        for (int i = 0; i < 2*dim+1; ++i) {
            cudaStreamCreate(&stream[i]);
        }

        cudaMemcpyAsync(&pNorm, pNorm_dev, sizeof(scalar), cudaMemcpyDeviceToHost, stream[0]);
        cudaMemcpyAsync(&uNorm, uNorm_dev, sizeof(scalar), cudaMemcpyDeviceToHost, stream[1]);
        cudaMemcpyAsync(&vNorm, vNorm_dev, sizeof(scalar), cudaMemcpyDeviceToHost, stream[2]);
        cudaMemcpyAsync(&ufNorm, ufNorm_dev, sizeof(scalar), cudaMemcpyDeviceToHost, stream[3]);
        cudaMemcpyAsync(&vfNorm, vfNorm_dev, sizeof(scalar), cudaMemcpyDeviceToHost, stream[4]);
        if (dim == 3) {
            cudaMemcpyAsync(&wNorm, wNorm_dev, sizeof(scalar), cudaMemcpyDeviceToHost, stream[5]);
            cudaMemcpyAsync(&wfNorm, wfNorm_dev, sizeof(scalar), cudaMemcpyDeviceToHost, stream[6]);
        }

        for (int i = 0; i < 2*dim+1; ++i) {
            cudaStreamSynchronize(stream[i]);
        }

        for (int i = 0; i < 2*dim+1; ++i) {
            cudaStreamDestroy(stream[i]);
        }

        pNorm = sqrt(pNorm / (nx * ny * nz));
        maxPNorm = max(pNorm, maxPNorm);
        uNorm = sqrt(uNorm / (nx * ny * nz));
        maxUNorm = max(uNorm, maxUNorm);
        vNorm = sqrt(vNorm / (nx * ny * nz));
        maxVNorm = max(vNorm, maxVNorm);
        ufNorm = sqrt(ufNorm / ((nx+1) * ny * nz));
        maxUfNorm = max(ufNorm, maxUfNorm);
        vfNorm = sqrt(vfNorm / (nx * (ny+1) * nz));
        maxVfNorm = max(vfNorm, maxVfNorm);
        if (dim == 3) {
            wNorm = sqrt(wNorm / (nx * ny * nz));
            maxWNorm = max(wNorm, maxWNorm);
            wfNorm = sqrt(wfNorm / (nx * ny * (nz+1)));
            maxWfNorm = max(wfNorm, maxWfNorm);
        }

        scalar relPNorm = pNorm / (maxPNorm + 1e-20);
        scalar relUNorm = uNorm / (maxUNorm + 1e-20);
        scalar relVNorm = vNorm / (maxVNorm + 1e-20);
        scalar relUfNorm = ufNorm / (maxUfNorm + 1e-20);
        scalar relVfNorm = vfNorm / (maxVfNorm + 1e-20);
        scalar relWNorm = 0, relWfNorm = 0;
        if (dim == 3) {
            relWNorm = wNorm / (maxWNorm + 1e-20);
            relWfNorm = wfNorm / (maxWfNorm + 1e-20);
        }

        cout << pNorm << " , " << maxPNorm << " , " << relPNorm << endl;

        if (relPNorm < outerTol && relUNorm < outerTol && relVNorm < outerTol && relWNorm < outerTol && relUfNorm < outerTol
            && relVfNorm < outerTol && relWfNorm < outerTol) {
            break;
        }
    }

    cudaMemcpy(_u.data(), u_dev, fieldSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(_v.data(), v_dev, fieldSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(_w.data(), w_dev, fieldSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(_p.data(), p_dev, fieldSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(_temp.data(), temp_dev, fieldSize, cudaMemcpyDeviceToHost);

    vector<scalar> uf((nx+1)*ny*nz, 0);

    cudaMemcpy(uf.data(), uf_dev, ufSize, cudaMemcpyDeviceToHost);

    for (int j = ny - 1; j >= 0; --j) {
        for (int i = 0; i < nx+1; ++i) {
            cout << uf[i+(nx+1)*j] << ' ';
        }
        cout << endl;
    }

    /* cout << "----------------------------------" << endl;

    vector<scalar> uCoef(nx*ny*nz*(1+2*dim), 0);

    cudaMemcpy(uCoef.data(), uCoef_dev, coefSize, cudaMemcpyDeviceToHost);

    for (int j = ny - 1; j >= 0; --j) {
        for (int i = 0; i < nx; ++i) {
            cout << uCoef[i+nx*(j+ny*aW)] << ' ';
        }
        cout << endl;
    }

    cout << "----------------------------------" << endl;

    vector<scalar> uSrcTerm(nx*ny*nz, 0);

    cudaMemcpy(uSrcTerm.data(), uSrcTerm_dev, srcTermSize, cudaMemcpyDeviceToHost);

    for (int j = ny - 1; j >= 0; --j) {
        for (int i = 0; i < nx; ++i) {
            cout << uSrcTerm[i+nx*j] << ' ';
        }
        cout << endl;
    }

    cout << "----------------------------------" << endl;

    for (int j = ny - 1; j >= 0; --j) {
        for (int i = 0; i < nx; ++i) {
            cout << _u[i+nx*j] << ' ';
        }
        cout << endl;
    } */

    cudaFree(u_dev);
    cudaFree(v_dev);
    cudaFree(w_dev);
    cudaFree(p_dev);
    cudaFree(temp_dev);
    cudaFree(pCorr_dev);

    cudaFree(uf_dev);
    cudaFree(vf_dev);
    cudaFree(wf_dev);

    cudaFree(uCoef_dev);
    cudaFree(vCoef_dev);
    cudaFree(wCoef_dev);
    cudaFree(pCorrCoef_dev);

    cudaFree(uSrcTerm_dev);
    cudaFree(vSrcTerm_dev);
    cudaFree(wSrcTerm_dev);
    cudaFree(pCorrSrcTerm_dev);

    cudaFree(uNorm_dev);
    cudaFree(vNorm_dev);
    cudaFree(wNorm_dev);
    cudaFree(ufNorm_dev);
    cudaFree(vfNorm_dev);
    cudaFree(wfNorm_dev);
    cudaFree(pNorm_dev);
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