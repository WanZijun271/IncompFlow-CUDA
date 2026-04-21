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

    size_t fieldSize = nx * ny * nz * sizeof(scalar);

    size_t ufSize = (nx+1) * ny * nz * sizeof(scalar);
    size_t vfSize = nx * (ny+1) * nz * sizeof(scalar);
    size_t wfSize = nx * ny * (nz+1) * sizeof(scalar);

    size_t coefSize = nx * ny * nz * (1+2*dim) * sizeof(scalar);

    cudaStream_t stream[18];

    for (int i = 0; i < 18; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    cudaMallocAsync(&_u_dev, fieldSize, stream[0]);
    cudaMallocAsync(&_v_dev, fieldSize, stream[1]);
    cudaMallocAsync(&_w_dev, fieldSize, stream[2]);
    cudaMallocAsync(&_p_dev, fieldSize, stream[3]);
    cudaMallocAsync(&_temp_dev, fieldSize, stream[4]);
    cudaMallocAsync(&_pCorr_dev, fieldSize, stream[5]);

    cudaMallocAsync(&_uf_dev, ufSize, stream[6]);
    cudaMallocAsync(&_vf_dev, vfSize, stream[7]);
    cudaMallocAsync(&_wf_dev, wfSize, stream[8]);

    cudaMallocAsync(&_uCoef_dev, coefSize, stream[9]);
    cudaMallocAsync(&_vCoef_dev, coefSize, stream[10]);
    cudaMallocAsync(&_wCoef_dev, coefSize, stream[11]);
    cudaMallocAsync(&_pCorrCoef_dev, coefSize, stream[12]);

    cudaMallocAsync(&_uSrcTerm_dev, fieldSize, stream[13]);
    cudaMallocAsync(&_vSrcTerm_dev, fieldSize, stream[14]);
    cudaMallocAsync(&_wSrcTerm_dev, fieldSize, stream[15]);
    cudaMallocAsync(&_pCorrSrcTerm_dev, fieldSize, stream[16]);

    cudaMallocAsync(&_uNorm_dev, sizeof(scalar), stream[17]);
    cudaMallocAsync(&_vNorm_dev, sizeof(scalar), stream[17]);
    cudaMallocAsync(&_wNorm_dev, sizeof(scalar), stream[17]);
    cudaMallocAsync(&_ufNorm_dev, sizeof(scalar), stream[17]);
    cudaMallocAsync(&_vfNorm_dev, sizeof(scalar), stream[17]);
    cudaMallocAsync(&_wfNorm_dev, sizeof(scalar), stream[17]);
    cudaMallocAsync(&_pNorm_dev, sizeof(scalar), stream[17]);
    cudaMallocAsync(&_massNorm_dev, sizeof(scalar), stream[17]);
    
    cudaMemcpyAsync(_u_dev, _u.data(), fieldSize, cudaMemcpyHostToDevice, stream[0]);
    cudaMemcpyAsync(_v_dev, _v.data(), fieldSize, cudaMemcpyHostToDevice, stream[1]);
    cudaMemcpyAsync(_w_dev, _w.data(), fieldSize, cudaMemcpyHostToDevice, stream[2]);
    cudaMemcpyAsync(_p_dev, _p.data(), fieldSize, cudaMemcpyHostToDevice, stream[3]);
    cudaMemcpyAsync(_temp_dev, _temp.data(), fieldSize, cudaMemcpyHostToDevice, stream[4]);
    cudaMemsetAsync(_pCorr_dev, 0, fieldSize, stream[5]);

    cudaMemsetAsync(_uf_dev, 0, ufSize, stream[6]);
    cudaMemsetAsync(_vf_dev, 0, vfSize, stream[7]);
    cudaMemsetAsync(_wf_dev, 0, wfSize, stream[8]);

    cudaMemsetAsync(_uCoef_dev, 0, coefSize, stream[9]);
    cudaMemsetAsync(_vCoef_dev, 0, coefSize, stream[10]);
    cudaMemsetAsync(_wCoef_dev, 0, coefSize, stream[11]);
    cudaMemsetAsync(_pCorrCoef_dev, 0, coefSize, stream[12]);

    cudaMemsetAsync(_uSrcTerm_dev, 0, fieldSize, stream[13]);
    cudaMemsetAsync(_vSrcTerm_dev, 0, fieldSize, stream[14]);
    cudaMemsetAsync(_wSrcTerm_dev, 0, fieldSize, stream[15]);
    cudaMemsetAsync(_pCorrSrcTerm_dev, 0, fieldSize, stream[16]);

    cudaMemsetAsync(_uNorm_dev, 0, sizeof(scalar), stream[17]);
    cudaMemsetAsync(_vNorm_dev, 0, sizeof(scalar), stream[17]);
    cudaMemsetAsync(_wNorm_dev, 0, sizeof(scalar), stream[17]);
    cudaMemsetAsync(_ufNorm_dev, 0, sizeof(scalar), stream[17]);
    cudaMemsetAsync(_vfNorm_dev, 0, sizeof(scalar), stream[17]);
    cudaMemsetAsync(_wfNorm_dev, 0, sizeof(scalar), stream[17]);
    cudaMemsetAsync(_pNorm_dev, 0, sizeof(scalar), stream[17]);
    cudaMemsetAsync(_massNorm_dev, 0, sizeof(scalar), stream[17]);

    for (int i = 0; i < 18; ++i) {
        cudaStreamSynchronize(stream[i]);
    }

    for (int i = 0; i < 18; ++i) {
        cudaStreamDestroy(stream[i]);
    }
}

Solver::~Solver() {

    cudaStream_t stream[18];

    for (int i = 0; i < 18; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    cudaFreeAsync(_u_dev, stream[0]);
    cudaFreeAsync(_v_dev, stream[1]);
    cudaFreeAsync(_w_dev, stream[2]);
    cudaFreeAsync(_p_dev, stream[3]);
    cudaFreeAsync(_temp_dev, stream[4]);
    cudaFreeAsync(_pCorr_dev, stream[5]);

    cudaFreeAsync(_uf_dev, stream[6]);
    cudaFreeAsync(_vf_dev, stream[7]);
    cudaFreeAsync(_wf_dev, stream[8]);

    cudaFreeAsync(_uCoef_dev, stream[9]);
    cudaFreeAsync(_vCoef_dev, stream[10]);
    cudaFreeAsync(_wCoef_dev, stream[11]);
    cudaFreeAsync(_pCorrCoef_dev, stream[12]);

    cudaFreeAsync(_uSrcTerm_dev, stream[13]);
    cudaFreeAsync(_vSrcTerm_dev, stream[14]);
    cudaFreeAsync(_wSrcTerm_dev, stream[15]);
    cudaFreeAsync(_pCorrSrcTerm_dev, stream[16]);

    cudaFreeAsync(_uNorm_dev, stream[17]);
    cudaFreeAsync(_vNorm_dev, stream[17]);
    cudaFreeAsync(_wNorm_dev, stream[17]);
    cudaFreeAsync(_ufNorm_dev, stream[17]);
    cudaFreeAsync(_vfNorm_dev, stream[17]);
    cudaFreeAsync(_wfNorm_dev, stream[17]);
    cudaFreeAsync(_pNorm_dev, stream[17]);
    cudaFreeAsync(_massNorm_dev, stream[17]);

    for (int i = 0; i < 18; ++i) {
        cudaStreamSynchronize(stream[i]);
    }

    for (int i = 0; i < 18; ++i) {
        cudaStreamDestroy(stream[i]);
    }
}

void Solver::solve() {

    size_t fieldSize = nx * ny * nz * sizeof(scalar);
    size_t coefSize = nx * ny * nz * (1+2*dim) * sizeof(scalar);

    scalar pNormMax = 0, uNormMax = 0, vNormMax = 0, wNormMax = 0, ufNormMax = 0, vfNormMax = 0, wfNormMax = 0;

    for (int it = 0; it < numOuterIter; ++it) {

        if (it == 0) {
            initFaceVel();
        }
        applyBCsToFaceVel();

        calcMomLinkCoef(_uCoef_dev);
        cudaMemcpy(_vCoef_dev, _uCoef_dev, coefSize, cudaMemcpyDeviceToDevice);
        if constexpr (dim == 3) {
            cudaMemcpy(_wCoef_dev, _uCoef_dev, coefSize, cudaMemcpyDeviceToDevice);
        }

        calcMomSrcTerm();

        applyBCsToMomEq();

        // pointJacobiIterate(_u_dev, fieldSize, _uCoef_dev, _uSrcTerm_dev, nIter_u, relax_u, tol_u);
        GaussSeidelIterate(_u_dev, _uCoef_dev, _uSrcTerm_dev, nIter_u, relax_u, tol_u);
        // pointJacobiIterate(_v_dev, fieldSize, _vCoef_dev, _vSrcTerm_dev, nIter_v, relax_v, tol_v);
        GaussSeidelIterate(_v_dev, _vCoef_dev, _vSrcTerm_dev, nIter_v, relax_v, tol_v);
        if constexpr (dim == 3) {
            // pointJacobiIterate(_w_dev, fieldSize, _wCoef_dev, _wSrcTerm_dev, nIter_w, relax_w, tol_w);
            GaussSeidelIterate(_w_dev, _wCoef_dev, _wSrcTerm_dev, nIter_w, relax_w, tol_w);
        }

        RhieChowInterpolate();
        applyBCsToFaceVel();

        calcPresCorrLinkCoef();

        calcPresCorrSrcTerm();

        cudaMemset(_pCorr_dev, 0, fieldSize);

        // pointJacobiIterate(_pCorr_dev, fieldSize, _pCorrCoef_dev, _pCorrSrcTerm_dev, nIter_p, relax_p, tol_p);
        GaussSeidelIterate(_pCorr_dev, _pCorrCoef_dev, _pCorrSrcTerm_dev, nIter_p, relax_p, tol_p);

        updateField();
        
        scalar pNorm = 0, uNorm = 0, vNorm = 0, wNorm = 0, ufNorm = 0, vfNorm = 0, wfNorm = 0, massNorm = 0;

        cudaStream_t stream[2*dim+2];
        for (int i = 0; i < 2*dim+2; ++i) {
            cudaStreamCreate(&stream[i]);
        }

        cudaMemcpyAsync(&pNorm, _pNorm_dev, sizeof(scalar), cudaMemcpyDeviceToHost, stream[0]);
        cudaMemcpyAsync(&uNorm, _uNorm_dev, sizeof(scalar), cudaMemcpyDeviceToHost, stream[1]);
        cudaMemcpyAsync(&vNorm, _vNorm_dev, sizeof(scalar), cudaMemcpyDeviceToHost, stream[2]);
        cudaMemcpyAsync(&ufNorm, _ufNorm_dev, sizeof(scalar), cudaMemcpyDeviceToHost, stream[3]);
        cudaMemcpyAsync(&vfNorm, _vfNorm_dev, sizeof(scalar), cudaMemcpyDeviceToHost, stream[4]);
        cudaMemcpyAsync(&massNorm, _massNorm_dev, sizeof(scalar), cudaMemcpyDeviceToHost, stream[5]);
        if constexpr (dim == 3) {
            cudaMemcpyAsync(&wNorm, _wNorm_dev, sizeof(scalar), cudaMemcpyDeviceToHost, stream[6]);
            cudaMemcpyAsync(&wfNorm, _wfNorm_dev, sizeof(scalar), cudaMemcpyDeviceToHost, stream[7]);
        }

        for (int i = 0; i < 2*dim+2; ++i) {
            cudaStreamSynchronize(stream[i]);
        }

        for (int i = 0; i < 2*dim+2; ++i) {
            cudaStreamDestroy(stream[i]);
        }

        pNorm = sqrt(pNorm / (nx * ny * nz));
        uNorm = sqrt(uNorm / (nx * ny * nz));
        vNorm = sqrt(vNorm / (nx * ny * nz));
        ufNorm = sqrt(ufNorm / ((nx-1) * ny * nz));
        vfNorm = sqrt(vfNorm / (nx * (ny-1) * nz));
        massNorm = sqrt(massNorm / (nx * ny * nz));
        if constexpr (dim == 3) {
            wNorm = sqrt(wNorm / (nx * ny * nz));
            wfNorm = sqrt(wfNorm / (nx * ny * (nz-1)));
        }

        pNormMax = max(pNormMax, pNorm);
        uNormMax = max(pNormMax, pNorm);
        vNormMax = max(pNormMax, pNorm);
        ufNormMax = max(pNormMax, pNorm);
        vfNormMax = max(pNormMax, pNorm);
        if constexpr (dim == 3) {
            wNormMax = max(pNormMax, pNorm);
            wfNormMax = max(pNormMax, pNorm);
        }

        scalar pRelNorm = pNorm / pNormMax;
        scalar uRelNorm = uNorm / uNormMax;
        scalar vRelNorm = vNorm / vNormMax;
        scalar ufRelNorm = ufNorm / ufNormMax;
        scalar vfRelNorm = vfNorm / vfNormMax;
        scalar wRelNorm, wfRelNorm;
        if constexpr (dim == 3) {
            wRelNorm = wNorm / wNormMax;
            wfRelNorm = wfNorm / wfNormMax;
        }

        bool isConverged = true;
        isConverged = isConverged && (pRelNorm < tol_p);
        isConverged = isConverged && (uRelNorm < tol_u);
        isConverged = isConverged && (vRelNorm < tol_v);
        isConverged = isConverged && (ufRelNorm < tol_u);
        isConverged = isConverged && (vfRelNorm < tol_v);
        if constexpr (dim == 3) {
            isConverged = isConverged && (wRelNorm < tol_w);
            isConverged = isConverged && (wfRelNorm < tol_w);
        }
        isConverged = isConverged || (massNorm < tol_mass);

        if (it == 0 || (it+1) % 1000 == 0 || isConverged || it == numOuterIter - 1) {
            printf("iter: %d\n", it+1);
            printf("pRelNorm : %.3E\n", pRelNorm);
            printf("uRelNorm : %.3E\n", uRelNorm);
            printf("vRelNorm : %.3E\n", vRelNorm);
            if constexpr (dim == 3) {
                printf("wRelNorm : %.3E\n", wRelNorm);
            }
            printf("ufRelNorm: %.3E\n", ufRelNorm);
            printf("vfRelNorm: %.3E\n", vfRelNorm);
            if constexpr (dim == 3) {
                printf("wfRelNorm: %.3E\n", wfRelNorm);
            }
            printf("massNorm : %.3E\n", massNorm);
            printf("------------------------------------------------------------\n");
        }

        if (isConverged) {
            break;
        }
    }

    cudaMemcpy(_u.data(), _u_dev, fieldSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(_v.data(), _v_dev, fieldSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(_w.data(), _w_dev, fieldSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(_p.data(), _p_dev, fieldSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(_temp.data(), _temp_dev, fieldSize, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
}

void Solver::writeVTK(const string &filename) const {

    filesystem::path filepath = filesystem::path("output") / filesystem::path(filename);
    filesystem::path parent = filepath.parent_path();

    if (!parent.empty() && !filesystem::exists(parent)) {
        filesystem::create_directories(parent);
    }

    ofstream file(filepath);

    if (!file.is_open()) {
        cerr << "Failed to open the file: " << filepath << std::endl;
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

    file << "FIELD FieldData 3" << endl;

    // write temperature data
    file << "temperature 1 " << ncell << " float" << endl;
    for (const scalar& temp : _temp) {
        file << temp << " ";
    }
    file << endl;

    // write velocity data
    file << "velocity 3 " << ncell << " float" << endl;
    for (int idx = 0; idx < ncell; ++idx) {
        file << _u[idx] << " " << _v[idx] << " " << _w[idx] << " ";
    }
    file << endl;

    // write pressure data
    file << "pressure 1 " << ncell << " float" << endl;
    for (const scalar& p : _p) {
        file << p << " ";
    }

    file.close();
}

void Solver::initFaceVel() {

    cudaStream_t stream[dim];
    for (int i = 0; i < dim; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    dim3 threadsPerBlock;
    dim3 numBlocks;
    if constexpr (dim == 2) {
        threadsPerBlock = dim3(32, 32, 1);
    } else if constexpr (dim == 3) {
        threadsPerBlock = dim3(16, 16, 8);
    }

    numBlocks.x = (nx - 1 + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;
    initFaceVelKernel<xDir><<<numBlocks, threadsPerBlock, 0, stream[0]>>>(_u_dev, _uf_dev);

    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny - 1 + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;
    initFaceVelKernel<yDir><<<numBlocks, threadsPerBlock, 0, stream[1]>>>(_v_dev, _vf_dev);

    if constexpr (dim == 3) {
        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = (nz - 1 + threadsPerBlock.z - 1) / threadsPerBlock.z;
        initFaceVelKernel<zDir><<<numBlocks, threadsPerBlock, 0, stream[2]>>>(_w_dev, _wf_dev);
    }

    for (int i = 0; i < dim; ++i) {
        cudaStreamSynchronize(stream[i]);
    }

    for (int i = 0; i < dim; ++i) {
        cudaStreamDestroy(stream[i]);
    }
}

void Solver::applyBCsToFaceVel() {

    cudaStream_t stream[2*dim];
    for (int i = 0; i < 2*dim; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    dim3 threadsPerBlock;
    dim3 numBlocks;

    if constexpr (dim == 2) {
        threadsPerBlock = dim3(1024, 1, 1);
    } else if constexpr (dim == 3) {
        threadsPerBlock = dim3(32, 32, 1);
    }

    numBlocks.x = (ny + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (nz + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;

    applyBCsToFaceVelKernel<west, velBCs::type[west]><<<numBlocks, threadsPerBlock, 0, stream[0]>>>(_uf_dev, _u_dev
        , velBCs::val[west][0]);

    applyBCsToFaceVelKernel<east, velBCs::type[east]><<<numBlocks, threadsPerBlock, 0, stream[1]>>>(_uf_dev, _u_dev
        , velBCs::val[east][0]);

    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (nz + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;

    applyBCsToFaceVelKernel<south, velBCs::type[south]><<<numBlocks, threadsPerBlock, 0, stream[2]>>>(_vf_dev, _v_dev
        , velBCs::val[south][1]);

    applyBCsToFaceVelKernel<north, velBCs::type[north]><<<numBlocks, threadsPerBlock, 0, stream[3]>>>(_vf_dev, _v_dev
        , velBCs::val[north][1]);

    if constexpr (dim == 3) {
        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = 1;

        applyBCsToFaceVelKernel<bottom, velBCs::type[bottom]><<<numBlocks, threadsPerBlock, 0, stream[4]>>>(_wf_dev, _w_dev
            , velBCs::val[bottom][2]);

        applyBCsToFaceVelKernel<top, velBCs::type[top]><<<numBlocks, threadsPerBlock, 0, stream[5]>>>(_wf_dev, _w_dev
            , velBCs::val[top][2]);
    }

    for (int i = 0; i < 2*dim; ++i) {
        cudaStreamSynchronize(stream[i]);
    }

    for (int i = 0; i < 2*dim; ++i) {
        cudaStreamDestroy(stream[i]);
    }
}

void Solver::calcMomLinkCoef(scalar *coef_dev) {

    dim3 threadsPerBlock;
    dim3 numBlocks; 

    if constexpr (dim == 2) {
        threadsPerBlock = dim3(32, 32, 1);
    } else if constexpr (dim == 3) {
        threadsPerBlock = dim3(16, 8, 8);
    }
    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;

    calcMomLinkCoefKernel<<<numBlocks, threadsPerBlock>>>(coef_dev, _uf_dev, _vf_dev, _wf_dev);

    cudaDeviceSynchronize();
}

void Solver::calcMomSrcTerm() {

    cudaStream_t stream[3*dim];
    for (int i = 0; i < 3*dim; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    dim3 threadsPerBlock;
    dim3 numBlocks; 

    if constexpr (dim == 2) {
        threadsPerBlock = dim3(32, 32, 1);
    } else if constexpr (dim == 3) {
        threadsPerBlock = dim3(16, 8, 8);
    }

    numBlocks.x = (nx-2 + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny   + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz   + threadsPerBlock.z - 1) / threadsPerBlock.z;

    calcInteriorMomSrcTermKernel<xDir><<<numBlocks, threadsPerBlock, 0, stream[0]>>>(_uSrcTerm_dev, _p_dev);

    numBlocks.x = (nx   + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny-2 + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz   + threadsPerBlock.z - 1) / threadsPerBlock.z;

    calcInteriorMomSrcTermKernel<yDir><<<numBlocks, threadsPerBlock, 0, stream[1]>>>(_vSrcTerm_dev, _p_dev);

    if constexpr (dim == 3) {
        numBlocks.x = (nx   + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny   + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = (nz-2 + threadsPerBlock.z - 1) / threadsPerBlock.z;

        calcInteriorMomSrcTermKernel<zDir><<<numBlocks, threadsPerBlock, 0, stream[6]>>>(_wSrcTerm_dev, _p_dev);
    }

    if constexpr (dim == 2) {
        threadsPerBlock = dim3(1024, 1, 1);
    } else if constexpr (dim == 3) {
        threadsPerBlock = dim3(32, 32, 1);
    }

    numBlocks.x = (ny + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (nz + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;

    calcBCMomSrcTermKernel<west, velBCs::type[west]><<<numBlocks, threadsPerBlock, 0, stream[2]>>>(_uSrcTerm_dev, _p_dev);

    calcBCMomSrcTermKernel<east, velBCs::type[east]><<<numBlocks, threadsPerBlock, 0, stream[3]>>>(_uSrcTerm_dev, _p_dev);

    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (nz + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;

    calcBCMomSrcTermKernel<south, velBCs::type[south]><<<numBlocks, threadsPerBlock, 0, stream[4]>>>(_vSrcTerm_dev, _p_dev);

    calcBCMomSrcTermKernel<north, velBCs::type[north]><<<numBlocks, threadsPerBlock, 0, stream[5]>>>(_vSrcTerm_dev, _p_dev);

    if constexpr (dim == 3) {
        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = 1;

        calcBCMomSrcTermKernel<bottom, velBCs::type[bottom]><<<numBlocks, threadsPerBlock, 0, stream[7]>>>(_wSrcTerm_dev, _p_dev);

        calcBCMomSrcTermKernel<top, velBCs::type[top]><<<numBlocks, threadsPerBlock, 0, stream[8]>>>(_wSrcTerm_dev, _p_dev);
    }

    for (int i = 0; i < 3*dim; ++i) {
        cudaStreamSynchronize(stream[i]);
    }

    for (int i = 0; i < 3*dim; ++i) {
        cudaStreamDestroy(stream[i]);
    }
}

void Solver::applyBCsToMomEq() {

    cudaStream_t stream[2*dim];
    for (int i = 0; i < 2*dim; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    dim3 threadsPerBlock;
    dim3 numBlocks;

    if constexpr (dim == 2) {
        threadsPerBlock = dim3(1024, 1, 1);
    } else if constexpr (dim == 3) {
        threadsPerBlock = dim3(32, 32, 1);
    }

    numBlocks.x = (ny + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (nz + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;

    applyBCsToMomEqKernel<east, velBCs::type[east]><<<numBlocks, threadsPerBlock, 0, stream[0]>>>(_uCoef_dev, _vCoef_dev, _wCoef_dev
        , _uSrcTerm_dev, _vSrcTerm_dev, _wSrcTerm_dev, velBCs::val[east][0], velBCs::val[east][1], velBCs::val[east][2]);

    applyBCsToMomEqKernel<west, velBCs::type[west]><<<numBlocks, threadsPerBlock, 0, stream[1]>>>(_uCoef_dev, _vCoef_dev, _wCoef_dev
        , _uSrcTerm_dev, _vSrcTerm_dev, _wSrcTerm_dev, velBCs::val[west][0], velBCs::val[west][1], velBCs::val[west][2]);

    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (nz + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;

    applyBCsToMomEqKernel<north, velBCs::type[north]><<<numBlocks, threadsPerBlock, 0, stream[2]>>>(_uCoef_dev, _vCoef_dev, _wCoef_dev
        , _uSrcTerm_dev, _vSrcTerm_dev, _wSrcTerm_dev, velBCs::val[north][0], velBCs::val[north][1], velBCs::val[north][2]);

    applyBCsToMomEqKernel<south, velBCs::type[south]><<<numBlocks, threadsPerBlock, 0, stream[3]>>>(_uCoef_dev, _vCoef_dev, _wCoef_dev
        , _uSrcTerm_dev, _vSrcTerm_dev, _wSrcTerm_dev, velBCs::val[south][0], velBCs::val[south][1], velBCs::val[south][2]);

    if constexpr (dim == 3) {
        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = 1;

        applyBCsToMomEqKernel<top, velBCs::type[top]><<<numBlocks, threadsPerBlock, 0, stream[4]>>>(_uCoef_dev, _vCoef_dev, _wCoef_dev
            , _uSrcTerm_dev, _vSrcTerm_dev, _wSrcTerm_dev, velBCs::val[top][0], velBCs::val[top][1], velBCs::val[top][2]);

        applyBCsToMomEqKernel<bottom, velBCs::type[bottom]><<<numBlocks, threadsPerBlock, 0, stream[5]>>>(_uCoef_dev, _vCoef_dev, _wCoef_dev
            , _uSrcTerm_dev, _vSrcTerm_dev, _wSrcTerm_dev, velBCs::val[bottom][0], velBCs::val[bottom][1], velBCs::val[bottom][2]);
    }

    for (int i = 0; i < 2*dim; ++i) {
        cudaStreamSynchronize(stream[i]);
    }

    for (int i = 0; i < 2*dim; ++i) {
        cudaStreamDestroy(stream[i]);
    }
}

void Solver::RhieChowInterpolate() {

    cudaStream_t stream[3*dim];
    for (int i = 0; i < 3*dim; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    dim3 threadsPerBlock;
    dim3 numBlocks;

    if constexpr (dim == 2) {
        threadsPerBlock = dim3(32, 32, 1);
    } else if constexpr (dim == 3) {
        threadsPerBlock = dim3(16, 8, 8);
    }

    numBlocks.x = (nx - 3 + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;

    interiorRhieChowInterpolateKernel<xDir><<<numBlocks, threadsPerBlock, 0, stream[0]>>>(_uf_dev, _u_dev, _uCoef_dev, _p_dev);

    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny - 3 + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;

    interiorRhieChowInterpolateKernel<yDir><<<numBlocks, threadsPerBlock, 0, stream[1]>>>(_vf_dev, _v_dev, _vCoef_dev, _p_dev);

    if constexpr (dim == 3) {
        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = (nz - 3 + threadsPerBlock.z - 1) / threadsPerBlock.z;

        interiorRhieChowInterpolateKernel<zDir><<<numBlocks, threadsPerBlock, 0, stream[6]>>>(_wf_dev, _w_dev, _wCoef_dev, _p_dev);
    }

    if constexpr (dim == 2) {
        threadsPerBlock = dim3(1024, 1, 1);
    } else if constexpr (dim == 3) {
        threadsPerBlock = dim3(32, 32, 1);
    }

    numBlocks.x = (ny + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (nz + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;

    bcRhieChowInterpolateKernel<west, velBCs::type[west]><<<numBlocks, threadsPerBlock, 0, stream[2]>>>(_uf_dev, _u_dev, _uCoef_dev, _p_dev);

    bcRhieChowInterpolateKernel<east, velBCs::type[east]><<<numBlocks, threadsPerBlock, 0, stream[3]>>>(_uf_dev, _u_dev, _uCoef_dev, _p_dev);

    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (nz + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;

    bcRhieChowInterpolateKernel<south, velBCs::type[south]><<<numBlocks, threadsPerBlock, 0, stream[4]>>>(_vf_dev, _v_dev, _vCoef_dev, _p_dev);

    bcRhieChowInterpolateKernel<north, velBCs::type[north]><<<numBlocks, threadsPerBlock, 0, stream[5]>>>(_vf_dev, _v_dev, _vCoef_dev, _p_dev);

    if constexpr (dim == 3) {
        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = 1;

        bcRhieChowInterpolateKernel<bottom, velBCs::type[bottom]><<<numBlocks, threadsPerBlock, 0, stream[7]>>>(_wf_dev, _w_dev, _wCoef_dev, _p_dev);

        bcRhieChowInterpolateKernel<top, velBCs::type[top]><<<numBlocks, threadsPerBlock, 0, stream[8]>>>(_wf_dev, _w_dev, _wCoef_dev, _p_dev);
    }

    for (int i = 0; i < 3*dim; ++i) {
        cudaStreamSynchronize(stream[i]);
    }

    for (int i = 0; i < 3*dim; ++i) {
        cudaStreamDestroy(stream[i]);
    }
}

void Solver::calcPresCorrLinkCoef() {

    dim3 threadsPerBlock;
    dim3 numBlocks; 

    if constexpr (dim == 2) {
        threadsPerBlock = dim3(32, 32, 1);
    } else if constexpr (dim == 3) {
        threadsPerBlock = dim3(16, 8, 8);
    }
    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;

    calcPresCorrLinkCoefKernel<<<numBlocks, threadsPerBlock>>>(_pCorrCoef_dev, _uCoef_dev, _vCoef_dev, _wCoef_dev);

    cudaDeviceSynchronize();
}

void Solver::calcPresCorrSrcTerm() {

    cudaMemset(_massNorm_dev, 0, sizeof(scalar));
    
    dim3 threadsPerBlock;
    dim3 numBlocks; 

    if constexpr (dim == 2) {
        threadsPerBlock = dim3(32, 32, 1);
    } else if constexpr (dim == 3) {
        threadsPerBlock = dim3(16, 8, 8);
    }
    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;

    int blockSize = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z;

    calcPresCorrSrcTermKernel<<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize>>>(_pCorrSrcTerm_dev, _uf_dev, _vf_dev, _wf_dev, _massNorm_dev);

    cudaDeviceSynchronize();
}

void Solver::updateField() {

    cudaStream_t stream[4*dim+1];

    for (int i = 0; i < 4*dim+1; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    cudaMemsetAsync(_pNorm_dev, 0, sizeof(scalar), stream[0]);
    cudaMemsetAsync(_uNorm_dev, 0, sizeof(scalar), stream[1]);
    cudaMemsetAsync(_vNorm_dev, 0, sizeof(scalar), stream[2]);
    cudaMemsetAsync(_ufNorm_dev, 0, sizeof(scalar), stream[3]);
    cudaMemsetAsync(_vfNorm_dev, 0, sizeof(scalar), stream[4]);
    if constexpr (dim == 3) {
        cudaMemsetAsync(_wNorm_dev, 0, sizeof(scalar), stream[9]);
        cudaMemsetAsync(_wfNorm_dev, 0, sizeof(scalar), stream[10]);
    }

    dim3 threadsPerBlock;
    dim3 numBlocks; 

    if constexpr (dim == 2) {
        threadsPerBlock = dim3(32, 32, 1);
    } else if constexpr (dim == 3) {
        threadsPerBlock = dim3(16, 8, 8);
    }

    int blockSize = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z;

    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;

    updatePresKernel<<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[0]>>>(_p_dev, _pCorr_dev, _pNorm_dev, relax_p);

    numBlocks.x = (nx-2 + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny   + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz   + threadsPerBlock.z - 1) / threadsPerBlock.z;

    updateInteriorVelKernel<xDir><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[1]>>>(_u_dev, _uCoef_dev, _pCorr_dev
        , _uNorm_dev, relax_u);

    numBlocks.x = (nx   + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny-2 + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz   + threadsPerBlock.z - 1) / threadsPerBlock.z;

    updateInteriorVelKernel<yDir><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[2]>>>(_v_dev, _vCoef_dev, _pCorr_dev
        , _vNorm_dev, relax_v);

    numBlocks.x = (nx-1 + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny   + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz   + threadsPerBlock.z - 1) / threadsPerBlock.z;

    updateFaceVelKernel<xDir><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[3]>>>(_uf_dev, _uCoef_dev, _pCorr_dev
        , _ufNorm_dev, relax_u);

    numBlocks.x = (nx   + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny-1 + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz   + threadsPerBlock.z - 1) / threadsPerBlock.z;

    updateFaceVelKernel<yDir><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[4]>>>(_vf_dev, _vCoef_dev, _pCorr_dev
        , _vfNorm_dev, relax_v);

    if constexpr (dim == 3) {
        numBlocks.x = (nx   + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny   + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = (nz-2 + threadsPerBlock.z - 1) / threadsPerBlock.z;

        updateInteriorVelKernel<zDir><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[9]>>>(_w_dev, _wCoef_dev, _pCorr_dev
            , _wNorm_dev, relax_w);

        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = (nz+1 + threadsPerBlock.z - 1) / threadsPerBlock.z;

        updateFaceVelKernel<zDir><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[10]>>>(_wf_dev, _wCoef_dev, _pCorr_dev
            , _wfNorm_dev, relax_w);
    }

    if constexpr (dim == 2) {
        threadsPerBlock = dim3(1024, 1, 1);
    } else if constexpr (dim == 3) {
        threadsPerBlock = dim3(32, 32, 1);
    }

    blockSize = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z;

    numBlocks.x = (ny + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (nz + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;

    updateBCVelKernel<west, velBCs::type[west]><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[5]>>>(_u_dev, _uCoef_dev, _pCorr_dev
        , _uNorm_dev, relax_u);

    updateBCVelKernel<east, velBCs::type[east]><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[6]>>>(_u_dev, _uCoef_dev, _pCorr_dev
        , _uNorm_dev, relax_u);

    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (nz + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;

    updateBCVelKernel<south, velBCs::type[south]><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[7]>>>(_v_dev, _vCoef_dev, _pCorr_dev
        , _vNorm_dev, relax_v);

    updateBCVelKernel<north, velBCs::type[north]><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[8]>>>(_v_dev, _vCoef_dev, _pCorr_dev
        , _vNorm_dev, relax_v);

    if constexpr (dim == 3) {
        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = 1;

        updateBCVelKernel<bottom, velBCs::type[bottom]><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[11]>>>(_w_dev, _wCoef_dev, _pCorr_dev
        , _wNorm_dev, relax_w);

        updateBCVelKernel<top, velBCs::type[top]><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[12]>>>(_w_dev, _wCoef_dev, _pCorr_dev
        , _wNorm_dev, relax_w);
    }

    for (int i = 0; i < 4*dim+1; ++i) {
        cudaStreamSynchronize(stream[i]);
    }

    for (int i = 0; i < 4*dim+1; ++i) {
        cudaStreamDestroy(stream[i]);
    }

}

void Solver::pointJacobiIterate(scalar *field_dev, size_t fieldSize, scalar *coef_dev, scalar *srcTerm_dev, int nIter, scalar relax, scalar tol) {

    scalar *field0_dev, *norm_dev;
    cudaMalloc(&field0_dev, fieldSize);
    cudaMalloc(&norm_dev, sizeof(scalar));

    cudaMemset(field0_dev, 0, fieldSize);

    dim3 threadsPerBlock;
    dim3 numBlocks;

    if constexpr (dim == 2) {
        threadsPerBlock = dim3(32, 32, 1);
    } else if constexpr (dim == 3) {
        threadsPerBlock = dim3(16, 16, 8);
    }
    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;

    int blockSize = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z;

    scalar maxNorm = -1e20;

    for (int it = 0; it < nIter; ++it) {
        
        scalar *tmp = field_dev;
        field_dev = field0_dev;
        field0_dev = tmp;

        scalar norm = 0.0;
        cudaMemset(norm_dev, 0, sizeof(scalar));

        pointJacobiIterateKernel<<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize>>>(field_dev, field0_dev, coef_dev
            , srcTerm_dev, norm_dev, relax);
        
        cudaDeviceSynchronize();
        
        cudaMemcpy(&norm, norm_dev, sizeof(scalar), cudaMemcpyDeviceToHost);

        norm = sqrt(norm / (nx * ny * nz));

        maxNorm = max(norm, maxNorm);

        scalar relNorm = norm / (maxNorm + 1e-20);    // relative residual
        if (relNorm < tol) {
            break;
        }
    }

    cudaFree(field0_dev);
    cudaFree(norm_dev);
}

void Solver::GaussSeidelIterate(scalar *field_dev, scalar *coef_dev, scalar *srcTerm_dev, int nIter, scalar relax, scalar tol) {

    scalar *norm_dev;
    cudaMalloc(&norm_dev, sizeof(scalar));

    dim3 threadsPerBlock;
    dim3 numBlocks;

    if constexpr (dim == 2) {
        threadsPerBlock = dim3(32, 32, 1);
    } else if constexpr (dim == 3) {
        threadsPerBlock = dim3(16, 16, 8);
    }
    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;

    int blockSize = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z;

    scalar maxNorm = -1e20;

    for (int it = 0; it < nIter; ++it) {

        scalar norm = 0.0;
        cudaMemset(norm_dev, 0, sizeof(scalar));

        GaussSeidelIterateKernel<<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize>>>(field_dev, coef_dev, srcTerm_dev
            , norm_dev, relax);
        
        cudaDeviceSynchronize();
        
        cudaMemcpy(&norm, norm_dev, sizeof(scalar), cudaMemcpyDeviceToHost);

        norm = sqrt(norm / (nx * ny * nz));

        maxNorm = max(norm, maxNorm);

        scalar relNorm = norm / (maxNorm + 1e-20);    // relative residual
        if (relNorm < tol) {
            break;
        }
    }

    cudaFree(norm_dev);
}