#include "kernels.cuh"
#include "config.h"
#include "constants.h"
#include <cmath>
#include <algorithm>

using namespace std;

__global__ void initUfKernel(scalar *u, scalar *uf) {

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int id_c  = i + (nx+1) * (j + ny * k);
        int id_W = (i-1) + nx * (j + ny * k);
        int id_E = i + nx * (j + ny * k);
        uf[id_c] = (u[id_W] + u[id_E]) / 2;
    }
}

__global__ void initVfKernel(scalar *v, scalar *vf) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int id_c = i + nx * (j + (ny+1) * k);
        int id_S = i + nx * (j-1 + ny * k);
        int id_N = i + nx * (j + ny * k);
        vf[id_c] = (v[id_S] + v[id_N]) / 2;
    }
}

__global__ void initWfKernel(scalar *w, scalar *wf) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i < nx && j < ny && k < nz) {
        int id_c = i + nx * (j + ny * k);
        int id_B = i + nx * (j + ny * (k-1));
        int id_T = i + nx * (j + ny * k);
        wf[id_c] = (w[id_B] + w[id_T]) / 2;
    }
}

void initFaceVel(scalar *u_dev, scalar *v_dev, scalar *w_dev, scalar *uf_dev, scalar *vf_dev, scalar *wf_dev) {

    cudaStream_t stream[dim];
    for (int i = 0; i < dim; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    dim3 threadsPerBlock;
    dim3 numBlocks;
    if (dim == 2) {
        threadsPerBlock = dim3(32, 32, 1);
    } else if (dim == 3) {
        threadsPerBlock = dim3(16, 16, 8);
    }

    numBlocks.x = (nx - 1 + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;
    initUfKernel<<<numBlocks, threadsPerBlock, 0, stream[0]>>>(u_dev, uf_dev);

    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny - 1 + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;
    initVfKernel<<<numBlocks, threadsPerBlock, 0, stream[1]>>>(v_dev, vf_dev);

    if (dim == 3) {
        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = (nz - 1 + threadsPerBlock.z - 1) / threadsPerBlock.z;
        initWfKernel<<<numBlocks, threadsPerBlock, 0, stream[2]>>>(w_dev, wf_dev);
    }

    for (int i = 0; i < dim; ++i) {
        cudaStreamSynchronize(stream[i]);
    }

    for (int i = 0; i < dim; ++i) {
        cudaStreamDestroy(stream[i]);
    }
}

__global__ void applyBCsToUfKernel(scalar *uf, scalar *u, int i, int type, scalar val) {

    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (j < ny && k < nz) {
        int id = i + (nx+1) * (j + ny * k);
        if (type == 0) { // "wall"
            uf[id] = 0;
        } else if (type == 1) { // "inlet"
            uf[id] = val;
        } else if (type == 2) { // "outlet"
            if (i == 0) {
                uf[id] = u[0 + nx * (j + ny * k)];
            } else if (i == nx) {
                uf[id] = u[nx-1 + nx * (j + ny * k)];
            }
        }
    }
}

__global__ void applyBCsToVfKernel(scalar *vf, scalar *v, int j, int type, scalar val) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && k < nz) {
        int id = i + nx * (j + (ny+1) * k);
        if (type == 0) { // "wall"
            vf[id] = 0;
        } else if (type == 1) { // "inlet"
            vf[id] = val;
        } else if (type == 2) { // "outlet"
            if (j == 0) {
                vf[id] = v[i + nx * (0 + ny * k)];
            } else if (j == ny) {
                vf[id] = v[i + nx * (ny-1 + ny * k)];
            }
        }
    }
}

__global__ void applyBCsToWfKernel(scalar *wf, scalar *w, int k, int type, scalar val) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx && j < ny) {
        int id = i + nx * (j + ny * k);
        if (type == 0) { // "wall"
            wf[id] = 0;
        } else if (type == 1) { // "inlet"
            wf[id] = val;
        } else if (type == 2) { // "outlet"
            if (k == 0) {
                wf[id] = w[i + nx * (j + ny * 0)];
            } else if (k == nz) {
                wf[id] = w[i + nx * (j + ny * (nz-1))];
            }
        }
    }
}

void applyBCsToFaceVel(scalar *uf_dev, scalar *vf_dev, scalar *wf_dev, scalar *u_dev, scalar *v_dev, scalar *w_dev) {

    cudaStream_t stream[2*dim];
    for (int i = 0; i < 2*dim; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    dim3 threadsPerBlock;
    dim3 numBlocks;

    if (dim == 2) {
        threadsPerBlock = dim3(1, 1024, 1);
    } else if (dim == 3) {
        threadsPerBlock = dim3(1, 32, 32);
    }
    numBlocks.x = 1;
    numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;
    applyBCsToUfKernel<<<numBlocks, threadsPerBlock, 0, stream[0]>>>(uf_dev, u_dev, 0, velBCs::type[west], velBCs::val[west][0]);
    applyBCsToUfKernel<<<numBlocks, threadsPerBlock, 0, stream[1]>>>(uf_dev, u_dev, nx, velBCs::type[east], velBCs::val[east][0]);

    if (dim == 2) {
        threadsPerBlock = dim3(1024, 1, 1);
    } else if (dim == 3) {
        threadsPerBlock = dim3(32, 1, 32);
    }
    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = 1;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;
    applyBCsToVfKernel<<<numBlocks, threadsPerBlock, 0, stream[2]>>>(vf_dev, v_dev, 0, velBCs::type[south], velBCs::val[south][1]);
    applyBCsToVfKernel<<<numBlocks, threadsPerBlock, 0, stream[3]>>>(vf_dev, v_dev, ny, velBCs::type[north], velBCs::val[north][1]);

    if (dim == 3) {
        threadsPerBlock = dim3(32, 32, 1);
        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = 1;
        applyBCsToWfKernel<<<numBlocks, threadsPerBlock, 0, stream[4]>>>(wf_dev, w_dev, 0, velBCs::type[bottom], velBCs::val[bottom][2]);
        applyBCsToWfKernel<<<numBlocks, threadsPerBlock, 0, stream[5]>>>(wf_dev, w_dev, nz, velBCs::type[top], velBCs::val[top][2]);
    }

    for (int i = 0; i < 2*dim; ++i) {
        cudaStreamSynchronize(stream[i]);
    }

    for (int i = 0; i < 2*dim; ++i) {
        cudaStreamDestroy(stream[i]);
    }
}

__global__ void calcMomLinkCoefKernel(scalar *coef, scalar *uf, scalar *vf, scalar *wf) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        scalar ue = uf[i+1 + (nx+1) * (j   + ny     * k)];
        scalar uw = uf[i   + (nx+1) * (j   + ny     * k)];
        scalar vn = vf[i   + nx     * (j+1 + (ny+1) * k)];
        scalar vs = vf[i   + nx     * (j   + (ny+1) * k)];

        int id_aE = i + nx * (j + ny * (k + aE));
        int id_aW = i + nx * (j + ny * (k + aW));
        int id_aN = i + nx * (j + ny * (k + aN));
        int id_aS = i + nx * (j + ny * (k + aS));
        int id_aC = i + nx * (j + ny * (k + aC));

        coef[id_aE] = (density * (ue  - abs(ue)) * areaX) / 2 - dynamicViscosity * areaX / dx;
        coef[id_aW] = (density * (-uw - abs(uw)) * areaX) / 2 - dynamicViscosity * areaX / dx;
        coef[id_aN] = (density * (vn  - abs(vn)) * areaY) / 2 - dynamicViscosity * areaY / dy;
        coef[id_aS] = (density * (-vs - abs(vs)) * areaY) / 2 - dynamicViscosity * areaY / dy;
        coef[id_aC] = -(coef[id_aE] + coef[id_aW] + coef[id_aN] + coef[id_aS]);

        if (dim == 3) {
            scalar wt = wf[k+1 + (nz+1) * (i + nx * j)];
            scalar wb = wf[k   + (nz+1) * (i + nx * j)];

            int id_aT = i + nx * (j + ny * (k + aT));
            int id_aB = i + nx * (j + ny * (k + aB));

            coef[id_aT] = (density * (wt  - abs(wt)) * areaZ) / 2 - dynamicViscosity * areaZ / dz;
            coef[id_aB] = (density * (-wb - abs(wb)) * areaZ) / 2 - dynamicViscosity * areaZ / dz;
            coef[id_aC] += -(coef[id_aT] + coef[id_aB]);
        }
    }
}

void calcMomLinkCoef(scalar *coef_dev, scalar *uf_dev, scalar *vf_dev, scalar *wf_dev) {

    dim3 threadsPerBlock;
    dim3 numBlocks; 

    if (dim == 2) {
        threadsPerBlock = dim3(32, 32, 1);
    } else if (dim == 3) {
        threadsPerBlock = dim3(16, 8, 8);
    }
    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;

    calcMomLinkCoefKernel<<<numBlocks, threadsPerBlock>>>(coef_dev, uf_dev, vf_dev, wf_dev);

    cudaDeviceSynchronize();
}

__global__ void calcMomXSrcTermKernel(scalar *uSrcTerm, scalar *p, int typeW, int typeE) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int id_C = i   + nx * (j + ny * k);
        int id_W = i-1 + nx * (j + ny * k);
        int id_E = i+1 + nx * (j + ny * k);

        if (i == 0) {
            if (typeW == 0 || typeW == 1) { // "wall" or "inlet"
                uSrcTerm[id_C] = 0.5 * (p[id_C] - p[id_E]) * areaX;
            } else if (typeW == 2) { // "outlet"
                uSrcTerm[id_C] = 0.5 * (p[id_C] + p[id_E]) * areaX;
            }
        } else if (i == nx - 1) {
            if (typeE == 0 || typeE == 1) { // "wall" or "inlet"
                uSrcTerm[id_C] = 0.5 * (p[id_W] - p[id_C]) * areaX;
            } else if (typeE == 2) { // "outlet"
                uSrcTerm[id_C] = 0.5 * (p[id_W] + p[id_C]) * areaX;
            }
        } else {
            uSrcTerm[id_C] = 0.5 * (p[id_W] - p[id_E]) * areaX;
        }
    }
}

__global__ void calcMomYSrcTermKernel(scalar *vSrcTerm, scalar *p, int typeS, int typeN) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int id_C = i + nx * (j + ny * k);
        int id_S = i + nx * (j-1 + ny * k);
        int id_N = i + nx * (j+1 + ny * k);

        if (j == 0) {
            if (typeS == 0 || typeS == 1) { // "wall" or "inlet"
                vSrcTerm[id_C] = 0.5 * (p[id_C] - p[id_N]) * areaY;
            } else if (typeS == 2) { // "outlet"
                vSrcTerm[id_C] = 0.5 * (p[id_C] + p[id_N]) * areaY;
            }
        } else if (j == ny - 1) {
            if (typeN == 0 || typeN == 1) { // "wall" or "inlet"
                vSrcTerm[id_C] = 0.5 * (p[id_S] - p[id_C]) * areaY;
            } else if (typeN == 2) { // "outlet"
                vSrcTerm[id_C] = 0.5 * (p[id_S] + p[id_C]) * areaY;
            }
        } else {
            vSrcTerm[id_C] = 0.5 * (p[id_S] - p[id_N]) * areaY;
        }
    }
}

__global__ void calcMomZSrcTermKernel(scalar *wSrcTerm, scalar *p, int typeB, int typeT) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int id_C = i + nx * (j + ny * k);
        int id_B = i + nx * (j + ny * k-1);
        int id_T = i + nx * (j + ny * k+1);

        if (k == 0) {
            if (typeB == 0 || typeB == 1) { // "wall" or "inlet"
                wSrcTerm[id_C] = 0.5 * (p[id_C] - p[id_T]) * areaZ;
            } else if (typeB == 2) { // "outlet"
                wSrcTerm[id_C] = 0.5 * (p[id_C] + p[id_T]) * areaZ;
            }
        } else if (k == nz - 1) {
            if (typeT == 0 || typeT == 1) { // "wall" or "inlet"
                wSrcTerm[id_C] = 0.5 * (p[id_B] - p[id_C]) * areaZ;
            } else if (typeT == 2) { // "outlet"
                wSrcTerm[id_C] = 0.5 * (p[id_B] + p[id_C]) * areaZ;
            }
        } else {
            wSrcTerm[id_C] = 0.5 * (p[id_B] - p[id_T]) * areaZ;
        }
    }
}

void calcMomSrcTerm(scalar *uSrcTerm_dev, scalar *vSrcTerm_dev, scalar *wSrcTerm_dev, scalar *p_dev) {

    cudaStream_t stream[dim];
    for (int i = 0; i < dim; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    dim3 threadsPerBlock;
    dim3 numBlocks; 

    if (dim == 2) {
        threadsPerBlock = dim3(32, 32, 1);
    } else if (dim == 3) {
        threadsPerBlock = dim3(16, 8, 8);
    }
    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;

    calcMomXSrcTermKernel<<<numBlocks, threadsPerBlock, 0, stream[0]>>>(uSrcTerm_dev, p_dev, velBCs::type[west], velBCs::type[east]);
    calcMomYSrcTermKernel<<<numBlocks, threadsPerBlock, 0, stream[1]>>>(vSrcTerm_dev, p_dev, velBCs::type[south], velBCs::type[north]);
    if (dim == 3) {
        calcMomZSrcTermKernel<<<numBlocks, threadsPerBlock, 0, stream[2]>>>(wSrcTerm_dev, p_dev, velBCs::type[bottom], velBCs::type[top]);
    }

    for (int i = 0; i < dim; ++i) {
        cudaStreamSynchronize(stream[i]);
    }

    for (int i = 0; i < dim; ++i) {
        cudaStreamDestroy(stream[i]);
    }
}

__global__ void applyBCsToMomEqKernel(scalar *uCoef, scalar *vCoef, scalar *wCoef, scalar *uSrcTerm, scalar *vSrcTerm
    , scalar *wSrcTerm, int location, int type, scalar uBC, scalar vBC, scalar wBC) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (((location == east  || location == west  ) && i < ny && j < nz) ||
        ((location == north || location == south ) && i < nx && j < nz) ||
        ((location == top   || location == bottom) && i < nx && j < ny)) {

        int id_aC, id_aNB, id_b;
        if (location == east) {
            id_aC  = nx-1 + nx * (i + ny * (j + aC));
            id_aNB = nx-1 + nx * (i + ny * (j + aE));
            id_b   = nx-1 + nx * (i + ny * j);
        } else if (location == west) {
            id_aC  = 0 + nx * (i + ny * (j + aC));
            id_aNB = 0 + nx * (i + ny * (j + aW));
            id_b   = 0 + nx * (i + ny * j);
        } else if (location == north) {
            id_aC  = i + nx * (ny-1 + ny * (j + aC));
            id_aNB = i + nx * (ny-1 + ny * (j + aN));
            id_b   = i + nx * (ny-1 + ny * j);
        } else if (location == south) {
            id_aC  = i + nx * (0 + ny * (j + aC));
            id_aNB = i + nx * (0 + ny * (j + aS));
            id_b   = i + nx * (0 + ny * j);
        } else if (location == top) {
            id_aC  = i + nx * (j + ny * (nz-1 + aC));
            id_aNB = i + nx * (j + ny * (nz-1 + aT));
            id_b   = i + nx * (j + ny * (nz-1));
        } else if (location == bottom) {
            id_aC  = i + nx * (j + ny * (0 + aC));
            id_aNB = i + nx * (j + ny * (0 + aB));
            id_b   = i + nx * (j + ny * 0);
        }

        if (type == 0) { // "wall"
            uCoef[id_aC] -= uCoef[id_aNB];
            if (location != east && location != west) {
                uSrcTerm[id_b] -= 2 * uCoef[id_aNB] * uBC;
            }
            uCoef[id_aNB] = 0;

            vCoef[id_aC] -= vCoef[id_aNB];
            if (location != north && location != south) {
                vSrcTerm[id_b] -= 2 * vCoef[id_aNB] * vBC;
            }
            vCoef[id_aNB] = 0;

            if (dim == 3) {
                wCoef[id_aC] -= wCoef[id_aNB];
                if (location != top && location != bottom) {
                    wSrcTerm[id_b] -= 2 * wCoef[id_aNB] * wBC;
                }
                wCoef[id_aNB] = 0;
            }
        } else if (type == 1) { // "inlet"
            uCoef[id_aC] -= uCoef[id_aNB];
            uSrcTerm[id_b] -= 2 * uCoef[id_aNB] * uBC;
            uCoef[id_aNB] = 0;

            vCoef[id_aC] -= vCoef[id_aNB];
            vSrcTerm[id_b] -= 2 * vCoef[id_aNB] * vBC;
            vCoef[id_aNB] = 0;

            if (dim == 3) {
                wCoef[id_aC] -= wCoef[id_aNB];
                wSrcTerm[id_b] -= 2 * wCoef[id_aNB] * wBC;
                wCoef[id_aNB] = 0;
            }
        } else if (type == 2) { // "outlet"
            uCoef[id_aC] += uCoef[id_aNB];
            uCoef[id_aNB] = 0;

            vCoef[id_aC] += vCoef[id_aNB];
            vCoef[id_aNB] = 0;

            if (dim == 3) {
                wCoef[id_aC] += wCoef[id_aNB];
                wCoef[id_aNB] = 0;
            }
        }
    }
}

void applyBCsToMomEq(scalar *uCoef_dev, scalar *uSrcTerm_dev, scalar *vCoef_dev, scalar *vSrcTerm_dev, scalar *wCoef_dev
    , scalar *wSrcTerm_dev) {

    cudaStream_t stream[2*dim];
    for (int i = 0; i < 2*dim; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    dim3 threadsPerBlock;
    dim3 numBlocks;

    if (dim == 2) {
        threadsPerBlock = dim3(1024, 1, 1);
    } else if (dim == 3) {
        threadsPerBlock = dim3(32, 32, 1);
    }

    numBlocks.x = (ny + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (nz + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;
    applyBCsToMomEqKernel<<<numBlocks, threadsPerBlock, 0, stream[0]>>>(uCoef_dev, vCoef_dev, wCoef_dev, uSrcTerm_dev, vSrcTerm_dev
        , wSrcTerm_dev, east, velBCs::type[east], velBCs::val[east][0], velBCs::val[east][1], velBCs::val[east][2]);
    applyBCsToMomEqKernel<<<numBlocks, threadsPerBlock, 0, stream[1]>>>(uCoef_dev, vCoef_dev, wCoef_dev, uSrcTerm_dev, vSrcTerm_dev
        , wSrcTerm_dev, west, velBCs::type[west], velBCs::val[west][0], velBCs::val[west][1], velBCs::val[west][2]);

    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (nz + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;
    applyBCsToMomEqKernel<<<numBlocks, threadsPerBlock, 0, stream[2]>>>(uCoef_dev, vCoef_dev, wCoef_dev, uSrcTerm_dev, vSrcTerm_dev
        , wSrcTerm_dev, north, velBCs::type[north], velBCs::val[north][0], velBCs::val[north][1], velBCs::val[north][2]);
    applyBCsToMomEqKernel<<<numBlocks, threadsPerBlock, 0, stream[3]>>>(uCoef_dev, vCoef_dev, wCoef_dev, uSrcTerm_dev, vSrcTerm_dev
        , wSrcTerm_dev, south, velBCs::type[south], velBCs::val[south][0], velBCs::val[south][1], velBCs::val[south][2]);

    if (dim == 3) {
        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = 1;
        applyBCsToMomEqKernel<<<numBlocks, threadsPerBlock, 0, stream[4]>>>(uCoef_dev, vCoef_dev, wCoef_dev, uSrcTerm_dev, vSrcTerm_dev
            , wSrcTerm_dev, top, velBCs::type[top], velBCs::val[top][0], velBCs::val[top][1], velBCs::val[top][2]);
        applyBCsToMomEqKernel<<<numBlocks, threadsPerBlock, 0, stream[5]>>>(uCoef_dev, vCoef_dev, wCoef_dev, uSrcTerm_dev, vSrcTerm_dev
            , wSrcTerm_dev, bottom, velBCs::type[bottom], velBCs::val[bottom][0], velBCs::val[bottom][1], velBCs::val[bottom][2]);
    }

    for (int i = 0; i < 2*dim; ++i) {
        cudaStreamSynchronize(stream[i]);
    }

    for (int i = 0; i < 2*dim; ++i) {
        cudaStreamDestroy(stream[i]);
    }
}

__global__ void pointJacobiIterateKernel(scalar *field, scalar* field0, scalar *coef, scalar *srcTerm, scalar *norm) {

    extern __shared__ scalar sharedNorm[];

    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int id_C = i + nx * (j + ny * k);
        scalar phi_C = field0[id_C];
        // east
        scalar phi_E = 0;
        if ( i != nx - 1 ) {
            int id_E = i+1 + nx * (j + ny * k);
            phi_E = field0[id_E];
        }
        // west
        scalar phi_W = 0;
        if ( i != 0 ) {
            int id_W = i-1 + nx * (j + ny * k);
            phi_W = field0[id_W];
        }
        // north
        scalar phi_N = 0;
        if ( j != ny - 1 ) {
            int id_N = i + nx * (j+1 + ny * k);
            phi_N = field0[id_N];
        }
        // south
        scalar phi_S = 0;
        if ( j != 0 ) {
            int id_S = i + nx * (j-1 + ny * k);
            phi_S = field0[id_S];
        }
        // top
        scalar phi_T = 0;
        if ( k != nz - 1 ) {
            int id_T = i + nx * (j + ny * (k+1));
            phi_T = field0[id_T];
        }
        // bottom
        scalar phi_B = 0;
        if ( k != 0 ) {
            int id_B = i + nx * (j + ny * (k-1));
            phi_B = field0[id_B];
        }
        
        scalar newPhi = srcTerm[id_C]
            - coef[i+nx*(j+ny*(k+aE))] * phi_E
            - coef[i+nx*(j+ny*(k+aW))] * phi_W
            - coef[i+nx*(j+ny*(k+aN))] * phi_N
            - coef[i+nx*(j+ny*(k+aS))] * phi_S;
        if (dim == 3) {
            newPhi -= coef[i+nx*(j+ny*(k+aT))] * phi_T + coef[i+nx*(j+ny*(k+aB))] * phi_B;
        }
        newPhi /= coef[i+nx*(j+ny*(k+aC))];

        scalar dPhi = relax * (newPhi - phi_C);

        field[id_C] = phi_C + dPhi;

        sharedNorm[tid] = dPhi*dPhi;
        __syncthreads();

        int stride = blockDim.x * blockDim.y * blockDim.z;
        for (int s = stride / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sharedNorm[tid] += sharedNorm[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            atomicAdd(norm, sharedNorm[0]);
        }
    }
}

void pointJacobiIterate(scalar *field_dev, size_t fieldSize, scalar *coef_dev, scalar *srcTerm_dev) {

    scalar *field0_dev, *norm_dev;
    cudaMalloc(&field0_dev, fieldSize);
    cudaMalloc(&norm_dev, sizeof(scalar));

    cudaMemset(field0_dev, 0.0, fieldSize);

    dim3 threadsPerBlock;
    dim3 numBlocks;
    if (dim == 2) {
        threadsPerBlock = dim3(32, 32, 1);

        numBlocks.x = (nx + 31) / 32;
        numBlocks.y = (ny + 31) / 32;
        numBlocks.z = 1;
    } else if (dim == 3) {
        threadsPerBlock = dim3(16, 16, 8);

        numBlocks.x = (nx + 15) / 16;
        numBlocks.y = (ny + 7) / 8;
        numBlocks.z = (nz + 7) / 8;
    }

    int blockSize = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z;

    scalar maxNorm = -1e20;

    for (int it = 0; it < numInnerIter; ++it) {
        
        scalar *tmp = field_dev;
        field_dev = field0_dev;
        field0_dev = tmp;

        scalar norm = 0.0;
        cudaMemcpy(norm_dev, &norm, sizeof(scalar), cudaMemcpyHostToDevice);

        pointJacobiIterateKernel<<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize>>>(field_dev, field0_dev, coef_dev, srcTerm_dev, norm_dev);
        
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

__global__ void GaussSeidelIterateKernel(scalar *field, scalar *coef, scalar *srcTerm, scalar *norm) {

    extern __shared__ scalar sharedNorm[];

    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int id_C = i + nx * (j + ny * k);
        scalar phi_C = field[id_C];
        // east
        scalar phi_E = 0;
        if ( i != nx - 1 ) {
            int id_E = i+1 + nx * (j + ny * k);
            phi_E = field[id_E];
        }
        // west
        scalar phi_W = 0;
        if ( i != 0 ) {
            int id_W = i-1 + nx * (j + ny * k);
            phi_W = field[id_W];
        }
        // north
        scalar phi_N = 0;
        if ( j != ny - 1 ) {
            int id_N = i + nx * (j+1 + ny * k);
            phi_N = field[id_N];
        }
        // south
        scalar phi_S = 0;
        if ( j != 0 ) {
            int id_S = i + nx * (j-1 + ny * k);
            phi_S = field[id_S];
        }
        // top
        scalar phi_T = 0;
        if ( k != nz - 1 ) {
            int id_T = i + nx * (j + ny * (k+1));
            phi_T = field[id_T];
        }
        // bottom
        scalar phi_B = 0;
        if ( k != 0 ) {
            int id_B = i + nx * (j + ny * (k-1));
            phi_B = field[id_B];
        }
        
        scalar newPhi = srcTerm[id_C]
            - coef[i+nx*(j+ny*(k+aE))] * phi_E
            - coef[i+nx*(j+ny*(k+aW))] * phi_W
            - coef[i+nx*(j+ny*(k+aN))] * phi_N
            - coef[i+nx*(j+ny*(k+aS))] * phi_S;
        if (dim == 3) {
            newPhi -= coef[i+nx*(j+ny*(k+aT))] * phi_T + coef[i+nx*(j+ny*(k+aB))] * phi_B;
        }
        newPhi /= coef[i+nx*(j+ny*(k+aC))];

        scalar dPhi = relax * (newPhi - phi_C);

        field[id_C] = phi_C + dPhi;

        sharedNorm[tid] = dPhi*dPhi;
        __syncthreads();

        int stride = blockDim.x * blockDim.y * blockDim.z;
        for (int s = stride / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sharedNorm[tid] += sharedNorm[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            atomicAdd(norm, sharedNorm[0]);
        }
    }
}

void GaussSeidelIterate(scalar *field_dev, scalar *coef_dev, scalar *srcTerm_dev) {

    scalar *norm_dev;
    cudaMalloc(&norm_dev, sizeof(scalar));

    dim3 threadsPerBlock;
    dim3 numBlocks;
    if (dim == 2) {
        threadsPerBlock = dim3(32, 32, 1);

        numBlocks.x = (nx + 31) / 32;
        numBlocks.y = (ny + 31) / 32;
        numBlocks.z = 1;
    } else if (dim == 3) {
        threadsPerBlock = dim3(16, 16, 8);

        numBlocks.x = (nx + 15) / 16;
        numBlocks.y = (ny + 7) / 8;
        numBlocks.z = (nz + 7) / 8;
    }

    int blockSize = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z;

    scalar maxNorm = -1e20;

    for (int it = 0; it < numInnerIter; ++it) {

        scalar norm = 0.0;
        cudaMemcpy(norm_dev, &norm, sizeof(scalar), cudaMemcpyHostToDevice);

        GaussSeidelIterateKernel<<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize>>>(field_dev, coef_dev, srcTerm_dev, norm_dev);
        
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

__global__ void RhieChowInterpolateUfKernel(scalar *uf, scalar *u, scalar *uCoef, scalar *p, int typeW, int typeE) {

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int id_c  = i   + (nx+1) * (j + ny * k);
        int id_WW = i-2 + nx     * (j + ny * k);
        int id_W  = i-1 + nx     * (j + ny * k);
        int id_E  = i   + nx     * (j + ny * k);
        int id_EE = i+1 + nx     * (j + ny * k);

        scalar u_W = u[id_W];
        scalar u_E = u[id_E];

        scalar aC_W = uCoef[i-1 + nx * (j + ny * (k + nz * aC))];
        scalar aC_E = uCoef[i   + nx * (j + ny * (k + nz * aC))];

        scalar p_WW, p_W, p_E, p_EE;
        p_W = p[id_W];
        p_E = p[id_E];
        if (i == 1) {
            if (typeW == 0 || typeW == 1) { // "wall" or "inlet"
                p_WW = p_W;
            } else if (typeW == 2) { // "outlet"
                p_WW = 0;
            }
        } else {
            p_WW = p[id_WW];
        }
        if (i == nx - 1) {
            if (typeE == 0 || typeE == 1) { // "wall" or "inlet"
                p_EE = p_E;
            } else if (typeE == 2) { // "outlet"
                p_EE = 0;
            }
        } else {
            p_EE = p[id_EE];
        }

        uf[id_c] = 0.5 * (u_W + u_E) + (p_E - p_WW) * areaX / (4 * aC_W) + (p_EE - p_W) * areaX / (4 * aC_E)
                 + 0.5 * (1/aC_W + 1/aC_E) * (p_W - p_E) * areaX;
    }
}

__global__ void RhieChowInterpolateVfKernel(scalar *vf, scalar *v, scalar *vCoef, scalar *p, int typeS, int typeN) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int id_c  = i + nx * (j   + (ny+1) * k);
        int id_SS = i + nx * (j-2 + ny     * k);
        int id_S  = i + nx * (j-1 + ny     * k);
        int id_N  = i + nx * (j   + ny     * k);
        int id_NN = i + nx * (j+1 + ny     * k);

        scalar v_S = v[id_S];
        scalar v_N = v[id_N];

        scalar aC_S = vCoef[i + nx * (j-1 + ny * (k + nz * aC))];
        scalar aC_N = vCoef[i + nx * (j   + ny * (k + nz * aC))];

        scalar p_SS, p_S, p_N, p_NN;
        p_S = p[id_S];
        p_N = p[id_N];
        if (j == 1) {
            if (typeS == 0 || typeS == 1) { // "wall" or "inlet"
                p_SS = p_S;
            } else if (typeS == 2) { // "outlet"
                p_SS = 0;
            }
        } else {
            p_SS = p[id_SS];
        }
        if (j == ny - 1) {
            if (typeN == 0 || typeN == 1) { // "wall" or "inlet"
                p_NN = p_N;
            } else if (typeN == 2) { // "outlet"
                p_NN = 0;
            }
        } else {
            p_NN = p[id_NN];
        }

        vf[id_c] = 0.5 * (v_S + v_N) + (p_N - p_SS) * areaY / (4 * aC_S) + (p_NN - p_S) * areaY / (4 * aC_N)
                 + 0.5 * (1/aC_S + 1/aC_N) * (p_S - p_N) * areaY;
    }
}

__global__ void RhieChowInterpolateWfKernel(scalar *wf, scalar *w, scalar *wCoef, scalar *p, int typeB, int typeT) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i < nx && j < ny && k < nz) {
        int id_c  = i + nx * (j + ny * k  );
        int id_BB = i + nx * (j + ny * k-2);
        int id_B  = i + nx * (j + ny * k-1);
        int id_T  = i + nx * (j + ny * k  );
        int id_TT = i + nx * (j + ny * k+1);

        scalar w_B = w[id_B];
        scalar w_T = w[id_T];

        scalar aC_B = wCoef[i + nx * (j + ny * (k-1 + nz * aC))];
        scalar aC_T = wCoef[i + nx * (j + ny * (k   + nz * aC))];

        scalar p_BB, p_B, p_T, p_TT;
        p_B = p[id_B];
        p_T = p[id_T];
        if (k == 1) {
            if (typeB == 0 || typeB == 1) { // "wall" or "inlet"
                p_BB = p_B;
            } else if (typeB == 2) { // "outlet"
                p_BB = 0;
            }
        } else {
            p_BB = p[id_BB];
        }
        if (k == nz - 1) {
            if (typeT == 0 || typeT == 1) { // "wall" or "inlet"
                p_TT = p_T;
            } else if (typeT == 2) { // "outlet"
                p_TT = 0;
            }
        } else {
            p_TT = p[id_TT];
        }

        wf[id_c] = 0.5 * (w_B + w_T) + (p_T - p_BB) * areaZ / (4 * aC_B) + (p_TT - p_B) * areaZ / (4 * aC_T)
                 + 0.5 * (1/aC_B + 1/aC_T) * (p_B - p_T) * areaZ;
    }
}

void RhieChowInterpolate(scalar *uf_dev, scalar *vf_dev, scalar *wf_dev, scalar *u_dev, scalar *v_dev, scalar *w_dev
    , scalar *uCoef_dev, scalar *vCoef_dev, scalar *wCoef_dev, scalar *p_dev) {

    cudaStream_t stream[dim];
    for (int i = 0; i < dim; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    dim3 threadsPerBlock;
    dim3 numBlocks;

    if (dim == 2) {
        threadsPerBlock = dim3(32, 32, 1);
    } else if (dim == 3) {
        threadsPerBlock = dim3(16, 8, 8);
    }

    numBlocks.x = (nx - 1 + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;
    RhieChowInterpolateUfKernel<<<numBlocks, threadsPerBlock, 0, stream[0]>>>(uf_dev, u_dev, uCoef_dev, p_dev
        , velBCs::type[west], velBCs::type[east]);

    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny - 1 + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;
    RhieChowInterpolateVfKernel<<<numBlocks, threadsPerBlock, 0, stream[1]>>>(vf_dev, v_dev, vCoef_dev, p_dev
        , velBCs::type[south], velBCs::type[north]);

    if (dim == 3) {
        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = (nz - 1 + threadsPerBlock.z - 1) / threadsPerBlock.z;
        RhieChowInterpolateWfKernel<<<numBlocks, threadsPerBlock, 0, stream[2]>>>(wf_dev, w_dev, wCoef_dev, p_dev
            , velBCs::type[bottom], velBCs::type[top]);
    }

    for (int i = 0; i < dim; ++i) {
        cudaStreamSynchronize(stream[i]);
    }

    for (int i = 0; i < dim; ++i) {
        cudaStreamDestroy(stream[i]);
    }
}