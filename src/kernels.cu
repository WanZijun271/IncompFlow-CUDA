#include "kernels.cuh"
#include "config.h"
#include "constants.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdio>

using namespace std;

__global__ void initUfKernel(scalar *u, scalar *uf) {

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int id_c = i     + (nx+1) * (j + ny * k);
        int id_W = (i-1) + nx     * (j + ny * k);
        int id_E = i     + nx     * (j + ny * k);
        uf[id_c] = (u[id_W] + u[id_E]) / 2;
    }
}

__global__ void initVfKernel(scalar *v, scalar *vf) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int id_c = i + nx * (j   + (ny+1) * k);
        int id_S = i + nx * (j-1 + ny     * k);
        int id_N = i + nx * (j   + ny     * k);
        vf[id_c] = (v[id_S] + v[id_N]) / 2;
    }
}

__global__ void initWfKernel(scalar *w, scalar *wf) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i < nx && j < ny && k < nz) {
        int id_c = i + nx * (j + ny * k    );
        int id_B = i + nx * (j + ny * (k-1));
        int id_T = i + nx * (j + ny * k    );
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

        int id_aE = i + nx * (j + ny * (k + nz * aE));
        int id_aW = i + nx * (j + ny * (k + nz * aW));
        int id_aN = i + nx * (j + ny * (k + nz * aN));
        int id_aS = i + nx * (j + ny * (k + nz * aS));
        int id_aC = i + nx * (j + ny * (k + nz * aC));

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

__global__ void calcMomSrcTermKernel(scalar *srcTerm, scalar *p, int dir, int typeL, int typeR) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int id_C = i   + nx * (j + ny * k);
        int id_L, id_R;
        scalar area;
        if (dir == xDir) {
            id_L = i-1 + nx * (j + ny * k);
            id_R = i+1 + nx * (j + ny * k);
            area = areaX;
        } else if (dir == yDir) {
            id_L = i + nx * (j-1 + ny * k);
            id_R = i + nx * (j+1 + ny * k);
            area = areaY;
        } else if (dir == zDir) {
            id_L = i + nx * (j + ny * (k-1));
            id_R = i + nx * (j + ny * (k+1));
            area = areaZ;
        }

        scalar p_L, p_R;
        if ((dir == xDir && i == 0) || (dir == yDir && j == 0) || (dir == zDir && k == 0)) {
            if (typeL == 0 || typeL == 1) { // "wall" or "inlet"
                p_L = p[id_C];
            } else if (typeL == 2) { // "outlet"
                p_L = 0;
            }
            p_R = p[id_R];
        } else if ((dir == xDir && i == nx-1) || (dir == yDir && j == ny-1) || (dir == zDir && k == nz-1)) {
            if (typeR == 0 || typeR == 1) { // "wall" or "inlet"
                p_R = p[id_C];
            } else if (typeR == 2) { // "outlet"
                p_R = 0;
            }
            p_L = p[id_L];
        } else {
            p_L = p[id_L];
            p_R = p[id_R];
        }
        srcTerm[id_C] = 0.5 * (p_L - p_R) * area;
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

    calcMomSrcTermKernel<<<numBlocks, threadsPerBlock, 0, stream[0]>>>(uSrcTerm_dev, p_dev, xDir, velBCs::type[west]
        , velBCs::type[east]);

    calcMomSrcTermKernel<<<numBlocks, threadsPerBlock, 0, stream[1]>>>(vSrcTerm_dev, p_dev, yDir, velBCs::type[south]
        , velBCs::type[north]);

    if (dim == 3) {
        calcMomSrcTermKernel<<<numBlocks, threadsPerBlock, 0, stream[2]>>>(wSrcTerm_dev, p_dev, zDir, velBCs::type[bottom]
            , velBCs::type[top]);
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

        int id_aC, id_aIn, id_aOut, id_b;
        if (location == east) {
            id_aC   = nx-1 + nx * (i + ny * (j + nz * aC));
            id_aIn  = nx-1 + nx * (i + ny * (j + nz * aW));
            id_aOut = nx-1 + nx * (i + ny * (j + nz * aE));
            id_b    = nx-1 + nx * (i + ny * j);
        } else if (location == west) {
            id_aC   = 0 + nx * (i + ny * (j + nz * aC));
            id_aIn  = 0 + nx * (i + ny * (j + nz * aE));
            id_aOut = 0 + nx * (i + ny * (j + nz * aW));
            id_b    = 0 + nx * (i + ny * j);
        } else if (location == north) {
            id_aC   = i + nx * (ny-1 + ny * (j + nz * aC));
            id_aIn  = i + nx * (ny-1 + ny * (j + nz * aS));
            id_aOut = i + nx * (ny-1 + ny * (j + nz * aN));
            id_b    = i + nx * (ny-1 + ny * j);
        } else if (location == south) {
            id_aC   = i + nx * (0 + ny * (j + nz * aC));
            id_aIn  = i + nx * (0 + ny * (j + nz * aN));
            id_aOut = i + nx * (0 + ny * (j + nz * aS));
            id_b    = i + nx * (0 + ny * j);
        } else if (location == top) {
            id_aC   = i + nx * (j + ny * (nz-1 + nz * aC));
            id_aIn  = i + nx * (i + ny * (nz-1 + nz * aB));
            id_aOut = i + nx * (j + ny * (nz-1 + nz * aT));
            id_b    = i + nx * (j + ny * (nz-1));
        } else if (location == bottom) {
            id_aC   = i + nx * (j + ny * (0 + nz * aC));
            id_aIn  = i + nx * (i + ny * (0 + nz * aT));
            id_aOut = i + nx * (j + ny * (0 + nz * aB));
            id_b    = i + nx * (j + ny * 0);
        }

        
        if (type == 0 || type == 1) { // "wall" or "inlet"
            uCoef[id_aC] -= 2 * uCoef[id_aOut];
            uCoef[id_aIn] += uCoef[id_aOut] / 3;
            uSrcTerm[id_b] -= 8/3 * uCoef[id_aOut] * uBC;
            uCoef[id_aOut] = 0;

            vCoef[id_aC] -= 2 * vCoef[id_aOut];
            vCoef[id_aIn] += vCoef[id_aOut] / 3;
            vSrcTerm[id_b] -= 8/3 * vCoef[id_aOut] * vBC;
            vCoef[id_aOut] = 0;

            if (dim == 3) {
                wCoef[id_aC] -= 2 * wCoef[id_aOut];
                wCoef[id_aIn] += wCoef[id_aOut] / 3;
                wSrcTerm[id_b] -= 8/3 * wCoef[id_aOut] * wBC;
                wCoef[id_aOut] = 0;
            }
        } else if (type == 2) { // "outlet"
            uCoef[id_aC] += uCoef[id_aOut];
            uCoef[id_aOut] = 0;

            vCoef[id_aC] += vCoef[id_aOut];
            vCoef[id_aOut] = 0;

            if (dim == 3) {
                wCoef[id_aC] += wCoef[id_aOut];
                wCoef[id_aOut] = 0;
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

__global__ void pointJacobiIterateKernel(scalar *field, scalar* field0, scalar *coef, scalar *srcTerm, scalar *norm, scalar relax) {

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
            - coef[i+nx*(j+ny*(k+nz*aE))] * phi_E
            - coef[i+nx*(j+ny*(k+nz*aW))] * phi_W
            - coef[i+nx*(j+ny*(k+nz*aN))] * phi_N
            - coef[i+nx*(j+ny*(k+nz*aS))] * phi_S;
        if (dim == 3) {
            newPhi -= coef[i+nx*(j+ny*(k+nz*aT))] * phi_T + coef[i+nx*(j+ny*(k+aB))] * phi_B;
        }
        newPhi /= coef[i+nx*(j+ny*(k+nz*aC))];

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

void pointJacobiIterate(scalar *field_dev, size_t fieldSize, scalar *coef_dev, scalar *srcTerm_dev, int nIter, scalar relax
    , scalar tol) {

    scalar *field0_dev, *norm_dev;
    cudaMalloc(&field0_dev, fieldSize);
    cudaMalloc(&norm_dev, sizeof(scalar));

    cudaMemset(field0_dev, 0, fieldSize);

    dim3 threadsPerBlock;
    dim3 numBlocks;

    if (dim == 2) {
        threadsPerBlock = dim3(32, 32, 1);
    } else if (dim == 3) {
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

__global__ void GaussSeidelIterateKernel(scalar *field, scalar *coef, scalar *srcTerm, scalar *norm, scalar relax) {

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

void GaussSeidelIterate(scalar *field_dev, scalar *coef_dev, scalar *srcTerm_dev, int nIter, scalar relax, scalar tol) {

    scalar *norm_dev;
    cudaMalloc(&norm_dev, sizeof(scalar));

    dim3 threadsPerBlock;
    dim3 numBlocks;

    if (dim == 2) {
        threadsPerBlock = dim3(32, 32, 1);
    } else if (dim == 3) {
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

__global__ void calcPresCorrLinkCoefKernel(scalar *pCorrCoef, scalar *uCoef, scalar *vCoef, scalar *wCoef) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int id_aC = i+nx*(j+ny*(k+nz*aC));
        int id_aE = i+nx*(j+ny*(k+nz*aE));
        int id_aW = i+nx*(j+ny*(k+nz*aW));
        int id_aN = i+nx*(j+ny*(k+nz*aN));
        int id_aS = i+nx*(j+ny*(k+nz*aS));
        int id_aT = i+nx*(j+ny*(k+nz*aT));
        int id_aB = i+nx*(j+ny*(k+nz*aB));

        if (i < nx - 1) {
            pCorrCoef[id_aE] = -0.5 * density * areaX * areaX * (1/uCoef[i+nx*(j+ny*(k+nz*aC))] + 1/uCoef[i+1+nx*(j+ny*(k+nz*aC))]);
        }
        if (i > 0) {
            pCorrCoef[id_aW] = -0.5 * density * areaX * areaX * (1/uCoef[i+nx*(j+ny*(k+nz*aC))] + 1/uCoef[i-1+nx*(j+ny*(k+nz*aC))]);
        }
        if (j < ny -1) {
            pCorrCoef[id_aN] = -0.5 * density * areaY * areaY * (1/vCoef[i+nx*(j+ny*(k+nz*aC))] + 1/vCoef[i+nx*(j+1+ny*(k+nz*aC))]);
        }
        if (j > 0) {
            pCorrCoef[id_aS] = -0.5 * density * areaY * areaY * (1/vCoef[i+nx*(j+ny*(k+nz*aC))] + 1/vCoef[i+nx*(j-1+ny*(k+nz*aC))]);
        }
        pCorrCoef[id_aC] = -(pCorrCoef[id_aE] + pCorrCoef[id_aW] + pCorrCoef[id_aN] + pCorrCoef[id_aS]);
        if (dim == 3) {
            if (k < nz - 1) {
                pCorrCoef[i+nx*(j+ny*(k+aT))] = -0.5 * density * areaZ * areaZ * (1/wCoef[i+nx*(j+ny*(k+nz*aC))] + 1/wCoef[i+nx*(j+ny*(k+1+nz*aC))]);
            }
            if (k > 0) {
                pCorrCoef[i+nx*(j+ny*(k+aB))] = -0.5 * density * areaZ * areaZ * (1/wCoef[i+nx*(j+ny*(k+nz*aC))] + 1/wCoef[i+nx*(j+ny*(k-1+nz*aC))]);
            }
            pCorrCoef[id_aC] += -(pCorrCoef[id_aT] + pCorrCoef[id_aB]);
        }
    }
}

void calcPresCorrLinkCoef(scalar *pCorrCoef_dev, scalar *uCoef_dev, scalar *vCoef_dev, scalar *wCoef_dev) {

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

    calcPresCorrLinkCoefKernel<<<numBlocks, threadsPerBlock>>>(pCorrCoef_dev, uCoef_dev, vCoef_dev, wCoef_dev);

    cudaDeviceSynchronize();
}

__global__ void calcPresCorrSrcTermKernel(scalar *pCorrSrcTerm, scalar *uf, scalar *vf, scalar *wf) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int id_C = i   + nx     * (j   + ny     * k);
        int id_e = i+1 + (nx+1) * (j   + ny     * k);
        int id_w = i   + (nx+1) * (j   + ny     * k);
        int id_n = i   + nx     * (j+1 + (ny+1) * k);
        int id_s = i   + nx     * (j   + (ny+1) * k);

        pCorrSrcTerm[id_C] = density * (areaX * (uf[id_w] - uf[id_e]) + areaY * (vf[id_s] - vf[id_n]));
        if (dim == 3) {
            int id_t = i + nx * (j + ny * (k+1));
            int id_b = i + nx * (j + ny * k    );

            pCorrSrcTerm[id_C] += density * areaZ * (wf[id_b] - wf[id_t]);
        }
    }
}

void calcPresCorrSrcTerm(scalar *pCorrSrcTerm_dev, scalar *uf_dev, scalar *vf_dev, scalar *wf_dev) {

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

    calcPresCorrSrcTermKernel<<<numBlocks, threadsPerBlock>>>(pCorrSrcTerm_dev, uf_dev, vf_dev, wf_dev);

    cudaDeviceSynchronize();
}

__global__ void updateVelKernel(scalar *vel, scalar *coef, scalar *pCorr, scalar *norm, int dir, int typeOnMin, int typeOnMax
    , scalar relax) {

    extern __shared__ scalar sharedNorm[];

    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int id_C = i + nx * (j + ny * k);
        int id_Prev, id_Next;
        scalar area;
        if (dir == xDir) { // x-direction
            id_Prev = i-1 + nx * (j + ny * k);
            id_Next = i+1 + nx * (j + ny * k);
            area = areaX;
        } else if (dir == yDir) { // y-direction
            id_Prev = i + nx * (j-1 + ny * k);
            id_Next = i + nx * (j+1 + ny * k);
            area = areaY;
        } else if (dir == zDir) { // z-direction
            id_Prev = i + nx * (j + ny * (k-1));
            id_Next = i + nx * (j + ny * (k+1));
            area = areaZ;
        }

        scalar pCorr_Prev;
        scalar pCorr_Next;
        scalar aC_C = coef[i+nx*(j+ny*(k+aC))];

        if ((dir == xDir && i == 0) || (dir == yDir && j == 0) || (dir == zDir && k == 0)) {
            if (typeOnMin == 0 || typeOnMin == 1) { // "wall" or "inlet"
                pCorr_Prev = pCorr[id_C];
            } else if (typeOnMin == 2) {
                pCorr_Prev = 0;
            }
            pCorr_Next = pCorr[id_Next];
        } else if ((dir == xDir && i == nx-1) || (dir == yDir && j == ny-1) || (dir == zDir && k == nz-1)) {
            if (typeOnMax == 0 || typeOnMax == 1) { // "wall" or "inlet"
                pCorr_Next = pCorr[id_C];
            } else if (typeOnMax == 2) {
                pCorr_Next = 0;
            }
            pCorr_Prev = pCorr[id_Prev];
        } else {
            pCorr_Prev = pCorr[id_Prev];
            pCorr_Next = pCorr[id_Next];
        }

        scalar dvel = relax * 0.5 * (pCorr_Prev - pCorr_Next) * area / aC_C;

        vel[id_C] += dvel;

        sharedNorm[tid] = dvel*dvel;
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

__global__ void updateFaceVelKernel(scalar *vel, scalar *coef, scalar *pCorr, scalar *norm, int dir, scalar relax) {

    extern __shared__ scalar sharedNorm[];

    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if ((dir == xDir && i < nx+1 && j < ny   && k < nz) ||
        (dir == yDir && i < nx   && j < ny+1 && k < nz) ||
        (dir == zDir && i < nx   && j < ny   && k < nz+1)) {

        int id_c, id_Prev, id_Next;
        scalar area;
        if (dir == xDir) { // x-direction
            id_c    = i   + (nx+1) * (j + ny * k);
            id_Prev = i-1 + nx     * (j + ny * k);
            id_Next = i   + nx     * (j + ny * k);
            area = areaX;
        } else if (dir == yDir) { // y-direction
            id_c    = i + nx * (j   + (ny+1) * k);
            id_Prev = i + nx * (j-1 + ny     * k);
            id_Next = i + nx * (j   + ny     * k);
            area = areaY;
        } else if (dir == zDir) { // z-direction
            id_c    = i + nx * (j + ny * k    );
            id_Prev = i + nx * (j + ny * (k-1));
            id_Next = i + nx * (j + ny * k    );
            area = areaZ;
        }

        scalar dvel;

        if ((dir == xDir && (i == 0 || i == nx)) || (dir == yDir && (j == 0 || j == ny)) || (dir == zDir && (k == 0 || k == nz))) {
            dvel = 0;
        } else {
            scalar aC_Prev = coef[id_Prev+nx*ny*nz*aC];
            scalar aC_Next = coef[id_Next+nx*ny*nz*aC];
            dvel = relax * 0.5 * (1/aC_Prev + 1/aC_Next) * (pCorr[id_Prev] - pCorr[id_Next]) * area;
        }

        vel[id_c] += dvel;

        sharedNorm[tid] = dvel*dvel;
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

__global__ void updatePresKernel(scalar *p, scalar *pCorr, scalar *norm, scalar relax) {

    extern __shared__ scalar sharedNorm[];

    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int id = i + nx * (j + ny * k);
        scalar dp = relax * pCorr[id];
        p[id] += dp;

        sharedNorm[tid] = dp*dp;
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

void updateField(scalar *u_dev, scalar *v_dev, scalar *w_dev, scalar *uNorm_dev, scalar *vNorm_dev, scalar *wNorm_dev
    , scalar *uf_dev, scalar *vf_dev, scalar *wf_dev, scalar *ufNorm_dev, scalar *vfNorm_dev, scalar *wfNorm_dev
    , scalar *p_dev, scalar *pNorm_dev, scalar *uCoef_dev, scalar *vCoef_dev, scalar *wCoef_dev, scalar *pCorr_dev) {

    cudaStream_t stream[2*dim+1];
    for (int i = 0; i < 2*dim+1; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    cudaMemsetAsync(pNorm_dev, 0, sizeof(scalar), stream[0]);
    cudaMemsetAsync(uNorm_dev, 0, sizeof(scalar), stream[1]);
    cudaMemsetAsync(vNorm_dev, 0, sizeof(scalar), stream[2]);
    cudaMemsetAsync(ufNorm_dev, 0, sizeof(scalar), stream[3]);
    cudaMemsetAsync(vfNorm_dev, 0, sizeof(scalar), stream[4]);
    if (dim == 3) {
        cudaMemsetAsync(wNorm_dev, 0, sizeof(scalar), stream[5]);
        cudaMemsetAsync(wfNorm_dev, 0, sizeof(scalar), stream[6]);
    }

    dim3 threadsPerBlock;
    dim3 numBlocks; 

    if (dim == 2) {
        threadsPerBlock = dim3(32, 32, 1);
    } else if (dim == 3) {
        threadsPerBlock = dim3(16, 8, 8);
    }

    int blockSize = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z;

    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;

    updatePresKernel<<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[0]>>>(p_dev, pCorr_dev, pNorm_dev, relax_p);

    updateVelKernel<<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[1]>>>(u_dev, uCoef_dev, pCorr_dev, uNorm_dev
        , xDir, velBCs::type[west], velBCs::type[east], relax_u);

    updateVelKernel<<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[2]>>>(v_dev, vCoef_dev, pCorr_dev, vNorm_dev
        , yDir, velBCs::type[south], velBCs::type[north], relax_v);

    numBlocks.x = (nx+1 + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;

    updateFaceVelKernel<<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[3]>>>(uf_dev, uCoef_dev, pCorr_dev
        , ufNorm_dev, xDir, relax_u);

    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny+1 + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;

    updateFaceVelKernel<<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[4]>>>(vf_dev, vCoef_dev, pCorr_dev
        , vfNorm_dev, yDir, relax_v);

    if (dim == 3) {
        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;

        updateVelKernel<<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[5]>>>(w_dev, wCoef_dev, pCorr_dev, wNorm_dev
            , zDir, velBCs::type[bottom], velBCs::type[top], relax_w);

        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = (nz+1 + threadsPerBlock.z - 1) / threadsPerBlock.z;

        updateFaceVelKernel<<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[6]>>>(wf_dev, wCoef_dev, pCorr_dev
            , wfNorm_dev, zDir, relax_w);
    }

    for (int i = 0; i < 2*dim+1; ++i) {
        cudaStreamSynchronize(stream[i]);
    }

    for (int i = 0; i < 2*dim+1; ++i) {
        cudaStreamDestroy(stream[i]);
    }

}