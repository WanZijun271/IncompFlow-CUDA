#include "kernels.cuh"
#include "config.h"
#include "constants.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdio>

using namespace std;

template<int dir>
__global__ void initFaceVelKernel(scalar *u, scalar *uf) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if constexpr (dir == xDir) {
        ++i;
    } else if constexpr (dir == yDir) {
        ++j;
    } else if constexpr (dir == zDir) {
        ++k;
    }

    if (i < nx && j < ny && k < nz) {

        int cId, LId, RId;

        if constexpr (dir == xDir) {
            cId = i     + (nx+1) * (j + ny * k);
            LId = (i-1) + nx     * (j + ny * k);
            RId = i     + nx     * (j + ny * k);
        } else if constexpr (dir == yDir) {
            cId = i + nx * (j   + (ny+1) * k);
            LId = i + nx * (j-1 + ny     * k);
            RId = i + nx * (j   + ny     * k);
        } else if constexpr (dir == zDir) {
            cId = i + nx * (j + ny * k    );
            LId = i + nx * (j + ny * (k-1));
            RId = i + nx * (j + ny * k    );
        }

        uf[cId] = (u[LId] + u[RId]) / 2;
    }
}

void initFaceVel(scalar *u_dev, scalar *v_dev, scalar *w_dev, scalar *uf_dev, scalar *vf_dev, scalar *wf_dev) {

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
    initFaceVelKernel<xDir><<<numBlocks, threadsPerBlock, 0, stream[0]>>>(u_dev, uf_dev);

    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny - 1 + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;
    initFaceVelKernel<yDir><<<numBlocks, threadsPerBlock, 0, stream[1]>>>(v_dev, vf_dev);

    if constexpr (dim == 3) {
        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = (nz - 1 + threadsPerBlock.z - 1) / threadsPerBlock.z;
        initFaceVelKernel<zDir><<<numBlocks, threadsPerBlock, 0, stream[2]>>>(w_dev, wf_dev);
    }

    for (int i = 0; i < dim; ++i) {
        cudaStreamSynchronize(stream[i]);
    }

    for (int i = 0; i < dim; ++i) {
        cudaStreamDestroy(stream[i]);
    }
}

template<int loc, int bcType>
__global__ void applyBCsToFaceVelKernel(scalar *uf, scalar *u, scalar bcVal) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    bool loopFlag;

    if constexpr (loc == west || loc == east) {
        loopFlag = (i < ny && j < nz);
    } else if constexpr (loc == south || loc == north) {
        loopFlag = (i < nx && j < nz);
    }  else if constexpr (loc == bottom || loc == top) {
        loopFlag = (i < nx && j < ny);
    }

    if (loopFlag) {

        int cId, NbId;

        if constexpr (loc == west) {
            cId  = 0 + (nx+1) * (i + ny * j);
            NbId = 0 + nx     * (i + ny * j);
        } else if constexpr (loc == east) {
            cId  = nx   + (nx+1) * (i + ny * j);
            NbId = nx-1 + nx     * (i + ny * j);
        } else if constexpr (loc == south) {
            cId  = i + nx * (0 + (ny+1) * j);
            NbId = i + nx * (0 + ny     * j);
        } else if constexpr (loc == north) {
            cId  = i + nx * (ny   + (ny+1) * j);
            NbId = i + nx * (ny-1 + ny     * j);
        } else if constexpr (loc == bottom) {
            cId  = i + nx * (j + ny * 0);
            NbId = i + nx * (j + ny * 0);
        } else if constexpr (loc == top) {
            cId  = i + nx * (j + ny * nz    );
            NbId = i + nx * (j + ny * (nz-1));
        }

        if constexpr (bcType == wall) {
            uf[cId] = 0;
        } else if constexpr (bcType == inlet) {
            uf[cId] = bcVal;
        } else if constexpr (bcType == outlet) {
            uf[cId] = u[NbId];
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

    if constexpr (dim == 2) {
        threadsPerBlock = dim3(1024, 1, 1);
    } else if constexpr (dim == 3) {
        threadsPerBlock = dim3(32, 32, 1);
    }

    numBlocks.x = (ny + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (nz + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;

    applyBCsToFaceVelKernel<west, velBCs::type[west]><<<numBlocks, threadsPerBlock, 0, stream[0]>>>(uf_dev, u_dev
        , velBCs::val[west][0]);

    applyBCsToFaceVelKernel<east, velBCs::type[east]><<<numBlocks, threadsPerBlock, 0, stream[1]>>>(uf_dev, u_dev
        , velBCs::val[east][0]);

    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (nz + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;

    applyBCsToFaceVelKernel<south, velBCs::type[south]><<<numBlocks, threadsPerBlock, 0, stream[2]>>>(vf_dev, v_dev
        , velBCs::val[south][1]);

    applyBCsToFaceVelKernel<north, velBCs::type[north]><<<numBlocks, threadsPerBlock, 0, stream[3]>>>(vf_dev, v_dev
        , velBCs::val[north][1]);

    if constexpr (dim == 3) {
        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = 1;

        applyBCsToFaceVelKernel<bottom, velBCs::type[bottom]><<<numBlocks, threadsPerBlock, 0, stream[4]>>>(wf_dev, w_dev
            , velBCs::val[bottom][2]);

        applyBCsToFaceVelKernel<top, velBCs::type[top]><<<numBlocks, threadsPerBlock, 0, stream[5]>>>(wf_dev, w_dev
            , velBCs::val[top][2]);
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

        int aEId = i + nx * (j + ny * (k + nz * aE));
        int aWId = i + nx * (j + ny * (k + nz * aW));
        int aNId = i + nx * (j + ny * (k + nz * aN));
        int aSId = i + nx * (j + ny * (k + nz * aS));
        int aCId = i + nx * (j + ny * (k + nz * aC));

        coef[aEId] = (density * min(ue , (scalar)0) * areaX) - dynamicViscosity * areaX / dx;
        coef[aWId] = (density * min(-uw, (scalar)0) * areaX) - dynamicViscosity * areaX / dx;
        coef[aNId] = (density * min(vn , (scalar)0) * areaY) - dynamicViscosity * areaY / dy;
        coef[aSId] = (density * min(-vs, (scalar)0) * areaY) - dynamicViscosity * areaY / dy;
        coef[aCId] = -(coef[aEId] + coef[aWId] + coef[aNId] + coef[aSId]);

        if constexpr (dim == 3) {
            scalar wt = wf[k+1 + (nz+1) * (i + nx * j)];
            scalar wb = wf[k   + (nz+1) * (i + nx * j)];

            int aTId = i + nx * (j + ny * (k + aT));
            int aBId = i + nx * (j + ny * (k + aB));

            coef[aTId] = (density * min(wt , (scalar)0) * areaZ) - dynamicViscosity * areaZ / dz;
            coef[aBId] = (density * min(-wb, (scalar)0) * areaZ) - dynamicViscosity * areaZ / dz;
            coef[aCId] += -(coef[aTId] + coef[aBId]);
        }
    }
}

void calcMomLinkCoef(scalar *coef_dev, scalar *uf_dev, scalar *vf_dev, scalar *wf_dev) {

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

    calcMomLinkCoefKernel<<<numBlocks, threadsPerBlock>>>(coef_dev, uf_dev, vf_dev, wf_dev);

    cudaDeviceSynchronize();
}

template<int dir>
__global__ void calcInteriorMomSrcTermKernel(scalar *srcTerm, scalar *p) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    bool loopFlag;

    if constexpr (dir == xDir) {
        ++i;
        loopFlag = (i < nx-1 && j < ny && k < nz);
    } else if constexpr (dir == yDir) {
        ++j;
        loopFlag = (i < nx && j < ny-1 && k < nz);
    } else if constexpr (dir == zDir) {
        ++k;
        loopFlag = (i < nx && j < ny && k < nz-1);
    }

    if (loopFlag) {

        int CId = i + nx * (j + ny * k);
        int LId, RId;
        scalar area;

        if constexpr (dir == xDir) {
            LId = i-1 + nx * (j + ny * k);
            RId = i+1 + nx * (j + ny * k);
            area = areaX;
        } else if constexpr (dir == yDir) {
            LId = i + nx * (j-1 + ny * k);
            RId = i + nx * (j+1 + ny * k);
            area = areaY;
        } else if constexpr (dir == zDir) {
            LId = i + nx * (j + ny * (k-1));
            RId = i + nx * (j + ny * (k+1));
            area = areaZ;
        }
        
        srcTerm[CId] = 0.5 * (p[LId] - p[RId]) * area;
    }
}

template<int loc, int bcType>
__global__ void calcBCMomSrcTermKernel(scalar *srcTerm, scalar *p) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    bool loopFlag;

    if constexpr (loc == west || loc == east) {
        loopFlag = (i < ny && j < nz);
    } else if constexpr (loc == south || loc == north) {
        loopFlag = (i < nx && j < nz);
    }  else if constexpr (loc == bottom || loc == top) {
        loopFlag = (i < nx && j < ny);
    }

    if (loopFlag) {

        int CId, LId, RId;

        if constexpr (loc == west) {
            CId = 0 + nx * (i + ny * j);
            RId = 1 + nx * (i + ny * j);
        } else if constexpr (loc == east) {
            CId = nx-1 + nx * (i + ny * j);
            LId = nx-2 + nx * (i + ny * j);
        } else if constexpr (loc == south) {
            CId = i + nx * (0 + ny * j);
            RId = i + nx * (1 + ny * j);
        } else if constexpr (loc == north) {
            CId = i + nx * (ny-1 + ny * j);
            LId = i + nx * (ny-2 + ny * j);
        } else if constexpr (loc == bottom) {
            CId = i + nx * (j + ny * 0);
            RId = i + nx * (j + ny * 1);
        } else if constexpr (loc == top) {
            CId = i + nx * (j + ny * (nz-1));
            LId = i + nx * (j + ny * (nz-2));
        }

        scalar area;

        if constexpr (loc == west || loc == east) {
            area = areaX;
        } else if constexpr (loc == south || loc == north) {
            area = areaY;
        } else if constexpr (loc == bottom || loc == top) {
            area = areaZ;
        }

        scalar p_L, p_R;

        if constexpr (loc == west || loc == south || loc == bottom) {
            if constexpr (bcType == wall || bcType == inlet) {
                p_L = p[CId];
            } else if constexpr (bcType == outlet) {
                p_L = 0;
            }
            p_R = p[RId];
        } else if constexpr (loc == east || loc == north || loc == top) {
            if constexpr (bcType == wall || bcType == inlet) {
                p_R = p[CId];
            } else if constexpr (bcType == outlet) {
                p_R = 0;
            }
            p_L = p[LId];
        }

        srcTerm[CId] = 0.5 * (p_L - p_R) * area;
    }
}

void calcMomSrcTerm(scalar *uSrcTerm_dev, scalar *vSrcTerm_dev, scalar *wSrcTerm_dev, scalar *p_dev) {

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

    calcInteriorMomSrcTermKernel<xDir><<<numBlocks, threadsPerBlock, 0, stream[0]>>>(uSrcTerm_dev, p_dev);

    numBlocks.x = (nx   + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny-2 + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz   + threadsPerBlock.z - 1) / threadsPerBlock.z;

    calcInteriorMomSrcTermKernel<yDir><<<numBlocks, threadsPerBlock, 0, stream[1]>>>(vSrcTerm_dev, p_dev);

    if constexpr (dim == 3) {
        numBlocks.x = (nx   + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny   + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = (nz-2 + threadsPerBlock.z - 1) / threadsPerBlock.z;

        calcInteriorMomSrcTermKernel<zDir><<<numBlocks, threadsPerBlock, 0, stream[6]>>>(wSrcTerm_dev, p_dev);
    }

    if constexpr (dim == 2) {
        threadsPerBlock = dim3(1024, 1, 1);
    } else if constexpr (dim == 3) {
        threadsPerBlock = dim3(32, 32, 1);
    }

    numBlocks.x = (ny + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (nz + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;

    calcBCMomSrcTermKernel<west, velBCs::type[west]><<<numBlocks, threadsPerBlock, 0, stream[2]>>>(uSrcTerm_dev, p_dev);

    calcBCMomSrcTermKernel<east, velBCs::type[east]><<<numBlocks, threadsPerBlock, 0, stream[3]>>>(uSrcTerm_dev, p_dev);

    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (nz + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;

    calcBCMomSrcTermKernel<south, velBCs::type[south]><<<numBlocks, threadsPerBlock, 0, stream[4]>>>(vSrcTerm_dev, p_dev);

    calcBCMomSrcTermKernel<north, velBCs::type[north]><<<numBlocks, threadsPerBlock, 0, stream[5]>>>(vSrcTerm_dev, p_dev);

    if constexpr (dim == 3) {
        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = 1;

        calcBCMomSrcTermKernel<bottom, velBCs::type[bottom]><<<numBlocks, threadsPerBlock, 0, stream[7]>>>(wSrcTerm_dev, p_dev);

        calcBCMomSrcTermKernel<top, velBCs::type[top]><<<numBlocks, threadsPerBlock, 0, stream[8]>>>(wSrcTerm_dev, p_dev);
    }

    for (int i = 0; i < 3*dim; ++i) {
        cudaStreamSynchronize(stream[i]);
    }

    for (int i = 0; i < 3*dim; ++i) {
        cudaStreamDestroy(stream[i]);
    }
}

template<int loc, int bcType>
__global__ void applyBCsToMomEqKernel(scalar *uCoef, scalar *vCoef, scalar *wCoef, scalar *uSrcTerm, scalar *vSrcTerm
    , scalar *wSrcTerm, scalar uBC, scalar vBC, scalar wBC) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    bool loopFlag;

    if constexpr (loc == east || loc == west) {
        loopFlag = (i < ny && j < nz);
    } else if constexpr (loc == north || loc == south) {
        loopFlag = (i < nx && j < nz);
    } else if constexpr (loc == top || loc == bottom) {
        loopFlag = (i < nx && j < ny);
    }

    if (loopFlag) {

        int aCId, aInId, aExId, CId;

        if constexpr (loc == east) {
            aCId  = nx-1 + nx * (i + ny * (j + nz * aC));
            aInId = nx-1 + nx * (i + ny * (j + nz * aW));
            aExId = nx-1 + nx * (i + ny * (j + nz * aE));
            CId   = nx-1 + nx * (i + ny * j);
        } else if constexpr (loc == west) {
            aCId  = 0 + nx * (i + ny * (j + nz * aC));
            aInId = 0 + nx * (i + ny * (j + nz * aE));
            aExId = 0 + nx * (i + ny * (j + nz * aW));
            CId   = 0 + nx * (i + ny * j);
        } else if constexpr (loc == north) {
            aCId  = i + nx * (ny-1 + ny * (j + nz * aC));
            aInId = i + nx * (ny-1 + ny * (j + nz * aS));
            aExId = i + nx * (ny-1 + ny * (j + nz * aN));
            CId   = i + nx * (ny-1 + ny * j);
        } else if constexpr (loc == south) {
            aCId  = i + nx * (0 + ny * (j + nz * aC));
            aInId = i + nx * (0 + ny * (j + nz * aN));
            aExId = i + nx * (0 + ny * (j + nz * aS));
            CId   = i + nx * (0 + ny * j);
        } else if constexpr (loc == top) {
            aCId  = i + nx * (j + ny * (nz-1 + nz * aC));
            aInId = i + nx * (i + ny * (nz-1 + nz * aB));
            aExId = i + nx * (j + ny * (nz-1 + nz * aT));
            CId   = i + nx * (j + ny * (nz-1));
        } else if constexpr (loc == bottom) {
            aCId  = i + nx * (j + ny * (0 + nz * aC));
            aInId = i + nx * (i + ny * (0 + nz * aT));
            aExId = i + nx * (j + ny * (0 + nz * aB));
            CId   = i + nx * (j + ny * 0);
        }
        
        if constexpr (bcType == wall || bcType == inlet) { // "wall" or "inlet"
            uCoef[aCId] -= 2 * uCoef[aExId];
            uCoef[aInId] += uCoef[aExId] / 3;
            uSrcTerm[CId] -= (scalar)8/3 * uCoef[aExId] * uBC;
            uCoef[aExId] = 0;

            vCoef[aCId] -= 2 * vCoef[aExId];
            vCoef[aInId] += vCoef[aExId] / 3;
            vSrcTerm[CId] -= (scalar)8/3 * vCoef[aExId] * vBC;
            vCoef[aExId] = 0;

            if constexpr (dim == 3) {
                wCoef[aCId] -= 2 * wCoef[aExId];
                wCoef[aInId] += wCoef[aExId] / 3;
                wSrcTerm[CId] -= (scalar)8/3 * wCoef[aExId] * wBC;
                wCoef[aExId] = 0;
            }
        } else if constexpr (bcType == 2) { // "outlet"
            uCoef[aCId] += uCoef[aExId];
            uCoef[aExId] = 0;

            vCoef[aCId] += vCoef[aExId];
            vCoef[aExId] = 0;

            if constexpr (dim == 3) {
                wCoef[aCId] += wCoef[aExId];
                wCoef[aExId] = 0;
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

    if constexpr (dim == 2) {
        threadsPerBlock = dim3(1024, 1, 1);
    } else if constexpr (dim == 3) {
        threadsPerBlock = dim3(32, 32, 1);
    }

    numBlocks.x = (ny + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (nz + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;

    applyBCsToMomEqKernel<east, velBCs::type[east]><<<numBlocks, threadsPerBlock, 0, stream[0]>>>(uCoef_dev, vCoef_dev, wCoef_dev
        , uSrcTerm_dev, vSrcTerm_dev, wSrcTerm_dev, velBCs::val[east][0], velBCs::val[east][1], velBCs::val[east][2]);

    applyBCsToMomEqKernel<west, velBCs::type[west]><<<numBlocks, threadsPerBlock, 0, stream[1]>>>(uCoef_dev, vCoef_dev, wCoef_dev
        , uSrcTerm_dev, vSrcTerm_dev, wSrcTerm_dev, velBCs::val[west][0], velBCs::val[west][1], velBCs::val[west][2]);

    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (nz + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;

    applyBCsToMomEqKernel<north, velBCs::type[north]><<<numBlocks, threadsPerBlock, 0, stream[2]>>>(uCoef_dev, vCoef_dev, wCoef_dev
        , uSrcTerm_dev, vSrcTerm_dev, wSrcTerm_dev, velBCs::val[north][0], velBCs::val[north][1], velBCs::val[north][2]);

    applyBCsToMomEqKernel<south, velBCs::type[south]><<<numBlocks, threadsPerBlock, 0, stream[3]>>>(uCoef_dev, vCoef_dev, wCoef_dev
        , uSrcTerm_dev, vSrcTerm_dev, wSrcTerm_dev, velBCs::val[south][0], velBCs::val[south][1], velBCs::val[south][2]);

    if constexpr (dim == 3) {
        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = 1;

        applyBCsToMomEqKernel<top, velBCs::type[top]><<<numBlocks, threadsPerBlock, 0, stream[4]>>>(uCoef_dev, vCoef_dev, wCoef_dev
            , uSrcTerm_dev, vSrcTerm_dev, wSrcTerm_dev, velBCs::val[top][0], velBCs::val[top][1], velBCs::val[top][2]);

        applyBCsToMomEqKernel<bottom, velBCs::type[bottom]><<<numBlocks, threadsPerBlock, 0, stream[5]>>>(uCoef_dev, vCoef_dev, wCoef_dev
            , uSrcTerm_dev, vSrcTerm_dev, wSrcTerm_dev, velBCs::val[bottom][0], velBCs::val[bottom][1], velBCs::val[bottom][2]);
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

        int CId = i + nx * (j + ny * k);
        scalar phi_C = field0[CId];
        // east
        scalar phi_E = 0;
        if ( i != nx - 1 ) {
            int EId = i+1 + nx * (j + ny * k);
            phi_E = field0[EId];
        }
        // west
        scalar phi_W = 0;
        if ( i != 0 ) {
            int WId = i-1 + nx * (j + ny * k);
            phi_W = field0[WId];
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
        
        scalar newPhi = srcTerm[CId]
            - coef[i+nx*(j+ny*(k+nz*aE))] * phi_E
            - coef[i+nx*(j+ny*(k+nz*aW))] * phi_W
            - coef[i+nx*(j+ny*(k+nz*aN))] * phi_N
            - coef[i+nx*(j+ny*(k+nz*aS))] * phi_S;
        if constexpr (dim == 3) {
            newPhi -= coef[i+nx*(j+ny*(k+nz*aT))] * phi_T + coef[i+nx*(j+ny*(k+aB))] * phi_B;
        }
        newPhi /= coef[i+nx*(j+ny*(k+nz*aC))];

        scalar dPhi = relax * (newPhi - phi_C);

        field[CId] = phi_C + dPhi;

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

__global__ void GaussSeidelIterateKernel(scalar *field, scalar *coef, scalar *srcTerm, scalar *norm, scalar relax) {

    extern __shared__ scalar sharedNorm[];

    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {

        int CId = i + nx * (j + ny * k);
        scalar phi_C = field[CId];
        // east
        scalar phi_E = 0;
        if ( i != nx - 1 ) {
            int EId = i+1 + nx * (j + ny * k);
            phi_E = field[EId];
        }
        // west
        scalar phi_W = 0;
        if ( i != 0 ) {
            int WId = i-1 + nx * (j + ny * k);
            phi_W = field[WId];
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
        
        scalar newPhi = srcTerm[CId]
            - coef[i+nx*(j+ny*(k+aE))] * phi_E
            - coef[i+nx*(j+ny*(k+aW))] * phi_W
            - coef[i+nx*(j+ny*(k+aN))] * phi_N
            - coef[i+nx*(j+ny*(k+aS))] * phi_S;
        if constexpr (dim == 3) {
            newPhi -= coef[i+nx*(j+ny*(k+aT))] * phi_T + coef[i+nx*(j+ny*(k+aB))] * phi_B;
        }
        newPhi /= coef[i+nx*(j+ny*(k+aC))];

        scalar dPhi = relax * (newPhi - phi_C);

        field[CId] = phi_C + dPhi;

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

template<int dir>
__global__ void interiorRhieChowInterpolateKernel(scalar *uf, scalar *u, scalar *coef, scalar *p) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    bool loopFlag;

    if constexpr (dir == xDir) {
        ++(++i);
        loopFlag = (i < nx-1 && j < ny && k < nz);
    } else if constexpr (dir == yDir) {
        ++(++j);
        loopFlag = (i < nx && j < ny-1 && k < nz);
    } else if constexpr (dir == zDir) {
        ++(++k);
        loopFlag = (i < nx && j < ny && k < nz-1);
    }

    if (loopFlag) {

        int cId, LLId, LId, RId, RRId;
        scalar aC_L, aC_R;
        scalar area;

        if constexpr (dir == xDir) {
            cId  = i   + (nx+1) * (j + ny * k);
            LLId = i-2 + nx     * (j + ny * k);
            LId  = i-1 + nx     * (j + ny * k);
            RId  = i   + nx     * (j + ny * k);
            RRId = i+1 + nx     * (j + ny * k);
            aC_L = coef[i-1 + nx * (j + ny * (k + nz * aC))];
            aC_R = coef[i   + nx * (j + ny * (k + nz * aC))];
            area = areaX;
        } else if constexpr (dir == yDir) {
            cId  = i + nx * (j   + (ny+1) * k);
            LLId = i + nx * (j-2 + ny     * k);
            LId  = i + nx * (j-1 + ny     * k);
            RId  = i + nx * (j   + ny     * k);
            RRId = i + nx * (j+1 + ny     * k);
            aC_L = coef[i + nx * (j-1 + ny * (k + nz * aC))];
            aC_R = coef[i + nx * (j   + ny * (k + nz * aC))];
            area = areaY;
        } else if constexpr (dir == zDir) {
            cId  = i + nx * (j + ny * k    );
            LLId = i + nx * (j + ny * (k-2));
            LId  = i + nx * (j + ny * (k-1));
            RId  = i + nx * (j + ny * k    );
            RRId = i + nx * (j + ny * (k+1));
            aC_L = coef[i + nx * (j + ny * (k-1 + nz * aC))];
            aC_R = coef[i + nx * (j + ny * (k   + nz * aC))];
            area = areaZ;
        }

        scalar u_L = u[LId];
        scalar u_R = u[RId];

        uf[cId] = 0.5 * (u_L + u_R) + (p[RId] - p[LLId]) * area / (4 * aC_L) + (p[RRId] - p[LId]) * area / (4 * aC_R)
                + 0.5 * (1/aC_L + 1/aC_R) * (p[LId] - p[RId]) * area;
    }
}

template<int loc, int bcType>
__global__ void bcRhieChowInterpolateKernel(scalar *uf, scalar *u, scalar *coef, scalar *p) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    bool loopFlag;

    if constexpr (loc == west || loc == east) {
        loopFlag = (i < ny && j < nz);
    } else if constexpr (loc == south || loc == north) {
        loopFlag = (i < nx && j < nz);
    }  else if constexpr (loc == bottom || loc == top) {
        loopFlag = (i < nx && j < ny);
    }

    if (loopFlag) {

        int cId, LLId, LId, RId, RRId;
        scalar aC_L, aC_R;

        if constexpr (loc == west) {
            cId  = 1 + (nx+1) * (i + ny * j);
            LId  = 0 + nx     * (i + ny * j);
            RId  = 1 + nx     * (i + ny * j);
            RRId = 2 + nx     * (i + ny * j);
            aC_L = coef[0 + nx * (i + ny * (j + nz * aC))];
            aC_R = coef[1 + nx * (i + ny * (j + nz * aC))];
        } else if constexpr (loc == east) {
            cId  = nx-1 + (nx+1) * (i + ny * j);
            LLId = nx-3 + nx     * (i + ny * j);
            LId  = nx-2 + nx     * (i + ny * j);
            RId  = nx-1 + nx     * (i + ny * j);
            aC_L = coef[nx-2 + nx * (i + ny * (j + nz * aC))];
            aC_R = coef[nx-1 + nx * (i + ny * (j + nz * aC))];
        } else if constexpr (loc == south) {
            cId  = i + nx * (1 + (ny+1) * j);
            LId  = i + nx * (0 + ny     * j);
            RId  = i + nx * (1 + ny     * j);
            RRId = i + nx * (2 + ny     * j);
            aC_L = coef[i + nx * (0 + ny * (j + nz * aC))];
            aC_R = coef[i + nx * (1 + ny * (j + nz * aC))];
        } else if constexpr (loc == north) {
            cId  = i + nx * (ny-1 + (ny+1) * j);
            LLId = i + nx * (ny-3 + ny     * j);
            LId  = i + nx * (ny-2 + ny     * j);
            RId  = i + nx * (ny-1 + ny     * j);
            aC_L = coef[i + nx * (ny-2 + ny * (j + nz * aC))];
            aC_R = coef[i + nx * (ny-1 + ny * (j + nz * aC))];
        } else if constexpr (loc == bottom) {
            cId  = i + nx * (j + ny * 1);
            LId  = i + nx * (j + ny * 0);
            RId  = i + nx * (j + ny * 1);
            RRId = i + nx * (j + ny * 2);
            aC_L = coef[i + nx * (j + ny * (0 + nz * aC))];
            aC_R = coef[i + nx * (j + ny * (1 + nz * aC))];
        } else if constexpr (loc == top) {
            cId  = i + nx * (j + ny * (nz-1));
            LLId = i + nx * (j + ny * (nz-3));
            LId  = i + nx * (j + ny * (nz-2));
            RId  = i + nx * (j + ny * (nz-1));
            aC_L = coef[i + nx * (j + ny * (nz-2 + nz * aC))];
            aC_R = coef[i + nx * (j + ny * (nz-1 + nz * aC))];
        }

        scalar area;

        if constexpr (loc == west || loc == east) {
            area = areaX;
        } else if constexpr (loc == south || loc == north) {
            area = areaY;
        } else if constexpr (loc == bottom || loc == top) {
            area = areaZ;
        }

        scalar u_L = u[LId];
        scalar u_R = u[RId];

        scalar p_LL, p_L, p_R, p_RR;
        p_L = p[LId];
        p_R = p[RId];

        if constexpr (loc == west || loc == south || loc == bottom) {
            if constexpr (bcType == wall || bcType == inlet) {
                p_LL = p_L;
            } else if constexpr (bcType == outlet) {
                p_LL = 0;
            }
            p_RR = p[RRId];
        } else if constexpr (loc == east || loc == north || loc == top) {
            if constexpr (bcType == wall || bcType == inlet) {
                p_RR = p_R;
            } else if constexpr (bcType == outlet) {
                p_RR = 0;
            }
            p_LL = p[LLId];
        }

        uf[cId] = 0.5 * (u_L + u_R) + (p_R - p_LL) * area / (4 * aC_L) + (p_RR - p_L) * area / (4 * aC_R)
                + 0.5 * (1/aC_L + 1/aC_R) * (p_L - p_R) * area;
    }
}

void RhieChowInterpolate(scalar *uf_dev, scalar *vf_dev, scalar *wf_dev, scalar *u_dev, scalar *v_dev, scalar *w_dev
    , scalar *uCoef_dev, scalar *vCoef_dev, scalar *wCoef_dev, scalar *p_dev) {

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

    interiorRhieChowInterpolateKernel<xDir><<<numBlocks, threadsPerBlock, 0, stream[0]>>>(uf_dev, u_dev, uCoef_dev, p_dev);

    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny - 3 + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;

    interiorRhieChowInterpolateKernel<yDir><<<numBlocks, threadsPerBlock, 0, stream[1]>>>(vf_dev, v_dev, vCoef_dev, p_dev);

    if constexpr (dim == 3) {
        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = (nz - 3 + threadsPerBlock.z - 1) / threadsPerBlock.z;

        interiorRhieChowInterpolateKernel<zDir><<<numBlocks, threadsPerBlock, 0, stream[6]>>>(wf_dev, w_dev, wCoef_dev, p_dev);
    }

    if constexpr (dim == 2) {
        threadsPerBlock = dim3(1024, 1, 1);
    } else if constexpr (dim == 3) {
        threadsPerBlock = dim3(32, 32, 1);
    }

    numBlocks.x = (ny + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (nz + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;

    bcRhieChowInterpolateKernel<west, velBCs::type[west]><<<numBlocks, threadsPerBlock, 0, stream[2]>>>(uf_dev, u_dev, uCoef_dev, p_dev);

    bcRhieChowInterpolateKernel<east, velBCs::type[east]><<<numBlocks, threadsPerBlock, 0, stream[3]>>>(uf_dev, u_dev, uCoef_dev, p_dev);

    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (nz + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;

    bcRhieChowInterpolateKernel<south, velBCs::type[south]><<<numBlocks, threadsPerBlock, 0, stream[4]>>>(vf_dev, v_dev, vCoef_dev, p_dev);

    bcRhieChowInterpolateKernel<north, velBCs::type[north]><<<numBlocks, threadsPerBlock, 0, stream[5]>>>(vf_dev, v_dev, vCoef_dev, p_dev);

    if constexpr (dim == 3) {
        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = 1;

        bcRhieChowInterpolateKernel<bottom, velBCs::type[bottom]><<<numBlocks, threadsPerBlock, 0, stream[7]>>>(wf_dev, w_dev, wCoef_dev, p_dev);

        bcRhieChowInterpolateKernel<top, velBCs::type[top]><<<numBlocks, threadsPerBlock, 0, stream[8]>>>(wf_dev, w_dev, wCoef_dev, p_dev);
    }

    for (int i = 0; i < 3*dim; ++i) {
        cudaStreamSynchronize(stream[i]);
    }

    for (int i = 0; i < 3*dim; ++i) {
        cudaStreamDestroy(stream[i]);
    }
}

__global__ void calcPresCorrLinkCoefKernel(scalar *pCorrCoef, scalar *uCoef, scalar *vCoef, scalar *wCoef) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int aCId = i+nx*(j+ny*(k+nz*aC));
        int aEId = i+nx*(j+ny*(k+nz*aE));
        int aWId = i+nx*(j+ny*(k+nz*aW));
        int aNId = i+nx*(j+ny*(k+nz*aN));
        int aSId = i+nx*(j+ny*(k+nz*aS));
        int aTId = i+nx*(j+ny*(k+nz*aT));
        int aBId = i+nx*(j+ny*(k+nz*aB));

        if (i < nx - 1) {
            pCorrCoef[aEId] = -0.5 * density * areaX * areaX * (1/uCoef[i+nx*(j+ny*(k+nz*aC))] + 1/uCoef[i+1+nx*(j+ny*(k+nz*aC))]);
        }
        if (i > 0) {
            pCorrCoef[aWId] = -0.5 * density * areaX * areaX * (1/uCoef[i+nx*(j+ny*(k+nz*aC))] + 1/uCoef[i-1+nx*(j+ny*(k+nz*aC))]);
        }
        if (j < ny -1) {
            pCorrCoef[aNId] = -0.5 * density * areaY * areaY * (1/vCoef[i+nx*(j+ny*(k+nz*aC))] + 1/vCoef[i+nx*(j+1+ny*(k+nz*aC))]);
        }
        if (j > 0) {
            pCorrCoef[aSId] = -0.5 * density * areaY * areaY * (1/vCoef[i+nx*(j+ny*(k+nz*aC))] + 1/vCoef[i+nx*(j-1+ny*(k+nz*aC))]);
        }
        pCorrCoef[aCId] = -(pCorrCoef[aEId] + pCorrCoef[aWId] + pCorrCoef[aNId] + pCorrCoef[aSId]);
        if constexpr (dim == 3) {
            if (k < nz - 1) {
                pCorrCoef[i+nx*(j+ny*(k+aT))] = -0.5 * density * areaZ * areaZ * (1/wCoef[i+nx*(j+ny*(k+nz*aC))] + 1/wCoef[i+nx*(j+ny*(k+1+nz*aC))]);
            }
            if (k > 0) {
                pCorrCoef[i+nx*(j+ny*(k+aB))] = -0.5 * density * areaZ * areaZ * (1/wCoef[i+nx*(j+ny*(k+nz*aC))] + 1/wCoef[i+nx*(j+ny*(k-1+nz*aC))]);
            }
            pCorrCoef[aCId] += -(pCorrCoef[aTId] + pCorrCoef[aBId]);
        }
    }
}

void calcPresCorrLinkCoef(scalar *pCorrCoef_dev, scalar *uCoef_dev, scalar *vCoef_dev, scalar *wCoef_dev) {

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

    calcPresCorrLinkCoefKernel<<<numBlocks, threadsPerBlock>>>(pCorrCoef_dev, uCoef_dev, vCoef_dev, wCoef_dev);

    cudaDeviceSynchronize();
}

__global__ void calcPresCorrSrcTermKernel(scalar *pCorrSrcTerm, scalar *uf, scalar *vf, scalar *wf) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int CId = i   + nx     * (j   + ny     * k);
        int eId = i+1 + (nx+1) * (j   + ny     * k);
        int wId = i   + (nx+1) * (j   + ny     * k);
        int nId = i   + nx     * (j+1 + (ny+1) * k);
        int sId = i   + nx     * (j   + (ny+1) * k);

        pCorrSrcTerm[CId] = density * (areaX * (uf[wId] - uf[eId]) + areaY * (vf[sId] - vf[nId]));
        if (dim == 3) {
            int tId = i + nx * (j + ny * (k+1));
            int bId = i + nx * (j + ny * k    );

            pCorrSrcTerm[CId] += density * areaZ * (wf[bId] - wf[tId]);
        }
    }
}

void calcPresCorrSrcTerm(scalar *pCorrSrcTerm_dev, scalar *uf_dev, scalar *vf_dev, scalar *wf_dev) {

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

    calcPresCorrSrcTermKernel<<<numBlocks, threadsPerBlock>>>(pCorrSrcTerm_dev, uf_dev, vf_dev, wf_dev);

    cudaDeviceSynchronize();
}

template<int dir>
__global__ void updateInteriorVelKernel(scalar *u, scalar *coef, scalar *pCorr, scalar *norm, scalar relax) {

    extern __shared__ scalar sharedNorm[];

    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    bool loopFlag;

    if constexpr (dir == xDir) {
        ++i;
        loopFlag = (i < nx-1 && j < ny && k < nz);
    } else if constexpr (dir == yDir) {
        ++j;
        loopFlag = (i < nx && j < ny-1 && k < nz);
    } else if constexpr (dir == zDir) {
        ++k;
        loopFlag = (i < nx && j < ny && k < nz-1);
    }

    if (loopFlag) {

        int CId = i + nx * (j + ny * k);
        int LId, RId;
        scalar area;

        if constexpr (dir == xDir) {
            LId = i-1 + nx * (j + ny * k);
            RId = i+1 + nx * (j + ny * k);
            area = areaX;
        } else if constexpr (dir == yDir) {
            LId = i + nx * (j-1 + ny * k);
            RId = i + nx * (j+1 + ny * k);
            area = areaY;
        } else if constexpr (dir == zDir) {
            LId = i + nx * (j + ny * (k-1));
            RId = i + nx * (j + ny * (k+1));
            area = areaZ;
        }

        scalar pCorr_L = pCorr[LId];
        scalar pCorr_R = pCorr[RId];
        scalar aC_C = coef[i+nx*(j+ny*(k+aC))];

        scalar du = relax * 0.5 * (pCorr_L - pCorr_R) * area / aC_C;

        u[CId] += du;

        sharedNorm[tid] = du*du;
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

template<int loc, int bcType>
__global__ void updateBCVelKernel(scalar *u, scalar *coef, scalar *pCorr, scalar *norm, scalar relax) {

    extern __shared__ scalar sharedNorm[];

    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    bool loopFlag;

    if constexpr (loc == west || loc == east) {
        loopFlag = (i < ny && j < nz);
    } else if constexpr (loc == south || loc == north) {
        loopFlag = (i < nx && j < nz);
    } else if constexpr (loc == bottom || loc == top) {
        loopFlag = (i < nx && j < ny);
    }

    if (loopFlag) {

        int CId, LId, RId;
        scalar aC_C;

        if constexpr (loc == west) {
            CId = 0 + nx * (i + ny * j);
            RId = 1 + nx * (i + ny * j);
            aC_C = coef[0+nx*(i+ny*(j+aC))];
        } else if constexpr (loc == east) {
            CId = nx-1 + nx * (i + ny * j);
            LId = nx-2 + nx * (i + ny * j);
            aC_C = coef[nx-1+nx*(i+ny*(j+aC))];
        } else if constexpr (loc == south) {
            CId = i + nx * (0 + ny * j);
            RId = i + nx * (1 + ny * j);
            aC_C = coef[i+nx*(0+ny*(j+aC))];
        } else if constexpr (loc == north) {
            CId = i + nx * (ny-1 + ny * j);
            LId = i + nx * (ny-2 + ny * j);
            aC_C = coef[i+nx*(ny-1+ny*(j+aC))];
        } else if constexpr (loc == bottom) {
            CId = i + nx * (j + ny * 0);
            RId = i + nx * (j + ny * 1);
            aC_C = coef[i+nx*(j+ny*(0+aC))];
        } else if constexpr (loc == bottom) {
            CId = i + nx * (j + ny * (nz-1));
            LId = i + nx * (j + ny * (nz-2));
            aC_C = coef[i+nx*(j+ny*(nz-1+aC))];
        }

        scalar area;

        if constexpr (loc == west || loc == east) {
            area = areaX;
        } else if constexpr (loc == south || loc == north) {
            area = areaY;
        } else if constexpr (loc == bottom || loc == top) {
            area = areaZ;
        }

        scalar pCorr_L, pCorr_R;

        if constexpr (loc == west || loc == south || loc == bottom) {
            if constexpr (bcType == wall || bcType == inlet) {
                pCorr_L = pCorr[CId];
            } else if constexpr (bcType == outlet) {
                pCorr_L = 0;
            }
            pCorr_R = pCorr[RId];
        } else if (loc == east || loc == north || loc == top) {
            if constexpr (bcType == wall || bcType == inlet) {
                pCorr_R = pCorr[CId];
            } else if constexpr (bcType == outlet) {
                pCorr_R = 0;
            }
            pCorr_L = pCorr[LId];
        }

        scalar du = relax * 0.5 * (pCorr_L - pCorr_R) * area / aC_C;

        u[CId] += du;

        sharedNorm[tid] = du*du;
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

template<int dir>
__global__ void updateFaceVelKernel(scalar *uf, scalar *coef, scalar *pCorr, scalar *norm, scalar relax) {

    extern __shared__ scalar sharedNorm[];

    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if constexpr (dir == xDir) {
        ++i;
    } else if constexpr (dir == yDir) {
        ++j;
    } else if constexpr (dir == zDir) {
        ++k;
    }

    if (i < nx && j < ny && k < nz) {

        int cId, LId, RId;
        scalar area;
        if constexpr (dir == xDir) {
            cId = i   + (nx+1) * (j + ny * k);
            LId = i-1 + nx     * (j + ny * k);
            RId = i   + nx     * (j + ny * k);
            area = areaX;
        } else if constexpr (dir == yDir) {
            cId = i + nx * (j   + (ny+1) * k);
            LId = i + nx * (j-1 + ny     * k);
            RId = i + nx * (j   + ny     * k);
            area = areaY;
        } else if constexpr (dir == zDir) {
            cId = i + nx * (j + ny * k    );
            LId = i + nx * (j + ny * (k-1));
            RId = i + nx * (j + ny * k    );
            area = areaZ;
        }

        scalar duf;

        scalar aC_L = coef[LId+nx*ny*nz*aC];
        scalar aC_R = coef[RId+nx*ny*nz*aC];
        duf = relax * 0.5 * (1/aC_L + 1/aC_R) * (pCorr[LId] - pCorr[RId]) * area;

        uf[cId] += duf;

        sharedNorm[tid] = duf*duf;
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

    cudaStream_t stream[4*dim+1];

    for (int i = 0; i < 4*dim+1; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    cudaMemsetAsync(pNorm_dev, 0, sizeof(scalar), stream[0]);
    cudaMemsetAsync(uNorm_dev, 0, sizeof(scalar), stream[1]);
    cudaMemsetAsync(vNorm_dev, 0, sizeof(scalar), stream[2]);
    cudaMemsetAsync(ufNorm_dev, 0, sizeof(scalar), stream[3]);
    cudaMemsetAsync(vfNorm_dev, 0, sizeof(scalar), stream[4]);
    if constexpr (dim == 3) {
        cudaMemsetAsync(wNorm_dev, 0, sizeof(scalar), stream[9]);
        cudaMemsetAsync(wfNorm_dev, 0, sizeof(scalar), stream[10]);
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

    updatePresKernel<<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[0]>>>(p_dev, pCorr_dev, pNorm_dev, relax_p);

    numBlocks.x = (nx-2 + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny   + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz   + threadsPerBlock.z - 1) / threadsPerBlock.z;

    updateInteriorVelKernel<xDir><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[1]>>>(u_dev, uCoef_dev, pCorr_dev
        , uNorm_dev, relax_u);

    numBlocks.x = (nx   + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny-2 + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz   + threadsPerBlock.z - 1) / threadsPerBlock.z;

    updateInteriorVelKernel<yDir><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[2]>>>(v_dev, vCoef_dev, pCorr_dev
        , vNorm_dev, relax_v);

    numBlocks.x = (nx-1 + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny   + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz   + threadsPerBlock.z - 1) / threadsPerBlock.z;

    updateFaceVelKernel<xDir><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[3]>>>(uf_dev, uCoef_dev, pCorr_dev
        , ufNorm_dev, relax_u);

    numBlocks.x = (nx   + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny-1 + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz   + threadsPerBlock.z - 1) / threadsPerBlock.z;

    updateFaceVelKernel<yDir><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[4]>>>(vf_dev, vCoef_dev, pCorr_dev
        , vfNorm_dev, relax_v);

    if constexpr (dim == 3) {
        numBlocks.x = (nx   + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny   + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = (nz-2 + threadsPerBlock.z - 1) / threadsPerBlock.z;

        updateInteriorVelKernel<zDir><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[9]>>>(w_dev, wCoef_dev, pCorr_dev
            , wNorm_dev, relax_w);

        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = (nz+1 + threadsPerBlock.z - 1) / threadsPerBlock.z;

        updateFaceVelKernel<zDir><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[10]>>>(wf_dev, wCoef_dev, pCorr_dev
            , wfNorm_dev, relax_w);
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

    updateBCVelKernel<west, velBCs::type[west]><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[5]>>>(u_dev, uCoef_dev, pCorr_dev
        , uNorm_dev, relax_u);

    updateBCVelKernel<east, velBCs::type[east]><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[6]>>>(u_dev, uCoef_dev, pCorr_dev
        , uNorm_dev, relax_u);

    numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (nz + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = 1;

    updateBCVelKernel<south, velBCs::type[south]><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[7]>>>(v_dev, vCoef_dev, pCorr_dev
        , vNorm_dev, relax_v);

    updateBCVelKernel<north, velBCs::type[north]><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[8]>>>(v_dev, vCoef_dev, pCorr_dev
        , vNorm_dev, relax_v);

    if constexpr (dim == 3) {
        numBlocks.x = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = 1;

        updateBCVelKernel<bottom, velBCs::type[bottom]><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[11]>>>(w_dev, wCoef_dev, pCorr_dev
        , wNorm_dev, relax_w);

        updateBCVelKernel<top, velBCs::type[top]><<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize, stream[12]>>>(w_dev, wCoef_dev, pCorr_dev
        , wNorm_dev, relax_w);
    }

    for (int i = 0; i < 4*dim+1; ++i) {
        cudaStreamSynchronize(stream[i]);
    }

    for (int i = 0; i < 4*dim+1; ++i) {
        cudaStreamDestroy(stream[i]);
    }

}