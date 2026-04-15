#include "kernels.cuh"
#include "config.h"
#include "constants.h"
#include <cmath>
#include <algorithm>

using namespace std;

__global__ void initUfKernel(scalar *u, scalar *uf) {

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (dim == 2 && i < nx && j < ny-1) {
        int id_c = i + j * (nx + 1);
        int id_W = i - 1 + j * nx;
        int id_E = i + j * nx;
        uf[id_c] = (u[id_W] + u[id_E]) / scalar(2.0);
    } else if (dim == 3 && i < nx && j < ny-1 && k < nz-1) {
        int id_c = i + j * (nx + 1) + k * (nx + 1) * ny;
        int id_W = i - 1 + j * nx + k * nx * ny;
        int id_E = i + j * nx + k * nx * ny;
        uf[id_c] = (u[id_W] + u[id_E]) / scalar(2.0);
    }
}

__global__ void initVfKernel(scalar *v, scalar *vf) {

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (dim == 2 && i < nx-1 && j < ny) {
        int id_c = j + i * (ny + 1);
        int id_S = j - 1 + i * ny;
        int id_N = j + i * ny;
        vf[id_c] = (v[id_S] + v[id_N]) / scalar(2.0);
    } else if (dim == 3 && i < nx-1 && j < ny && k < nz-1) {
        int id_c = j + k * (ny + 1) + i * (ny + 1) * nz;
        int id_S = j - 1 + k * ny + i * ny * nz;
        int id_N = j + k * ny + i * ny * nz;
        vf[id_c] = (v[id_S] + v[id_N]) / scalar(2.0);
    }
}

__global__ void initWfKernel(scalar *w, scalar *wf) {

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (dim == 3 && i < nx-1 && j < ny-1 && k < nz) {
        int id_c = k + i * (nz + 1) + j * (nz + 1) * nx;
        int id_B = k - 1 + i * nz + j * nz * nx;
        int id_T = k + i * nz + j * nz * nx;
        wf[id_c] = (w[id_B] + w[id_T]) / scalar(2.0);
    }
}

void initVelOnFace(scalar *u_dev, scalar *v_dev, scalar *w_dev, scalar *uf_dev, scalar *vf_dev, scalar *wf_dev) {

    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    if (dim == 3) {
        cudaStreamCreate(&stream3);
    }

    dim3 threadsPerBlock;
    dim3 numBlocks;
    if (dim == 2) {
        threadsPerBlock.x = 32;
        threadsPerBlock.y = 32;
        threadsPerBlock.z = 1;
    } else if (dim == 3) {
        threadsPerBlock.x = 16;
        threadsPerBlock.y = 8;
        threadsPerBlock.z = 8;
    }

    numBlocks.x = (nx - 1 + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny - 2 + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz - 2 + threadsPerBlock.z - 1) / threadsPerBlock.z;
    numBlocks.z = max(1, numBlocks.z);
    initUfKernel<<<numBlocks, threadsPerBlock, 0, stream1>>>(u_dev, uf_dev);

    numBlocks.x = (nx - 2 + threadsPerBlock.x - 1) / threadsPerBlock.x;
    numBlocks.y = (ny - 1 + threadsPerBlock.y - 1) / threadsPerBlock.y;
    numBlocks.z = (nz - 2 + threadsPerBlock.z - 1) / threadsPerBlock.z;
    numBlocks.z = max(1, numBlocks.z);
    initVfKernel<<<numBlocks, threadsPerBlock, 0, stream2>>>(v_dev, vf_dev);

    if (dim == 3) {
        numBlocks.x = (nx - 2 + threadsPerBlock.x - 1) / threadsPerBlock.x;
        numBlocks.y = (ny - 2 + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks.z = (nz - 1 + threadsPerBlock.z - 1) / threadsPerBlock.z;
        initWfKernel<<<numBlocks, threadsPerBlock, 0, stream3>>>(w_dev, wf_dev);
    }

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    if (dim == 3) {
        cudaStreamSynchronize(stream3);
    }

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    if (dim == 3) {
        cudaStreamDestroy(stream3);
    }
}

__global__ void pointJacobiIterateKernel(scalar *tempField, scalar* tempField0, scalar *coef, scalar *norm) {

    extern __shared__ scalar sharedNorm[];

    int threadId = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int idP = i + j * nx + k * nx * ny;
        scalar tempP = tempField0[idP];
        // tE
        scalar tempE = 0.0;
        if ( i != nx - 1 ) {
            int idE = (i+1) + j * nx + k * nx * ny;
            tempE = tempField0[idE];
        }
        // tW
        scalar tempW = 0.0;
        if ( i != 0 ) {
            int idW = (i-1) + j * nx + k * nx * ny;
            tempW = tempField0[idW];
        }
        // tN
        scalar tempN = 0.0;
        if ( j != ny - 1 ) {
            int idN = i + (j+1) * nx + k * nx * ny;
            tempN = tempField0[idN];
        }
        // tS
        scalar tempS = 0.0;
        if ( j != 0 ) {
            int idS = i + (j-1) * nx + k * nx * ny;
            tempS = tempField0[idS];
        }
        // tT
        scalar tempT = 0.0;
        if ( k != nz - 1 ) {
            int idT = i + j * nx + (k+1) * nx * ny;
            tempT = tempField0[idT];
        }
        // tB
        scalar tempB = 0.0;
        if ( k != 0 ) {
            int idB = i + j * nx + (k-1) * nx * ny;
            tempB = tempField0[idB];
        }
        
        scalar newTemp = tempP;
        if (dim == 2) {
            int id = i * 6 + j * nx * 6;
            newTemp = coef[id+id_b]
                - coef[id+id_aE] * tempE
                - coef[id+id_aW] * tempW
                - coef[id+id_aN] * tempN
                - coef[id+id_aS] * tempS;
            newTemp /= coef[id+id_aP];
        } else if (dim == 3) {
            int id = i * 8 + j * nx * 8 + k * nx * ny * 8;
            newTemp = coef[id+id_b]
                - coef[id+id_aE] * tempE
                - coef[id+id_aW] * tempW
                - coef[id+id_aN] * tempN
                - coef[id+id_aS] * tempS
                - coef[id+id_aT] * tempT
                - coef[id+id_aB] * tempB;
            newTemp /= coef[id+id_aP];
        }

        scalar dT = relax * (newTemp - tempP);

        tempField[idP] = tempP + dT;

        sharedNorm[threadId] = dT*dT;
        __syncthreads();

        int stride = blockDim.x * blockDim.y * blockDim.z;
        for (int s = stride / 2; s > 0; s >>= 1) {
            if (threadId < s) {
                sharedNorm[threadId] += sharedNorm[threadId + s];
            }
            __syncthreads();
        }

        if (threadId == 0) {
            atomicAdd(norm, sharedNorm[0]);
        }
    }
}

void pointJacobiIterate(vector<scalar> &tempField, const vector<scalar> &coef) {

    size_t tempFieldSize = tempField.size() * sizeof(scalar);
    size_t coefSize = coef.size() * sizeof(scalar);

    scalar *devTempField, *devTempField0, *devCoef, *devNorm;
    cudaMalloc(&devTempField, tempFieldSize);
    cudaMalloc(&devTempField0, tempFieldSize);
    cudaMalloc(&devCoef, coefSize);
    cudaMalloc(&devNorm, sizeof(scalar));

    cudaMemcpy(devTempField, tempField.data(), tempFieldSize, cudaMemcpyHostToDevice);
    cudaMemset(devTempField0, 0.0, tempFieldSize);
    cudaMemcpy(devCoef, coef.data(), coefSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock;
    dim3 numBlocks;
    if (dim == 2) {
        threadsPerBlock.x = 32;
        threadsPerBlock.y = 32;
        threadsPerBlock.z = 1;

        numBlocks.x = (nx + 31) / 32;
        numBlocks.y = (ny + 31) / 32;
        numBlocks.z = 1;
    } else if (dim == 3) {
        threadsPerBlock.x = 16;
        threadsPerBlock.y = 8;
        threadsPerBlock.z = 8;

        numBlocks.x = (nx + 15) / 16;
        numBlocks.y = (ny + 7) / 8;
        numBlocks.z = (nz + 7) / 8;
    }

    int blockSize = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z;

    scalar maxNorm = -1e20;

    for (int it = 0; it < niter; ++it) {
        
        scalar *tmp = devTempField;
        devTempField = devTempField0;
        devTempField0 = tmp;

        scalar norm = 0.0;
        cudaMemcpy(devNorm, &norm, sizeof(scalar), cudaMemcpyHostToDevice);

        pointJacobiIterateKernel<<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize>>>(devTempField, devTempField0, devCoef, devNorm);
        
        cudaMemcpy(&norm, devNorm, sizeof(scalar), cudaMemcpyDeviceToHost);

        norm = sqrt(norm / (nx * ny * nz));

        maxNorm = max(norm, maxNorm);

        scalar relNorm = norm / (maxNorm + 1e-20);    // relative residual
        if (relNorm < tol) {
            break;
        }
    }

    cudaMemcpy(tempField.data(), devTempField, tempFieldSize, cudaMemcpyDeviceToHost);

    cudaFree(devTempField);
    cudaFree(devTempField0);
    cudaFree(devCoef);
    cudaFree(devNorm);
}

__global__ void GaussSeidelIterateKernel(scalar *tempField, scalar *coef, scalar *norm) {

    extern __shared__ scalar sharedNorm[];

    int threadId = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int idP = i + j * nx + k * nx * ny;
        scalar tempP = tempField[idP];
        // tE
        scalar tempE = 0.0;
        if ( i != nx - 1 ) {
            int idE = (i+1) + j * nx + k * nx * ny;
            tempE = tempField[idE];
        }
        // tW
        scalar tempW = 0.0;
        if ( i != 0 ) {
            int idW = (i-1) + j * nx + k * nx * ny;
            tempW = tempField[idW];
        }
        // tN
        scalar tempN = 0.0;
        if ( j != ny - 1 ) {
            int idN = i + (j+1) * nx + k * nx * ny;
            tempN = tempField[idN];
        }
        // tS
        scalar tempS = 0.0;
        if ( j != 0 ) {
            int idS = i + (j-1) * nx + k * nx * ny;
            tempS = tempField[idS];
        }
        // tT
        scalar tempT = 0.0;
        if ( k != nz - 1 ) {
            int idT = i + j * nx + (k+1) * nx * ny;
            tempT = tempField[idT];
        }
        // tB
        scalar tempB = 0.0;
        if ( k != 0 ) {
            int idB = i + j * nx + (k-1) * nx * ny;
            tempB = tempField[idB];
        }
        
        scalar newTemp = tempP;
        if (dim == 2) {
            int id = i * 6 + j * nx * 6;
            newTemp = coef[id+id_b]
                - coef[id+id_aE] * tempE
                - coef[id+id_aW] * tempW
                - coef[id+id_aN] * tempN
                - coef[id+id_aS] * tempS;
            newTemp /= coef[id+id_aP];
        } else if (dim == 3) {
            int id = i * 8 + j * nx * 8 + k * nx * ny * 8;
            newTemp = coef[id+id_b]
                - coef[id+id_aE] * tempE
                - coef[id+id_aW] * tempW
                - coef[id+id_aN] * tempN
                - coef[id+id_aS] * tempS
                - coef[id+id_aT] * tempT
                - coef[id+id_aB] * tempB;
            newTemp /= coef[id+id_aP];
        }

        scalar dT = relax * (newTemp - tempP);

        tempField[idP] = tempP + dT;

        sharedNorm[threadId] = dT*dT;
        __syncthreads();

        int stride = blockDim.x * blockDim.y * blockDim.z;
        for (int s = stride / 2; s > 0; s >>= 1) {
            if (threadId < s) {
                sharedNorm[threadId] += sharedNorm[threadId + s];
            }
            __syncthreads();
        }

        if (threadId == 0) {
            atomicAdd(norm, sharedNorm[0]);
        }
    }
}

void GaussSeidelIterate(vector<scalar> &tempField, const vector<scalar> &coef) {

    size_t tempFieldSize = tempField.size() * sizeof(scalar);
    size_t coefSize = coef.size() * sizeof(scalar);

    scalar *devTempField, *devCoef, *devNorm;
    cudaMalloc(&devTempField, tempFieldSize);
    cudaMalloc(&devCoef, coefSize);
    cudaMalloc(&devNorm, sizeof(scalar));

    cudaMemcpy(devTempField, tempField.data(), tempFieldSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devCoef, coef.data(), coefSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock;
    dim3 numBlocks;
    if (dim == 2) {
        threadsPerBlock.x = 32;
        threadsPerBlock.y = 32;
        threadsPerBlock.z = 1;

        numBlocks.x = (nx + 31) / 32;
        numBlocks.y = (ny + 31) / 32;
        numBlocks.z = 1;
    } else if (dim == 3) {
        threadsPerBlock.x = 16;
        threadsPerBlock.y = 8;
        threadsPerBlock.z = 8;

        numBlocks.x = (nx + 15) / 16;
        numBlocks.y = (ny + 7) / 8;
        numBlocks.z = (nz + 7) / 8;
    }

    int blockSize = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z;

    scalar maxNorm = -1e20;

    for (int it = 0; it < niter; ++it) {

        scalar norm = 0.0;
        cudaMemcpy(devNorm, &norm, sizeof(scalar), cudaMemcpyHostToDevice);

        GaussSeidelIterateKernel<<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize>>>(devTempField, devCoef, devNorm);
        
        cudaMemcpy(&norm, devNorm, sizeof(scalar), cudaMemcpyDeviceToHost);

        norm = sqrt(norm / (nx * ny * nz));

        maxNorm = max(norm, maxNorm);

        scalar relNorm = norm / (maxNorm + 1e-20);    // relative residual
        if (relNorm < tol) {
            break;
        }
    }

    cudaMemcpy(tempField.data(), devTempField, tempFieldSize, cudaMemcpyDeviceToHost);

    cudaFree(devTempField);
    cudaFree(devCoef);
    cudaFree(devNorm);
}