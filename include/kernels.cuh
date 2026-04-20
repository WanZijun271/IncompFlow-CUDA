#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "types.h"
#include <vector>

void initFaceVel(scalar *u_dev, scalar *v_dev, scalar *w_dev, scalar *uf_dev, scalar *vf_dev, scalar *wf_dev);

void applyBCsToFaceVel(scalar *uf_dev, scalar *vf_dev, scalar *wf_dev, scalar *u_dev, scalar *v_dev, scalar *w_dev);

void calcMomLinkCoef(scalar *coef_dev, scalar *uf_dev, scalar *vf_dev, scalar *wf_dev);

void calcMomSrcTerm(scalar *uSrcTerm_dev, scalar *vSrcTerm_dev, scalar *wSrcTerm_dev, scalar *p_dev);

void applyBCsToMomEq(scalar *uCoef_dev, scalar *uSrcTerm_dev, scalar *vCoef_dev, scalar *vSrcTerm_dev, scalar *wCoef_dev
    , scalar *wSrcTerm_dev);

void pointJacobiIterate(scalar *field_dev, size_t fieldSize, scalar *coef_dev, scalar *srcTerm_dev, int nIter, scalar relax
    , scalar tol);

void GaussSeidelIterate(scalar *field_dev, scalar *coef_dev, scalar *srcTerm_dev, int nIter, scalar relax, scalar tol);

void RhieChowInterpolate(scalar *uf_dev, scalar *vf_dev, scalar *wf_dev, scalar *u_dev, scalar *v_dev, scalar *w_dev
    , scalar *uCoef_dev, scalar *vCoef_dev, scalar *wCoef_dev, scalar *p_dev);

void calcPresCorrLinkCoef(scalar *pCorrCoef_dev, scalar *uCoef_dev, scalar *vCoef_dev, scalar *wCoef_dev);

void calcPresCorrSrcTerm(scalar *pCorrSrcTerm_dev, scalar *uf_dev, scalar *vf_dev, scalar *wf_dev);

void updateField(scalar *u_dev, scalar *v_dev, scalar *w_dev, scalar *uNorm_dev, scalar *vNorm_dev, scalar *wNorm_dev
    , scalar *uf_dev, scalar *vf_dev, scalar *wf_dev, scalar *ufNorm_dev, scalar *vfNorm_dev, scalar *wfNorm_dev
    , scalar *p_dev, scalar *pNorm_dev, scalar *uCoef_dev, scalar *vCoef_dev, scalar *wCoef_dev, scalar *pCorr_dev);

template<int dir>
__global__ void initFaceVelKernel(scalar *u, scalar *uf);

template<int loc, int bcType>
__global__ void applyBCsToFaceVelKernel(scalar *uf, scalar *u, scalar bcVal);

__global__ void calcMomLinkCoefKernel(scalar *coef, scalar *uf, scalar *vf, scalar *wf);

template<int dir>
__global__ void calcInteriorMomSrcTermKernel(scalar *srcTerm, scalar *p);

template<int loc, int bcType>
__global__ void calcBCMomSrcTermKernel(scalar *srcTerm, scalar *p);

template<int loc, int bcType>
__global__ void applyBCsToMomEqKernel(scalar *uCoef, scalar *vCoef, scalar *wCoef, scalar *uSrcTerm, scalar *vSrcTerm
    , scalar *wSrcTerm, scalar uBC, scalar vBC, scalar wBC);

__global__ void pointJacobiIterateKernel(scalar *field, scalar* field0, scalar *coef, scalar *srcTerm, scalar *norm, scalar relax);

__global__ void GaussSeidelIterateKernel(scalar *field, scalar *coef, scalar *srcTerm, scalar *norm, scalar relax);

template<int dir>
__global__ void interiorRhieChowInterpolateKernel(scalar *uf, scalar *u, scalar *coef, scalar *p);

template<int loc, int bcType>
__global__ void bcRhieChowInterpolateKernel(scalar *uf, scalar *u, scalar *coef, scalar *p);

__global__ void calcPresCorrLinkCoefKernel(scalar *pCorrCoef, scalar *uCoef, scalar *vCoef, scalar *wCoef);

__global__ void calcPresCorrSrcTermKernel(scalar *pCorrSrcTerm, scalar *uf, scalar *vf, scalar *wf);

template<int dir>
__global__ void updateInteriorVelKernel(scalar *u, scalar *coef, scalar *pCorr, scalar *norm, scalar relax);

template<int loc, int bcType>
__global__ void updateBCVelKernel(scalar *u, scalar *coef, scalar *pCorr, scalar *norm, scalar relax);

template<int dir>
__global__ void updateFaceVelKernel(scalar *uf, scalar *coef, scalar *pCorr, scalar *norm, scalar relax);

__global__ void updatePresKernel(scalar *p, scalar *pCorr, scalar *norm, scalar relax);

#endif