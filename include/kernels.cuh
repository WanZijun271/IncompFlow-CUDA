#ifndef KERNELS_H
#define KERNELS_H

#include "types.h"
#include <vector>

void initFaceVel(scalar *u_dev, scalar *v_dev, scalar *w_dev, scalar *uf_dev, scalar *vf_dev, scalar *wf_dev);

void applyBCsToFaceVel(scalar *uf_dev, scalar *vf_dev, scalar *wf_dev, scalar *u_dev, scalar *v_dev, scalar *w_dev);

void calcMomLinkCoef(scalar *coef_dev, scalar *uf_dev, scalar *vf_dev, scalar *wf_dev);

void calcMomSrcTerm(scalar *uSrcTerm_dev, scalar *vSrcTerm_dev, scalar *wSrcTerm_dev, scalar *p_dev);

void applyBCsToMomEq(scalar *uCoef_dev, scalar *uSrcTerm_dev, scalar *vCoef_dev, scalar *vSrcTerm_dev, scalar *wCoef_dev
    , scalar *wSrcTerm_dev);

void pointJacobiIterate(scalar *field_dev, size_t fieldSize, scalar *coef_dev, scalar *srcTerm_dev);

void GaussSeidelIterate(scalar *field_dev, scalar *coef_dev, scalar *srcTerm_dev);

void RhieChowInterpolate(scalar *uf_dev, scalar *vf_dev, scalar *wf_dev, scalar *u_dev, scalar *v_dev, scalar *w_dev
    , scalar *uCoef_dev, scalar *vCoef_dev, scalar *wCoef_dev, scalar *p_dev);

#endif