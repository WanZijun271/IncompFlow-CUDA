#ifndef KERNELS_H
#define KERNELS_H

#include "types.h"
#include <vector>

void initVelOnFace(scalar *u_dev, scalar *v_dev, scalar *w_dev, scalar *uf_dev, scalar *vf_dev, scalar *wf_dev);

void pointJacobiIterate(std::vector<scalar> &tempField, const std::vector<scalar> &coef);

void GaussSeidelIterate(std::vector<scalar> &tempField, const std::vector<scalar> &coef);

#endif