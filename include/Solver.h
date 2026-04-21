#ifndef SOLVER_H
#define SOLVER_H

#include <vector>
#include <string>
#include "types.h"

class Solver {
private:
    std::vector<scalar> _u;
    std::vector<scalar> _v;
    std::vector<scalar> _w;
    std::vector<scalar> _p;
    std::vector<scalar> _temp;

    scalar *_u_dev;
    scalar *_v_dev;
    scalar *_w_dev;
    scalar *_p_dev;
    scalar *_temp_dev;
    scalar *_pCorr_dev;

    scalar *_uf_dev;
    scalar *_vf_dev;
    scalar *_wf_dev;

    scalar *_uCoef_dev;
    scalar *_vCoef_dev;
    scalar *_wCoef_dev;
    scalar *_pCorrCoef_dev;

    scalar *_uSrcTerm_dev;
    scalar *_vSrcTerm_dev;
    scalar *_wSrcTerm_dev;
    scalar *_pCorrSrcTerm_dev;

    scalar *_uNorm_dev;
    scalar *_vNorm_dev;
    scalar *_wNorm_dev;
    scalar *_ufNorm_dev;
    scalar *_vfNorm_dev;
    scalar *_wfNorm_dev;
    scalar *_pNorm_dev;
    scalar *_massNorm_dev;
public:
    Solver(scalar u=0.0, scalar v=0.0, scalar w=0.0, scalar p=0.0, scalar temp=273.0);

    ~Solver();

    void solve();

    void writeVTK(const std::string &filepath) const;    // write the result to .vtk file
private:
    void initFaceVel();

    void applyBCsToFaceVel();

    void calcMomLinkCoef(scalar *coef_dev);
    void calcMomSrcTerm();

    void applyBCsToMomEq();

    void RhieChowInterpolate();

    void calcPresCorrLinkCoef();
    void calcPresCorrSrcTerm();

    void updateField();

    static void pointJacobiIterate(scalar *field_dev, size_t fieldSize, scalar *coef_dev, scalar *srcTerm_dev, int nIter, scalar relax, scalar tol);
    static void GaussSeidelIterate(scalar *field_dev, scalar *coef_dev, scalar *srcTerm_dev, int nIter, scalar relax, scalar tol);
};

#endif