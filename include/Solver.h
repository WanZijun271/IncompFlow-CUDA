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

    std::vector<scalar> _temp;        // temperature field

    // std::vector<scalar> _coef;             // coefficient

public:
    Solver(scalar u=0.0, scalar v=0.0, scalar w=0.0, scalar p=0.0, scalar temp=273.0);
    void solve();
    // void pointJacobiSolver();                            // solve the equation using the Jacobi interative method
    // void GaussSeidelSolver();                            // solve the equation using the Gauss-Seidel interative method
    void writeVTK(const std::string &filepath) const;    // write the result to .vtk file
private:
    // void calcCoef();                                     // calculate the coefficients of the equation
};

#endif