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

public:
    Solver(scalar u=0.0, scalar v=0.0, scalar w=0.0, scalar p=0.0, scalar temp=273.0);
    void solve();
    void writeVTK(const std::string &filepath) const;    // write the result to .vtk file
};

#endif