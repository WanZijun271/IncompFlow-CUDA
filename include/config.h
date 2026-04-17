#ifndef CONFIG_H
#define CONFIG_H

#include "types.h"

constexpr int dim = 2;

constexpr int nx = 64;
constexpr int ny = 64;
constexpr int nz = 1;

constexpr scalar xmin = 0.0;
constexpr scalar xmax = 1.0;
constexpr scalar ymin = 0.0;
constexpr scalar ymax = 1.0;
constexpr scalar zmin = 0.0;
constexpr scalar zmax = 1.0;

constexpr scalar thermalConductivity = 0.598;                                                   // thermal conductivity 热导率
constexpr scalar specificHeatCapacity = 4186.0;                                                   // specific heat capacity 比热容
constexpr scalar density = 998.21;                                                                // density 密度 
constexpr scalar dynamicViscosity = 1.002e-3;
constexpr scalar thermalDiffusivity = thermalConductivity / specificHeatCapacity / density;    // thermal diffusivity 热扩散率

struct velBCs{
    static constexpr int type[6] = { // 0 for the "wall" type; 1 for the "inlet" type; 2 for the "outlet" type
        1,    // east
        2,    // west
        0,    // north
        0,    // south
        0,    // top
        0     // bottom
    };

    static constexpr scalar val[6][3] = {
        {1.0, 0.0, 0.0},    // east
        {0.0, 0.0, 0.0},    // west
        {0.0, 0.0, 0.0},    // north 
        {0.0, 0.0, 0.0},    // south
        {0.0, 0.0, 0.0},    // top
        {0.0, 0.0, 0.0}     // bottom
    };
};

struct TempBCs{
    static constexpr int type[6] = { // 0 for the "Dirichlet" type; 1 for the "Neumann" type
        1,    // east
        1,    // west
        0,    // north
        0,    // south
        1,    // top
        1     // bottom
    };

    static constexpr scalar val[6] = {
        0.0,      // east
        0.0,      // west
        293.0,    // north 
        373.0,    // south
        0.0,      // top
        0.0       // bottom
    };
};

constexpr int numOuterIter = 1000;      // iteration times 迭代次数
constexpr int numInnerIter = 1000;
constexpr scalar relax = 0.75;     // 松弛因子
constexpr scalar innerTol = 1e-2;       // tolerance of relative residual
constexpr scalar outerTol = 1e-4;

#endif