#ifndef CONFIG_H
#define CONFIG_H

#include "types.h"

constexpr int dim = 2;

constexpr int nx = 512;
constexpr int ny = 128;
constexpr int nz = 1;

constexpr scalar xmin = -0.0625;
constexpr scalar xmax = 0.0625;
constexpr scalar ymin = -0.0125;
constexpr scalar ymax = 0.0125;
constexpr scalar zmin = 0.0;
constexpr scalar zmax = 1.0;

constexpr scalar thermalConductivity = 0.598;                                                   // thermal conductivity 热导率
constexpr scalar specificHeatCapacity = 4186.0;                                                   // specific heat capacity 比热容
constexpr scalar density = 998.2;                                                                // density 密度 
constexpr scalar dynamicViscosity = 1.002e-3;
constexpr scalar thermalDiffusivity = thermalConductivity / specificHeatCapacity / density;    // thermal diffusivity 热扩散率

struct velBCs{
    static constexpr int type[6] = { // 0 for the "wall" type; 1 for the "inlet" type; 2 for the "outlet" type
        2,    // east
        1,    // west
        0,    // north
        0,    // south
        0,    // top
        0     // bottom
    };

    static constexpr scalar val[6][3] = {
        {0, 0, 0},    // east
        {0.001, 0, 0},    // west
        {0, 0, 0},    // north 
        {0, 0, 0},    // south
        {0, 0, 0},    // top
        {0, 0, 0}     // bottom
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

constexpr int numOuterIter = 100000;      // iteration times 迭代次数

constexpr int nIter_u = 10;
constexpr int nIter_v = 10;
constexpr int nIter_w = 10;
constexpr int nIter_p = 20;

constexpr scalar relax_u = 0.75;     // 松弛因子
constexpr scalar relax_v = 0.75;
constexpr scalar relax_w = 0.75;
constexpr scalar relax_p = 0.25;

constexpr scalar tol_u = 1e-4;
constexpr scalar tol_v = 1e-4;
constexpr scalar tol_w = 1e-4;
constexpr scalar tol_p = 1e-6;
constexpr scalar tol_mass = 1e-11;

#endif