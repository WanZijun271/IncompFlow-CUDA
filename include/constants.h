#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "config.h"
#include "types.h"

constexpr int east   = 0;
constexpr int west   = 1;
constexpr int north  = 2;
constexpr int south  = 3;
constexpr int top    = 4;
constexpr int bottom = 5;

constexpr int aC = 0;
constexpr int aE = 1;
constexpr int aW = 2;
constexpr int aN = 3;
constexpr int aS = 4;
constexpr int aT = 5;
constexpr int aB = 6;

constexpr scalar dx = (xmax - xmin) / (scalar)nx;
constexpr scalar dy = (ymax - ymin) / (scalar)ny;
constexpr scalar dz = (zmax - zmin) / (scalar)nz;

constexpr scalar areaX = dy * dz;
constexpr scalar areaY = dz * dx;
constexpr scalar areaZ = dx * dy;

#endif