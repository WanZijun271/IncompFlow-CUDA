#include <iostream>
#include <chrono>
#include "Solver.h"

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    Solver solver{0, 0, 0, 0, 0};
    solver.solve();
    solver.writeVTK("test1.vtk");

    auto end = std::chrono::high_resolution_clock::now();

    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed time: " << duration_ms.count() << " ms\n";
    return 0;
}