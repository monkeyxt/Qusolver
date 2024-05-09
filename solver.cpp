/******************************************************************************
 * name     solver.cpp
 * author   Tian Xia (tian.xia.th@dartmouth.edu)
 * date     Apr 23, 2024
 *
 * Entry point for the lindblad solver
******************************************************************************/
#include "qusolver.hpp"

using StateType = GPUSolver::DeviceMatrix<DefaultPrecisionType>;
typedef rk4Solver<StateType, CudaComplex<DefaultPrecisionType>::Type,
        DefaultPrecisionType,
        ComplexMatrixAlgebra,
        ComplexMatrixOperations> rk4Type;

/******************************************************************************
 * Example simulating for a small open quantum system
******************************************************************************/
int main() {
    /// Define the initial parameters for the lindblad solver
    /// All of these are in column-major order for solving on cuBLAS
    HostMatrix hamiltonian {{{500.0, 0}, {0, 0}, {0, 0}, {-500.0, 0}}, 2, 2};
    HostMatrix density {{{0, 0}, {0, 0}, {0, 0}, {1, 0}}, 2, 2};

    HostMatrix observationOp1 {{{1, 0}, {0, 0}, {0, 0}, {-1, 0}}, 2, 2};
    std::vector<HostMatrix<DefaultPrecisionType>> observationOps;
    observationOps.push_back(observationOp1);

    HostMatrix dissipationOp1 {{{0, 0}, {0, 0}, {1, 0}, {0, 0}}, 2, 2};
    HostMatrix dissipationOp2 {{{0, 0}, {1, 0}, {0, 0}, {0, 0}}, 2, 2};
    std::vector<HostMatrix<DefaultPrecisionType>> dissipationOps;
    dissipationOps.push_back(dissipationOp1);
    dissipationOps.push_back(dissipationOp2);

    const std::vector<DefaultPrecisionType> couplings = {2, 0};
    Lindblad system{hamiltonian,
                    density,
                    observationOps[0],
                    dissipationOps,
                    couplings};

    /// Define the simulation timeframe
    const int steps = 10000;
    const double dt = 0.1;

    rk4Type solver;
    std::cout << "timepoint" << "," << "observed" << std::endl;
    for (std::size_t n = 0; n < steps; n++) {
        solver.doStep(system, system.evolved, n * dt, dt);
        auto observed = system.observe(system.evolved);
        std::cout << n * dt << ","  << observed.x << std::endl;
    }
    return 0;
}