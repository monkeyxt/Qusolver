/******************************************************************************
 * name     solver.hpp
 * author   Tian Xia (tian.xia.th@dartmouth.edu)
 * date     Apr 23, 2024
 *
 * Simple solver for open quantum systems
******************************************************************************/
#pragma once

#include <iostream>
#include <concepts>
#include <complex>
#include <cmath>
#include <vector>

#include <cuComplex.h>

#include "rk4.hpp"
#include "lindblad.hpp"
#include "operators.hpp"
#include "gpu.hpp"

#define _SOVLER_H_

/******************************************************************************
 * Class definition for dynamics solver.
******************************************************************************/
template<ScalarType T = DefaultPrecisionType>
class Solver {
public:

    using StateType = GPUSolver::DeviceMatrix<T>;
    using ComplexType = CudaComplex<T>::Type;
    typedef rk4Solver<StateType, ComplexType, T, ComplexMatrixAlgebra,
            ComplexMatrixOperations> rk4Type;

    /// Constructors
    Solver(Lindblad<T>& _system) : system(_system){}
    ~Solver() = default;

    /// Delete the other constructors
    Solver() = delete;
    Solver(const Lindblad<T>&) = delete;
    Solver& operator=(const Lindblad<T>&) = delete;
    Solver(Lindblad<T>&&) = delete;
    Solver& operator=(Lindblad<T>&&) = delete;

    /// Returns the evolved trajectory of the system 
    std::vector<std::complex<T>> evolve (const int steps, const double dt) {
        std::vector<std::complex<T>> res{};
        for (std::size_t n = 0; n < steps; n++) {
            rk4.doStep(system, system.evolved, n * dt, dt);
            auto observed = system.observe(system.evolved);
            res.push_back({observed.x, observed.y});
        }
        return res;
    }
    
private:
    rk4Type rk4;
    Lindblad<T>& system;
};