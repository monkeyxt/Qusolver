/******************************************************************************
 * name     system.hpp
 * author   Tian Xia (tian.xia.th@dartmouth.edu)
 * date     Apr 23, 2024
 * 
 * Class definition for the current quantum system.
******************************************************************************/
#pragma once

#include <vector>

#include "operators.hpp"
#include "gpu.hpp"
#include "types.hpp"

/******************************************************************************
 * Class definition of the current quantum system state. The class keeps track
 * of the density, hamiltonian, and spin operators
******************************************************************************/
template<ScalarType T = DefaultPrecisionType>
class Lindblad {
public:
    using StateType = GPUSolver::DeviceMatrix<T>;
    using Matrix = GPUSolver::DeviceMatrix<T>;

    /// Constructors
    Lindblad(const HostMatrix<T>& _hamiltonian,
             const HostMatrix<T>& _density,
             const HostMatrix<T>& _observable,
             const std::vector<HostMatrix<T>>& _dissipationOps,
             const std::vector<T>& _couplingConstants)
           : systemSize(_hamiltonian.rows()),
           couplingConstants(_couplingConstants){
        /// Upload to device
        initialHamiltonian = _hamiltonian;
        evolved = _density;
        initialDensity = _density;
        observables = _observable;

        scratchpad.resize(systemSize, systemSize);

        /// Calculate the hermitian conjugates of the dissipation ops
        for (auto& ops : _dissipationOps) {
            dissipationOps.push_back(ops);
            GPUSolver::DeviceMatrix conjugate {ops.rows(), ops.cols()};
            GPUSolver::hermitian(conjugate, dissipationOps.back());
            dissipationOpsH.push_back(conjugate);
            lindbladian.push_back(dissipationOpsH.back()
                * dissipationOps.back());
        }
    }
    ~Lindblad() = default;

    /// Delete the other constructors
    Lindblad() = delete;
    Lindblad(const Lindblad&) = delete;
    Lindblad& operator=(const Lindblad&) = delete;
    Lindblad(Lindblad&&) = delete;
    Lindblad& operator=(Lindblad&&) = delete;

    /// Overloaded evolve for RK4, x is the current system state and `dxdt`
    /// is the next system state
    void operator()(const StateType& x, StateType& dxdt, T t, T dt) {
        typename CudaComplex<T>::Type complexTime = {dt, 0};
        dxdt = (vonNeumann(initialHamiltonian, x) + lindblad(x)) * complexTime;
    }

    /// Returns the observed of the current density state
    typename CudaComplex<T>::Type observe(const StateType& x) {
        return GPUSolver::matTr(x * observables);
    }

public:
    /// Evolved system state.
    Matrix evolved;

private:
    /// Returns the vonNeumann term for the current system
    Matrix vonNeumann(const Matrix& hamiltonian, const Matrix& density) {
        Matrix temp = hamiltonian * density;
        return (hamiltonian * density - density * hamiltonian) * CudaComplex<T>::img;
    }

    /// Returns the second term in the lindblad equation
    Matrix lindblad(const Matrix& density) {
        GPUSolver::zeroVec(scratchpad.size(), scratchpad);
        for(std::size_t i = 0; i < dissipationOps.size(); i++) {
            scratchpad += ((dissipationOps[i] * density * dissipationOpsH[i]
                           - (lindbladian[i] * density
                           + density * lindbladian[i]) 
                           * CudaComplex<T>::half)
                           * couplingConstants[i]);
        }
        return scratchpad;
    }

    /// Scratchpads for computation
    Matrix scratchpad;

    /// Size of the current system
    std::size_t systemSize{};

    /// Computable matrices in device
    Matrix observables;
    Matrix initialHamiltonian;
    Matrix initialDensity;
    std::vector<Matrix> dissipationOps;
    std::vector<Matrix> dissipationOpsH;
    std::vector<Matrix> lindbladian;

    /// The coupling constants converted to complex format for computation
    std::vector<T> couplingConstants;
};