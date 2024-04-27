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

using Matrix = GPUSolver::DeviceMatrix;
/******************************************************************************
 * Class definition of the current quantum system state. The class keeps track
 * of the density, hamiltonian, and spin operators
******************************************************************************/
class Lindblad {
public:
    const cuDoubleComplex img = {0, 1};
    const cuDoubleComplex half = {0.5, 0};
    using stateType = GPUSolver::DeviceMatrix;

    /// Constructors
    Lindblad(const HostMatrix& _hamiltonian,
             const HostMatrix& _density,
             const HostMatrix& _observable,
             const std::vector<HostMatrix>& _dissipationOps,
             const std::vector<double>& _couplingConstants)
           : systemSize(_hamiltonian.rows()),
           couplingConstants(_couplingConstants){
        /// Upload to device
        initialHamiltonian = _hamiltonian;
        evolved = _hamiltonian;
        initialDensity = _density;
        observables = _observable;

        scratchpad.resize(systemSize, systemSize);

        /// Calculate the hermitian conjugates of the dissipation ops
        for (auto& ops : _dissipationOps) {
            dissipationOps.push_back(ops);
            GPUSolver::DeviceMatrix conjugate {ops.rows(), ops.cols()};
            GPUSolver::hermitian(conjugate, dissipationOps.back());
            dissipationOpsH.push_back(conjugate);
            lindbladian.push_back(dissipationOps.back() * dissipationOpsH.back());
        }
    }

    Lindblad() = delete;
    ~Lindblad() = default;

    /// Overloaded evolve for RK4, x is the current system state and `dxdt`
    /// is the next system state
    void operator()(const stateType& x, stateType& dxdt, double t) {
        cuDoubleComplex complexT = {t, 0};
        dxdt = (vonNeumann(initialHamiltonian, x) + lindblad(x)) * complexT;
    }

    /// Returns the observed of the current density state
    cuDoubleComplex observe(const stateType& x) {
        return GPUSolver::matTr(observables * x);
    }

public:
    /// System initial states.
    Matrix evolved;
    Matrix initialDensity;

private:
    /// Returns the vonNeumann term for the current system
    Matrix vonNeumann(const Matrix& hamiltonian, const Matrix& density) {
        Matrix temp = hamiltonian * density;
        return (hamiltonian * density - density * hamiltonian) * img;
    }

    /// Returns the second term in the lindblad equation
    Matrix lindblad(const Matrix& density) {
        GPUSolver::zeroVec(scratchpad.size(), scratchpad);
        for(std::size_t i = 0; i < dissipationOps.size(); i++) {
            scratchpad += (dissipationOps[i] * density * dissipationOpsH[i]
                           - (lindbladian[i] * density
                           + density * lindbladian[i]) * half)
                                   * couplingConstants[i];
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
    std::vector<Matrix> dissipationOps;
    std::vector<Matrix> dissipationOpsH;
    std::vector<Matrix> lindbladian;

    /// The coupling constants converted to complex format for computation
    std::vector<double> couplingConstants;
};