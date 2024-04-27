/******************************************************************************
 * name     operators.hpp
 * author   Tian Xia (tian.xia.th@dartmouth.edu)
 * date     Apr 23, 2024
 * 
 * Class definition for Hamiltonian matrices, density matrices, and Pauli spin
 * operators.
******************************************************************************/
#pragma once

#include <concepts>
#include <vector>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuComplex.h"

/******************************************************************************
 * Class Definition of `HostMatrix`, a wrapper around raw host matrix pointer,
 * which lives in the GPU.
******************************************************************************/
class HostMatrix {
public:
    HostMatrix() : rowdim{}, coldim{}, mat{} {}
    ~HostMatrix() = default;

    HostMatrix(std::vector<cuDoubleComplex>& m,
               std::size_t _rowdim,
               std::size_t _coldim)
               : mat(m), rowdim(_rowdim), coldim(_coldim) {}

    HostMatrix(std::vector<cuDoubleComplex>&& m,
               std::size_t _rowdim,
               std::size_t _coldim)
            : mat(m), rowdim(_rowdim), coldim(_coldim) {}

    [[nodiscard]] std::size_t rows() const { return rowdim; }
    [[nodiscard]] std::size_t cols() const { return coldim; }
    [[nodiscard]] std::size_t size() const { return rowdim * coldim; }

    /// This is rather dangerous as it should only be used for copying data
    /// host to the GPU device.
    [[nodiscard]] std::vector<cuDoubleComplex> * data() const {
        return (std::vector<cuDoubleComplex> *) mat.data();
    }
private:
    std::vector<cuDoubleComplex> mat;
    std::size_t rowdim;
    std::size_t coldim;
};

/******************************************************************************
 * Class Definition of Spin operators
******************************************************************************/
class SpinOperators : public HostMatrix {

};