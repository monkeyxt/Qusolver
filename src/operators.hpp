/******************************************************************************
 * name     operators.hpp
 * author   Tian Xia (tian.xia.th@dartmouth.edu)
 * date     Apr 23, 2024
 * 
 * Class definition for operators, which includes matrices that live in the
 * CPU.
******************************************************************************/
#pragma once

#include <vector>
#include <iostream>

#include "types.hpp"    /// For ScalarType, DefaultPrecisionType, CudaComplex..

/******************************************************************************
 * Class Definition of `HostMatrix`, a wrapper around raw host matrix pointer,
 * which lives in the GPU.
******************************************************************************/
template<ScalarType T = DefaultPrecisionType>
class HostMatrix {
public:
    HostMatrix() : rowdim{}, coldim{}, mat{} {}
    ~HostMatrix() = default;

    HostMatrix(std::vector<CudaComplexType<T>>& m,
               std::size_t _rowdim,
               std::size_t _coldim)
               : mat(m), rowdim(_rowdim), coldim(_coldim), 
                 itemSz(_rowdim * _coldim) {}

    HostMatrix(std::vector<CudaComplexType<T>>&& m,
               std::size_t _rowdim,
               std::size_t _coldim)
               : mat(m), rowdim(_rowdim), coldim(_coldim),
                 itemSz(_rowdim * _coldim) {}

    /// Interface with pybind, converting from numpy buffer to our matrix type
    /// Probably not the most efficient, going to be optimized away.
    HostMatrix(std::vector<std::complex<T>>* ptr,
               std::size_t _rowdim,
               std::size_t _coldim)
               : rowdim(_rowdim), coldim(_coldim), itemSz(_rowdim * _coldim) {
                   mat.resize(itemSz);
                   for (std::size_t i = 0; i < itemSz; i++) {
                       mat.push_back({reinterpret_cast<T(&)[2]>(ptr[i])[0], 
                                      reinterpret_cast<T(&)[2]>(ptr[i])[0]});
                   }
               }
               
    [[nodiscard]] std::size_t rows() const { return rowdim; }
    [[nodiscard]] std::size_t cols() const { return coldim; }
    [[nodiscard]] std::size_t size() const { return rowdim * coldim; }

    /// Operator overloads
    CudaComplex<T> operator() (std::size_t rowIdx, std::size_t colIdx) const {
        return mat[colIdx * rowdim + rowIdx];
    }

    friend std::ostream& operator<< (std::ostream& stream,
                                     const HostMatrix<T>& matrix) {
        for (std::size_t i = 0; i < matrix.rowdim; i++) {
            for (std::size_t j = 0; j < matrix.coldim; j++) {
                std::cout << matrix(i, j) << " ";
            }
            std::cout << std::endl;
        }
        return stream;
    }

    /// This is rather dangerous as it should only be used for copying data
    /// host to the GPU device.
    [[nodiscard]] std::vector<CudaComplexType<T>> * data() const {
        return (std::vector<CudaComplexType<T>> *) mat.data();
    }

private:
    std::vector<CudaComplexType<T>> mat;
    std::size_t rowdim;
    std::size_t coldim;
    std::size_t itemSz;
};