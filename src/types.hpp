/******************************************************************************
 * name     types.hpp
 * author   Tian Xia (tian.xia.th@dartmouth.edu)
 * date     Apr 23, 2024
 *
 * Define floating point numerical types, as well as floating
 * type constants
******************************************************************************/
#pragma once

#include <concepts>
#include <complex>
#include <cuComplex.h>  /// for cuComplex, cuDoubleComplex

#ifndef TYPES_H
#define TYPES_H

using DefaultPrecisionType = double;

template<typename T>
concept RealFpType = std::floating_point<T>;
template<typename T>
concept ComplexFpType = std::convertible_to<T, std::complex<double>>;
template<typename T>
concept ScalarType = RealFpType<T> || ComplexFpType<T>;

template<ScalarType T>
struct CudaComplex;

template<>
struct CudaComplex<float> {
    using Type = cuComplex;
    static constexpr Type img = {0.0, 1.0};
    static constexpr Type one = {1.0, 0.0};
    static constexpr Type half = {0.5, 0.0};
};

template<>
struct CudaComplex<double> {
    using Type = cuDoubleComplex;
    static constexpr Type img = {0.0, 1.0};
    static constexpr Type one = {1.0, 0.0};
    static constexpr Type half = {0.5, 0.0};
};

template<ScalarType T>
using CudaComplexType = typename CudaComplex<T>::Type;
template<ScalarType T>
using CudaComplexPtr = CudaComplexType<T> *;
template<ScalarType T>
using CudaComplexConstPtr = const CudaComplexType<T> *;

#endif  /// TYPES_H