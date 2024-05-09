/******************************************************************************
 * name     rk4.hpp
 * author   Tian Xia (tian.xia.th@dartmouth.edu)
 * date     Mar 14, 2024
 *
 * Generic definitions of RK4 that supports custom state, value, time, algebra
 * and operations type. Custom backends are supported to allow GPU/CUDA
 * acceleration.
******************************************************************************/
#pragma once

#include <iostream>

#include "cuComplex.h"  /// For cuDoubleComplex
#include "gpu.hpp"      /// For DeviceMatrix

/******************************************************************************
 * Default operations for RK4 solver that only solves numerical values.
******************************************************************************/
struct DefaultOperations {
    template<class Fac1 = double, class Fac2 = Fac1>
    struct ScaleSum2 {
        typedef void ResultType;
        const Fac1 alpha1;
        const Fac2 alpha2;
        ScaleSum2(Fac1 alpha1, Fac2 alpha2) : alpha1(alpha1), alpha2(alpha2) {}
        template<class T0, class T1, class T2>
        void operator()(T0 &t0, const T1 &t1, const T2 &t2) const {
            t0 = alpha1 * t1 + alpha2 * t2;
        }
    };
};

struct ContainerAlgebra {
    template<class S1, class S2, class S3, class Op>
    static void forEach3(S1 &s1, S2 &s2, S3 &s3, Op op) {
        const std::size_t dim = s1.size();
        for(std::size_t n = 0; n < dim; ++n)
            op(s1[n], s2[n], s3[n]);
    }
};

/******************************************************************************
 * GPU operations for RK4 solver that applies rk4 to matrices
******************************************************************************/
struct ComplexMatrixOperations {
    template<class Fac1 = CudaComplex<DefaultPrecisionType>::Type, 
            class Fac2 = double>
    struct ScaleSum2 {
        typedef void ResultType;
        const Fac1 alpha1;
        const Fac2 alpha2;
        ScaleSum2(Fac1 alpha1, Fac2 alpha2) : alpha1(alpha1), alpha2(alpha2) {}
        template<class T0, class T1, class T2>
        void operator()(T0 &t0, const T1 &t1, const T2 &t2) const {
            t0 = t1  * alpha1 + t2 * alpha2;
        }
    };
    template<class Fac1 = CudaComplex<DefaultPrecisionType>::Type, 
            class Fac2 = double,
            class Fac3 = Fac2, class Fac4 = Fac2, class Fac5 = Fac2>
    struct ScaleSum5 {
        typedef void ResultType;
        const Fac1 alpha1;
        const Fac2 alpha2;
        const Fac3 alpha3;
        const Fac4 alpha4;
        const Fac5 alpha5;
        ScaleSum5(Fac1 alpha1, Fac2 alpha2, Fac3 alpha3, Fac4 alpha4,
                  Fac5 alpha5)
                : alpha1(alpha1), alpha2(alpha2), alpha3(alpha3),
                  alpha4(alpha4), alpha5(alpha5) {}
        template<class T0, class T1, class T2, class T3, class T4, class T5>
        void operator()(T0 &t0, const T1 &t1, const T2 &t2, const T3& t3,
                        const T4& t4, const T5& t5) const {
            t0 = t1  * alpha1 + t2 * alpha2 + t3 * alpha3 + t4 * alpha4
                 + t5 * alpha5;
        }
    };
};

struct ComplexMatrixAlgebra {
    template<class S1, class S2, class S3, class Op>
    static void forEach3(S1 &s1, S2 &s2, S3 &s3, Op op) {
        op(s1, s2, s3);
    }
    template<class S1, class S2, class S3, class S4, class S5, class S6,
            class Op>
    static void forEach6(S1 &s1, S2 &s2, S3 &s3, S4 &s4, S5 &s5, S6 &s6,
                         Op op) {
        op(s1, s2, s3, s4, s5, s6);
    }
};

template<class StateType>
void resize(const StateType &in, StateType &out) {
    out.resize(in.rows(), in.cols());
}

/******************************************************************************
 * Class definition for RK4 solver.
******************************************************************************/
template<class StateType, class ValueType = double,
        class TimeType = ValueType,
        class Algebra = ContainerAlgebra,
        class Operations = DefaultOperations>
class rk4Solver {
public:
    template<typename System>
    void doStep(System &system, StateType &x, TimeType t, TimeType dt) {
        adjustSize(x);
        const auto one = CudaComplex<TimeType>::one;
        const TimeType dt2 = dt / 2, dt3 = dt/3, dt6 = dt / 6;
        typedef typename Operations::template ScaleSum2<
                ValueType, TimeType> ScaleSum2;
        typedef typename Operations::template ScaleSum5<
                ValueType, TimeType, TimeType, TimeType, TimeType> ScaleSum5;
        system(x, k1, t, dt);
        Algebra::forEach3(tmp, x, k1, ScaleSum2(one, dt2));
        system(tmp, k2, t + dt2, dt2);
        Algebra::forEach3(tmp, x, k2, ScaleSum2(one, dt2));
        system(tmp, k3, t + dt, dt);
        Algebra::forEach3(tmp, x, k3, ScaleSum2(one, dt));
        system(tmp, k4, t + dt, dt);
        Algebra::forEach6(x, x, k1, k2, k3, k4,
                          ScaleSum5(one, dt6, dt3, dt3, dt6));
    }

private:
    StateType tmp, k1, k2, k3, k4;
    void adjustSize(const StateType &x) {
        resize(x, tmp);
        resize(x, k1);
        resize(x, k2);
        resize(x, k3);
        resize(x, k4);
    }
};