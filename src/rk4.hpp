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
#include "cuComplex.h"
#include "gpu.hpp"

/******************************************************************************
 * Default operations for RK4 solver that only solves numerical values.
******************************************************************************/
struct default_operations {
    template<class Fac1 = double, class Fac2 = Fac1>
    struct scale_sum2 {
        typedef void result_type;
        const Fac1 alpha1;
        const Fac2 alpha2;
        scale_sum2(Fac1 alpha1, Fac2 alpha2) : alpha1(alpha1), alpha2(alpha2) {}
        template<class T0, class T1, class T2>
        void operator()(T0 &t0, const T1 &t1, const T2 &t2) const {
            t0 = alpha1 * t1 + alpha2 * t2;
        }
    };
};

struct container_algebra {
template<class S1, class S2, class S3, class Op>
static void for_each3(S1 &s1, S2 &s2, S3 &s3, Op op) {
    const size_t dim = s1.size();
    for(size_t n = 0; n < dim; ++n)
        op(s1[n], s2[n], s3[n]);
    }
};

template<class state_type>
void resize(const state_type &in, state_type &out) {
    out.resize(in.rows(), in.cols());
}

/******************************************************************************
 * GPU operations for RK4 solver that applies rk4 to matrices
******************************************************************************/
struct complexMatrixOperations {
    template<class Fac1 = cuDoubleComplex , class Fac2 = double>
    struct scale_sum2 {
        typedef void result_type;
        const Fac1 alpha1;
        const Fac2 alpha2;
        scale_sum2(Fac1 alpha1, Fac2 alpha2) : alpha1(alpha1), alpha2(alpha2) {}
        template<class T0, class T1, class T2>
        void operator()(T0 &t0, const T1 &t1, const T2 &t2) const {
            t0 = t1  * alpha1 + t2 * alpha2;
        }
    };
    template<class Fac1 = cuDoubleComplex, class Fac2 = double,
            class Fac3 = Fac2, class Fac4 = Fac2, class Fac5 = Fac2>
    struct scale_sum5 {
        typedef void result_type;
        const Fac1 alpha1;
        const Fac2 alpha2;
        const Fac3 alpha3;
        const Fac4 alpha4;
        const Fac5 alpha5;
        scale_sum5(Fac1 alpha1, Fac2 alpha2, Fac3 alpha3, Fac4 alpha4,
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

struct complexMatrixAlgebra {
    template<class S1, class S2, class S3, class Op>
    static void for_each3(S1 &s1, S2 &s2, S3 &s3, Op op) {
        op(s1, s2, s3);
    }
    template<class S1, class S2, class S3, class S4, class S5, class S6,
            class Op>
    static void for_each6(S1 &s1, S2 &s2, S3 &s3, S4 &s4, S5 &s5, S6 &s6,
                          Op op) {
        op(s1, s2, s3, s4, s5, s6);
    }
};

/******************************************************************************
 * Class definition for RK4 solver.
******************************************************************************/
template<class state_type, class value_type = double,
    class time_type = value_type,
    class algebra = container_algebra,
    class operations = default_operations>
class rk4Solver {
public:
    template<typename System>
    void do_step(System &system, state_type &x, time_type t, time_type dt) {
        adjust_size(x);

        /// We assume here that we are always dealing with complex type
        /// Will change to more generic form in later iterations
        const value_type one = {1.0, 0};
        const time_type dt2 = dt / 2, dt3 = dt/3, dt6 = dt / 6;
        typedef typename operations::template scale_sum2<
            value_type, time_type> scale_sum2;
        typedef typename operations::template scale_sum5<
            value_type, time_type, time_type,
            time_type, time_type> scale_sum5;
        system(x, k1, t, dt);
        algebra::for_each3(x_tmp, x, k1, scale_sum2(one, dt2));
        system(x_tmp, k2, t + dt2, dt2);
        algebra::for_each3(x_tmp, x, k2, scale_sum2(one, dt2));
        system(x_tmp, k3, t + dt, dt);
        algebra::for_each3(x_tmp, x, k3, scale_sum2(one, dt));
        system(x_tmp, k4, t + dt, dt);
        algebra::for_each6(x, x, k1, k2, k3, k4,
                           scale_sum5(one, dt6, dt3, dt3, dt6));
    }    

private:
    state_type x_tmp, k1, k2, k3, k4;
    void adjust_size(const state_type &x) {
        resize(x, x_tmp);
        resize(x, k1);
        resize(x, k2);
        resize(x, k3);
        resize(x, k4);
    }
};