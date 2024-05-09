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
 * Header for solving open quantum systems
******************************************************************************/
namespace Solver {

    /// Define the floating type to use

}