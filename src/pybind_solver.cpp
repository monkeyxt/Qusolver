/******************************************************************************
 * name     pybind_solver.cpp
 * author   Tian Xia (tian.xia.th@dartmouth.edu)
 * date     Jun 5, 2024
 * 
 * Pybind interface for the open dynamics solver
******************************************************************************/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>
#include <pybind11/complex.h>

#include "solver.hpp"     /// For solver

namespace py = pybind11;

namespace qusolver {
PYBIND11_MODULE(rk4Solver, m) {

    /// Bindings to `HostMatrix`
    py::class_<HostMatrix<float>>(m, "HostMatrix", py::buffer_protocol())
        .def(py::init([](py::buffer b) {
            py::buffer_info info = b.request();

            /// Validation checks
            if (info.format 
                != py::format_descriptor<std::complex<float>>::format()) {
                throw std::runtime_error("Incompatible format: expected a complex float array!");
            }

            if (info.ndim !=2) {
                throw std::runtime_error("Incompatible buffer dimension!");
            }

            auto ptr = static_cast<std::vector<std::complex<float>>*>(info.ptr);
            return HostMatrix<float>(ptr, 
                                     static_cast<std::size_t>(info.shape[0]), 
                                     static_cast<std::size_t>(info.shape[1]));
    }));

    py::class_<HostMatrix<double>>(m, "HostMatrix", py::buffer_protocol())
        .def(py::init([](py::buffer b) {
            py::buffer_info info = b.request();

            /// Validation checks
            if (info.format 
                != py::format_descriptor<std::complex<double>>::format()) {
                throw std::runtime_error("Incompatible format: expected a complex double array!");
            }

            if (info.ndim !=2) {
                throw std::runtime_error("Incompatible buffer dimension!");
            }

            auto ptr = static_cast<std::vector<std::complex<double>>*>(info.ptr);
            return HostMatrix<double>(ptr, 
                                     static_cast<std::size_t>(info.shape[0]), 
                                     static_cast<std::size_t>(info.shape[1]));
    }));
}
}