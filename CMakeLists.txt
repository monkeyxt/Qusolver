cmake_minimum_required(VERSION 3.19)
project(qusolver)

set(CMAKE_CXX_STANDARD 20)

find_package(CUDA REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

include_directories(src)

add_library(qusolver MODULE 
        src/gpu.hpp
        src/lindblad.hpp
        src/rk4.hpp
        src/operators.hpp
        src/types.hpp
        src/solver.hpp
        src/pybind_solver.cpp)
        
target_link_libraries(qusolver PUBLIC pybind11::module cudart cuda cublas)