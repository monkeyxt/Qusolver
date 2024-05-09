cmake_minimum_required(VERSION 3.19)
project(qusolver)

set(CMAKE_CXX_STANDARD 20)

include_directories(src)

add_executable(qusolver
        src/gpu.hpp
        src/lindblad.hpp
        src/rk4.hpp
        src/operators.hpp
        src/types.hpp
        src/qusolver.hpp
        solver.cpp)

find_package(CUDA REQUIRED)
target_link_libraries(qusolver PRIVATE cudart cuda cublas)