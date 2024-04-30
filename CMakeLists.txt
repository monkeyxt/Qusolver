cmake_minimum_required(VERSION 3.19)
project(cuSolver)

set(CMAKE_CXX_STANDARD 17)

include_directories(src)

add_executable(cuSolver
        src/gpu.hpp
        src/lindblad.hpp
        src/rk4.hpp
        src/operators.hpp
        src/solver.cpp)

find_package(CUDA REQUIRED)
target_link_libraries(cuSolver PRIVATE cudart cuda cublas)