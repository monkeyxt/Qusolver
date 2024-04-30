/******************************************************************************
 * name     gpu.hpp
 * author   Tian Xia (tian.xia.th@dartmouth.edu)
 * date     Apr 23, 2024
 * 
 * This file implements a wrapper around the CUDA backend, and corresponding
 * cuBLAS calls.
******************************************************************************/
#pragma once
#include <concepts>
#include <memory>
#include <vector>
#include <cassert>

#include "cuComplex.h"
#include "cublas_v2.h"
#include "operators.hpp"

/******************************************************************************
 * Macro Definitions for CUDA calls.
******************************************************************************/
#define CUDA_CALL(x)						                \
    do {							                        \
	auto __SOLVER__res = (x);				                \
	if (__SOLVER__res != cudaSuccess) {			            \
        std::cerr << "CUDA Runtime Error" << std::endl;     \
        std::exit(EXIT_FAILURE);                            \
	}							                            \
    } while(0)
    
#define CUDA_ALLOC(p, sz)			                        \
    CUDA_CALL(cudaMalloc((void **)&(p), (sz)));	            \
    if ((p) == nullptr) {			                        \
        std::cerr << "CUDA Alloc Error" << std::endl;       \
        std::exit(EXIT_FAILURE);                            \
    }

#define CUBLAS_CALL(x)						                \
    do {							                        \
	auto __SOLVER__res = (x);				                \
	if (__SOLVER__res != CUBLAS_STATUS_SUCCESS) {		    \
        std::cerr << "cuBLAS Runtime Error" << std::endl;   \
        std::exit(EXIT_FAILURE);                            \
	}							                            \
    } while(0)


using CudaComplexPtr = cuDoubleComplex *;
using CudaComplexConstPtr = const cuDoubleComplex *;

/// Forward declaration
class GPUSolver;

/******************************************************************************
 * Class definition for `GPUSolver`. This class is a wrapper around cuBLAS
 * calls and keeps track of the current cuBLAS context.
******************************************************************************/
class GPUSolver {
public:
    /// Implementation of Device Matrix, a wrapper around raw device matrix
    /// pointer.
    class DeviceMatrix {
    public:
        DeviceMatrix() : rowdim{}, coldim{}, devPtr {} {}
        ~DeviceMatrix() { release(); }

        DeviceMatrix(std::size_t rowdim, std::size_t coldim) {
            allocate(rowdim, coldim);
        }
        DeviceMatrix(const DeviceMatrix& m) : DeviceMatrix(m.rowdim, m.coldim) {
            *this = m;
        }
        DeviceMatrix(DeviceMatrix&& m)  noexcept : rowdim{m.rowdim},
                                                   coldim{m.coldim},
                                                   devPtr{m.devPtr} {
            m.rowdim = 0;
            m.coldim = 0;
            m.devPtr = nullptr;
        }

        DeviceMatrix(const HostMatrix &m) : DeviceMatrix(m.rows(), m.cols()) {
            *this = m;
        }

        DeviceMatrix &operator=(const HostMatrix &m) {
            resize(m.rows(), m.cols());
            if (datasize()) {
                CUDA_CALL(cudaMemcpy(devPtr, m.data(), datasize(),
                                     cudaMemcpyHostToDevice));
            }
            return *this;
        }

        DeviceMatrix &operator=(const DeviceMatrix &m) {
            if (this == &m) { return *this; }
            resize(m.rowdim, m.coldim);
            if (datasize()) {
                CUDA_CALL(cudaMemcpy(devPtr, m.devPtr, datasize(),
                                     cudaMemcpyDeviceToDevice));
            }
            return *this;
        }

        DeviceMatrix &operator=(DeviceMatrix &&rhs)  noexcept {
            if (this == &rhs) { return *this; }
            std::swap(rowdim, rhs.rowdim);
            std::swap(coldim, rhs.coldim);
            std::swap(devPtr, rhs.devPtr);
            return *this;
        }

        cuDoubleComplex operator() (std::size_t rowIdx,
                                    std::size_t colIdx) const {
            cuDoubleComplex res;
            if (!(rowIdx < rowdim && colIdx < coldim)) {
                throw std::exception();
            }
            CUDA_CALL(cudaMemcpy(&res, devPtr + colIdx * rowdim + rowIdx,
                                 sizeof(cuDoubleComplex),
                                 cudaMemcpyDeviceToHost));
            return res;
        }

        void resize(std::size_t newRowdim, std::size_t newColdim) {
            if((rowdim != newRowdim) || coldim != (newColdim)) {
                release();
                allocate(newRowdim, newColdim);
            }
        }

        void allocate(std::size_t newRowdim, std::size_t newColdim) {
            rowdim = newRowdim;
            coldim = newColdim;
            if (datasize()) {
                CUDA_ALLOC(devPtr, datasize());
            } else {
                devPtr = nullptr;
            }
        }

        void release() {
            if (devPtr) {
                cudaFree(devPtr);
            }
            devPtr = nullptr;
            rowdim = 0;
            coldim = 0;
        }

        friend std::ostream& operator<< (std::ostream& stream,
                                         const DeviceMatrix& matrix){
            for (int i = 0; i < matrix.rowdim; i++) {
                for (int j = 0; j < matrix.coldim; j++) {
                    cuDoubleComplex ele = matrix(i, j);
                    std::cout << ele.x << "," << ele.y << "i" << " ";
                }
                std::cout << std::endl;
            }
            return stream;
        }

        DeviceMatrix operator*(const DeviceMatrix &other) const {
            if (coldim != other.rowdim) {
                /// Throw exception here
            }
            DeviceMatrix res(rowdim, other.coldim);
            matMul(res, *this, other);
            return res;
        }

        DeviceMatrix operator*=(const cuDoubleComplex s) const {
            scaleVec(*this, s);
            return *this;
        }

        DeviceMatrix operator*(const cuDoubleComplex s) const {
            scaleVec(*this, s);
            return *this;
        }

        DeviceMatrix operator*=(const double s) const {
            scaleVec(*this, s);
            return *this;
        }

        DeviceMatrix operator*(const double s) const {
            scaleVec(*this, s);
            return *this;
        }

        DeviceMatrix operator+=(const DeviceMatrix &other) const {
            if (coldim != other.coldim || rowdim != other.rowdim) {
                /// Throw exception here
            }
            addVec(*this, other);
            return *this;
        }

        DeviceMatrix operator+(const DeviceMatrix &other) const {
            if (coldim != other.coldim || rowdim != other.rowdim) {
                /// Throw exception here
            }
            DeviceMatrix res(rowdim, coldim);
            addVec(res, *this, other);
            return res;
        }

        DeviceMatrix operator-=(const DeviceMatrix &other) const {
            if (coldim != other.coldim || rowdim != other.rowdim) {
                /// Throw exception here
            }
            subVec(*this, other);
            return *this;
        }

        DeviceMatrix operator-(const DeviceMatrix &other) const {
            if (coldim != other.coldim || rowdim != other.rowdim) {
                /// Throw exception here
            }
            DeviceMatrix res(rowdim, coldim);
            subVec(res, *this, other);
            return res;
        }


        [[nodiscard]] std::size_t rows() const { return rowdim; }
        [[nodiscard]] std::size_t cols() const { return coldim; }
        [[nodiscard]] std::size_t size() const { return rowdim * coldim; }
        [[nodiscard]] std::size_t datasize() const {
            return rowdim * coldim * sizeof(cuDoubleComplex);
        }

    public:
        /// Pointer to the device matrix
        cuDoubleComplex *devPtr{};
        /// Dimensions of the device matrix
        std::size_t rowdim{};
        std::size_t coldim{};
    };

public:
    GPUSolver() = default;
    ~GPUSolver() = default;
    GPUSolver(const GPUSolver&) = delete;
    GPUSolver& operator=(const GPUSolver&) = delete;
    GPUSolver(GPUSolver&&) = delete;
    GPUSolver& operator=(GPUSolver&&) = delete;

    /// Sets the current vector to all zeros
    static void zeroVec(int32_t size, DeviceMatrix &v) {
        CUDA_CALL(cudaMemset(v.devPtr, 0, size * sizeof(cuDoubleComplex)));
    }

    /// Set the current vector to all ones
    static void oneVec(int32_t size, DeviceMatrix &v) {
        std::vector<cuDoubleComplex> ones (size, {1, 0});
        HostMatrix onesHost(ones, size, 1);
        v = onesHost;
    }

    /// Copies device vector v1 into device vector v0
    static void copyVec(int32_t size, DeviceMatrix& v0,
                        const DeviceMatrix& v1) {
        CUDA_CALL(cudaMemcpy(v0.devPtr, v1.devPtr,
                             size * sizeof(cuDoubleComplex),
                             cudaMemcpyDeviceToDevice));
    }

    /// Computes the addition of two device matrices
    static void addVec(const DeviceMatrix& v0,
                       const DeviceMatrix& v1) {
        assert(v0.rows() == v1.rows());
        assert(v0.cols() == v1.cols());
        CUBLAS_CALL(cublas_geam(v0.rows(), v0.cols(), v0.devPtr, {1, 0},
                                v0.devPtr, v1.devPtr, {1, 0}, false, false));
    }

    /// Computes the subtraction of two device matrices
    static void subVec(const DeviceMatrix& v0,
                       const DeviceMatrix& v1) {
        assert(v0.rows() == v1.rows());
        assert(v0.cols() == v1.cols());
        CUBLAS_CALL(cublas_geam(v0.rows(), v0.cols(), v0.devPtr, {1, 0},
                                v0.devPtr, v1.devPtr, {-1, 0}, false, false));
    }

    /// Computes device vector res = v0 + v1
    static void addVec(DeviceMatrix& res,
                       const DeviceMatrix& v0,
                       const DeviceMatrix& v1) {
        assert(res.rows() == v0.rows());
        assert(res.cols() == v0.cols());
        assert(v0.rows() == v1.rows());
        assert(v0.cols() == v1.cols());
        CUBLAS_CALL(cublas_geam(v0.rows(), v0.cols(), res.devPtr, {1, 0},
                                v0.devPtr, v1.devPtr, {1, 0}, false, false));
    }

    /// Computes device vector res = v0 + v1
    static void subVec(DeviceMatrix& res,
                       const DeviceMatrix& v0,
                       const DeviceMatrix& v1) {
        assert(res.rows() == v0.rows());
        assert(res.cols() == v0.cols());
        assert(v0.rows() == v1.rows());
        assert(v0.cols() == v1.cols());
        CUBLAS_CALL(cublas_geam(v0.rows(), v0.cols(), res.devPtr, {1, 0},
                                v0.devPtr, v1.devPtr, {-1, 0}, false, false));
    }

    /// Computes v *= s where s is a complex scalar
    static void scaleVec(const DeviceMatrix& v, cuDoubleComplex s) {
        CUBLAS_CALL(cublas_scal(v.rowdim * v.coldim, s, v.devPtr));
    }

    /// Computes v *= s where s is a real scalar
    static void scaleVec(const DeviceMatrix& v, double s) {
        CUBLAS_CALL(cublas_scal(v.rowdim * v.coldim, {s, 0}, v.devPtr));
    }


    /// Computes the matrix multiplication res = mat0 * mat1
    static void matMul (DeviceMatrix &res, const DeviceMatrix &mat0,
                        const DeviceMatrix &mat1) {
        assert(res.rows() == mat0.rows());
        assert(mat0.cols() == mat1.rows());
        assert(mat1.cols() == res.cols());
        CUBLAS_CALL(cublas_gemm(res.rowdim, mat0.coldim, mat1.coldim,
                                res.devPtr, {1, 0}, mat0.devPtr, mat1.devPtr,
                                {}, false, false));
    }

    /// Computes the trace of the matrix
    static cuDoubleComplex matTr(const DeviceMatrix &mat){
        DeviceMatrix ones {mat.rows(), mat.cols()};
        oneVec(mat.rows() * mat.cols(), ones);
        cuDoubleComplex res;
        CUBLAS_CALL(cublas_dotc(mat.cols(), res, mat.devPtr, ones.devPtr,
                                mat.cols() + 1, 0));
        return res;
    }

    /// Computes the complex transpose the the matrix
    static void hermitian(DeviceMatrix& v0, const DeviceMatrix& v1) {
        assert(v0.rows() == v1.cols());
        assert(v0.cols() == v1.rows());
        CUBLAS_CALL(cublas_geam(v0.rows(), v0.cols(), v0.devPtr, {0, 0},
                                v0.devPtr, v1.devPtr, {1, 0}, false, true));
    }

    /// GPU context, keeps track of the cuBLAS handle.
    struct GPUContext {
        GPUContext () {
            CUBLAS_CALL(cublasCreate(&handle));
        }
        cublasHandle_t handle;
    };

    /// Context for managing GPU status
    inline static GPUContext context;

private:

    /// cuBLAS calls
    static cublasStatus_t cublas_gemm(std::size_t rowdim, std::size_t coldimA,
                                      std::size_t coldimB,
                                      cuDoubleComplex *res,
                                      const cuDoubleComplex alpha,
                                      const cuDoubleComplex *A,
                                      const cuDoubleComplex *B,
                                      const cuDoubleComplex beta,
                                      bool adjointA = false,
                                      bool adjointB = false) {
        return cublasZgemm_v2(context.handle,
                              adjointA ? CUBLAS_OP_C : CUBLAS_OP_N,
                              adjointB ? CUBLAS_OP_C : CUBLAS_OP_N,
                              static_cast<int64_t>(rowdim),
                              static_cast<int64_t>(coldimB),
                              static_cast<int64_t>(coldimA),
                              static_cast<CudaComplexConstPtr>(&alpha),
                              static_cast<CudaComplexConstPtr>(A),
                              static_cast<int64_t>(rowdim),
                              static_cast<CudaComplexConstPtr>(B),
                              static_cast<int64_t>(coldimA),
                              static_cast<CudaComplexConstPtr>(&beta),
                              res,
                              static_cast<int64_t>(rowdim));
    }

    static cublasStatus_t cublas_axpy(std::size_t dim, cuDoubleComplex s,
                                      const cuDoubleComplex *x,
                                      cuDoubleComplex* y) {
        return cublasZaxpy_v2(context.handle, dim,
                              static_cast<CudaComplexConstPtr>(&s),
                              static_cast<CudaComplexConstPtr>(x), 1,
                              static_cast<CudaComplexPtr>(y), 1);
    }

    /// cuBLAS scaling handle
    static cublasStatus_t cublas_scal(std::size_t dim, cuDoubleComplex s,
                                      cuDoubleComplex* x) {
        return cublasZscal_v2(context.handle, dim,
                              static_cast<CudaComplexConstPtr>(&s),
                              static_cast<CudaComplexPtr>(x), 1);
    }

    /// cublas addition handle
    static cublasStatus_t cublas_geam(std::size_t rowdim, std::size_t coldimA,
                                      cuDoubleComplex *res,
                                      const cuDoubleComplex alpha,
                                      const cuDoubleComplex *A,
                                      const cuDoubleComplex *B,
                                      const cuDoubleComplex beta,
                                      bool adjointA = false,
                                      bool adjointB = false) {
        return cublasZgeam(context.handle,
                           adjointA ? CUBLAS_OP_C : CUBLAS_OP_N,
                           adjointB ? CUBLAS_OP_C : CUBLAS_OP_N,
                           static_cast<int64_t>(rowdim),
                           static_cast<int64_t>(coldimA),
                           static_cast<CudaComplexConstPtr>(&alpha),
                           static_cast<CudaComplexConstPtr>(A),
                           static_cast<int64_t>(rowdim),
                           static_cast<CudaComplexConstPtr>(&beta),
                           static_cast<CudaComplexConstPtr>(B),
                           static_cast<int64_t>(rowdim),
                           res,
                           static_cast<int64_t>(rowdim));
    }

    /// Calculate the dot product
    static cublasStatus_t cublas_dotc(std::size_t n,
                                      cuDoubleComplex &res,
                                      const cuDoubleComplex *x,
                                      const cuDoubleComplex *y,
                                      int incx,
                                      int incy) {
        return cublasZdotc_v2(context.handle, n,
                              static_cast<CudaComplexConstPtr>(x), incx,
                              static_cast<CudaComplexConstPtr>(y), incy,
                              static_cast<CudaComplexPtr>(&res));
    }
};