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

#include "cublas_v2.h"
#include "cuComplex.h"
#include "cuda.h"
#include "cuda_runtime.h"

#include "operators.hpp"
#include "types.hpp"

/******************************************************************************
 * Macro Definitions for CUDA calls.
 ******************************************************************************/
#define CUDA_CALL(x)                                        \
    do                                                      \
    {                                                       \
        auto __SOLVER__res = (x);                           \
        if (__SOLVER__res != cudaSuccess)                   \
        {                                                   \
            std::cerr << "CUDA Runtime Error" << std::endl; \
            std::exit(EXIT_FAILURE);                        \
        }                                                   \
    } while (0)

#define CUDA_ALLOC(p, sz)                             \
    CUDA_CALL(cudaMalloc((void **)&(p), (sz)));       \
    if ((p) == nullptr)                               \
    {                                                 \
        std::cerr << "CUDA Alloc Error" << std::endl; \
        std::exit(EXIT_FAILURE);                      \
    }

#define CUBLAS_CALL(x)                                        \
    do                                                        \
    {                                                         \
        auto __SOLVER__res = (x);                             \
        if (__SOLVER__res != CUBLAS_STATUS_SUCCESS)           \
        {                                                     \
            std::cerr << "cuBLAS Runtime Error" << std::endl; \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)

/// Forward declaration
class GPUSolver;

/******************************************************************************
 * Define cuBLAS call types for different precisions.
 ******************************************************************************/
template <ScalarType T>
struct BLASType;

template <>
struct BLASType<float> {
    static constexpr auto cublas_axpy = cublasCaxpy;
    static constexpr auto cublas_scal = cublasSscal;
    static constexpr auto cublas_cscal = cublasCscal;
    static constexpr auto cublas_dotc = cublasCdotc;
    static constexpr auto cublas_gemv = cublasCgemv;
    static constexpr auto cublas_gemm = cublasCgemm;
    static constexpr auto cublas_geam = cublasCgeam;
};

template <>
struct BLASType<double> {
    static constexpr auto cublas_axpy = cublasZaxpy;
    static constexpr auto cublas_scal = cublasDscal;
    static constexpr auto cublas_cscal = cublasZscal;
    static constexpr auto cublas_dotc = cublasZdotc;
    static constexpr auto cublas_gemv = cublasZgemv;
    static constexpr auto cublas_gemm = cublasZgemm;
    static constexpr auto cublas_geam = cublasZgeam;
};

/******************************************************************************
 * Class definition for `GPUSolver`. This class is a wrapper around cuBLAS
 * calls and keeps track of the current cuBLAS context.
 ******************************************************************************/
class GPUSolver {
public:
    /// Implementation of DeviceMatrix
    template <ScalarType T = DefaultPrecisionType>
    class DeviceMatrix {
    public:
        using Matrix = HostMatrix<T>;
        DeviceMatrix() : rowdim{}, coldim{}, devPtr{} {}
        ~DeviceMatrix() { release(); }

        DeviceMatrix(std::size_t rowdim, std::size_t coldim) {
            allocate(rowdim, coldim);
        }
        DeviceMatrix(const DeviceMatrix &m) : DeviceMatrix(m.rowdim, m.coldim) {
            *this = m;
        }
        DeviceMatrix(DeviceMatrix &&m) noexcept : rowdim{m.rowdim},
                                                  coldim{m.coldim},
                                                  devPtr{m.devPtr} {
            m.rowdim = 0;
            m.coldim = 0;
            m.devPtr = nullptr;
        }

        DeviceMatrix(const Matrix &m) : DeviceMatrix(m.rows(), m.cols()) {
            *this = m;
        }

        DeviceMatrix &operator=(const Matrix &m) {
            resize(m.rows(), m.cols());
            if (datasize()) {
                CUDA_CALL(cudaMemcpy(devPtr, m.data(), datasize(),
                                     cudaMemcpyHostToDevice));
            }
            return *this;
        }

        DeviceMatrix &operator=(const DeviceMatrix &m) {
            if (this == &m) {
                return *this;
            }
            resize(m.rowdim, m.coldim);
            if (datasize()) {
                CUDA_CALL(cudaMemcpy(devPtr, m.devPtr, datasize(),
                                     cudaMemcpyDeviceToDevice));
            }
            return *this;
        }

        DeviceMatrix &operator=(DeviceMatrix &&rhs) noexcept {
            if (this == &rhs) {
                return *this;
            }
            std::swap(rowdim, rhs.rowdim);
            std::swap(coldim, rhs.coldim);
            std::swap(devPtr, rhs.devPtr);
            return *this;
        }

        CudaComplexType<T> operator()(std::size_t rowIdx,
                                   std::size_t colIdx) const {
            CudaComplexType<T> res;
            if (!(rowIdx < rowdim && colIdx < coldim)) {
                throw std::exception();
            }
            CUDA_CALL(cudaMemcpy(&res, devPtr + colIdx * rowdim + rowIdx,
                                 sizeof(CudaComplexType<T>),
                                 cudaMemcpyDeviceToHost));
            return res;
        }

        void resize(std::size_t newRowdim, std::size_t newColdim) {
            if ((rowdim != newRowdim) || coldim != (newColdim)) {
                release();
                allocate(newRowdim, newColdim);
            }
        }

        void allocate(std::size_t newRowdim, std::size_t newColdim) {
            rowdim = newRowdim;
            coldim = newColdim;
            if (datasize()) {
                CUDA_ALLOC(devPtr, datasize());
            }
            else {
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

        /// Handy print overload
        friend std::ostream &operator<<(std::ostream &stream,
                                        const DeviceMatrix &matrix) {
            for (std::size_t i = 0; i < matrix.rowdim; i++) {
                for (std::size_t j = 0; j < matrix.coldim; j++) {
                    CudaComplexType<T> ele = matrix(i, j);
                    std::cout << ele << " ";
                }
                std::cout << std::endl;
            }
            return stream;
        }

        DeviceMatrix operator*(const DeviceMatrix &other) const {
            if (coldim != other.rowdim) {
                throw std::invalid_argument( "Matrix index out of bounds!" );
            }
            DeviceMatrix res(rowdim, other.coldim);
            matMul(res, *this, other);
            return res;
        }

        DeviceMatrix operator*=(const CudaComplexType<T> s) const {
            scaleVec(*this, s);
            return *this;
        }

        DeviceMatrix operator*(const CudaComplexType<T> s) const {
            scaleVec(*this, s);
            return *this;
        }

        DeviceMatrix operator*=(const T s) const {
            scaleVec(*this, s);
            return *this;
        }

        DeviceMatrix operator*(const T s) const {
            scaleVec(*this, s);
            return *this;
        }

        DeviceMatrix operator+=(const DeviceMatrix &other) const {
            if (coldim != other.coldim || rowdim != other.rowdim) {
                throw std::invalid_argument( "Matrix index out of bounds!" );
            }
            addVec(*this, other);
            return *this;
        }

        DeviceMatrix operator+(const DeviceMatrix &other) const {
            if (coldim != other.coldim || rowdim != other.rowdim) {
                throw std::invalid_argument( "Matrix index out of bounds!" );
            }
            DeviceMatrix res(rowdim, coldim);
            addVec(res, *this, other);
            return res;
        }

        DeviceMatrix operator-=(const DeviceMatrix &other) const {
            if (coldim != other.coldim || rowdim != other.rowdim) {
                throw std::invalid_argument( "Matrix index out of bounds!" );
            }
            subVec(*this, other);
            return *this;
        }

        DeviceMatrix operator-(const DeviceMatrix &other) const {
            if (coldim != other.coldim || rowdim != other.rowdim) {
                throw std::invalid_argument( "Matrix index out of bounds!" );
            }
            DeviceMatrix res(rowdim, coldim);
            subVec(res, *this, other);
            return res;
        }

        [[nodiscard]] std::size_t rows() const { return rowdim; }
        [[nodiscard]] std::size_t cols() const { return coldim; }
        [[nodiscard]] std::size_t size() const { return rowdim * coldim; }
        [[nodiscard]] std::size_t datasize() const {
            return rowdim * coldim * sizeof(CudaComplexType<T>);
        }

    public:
        /// Pointer to the device matrix
        CudaComplexType<T> *devPtr{};
        /// Dimensions of the device matrix
        std::size_t rowdim{};
        std::size_t coldim{};
    };

public:
    GPUSolver() = default;
    ~GPUSolver() = default;
    GPUSolver(const GPUSolver &) = delete;
    GPUSolver &operator=(const GPUSolver &) = delete;
    GPUSolver(GPUSolver &&) = delete;
    GPUSolver &operator=(GPUSolver &&) = delete;

    /// Sets the current vector to all zeros
    template <ScalarType T>
    static void zeroVec(int32_t size, DeviceMatrix<T> &v) {
        CUDA_CALL(cudaMemset(v.devPtr, 0, size * sizeof(CudaComplexType<T>)));
    }

    /// Set the current vector to all ones
    template <ScalarType T>
    static void oneVec(int32_t size, DeviceMatrix<T> &v) {
        std::vector<CudaComplexType<T>> ones(size, {1, 0});
        HostMatrix onesHost(ones, size, 1);
        v = onesHost;
    }

    /// Copies device vector v1 into device vector v0
    template <ScalarType T>
    static void copyVec(int32_t size, DeviceMatrix<T> &v0,
                        const DeviceMatrix<T> &v1) {
        CUDA_CALL(cudaMemcpy(v0.devPtr, v1.devPtr,
                             size * sizeof(CudaComplexType<T>),
                             cudaMemcpyDeviceToDevice));
    }

    /// Computes the addition of two device matrices
    template <ScalarType T>
    static void addVec(const DeviceMatrix<T> &v0, const DeviceMatrix<T> &v1) {
        assert(v0.rows() == v1.rows());
        assert(v0.cols() == v1.cols());
        CUBLAS_CALL(cublas_geam<T>(v0.rows(), v0.cols(), v0.devPtr, {1, 0},
                                   v0.devPtr, v1.devPtr, {1, 0}, false, false));
    }

    /// Computes the subtraction of two device matrices
    template <ScalarType T>
    static void subVec(const DeviceMatrix<T> &v0, const DeviceMatrix<T> &v1) {
        assert(v0.rows() == v1.rows());
        assert(v0.cols() == v1.cols());
        CUBLAS_CALL(cublas_geam<T>(v0.rows(), v0.cols(), v0.devPtr, {1, 0},
                                   v0.devPtr, v1.devPtr, {-1, 0}, false, false));
    }

    /// Computes device vector res = v0 + v1
    template <ScalarType T>
    static void addVec(DeviceMatrix<T> &res,
                       const DeviceMatrix<T> &v0,
                       const DeviceMatrix<T> &v1) {
        assert(res.rows() == v0.rows());
        assert(res.cols() == v0.cols());
        assert(v0.rows() == v1.rows());
        assert(v0.cols() == v1.cols());
        CUBLAS_CALL(cublas_geam<T>(v0.rows(), v0.cols(), res.devPtr, {1, 0},
                                   v0.devPtr, v1.devPtr, {1, 0}, false, false));
    }

    /// Computes device vector res = v0 + v1
    template <ScalarType T>
    static void subVec(DeviceMatrix<T> &res,
                       const DeviceMatrix<T> &v0,
                       const DeviceMatrix<T> &v1) {
        assert(res.rows() == v0.rows());
        assert(res.cols() == v0.cols());
        assert(v0.rows() == v1.rows());
        assert(v0.cols() == v1.cols());
        CUBLAS_CALL(cublas_geam<T>(v0.rows(), v0.cols(), res.devPtr, {1, 0},
                                   v0.devPtr, v1.devPtr, {-1, 0}, false, false));
    }

    /// Computes v *= s where s is a complex scalar
    template <ScalarType T>
    static void scaleVec(const DeviceMatrix<T> &v, CudaComplexType<T> s) {
        CUBLAS_CALL(cublas_scal<T>(v.rowdim * v.coldim, s, v.devPtr));
    }

    /// Computes v *= s where s is a real scalar
    template <ScalarType T>
    static void scaleVec(const DeviceMatrix<T> &v, T s) {
        CUBLAS_CALL(cublas_scal<T>(v.rowdim * v.coldim, {s, 0}, v.devPtr));
    }

    /// Computes the matrix multiplication res = mat0 * mat1
    template <ScalarType T>
    static void matMul(DeviceMatrix<T> &res, const DeviceMatrix<T> &mat0,
                       const DeviceMatrix<T> &mat1) {
        assert(res.rows() == mat0.rows());
        assert(mat0.cols() == mat1.rows());
        assert(mat1.cols() == res.cols());
        CUBLAS_CALL(cublas_gemm<T>(res.rowdim, mat0.coldim, mat1.coldim,
                                   res.devPtr, {1, 0}, mat0.devPtr, mat1.devPtr,
                                   {}, false, false));
    }

    /// Computes the trace of the matrix
    template <ScalarType T>
    static cuDoubleComplex matTr(const DeviceMatrix<T> &mat) {
        DeviceMatrix ones{mat.rows(), mat.cols()};
        oneVec(mat.rows() * mat.cols(), ones);
        cuDoubleComplex res;
        CUBLAS_CALL(cublas_dotc<T>(mat.cols(), res, mat.devPtr, ones.devPtr,
                                   mat.cols() + 1, 0));
        return res;
    }

    /// Computes the complex transpose the the matrix
    template <ScalarType T>
    static void hermitian(DeviceMatrix<T> &v0, const DeviceMatrix<T> &v1) {
        assert(v0.rows() == v1.cols());
        assert(v0.cols() == v1.rows());
        CUBLAS_CALL(cublas_geam<T>(v0.rows(), v0.cols(), v0.devPtr, {0, 0},
                                   v0.devPtr, v1.devPtr, {1, 0}, false, true));
    }

    /// GPU context, keeps track of the cuBLAS handle.
    struct GPUContext {
        GPUContext() {
            CUBLAS_CALL(cublasCreate(&handle));
        }
        ~GPUContext() = default;
        cublasHandle_t handle;
    };

    /// Context for managing GPU status
    inline static GPUContext context;

private:
    /// cuBLAS matrix multiplication handle
    template <ScalarType T>
    static cublasStatus_t cublas_gemm(std::size_t rowdim, std::size_t coldimA,
                                      std::size_t coldimB,
                                      CudaComplexPtr<T> res,
                                      const CudaComplexType<T> alpha,
                                      const CudaComplexPtr<T> A,
                                      const CudaComplexPtr<T> B,
                                      const CudaComplexType<T> beta,
                                      bool adjointA = false,
                                      bool adjointB = false) {
        return BLASType<T>::cublas_gemm(context.handle,
                                        adjointA ? CUBLAS_OP_C : CUBLAS_OP_N,
                                        adjointB ? CUBLAS_OP_C : CUBLAS_OP_N,
                                        static_cast<int64_t>(rowdim),
                                        static_cast<int64_t>(coldimB),
                                        static_cast<int64_t>(coldimA),
                                        static_cast<CudaComplexConstPtr<T>>(&alpha),
                                        static_cast<CudaComplexConstPtr<T>>(A),
                                        static_cast<int64_t>(rowdim),
                                        static_cast<CudaComplexConstPtr<T>>(B),
                                        static_cast<int64_t>(coldimA),
                                        static_cast<CudaComplexConstPtr<T>>(&beta),
                                        res,
                                        static_cast<int64_t>(rowdim));
    }

    template <ScalarType T>
    static cublasStatus_t cublas_axpy(std::size_t dim, CudaComplexType<T> s,
                                      const CudaComplexType<T> *x,
                                      CudaComplexType<T> *y) {
        return BLASType<T>::cublas_axpy(context.handle, dim,
                                        static_cast<CudaComplexConstPtr<T>>(&s),
                                        static_cast<CudaComplexConstPtr<T>>(x), 1,
                                        static_cast<CudaComplexPtr<T>>(y), 1);
    }

    /// cuBLAS scaling handle
    template <ScalarType T>
    static cublasStatus_t cublas_scal(std::size_t dim, CudaComplexType<T> s,
                                      CudaComplexType<T> *x) {
        return BLASType<T>::cublas_cscal(context.handle, dim,
                                         static_cast<CudaComplexConstPtr<T>>(&s),
                                         static_cast<CudaComplexPtr<T>>(x), 1);
    }

    /// cublas addition handle
    template <ScalarType T>
    static cublasStatus_t cublas_geam(std::size_t rowdim, std::size_t coldimA,
                                      CudaComplexType<T> *res,
                                      const CudaComplexType<T> alpha,
                                      const CudaComplexType<T> *A,
                                      const CudaComplexType<T> *B,
                                      const CudaComplexType<T> beta,
                                      bool adjointA = false,
                                      bool adjointB = false) {
        return BLASType<T>::cublas_geam(context.handle,
                                        adjointA ? CUBLAS_OP_C : CUBLAS_OP_N,
                                        adjointB ? CUBLAS_OP_C : CUBLAS_OP_N,
                                        static_cast<int64_t>(rowdim),
                                        static_cast<int64_t>(coldimA),
                                        static_cast<CudaComplexConstPtr<T>>(&alpha),
                                        static_cast<CudaComplexConstPtr<T>>(A),
                                        static_cast<int64_t>(rowdim),
                                        static_cast<CudaComplexConstPtr<T>>(&beta),
                                        static_cast<CudaComplexConstPtr<T>>(B),
                                        static_cast<int64_t>(rowdim),
                                        res,
                                        static_cast<int64_t>(rowdim));
    }

    /// Calculate the dot product
    template <ScalarType T>
    static cublasStatus_t cublas_dotc(std::size_t n,
                                      CudaComplexType<T> &res,
                                      const CudaComplexType<T> *x,
                                      const CudaComplexType<T> *y,
                                      int incx,
                                      int incy) {
        return BLASType<T>::cublas_dotc(context.handle, n,
                                        static_cast<CudaComplexConstPtr<T>>(x),
                                        incx,
                                        static_cast<CudaComplexConstPtr<T>>(y),
                                        incy,
                                        static_cast<CudaComplexPtr<T>>(&res));
    }
};