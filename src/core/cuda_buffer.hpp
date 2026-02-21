/**
 * @file cuda_buffer.hpp
 * @brief RAII wrappers for CUDA device and pinned host memory.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <utility>
#include "util/cuda_check.hpp"

namespace bytron {

    /**
     * @brief RAII wrapper for a CUDA device (GPU) memory allocation.
     *
     * Owns a single cudaMalloc'd buffer. Move-only; the moved-from
     * object becomes empty (ptr() == nullptr). Destructor calls cudaFree.
     *
     * @note Thread safety: Not thread-safe. Access from one stream at a
     *       time or synchronize externally.
     */
    class CudaDeviceBuffer {
    public:
        CudaDeviceBuffer() = default;

        explicit CudaDeviceBuffer(std::size_t bytes) : size_(bytes) {
            CUDA_CHECK(cudaMalloc(&ptr_, bytes));
        }

        ~CudaDeviceBuffer() {
            if (ptr_) cudaFree(ptr_);
        }

        CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept
            : ptr_(std::exchange(other.ptr_, nullptr))
            , size_(std::exchange(other.size_, 0)) {}

        CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept {
            if (this != &other) {
                if (ptr_) cudaFree(ptr_);
                ptr_  = std::exchange(other.ptr_, nullptr);
                size_ = std::exchange(other.size_, 0);
            }
            return *this;
        }

        CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
        CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

        [[nodiscard]] void*       ptr()   const noexcept { return ptr_; }
        [[nodiscard]] std::size_t size()  const noexcept { return size_; }
        [[nodiscard]] bool        empty() const noexcept { return ptr_ == nullptr; }

        /** @brief Reinterpret the buffer pointer as a typed pointer. */
        template <typename T>
        [[nodiscard]] T* as() const noexcept { return static_cast<T*>(ptr_); }

        /**
         * @brief Async copy from host memory into this device buffer.
         * @pre @p src should be page-locked (cudaMallocHost) for true async behavior.
         */
        void copy_from_host_async(const void* src, std::size_t bytes, cudaStream_t stream) {
            CUDA_CHECK(cudaMemcpyAsync(ptr_, src, bytes, cudaMemcpyHostToDevice, stream));
        }

        /**
         * @brief Async copy from this device buffer to host memory.
         * @pre @p dst should be page-locked for true async behavior.
         */
        void copy_to_host_async(void* dst, std::size_t bytes, cudaStream_t stream) const {
            CUDA_CHECK(cudaMemcpyAsync(dst, ptr_, bytes, cudaMemcpyDeviceToHost, stream));
        }

    private:
        void*       ptr_  = nullptr;
        std::size_t size_ = 0;
    };

    /**
     * @brief RAII wrapper for page-locked (pinned) host memory.
     *
     * Allocated via cudaMallocHost for overlap of H2D/D2H transfers
     * with GPU compute. Move-only; destructor calls cudaFreeHost.
     *
     * @note Thread safety: Same as CudaDeviceBuffer.
     */
    class CudaPinnedBuffer {
    public:
        CudaPinnedBuffer() = default;

        explicit CudaPinnedBuffer(std::size_t bytes) : size_(bytes) {
            CUDA_CHECK(cudaMallocHost(&ptr_, bytes));
        }

        ~CudaPinnedBuffer() {
            if (ptr_) cudaFreeHost(ptr_);
        }

        CudaPinnedBuffer(CudaPinnedBuffer&& other) noexcept
            : ptr_(std::exchange(other.ptr_, nullptr))
            , size_(std::exchange(other.size_, 0)) {}

        CudaPinnedBuffer& operator=(CudaPinnedBuffer&& other) noexcept {
            if (this != &other) {
                if (ptr_) cudaFreeHost(ptr_);
                ptr_  = std::exchange(other.ptr_, nullptr);
                size_ = std::exchange(other.size_, 0);
            }
            return *this;
        }

        CudaPinnedBuffer(const CudaPinnedBuffer&) = delete;
        CudaPinnedBuffer& operator=(const CudaPinnedBuffer&) = delete;

        [[nodiscard]] void*       ptr()  const noexcept { return ptr_; }
        [[nodiscard]] std::size_t size() const noexcept { return size_; }

        template <typename T>
        [[nodiscard]] T* as() const noexcept { return static_cast<T*>(ptr_); }

    private:
        void*       ptr_  = nullptr;
        std::size_t size_ = 0;
    };

}
