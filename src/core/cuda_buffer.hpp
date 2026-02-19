#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <utility>
#include "util/cuda_check.hpp"

namespace bytron {

    /// <summary>
    /// RAII wrapper for a cudaMalloc'd device buffer.
    /// Move-only; automatically frees on destruction.
    /// </summary>
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
                    ptr_ = std::exchange(other.ptr_, nullptr);
                    size_ = std::exchange(other.size_, 0);
                }
                return *this;
            }

            CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
            CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

            [[nodiscard]] void* ptr() const noexcept { return ptr_; }
            [[nodiscard]] std::size_t size() const noexcept { return size_; }
            [[nodiscard]] bool empty() const noexcept { return ptr_ == nullptr; }

            template <typename T>
            [[nodiscard]] T* as() const noexcept { return static_cast<T*>(ptr_); }

            /// <summary>
            /// Async host-to-device copy into this buffer.
            /// </summary>
            void copy_from_host_async(const void* src, std::size_t bytes, cudaStream_t stream) {
                CUDA_CHECK(cudaMemcpyAsync(ptr_, src, bytes, cudaMemcpyHostToDevice, stream));
            }

            /// <summary>
            /// Async device-to-host copy out of this buffer.
            /// </summary>
            void copy_to_host_async(void* dst, std::size_t bytes, cudaStream_t stream) const {
                CUDA_CHECK(cudaMemcpyAsync(dst, ptr_, bytes, cudaMemcpyDeviceToHost, stream));
            }

        private:
            void* ptr_ = nullptr;
            std::size_t size_ = 0;
    };

    /// <summary>
    /// RAII wrapper for a cudaMallocHost'd pinned buffer.
    /// Move-only; automatically frees on destruction.
    /// </summary>
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
                    ptr_ = std::exchange(other.ptr_, nullptr);
                    size_ = std::exchange(other.size_, 0);
                }
                return *this;
            }

            CudaPinnedBuffer(const CudaPinnedBuffer&) = delete;
            CudaPinnedBuffer& operator=(const CudaPinnedBuffer&) = delete;

            [[nodiscard]] void* ptr() const noexcept { return ptr_; }
            [[nodiscard]] std::size_t size() const noexcept { return size_; }

            template <typename T>
            [[nodiscard]] T* as() const noexcept { return static_cast<T*>(ptr_); }

        private:
            void* ptr_ = nullptr;
            std::size_t size_ = 0;
    };
}
