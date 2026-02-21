/**
 * @file desktop_capture.hpp
 * @brief DXGI Desktop Duplication capture with D3D11-CUDA interop.
 */

#pragma once

#include "core/cuda_buffer.hpp"

#define NOMINMAX
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

namespace bytron {

    /** @brief Result of a frame acquisition attempt. */
    enum class AcquireResult {
        kNewFrame,   ///< Successfully acquired a new desktop frame
        kTimeout,    ///< No new frame within the timeout period
        kError,      ///< Unrecoverable error; capture is unusable
        kAccessLost  ///< Duplication lost (display mode change); call reacquire()
    };

    /**
     * @brief Captures the desktop via DXGI Output Duplication with zero-copy CUDA interop.
     *
     * Interop path: DXGI AcquireNextFrame -> D3D11 CopyResource to shared staging
     * texture -> cudaGraphicsD3D11RegisterResource -> cudaMemcpy2DFromArrayAsync
     * to destination CudaDeviceBuffer. Frames are BGRA (B8G8R8A8_UNORM).
     *
     * @note Thread safety: Not thread-safe. Call all methods from the capture thread.
     *
     * @warning Only one DXGI output duplication can exist per output per process.
     */
    class DesktopCapture {
    public:
        /**
         * @brief Initialize D3D11 device and DXGI output duplication.
         * @param monitor_index DXGI output index (0 = primary monitor).
         * @throws std::runtime_error On D3D11/DXGI/CUDA initialization failure.
         */
        explicit DesktopCapture(int monitor_index = 0);
        ~DesktopCapture();

        DesktopCapture(const DesktopCapture&) = delete;
        DesktopCapture& operator=(const DesktopCapture&) = delete;

        /**
         * @brief Acquire the next desktop frame into a device buffer.
         *
         * @param [out] dst        Device buffer receiving BGRA frame data.
         * @param [in]  stream     CUDA stream for async copy operations.
         * @param [in]  timeout_ms DXGI wait timeout (~16ms = one 60 Hz frame).
         * @return AcquireResult indicating success, timeout, or error.
         * @pre dst.size() >= width() * height() * 4
         */
        [[nodiscard]] AcquireResult acquire_frame_blocking(CudaDeviceBuffer& dst,
                                                          cudaStream_t stream,
                                                          UINT timeout_ms = 16);

        /**
         * @brief Re-acquire DXGI output duplication after access loss.
         * @return true if duplication was re-acquired successfully.
         */
        [[nodiscard]] bool reacquire();

        [[nodiscard]] bool is_open() const noexcept { return open_; }
        [[nodiscard]] int  width()   const noexcept { return width_; }   ///< Capture width in pixels
        [[nodiscard]] int  height()  const noexcept { return height_; }  ///< Capture height in pixels
        [[nodiscard]] int  pitch()   const noexcept { return width_ * 4; } ///< Row stride in bytes

    private:
        template <typename T>
        using ComPtr = Microsoft::WRL::ComPtr<T>;

        ComPtr<ID3D11Device>           d3d_device_;
        ComPtr<ID3D11DeviceContext>    d3d_context_;
        ComPtr<IDXGIOutputDuplication> duplication_;
        ComPtr<ID3D11Texture2D>        staging_tex_;

        int monitor_index_ = 0;
        cudaGraphicsResource_t cuda_resource_ = nullptr;

        int  width_  = 0;
        int  height_ = 0;
        bool open_   = false;
    };

}
