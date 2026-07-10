/**
 * @file desktop_capture.hpp
 * @brief DXGI Desktop Duplication capture with D3D11-CUDA interop.
 */

#pragma once

#include "capture/capture_source.hpp"

#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <cuda_d3d11_interop.h>

namespace bytron {

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
    class DesktopCapture final : public ICaptureSource {
    public:
        /**
         * @brief Initialize D3D11 device and DXGI output duplication.
         * @param monitor_index DXGI output index (0 = primary monitor).
         * @throws std::runtime_error On D3D11/DXGI/CUDA initialization failure.
         */
        explicit DesktopCapture(int monitor_index = 0);
        ~DesktopCapture() override;

        DesktopCapture(const DesktopCapture&) = delete;
        DesktopCapture& operator=(const DesktopCapture&) = delete;

        [[nodiscard]] AcquireResult acquire_frame(CudaDeviceBuffer& dst,
                                                  cudaStream_t stream,
                                                  unsigned timeout_ms) override;

        [[nodiscard]] bool reacquire() override;

        [[nodiscard]] bool is_open() const noexcept override { return open_; }
        [[nodiscard]] int  width()   const noexcept override { return width_; }
        [[nodiscard]] int  height()  const noexcept override { return height_; }
        [[nodiscard]] int  pitch()   const noexcept override { return width_ * 4; }
        [[nodiscard]] const char* backend_name() const noexcept override {
            return "DXGI Desktop Duplication";
        }

    private:
        template <typename T>
        using ComPtr = Microsoft::WRL::ComPtr<T>;

        /**
         * @brief (Re)create the DXGI output duplication for monitor_index_.
         *
         * Shared by the constructor and reacquire(). Logs the failing step and
         * updates width_/height_ to the output's current dimensions (callers
         * recreate the staging texture when the size changes).
         */
        [[nodiscard]] bool create_duplication();

        /**
         * @brief (Re)create the CUDA-registered BGRA staging texture at width_ x height_.
         *
         * Unregisters and releases any prior texture/resource first, so it is
         * safe to call again after a resolution change.
         * @throws std::runtime_error / CudaError on failure.
         */
        void create_staging_and_register();

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
