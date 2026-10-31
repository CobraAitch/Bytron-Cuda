/**
 * @file graphics_capture.hpp
 * @brief Windows.Graphics.Capture (WGC) capture source with D3D11-CUDA interop.
 */

#pragma once

#include "capture/capture_source.hpp"

#include <memory>

namespace bytron {

    /**
     * @brief Captures a monitor via the modern Windows.Graphics.Capture WinRT API.
     *
     * Compositor-driven, GPU-resident capture: a free-threaded
     * Direct3D11CaptureFramePool delivers frames whose D3D11 textures are copied
     * into a CUDA-registered staging texture, then DMA'd into the destination
     * device buffer, the same zero-host-copy interop path as the DXGI backend.
     *
     * Compared to Desktop Duplication it is borderless and cursorless by default,
     * handles DPI/HDR/multi-monitor cleanly, and recovers automatically from
     * resolution changes by recreating the frame pool.
     *
     * All WinRT/Direct3D state is hidden behind a PImpl so this header stays free
     * of WinRT includes (which must not reach CUDA translation units).
     *
     * @note Thread safety: acquire_frame()/reacquire() run on the capture thread;
     *       the frame-arrived callback only signals a condition variable.
     */
    class GraphicsCapture final : public ICaptureSource {
    public:
        /**
         * @brief Initialize D3D11, the WGC session, and CUDA interop for a monitor.
         * @param monitor_index Monitor index (0 = primary), matching DXGI output order.
         * @throws std::runtime_error if WGC is unsupported or initialization fails.
         */
        explicit GraphicsCapture(int monitor_index);
        ~GraphicsCapture() override;

        GraphicsCapture(const GraphicsCapture&) = delete;
        GraphicsCapture& operator=(const GraphicsCapture&) = delete;

        [[nodiscard]] AcquireResult acquire_frame(CudaDeviceBuffer& dst,
                                                  cudaStream_t stream,
                                                  unsigned timeout_ms) override;

        [[nodiscard]] bool reacquire() override;

        [[nodiscard]] bool is_open() const noexcept override;
        [[nodiscard]] int  width()   const noexcept override;
        [[nodiscard]] int  height()  const noexcept override;
        [[nodiscard]] int  pitch()   const noexcept override;
        [[nodiscard]] const char* backend_name() const noexcept override {
            return "Windows.Graphics.Capture";
        }

        /** @brief Whether WGC is available on this OS (GraphicsCaptureSession::IsSupported). */
        [[nodiscard]] static bool is_supported() noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

}
