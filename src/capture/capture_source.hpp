/**
 * @file capture_source.hpp
 * @brief Abstract screen-capture interface and backend factory.
 */

#pragma once

#include "core/cuda_buffer.hpp"
#include "bytron/types.hpp"

#include <cuda_runtime.h>
#include <memory>

namespace bytron {

    /** @brief Result of a frame acquisition attempt. */
    enum class AcquireResult {
        kNewFrame,   ///< Successfully acquired a new frame into the destination buffer
        kTimeout,    ///< No new frame within the timeout period
        kError,      ///< Unrecoverable error; capture is unusable
        kAccessLost  ///< Capture lost (display mode change / item closed); call reacquire()
    };

    /**
     * @brief Common interface for GPU-resident desktop capture sources.
     *
     * Implementations deliver BGRA (B8G8R8A8) frames into a CUDA device buffer
     * with zero host round-trips. All methods are called from a single capture
     * thread; implementations are not required to be thread-safe.
     */
    class ICaptureSource {
    public:
        virtual ~ICaptureSource() = default;

        /**
         * @brief Acquire the next frame into a device buffer.
         *
         * @param [out] dst        Device buffer receiving BGRA frame data.
         * @param [in]  stream     CUDA stream for async copy operations.
         * @param [in]  timeout_ms Wait budget for a new frame (~16ms = one 60 Hz frame).
         * @return AcquireResult indicating success, timeout, access loss, or error.
         * @pre dst.size() >= width() * height() * 4
         */
        [[nodiscard]] virtual AcquireResult acquire_frame(CudaDeviceBuffer& dst,
                                                          cudaStream_t stream,
                                                          unsigned timeout_ms) = 0;

        /** @brief Rebuild capture after access loss. @return true on success. */
        [[nodiscard]] virtual bool reacquire() = 0;

        [[nodiscard]] virtual bool is_open() const noexcept = 0;
        [[nodiscard]] virtual int  width()   const noexcept = 0;  ///< Capture width in pixels
        [[nodiscard]] virtual int  height()  const noexcept = 0;  ///< Capture height in pixels
        [[nodiscard]] virtual int  pitch()   const noexcept = 0;  ///< Row stride in bytes

        /** @brief Human-readable backend name for logging. */
        [[nodiscard]] virtual const char* backend_name() const noexcept = 0;
    };

    /**
     * @brief Construct the requested capture backend for a monitor.
     *
     * When @p backend is GraphicsCapture but WGC is unsupported or fails to
     * initialize, transparently falls back to DXGI Desktop Duplication.
     *
     * @throws std::runtime_error if no backend could be initialized.
     */
    [[nodiscard]] std::unique_ptr<ICaptureSource> make_capture_source(
        CaptureBackend backend, int monitor_index);

}
