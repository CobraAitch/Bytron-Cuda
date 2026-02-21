/**
 * @file output_sink.hpp
 * @brief Abstract interface for consuming inference results and rendered frames.
 */

#pragma once

#include "core/frame_context.hpp"
#include <cuda_runtime.h>

namespace bytron {

    /**
     * @brief Base class for pipeline output consumers.
     *
     * The pipeline calls on_frame() from the inference thread for each
     * completed frame. Implementations that forward to another thread
     * must handle synchronization internally.
     *
     * @note Thread safety: on_frame() is called from the inference thread.
     */
    class OutputSink {
    public:
        virtual ~OutputSink() = default;

        /**
         * @brief Called for each completed inference frame.
         *
         * @param [in]  ctx             Detection results and frame metadata.
         * @param [in]  device_frame    Device pointer to BGRA frame data. Valid only
         *                              until copy_done_event is recorded by the sink.
         * @param [in]  width           Frame width in pixels.
         * @param [in]  height          Frame height in pixels.
         * @param [in]  pitch           Row stride in bytes (width * 4).
         * @param [in]  stream          CUDA stream the frame was processed on.
         * @param [in,out] copy_done_event  The sink must record this event after
         *                              finishing its copy of device_frame. The pipeline
         *                              waits on it before reusing the slot's raw_buffer.
         */
        virtual void on_frame(const FrameContext& ctx,
                              void* device_frame,
                              int width, int height, int pitch,
                              cudaStream_t stream,
                              cudaEvent_t copy_done_event = nullptr) = 0;

        /**
         * @brief Polled each iteration; return false to signal graceful shutdown.
         */
        [[nodiscard]] virtual bool should_continue() const noexcept { return true; }
    };

}
