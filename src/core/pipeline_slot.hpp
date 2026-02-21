/**
 * @file pipeline_slot.hpp
 * @brief Per-slot GPU resources for the pipelined inference engine.
 */

#pragma once

#include <NvInfer.h>
#include <memory>
#include "core/cuda_stream.hpp"
#include "core/cuda_event.hpp"
#include "core/cuda_buffer.hpp"
#include "core/frame_context.hpp"

namespace bytron {

    /** @brief TRT 10.x custom deleter (uses `delete`, not `destroy()`). */
    template <typename T>
    struct TrtDeleter {
        void operator()(T* p) const noexcept { delete p; }
    };

    template <typename T>
    using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

    /**
     * @brief Holds all GPU resources needed for one concurrent inference slot.
     *
     * Each slot has its own CUDA stream, TRT execution context, and device
     * buffers, allowing multiple frames to be in-flight simultaneously.
     *
     * @note Thread safety: Accessed by the inference thread only, except
     *       display_copy_done which is recorded by the output sink's thread.
     */
    struct PipelineSlot {
        int index = 0;  ///< Slot index within the pipeline [0, MAX_SLOTS)

        CudaStream stream;              ///< Dedicated CUDA stream for this slot's work
        CudaEvent  ready_event;         ///< Signaled when inference + D2H copy completes
        CudaEvent  display_copy_done;   ///< Signaled by OutputSink after copying frame data

        TrtUniquePtr<nvinfer1::IExecutionContext> trt_context;  ///< Per-slot TRT execution context

        CudaDeviceBuffer raw_buffer;     ///< Captured BGRA frame (device, max_w * max_h * 4)
        CudaDeviceBuffer input_buffer;   ///< Preprocessed CHW float tensor (device, 3 * H * W * 4)
        CudaDeviceBuffer output_buffer;  ///< Raw TRT output (device, max_dets * stride * sizeof(float))
        CudaPinnedBuffer output_pinned;  ///< Host copy of output for CPU postprocessing

        FrameContext frame_ctx;  ///< Per-frame metadata populated during processing

        /**
         * @brief Allocate all device and pinned buffers for this slot.
         *
         * @param max_frame_w Maximum capture frame width (for raw_buffer sizing).
         * @param max_frame_h Maximum capture frame height.
         * @param model_w     Model input width (for input_buffer sizing).
         * @param model_h     Model input height.
         * @param max_dets    Maximum detections (for output_buffer sizing).
         * @param det_stride  Values per detection from model output (e.g. 6).
         */
        void allocate(int max_frame_w, int max_frame_h,
                      int model_w, int model_h, int max_dets, int det_stride = 6) {
            raw_buffer    = CudaDeviceBuffer(std::size_t(max_frame_w) * max_frame_h * 4);
            input_buffer  = CudaDeviceBuffer(std::size_t(3) * model_h * model_w * sizeof(float));
            output_buffer = CudaDeviceBuffer(std::size_t(max_dets) * det_stride * sizeof(float));
            output_pinned = CudaPinnedBuffer(std::size_t(max_dets) * det_stride * sizeof(float));
        }
    };

}
