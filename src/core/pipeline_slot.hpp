#pragma once

#include <NvInfer.h>
#include <memory>
#include "core/cuda_stream.hpp"
#include "core/cuda_event.hpp"
#include "core/cuda_buffer.hpp"
#include "core/frame_context.hpp"

namespace bytron {

    /// <summary>
    /// Custom deleter for TensorRT objects allocated by the runtime.
    /// </summary>
    template <typename T>
    struct TrtDeleter {
        void operator()(T* p) const noexcept { delete p; }
    };

    template <typename T>
    using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

    /// <summary>
    /// One slot in the triple-buffered pipeline. Each slot owns its own stream,
    /// buffers, and TRT execution context so multiple frames can overlap on the GPU.
    /// </summary>
    struct PipelineSlot {
        int index = 0;

        CudaStream stream;
        CudaEvent  ready_event;

        TrtUniquePtr<nvinfer1::IExecutionContext> trt_context;

        CudaDeviceBuffer raw_buffer;     // BGR source frame on device
        CudaDeviceBuffer input_buffer;   // CHW float32 preprocessed tensor
        CudaDeviceBuffer output_buffer;  // [max_det, 6] on device

        CudaPinnedBuffer output_pinned;  // [max_det, 6] host-side for postprocess

        FrameContext frame_ctx;

        /// <summary>
        /// Allocate all GPU and pinned buffers for this slot.
        /// </summary>
        void allocate(int max_frame_width, int max_frame_height,
                      int model_w, int model_h, int max_detections) {
            raw_buffer = CudaDeviceBuffer(
                static_cast<std::size_t>(max_frame_width) * max_frame_height * 3);

            input_buffer = CudaDeviceBuffer(
                static_cast<std::size_t>(3) * model_h * model_w * sizeof(float));

            output_buffer = CudaDeviceBuffer(
                static_cast<std::size_t>(max_detections) * 6 * sizeof(float));

            output_pinned = CudaPinnedBuffer(
                static_cast<std::size_t>(max_detections) * 6 * sizeof(float));
        }
    };
}
