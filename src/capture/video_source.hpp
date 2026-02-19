#pragma once

#include "core/cuda_buffer.hpp"
#include <cuda_runtime.h>

namespace bytron {
    /// <summary>
    /// Abstract base class representing a video source capable of acquiring frames into CUDA device buffers.
    /// Implementations provide access to video properties such as resolution and frame rate,
    /// and support asynchronous frame acquisition using CUDA streams.
    /// </summary>
    class VideoSource {
        public:
            /// <summary>
            /// Virtual destructor for the VideoSource class.
            /// </summary>
            virtual ~VideoSource() = default;

            /// <summary>
            /// Acquires a frame and stores it in the destination CUDA device buffer.
            /// </summary>
            /// <param name="dst">The destination CUDA device buffer where the acquired frame will be stored.</param>
            /// <param name="stream">The CUDA stream to use for asynchronous operations.</param>
            /// <param name="out_width">Output parameter that receives the width of the acquired frame.</param>
            /// <param name="out_height">Output parameter that receives the height of the acquired frame.</param>
            /// <returns>True if the frame was successfully acquired; otherwise, false.</returns>
            virtual bool acquire_frame(CudaDeviceBuffer& dst, cudaStream_t stream,
                                       int& out_width, int& out_height) = 0;

            /// <summary>
            /// Checks whether the video source is currently open and available for frame acquisition.
            /// </summary>
            /// <returns>True if the video source is open; otherwise, false.</returns>
            [[nodiscard]] virtual bool is_open() const = 0;

            /// <summary>
            /// Gets the frame rate of the video source in frames per second.
            /// </summary>
            /// <returns>The frame rate of the video source.</returns>
            [[nodiscard]] virtual double fps() const = 0;

            /// <summary>
            /// Gets the width of the video source in pixels.
            /// </summary>
            /// <returns>The width of the video frames.</returns>
            [[nodiscard]] virtual int width() const = 0;

            /// <summary>
            /// Gets the height of the video source in pixels.
            /// </summary>
            /// <returns>The height of the video frames.</returns>
            [[nodiscard]] virtual int height() const = 0;
    };
}
