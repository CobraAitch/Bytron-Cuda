/**
 * @file overlay_kernels.cuh
 * @brief CUDA kernel launch wrapper for drawing detection bounding boxes on BGRA frames.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "bytron/types.hpp"

namespace bytron {

    /**
     * @brief Draw green bounding boxes around detections directly on a device BGRA frame.
     *
     * Launches a 2D grid of 32x32 thread blocks. Each thread tests its pixel
     * against all detection bounding boxes and writes opaque green (BGRA 0,255,0,255)
     * if the pixel falls on a box border.
     *
     * @param [in,out] frame_bgra        Device pointer to BGRA frame buffer.
     * @param [in]     width             Frame width in pixels.
     * @param [in]     height            Frame height in pixels.
     * @param [in]     pitch             Row stride in bytes (may exceed width*4 due to alignment).
     * @param [in]     detections_device Device pointer to Detection array.
     * @param [in]     num_detections    Number of detections to draw (0 = no-op).
     * @param [in]     thickness         Border thickness in pixels.
     * @param [in]     stream            CUDA stream for asynchronous launch.
     */
    void launch_draw_detection_boxes(
        std::uint8_t* frame_bgra,
        int width, int height, int pitch,
        const Detection* detections_device,
        int num_detections,
        int thickness,
        cudaStream_t stream);

}
