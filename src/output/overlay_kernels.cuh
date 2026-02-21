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
     * Launches one block per detection (256 threads). Each block decomposes the
     * box border into 4 non-overlapping strips (top/bottom/left/right) and threads
     * cooperatively write opaque green (BGRA 0,255,0,255) pixels.
     * Work is O(N_dets * box_perimeter * thickness) instead of O(frame_pixels * N_dets).
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
