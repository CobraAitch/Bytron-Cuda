/**
 * @file overlay_kernels.cu
 * @brief CUDA kernel implementation for drawing detection bounding boxes on BGRA frames.
 */

#include "output/overlay_kernels.cuh"
#include "util/cuda_check.hpp"

namespace bytron {

    __device__ __forceinline__ void write_pixel(std::uint8_t* frame, int px, int py, int pitch) {
        // BGRA green (B=0, G=255, R=0, A=255) as a single aligned 32-bit store.
        // Pixel addresses are 4-byte aligned (pitch is a multiple of 4).
        *reinterpret_cast<std::uint32_t*>(frame + py * pitch + px * 4) = 0xFF00FF00u;
    }

    __global__ void draw_boxes_kernel(
        std::uint8_t* __restrict__ frame,
        int width, int height, int pitch,
        const Detection* __restrict__ dets,
        int num_dets, int thickness
    ) {
        const int d = blockIdx.x;
        if (d >= num_dets) return;

        int x1 = max(0, static_cast<int>(dets[d].box.x1));
        int y1 = max(0, static_cast<int>(dets[d].box.y1));
        int x2 = min(width - 1, static_cast<int>(dets[d].box.x2));
        int y2 = min(height - 1, static_cast<int>(dets[d].box.y2));

        int box_w = x2 - x1 + 1;
        int box_h = y2 - y1 + 1;
        if (box_w <= 0 || box_h <= 0) return;

        int t = min(thickness, min(box_w / 2, box_h / 2));
        if (t <= 0) t = 1;

        int inner_h = max(0, box_h - 2 * t);

        // 4 separate uniform loops, no warp divergence within each loop.
        // Top strip: (x1..x2, y1..y1+t-1)
        int top_count = box_w * t;
        for (int i = threadIdx.x; i < top_count; i += blockDim.x)
            write_pixel(frame, x1 + i % box_w, y1 + i / box_w, pitch);

        // Bottom strip: (x1..x2, y2-t+1..y2)
        int bot_count = box_w * t;
        for (int i = threadIdx.x; i < bot_count; i += blockDim.x)
            write_pixel(frame, x1 + i % box_w, (y2 - t + 1) + i / box_w, pitch);

        // Left strip: (x1..x1+t-1, y1+t..y2-t)
        int side_count = t * inner_h;
        for (int i = threadIdx.x; i < side_count; i += blockDim.x)
            write_pixel(frame, x1 + i % t, (y1 + t) + i / t, pitch);

        // Right strip: (x2-t+1..x2, y1+t..y2-t)
        for (int i = threadIdx.x; i < side_count; i += blockDim.x)
            write_pixel(frame, (x2 - t + 1) + i % t, (y1 + t) + i / t, pitch);
    }

    void launch_draw_detection_boxes(
        std::uint8_t* frame_bgra, int width, int height, int pitch,
        const Detection* detections_device, int num_detections,
        int thickness, cudaStream_t stream
    ) {
        if (num_detections == 0) return;

        draw_boxes_kernel<<<num_detections, 256, 0, stream>>>(
            frame_bgra, width, height, pitch,
            detections_device, num_detections, thickness);
        CUDA_CHECK(cudaGetLastError());
    }

}
