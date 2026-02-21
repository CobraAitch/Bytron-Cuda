/**
 * @file overlay_kernels.cu
 * @brief CUDA kernel implementation for drawing detection bounding boxes on BGRA frames.
 */

#include "output/overlay_kernels.cuh"

namespace bytron {

    __global__ void draw_boxes_kernel(
        std::uint8_t* __restrict__ frame,
        int width, int height, int pitch,
        const Detection* __restrict__ dets,
        int num_dets, int thickness
    ) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;

        for (int d = 0; d < num_dets; ++d) {
            int x1 = max(0, static_cast<int>(dets[d].box.x1));
            int y1 = max(0, static_cast<int>(dets[d].box.y1));
            int x2 = min(width  - 1, static_cast<int>(dets[d].box.x2));
            int y2 = min(height - 1, static_cast<int>(dets[d].box.y2));

            bool on_border = (y >= y1 && y < y1 + thickness && x >= x1 && x <= x2)
                          || (y > y2 - thickness && y <= y2 && x >= x1 && x <= x2)
                          || (x >= x1 && x < x1 + thickness && y >= y1 && y <= y2)
                          || (x > x2 - thickness && x <= x2 && y >= y1 && y <= y2);

            if (on_border) {
                auto* pixel = frame + y * pitch + x * 4;
                // BGRA green: B=0, G=255, R=0, A=255 (fully opaque).
                pixel[0] = 0; pixel[1] = 255; pixel[2] = 0; pixel[3] = 255;
                return;
            }
        }
    }

    void launch_draw_detection_boxes(
        std::uint8_t* frame_bgra, int width, int height, int pitch,
        const Detection* detections_device, int num_detections,
        int thickness, cudaStream_t stream
    ) {
        if (num_detections == 0) return;

        dim3 block(32, 32);
        dim3 grid((width + 31) / 32, (height + 31) / 32);

        draw_boxes_kernel<<<grid, block, 0, stream>>>(
            frame_bgra, width, height, pitch,
            detections_device, num_detections, thickness);
    }

}
