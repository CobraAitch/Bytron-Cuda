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

        // Decompose border into 4 non-overlapping strips:
        //   top:    (x1..x2, y1..y1+t-1)
        //   bottom: (x1..x2, y2-t+1..y2)
        //   left:   (x1..x1+t-1, y1+t..y2-t)
        //   right:  (x2-t+1..x2, y1+t..y2-t)
        int inner_h    = max(0, box_h - 2 * t);
        int top_count  = box_w * t;
        int bot_count  = box_w * t;
        int side_count = t * inner_h;
        int total      = top_count + bot_count + 2 * side_count;

        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            int px, py;
            if (i < top_count) {
                px = x1 + i % box_w;
                py = y1 + i / box_w;
            } else if (i < top_count + bot_count) {
                int j = i - top_count;
                px = x1 + j % box_w;
                py = (y2 - t + 1) + j / box_w;
            } else if (i < top_count + bot_count + side_count) {
                int j = i - top_count - bot_count;
                px = x1 + j % t;
                py = (y1 + t) + j / t;
            } else {
                int j = i - top_count - bot_count - side_count;
                px = (x2 - t + 1) + j % t;
                py = (y1 + t) + j / t;
            }

            auto* pixel = frame + py * pitch + px * 4;
            // BGRA green: B=0, G=255, R=0, A=255 (fully opaque).
            pixel[0] = 0; pixel[1] = 255; pixel[2] = 0; pixel[3] = 255;
        }
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
    }

}
