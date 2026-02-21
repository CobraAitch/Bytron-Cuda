/**
 * @file preprocess_kernels.cu
 * @brief Fused BGRA preprocessing CUDA kernel implementation.
 */

#include "preprocess/preprocess_kernels.cuh"

namespace bytron {

    __global__ void fused_preprocess_kernel_bgra(
        const std::uint8_t* __restrict__ src,
        float*              __restrict__ dst,
        int src_w, int src_h, int src_pitch,
        int dst_w, int dst_h,
        float scale, int pad_x, int pad_y, int new_w, int new_h
    ) {
        const int dx = blockIdx.x * blockDim.x + threadIdx.x;
        const int dy = blockIdx.y * blockDim.y + threadIdx.y;
        if (dx >= dst_w || dy >= dst_h) return;

        constexpr float pad_val = 114.0f / 255.0f;
        const int rx = dx - pad_x;
        const int ry = dy - pad_y;

        if (rx >= 0 && rx < new_w && ry >= 0 && ry < new_h) {
            const float sx = (static_cast<float>(rx) + 0.5f) / scale - 0.5f;
            const float sy = (static_cast<float>(ry) + 0.5f) / scale - 0.5f;

            const int x0  = __float2int_rd(sx);
            const int y0  = __float2int_rd(sy);
            const int x1  = min(x0 + 1, src_w - 1);
            const int y1  = min(y0 + 1, src_h - 1);
            const float fx = sx - static_cast<float>(x0);
            const float fy = sy - static_cast<float>(y0);
            const int cx0 = max(0, x0);
            const int cy0 = max(0, y0);

            const int plane_size = dst_h * dst_w;

            #pragma unroll
            for (int c = 0; c < 3; ++c) {
                // BGRA-to-RGB: output channel c reads source channel (2-c).
                // c=0(R)->sc=2(B byte), c=1(G)->sc=1(G byte), c=2(B)->sc=0(R byte).
                const int sc = 2 - c;
                const float v00 = static_cast<float>(src[cy0 * src_pitch + cx0 * 4 + sc]);
                const float v01 = static_cast<float>(src[cy0 * src_pitch + x1  * 4 + sc]);
                const float v10 = static_cast<float>(src[y1  * src_pitch + cx0 * 4 + sc]);
                const float v11 = static_cast<float>(src[y1  * src_pitch + x1  * 4 + sc]);

                const float val = v00 * (1.0f - fx) * (1.0f - fy)
                                + v01 * fx * (1.0f - fy)
                                + v10 * (1.0f - fx) * fy
                                + v11 * fx * fy;

                dst[c * plane_size + dy * dst_w + dx] = val / 255.0f;
            }
            return;
        }

        const int plane_size = dst_h * dst_w;
        dst[0 * plane_size + dy * dst_w + dx] = pad_val;
        dst[1 * plane_size + dy * dst_w + dx] = pad_val;
        dst[2 * plane_size + dy * dst_w + dx] = pad_val;
    }

    void launch_fused_preprocess_bgra(
        const std::uint8_t* src, float* dst,
        int src_w, int src_h, int src_pitch,
        int dst_w, int dst_h,
        cudaStream_t stream
    ) {
        auto params = compute_letterbox(src_w, src_h, dst_w, dst_h);

        dim3 block(32, 32);
        dim3 grid((dst_w + 31) / 32, (dst_h + 31) / 32);

        fused_preprocess_kernel_bgra<<<grid, block, 0, stream>>>(
            src, dst, src_w, src_h, src_pitch, dst_w, dst_h,
            params.scale, params.pad_x, params.pad_y, params.new_w, params.new_h);
    }

}
