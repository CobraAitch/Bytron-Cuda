/**
 * @file preprocess_kernels.cu
 * @brief Fused BGRA preprocessing CUDA kernel implementation.
 */

#include "preprocess/preprocess_kernels.cuh"
#include "util/cuda_check.hpp"

namespace bytron {

    __global__ void fused_preprocess_kernel_bgra(
        const std::uint8_t* __restrict__ src,
        float*              __restrict__ dst,
        int src_w, int src_h, int src_pitch,
        int dst_w, int dst_h, int plane_size,
        float inv_scale, int pad_x, int pad_y, int new_w, int new_h
    ) {
        const int dx = blockIdx.x * blockDim.x + threadIdx.x;
        const int dy = blockIdx.y * blockDim.y + threadIdx.y;
        if (dx >= dst_w || dy >= dst_h) return;

        constexpr float pad_val = 114.0f / 255.0f;
        const int rx = dx - pad_x;
        const int ry = dy - pad_y;
        const int dst_idx = dy * dst_w + dx;

        if (rx >= 0 && rx < new_w && ry >= 0 && ry < new_h) {
            const float sx = (static_cast<float>(rx) + 0.5f) * inv_scale - 0.5f;
            const float sy = (static_cast<float>(ry) + 0.5f) * inv_scale - 0.5f;

            const int x0  = __float2int_rd(sx);
            const int y0  = __float2int_rd(sy);
            const int x1  = min(x0 + 1, src_w - 1);
            const int y1  = min(y0 + 1, src_h - 1);
            const float fx = sx - static_cast<float>(x0);
            const float fy = sy - static_cast<float>(y0);
            const int cx0 = max(0, x0);
            const int cy0 = max(0, y0);

            // Load 4 BGRA pixels as uint32_t via read-only texture cache.
            const auto* src32 = reinterpret_cast<const unsigned int*>(src);
            const int pitch32 = src_pitch >> 2;
            const unsigned int p00 = __ldg(&src32[cy0 * pitch32 + cx0]);
            const unsigned int p01 = __ldg(&src32[cy0 * pitch32 + x1]);
            const unsigned int p10 = __ldg(&src32[y1  * pitch32 + cx0]);
            const unsigned int p11 = __ldg(&src32[y1  * pitch32 + x1]);

            // Precompute bilinear weights once (shared across all 3 channels).
            const float w00 = (1.0f - fx) * (1.0f - fy);
            const float w01 = fx * (1.0f - fy);
            const float w10 = (1.0f - fx) * fy;
            const float w11 = fx * fy;

            constexpr float inv255 = 1.0f / 255.0f;

            #pragma unroll
            for (int c = 0; c < 3; ++c) {
                // BGRA-to-RGB: c=0(R)->shift 16, c=1(G)->shift 8, c=2(B)->shift 0.
                const int shift = (2 - c) * 8;
                const float v00 = static_cast<float>((p00 >> shift) & 0xFFu);
                const float v01 = static_cast<float>((p01 >> shift) & 0xFFu);
                const float v10 = static_cast<float>((p10 >> shift) & 0xFFu);
                const float v11 = static_cast<float>((p11 >> shift) & 0xFFu);

                dst[c * plane_size + dst_idx] =
                    (v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11) * inv255;
            }
            return;
        }

        dst[0 * plane_size + dst_idx] = pad_val;
        dst[1 * plane_size + dst_idx] = pad_val;
        dst[2 * plane_size + dst_idx] = pad_val;
    }

    void launch_fused_preprocess_bgra(
        const std::uint8_t* src, float* dst,
        int src_w, int src_h, int src_pitch,
        int dst_w, int dst_h,
        const LetterboxParams& params,
        cudaStream_t stream
    ) {
        dim3 block(32, 8);
        dim3 grid((dst_w + 31) / 32, (dst_h + 7) / 8);

        // Reciprocal computed once on the host so every thread multiplies
        // instead of dividing.
        const float inv_scale = 1.0f / params.scale;

        fused_preprocess_kernel_bgra<<<grid, block, 0, stream>>>(
            src, dst, src_w, src_h, src_pitch, dst_w, dst_h,
            dst_h * dst_w,
            inv_scale, params.pad_x, params.pad_y, params.new_w, params.new_h);
        CUDA_CHECK(cudaGetLastError());
    }

}
