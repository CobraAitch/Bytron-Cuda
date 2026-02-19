#include "preprocess/preprocess_kernels.cuh"

namespace bytron {

    __global__ void fused_preprocess_kernel(
        const uint8_t* __restrict__ src,
        float*         __restrict__ dst,
        int src_w, int src_h, int src_pitch,
        int dst_w, int dst_h,
        float scale,
        int pad_x, int pad_y,
        int new_w, int new_h
    ) {
        const int dx = blockIdx.x * blockDim.x + threadIdx.x;
        const int dy = blockIdx.y * blockDim.y + threadIdx.y;

        if (dx >= dst_w || dy >= dst_h) return;

        // Letterbox padding value: 114/255 (standard YOLO gray)
        constexpr float pad_val = 114.0f / 255.0f;

        const int rx = dx - pad_x;
        const int ry = dy - pad_y;

        if (rx >= 0 && rx < new_w && ry >= 0 && ry < new_h) {
            // Bilinear interpolation: map output pixel back to source coordinates
            const float sx = (static_cast<float>(rx) + 0.5f) / scale - 0.5f;
            const float sy = (static_cast<float>(ry) + 0.5f) / scale - 0.5f;

            const int x0 = __float2int_rd(sx);
            const int y0 = __float2int_rd(sy);
            const int x1 = min(x0 + 1, src_w - 1);
            const int y1 = min(y0 + 1, src_h - 1);

            const float fx = sx - static_cast<float>(x0);
            const float fy = sy - static_cast<float>(y0);

            const int cx0 = max(0, x0);
            const int cy0 = max(0, y0);

            const int plane_size = dst_h * dst_w;

            // BGR -> RGB with bilinear + normalize to [0,1], output in CHW layout
            #pragma unroll
            for (int c = 0; c < 3; ++c) {
                const int sc = 2 - c; // R(0)->src[2], G(1)->src[1], B(2)->src[0]

                const float v00 = static_cast<float>(src[cy0 * src_pitch + cx0 * 3 + sc]);
                const float v01 = static_cast<float>(src[cy0 * src_pitch + x1  * 3 + sc]);
                const float v10 = static_cast<float>(src[y1  * src_pitch + cx0 * 3 + sc]);
                const float v11 = static_cast<float>(src[y1  * src_pitch + x1  * 3 + sc]);

                const float val = v00 * (1.0f - fx) * (1.0f - fy)
                                + v01 * fx * (1.0f - fy)
                                + v10 * (1.0f - fx) * fy
                                + v11 * fx * fy;

                dst[c * plane_size + dy * dst_w + dx] = val / 255.0f;
            }
            return;
        }

        // Padding region: fill all 3 channels with gray
        const int plane_size = dst_h * dst_w;
        dst[0 * plane_size + dy * dst_w + dx] = pad_val;
        dst[1 * plane_size + dy * dst_w + dx] = pad_val;
        dst[2 * plane_size + dy * dst_w + dx] = pad_val;
    }

    LetterboxParams compute_letterbox(int src_w, int src_h, int dst_w, int dst_h) {
        LetterboxParams p;
        float scale_w = static_cast<float>(dst_w) / static_cast<float>(src_w);
        float scale_h = static_cast<float>(dst_h) / static_cast<float>(src_h);
        p.scale = (scale_w < scale_h) ? scale_w : scale_h;
        p.new_w = static_cast<int>(src_w * p.scale);
        p.new_h = static_cast<int>(src_h * p.scale);
        p.pad_x = (dst_w - p.new_w) / 2;
        p.pad_y = (dst_h - p.new_h) / 2;
        return p;
    }

    void launch_fused_preprocess(
        const uint8_t* src, float* dst,
        int src_w, int src_h, int src_pitch,
        int dst_w, int dst_h,
        cudaStream_t stream
    ) {
        auto params = compute_letterbox(src_w, src_h, dst_w, dst_h);

        dim3 block(32, 32);
        dim3 grid(
            (dst_w + block.x - 1) / block.x,
            (dst_h + block.y - 1) / block.y
        );

        fused_preprocess_kernel<<<grid, block, 0, stream>>>(
            src, dst,
            src_w, src_h, src_pitch,
            dst_w, dst_h,
            params.scale,
            params.pad_x, params.pad_y,
            params.new_w, params.new_h
        );
    }
}
