/**
 * @file preprocess_kernels.cuh
 * @brief Fused CUDA preprocessing kernel declarations and letterbox computation.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace bytron {

    /**
     * @brief Pre-computed letterbox transform for aspect-ratio-preserving resize.
     *
     * Padding regions are filled with 114/255 gray to match YOLO training conventions.
     */
    struct LetterboxParams {
        float scale;       ///< Uniform scale factor (min of w_ratio, h_ratio)
        int   new_w, new_h; ///< Scaled image dimensions before padding
        int   pad_x, pad_y; ///< Padding offset in destination pixels (left/top)
    };

    /** @brief Compute letterbox parameters for resizing src into dst with preserved aspect ratio. */
    [[nodiscard]] inline constexpr LetterboxParams compute_letterbox(int src_w, int src_h, int dst_w, int dst_h) {
        float scale_w = static_cast<float>(dst_w) / static_cast<float>(src_w);
        float scale_h = static_cast<float>(dst_h) / static_cast<float>(src_h);
        float scale   = (scale_w < scale_h) ? scale_w : scale_h;
        int   new_w   = static_cast<int>(src_w * scale);
        int   new_h   = static_cast<int>(src_h * scale);
        return { scale, new_w, new_h, (dst_w - new_w) / 2, (dst_h - new_h) / 2 };
    }

    /**
     * @brief Launch the fused BGRA preprocessing kernel.
     *
     * Single-pass GPU kernel: bilinear resize + letterbox padding (114/255 gray)
     * + BGRA-to-RGB channel reorder + [0,255]-to-[0,1] normalize + HWC-to-CHW transpose.
     *
     * @param [in]  src       Device pointer to BGRA source frame.
     * @param [out] dst       Device pointer to CHW float output (3 * dst_h * dst_w floats).
     * @param [in]  src_pitch Source row stride in bytes (typically src_w * 4).
     * @param [in]  stream    CUDA stream for async execution.
     *
     * @note Kernel config: 32x8 thread blocks, grid covers (dst_w, dst_h).
     *       Source pixels loaded as uint32_t via __ldg() for coalesced access.
     */
    void launch_fused_preprocess_bgra(
        const std::uint8_t* src, float* dst,
        int src_w, int src_h, int src_pitch,
        int dst_w, int dst_h,
        cudaStream_t stream);

}
