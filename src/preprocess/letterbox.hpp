/**
 * @file letterbox.hpp
 * @brief Letterbox transform math shared by the GPU pipeline and CPU-only tests.
 */

#pragma once

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
        if (src_w <= 0 || src_h <= 0)
            return { 1.0f, dst_w, dst_h, 0, 0 };
        float scale_w = static_cast<float>(dst_w) / static_cast<float>(src_w);
        float scale_h = static_cast<float>(dst_h) / static_cast<float>(src_h);
        float scale   = (scale_w < scale_h) ? scale_w : scale_h;
        int   new_w   = static_cast<int>(src_w * scale + 0.5f);
        int   new_h   = static_cast<int>(src_h * scale + 0.5f);
        return { scale, new_w, new_h, (dst_w - new_w) / 2, (dst_h - new_h) / 2 };
    }

}
