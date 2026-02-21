/**
 * @file postprocessor.hpp
 * @brief CPU-side postprocessing for NMS-free detection model output.
 */

#pragma once

#include "bytron/types.hpp"
#include "core/frame_context.hpp"
#include <span>

namespace bytron {

    /**
     * @brief Filters and rescales detections from model space to source pixels.
     *
     * Uses a DetectionSchema to map tensor indices to semantic fields (box
     * corners, confidence, class ID), making it model-agnostic. Applies the
     * confidence threshold and inverse-letterbox transform:
     * `src_coord = (model_coord - pad) / scale`, clamped to [0, src_dim].
     *
     * Default schema matches YOLOv26 output: [1, 300, 6] tensor with
     * [x1, y1, x2, y2, confidence, class_id] per detection.
     *
     * @note Thread safety: Stateless; safe for concurrent calls with different contexts.
     */
    class Postprocessor {
    public:
        Postprocessor(float conf_threshold, int max_detections,
                      int stride = 6,
                      DetectionSchema schema = {}) noexcept;

        void process(std::span<const float> output_host, FrameContext& ctx);

    private:
        float           conf_threshold_;
        int             max_detections_;
        int             stride_;   ///< Values per detection in model output (>= 6).
        DetectionSchema schema_;   ///< Maps tensor indices to semantic fields.
    };

}
