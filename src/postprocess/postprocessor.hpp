/**
 * @file postprocessor.hpp
 * @brief CPU-side postprocessing for YOLOv26 NMS-free detection output.
 */

#pragma once

#include "bytron/types.hpp"
#include "core/frame_context.hpp"
#include <span>

namespace bytron {

    /**
     * @brief Filters and rescales YOLOv26 detections from model space to source pixels.
     *
     * YOLOv26 outputs a fixed [1, 300, 6] tensor: [x1, y1, x2, y2, confidence,
     * class_id] in model input pixel space (NMS-free). This postprocessor applies
     * the confidence threshold and inverse-letterbox transform:
     * `src_coord = (model_coord - pad) / scale`, clamped to [0, src_dim].
     *
     * @note Thread safety: Stateless; safe for concurrent calls with different contexts.
     */
    class Postprocessor {
    public:
        Postprocessor(float conf_threshold, int max_detections,
                      int stride = 6) noexcept;

        void process(std::span<const float> output_host, FrameContext& ctx);

    private:
        float conf_threshold_;
        int   max_detections_;
        int   stride_;  ///< Values per detection in model output (>= 6).
    };

}
