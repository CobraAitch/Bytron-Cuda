/**
 * @file postprocessor.cpp
 * @brief Schema-driven confidence filtering and inverse-letterbox coordinate transform.
 */

#include "postprocess/postprocessor.hpp"

#include <algorithm>

namespace bytron {

    Postprocessor::Postprocessor(float conf_threshold, int max_detections,
                                 int stride, DetectionSchema schema) noexcept
        : conf_threshold_(conf_threshold)
        , max_detections_(max_detections)
        , stride_(stride)
        , schema_(schema) {}

    void Postprocessor::process(std::span<const float> output_host, FrameContext& ctx) {
        ctx.detections.clear();

        auto w = static_cast<float>(ctx.src_width);
        auto h = static_cast<float>(ctx.src_height);

        for (int i = 0; i < max_detections_; ++i) {
            const float* det = output_host.data() + i * stride_;
            if (det[schema_.confidence_idx] < conf_threshold_) continue;

            // Inverse letterbox: subtract padding, divide by scale, clamp to frame bounds.
            ctx.detections.push_back(Detection{
                .box = BBox{
                    std::clamp((det[schema_.x1_idx] - ctx.pad_x) / ctx.scale, 0.0f, w),
                    std::clamp((det[schema_.y1_idx] - ctx.pad_y) / ctx.scale, 0.0f, h),
                    std::clamp((det[schema_.x2_idx] - ctx.pad_x) / ctx.scale, 0.0f, w),
                    std::clamp((det[schema_.y2_idx] - ctx.pad_y) / ctx.scale, 0.0f, h),
                },
                .confidence = det[schema_.confidence_idx],
                .class_id   = static_cast<int>(det[schema_.class_id_idx])
            });
        }
    }

}
