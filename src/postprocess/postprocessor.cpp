/**
 * @file postprocessor.cpp
 * @brief YOLOv26 confidence filtering and inverse-letterbox coordinate transform.
 */

#include "postprocess/postprocessor.hpp"

#include <algorithm>

namespace bytron {

    Postprocessor::Postprocessor(float conf_threshold, int max_detections) noexcept
        : conf_threshold_(conf_threshold)
        , max_detections_(max_detections) {}

    void Postprocessor::process(std::span<const float> output_host, FrameContext& ctx) {
        ctx.detections.clear();

        auto w = static_cast<float>(ctx.src_width);
        auto h = static_cast<float>(ctx.src_height);

        for (int i = 0; i < max_detections_; ++i) {
            const float* det = output_host.data() + i * 6;
            if (det[4] < conf_threshold_) continue;

            // Inverse letterbox: subtract padding, divide by scale, clamp to frame bounds.
            ctx.detections.push_back(Detection{
                .box = BBox{
                    std::clamp((det[0] - ctx.pad_x) / ctx.scale, 0.0f, w),
                    std::clamp((det[1] - ctx.pad_y) / ctx.scale, 0.0f, h),
                    std::clamp((det[2] - ctx.pad_x) / ctx.scale, 0.0f, w),
                    std::clamp((det[3] - ctx.pad_y) / ctx.scale, 0.0f, h),
                },
                .confidence = det[4],
                .class_id   = static_cast<int>(det[5])
            });
        }
    }

}
