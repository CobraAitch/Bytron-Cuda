#include "postprocess/postprocessor.hpp"

#include <algorithm>

namespace bytron {

    Postprocessor::Postprocessor(float conf_threshold, int max_detections)
        : conf_threshold_(conf_threshold)
        , max_detections_(max_detections) {}

    void Postprocessor::process(const float* output_host, FrameContext& ctx) {
        ctx.detections.clear();

        // YOLOv26 output layout: [max_detections, 6] = [x1, y1, x2, y2, conf, class_id]
        for (int i = 0; i < max_detections_; ++i) {
            const float* det = output_host + i * 6;

            float conf = det[4];
            if (conf < conf_threshold_) continue;

            float x1 = det[0];
            float y1 = det[1];
            float x2 = det[2];
            float y2 = det[3];
            int class_id = static_cast<int>(det[5]);

            // Undo letterbox: subtract padding, divide by scale
            x1 = (x1 - ctx.pad_x) / ctx.scale;
            y1 = (y1 - ctx.pad_y) / ctx.scale;
            x2 = (x2 - ctx.pad_x) / ctx.scale;
            y2 = (y2 - ctx.pad_y) / ctx.scale;

            // Clamp to source image bounds
            x1 = std::clamp(x1, 0.0f, static_cast<float>(ctx.src_width));
            y1 = std::clamp(y1, 0.0f, static_cast<float>(ctx.src_height));
            x2 = std::clamp(x2, 0.0f, static_cast<float>(ctx.src_width));
            y2 = std::clamp(y2, 0.0f, static_cast<float>(ctx.src_height));

            ctx.detections.push_back(Detection{
                .box = BBox{x1, y1, x2, y2},
                .confidence = conf,
                .class_id = class_id
            });
        }
    }
}
