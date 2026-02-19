#pragma once

#include "bytron/types.hpp"
#include "core/frame_context.hpp"
#include <vector>

namespace bytron {

    /// <summary>
    /// Postprocesses raw YOLOv26 output: confidence threshold + rescale from
    /// letterboxed 640x640 coords back to original image space.
    /// No NMS needed -- v26 output is already NMS-free.
    /// </summary>
    class Postprocessor {
        public:
            Postprocessor(float conf_threshold, int max_detections);

            /// <summary>
            /// Filter detections by confidence and rescale bounding boxes.
            /// Results are written into ctx.detections.
            /// </summary>
            void process(const float* output_host, FrameContext& ctx);

        private:
            float conf_threshold_;
            int   max_detections_;
    };
}
