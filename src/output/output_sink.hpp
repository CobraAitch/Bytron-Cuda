#pragma once

#include "core/frame_context.hpp"

namespace bytron {

    /// <summary>
    /// Abstract output sink. Receives completed frames with detection results.
    /// </summary>
    class OutputSink {
        public:
            virtual ~OutputSink() = default;

            /// <summary>
            /// Called for each completed frame. raw_frame_bgr may be nullptr in headless mode.
            /// </summary>
            virtual void on_frame(const FrameContext& ctx,
                                  const void* raw_frame_bgr,
                                  int width, int height) = 0;
    };
}
