#pragma once

#include "output/output_sink.hpp"
#include <string>

namespace bytron {

    /// <summary>
    /// Draws detection boxes on the frame and shows it in a cv::imshow window.
    /// ESC key or window close sets should_continue() to false.
    /// </summary>
    class DisplaySink : public OutputSink {
        public:
            explicit DisplaySink(const std::string& window_name = "Bytron");

            void on_frame(const FrameContext& ctx,
                          const void* raw_frame_bgr,
                          int width, int height) override;

            [[nodiscard]] bool should_continue() const noexcept { return running_; }

        private:
            std::string window_name_;
            bool running_ = true;
    };
}
