#pragma once

#include "output/output_sink.hpp"
#include <functional>

namespace bytron {

    /// <summary>
    /// Lightweight sink that forwards each frame to a user-provided callback.
    /// Useful for headless/benchmark mode.
    /// </summary>
    class CallbackSink : public OutputSink {
        public:
            using Callback = std::function<void(const FrameContext&)>;

            explicit CallbackSink(Callback cb) : callback_(std::move(cb)) {}

            void on_frame(const FrameContext& ctx,
                          const void* /*raw_frame_bgr*/,
                          int /*width*/, int /*height*/) override {
                if (callback_) callback_(ctx);
            }

        private:
            Callback callback_;
    };
}
