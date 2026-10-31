/**
 * @file callback_sink.hpp
 * @brief Minimal output sink that forwards detections to a user callback.
 */

#pragma once

#include "output/output_sink.hpp"
#include <functional>

namespace bytron {

    /**
     * @brief Invokes a std::function on each frame's FrameContext.
     *
     * Discards the device frame data entirely, suitable for headless
     * operation or when only detection metadata is needed.
     *
     * @note Thread safety: The callback is invoked on the inference thread.
     */
    class CallbackSink : public OutputSink {
    public:
        using Callback = std::function<void(const FrameContext&)>;

        explicit CallbackSink(Callback cb) : callback_(std::move(cb)) {}

        void on_frame(const FrameContext& ctx,
                      void*, int, int, int,
                      cudaStream_t, cudaEvent_t) override {
            if (callback_) callback_(ctx);
        }

    private:
        Callback callback_;
    };

}
