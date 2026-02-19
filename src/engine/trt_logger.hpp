#pragma once

#include <NvInfer.h>
#include <spdlog/spdlog.h>

namespace bytron {

    /// <summary>
    /// Bridges TensorRT's logging into spdlog at matching severity levels.
    /// </summary>
    class TrtLogger : public nvinfer1::ILogger {
        public:
            void log(Severity severity, const char* msg) noexcept override {
                switch (severity) {
                    case Severity::kINTERNAL_ERROR: spdlog::critical("[TRT] {}", msg); break;
                    case Severity::kERROR:          spdlog::error("[TRT] {}", msg);    break;
                    case Severity::kWARNING:        spdlog::warn("[TRT] {}", msg);     break;
                    case Severity::kINFO:           spdlog::info("[TRT] {}", msg);     break;
                    case Severity::kVERBOSE:        spdlog::debug("[TRT] {}", msg);    break;
                }
            }

            static TrtLogger& instance() {
                static TrtLogger logger;
                return logger;
            }
    };
}
