/**
 * @file trt_logger.hpp
 * @brief TensorRT-to-spdlog severity bridge.
 */

#pragma once

#include <NvInfer.h>
#include <spdlog/spdlog.h>

namespace bytron {

    /**
     * @brief Routes TensorRT log messages to spdlog with mapped severity.
     *
     * Mapping: kINTERNAL_ERROR -> critical, kERROR -> error,
     * kWARNING -> warn, kINFO -> info, kVERBOSE -> debug.
     *
     * @note Thread safety: spdlog is thread-safe; usable from TRT internal threads.
     */
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

        [[nodiscard]] static TrtLogger& instance() {
            static TrtLogger logger;
            return logger;
        }
    };

}
