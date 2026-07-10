/**
 * @file capture_source.cpp
 * @brief Capture backend factory with automatic fallback.
 */

#include "capture/capture_source.hpp"
#include "capture/desktop_capture.hpp"
#include "capture/graphics_capture.hpp"

#include <spdlog/spdlog.h>

namespace bytron {

    std::unique_ptr<ICaptureSource> make_capture_source(
        CaptureBackend backend, int monitor_index)
    {
        if (backend == CaptureBackend::GraphicsCapture) {
            try {
                auto src = std::make_unique<GraphicsCapture>(monitor_index);
                spdlog::info("Capture backend: {}", src->backend_name());
                return src;
            } catch (const std::exception& e) {
                spdlog::warn("Graphics Capture unavailable ({}), "
                             "falling back to Desktop Duplication", e.what());
            }
        }

        auto src = std::make_unique<DesktopCapture>(monitor_index);
        spdlog::info("Capture backend: {}", src->backend_name());
        return src;
    }

}
