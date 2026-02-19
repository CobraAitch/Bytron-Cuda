#include "capture/video_source_factory.hpp"
#include "capture/video_source_opencv.hpp"

#include <spdlog/spdlog.h>

namespace bytron {

    std::unique_ptr<VideoSource> create_video_source(const std::string& uri) {
        spdlog::info("Using OpenCV for: {}", uri);
        return std::make_unique<VideoSourceOpenCV>(uri);
    }
}
