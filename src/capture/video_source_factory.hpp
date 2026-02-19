#pragma once

#include "capture/video_source.hpp"
#include <memory>
#include <string>

namespace bytron {

    /// <summary>
    /// Factory that creates the appropriate VideoSource for the given URI.
    /// Currently always returns an OpenCV-backed source.
    /// </summary>
    std::unique_ptr<VideoSource> create_video_source(const std::string& uri);
}
