#include "capture/video_source_opencv.hpp"
#include "util/cuda_check.hpp"

#include <spdlog/spdlog.h>
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace bytron {

    VideoSourceOpenCV::VideoSourceOpenCV(const std::string& uri) {
        // If the URI is all digits, treat it as a camera index
        bool is_camera = !uri.empty() && std::all_of(uri.begin(), uri.end(), ::isdigit);

        if (is_camera) {
            cap_.open(std::stoi(uri), cv::CAP_ANY);
        } else {
            cap_.open(uri, cv::CAP_ANY);
        }

        if (!cap_.isOpened()) {
            throw std::runtime_error("Failed to open video source: " + uri);
        }

        width_  = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
        height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
        fps_    = cap_.get(cv::CAP_PROP_FPS);
        if (fps_ <= 0.0) fps_ = 30.0; // fallback for cameras that don't report FPS

        std::size_t frame_bytes = static_cast<std::size_t>(width_) * height_ * 3;
        pinned_staging_ = CudaPinnedBuffer(frame_bytes);

        spdlog::info("OpenCV source: {}x{} @ {:.1f} FPS", width_, height_, fps_);
    }

    bool VideoSourceOpenCV::acquire_frame(
        CudaDeviceBuffer& dst, cudaStream_t stream,
        int& out_width, int& out_height
    ) {
        if (!cap_.read(cpu_frame_)) return false;

        out_width = cpu_frame_.cols;
        out_height = cpu_frame_.rows;

        std::size_t frame_bytes = static_cast<std::size_t>(out_width) * out_height * 3;

        // CPU -> pinned -> GPU (async)
        std::memcpy(pinned_staging_.ptr(), cpu_frame_.data, frame_bytes);
        dst.copy_from_host_async(pinned_staging_.ptr(), frame_bytes, stream);

        return true;
    }

    bool VideoSourceOpenCV::is_open() const { return cap_.isOpened(); }
    double VideoSourceOpenCV::fps() const { return fps_; }
    int VideoSourceOpenCV::width() const { return width_; }
    int VideoSourceOpenCV::height() const { return height_; }
}
