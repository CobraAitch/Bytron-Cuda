#pragma once

#include "capture/video_source.hpp"
#include "core/cuda_buffer.hpp"

#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <string>

namespace bytron {

    /// <summary>
    /// VideoSource backed by OpenCV's VideoCapture.
    /// Reads frames on the CPU, then uploads through pinned staging to the GPU.
    /// </summary>
    class VideoSourceOpenCV : public VideoSource {
        public:
            explicit VideoSourceOpenCV(const std::string& uri);

            bool acquire_frame(CudaDeviceBuffer& dst, cudaStream_t stream,
                               int& out_width, int& out_height) override;

            [[nodiscard]] bool is_open() const override;
            [[nodiscard]] double fps() const override;
            [[nodiscard]] int width() const override;
            [[nodiscard]] int height() const override;

        private:
            cv::VideoCapture cap_;
            cv::Mat          cpu_frame_;
            CudaPinnedBuffer pinned_staging_;
            int width_ = 0;
            int height_ = 0;
            double fps_ = 0.0;
    };
}
