#include "output/display_sink.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace bytron {

    DisplaySink::DisplaySink(const std::string& window_name)
        : window_name_(window_name) {
        cv::namedWindow(window_name_, cv::WINDOW_AUTOSIZE);
    }

    void DisplaySink::on_frame(
        const FrameContext& ctx,
        const void* raw_frame_bgr,
        int width, int height
    ) {
        // Wrap raw BGR pixels in a cv::Mat without copying
        cv::Mat frame(height, width, CV_8UC3,
                      const_cast<void*>(raw_frame_bgr));

        for (const auto& det : ctx.detections) {
            cv::Rect rect(
                static_cast<int>(det.box.x1),
                static_cast<int>(det.box.y1),
                static_cast<int>(det.box.x2 - det.box.x1),
                static_cast<int>(det.box.y2 - det.box.y1)
            );

            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);

            std::string label = std::to_string(det.class_id) + " "
                              + std::to_string(static_cast<int>(det.confidence * 100)) + "%";
            cv::putText(frame, label,
                        cv::Point(rect.x, rect.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 255, 0), 1);
        }

        cv::imshow(window_name_, frame);
        int key = cv::waitKey(1);
        if (key == 27) { // ESC
            running_ = false;
        }
    }
}
