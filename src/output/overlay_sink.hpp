/**
 * @file overlay_sink.hpp
 * @brief Transparent overlay sink using CPU-rendered detection boxes.
 */

#pragma once

#include "output/output_sink.hpp"
#include "output/overlay_window.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stop_token>
#include <thread>
#include <vector>

namespace bytron {

    /**
     * @brief Renders detection boxes as a transparent overlay on a Win32 layered window.
     *
     * Discards GPU frame data; only uses detection coordinates to draw green
     * rectangles via CPU on a transparent BGRA buffer. The overlay is
     * click-through and excluded from desktop capture.
     *
     * Rendering is incremental: previous boxes are erased by redrawing their
     * outlines transparent, and only the union of old + new box regions is
     * uploaded to the layered window — work stays proportional to box area,
     * not monitor area.
     *
     * Handoff uses a single latest-wins snapshot: on_frame() replaces the
     * pending detection set, so a slow display never backs up inference.
     *
     * @note Thread safety: on_frame() from inference thread, display_loop()
     *       on dedicated jthread. The snapshot is copied in and out under mtx_.
     */
    class OverlaySink : public OutputSink {
    public:
        OverlaySink(int monitor_width, int monitor_height);
        ~OverlaySink();

        OverlaySink(const OverlaySink&) = delete;
        OverlaySink& operator=(const OverlaySink&) = delete;

        void on_frame(const FrameContext& ctx,
                      void* device_frame,
                      int width, int height, int pitch,
                      cudaStream_t stream,
                      cudaEvent_t copy_done_event = nullptr) override;

        [[nodiscard]] bool should_continue() const noexcept override;

    private:
        void display_loop(std::stop_token stoken);

        static void render_boxes_cpu(void* bgra_buffer, int buf_w, int buf_h,
                                     const std::vector<Detection>& detections,
                                     int thickness, std::uint32_t color);

        int monitor_w_;
        int monitor_h_;

        std::unique_ptr<OverlayWindow> window_;

        /// Latest-wins snapshot of the newest frame's detections.
        std::mutex                  mtx_;
        std::condition_variable_any cv_;
        std::vector<Detection>      latest_dets_;
        std::uint64_t               latest_seq_ = 0;  ///< Incremented per on_frame().

        std::atomic<bool> window_open_{true};
        std::jthread      display_thread_;
    };
}
