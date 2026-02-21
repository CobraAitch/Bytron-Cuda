/**
 * @file overlay_sink.cpp
 * @brief OverlaySink CPU box rendering and Win32 layered window present loop.
 */

#include "output/overlay_sink.hpp"
#include "util/cuda_check.hpp"

#include <spdlog/spdlog.h>
#include <cstring>
#include <algorithm>
#include <cstdint>

namespace bytron {

    OverlaySink::OverlaySink(int monitor_width, int monitor_height)
        : monitor_w_(monitor_width)
        , monitor_h_(monitor_height)
    {
        window_ = std::make_unique<OverlayWindow>(monitor_w_, monitor_h_);
        display_thread_ = std::jthread([this](std::stop_token st) { display_loop(st); });
    }

    OverlaySink::~OverlaySink() {
        display_thread_.request_stop();
        cv_.notify_one();
    }

    void OverlaySink::on_frame(
        const FrameContext& ctx,
        void* /*device_frame*/,
        int width, int height, int /*pitch*/,
        cudaStream_t /*stream*/,
        cudaEvent_t copy_done_event
    ) {
        if (display_thread_.get_stop_token().stop_requested()) return;

        auto& mb = mailbox_[write_idx_];
        mb.detections = ctx.detections;
        mb.src_width  = width;
        mb.src_height = height;

        if (copy_done_event)
            cudaEventRecord(copy_done_event, nullptr);

        {
            std::lock_guard lock(mtx_);
            pending_idx_ = write_idx_;
        }
        cv_.notify_one();

        write_idx_ = (write_idx_ + 1) % MAILBOX_SLOTS;
    }

    void OverlaySink::display_loop(std::stop_token stoken) {
        auto* overlay = window_.get();

        while (!stoken.stop_requested()) {
            int slot_idx;
            {
                std::unique_lock lock(mtx_);
                cv_.wait_for(lock, std::chrono::milliseconds(4), [&] {
                    return pending_idx_ >= 0 || stoken.stop_requested();
                });
                if (stoken.stop_requested()) break;

                if (pending_idx_ < 0) {
                    overlay->poll_events();
                    if (overlay->should_close()) {
                        window_open_ = false;
                        return;
                    }
                    continue;
                }
                slot_idx = pending_idx_;
                pending_idx_ = -1;
            }

            auto& mb = mailbox_[slot_idx];

            auto* pixels = overlay->pixels();
            std::memset(pixels, 0, overlay->pixel_bytes());

            render_boxes_cpu(pixels,
                             overlay->width(), overlay->height(),
                             mb.detections, 2);

            overlay->present();

            if (overlay->should_close()) {
                window_open_ = false;
                return;
            }
        }
    }

    void OverlaySink::render_boxes_cpu(
        void* bgra_buffer, int buf_w, int buf_h,
        const std::vector<Detection>& detections,
        int thickness
    ) {
        auto* pixels = static_cast<std::uint32_t*>(bgra_buffer);
        constexpr std::uint32_t green = 0xFF00FF00u;

        for (const auto& det : detections) {
            int x1 = std::max(0, static_cast<int>(det.box.x1));
            int y1 = std::max(0, static_cast<int>(det.box.y1));
            int x2 = std::min(buf_w - 1, static_cast<int>(det.box.x2));
            int y2 = std::min(buf_h - 1, static_cast<int>(det.box.y2));

            if (x1 > x2 || y1 > y2) continue;

            for (int t = 0; t < thickness; ++t) {
                int row_top = y1 + t;
                if (row_top <= y2) {
                    for (int x = x1; x <= x2; ++x)
                        pixels[row_top * buf_w + x] = green;
                }
                int row_bot = y2 - t;
                if (row_bot >= y1 + thickness) {
                    for (int x = x1; x <= x2; ++x)
                        pixels[row_bot * buf_w + x] = green;
                }
            }

            for (int y = y1 + thickness; y <= y2 - thickness; ++y) {
                for (int t = 0; t < thickness; ++t) {
                    int col_left = x1 + t;
                    if (col_left <= x2)
                        pixels[y * buf_w + col_left] = green;
                    int col_right = x2 - t;
                    if (col_right >= x1 + thickness)
                        pixels[y * buf_w + col_right] = green;
                }
            }
        }
    }

    bool OverlaySink::should_continue() const noexcept {
        return window_open_.load(std::memory_order_relaxed);
    }
}
