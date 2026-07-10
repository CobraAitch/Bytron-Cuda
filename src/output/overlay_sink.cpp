/**
 * @file overlay_sink.cpp
 * @brief OverlaySink incremental CPU box rendering and Win32 layered window present loop.
 */

#include "output/overlay_sink.hpp"
#include "util/cuda_check.hpp"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstdint>

namespace bytron {

    namespace {
        constexpr int           BOX_THICKNESS = 2;
        constexpr std::uint32_t BOX_COLOR     = 0xFF00FF00u;  // Opaque green (BGRA).
        constexpr std::uint32_t CLEAR_COLOR   = 0x00000000u;  // Fully transparent.

        /**
         * @brief Union of all clamped box rects from both detection sets.
         * @return false when both sets contain no visible boxes.
         *         @p out is a GDI RECT (right/bottom exclusive).
         */
        bool dirty_union(const std::vector<Detection>& a,
                         const std::vector<Detection>& b,
                         int w, int h, RECT& out) {
            int x1 = w, y1 = h, x2 = -1, y2 = -1;

            auto add = [&](const std::vector<Detection>& dets) {
                for (const auto& det : dets) {
                    int bx1 = std::max(0, static_cast<int>(det.box.x1));
                    int by1 = std::max(0, static_cast<int>(det.box.y1));
                    int bx2 = std::min(w - 1, static_cast<int>(det.box.x2));
                    int by2 = std::min(h - 1, static_cast<int>(det.box.y2));
                    if (bx1 > bx2 || by1 > by2) continue;

                    x1 = std::min(x1, bx1);
                    y1 = std::min(y1, by1);
                    x2 = std::max(x2, bx2);
                    y2 = std::max(y2, by2);
                }
            };
            add(a);
            add(b);

            if (x2 < 0) return false;
            out = RECT{ x1, y1, x2 + 1, y2 + 1 };
            return true;
        }
    }

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
        int /*width*/, int /*height*/, int /*pitch*/,
        cudaStream_t stream,
        cudaEvent_t copy_done_event
    ) {
        if (display_thread_.get_stop_token().stop_requested()) return;

        {
            std::lock_guard lock(mtx_);
            latest_dets_.assign(ctx.detections.begin(), ctx.detections.end());
            ++latest_seq_;
        }
        cv_.notify_one();

        // This sink never reads the GPU frame; release the slot's raw_buffer
        // as soon as the work already queued on its stream completes.
        if (copy_done_event)
            CUDA_CHECK(cudaEventRecord(copy_done_event, stream));
    }

    void OverlaySink::display_loop(std::stop_token stoken) {
        auto* overlay = window_.get();

        std::vector<Detection> current;  ///< Boxes to draw this iteration.
        std::vector<Detection> visible;  ///< Boxes on screen (erased next iteration).
        std::uint64_t consumed_seq  = 0;
        bool          first_present = true;

        while (!stoken.stop_requested()) {
            bool has_frame = false;
            {
                std::unique_lock lock(mtx_);
                cv_.wait_for(lock, std::chrono::milliseconds(4), [&] {
                    return latest_seq_ != consumed_seq || stoken.stop_requested();
                });
                if (stoken.stop_requested()) break;

                if (latest_seq_ != consumed_seq) {
                    consumed_seq = latest_seq_;
                    current.assign(latest_dets_.begin(), latest_dets_.end());
                    has_frame = true;
                }
            }

            if (!has_frame) {
                overlay->poll_events();
                if (overlay->should_close()) {
                    window_open_ = false;
                    return;
                }
                continue;
            }

            auto* pixels = overlay->pixels();
            const int w  = overlay->width();
            const int h  = overlay->height();

            // Erase previous boxes by redrawing their outlines transparent,
            // then draw the new set — touches only box outlines instead of
            // clearing the whole monitor-sized buffer every frame.
            render_boxes_cpu(pixels, w, h, visible, BOX_THICKNESS, CLEAR_COLOR);
            render_boxes_cpu(pixels, w, h, current, BOX_THICKNESS, BOX_COLOR);

            RECT dirty{};
            const bool any_change = dirty_union(visible, current, w, h, dirty);

            if (first_present) {
                overlay->present();  // Full upload primes the layered window.
                first_present = false;
            } else if (any_change) {
                overlay->present(&dirty);
            } else {
                overlay->poll_events();  // Nothing on screen changed.
            }

            visible.swap(current);

            if (overlay->should_close()) {
                window_open_ = false;
                return;
            }
        }
    }

    void OverlaySink::render_boxes_cpu(
        void* bgra_buffer, int buf_w, int buf_h,
        const std::vector<Detection>& detections,
        int thickness, std::uint32_t color
    ) {
        auto* pixels = static_cast<std::uint32_t*>(bgra_buffer);

        for (const auto& det : detections) {
            int x1 = std::max(0, static_cast<int>(det.box.x1));
            int y1 = std::max(0, static_cast<int>(det.box.y1));
            int x2 = std::min(buf_w - 1, static_cast<int>(det.box.x2));
            int y2 = std::min(buf_h - 1, static_cast<int>(det.box.y2));

            if (x1 > x2 || y1 > y2) continue;

            for (int t = 0; t < thickness; ++t) {
                int row_top = y1 + t;
                if (row_top <= y2) {
                    std::fill_n(pixels + row_top * buf_w + x1, x2 - x1 + 1, color);
                }
                int row_bot = y2 - t;
                if (row_bot >= y1 + thickness) {
                    std::fill_n(pixels + row_bot * buf_w + x1, x2 - x1 + 1, color);
                }
            }

            for (int y = y1 + thickness; y <= y2 - thickness; ++y) {
                for (int t = 0; t < thickness; ++t) {
                    int col_left = x1 + t;
                    if (col_left <= x2)
                        pixels[y * buf_w + col_left] = color;
                    int col_right = x2 - t;
                    if (col_right >= x1 + thickness)
                        pixels[y * buf_w + col_right] = color;
                }
            }
        }
    }

    bool OverlaySink::should_continue() const noexcept {
        return window_open_.load(std::memory_order_relaxed);
    }
}
