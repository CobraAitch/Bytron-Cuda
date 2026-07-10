/**
 * @file display_sink.hpp
 * @brief Full-frame Vulkan display sink with GPU overlay rendering.
 */

#pragma once

#include "output/output_sink.hpp"
#include "output/vulkan_display.hpp"
#include "core/cuda_buffer.hpp"
#include "core/cuda_stream.hpp"
#include "core/cuda_event.hpp"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

namespace bytron {

    /**
     * @brief Copies frames to a 3-slot mailbox, draws detection overlays with CUDA,
     *        and presents via VulkanDisplay on a dedicated thread.
     *
     * Uses copy_stream_ for D2D frame copy and overlay_stream_ for detection
     * box rendering + Vulkan present. A condition_variable_any mailbox decouples
     * inference from display with latest-wins semantics: a pending frame the
     * display hasn't claimed yet is superseded by a newer one.
     *
     * Slot ownership: the writer never selects the slot the display thread is
     * reading (reading_idx_) nor the currently pending slot (pending_idx_), so
     * with 3 slots there is always a free slot and no metadata races. GPU reuse
     * of a slot is gated on its present_done event.
     *
     * @note Thread safety: on_frame() from inference thread, display_loop() on
     *       dedicated jthread. Mutex + CV for slot ownership and metadata,
     *       CUDA events for GPU data.
     */
    class DisplaySink : public OutputSink {
    public:
        DisplaySink(const std::string& window_name,
                    int capture_w, int capture_h,
                    int max_detections = 300);
        ~DisplaySink();

        DisplaySink(const DisplaySink&) = delete;
        DisplaySink& operator=(const DisplaySink&) = delete;

        void on_frame(const FrameContext& ctx, void* device_frame,
                      int width, int height, int pitch,
                      cudaStream_t stream,
                      cudaEvent_t copy_done_event = nullptr) override;

        [[nodiscard]] bool should_continue() const noexcept override;

    private:
        void display_loop(std::stop_token stoken);

        int         capture_w_;
        int         capture_h_;
        std::string window_name_;

        CudaStream copy_stream_;
        CudaStream overlay_stream_;

        /** @brief Frame slot for inference-to-display handoff. */
        struct FrameMailbox {
            CudaDeviceBuffer  frame_buf;
            CudaDeviceBuffer  det_buf;
            CudaPinnedBuffer  det_pinned;    ///< Pinned host copy for truly async H2D transfer.
            CudaEvent         copy_done;     ///< D2D frame copy finished (writer side).
            CudaEvent         present_done;  ///< Display's last GPU use of this slot finished.
            int               width          = 0;
            int               height         = 0;
            int               pitch          = 0;
            int               num_detections = 0;
        };

        /// 3 slots: one being read by the display, one pending, one free for the
        /// writer — guarantees a free slot without ever overwriting a live one.
        static constexpr int MAILBOX_SLOTS = 3;
        FrameMailbox mailbox_[MAILBOX_SLOTS];

        std::mutex                  mtx_;
        std::condition_variable_any cv_;
        int                         pending_idx_ = -1;  ///< Newest unclaimed frame (-1 = none).
        int                         reading_idx_ = -1;  ///< Slot the display thread is using (-1 = none).

        std::atomic<bool> window_open_{true};
        std::jthread      display_thread_;
    };

}
