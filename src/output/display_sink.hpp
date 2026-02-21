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
     * @brief Copies frames to a 2-slot mailbox, draws detection overlays with CUDA,
     *        and presents via VulkanDisplay on a dedicated thread.
     *
     * Uses copy_stream_ for D2D frame copy and overlay_stream_ for detection
     * box rendering + Vulkan present. A condition_variable_any mailbox decouples
     * inference from display.
     *
     * @note Thread safety: on_frame() from inference thread, display_loop() on
     *       dedicated jthread. Mutex + CV for metadata, CUDA events for GPU data.
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

        /** @brief Double-buffered frame slot for inference-to-display handoff. */
        struct FrameMailbox {
            CudaDeviceBuffer       frame_buf;
            CudaDeviceBuffer       det_buf;
            CudaEvent              copy_done;
            int                    width          = 0;
            int                    height         = 0;
            int                    pitch          = 0;
            int                    num_detections = 0;
            std::vector<Detection> detections;
        };

        static constexpr int MAILBOX_SLOTS = 2;
        FrameMailbox mailbox_[MAILBOX_SLOTS];
        int          write_idx_ = 0;

        std::mutex                  mtx_;
        std::condition_variable_any cv_;
        int                         pending_idx_ = -1;

        std::atomic<bool> window_open_{true};
        std::jthread      display_thread_;
    };

}
