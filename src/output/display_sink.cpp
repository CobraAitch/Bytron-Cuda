/**
 * @file display_sink.cpp
 * @brief DisplaySink mailbox handoff, CUDA overlay rendering, and Vulkan present loop.
 */

#include "output/display_sink.hpp"
#include "output/overlay_kernels.cuh"
#include "util/cuda_check.hpp"

#include <spdlog/spdlog.h>
#include <cstdint>
#include <cstring>

namespace bytron {

    DisplaySink::DisplaySink(const std::string& window_name,
                             int capture_w, int capture_h,
                             int max_detections)
        : capture_w_(capture_w)
        , capture_h_(capture_h)
        , window_name_(window_name)
    {
        auto frame_bytes = std::size_t(capture_w) * capture_h * 4;
        auto det_bytes   = std::size_t(max_detections) * sizeof(Detection);

        for (auto& mb : mailbox_) {
            mb.frame_buf   = CudaDeviceBuffer(frame_bytes);
            mb.det_buf     = CudaDeviceBuffer(det_bytes);
            mb.det_pinned  = CudaPinnedBuffer(det_bytes);
        }

        display_thread_ = std::jthread([this](std::stop_token st) { display_loop(st); });
    }

    DisplaySink::~DisplaySink() {
        display_thread_.request_stop();
        cv_.notify_one();
    }

    void DisplaySink::on_frame(
        const FrameContext& ctx, void* device_frame,
        int width, int height, int,
        cudaStream_t, cudaEvent_t copy_done_event
    ) {
        if (!device_frame || display_thread_.get_stop_token().stop_requested()) return;

        auto& mb = mailbox_[write_idx_];

        auto frame_bytes = std::size_t(width) * 4 * height;
        CUDA_CHECK(cudaMemcpyAsync(mb.frame_buf.ptr(), device_frame,
                                    frame_bytes, cudaMemcpyDeviceToDevice,
                                    copy_stream_.get()));

        if (copy_done_event)
            CUDA_CHECK(cudaEventRecord(copy_done_event, copy_stream_.get()));

        mb.copy_done.record(copy_stream_.get());
        mb.width          = width;
        mb.height         = height;
        mb.pitch          = width * 4;
        mb.num_detections = static_cast<int>(ctx.detections.size());

        // Copy detections into pinned memory for truly async H2D transfer later.
        if (mb.num_detections > 0) {
            std::memcpy(mb.det_pinned.ptr(), ctx.detections.data(),
                        mb.num_detections * sizeof(Detection));
        }

        {
            std::lock_guard lock(mtx_);
            pending_idx_ = write_idx_;
        }
        cv_.notify_one();

        write_idx_ = (write_idx_ + 1) % MAILBOX_SLOTS;
    }

    void DisplaySink::display_loop(std::stop_token stoken) {
        auto display = std::make_unique<VulkanDisplay>(
            capture_w_, capture_h_, window_name_.c_str());

        int last_presented = -1;

        while (!stoken.stop_requested()) {
            int slot_idx;
            {
                std::unique_lock lock(mtx_);
                cv_.wait_for(lock, std::chrono::milliseconds(4), [&] {
                    return pending_idx_ >= 0 || stoken.stop_requested();
                });
                if (stoken.stop_requested()) break;
                if (pending_idx_ < 0) {
                    if (last_presented >= 0) {
                        auto& mb = mailbox_[last_presented];
                        display->present(mb.frame_buf.ptr(), mb.width, mb.height,
                                         overlay_stream_.get());
                    } else {
                        display->present(nullptr, 0, 0);
                    }
                    if (display->should_close()) {
                        window_open_ = false;
                        return;
                    }
                    continue;
                }
                slot_idx = pending_idx_;
                pending_idx_ = -1;
            }

            auto& mb = mailbox_[slot_idx];
            auto os = overlay_stream_.get();

            // GPU-side wait: overlay_stream waits for copy_stream's D2D frame copy
            // to complete, without blocking the CPU.
            CUDA_CHECK(cudaStreamWaitEvent(os, mb.copy_done.get(), 0));

            if (mb.num_detections > 0) {
                auto det_bytes = mb.num_detections * sizeof(Detection);
                CUDA_CHECK(cudaMemcpyAsync(mb.det_buf.ptr(), mb.det_pinned.ptr(),
                                            det_bytes, cudaMemcpyHostToDevice, os));

                launch_draw_detection_boxes(
                    mb.frame_buf.as<std::uint8_t>(),
                    mb.width, mb.height, mb.pitch,
                    mb.det_buf.as<const Detection>(),
                    mb.num_detections, 2, os);
            }

            display->present(mb.frame_buf.ptr(), mb.width, mb.height, os);
            last_presented = slot_idx;

            if (display->should_close()) {
                window_open_ = false;
                return;
            }
        }
    }

    bool DisplaySink::should_continue() const noexcept {
        return window_open_.load(std::memory_order_relaxed);
    }

}
