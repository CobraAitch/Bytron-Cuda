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
        // The pipeline gates slot reuse on the D2D copy this stream performs,
        // so it must not be starved by other GPU work. Overlay draw + present
        // stay low priority: they never block inference.
        , copy_stream_(StreamPriority::High)
        , overlay_stream_(StreamPriority::Low)
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

        // Pick a slot the display thread is neither reading nor about to claim.
        // Replacing pending_idx_ below (not the slot itself) gives latest-wins:
        // an unclaimed older frame is simply superseded.
        int idx = 0;
        {
            std::lock_guard lock(mtx_);
            while (idx == reading_idx_ || idx == pending_idx_) ++idx;
        }
        auto& mb = mailbox_[idx];

        // The display's last GPU use of this slot (H2D dets + overlay draw +
        // staging copy) must finish before its buffers are overwritten. That
        // was 2+ presents ago, so this almost never blocks.
        mb.present_done.synchronize();

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

        if (mb.num_detections > 0) {
            std::memcpy(mb.det_pinned.ptr(), ctx.detections.data(),
                        mb.num_detections * sizeof(Detection));
        }

        {
            std::lock_guard lock(mtx_);
            pending_idx_ = idx;
        }
        cv_.notify_one();
    }

    void DisplaySink::display_loop(std::stop_token stoken) {
        auto display = std::make_unique<VulkanDisplay>(
            capture_w_, capture_h_, window_name_.c_str());

        while (!stoken.stop_requested()) {
            int slot_idx;
            {
                std::unique_lock lock(mtx_);
                cv_.wait_for(lock, std::chrono::milliseconds(4), [&] {
                    return pending_idx_ >= 0 || stoken.stop_requested();
                });
                if (stoken.stop_requested()) break;
                if (pending_idx_ < 0) {
                    // No new frame: poll GLFW events (for close button) without
                    // re-presenting. MAILBOX mode retains the last image on screen.
                    // present(nullptr) polls events + returns early, no GPU work.
                    display->present(nullptr, 0, 0);
                    if (display->should_close()) {
                        window_open_ = false;
                        return;
                    }
                    continue;
                }
                slot_idx = pending_idx_;
                pending_idx_ = -1;
                reading_idx_ = slot_idx;  // Claim: writer won't touch this slot now.
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

            // All GPU reads of this slot are enqueued on os; mark the point the
            // writer must wait for before reusing the slot, then release it.
            mb.present_done.record(os);
            {
                std::lock_guard lock(mtx_);
                reading_idx_ = -1;
            }

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
