/**
 * @file pipeline.cpp
 * @brief Pipeline capture loop, inference loop, CUDA graph capture, and frame delivery.
 */

#include "core/pipeline.hpp"
#include "preprocess/preprocess_kernels.cuh"
#include "util/cuda_check.hpp"
#include "util/timer.hpp"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <span>
#include <stdexcept>

namespace bytron {

    Pipeline::Pipeline(const Config& cfg)
        : cfg_(cfg)
        , num_slots_(std::clamp(cfg.num_pipeline_slots, 2, MAX_SLOTS))
        , use_cuda_graphs_(cfg.use_cuda_graphs)
    {
        capture_ = std::make_unique<DesktopCapture>(cfg_.monitor_index);
        if (!capture_->is_open())
            throw std::runtime_error("Failed to open desktop capture");

        TrtEngine::BuildConfig engine_cfg{
            .onnx_path              = cfg_.onnx_path,
            .engine_cache_path      = cfg_.engine_cache_path(),
            .precision              = cfg_.precision,
            .max_batch_size         = 1,
            .opt_batch_size         = 1,
            .max_workspace_bytes    = 1ULL << 30,
            .max_aux_streams        = 4,
            .calibration_data_dir   = cfg_.calibration_images_dir,
            .calibration_batch_size = cfg_.calibration_batch_size,
        };
        engine_ = std::make_unique<TrtEngine>(engine_cfg);

        preprocessor_ = std::make_unique<GpuPreprocessor>(
            engine_->model_width(), engine_->model_height());

        postprocessor_ = std::make_unique<Postprocessor>(
            cfg_.confidence_threshold, engine_->max_detections());

        slots_.resize(num_slots_);
        for (int i = 0; i < num_slots_; ++i) {
            slots_[i].index = i;
            slots_[i].allocate(cfg_.max_frame_width, cfg_.max_frame_height,
                               engine_->model_width(), engine_->model_height(),
                               engine_->max_detections());
            slots_[i].trt_context = engine_->create_context();
        }

        capture_mailbox_.allocate(cfg_.max_frame_width, cfg_.max_frame_height);

        graph_state_.resize(num_slots_);
        slot_warmed_up_.resize(num_slots_, false);
        timing_.resize(num_slots_);

        spdlog::info("Pipeline initialized: {} slots, model {}x{}, max {} detections, "
                     "CUDA graphs: {}, desktop BGRA capture, three-thread architecture",
                     num_slots_, engine_->model_width(), engine_->model_height(),
                     engine_->max_detections(), use_cuda_graphs_ ? "ON" : "OFF");
    }

    Pipeline::~Pipeline() {
        stop();
        capture_thread_   = {};
        inference_thread_ = {};

        for (auto& gs : graph_state_) {
            if (gs.exec)  cudaGraphExecDestroy(gs.exec);
            if (gs.graph) cudaGraphDestroy(gs.graph);
        }
    }

    void Pipeline::run(OutputSink& sink) {
        frame_counter_    = 0;
        completed_frames_ = 0;
        total_latency_ms_ = 0.0;

        capture_thread_   = std::jthread([this](std::stop_token st) { capture_loop(st); });
        inference_thread_ = std::jthread(
            [this, &sink](std::stop_token st) { inference_loop(st, sink); });

        inference_thread_.join();

        capture_thread_.request_stop();
        capture_mailbox_.cv.notify_all();
        capture_thread_.join();
    }

    void Pipeline::stop() noexcept {
        if (capture_thread_.joinable())   capture_thread_.request_stop();
        if (inference_thread_.joinable()) inference_thread_.request_stop();
        capture_mailbox_.cv.notify_all();
    }

    void Pipeline::capture_loop(std::stop_token stoken) {
        CUDA_CHECK(cudaSetDevice(0));
        spdlog::debug("Capture thread started");

        int write_idx = 0;

        while (!stoken.stop_requested()) {
            auto& entry = capture_mailbox_.entries[write_idx];

            // Wait until the inference thread has finished reading from this entry
            // before overwriting it with a new capture frame.
            CUDA_CHECK(cudaStreamWaitEvent(capture_stream_.get(), entry.read_done.get(), 0));

            auto result = capture_->acquire_frame_blocking(
                entry.frame_buf, capture_stream_.get(), 16);

            switch (result) {
                case AcquireResult::kNewFrame: {
                    entry.capture_done.record(capture_stream_.get());
                    entry.width  = capture_->width();
                    entry.height = capture_->height();

                    {
                        std::lock_guard lock(capture_mailbox_.mtx);
                        capture_mailbox_.latest_idx = write_idx;
                        ++capture_mailbox_.sequence;
                    }
                    capture_mailbox_.cv.notify_one();

                    write_idx = (write_idx + 1) % CaptureMailbox::NUM_ENTRIES;
                    break;
                }

                case AcquireResult::kTimeout:
                    continue;

                case AcquireResult::kAccessLost:
                    spdlog::warn("Capture: access lost, re-acquiring...");
                    for (int attempt = 0; attempt < 20 && !stoken.stop_requested(); ++attempt) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        if (capture_->reacquire()) break;
                    }
                    if (!capture_->is_open()) {
                        spdlog::error("Capture: failed to re-acquire after retries");
                        capture_mailbox_.cv.notify_all();
                        return;
                    }
                    continue;

                case AcquireResult::kError:
                    spdlog::error("Capture: unrecoverable error, exiting");
                    capture_mailbox_.cv.notify_all();
                    return;
            }
        }

        spdlog::debug("Capture thread exiting");
    }

    void Pipeline::inference_loop(std::stop_token stoken, OutputSink& sink) {
        CUDA_CHECK(cudaSetDevice(0));
        spdlog::debug("Inference thread started");

        Timer fps_timer;
        int next_slot = 0;
        std::vector<bool> slot_in_flight(num_slots_, false);

        while (!stoken.stop_requested() && sink.should_continue()) {
            int read_idx;
            {
                std::unique_lock lock(capture_mailbox_.mtx);
                if (!capture_mailbox_.cv.wait(lock, stoken, [&] {
                    return capture_mailbox_.sequence > capture_mailbox_.consumed_sequence;
                })) {
                    break;
                }

                read_idx = capture_mailbox_.latest_idx;
                capture_mailbox_.consumed_sequence = capture_mailbox_.sequence;
            }

            auto& slot = slots_[next_slot % num_slots_];
            int idx = slot.index;

            if (slot_in_flight[idx]) {
                slot.ready_event.synchronize();

                float frame_ms = CudaEvent::elapsed_ms(timing_[idx].start, timing_[idx].end);
                total_latency_ms_ += frame_ms;
                ++completed_frames_;

                harvest_and_deliver(slot, sink);
                slot_in_flight[idx] = false;
            }

            auto stream = slot.stream.get();
            timing_[idx].start.record(stream);

            // Guard: wait until the display sink has finished copying this slot's
            // raw_buffer before we overwrite it with a new capture frame.
            // Skipped for the first N frames (no prior frame in this slot yet).
            if (frame_counter_ >= static_cast<std::uint64_t>(num_slots_))
                CUDA_CHECK(cudaStreamWaitEvent(stream, slot.display_copy_done.get(), 0));

            auto& entry = capture_mailbox_.entries[read_idx];
            CUDA_CHECK(cudaStreamWaitEvent(stream, entry.capture_done.get(), 0));

            auto frame_bytes = std::size_t(entry.width) * entry.height * 4;
            CUDA_CHECK(cudaMemcpyAsync(
                slot.raw_buffer.ptr(), entry.frame_buf.ptr(),
                frame_bytes, cudaMemcpyDeviceToDevice, stream));

            entry.read_done.record(stream);

            int w = entry.width;
            int h = entry.height;

            slot.frame_ctx.frame_id   = frame_counter_++;
            slot.frame_ctx.slot_index = idx;

            if (use_cuda_graphs_ && slot_warmed_up_[idx]) {
                auto& gs = graph_state_[idx];

                if (!gs.captured || gs.captured_w != w || gs.captured_h != h) {
                    capture_graph(slot, w, h);
                    spdlog::debug("CUDA graph re-captured for slot {} ({}x{} -> {}x{})",
                                  idx, gs.captured_w, gs.captured_h, w, h);
                }

                auto params = compute_letterbox(w, h, engine_->model_width(), engine_->model_height());
                slot.frame_ctx.src_width  = w;
                slot.frame_ctx.src_height = h;
                slot.frame_ctx.scale = params.scale;
                slot.frame_ctx.pad_x = static_cast<float>(params.pad_x);
                slot.frame_ctx.pad_y = static_cast<float>(params.pad_y);

                enqueue_graph(slot);
            } else {
                preprocessor_->preprocess_bgra(
                    slot.raw_buffer, w, h, w * 4,
                    slot.input_buffer, slot.frame_ctx, stream);

                auto& ctx = slot.trt_context;
                ctx->setInputShape(engine_->input_name(),
                    nvinfer1::Dims4{1, 3, engine_->model_height(), engine_->model_width()});
                ctx->setTensorAddress(engine_->input_name(), slot.input_buffer.ptr());
                ctx->setTensorAddress(engine_->output_name(), slot.output_buffer.ptr());

                if (!ctx->enqueueV3(stream))
                    throw std::runtime_error("TRT enqueueV3 failed on slot " + std::to_string(idx));

                auto output_bytes = std::size_t(engine_->max_detections()) * 6 * sizeof(float);
                slot.output_buffer.copy_to_host_async(slot.output_pinned.ptr(), output_bytes, stream);

                // Warm-up: run one eager pass before graph capture to let TRT
                // auto-tune kernels. Sync ensures all work completes before capture.
                if (use_cuda_graphs_ && !slot_warmed_up_[idx]) {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    slot_warmed_up_[idx] = true;
                    capture_graph(slot, w, h);
                    spdlog::debug("CUDA graph captured for slot {} ({}x{})", idx, w, h);
                }
            }

            timing_[idx].end.record(stream);
            slot.ready_event.record(stream);
            slot_in_flight[idx] = true;
            ++next_slot;

            if (completed_frames_ > 0 && completed_frames_ % 100 == 0) {
                auto frames = completed_frames_.load();
                double elapsed = fps_timer.elapsed_ms();
                spdlog::info("Frames: {}  FPS: {:.1f}  Avg latency: {:.2f} ms",
                             frames, (frames * 1000.0) / elapsed,
                             total_latency_ms_.load() / frames);
            }
        }

        for (int i = 0; i < num_slots_; ++i) {
            if (slot_in_flight[i]) {
                slots_[i].ready_event.synchronize();
                total_latency_ms_ += CudaEvent::elapsed_ms(timing_[i].start, timing_[i].end);
                ++completed_frames_;
                harvest_and_deliver(slots_[i], sink);
            }
        }

        auto final_frames = completed_frames_.load();
        double elapsed = fps_timer.elapsed_ms();
        spdlog::info("Pipeline finished: {} frames, {:.1f} FPS, {:.2f} ms avg latency",
                     final_frames, (final_frames * 1000.0) / elapsed,
                     final_frames > 0 ? total_latency_ms_.load() / final_frames : 0.0);
    }

    void Pipeline::capture_graph(PipelineSlot& slot, int src_w, int src_h) {
        auto stream = slot.stream.get();
        int idx = slot.index;
        auto& gs = graph_state_[idx];

        if (gs.exec)  { cudaGraphExecDestroy(gs.exec);  gs.exec  = nullptr; }
        if (gs.graph) { cudaGraphDestroy(gs.graph);      gs.graph = nullptr; }

        auto& ctx = slot.trt_context;
        ctx->setInputShape(engine_->input_name(),
                           nvinfer1::Dims4{1, 3, engine_->model_height(), engine_->model_width()});
        ctx->setTensorAddress(engine_->input_name(), slot.input_buffer.ptr());
        ctx->setTensorAddress(engine_->output_name(), slot.output_buffer.ptr());

        // cudaStreamCaptureModeGlobal: any kernel launch on any stream while
        // capture is active is included in the graph. Required because TRT
        // may launch kernels on internal auxiliary streams.
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

        launch_fused_preprocess_bgra(
            static_cast<const std::uint8_t*>(slot.raw_buffer.ptr()),
            static_cast<float*>(slot.input_buffer.ptr()),
            src_w, src_h, src_w * 4,
            engine_->model_width(), engine_->model_height(),
            stream);

        if (!ctx->enqueueV3(stream)) {
            // Abort graph capture before throwing to leave the stream in a valid state.
            cudaStreamEndCapture(stream, &gs.graph);
            throw std::runtime_error("TRT enqueueV3 failed during graph capture on slot "
                                     + std::to_string(idx));
        }

        auto output_bytes = std::size_t(engine_->max_detections()) * 6 * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(slot.output_pinned.ptr(), slot.output_buffer.ptr(),
                                    output_bytes, cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamEndCapture(stream, &gs.graph));
        CUDA_CHECK(cudaGraphInstantiate(&gs.exec, gs.graph, 0));

        gs.captured   = true;
        gs.captured_w = src_w;
        gs.captured_h = src_h;
    }

    void Pipeline::enqueue_graph(PipelineSlot& slot) {
        CUDA_CHECK(cudaGraphLaunch(graph_state_[slot.index].exec, slot.stream.get()));
    }

    void Pipeline::harvest_and_deliver(PipelineSlot& slot, OutputSink& sink) {
        postprocessor_->process(
            {slot.output_pinned.as<float>(), static_cast<std::size_t>(engine_->max_detections()) * 6},
            slot.frame_ctx);

        int w = slot.frame_ctx.src_width;
        int h = slot.frame_ctx.src_height;
        sink.on_frame(slot.frame_ctx, slot.raw_buffer.ptr(),
                      w, h, w * 4, slot.stream.get(), slot.display_copy_done.get());
    }

    Pipeline::Metrics Pipeline::metrics() const noexcept {
        auto frames = completed_frames_.load();
        return Metrics{
            .avg_latency_ms = frames > 0 ? total_latency_ms_.load() / frames : 0.0,
            .total_frames   = frames
        };
    }

    int Pipeline::capture_width()  const noexcept { return capture_ ? capture_->width()  : 0; }
    int Pipeline::capture_height() const noexcept { return capture_ ? capture_->height() : 0; }

}
