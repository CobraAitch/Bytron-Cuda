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

#include <windows.h>

namespace bytron {

    Pipeline::Pipeline(const Config& cfg)
        : cfg_(cfg)
        , num_slots_(std::clamp(cfg.num_pipeline_slots, 2, MAX_SLOTS))
        , use_cuda_graphs_(cfg.use_cuda_graphs)
    {
        CUDA_CHECK(cudaSetDevice(cfg_.gpu_device));

        capture_ = make_capture_source(cfg_.capture_backend, cfg_.monitor_index);
        if (!capture_->is_open())
            throw std::runtime_error("Failed to open screen capture");

        // Raise the CUDA device-level L2 persistent cache limit so TRT can
        // actually use the full requested size.  Without this, the driver
        // default is often smaller (e.g. 9 MB on RTX 4070 Ti) and TRT logs a
        // warning about the shortfall.
        std::size_t effective_l2 = cfg_.l2_cache_bytes;
        if (effective_l2 > 0) {
            int max_l2 = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(
                &max_l2, cudaDevAttrMaxPersistingL2CacheSize, cfg_.gpu_device));

            effective_l2 = std::min(effective_l2, static_cast<std::size_t>(max_l2));
            if (effective_l2 > 0) {
                CUDA_CHECK(cudaDeviceSetLimit(
                    cudaLimitPersistingL2CacheSize, effective_l2));
                spdlog::info("L2 persistent cache: requested {} MB, hw max {} MB, set {} MB",
                             cfg_.l2_cache_bytes >> 20, max_l2 >> 20, effective_l2 >> 20);
            } else {
                spdlog::info("GPU does not support L2 persistent cache, disabled");
            }
        }

        TrtEngine::BuildConfig engine_cfg{
            .onnx_path              = cfg_.onnx_path,
            .engine_cache_path      = cfg_.engine_cache_path(),
            .precision              = cfg_.precision,
            .max_batch_size         = 1,
            .opt_batch_size         = 1,
            .max_workspace_bytes    = cfg_.max_workspace_bytes,
            .max_aux_streams        = cfg_.max_aux_streams,
            .builder_opt_level      = cfg_.builder_opt_level,
            .l2_cache_bytes         = effective_l2,
            .calibration_data_dir   = cfg_.calibration_images_dir,
            .calibration_batch_size = cfg_.calibration_batch_size,
        };
        engine_ = std::make_unique<TrtEngine>(engine_cfg);

        preprocessor_ = std::make_unique<GpuPreprocessor>(
            engine_->model_width(), engine_->model_height());

        postprocessor_ = std::make_unique<Postprocessor>(
            cfg_.confidence_threshold, engine_->max_detections(),
            engine_->output_stride(), cfg_.detection_schema);

        slots_.resize(num_slots_);
        for (int i = 0; i < num_slots_; ++i) {
            slots_[i].index  = i;
            slots_[i].stream = CudaStream(StreamPriority::High);
            slots_[i].allocate(cfg_.max_frame_width, cfg_.max_frame_height,
                               engine_->model_width(), engine_->model_height(),
                               engine_->max_detections(), engine_->output_stride());
            slots_[i].trt_context = engine_->create_context();

            // Input shape and tensor addresses are fixed per slot; TRT keeps
            // them across enqueues, so set them once here instead of per frame.
            auto& ctx = slots_[i].trt_context;
            if (!ctx->setInputShape(engine_->input_name(),
                    nvinfer1::Dims4{1, 3, engine_->model_height(), engine_->model_width()}))
                throw std::runtime_error("Failed to set TRT input shape on slot " + std::to_string(i));
            if (!ctx->setTensorAddress(engine_->input_name(), slots_[i].input_buffer.ptr())
                || !ctx->setTensorAddress(engine_->output_name(), slots_[i].output_buffer.ptr()))
                throw std::runtime_error("Failed to set TRT tensor addresses on slot " + std::to_string(i));
        }

        capture_mailbox_.allocate(cfg_.max_frame_width, cfg_.max_frame_height);

        graph_cache_.reserve(num_slots_);
        for (int i = 0; i < num_slots_; ++i)
            graph_cache_.push_back(std::make_unique<SlotGraphCache>());
        slot_warmed_up_.resize(num_slots_, false);
        slot_eager_exec_.resize(num_slots_, true);
        timing_.resize(num_slots_);

        if (cfg_.max_aux_streams > 0)
            create_aux_streams();

        spdlog::info("Pipeline initialized: {} slots, model {}x{}, max {} detections, "
                     "CUDA graphs: {}, aux streams: {}, stream priorities: HIGH(inference)/NORMAL(capture), "
                     "capture: {}, three-thread architecture",
                     num_slots_, engine_->model_width(), engine_->model_height(),
                     engine_->max_detections(), use_cuda_graphs_ ? "ON" : "OFF",
                     num_aux_streams_, capture_->backend_name());
    }

    Pipeline::~Pipeline() {
        stop();
        capture_thread_   = {};
        inference_thread_ = {};
        // GraphState destructors handle cudaGraphDestroy / cudaGraphExecDestroy via RAII.
        destroy_aux_streams();
    }

    void Pipeline::store_exception(std::exception_ptr ep) noexcept {
        std::lock_guard lock(exception_mtx_);
        if (!thread_exception_)
            thread_exception_ = ep;
    }

    void Pipeline::run(OutputSink& sink) {
        frame_counter_    = 0;
        completed_frames_ = 0;
        eager_frames_     = 0;
        total_latency_us_ = 0;
        last_pre_ms_      = 0.0f;
        last_trt_ms_      = 0.0f;
        last_d2h_ms_      = 0.0f;
        thread_exception_ = nullptr;
        capture_dead_.store(false, std::memory_order_relaxed);

        capture_thread_   = std::jthread([this](std::stop_token st) { capture_loop(st); });
        inference_thread_ = std::jthread(
            [this, &sink](std::stop_token st) { inference_loop(st, sink); });

        inference_thread_.join();

        capture_thread_.request_stop();
        capture_mailbox_.cv.notify_all();
        capture_thread_.join();

        if (thread_exception_)
            std::rethrow_exception(thread_exception_);
    }

    void Pipeline::stop() noexcept {
        if (capture_thread_.joinable())   capture_thread_.request_stop();
        if (inference_thread_.joinable()) inference_thread_.request_stop();
        capture_mailbox_.cv.notify_all();
    }

    void Pipeline::capture_loop(std::stop_token stoken) {
        try {
            CUDA_CHECK(cudaSetDevice(cfg_.gpu_device));
            // Capture feeds the whole pipeline; don't let background CPU load
            // add scheduling jitter to frame acquisition.
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
            spdlog::debug("Capture thread started");

            int write_idx = 0;

            while (!stoken.stop_requested()) {
                auto& entry = capture_mailbox_.entries[write_idx];

                CUDA_CHECK(cudaStreamWaitEvent(capture_stream_.get(), entry.read_done.get(), 0));

                auto result = capture_->acquire_frame(
                    entry.frame_buf, capture_stream_.get(), 16);

                switch (result) {
                    case AcquireResult::kNewFrame: {
                        entry.capture_done.record(capture_stream_.get());

                        // Entry metadata is written under the mailbox mutex so the
                        // inference thread never reads torn width/height/pitch if
                        // capture laps the mailbox.
                        {
                            std::lock_guard lock(capture_mailbox_.mtx);
                            entry.width  = capture_->width();
                            entry.height = capture_->height();
                            entry.pitch  = capture_->pitch();
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
                            capture_dead_.store(true, std::memory_order_release);
                            capture_mailbox_.cv.notify_all();
                            return;
                        }
                        continue;

                    case AcquireResult::kError:
                        spdlog::error("Capture: unrecoverable error, exiting");
                        capture_dead_.store(true, std::memory_order_release);
                        capture_mailbox_.cv.notify_all();
                        return;
                }
            }

            spdlog::debug("Capture thread exiting");
        } catch (...) {
            store_exception(std::current_exception());
            capture_dead_.store(true, std::memory_order_release);
            capture_mailbox_.cv.notify_all();
        }
    }

    void Pipeline::finalize_slot(PipelineSlot& slot, OutputSink& sink) {
        const int idx = slot.index;
        slot.ready_event.synchronize();

        float frame_ms = CudaEvent::elapsed_ms(timing_[idx].start, timing_[idx].end);
        {
            std::lock_guard lock(metrics_mtx_);
            total_latency_us_ += static_cast<std::uint64_t>(frame_ms * 1000.0f);
            ++completed_frames_;

            // Per-stage breakdown: snapshot latest eager-frame timings.
            // Graph replays fuse all stages so intermediate events are invalid.
            if (slot_eager_exec_[idx]) {
                last_pre_ms_ = CudaEvent::elapsed_ms(timing_[idx].start, timing_[idx].preprocess_done);
                last_trt_ms_ = CudaEvent::elapsed_ms(timing_[idx].preprocess_done, timing_[idx].trt_done);
                last_d2h_ms_ = CudaEvent::elapsed_ms(timing_[idx].trt_done, timing_[idx].end);
                ++eager_frames_;
            }
        }

        harvest_and_deliver(slot, sink);
    }

    void Pipeline::launch_inference(PipelineSlot& slot, int w, int h, int pitch) {
        const int idx = slot.index;
        auto stream = slot.stream.get();
        auto key = res_key(w, h);

        // Periodically force an eager frame to refresh per-stage timing
        // breakdown. At ~275 FPS one eager frame every 500 is <0.1% overhead.
        static constexpr std::uint64_t PROFILE_INTERVAL = 500;
        const bool force_profile = use_cuda_graphs_ && slot_warmed_up_[idx]
                                && slot.frame_ctx.frame_id % PROFILE_INTERVAL == 0;

        if (use_cuda_graphs_ && slot_warmed_up_[idx] && !force_profile) {
            auto& cache = *graph_cache_[idx];
            auto it = cache.map.find(key);

            if (it == cache.map.end()) {
                if (static_cast<int>(cache.map.size()) >= MAX_CACHED_GRAPHS)
                    cache.evict_oldest();

                auto& gs = cache.map[key];
                capture_graph(slot, w, h, pitch, gs);
                spdlog::debug("CUDA graph captured for slot {} ({}x{}), cache size {}",
                              idx, w, h, cache.map.size());
                it = cache.map.find(key);
            }
            cache.touch(key);

            auto params = compute_letterbox(w, h, engine_->model_width(), engine_->model_height());
            slot.frame_ctx.src_width  = w;
            slot.frame_ctx.src_height = h;
            slot.frame_ctx.src_pitch  = pitch;
            slot.frame_ctx.scale = params.scale;
            slot.frame_ctx.pad_x = static_cast<float>(params.pad_x);
            slot.frame_ctx.pad_y = static_cast<float>(params.pad_y);

            CUDA_CHECK(cudaGraphLaunch(it->second.exec, stream));
            slot_eager_exec_[idx] = false;
            return;
        }

        preprocessor_->preprocess_bgra(
            slot.raw_buffer, w, h, pitch,
            slot.input_buffer, slot.frame_ctx, stream);

        timing_[idx].preprocess_done.record(stream);

        // Input shape and tensor addresses were bound once at construction.
        if (!slot.trt_context->enqueueV3(stream))
            throw std::runtime_error("TRT enqueueV3 failed on slot " + std::to_string(idx));

        // Join aux streams so trt_done reflects full inference time
        // including aux stream work. Without this, trt_done fires when
        // the main stream's portion completes and the D2H memcpy
        // absorbs residual aux inference time.
        if (num_aux_streams_ > 0) {
            auto& aux  = slot_aux_streams_[idx];
            auto& sync = slot_aux_sync_events_[idx];
            for (int a = 0; a < num_aux_streams_; ++a) {
                sync[a].record(aux[a]);
                CUDA_CHECK(cudaStreamWaitEvent(stream, sync[a].get(), 0));
            }
        }

        timing_[idx].trt_done.record(stream);

        auto output_bytes = std::size_t(engine_->max_detections()) * engine_->output_stride() * sizeof(float);
        slot.output_buffer.copy_to_host_async(slot.output_pinned.ptr(), output_bytes, stream);

        // Warm-up frames include TRT auto-tuning which distorts
        // per-stage timings, so exclude them from profiling metrics.
        slot_eager_exec_[idx] = !(use_cuda_graphs_ && !slot_warmed_up_[idx]);

        // Warm-up: run one eager pass before graph capture to let TRT
        // auto-tune kernels. Sync ensures all work completes before capture.
        if (use_cuda_graphs_ && !slot_warmed_up_[idx]) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            slot_warmed_up_[idx] = true;
            auto& cache = *graph_cache_[idx];
            auto& gs = cache.map[key];
            capture_graph(slot, w, h, pitch, gs);
            cache.touch(key);
            spdlog::debug("CUDA graph captured for slot {} ({}x{})", idx, w, h);
        }
    }

    void Pipeline::inference_loop(std::stop_token stoken, OutputSink& sink) {
      try {
        CUDA_CHECK(cudaSetDevice(cfg_.gpu_device));
        // Inference paces the whole pipeline; shield it from CPU scheduling jitter.
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
        spdlog::debug("Inference thread started");

        Timer fps_timer;
        int next_slot = 0;
        std::vector<bool> slot_in_flight(num_slots_, false);

        while (!stoken.stop_requested() && sink.should_continue()) {
            // Wait until at least one unconsumed capture exists (or capture died).
            {
                std::unique_lock lock(capture_mailbox_.mtx);
                if (!capture_mailbox_.cv.wait(lock, stoken, [&] {
                    return capture_mailbox_.sequence > capture_mailbox_.consumed_sequence
                        || capture_dead_.load(std::memory_order_acquire);
                })) {
                    break;
                }
                if (capture_dead_.load(std::memory_order_acquire)
                    && capture_mailbox_.sequence <= capture_mailbox_.consumed_sequence)
                    break;
            }

            auto& slot = slots_[next_slot];

            // Finalizing the oldest in-flight slot can block on the GPU, so do
            // it *before* choosing which capture to process: the mailbox may
            // receive newer frames while we wait.
            if (slot_in_flight[slot.index]) {
                finalize_slot(slot, sink);
                slot_in_flight[slot.index] = false;
            }

            // Consume the freshest capture entry (latest-wins mailbox semantics).
            int read_idx, w, h, pitch;
            {
                std::lock_guard lock(capture_mailbox_.mtx);
                read_idx = capture_mailbox_.latest_idx;
                capture_mailbox_.consumed_sequence = capture_mailbox_.sequence;
                const auto& e = capture_mailbox_.entries[read_idx];
                w     = e.width;
                h     = e.height;
                pitch = e.pitch;
            }

            auto stream = slot.stream.get();
            timing_[slot.index].start.record(stream);

            // Guard: wait until the display sink has finished copying this slot's
            // raw_buffer before we overwrite it with a new capture frame.
            // Skipped for the first N frames (no prior frame in this slot yet).
            if (frame_counter_ >= static_cast<std::uint64_t>(num_slots_))
                CUDA_CHECK(cudaStreamWaitEvent(stream, slot.display_copy_done.get(), 0));

            auto& entry = capture_mailbox_.entries[read_idx];
            CUDA_CHECK(cudaStreamWaitEvent(stream, entry.capture_done.get(), 0));

            auto frame_bytes = std::size_t(w) * h * 4;
            CUDA_CHECK(cudaMemcpyAsync(
                slot.raw_buffer.ptr(), entry.frame_buf.ptr(),
                frame_bytes, cudaMemcpyDeviceToDevice, stream));

            entry.read_done.record(stream);

            slot.frame_ctx.frame_id   = frame_counter_++;
            slot.frame_ctx.slot_index = slot.index;

            launch_inference(slot, w, h, pitch);

            timing_[slot.index].end.record(stream);
            slot.ready_event.record(stream);
            slot_in_flight[slot.index] = true;
            next_slot = (next_slot + 1) % num_slots_;

            if (completed_frames_ > 0 && completed_frames_ % 100 == 0) {
                std::lock_guard lock(metrics_mtx_);
                double elapsed = fps_timer.elapsed_ms();
                double avg_ms  = total_latency_us_ / (completed_frames_ * 1000.0);

                if (eager_frames_ > 0) {
                    spdlog::info("Frames: {}  FPS: {:.1f}  Avg: {:.2f} ms "
                                 "[pre {:.2f} | trt {:.2f} | d2h {:.2f}]",
                                 completed_frames_, (completed_frames_ * 1000.0) / elapsed, avg_ms,
                                 last_pre_ms_, last_trt_ms_, last_d2h_ms_);
                } else {
                    spdlog::info("Frames: {}  FPS: {:.1f}  Avg: {:.2f} ms (graph replay)",
                                 completed_frames_, (completed_frames_ * 1000.0) / elapsed, avg_ms);
                }
            }
        }

        // Drain in-flight slots in FIFO order so frames are delivered oldest-first.
        for (int n = 0; n < num_slots_; ++n) {
            int i = (next_slot + n) % num_slots_;
            if (slot_in_flight[i])
                finalize_slot(slots_[i], sink);
        }

        {
            std::lock_guard lock(metrics_mtx_);
            double elapsed = fps_timer.elapsed_ms();
            double avg_ms  = completed_frames_ > 0 ? total_latency_us_ / (completed_frames_ * 1000.0) : 0.0;

            if (eager_frames_ > 0) {
                spdlog::info("Pipeline finished: {} frames, {:.1f} FPS, {:.2f} ms avg "
                             "[pre {:.2f} | trt {:.2f} | d2h {:.2f}]",
                             completed_frames_, (completed_frames_ * 1000.0) / elapsed, avg_ms,
                             last_pre_ms_, last_trt_ms_, last_d2h_ms_);
            } else {
                spdlog::info("Pipeline finished: {} frames, {:.1f} FPS, {:.2f} ms avg latency",
                             completed_frames_, (completed_frames_ * 1000.0) / elapsed, avg_ms);
            }
        }
      } catch (...) {
          store_exception(std::current_exception());
          // Request capture thread to stop so run() can join and rethrow.
          if (capture_thread_.joinable()) capture_thread_.request_stop();
          capture_mailbox_.cv.notify_all();
      }
    }

    void Pipeline::create_aux_streams() {
        num_aux_streams_ = cfg_.max_aux_streams;
        slot_aux_streams_.resize(num_slots_);
        slot_aux_sync_events_.resize(num_slots_);

        // Aux streams carry TRT layer-parallel kernels on the critical path:
        // create them non-blocking (never serialize with the legacy default
        // stream) and at the same high priority as the slot streams.
        int lo = 0, hi = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));

        for (int i = 0; i < num_slots_; ++i) {
            slot_aux_streams_[i].resize(num_aux_streams_);
            for (int j = 0; j < num_aux_streams_; ++j)
                CUDA_CHECK(cudaStreamCreateWithPriority(
                    &slot_aux_streams_[i][j], cudaStreamNonBlocking, hi));

            // Non-timing events for joining aux stream work back to the
            // main stream before profiling event recording.
            slot_aux_sync_events_[i].reserve(num_aux_streams_);
            for (int j = 0; j < num_aux_streams_; ++j)
                slot_aux_sync_events_[i].emplace_back(false);

            // Tell TRT to use our managed aux streams instead of its internal ones.
            // During graph capture these will be forked from the capture stream,
            // ensuring all TRT kernel launches land on captured streams.
            slots_[i].trt_context->setAuxStreams(
                slot_aux_streams_[i].data(), num_aux_streams_);
        }

        spdlog::debug("Created {} aux streams per slot with sync events",
                       num_aux_streams_);
    }

    void Pipeline::destroy_aux_streams() noexcept {
        slot_aux_sync_events_.clear();
        for (auto& slot_streams : slot_aux_streams_)
            for (auto s : slot_streams)
                cudaStreamDestroy(s);
        slot_aux_streams_.clear();
    }

    void Pipeline::capture_graph(PipelineSlot& slot, int src_w, int src_h, int src_pitch, GraphState& gs) {
        auto stream = slot.stream.get();

        // Input shape and tensor addresses were bound once at construction.

        // Relaxed mode: only streams explicitly forked from the capture stream
        // are included in the graph. This is required for TRT multi-stream:
        // we fork our managed aux streams below so TRT kernels are captured,
        // while any other incidental stream activity doesn't cause failures.
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));

        // Fork: make each aux stream a dependency of the capture stream so that
        // TRT kernel launches on them become part of the captured graph.
        auto& aux = slot_aux_streams_[slot.index];
        for (int i = 0; i < num_aux_streams_; ++i) {
            cudaEvent_t fork_event;
            CUDA_CHECK(cudaEventCreate(&fork_event));
            CUDA_CHECK(cudaEventRecord(fork_event, stream));
            CUDA_CHECK(cudaStreamWaitEvent(aux[i], fork_event, 0));
            CUDA_CHECK(cudaEventDestroy(fork_event));
        }

        launch_fused_preprocess_bgra(
            static_cast<const std::uint8_t*>(slot.raw_buffer.ptr()),
            static_cast<float*>(slot.input_buffer.ptr()),
            src_w, src_h, src_pitch,
            engine_->model_width(), engine_->model_height(),
            compute_letterbox(src_w, src_h, engine_->model_width(), engine_->model_height()),
            stream);

        if (!slot.trt_context->enqueueV3(stream)) {
            // Abort graph capture to leave the stream in a valid state.
            cudaGraph_t aborted = nullptr;
            cudaStreamEndCapture(stream, &aborted);
            if (aborted) cudaGraphDestroy(aborted);
            throw std::runtime_error("TRT enqueueV3 failed during graph capture on slot "
                                     + std::to_string(slot.index));
        }

        // Join: merge aux stream work back into the main capture stream so
        // cudaStreamEndCapture sees a single-rooted graph.
        for (int i = 0; i < num_aux_streams_; ++i) {
            cudaEvent_t join_event;
            CUDA_CHECK(cudaEventCreate(&join_event));
            CUDA_CHECK(cudaEventRecord(join_event, aux[i]));
            CUDA_CHECK(cudaStreamWaitEvent(stream, join_event, 0));
            CUDA_CHECK(cudaEventDestroy(join_event));
        }

        auto output_bytes = std::size_t(engine_->max_detections()) * engine_->output_stride() * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(slot.output_pinned.ptr(), slot.output_buffer.ptr(),
                                    output_bytes, cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamEndCapture(stream, &gs.graph));
        CUDA_CHECK(cudaGraphInstantiate(&gs.exec, gs.graph, 0));
    }

    void Pipeline::harvest_and_deliver(PipelineSlot& slot, OutputSink& sink) {
        postprocessor_->process(
            {slot.output_pinned.as<float>(), static_cast<std::size_t>(engine_->max_detections()) * engine_->output_stride()},
            slot.frame_ctx);

        int w = slot.frame_ctx.src_width;
        int h = slot.frame_ctx.src_height;
        sink.on_frame(slot.frame_ctx, slot.raw_buffer.ptr(),
                      w, h, slot.frame_ctx.src_pitch, slot.stream.get(), slot.display_copy_done.get());
    }

    Pipeline::Metrics Pipeline::metrics() const noexcept {
        std::lock_guard lock(metrics_mtx_);
        return Metrics{
            .avg_latency_ms = completed_frames_ > 0 ? total_latency_us_ / (completed_frames_ * 1000.0) : 0.0,
            .last_pre_ms    = last_pre_ms_,
            .last_trt_ms    = last_trt_ms_,
            .last_d2h_ms    = last_d2h_ms_,
            .total_frames   = completed_frames_,
            .eager_frames   = eager_frames_
        };
    }

    int Pipeline::capture_width()  const noexcept { return capture_ ? capture_->width()  : 0; }
    int Pipeline::capture_height() const noexcept { return capture_ ? capture_->height() : 0; }

}
