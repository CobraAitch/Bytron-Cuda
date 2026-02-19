#include "core/pipeline.hpp"
#include "capture/video_source_factory.hpp"
#include "preprocess/preprocess_kernels.cuh"
#include "util/cuda_check.hpp"
#include "util/timer.hpp"

#include <spdlog/spdlog.h>
#include <stdexcept>
#include <algorithm>

namespace bytron {

    Pipeline::Pipeline(const Config& cfg)
        : cfg_(cfg)
        , num_slots_(std::clamp(cfg.num_pipeline_slots, 2, MAX_SLOTS))
        , use_cuda_graphs_(cfg.use_cuda_graphs)
    {
        source_ = create_video_source(cfg_.video_uri);
        if (!source_ || !source_->is_open()) {
            throw std::runtime_error("Failed to open video source: " + cfg_.video_uri);
        }

        TrtEngine::BuildConfig engine_cfg{
            .onnx_path = cfg_.onnx_path,
            .engine_cache_path = cfg_.engine_cache_path(),
            .precision = cfg_.precision,
            .max_batch_size = 1,
            .opt_batch_size = 1,
            .max_workspace_bytes = 1ULL << 30,
            .max_aux_streams = 4,
            .calibration_data_dir = cfg_.calibration_images_dir,
            .calibration_batch_size = cfg_.calibration_batch_size,
        };
        engine_ = std::make_unique<TrtEngine>(engine_cfg);

        preprocessor_ = std::make_unique<GpuPreprocessor>(
            engine_->model_width(), engine_->model_height());

        postprocessor_ = std::make_unique<Postprocessor>(
            cfg_.confidence_threshold, engine_->max_detections());

        // Allocate per-slot resources (stream, buffers, TRT context)
        slots_.resize(num_slots_);
        for (int i = 0; i < num_slots_; ++i) {
            slots_[i].index = i;
            slots_[i].allocate(
                cfg_.max_frame_width, cfg_.max_frame_height,
                engine_->model_width(), engine_->model_height(),
                engine_->max_detections());
            slots_[i].trt_context = engine_->create_context();
        }

        graph_state_.resize(num_slots_);
        slot_warmed_up_.resize(num_slots_, false);

        if (cfg_.display) {
            display_staging_ = CudaPinnedBuffer(
                static_cast<std::size_t>(cfg_.max_frame_width) * cfg_.max_frame_height * 3);
        }

        timing_.resize(num_slots_);

        spdlog::info("Pipeline initialized: {} slots, model {}x{}, max {} detections, CUDA graphs: {}",
                     num_slots_, engine_->model_width(), engine_->model_height(),
                     engine_->max_detections(), use_cuda_graphs_ ? "ON" : "OFF");
    }

    Pipeline::~Pipeline() {
        for (auto& gs : graph_state_) {
            if (gs.exec) cudaGraphExecDestroy(gs.exec);
            if (gs.graph) cudaGraphDestroy(gs.graph);
        }
    }

    void Pipeline::run(OutputSink& sink) {
        running_ = true;
        frame_counter_ = 0;
        completed_frames_ = 0;
        total_latency_ms_ = 0.0;

        Timer fps_timer;

        // Prime the pipeline: fill all slots before harvesting
        int slots_in_flight = 0;
        for (int i = 0; i < num_slots_ && running_; ++i) {
            if (enqueue_frame(slots_[i])) {
                ++slots_in_flight;
            } else {
                break;
            }
        }

        int next_harvest = 0;

        // Steady state: harvest oldest, re-enqueue into that slot
        while (running_ && slots_in_flight > 0) {
            auto& slot = slots_[next_harvest % num_slots_];

            slot.ready_event.synchronize();

            float frame_ms = CudaEvent::elapsed_ms(
                timing_[slot.index].start, timing_[slot.index].end);
            total_latency_ms_ += frame_ms;
            ++completed_frames_;

            harvest_and_deliver(slot, sink);

            if (running_ && !enqueue_frame(slot)) {
                --slots_in_flight;
            }

            ++next_harvest;

            if (completed_frames_ % 100 == 0) {
                double elapsed = fps_timer.elapsed_ms();
                double fps = (completed_frames_ * 1000.0) / elapsed;
                double avg_lat = total_latency_ms_ / completed_frames_;
                spdlog::info("Frames: {}  FPS: {:.1f}  Avg latency: {:.2f} ms",
                             completed_frames_, fps, avg_lat);
            }
        }

        double elapsed = fps_timer.elapsed_ms();
        double fps = (completed_frames_ * 1000.0) / elapsed;
        spdlog::info("Pipeline finished: {} frames, {:.1f} FPS, {:.2f} ms avg latency",
                     completed_frames_, fps,
                     completed_frames_ > 0 ? total_latency_ms_ / completed_frames_ : 0.0);
    }

    void Pipeline::stop() noexcept {
        running_ = false;
    }

    bool Pipeline::enqueue_frame(PipelineSlot& slot) {
        auto stream = slot.stream.get();
        int idx = slot.index;

        timing_[idx].start.record(stream);

        int w = 0, h = 0;
        if (!source_->acquire_frame(slot.raw_buffer, stream, w, h)) {
            return false;
        }

        slot.frame_ctx.frame_id = frame_counter_++;
        slot.frame_ctx.slot_index = idx;

        if (use_cuda_graphs_ && slot_warmed_up_[idx]) {
            // Fast path: replay captured graph (skips driver overhead)
            auto& gs = graph_state_[idx];

            // Re-capture if source resolution changed
            if (!gs.captured || gs.captured_w != w || gs.captured_h != h) {
                capture_graph(slot, w, h);
                spdlog::debug("CUDA graph re-captured for slot {} ({}x{} -> {}x{})",
                              idx, gs.captured_w, gs.captured_h, w, h);
            }

            auto params = compute_letterbox(w, h, engine_->model_width(), engine_->model_height());
            slot.frame_ctx.src_width = w;
            slot.frame_ctx.src_height = h;
            slot.frame_ctx.scale = params.scale;
            slot.frame_ctx.pad_x = static_cast<float>(params.pad_x);
            slot.frame_ctx.pad_y = static_cast<float>(params.pad_y);

            enqueue_graph(slot);

        } else {
            // Slow path: individual kernel launches (used for warmup + graph capture)
            preprocessor_->preprocess(
                slot.raw_buffer, w, h,
                slot.input_buffer, slot.frame_ctx,
                stream);

            auto& ctx = slot.trt_context;
            ctx->setInputShape(engine_->input_name(),
                               nvinfer1::Dims4{1, 3, engine_->model_height(), engine_->model_width()});
            ctx->setTensorAddress(engine_->input_name(), slot.input_buffer.ptr());
            ctx->setTensorAddress(engine_->output_name(), slot.output_buffer.ptr());
            ctx->enqueueV3(stream);

            std::size_t output_bytes = static_cast<std::size_t>(engine_->max_detections()) * 6 * sizeof(float);
            slot.output_buffer.copy_to_host_async(slot.output_pinned.ptr(), output_bytes, stream);

            // After the first real run, capture a graph for all future frames
            if (use_cuda_graphs_ && !slot_warmed_up_[idx]) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                slot_warmed_up_[idx] = true;
                capture_graph(slot, w, h);
                spdlog::debug("CUDA graph captured for slot {} ({}x{})", idx, w, h);
            }
        }

        timing_[idx].end.record(stream);
        slot.ready_event.record(stream);

        return true;
    }

    void Pipeline::capture_graph(PipelineSlot& slot, int src_w, int src_h) {
        auto stream = slot.stream.get();
        int idx = slot.index;
        auto& gs = graph_state_[idx];

        if (gs.exec) { cudaGraphExecDestroy(gs.exec); gs.exec = nullptr; }
        if (gs.graph) { cudaGraphDestroy(gs.graph); gs.graph = nullptr; }

        // Bind TRT context before capture
        auto& ctx = slot.trt_context;
        ctx->setInputShape(engine_->input_name(),
                           nvinfer1::Dims4{1, 3, engine_->model_height(), engine_->model_width()});
        ctx->setTensorAddress(engine_->input_name(), slot.input_buffer.ptr());
        ctx->setTensorAddress(engine_->output_name(), slot.output_buffer.ptr());

        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

        launch_fused_preprocess(
            static_cast<const uint8_t*>(slot.raw_buffer.ptr()),
            static_cast<float*>(slot.input_buffer.ptr()),
            src_w, src_h, src_w * 3,
            engine_->model_width(), engine_->model_height(),
            stream
        );

        ctx->enqueueV3(stream);

        std::size_t output_bytes = static_cast<std::size_t>(engine_->max_detections()) * 6 * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(slot.output_pinned.ptr(), slot.output_buffer.ptr(),
                                    output_bytes, cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamEndCapture(stream, &gs.graph));
        CUDA_CHECK(cudaGraphInstantiate(&gs.exec, gs.graph, 0));

        gs.captured = true;
        gs.captured_w = src_w;
        gs.captured_h = src_h;
    }

    void Pipeline::enqueue_graph(PipelineSlot& slot) {
        auto& gs = graph_state_[slot.index];
        CUDA_CHECK(cudaGraphLaunch(gs.exec, slot.stream.get()));
    }

    void Pipeline::harvest_and_deliver(PipelineSlot& slot, OutputSink& sink) {
        postprocessor_->process(
            slot.output_pinned.as<float>(),
            slot.frame_ctx);

        if (cfg_.display) {
            // D2H copy of the raw BGR frame for display overlay
            std::size_t frame_bytes = static_cast<std::size_t>(slot.frame_ctx.src_width)
                                    * slot.frame_ctx.src_height * 3;
            CUDA_CHECK(cudaMemcpy(display_staging_.ptr(), slot.raw_buffer.ptr(),
                                  frame_bytes, cudaMemcpyDeviceToHost));

            sink.on_frame(slot.frame_ctx, display_staging_.ptr(),
                          slot.frame_ctx.src_width, slot.frame_ctx.src_height);
        } else {
            sink.on_frame(slot.frame_ctx, nullptr, 0, 0);
        }
    }

    Pipeline::Metrics Pipeline::metrics() const noexcept {
        return Metrics{
            .avg_latency_ms = completed_frames_ > 0 ? total_latency_ms_ / completed_frames_ : 0.0,
            .total_frames = completed_frames_
        };
    }
}
