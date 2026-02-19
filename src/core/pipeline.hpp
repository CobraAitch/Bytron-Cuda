#pragma once

#include "core/pipeline_slot.hpp"
#include "core/cuda_event.hpp"
#include "capture/video_source.hpp"
#include "preprocess/gpu_preprocessor.hpp"
#include "engine/trt_engine.hpp"
#include "postprocess/postprocessor.hpp"
#include "output/output_sink.hpp"
#include "util/config.hpp"

#include <atomic>
#include <memory>
#include <cstdint>

namespace bytron {

    /// <summary>
    /// Triple-buffered inference pipeline. Overlaps capture, preprocess+infer,
    /// and postprocess across multiple CUDA streams for maximum GPU utilization.
    /// </summary>
    class Pipeline {
        public:
            static constexpr int MAX_SLOTS = 4;

            explicit Pipeline(const Config& cfg);
            ~Pipeline();

            Pipeline(const Pipeline&) = delete;
            Pipeline& operator=(const Pipeline&) = delete;

            /// <summary>
            /// Main loop: feeds frames through the pipeline until the source is
            /// exhausted or stop() is called. Blocks the calling thread.
            /// </summary>
            void run(OutputSink& sink);

            void stop() noexcept;

            struct Metrics {
                double avg_latency_ms = 0.0;
                std::uint64_t total_frames = 0;
            };

            [[nodiscard]] Metrics metrics() const noexcept;

        private:
            // Submit one frame into the given slot's stream (preprocess + infer + D2H copy)
            bool enqueue_frame(PipelineSlot& slot);

            // Replay a previously captured CUDA graph
            void enqueue_graph(PipelineSlot& slot);

            // Record preprocess+infer+copy as a CUDA graph for this slot/resolution
            void capture_graph(PipelineSlot& slot, int src_w, int src_h);

            // Wait for the slot, run postprocess, push results to the sink
            void harvest_and_deliver(PipelineSlot& slot, OutputSink& sink);

            Config cfg_;
            int    num_slots_;

            std::unique_ptr<VideoSource>     source_;
            std::unique_ptr<GpuPreprocessor> preprocessor_;
            std::unique_ptr<TrtEngine>       engine_;
            std::unique_ptr<Postprocessor>   postprocessor_;

            std::vector<PipelineSlot> slots_;

            // Per-slot CUDA graph state (captured on first frame, re-captured on resolution change)
            struct GraphState {
                cudaGraph_t     graph = nullptr;
                cudaGraphExec_t exec = nullptr;
                bool            captured = false;
                int             captured_w = 0;
                int             captured_h = 0;
            };
            std::vector<GraphState> graph_state_;
            bool use_cuda_graphs_ = false;

            std::vector<bool> slot_warmed_up_;

            std::atomic<bool> running_{false};
            std::uint64_t frame_counter_ = 0;

            // Pinned staging for display D2H copy (only allocated when display is on)
            CudaPinnedBuffer display_staging_;

            double total_latency_ms_ = 0.0;
            std::uint64_t completed_frames_ = 0;

            struct SlotTimingEvents {
                CudaEvent start{true};
                CudaEvent end{true};
            };
            std::vector<SlotTimingEvents> timing_;
    };
}
