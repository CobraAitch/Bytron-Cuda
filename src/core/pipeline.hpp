/**
 * @file pipeline.hpp
 * @brief Pipelined CUDA + TensorRT inference orchestrator with DXGI desktop capture.
 */

#pragma once

#include "core/pipeline_slot.hpp"
#include "core/capture_mailbox.hpp"
#include "core/cuda_event.hpp"
#include "core/cuda_stream.hpp"
#include "capture/capture_source.hpp"
#include "preprocess/gpu_preprocessor.hpp"
#include "engine/trt_engine.hpp"
#include "postprocess/postprocessor.hpp"
#include "output/output_sink.hpp"
#include "util/config.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stop_token>
#include <thread>
#include <unordered_map>

namespace bytron {

    /**
     * @brief Multi-slot pipelined inference engine with capture, preprocess, TRT
     *        inference, postprocess, and output delivery.
     *
     * Architecture uses two jthreads:
     *  - **Capture thread** acquires DXGI frames into a triple-buffered
     *    CaptureMailbox (BGRA, GPU-resident, zero CPU copies).
     *  - **Inference thread** cycles through N PipelineSlots, each with its own
     *    high-priority CUDA stream and TRT execution context: copy captured frame
     *    → fused preprocess → TRT enqueue → D2H output copy → postprocess →
     *    deliver to OutputSink.
     *
     * Optionally captures a CUDA graph per slot after a warm-up pass, replaying
     * the graph on subsequent frames for reduced launch overhead. Graphs are
     * re-captured when capture resolution changes.
     *
     * @code
     * Config cfg = Config::from_file("config.json");
     * Pipeline pipeline(cfg);
     *
     * CallbackSink sink([](const FrameContext& ctx) {
     *     for (auto& det : ctx.detections)
     *         spdlog::info("class {} conf {:.2f}", det.class_id, det.confidence);
     * });
     *
     * pipeline.run(sink);  // blocks until sink.should_continue() returns false
     * @endcode
     *
     * @note Thread safety: run() blocks the calling thread. stop() is safe to
     *       call from any thread (e.g. signal handler).
     */
    class Pipeline {
    public:
        static constexpr int MAX_SLOTS = 4;  ///< Maximum pipeline depth (limited by GPU memory).

        /**
         * @brief Construct the pipeline: build/load TRT engine, allocate slots and mailbox.
         * @param [in] cfg Validated configuration.
         * @throws std::runtime_error on capture, engine build, or allocation failure.
         */
        explicit Pipeline(const Config& cfg);
        ~Pipeline();

        Pipeline(const Pipeline&) = delete;
        Pipeline& operator=(const Pipeline&) = delete;

        /**
         * @brief Start capture and inference threads, blocking until the sink
         *        signals to stop or a thread exits.
         *
         * Joins inference_thread_ first, then requests capture_thread_ stop.
         *
         * @param [in] sink Output sink receiving processed frames. Must outlive
         *                  this call. sink.should_continue() is polled each iteration.
         */
        void run(OutputSink& sink);

        /**
         * @brief Request both threads to stop. Safe to call from signal handlers.
         */
        void stop() noexcept;

        /**
         * @brief Accumulated pipeline performance counters with per-stage breakdown.
         *
         * Per-stage averages are only available for eager-mode frames (not CUDA
         * graph replays, where stages are fused into a single launch).
         */
        struct Metrics {
            double        avg_latency_ms    = 0.0;  ///< Mean total GPU latency per frame (ms).
            float         last_pre_ms       = 0.0f; ///< Latest preprocess time from profiling frame (ms).
            float         last_trt_ms       = 0.0f; ///< Latest TRT inference time from profiling frame (ms).
            float         last_d2h_ms       = 0.0f; ///< Latest D2H copy time from profiling frame (ms).
            std::uint64_t total_frames      = 0;    ///< Number of frames fully processed.
            std::uint64_t eager_frames      = 0;    ///< Total profiling frames executed.
        };

        [[nodiscard]] Metrics metrics()        const noexcept;
        [[nodiscard]] int     capture_width()  const noexcept;  ///< Current capture source width.
        [[nodiscard]] int     capture_height() const noexcept;  ///< Current capture source height.

    private:
        /**
         * @brief Capture thread: acquires DXGI frames into the triple-buffered mailbox.
         *
         * Waits on read_done before overwriting each entry to avoid tearing.
         * On access-lost, retries reacquire() up to 20 times at 100ms intervals.
         */
        void capture_loop(std::stop_token stoken);

        /**
         * @brief Inference thread: cycles slots through preprocess → TRT → postprocess → deliver.
         *
         * Waits for the mailbox CV, finalizes the oldest in-flight slot, then
         * consumes the *freshest* capture entry (which may be newer than the one
         * that triggered the wakeup). Slot reuse is gated on ready_event completion
         * and display_copy_done to prevent overwriting a frame the display sink
         * hasn't finished copying.
         */
        void inference_loop(std::stop_token stoken, OutputSink& sink);

        /**
         * @brief Enqueue preprocess + TRT + D2H for a frame already copied into the slot.
         *
         * Dispatches to CUDA graph replay when warmed up (with periodic eager
         * profiling frames), otherwise runs the eager path and captures a graph
         * after the first warm-up pass.
         */
        void launch_inference(PipelineSlot& slot, int w, int h, int pitch);

        /**
         * @brief Wait for a slot's in-flight work, record metrics, and deliver results.
         *
         * Blocks on ready_event, accumulates GPU latency into the metrics counters
         * (per-stage breakdown for eager frames), then postprocesses and delivers.
         */
        void finalize_slot(PipelineSlot& slot, OutputSink& sink);

        /**
         * @brief Postprocess TRT output and deliver to the output sink.
         *
         * Reads pinned output buffer, runs confidence filtering and inverse-letterbox
         * transform, then calls sink.on_frame() with the raw BGRA buffer and metadata.
         */
        void harvest_and_deliver(PipelineSlot& slot, OutputSink& sink);

        Config cfg_;
        int    num_slots_;  ///< Actual slot count [2, MAX_SLOTS], clamped from config.

        std::unique_ptr<ICaptureSource>  capture_;        ///< Screen capture source (WGC or DXGI).
        std::unique_ptr<GpuPreprocessor> preprocessor_;   ///< Fused resize+letterbox+normalize kernel.
        std::unique_ptr<TrtEngine>       engine_;         ///< TensorRT engine (owns serialized plan).
        std::unique_ptr<Postprocessor>   postprocessor_;  ///< Confidence filter + inverse letterbox.

        std::vector<PipelineSlot> slots_;  ///< N independent inference slots with buffers and streams.

        CudaStream     capture_stream_;   ///< Dedicated stream for DXGI acquire + D2D copy.
        CaptureMailbox capture_mailbox_;  ///< Triple-buffered capture → inference handoff.

        std::jthread capture_thread_;    ///< Runs capture_loop().
        std::jthread inference_thread_;  ///< Runs inference_loop().
        std::atomic<bool> capture_dead_{false};  ///< Set by capture thread on unrecoverable exit.

        /** @brief Per-resolution CUDA graph state for replay-based inference. */
        struct GraphState {
            cudaGraph_t     graph    = nullptr;  ///< Captured graph (preprocess + TRT + D2H copy).
            cudaGraphExec_t exec     = nullptr;  ///< Instantiated executable graph.

            ~GraphState() { destroy(); }
            GraphState() = default;
            GraphState(GraphState&& o) noexcept
                : graph(std::exchange(o.graph, nullptr))
                , exec(std::exchange(o.exec, nullptr)) {}
            GraphState& operator=(GraphState&& o) noexcept {
                if (this != &o) { destroy(); graph = std::exchange(o.graph, nullptr); exec = std::exchange(o.exec, nullptr); }
                return *this;
            }
            GraphState(const GraphState&) = delete;
            GraphState& operator=(const GraphState&) = delete;
        private:
            void destroy() noexcept {
                if (exec)  { cudaGraphExecDestroy(exec);  exec  = nullptr; }
                if (graph) { cudaGraphDestroy(graph);      graph = nullptr; }
            }
        };

        static constexpr int MAX_CACHED_GRAPHS = 4;  ///< Max cached resolutions per slot.

        /** @brief Pack (w, h) into a 64-bit key for graph cache lookup. */
        static std::uint64_t res_key(int w, int h) noexcept {
            return (static_cast<std::uint64_t>(w) << 32) | static_cast<std::uint32_t>(h);
        }

        /**
         * @brief Capture the preprocess → TRT → D2H sequence as a CUDA graph.
         *
         * Uses cudaStreamCaptureModeGlobal so TRT internal streams are included.
         * Called once after warm-up, and again on resolution cache miss.
         */
        void capture_graph(PipelineSlot& slot, int src_w, int src_h, int src_pitch, GraphState& gs);

        /// Per-slot map from resolution key → cached graph, with LRU eviction order.
        struct SlotGraphCache {
            std::unordered_map<std::uint64_t, GraphState> map;
            std::vector<std::uint64_t> lru;  ///< Front = oldest (evict first), back = most recent.

            void touch(std::uint64_t key) {
                std::erase(lru, key);
                lru.push_back(key);
            }

            void evict_oldest() {
                if (lru.empty()) return;
                map.erase(lru.front());
                lru.erase(lru.begin());
            }
        };
        std::vector<std::unique_ptr<SlotGraphCache>> graph_cache_;
        bool use_cuda_graphs_ = false;   ///< From config; enables graph capture after warm-up.

        /// Per-slot auxiliary CUDA streams for TRT multi-stream execution.
        /// Created non-blocking at high priority: they carry inference kernels on
        /// the critical path, so they must match the slot stream's priority and
        /// never serialize with the legacy default stream.
        /// Registered via setAuxStreams() so we control TRT's parallel streams.
        /// Used for: (1) graph capture fork/join, (2) eager-path profiling
        /// sync (joining aux work before recording trt_done event).
        int num_aux_streams_ = 0;
        std::vector<std::vector<cudaStream_t>> slot_aux_streams_;
        std::vector<std::vector<CudaEvent>> slot_aux_sync_events_;

        void create_aux_streams();
        void destroy_aux_streams() noexcept;

        std::vector<bool> slot_warmed_up_;  ///< Per-slot: true after first eager execution pass.

        std::uint64_t frame_counter_{0};  ///< Monotonic frame ID generator (inference thread only).

        /// Accumulated metrics protected by metrics_mtx_ for consistent snapshots.
        mutable std::mutex metrics_mtx_;
        std::uint64_t total_latency_us_{0};
        std::uint64_t completed_frames_{0};
        std::uint64_t eager_frames_{0};

        /// Latest per-stage timings from the most recent profiling frame.
        /// Updated every PROFILE_INTERVAL frames; warm-up frames are excluded.
        float last_pre_ms_{0.0f};
        float last_trt_ms_{0.0f};
        float last_d2h_ms_{0.0f};

        /**
         * @brief Per-slot CUDA events for GPU-side latency measurement.
         *
         * In eager mode, all four events are recorded per frame for per-stage
         * breakdown. In graph replay mode, only start and end are valid.
         */
        struct SlotTiming {
            CudaEvent start{true};           ///< Recorded before preprocess. Timing-enabled.
            CudaEvent preprocess_done{true}; ///< Recorded after preprocess, before TRT. Eager only.
            CudaEvent trt_done{true};        ///< Recorded after TRT enqueue, before D2H. Eager only.
            CudaEvent end{true};             ///< Recorded after D2H output copy. Timing-enabled.
        };
        std::vector<SlotTiming> timing_;

        std::vector<bool> slot_eager_exec_;  ///< Per-slot: true if last execution was eager (per-stage events valid).

        /// Exception propagation from worker threads to run().
        std::mutex              exception_mtx_;
        std::exception_ptr      thread_exception_;

        void store_exception(std::exception_ptr ep) noexcept;
    };

}
