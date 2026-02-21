/**
 * @file pipeline.hpp
 * @brief Pipelined CUDA + TensorRT inference orchestrator with DXGI desktop capture.
 */

#pragma once

#include "core/pipeline_slot.hpp"
#include "core/capture_mailbox.hpp"
#include "core/cuda_event.hpp"
#include "core/cuda_stream.hpp"
#include "capture/desktop_capture.hpp"
#include "preprocess/gpu_preprocessor.hpp"
#include "engine/trt_engine.hpp"
#include "postprocess/postprocessor.hpp"
#include "output/output_sink.hpp"
#include "util/config.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
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
     *    CUDA stream and TRT execution context: copy captured frame → fused
     *    preprocess → TRT enqueue → D2H output copy → postprocess → deliver
     *    to OutputSink.
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

        /** @brief Accumulated pipeline performance counters. */
        struct Metrics {
            double        avg_latency_ms = 0.0;  ///< Mean GPU latency per frame (ms).
            std::uint64_t total_frames   = 0;     ///< Number of frames fully processed.
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
         * Waits for the mailbox CV, then processes the latest capture entry. Slot
         * reuse is gated on ready_event completion and display_copy_done to prevent
         * overwriting a frame the display sink hasn't finished copying.
         */
        void inference_loop(std::stop_token stoken, OutputSink& sink);

        /**
         * @brief Postprocess TRT output and deliver to the output sink.
         *
         * Reads pinned output buffer, runs confidence filtering and inverse-letterbox
         * transform, then calls sink.on_frame() with the raw BGRA buffer and metadata.
         */
        void harvest_and_deliver(PipelineSlot& slot, OutputSink& sink);

        Config cfg_;
        int    num_slots_;  ///< Actual slot count [2, MAX_SLOTS], clamped from config.

        std::unique_ptr<DesktopCapture>  capture_;        ///< DXGI desktop duplication source.
        std::unique_ptr<GpuPreprocessor> preprocessor_;   ///< Fused resize+letterbox+normalize kernel.
        std::unique_ptr<TrtEngine>       engine_;         ///< TensorRT engine (owns serialized plan).
        std::unique_ptr<Postprocessor>   postprocessor_;  ///< Confidence filter + inverse letterbox.

        std::vector<PipelineSlot> slots_;  ///< N independent inference slots with buffers and streams.

        CudaStream     capture_stream_;   ///< Dedicated stream for DXGI acquire + D2D copy.
        CaptureMailbox capture_mailbox_;  ///< Triple-buffered capture → inference handoff.

        std::jthread capture_thread_;    ///< Runs capture_loop().
        std::jthread inference_thread_;  ///< Runs inference_loop().

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
        void capture_graph(PipelineSlot& slot, int src_w, int src_h, GraphState& gs);

        /// Per-slot map from resolution key → cached graph.
        std::vector<std::unique_ptr<std::unordered_map<std::uint64_t, GraphState>>> graph_cache_;
        bool use_cuda_graphs_ = false;   ///< From config; enables graph capture after warm-up.

        std::vector<bool> slot_warmed_up_;  ///< Per-slot: true after first eager execution pass.

        std::atomic<std::uint64_t> frame_counter_{0};    ///< Monotonic frame ID generator.
        std::atomic<double>        total_latency_ms_{0.0};  ///< Accumulated GPU latency for averaging.
        std::atomic<std::uint64_t> completed_frames_{0};    ///< Frames fully processed (through postprocess).

        /** @brief Per-slot CUDA event pair for GPU-side latency measurement. */
        struct SlotTiming {
            CudaEvent start{true};  ///< Recorded before preprocess. Timing-enabled.
            CudaEvent end{true};    ///< Recorded after output copy. Timing-enabled.
        };
        std::vector<SlotTiming> timing_;
    };

}
