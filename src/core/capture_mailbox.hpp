/**
 * @file capture_mailbox.hpp
 * @brief Triple-buffered mailbox for passing captured frames between threads.
 */

#pragma once

#include "core/cuda_buffer.hpp"
#include "core/cuda_event.hpp"

#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace bytron {

    /**
     * @brief Triple-buffered mailbox for the capture-to-inference handoff.
     *
     * Protocol: capture thread writes to entries[write_idx], records
     * capture_done, updates latest_idx + sequence under mtx, and notifies cv.
     * Inference thread waits on cv, reads latest_idx, waits on capture_done,
     * copies frame data, then records read_done. Capture thread waits on
     * read_done before overwriting that entry.
     *
     * Uses std::condition_variable_any (not condition_variable) because
     * stop_token-aware wait() requires it.
     *
     * @note Thread safety: mtx + cv protect latest_idx and sequence.
     *       GPU synchronization uses CUDA events.
     */
    struct CaptureMailbox {
        static constexpr int NUM_ENTRIES = 3;  ///< Triple-buffered

        struct Entry {
            CudaDeviceBuffer frame_buf;    ///< BGRA device buffer (max_w * max_h * 4 bytes)
            CudaEvent        capture_done; ///< Signaled after capture writes frame data
            CudaEvent        read_done;    ///< Signaled after inference finishes reading
            int width  = 0;  ///< Actual frame width for this capture
            int height = 0;  ///< Actual frame height for this capture
        };

        Entry entries[NUM_ENTRIES];

        std::mutex                  mtx;  ///< Protects latest_idx and sequence
        std::condition_variable_any cv;   ///< Notified when a new frame is available

        int           latest_idx        = -1;  ///< Most recently written entry index
        std::uint64_t sequence          = 0;   ///< Incremented on each capture
        std::uint64_t consumed_sequence = 0;   ///< Last sequence consumed by inference

        /**
         * @brief Pre-allocate device buffers and prime read_done events.
         *
         * Records initial read_done events so the capture thread's first
         * iteration does not stall waiting for a read that never happened.
         */
        void allocate(int max_width, int max_height) {
            auto frame_bytes = std::size_t(max_width) * max_height * 4;
            for (auto& entry : entries) {
                entry.frame_buf = CudaDeviceBuffer(frame_bytes);
                CUDA_CHECK(cudaEventRecord(entry.read_done.get(), nullptr));
            }
        }
    };

}
