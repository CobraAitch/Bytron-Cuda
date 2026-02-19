#pragma once

#include <cstdint>
#include <vector>
#include "bytron/types.hpp"

namespace bytron {

    /// <summary>
    /// Per-frame metadata that travels through the pipeline alongside its buffers.
    /// Holds source dimensions, letterbox transform params, and final detection results.
    /// </summary>
    struct FrameContext {
        std::uint64_t frame_id = 0;
        int slot_index = 0;

        int src_width = 0;
        int src_height = 0;

        // Letterbox transform (needed to rescale detections back to source coords)
        float scale = 1.0f;
        float pad_x = 0.0f;
        float pad_y = 0.0f;

        std::vector<Detection> detections;
    };
}
