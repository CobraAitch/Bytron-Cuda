/**
 * @file frame_context.hpp
 * @brief Per-frame metadata and detection results passed through the pipeline.
 */

#pragma once

#include <cstdint>
#include <vector>
#include "bytron/types.hpp"

namespace bytron {

    /**
     * @brief Carries frame identity, letterbox transform, and detection results.
     *
     * Populated during preprocessing (letterbox params) and postprocessing
     * (detections). The letterbox params map model-space coordinates back
     * to source resolution: `src_coord = (model_coord - pad) / scale`.
     *
     * @note Thread safety: Owned by one PipelineSlot; not shared across
     *       threads simultaneously.
     */
    struct FrameContext {
        std::uint64_t frame_id   = 0;  ///< Monotonically increasing frame counter
        int           slot_index = 0;  ///< Pipeline slot that processed this frame

        int   src_width  = 0;    ///< Original capture frame width in pixels
        int   src_height = 0;    ///< Original capture frame height in pixels
        int   src_pitch  = 0;    ///< Source row stride in bytes
        float scale      = 1.0f; ///< Letterbox scale factor (min of w_scale, h_scale)
        float pad_x      = 0.0f; ///< Horizontal letterbox padding in model pixels
        float pad_y      = 0.0f; ///< Vertical letterbox padding in model pixels

        std::vector<Detection> detections;  ///< Post-threshold detections in source-pixel coords
    };

}
