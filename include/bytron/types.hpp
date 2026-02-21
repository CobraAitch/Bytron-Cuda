/**
 * @file types.hpp
 * @brief Core detection types and precision enum for the Bytron public API.
 */

#pragma once

#include <cstdint>
#include <type_traits>
#include <vector>

namespace bytron {

    /**
     * @brief Axis-aligned bounding box in pixel coordinates.
     *
     * Coordinates are in the original source frame's pixel space after
     * inverse letterbox rescaling (not model input space).
     */
    struct BBox {
        float x1, y1;  ///< Top-left corner (pixels, origin = top-left of frame)
        float x2, y2;  ///< Bottom-right corner (pixels)
    };

    /**
     * @brief Single object detection result.
     *
     * Layout is trivially copyable (24 bytes) to allow direct GPU memcpy
     * between device and host buffers.
     */
    struct Detection {
        BBox  box;         ///< Bounding box in source-frame pixel coordinates
        float confidence;  ///< Detection confidence in [0.0, 1.0]
        int   class_id;    ///< Class index from the model's label set
    };

    static_assert(std::is_trivially_copyable_v<BBox>);
    static_assert(std::is_trivially_copyable_v<Detection>);
    static_assert(sizeof(Detection) == 24, "Detection layout changed — update GPU code");

    /** @brief TensorRT inference precision mode. */
    enum class Precision {
        FP16,  ///< Half-precision floating point (default)
        INT8   ///< 8-bit integer quantization (requires calibration data)
    };
}
