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
    static_assert(sizeof(Detection) == 24, "Detection layout changed: update GPU code");

    /** @brief TensorRT inference precision mode. */
    enum class Precision {
        FP16,  ///< Half-precision floating point (default)
        INT8   ///< 8-bit integer quantization (requires calibration data)
    };

    /**
     * @brief Screen capture backend selection.
     *
     * GraphicsCapture is the modern Windows.Graphics.Capture (WinRT) path:
     * compositor-driven, GPU-resident, borderless/cursorless, and the
     * preferred default. DesktopDuplication is the legacy DXGI Output
     * Duplication path, used automatically when WGC is unavailable.
     */
    enum class CaptureBackend {
        GraphicsCapture,    ///< Windows.Graphics.Capture (Windows 10 1903+/11)
        DesktopDuplication  ///< DXGI Output Duplication (fallback)
    };

    /**
     * @brief Describes the per-detection field layout in the model output tensor.
     *
     * Maps semantic fields (bounding box corners, confidence, class ID) to
     * indices within each detection's stride in the raw output tensor. This
     * decouples the postprocessor from any specific model format: change
     * the schema to support different YOLO variants or custom models without
     * modifying postprocessor code.
     *
     * Default layout matches YOLOv26: [x1, y1, x2, y2, confidence, class_id].
     */
    struct DetectionSchema {
        int x1_idx         = 0;  ///< Index of top-left x coordinate
        int y1_idx         = 1;  ///< Index of top-left y coordinate
        int x2_idx         = 2;  ///< Index of bottom-right x coordinate
        int y2_idx         = 3;  ///< Index of bottom-right y coordinate
        int confidence_idx = 4;  ///< Index of detection confidence score
        int class_id_idx   = 5;  ///< Index of class ID
    };
}
