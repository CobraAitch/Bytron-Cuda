#pragma once

#include <cstdint>
#include <vector>

namespace bytron {

    /// <summary>
    /// Represents a bounding box defined by two corner points.
    /// </summary>
    struct BBox {
        /// <summary>
        /// Float variables, used to store coordinates for two points (x1, y1) top-left and (x2, y2) bottom right.
        /// </summary>
        float x1, y1, x2, y2;
    };

    /// <summary>
    /// Represents a detection result from an object detection model.
    /// </summary>
    struct Detection {
        /// <summary>
        /// A bounding box variable.
        /// </summary>
        BBox box;

        /// <summary>
        /// The confidence score for the result predicted by the model.
        /// </summary>
        float confidence;

        /// <summary>
        /// The class id of the current detection by the model.
        /// </summary>
        int class_id;
    };


    /// <summary>
    /// Specifies the precision mode for YOLO model inference.
    /// </summary>
    enum class Precision {
        /// <summary>
        /// 16-bit floating point precision for YOLO model inference.
        /// </summary>
        FP16,

        /// <summary>
        /// 8-bit integer quantized precision for YOLO model inference.
        /// </summary>
        INT8
    };
}
