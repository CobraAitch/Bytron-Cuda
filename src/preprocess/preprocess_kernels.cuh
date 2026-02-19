w#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace bytron {

    /// <summary>
    /// Fused preprocessing kernel: resize + letterbox + BGR to RGB + normalize + HWC to CHW.
    /// One thread per output pixel; all 3 channels computed per thread.
    /// </summary>
    void launch_fused_preprocess(
        const std::uint8_t* src,
        float*              dst,
        int src_w, int src_h,
        int src_pitch,
        int dst_w, int dst_h,
        cudaStream_t stream
    );

    /// <summary>
    /// Letterbox parameters computed on the CPU side.
    /// </summary>
    struct LetterboxParams {
        float scale;
        int   new_w;
        int   new_h;
        int   pad_x; // left padding
        int   pad_y; // top padding
    };

    LetterboxParams compute_letterbox(int src_w, int src_h, int dst_w, int dst_h);
}
