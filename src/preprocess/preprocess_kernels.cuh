/**
 * @file preprocess_kernels.cuh
 * @brief Fused CUDA preprocessing kernel declarations.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "preprocess/letterbox.hpp"

namespace bytron {

    /**
     * @brief Launch the fused BGRA preprocessing kernel.
     *
     * Single-pass GPU kernel: bilinear resize + letterbox padding (114/255 gray)
     * + BGRA-to-RGB channel reorder + [0,255]-to-[0,1] normalize + HWC-to-CHW transpose.
     *
     * @param [in]  src       Device pointer to BGRA source frame.
     * @param [out] dst       Device pointer to CHW float output (3 * dst_h * dst_w floats).
     * @param [in]  src_pitch Source row stride in bytes (typically src_w * 4).
     * @param [in]  params    Letterbox transform from compute_letterbox(src, dst dims).
     * @param [in]  stream    CUDA stream for async execution.
     *
     * @note Kernel config: 32x8 thread blocks, grid covers (dst_w, dst_h).
     *       Source pixels loaded as uint32_t via __ldg() for coalesced access.
     */
    void launch_fused_preprocess_bgra(
        const std::uint8_t* src, float* dst,
        int src_w, int src_h, int src_pitch,
        int dst_w, int dst_h,
        const LetterboxParams& params,
        cudaStream_t stream);

}
