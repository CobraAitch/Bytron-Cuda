/**
 * @file test_postprocessor.cpp
 * @brief Unit tests for confidence filtering and inverse-letterbox transform.
 */

#include <gtest/gtest.h>
#include "postprocess/postprocessor.hpp"
#include "core/frame_context.hpp"

#include <vector>

using namespace bytron;

namespace {
    // Build one raw detection row in the default schema layout:
    // [x1, y1, x2, y2, confidence, class_id].
    void put(std::vector<float>& buf, float x1, float y1, float x2, float y2,
             float conf, int cls) {
        buf.insert(buf.end(), {x1, y1, x2, y2, conf, static_cast<float>(cls)});
    }

    FrameContext identity_ctx(int w, int h, float scale) {
        FrameContext ctx;
        ctx.src_width  = w;
        ctx.src_height = h;
        ctx.scale = scale;
        ctx.pad_x = 0.0f;
        ctx.pad_y = 0.0f;
        return ctx;
    }
}

TEST(Postprocessor, FiltersByConfidence) {
    Postprocessor pp(0.5f, /*max_detections=*/3, /*stride=*/6);

    std::vector<float> out;
    put(out, 100, 100, 200, 200, 0.9f, 1);  // keep
    put(out,  50,  50,  60,  60, 0.2f, 2);  // drop (< 0.5)
    put(out, 300, 300, 400, 400, 0.75f, 3); // keep

    auto ctx = identity_ctx(1280, 720, 0.5f);
    pp.process(out, ctx);

    ASSERT_EQ(ctx.detections.size(), 2u);
    EXPECT_EQ(ctx.detections[0].class_id, 1);
    EXPECT_EQ(ctx.detections[1].class_id, 3);
}

TEST(Postprocessor, AppliesInverseLetterbox) {
    Postprocessor pp(0.5f, 1, 6);

    std::vector<float> out;
    put(out, 100, 100, 200, 200, 0.9f, 7);

    // scale 0.5, no padding: src = model / 0.5 = model * 2.
    auto ctx = identity_ctx(1280, 720, 0.5f);
    pp.process(out, ctx);

    ASSERT_EQ(ctx.detections.size(), 1u);
    const auto& b = ctx.detections[0].box;
    EXPECT_FLOAT_EQ(b.x1, 200.0f);
    EXPECT_FLOAT_EQ(b.y1, 200.0f);
    EXPECT_FLOAT_EQ(b.x2, 400.0f);
    EXPECT_FLOAT_EQ(b.y2, 400.0f);
    EXPECT_FLOAT_EQ(ctx.detections[0].confidence, 0.9f);
    EXPECT_EQ(ctx.detections[0].class_id, 7);
}

TEST(Postprocessor, AccountsForPadding) {
    Postprocessor pp(0.5f, 1, 6);

    // Landscape letterbox: scale 1/3, pad_x 0, pad_y 140 (as in 1920x1080->640).
    // Model coords chosen to map to an interior source box (300,150)-(600,450):
    //   model = src * scale + pad.
    std::vector<float> out;
    put(out, 100, 190, 200, 290, 0.8f, 0);

    FrameContext ctx = identity_ctx(1920, 1080, 640.0f / 1920.0f);
    ctx.pad_y = 140.0f;
    pp.process(out, ctx);

    ASSERT_EQ(ctx.detections.size(), 1u);
    const auto& b = ctx.detections[0].box;
    EXPECT_NEAR(b.x1, 300.0f, 1.0f);
    EXPECT_NEAR(b.y1, 150.0f, 1.0f);
    EXPECT_NEAR(b.x2, 600.0f, 1.0f);
    EXPECT_NEAR(b.y2, 450.0f, 1.0f);
}

TEST(Postprocessor, ClampsToFrameBounds) {
    Postprocessor pp(0.5f, 1, 6);

    std::vector<float> out;
    // Box that maps outside the source frame after scaling.
    put(out, -50, -50, 5000, 5000, 0.9f, 0);

    auto ctx = identity_ctx(1280, 720, 1.0f);
    pp.process(out, ctx);

    ASSERT_EQ(ctx.detections.size(), 1u);
    const auto& b = ctx.detections[0].box;
    EXPECT_GE(b.x1, 0.0f);
    EXPECT_GE(b.y1, 0.0f);
    EXPECT_LE(b.x2, 1280.0f);
    EXPECT_LE(b.y2, 720.0f);
}

TEST(Postprocessor, ShortBufferProducesNoDetections) {
    Postprocessor pp(0.1f, 4, 6);  // expects 24 floats

    std::vector<float> out(6, 0.5f);  // only room for one row
    auto ctx = identity_ctx(640, 640, 1.0f);
    ctx.detections.resize(3);  // must be cleared regardless

    pp.process(out, ctx);
    EXPECT_TRUE(ctx.detections.empty());
}

TEST(Postprocessor, RejectsSchemaIndexOutsideStride) {
    DetectionSchema bad;
    bad.class_id_idx = 6;  // stride is 6 -> index 6 is out of range
    EXPECT_THROW(Postprocessor(0.5f, 10, 6, bad), std::runtime_error);
}

TEST(Postprocessor, HonoursWiderStride) {
    // Stride 7 with default schema (max index 5) must be accepted and skip the
    // trailing padding value per row.
    Postprocessor pp(0.5f, 2, 7);

    std::vector<float> out;
    // row layout: x1,y1,x2,y2,conf,class,<pad>
    out.insert(out.end(), {10, 10, 20, 20, 0.9f, 4, 0});
    out.insert(out.end(), {30, 30, 40, 40, 0.1f, 5, 0});  // dropped

    auto ctx = identity_ctx(640, 640, 1.0f);
    pp.process(out, ctx);

    ASSERT_EQ(ctx.detections.size(), 1u);
    EXPECT_EQ(ctx.detections[0].class_id, 4);
}
