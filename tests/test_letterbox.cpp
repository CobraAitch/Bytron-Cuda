/**
 * @file test_letterbox.cpp
 * @brief Unit tests for the letterbox transform (compute_letterbox).
 */

#include <gtest/gtest.h>
#include "preprocess/letterbox.hpp"

using bytron::compute_letterbox;

TEST(Letterbox, IdentitySquare) {
    auto p = compute_letterbox(640, 640, 640, 640);
    EXPECT_FLOAT_EQ(p.scale, 1.0f);
    EXPECT_EQ(p.new_w, 640);
    EXPECT_EQ(p.new_h, 640);
    EXPECT_EQ(p.pad_x, 0);
    EXPECT_EQ(p.pad_y, 0);
}

TEST(Letterbox, LandscapePadsTopBottom) {
    // 1920x1080 into 640x640: width is the binding dimension.
    auto p = compute_letterbox(1920, 1080, 640, 640);
    EXPECT_NEAR(p.scale, 640.0f / 1920.0f, 1e-6f);
    EXPECT_EQ(p.new_w, 640);
    EXPECT_EQ(p.new_h, 360);
    EXPECT_EQ(p.pad_x, 0);
    EXPECT_EQ(p.pad_y, (640 - 360) / 2);  // 140
}

TEST(Letterbox, PortraitPadsLeftRight) {
    auto p = compute_letterbox(1080, 1920, 640, 640);
    EXPECT_NEAR(p.scale, 640.0f / 1920.0f, 1e-6f);
    EXPECT_EQ(p.new_w, 360);
    EXPECT_EQ(p.new_h, 640);
    EXPECT_EQ(p.pad_x, (640 - 360) / 2);  // 140
    EXPECT_EQ(p.pad_y, 0);
}

TEST(Letterbox, NonSquareModel) {
    // Scale is the min of both ratios; the tighter dimension binds.
    auto p = compute_letterbox(1280, 720, 512, 288);
    EXPECT_NEAR(p.scale, 512.0f / 1280.0f, 1e-6f);  // 0.4 == 288/720, both equal here
    EXPECT_EQ(p.new_w, 512);
    EXPECT_EQ(p.new_h, 288);
    EXPECT_EQ(p.pad_x, 0);
    EXPECT_EQ(p.pad_y, 0);
}

TEST(Letterbox, DegenerateSourceReturnsSafeDefaults) {
    auto p = compute_letterbox(0, 0, 640, 640);
    EXPECT_FLOAT_EQ(p.scale, 1.0f);
    EXPECT_EQ(p.new_w, 640);
    EXPECT_EQ(p.new_h, 640);
    EXPECT_EQ(p.pad_x, 0);
    EXPECT_EQ(p.pad_y, 0);
}

TEST(Letterbox, RoundTripMapsModelBackToSource) {
    // A detection at the centre of the scaled content must map back near the
    // centre of the source frame: src = (model - pad) / scale.
    const int sw = 1920, sh = 1080;
    auto p = compute_letterbox(sw, sh, 640, 640);

    const float model_cx = p.pad_x + p.new_w * 0.5f;
    const float model_cy = p.pad_y + p.new_h * 0.5f;

    const float src_x = (model_cx - p.pad_x) / p.scale;
    const float src_y = (model_cy - p.pad_y) / p.scale;

    EXPECT_NEAR(src_x, sw * 0.5f, 1.0f);
    EXPECT_NEAR(src_y, sh * 0.5f, 1.0f);
}

TEST(Letterbox, IsConstexpr) {
    // Guard against accidental loss of compile-time evaluability.
    constexpr auto p = compute_letterbox(1920, 1080, 640, 640);
    static_assert(p.new_h == 360, "compute_letterbox must be constexpr-evaluable");
    SUCCEED();
}
