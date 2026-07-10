/**
 * @file test_config.cpp
 * @brief Unit tests for Config CLI parsing, JSON loading, and validation.
 */

#include <gtest/gtest.h>
#include "util/config.hpp"

#include <fstream>
#include <filesystem>
#include <string>
#include <vector>

using namespace bytron;

namespace {
    // Adapt string args to the (int, char**) signature apply_cli expects.
    void apply(Config& cfg, std::vector<std::string> args) {
        std::vector<char*> argv;
        argv.reserve(args.size());
        for (auto& a : args) argv.push_back(a.data());
        cfg.apply_cli(static_cast<int>(argv.size()), argv.data());
    }
}

TEST(Config, Defaults) {
    Config cfg;
    EXPECT_EQ(cfg.gpu_device, 0);
    EXPECT_EQ(cfg.num_pipeline_slots, 3);
    EXPECT_EQ(cfg.precision, Precision::FP16);
    EXPECT_EQ(cfg.capture_backend, CaptureBackend::GraphicsCapture);
    EXPECT_TRUE(cfg.display);
    EXPECT_TRUE(cfg.overlay);
    EXPECT_FLOAT_EQ(cfg.confidence_threshold, 0.25f);
}

TEST(Config, CliOverrides) {
    Config cfg;
    apply(cfg, {"bytron", "--gpu", "1", "--conf", "0.4", "--slots", "4",
                "--precision", "int8", "--onnx", "m.onnx",
                "--no-overlay", "--no-cuda-graphs"});

    EXPECT_EQ(cfg.gpu_device, 1);
    EXPECT_FLOAT_EQ(cfg.confidence_threshold, 0.4f);
    EXPECT_EQ(cfg.num_pipeline_slots, 4);
    EXPECT_EQ(cfg.precision, Precision::INT8);
    EXPECT_EQ(cfg.onnx_path, std::filesystem::path("m.onnx"));
    EXPECT_FALSE(cfg.overlay);
    EXPECT_FALSE(cfg.use_cuda_graphs);
}

TEST(Config, CaptureBackendFlag) {
    {
        Config cfg;
        apply(cfg, {"bytron", "--capture", "dxgi"});
        EXPECT_EQ(cfg.capture_backend, CaptureBackend::DesktopDuplication);
    }
    {
        Config cfg;
        apply(cfg, {"bytron", "--capture", "wgc"});
        EXPECT_EQ(cfg.capture_backend, CaptureBackend::GraphicsCapture);
    }
    {
        Config cfg;  // unrecognized value falls back to the WGC default
        apply(cfg, {"bytron", "--capture", "banana"});
        EXPECT_EQ(cfg.capture_backend, CaptureBackend::GraphicsCapture);
    }
}

TEST(Config, MissingCliValueThrows) {
    Config cfg;
    EXPECT_THROW(apply(cfg, {"bytron", "--gpu"}), std::runtime_error);
}

TEST(Config, EngineCachePath) {
    Config cfg;
    cfg.onnx_path        = "models/yolo26m.onnx";
    cfg.engine_cache_dir = "engines";
    cfg.precision        = Precision::FP16;
    EXPECT_EQ(cfg.engine_cache_path(), std::filesystem::path("engines") / "yolo26m_fp16.engine");

    cfg.precision = Precision::INT8;
    EXPECT_EQ(cfg.engine_cache_path().filename(), std::filesystem::path("yolo26m_int8.engine"));
}

TEST(Config, ValidateRejectsBadRanges) {
    {
        Config cfg; cfg.num_pipeline_slots = 0;
        EXPECT_THROW(cfg.validate(), std::runtime_error);
    }
    {
        Config cfg; cfg.confidence_threshold = 2.0f;
        EXPECT_THROW(cfg.validate(), std::runtime_error);
    }
    {
        Config cfg; cfg.builder_opt_level = 6;
        EXPECT_THROW(cfg.validate(), std::runtime_error);
    }
    {
        Config cfg; cfg.max_detections = 0;
        EXPECT_THROW(cfg.validate(), std::runtime_error);
    }
    {
        Config cfg;  // defaults are valid
        EXPECT_NO_THROW(cfg.validate());
    }
}

TEST(Config, FromJsonFile) {
    auto path = std::filesystem::temp_directory_path() / "bytron_test_config.json";
    {
        std::ofstream f(path);
        f << R"({
            "gpu_device": 2,
            "precision": "int8",
            "num_pipeline_slots": 5,
            "confidence_threshold": 0.6,
            "capture_backend": "dxgi",
            "display": false
        })";
    }

    Config cfg = Config::from_file(path);
    EXPECT_EQ(cfg.gpu_device, 2);
    EXPECT_EQ(cfg.precision, Precision::INT8);
    EXPECT_EQ(cfg.num_pipeline_slots, 5);
    EXPECT_FLOAT_EQ(cfg.confidence_threshold, 0.6f);
    EXPECT_EQ(cfg.capture_backend, CaptureBackend::DesktopDuplication);
    EXPECT_FALSE(cfg.display);

    std::filesystem::remove(path);
}
