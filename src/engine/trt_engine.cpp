/**
 * @file trt_engine.cpp
 * @brief TensorRT engine build, serialization, cache management, and timing cache.
 */

#include "engine/trt_engine.hpp"
#include "engine/trt_logger.hpp"
#include "engine/calibrator.hpp"

#pragma warning(push)
#pragma warning(disable: 4996)

#include <NvOnnxParser.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <format>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace bytron {

    static std::string dims_to_string(nvinfer1::Dims d) {
        std::string s = "[";
        for (int i = 0; i < d.nbDims; ++i) {
            if (i > 0) s += ", ";
            s += std::to_string(d.d[i]);
        }
        return s + "]";
    }

    TrtEngine::TrtEngine(const BuildConfig& config)
        : l2_cache_bytes_(config.l2_cache_bytes)
    {
        // Cache-or-build: load serialized .engine if exists (fast), else build from ONNX (slow).
        if (std::filesystem::exists(config.engine_cache_path)) {
            spdlog::info("Loading cached TensorRT engine: {}",
                         config.engine_cache_path.string());
            load_from_cache(config.engine_cache_path);
        } else {
            spdlog::info("Building TensorRT engine from ONNX: {}",
                         config.onnx_path.string());
            build_from_onnx(config);
            serialize_to_file(config.engine_cache_path);
            spdlog::info("Engine cached to: {}", config.engine_cache_path.string());
        }

        auto num_io = engine_->getNbIOTensors();
        if (num_io < 2)
            throw std::runtime_error(std::format("Engine has {} I/O tensors, expected >= 2", num_io));

        input_name_  = engine_->getIOTensorName(0);
        output_name_ = engine_->getIOTensorName(1);

        spdlog::info("Input:  '{}' {}", input_name_,
                     dims_to_string(engine_->getTensorShape(input_name_.c_str())));
        spdlog::info("Output: '{}' {}", output_name_,
                     dims_to_string(engine_->getTensorShape(output_name_.c_str())));

        // Validate output shape: expect [N, max_detections, values_per_det] where
        // values_per_det >= 6 (x1, y1, x2, y2, confidence, class_id).
        auto out_dims = output_dims();
        if (out_dims.nbDims < 3 || out_dims.d[2] < 6) {
            throw std::runtime_error(std::format(
                "Unsupported output shape {}: expected [N, D, >=6] (NMS-free detection format)",
                dims_to_string(out_dims)));
        }
        if (out_dims.d[2] > 6) {
            spdlog::info("Output stride {} > 6 — extra values per detection will be ignored",
                         out_dims.d[2]);
        }

        if (num_io > 2) {
            spdlog::warn("Engine has {} I/O tensors (expected 2) — extra tensors:", num_io);
            for (int i = 2; i < num_io; ++i) {
                auto name = engine_->getIOTensorName(i);
                spdlog::warn("  [{}] '{}' {}", i, name,
                             dims_to_string(engine_->getTensorShape(name)));
            }
        }
    }

    void TrtEngine::build_from_onnx(const BuildConfig& config) {
        auto& logger = TrtLogger::instance();

        auto builder = TrtUniquePtr<nvinfer1::IBuilder>(
            nvinfer1::createInferBuilder(logger));
        if (!builder) throw std::runtime_error("Failed to create TensorRT builder");

        auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(
            builder->createNetworkV2(0));
        if (!network) throw std::runtime_error("Failed to create network definition");

        auto parser = TrtUniquePtr<nvonnxparser::IParser>(
            nvonnxparser::createParser(*network, logger));
        if (!parser) throw std::runtime_error("Failed to create ONNX parser");

        if (!parser->parseFromFile(config.onnx_path.string().c_str(),
                                   static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            std::string errors;
            for (int i = 0; i < parser->getNbErrors(); ++i)
                errors += std::format("\n  [{}] {}", i, parser->getError(i)->desc());
            throw std::runtime_error("Failed to parse ONNX:" + errors);
        }

        auto builder_config = TrtUniquePtr<nvinfer1::IBuilderConfig>(
            builder->createBuilderConfig());
        if (!builder_config) throw std::runtime_error("Failed to create builder config");

        auto* input = network->getInput(0);
        auto dims = input->getDimensions();

        builder_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                            config.max_workspace_bytes);

        if (config.max_aux_streams > 0)
            builder_config->setMaxAuxStreams(config.max_aux_streams);

        builder_config->setBuilderOptimizationLevel(config.builder_opt_level);
        builder_config->setFlag(nvinfer1::BuilderFlag::kDIRECT_IO);
        builder_config->setFlag(nvinfer1::BuilderFlag::kREJECT_EMPTY_ALGORITHMS);

        auto timing_cache_path = config.engine_cache_path;
        timing_cache_path.replace_extension(".timing");

        // Timing cache persists kernel autotuning results, dramatically reducing rebuild time.
        std::vector<char> timing_cache_data;
        if (std::filesystem::exists(timing_cache_path)) {
            std::ifstream tc_file(timing_cache_path, std::ios::binary | std::ios::ate);
            if (tc_file.is_open()) {
                auto tc_size = tc_file.tellg();
                tc_file.seekg(0, std::ios::beg);
                timing_cache_data.resize(tc_size);
                tc_file.read(timing_cache_data.data(), tc_size);
                spdlog::info("Loaded timing cache: {} ({} bytes)",
                             timing_cache_path.string(), std::size_t(tc_size));
            }
        }

        auto timing_cache = TrtUniquePtr<nvinfer1::ITimingCache>(
            builder_config->createTimingCache(
                timing_cache_data.data(), timing_cache_data.size()));
        if (timing_cache)
            builder_config->setTimingCache(*timing_cache, false);

        if (config.precision == Precision::FP16 || config.precision == Precision::INT8)
            builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);

        std::unique_ptr<Int8EntropyCalibrator> calibrator;
        if (config.precision == Precision::INT8) {
            if (config.calibration_data_dir.empty() ||
                !std::filesystem::exists(config.calibration_data_dir)) {
                spdlog::warn("INT8 requested but no calibration dir -- falling back to FP16");
            } else {
                builder_config->setFlag(nvinfer1::BuilderFlag::kINT8);
                builder_config->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
                calibrator = std::make_unique<Int8EntropyCalibrator>(
                    config.calibration_data_dir, config.calibration_batch_size,
                    static_cast<int>(dims.d[3]), static_cast<int>(dims.d[2]),
                    (config.engine_cache_path.parent_path() / "calibration.cache").string());
                builder_config->setInt8Calibrator(calibrator.get());
                spdlog::info("INT8 calibration enabled with batch size {}",
                             config.calibration_batch_size);
            }
        }

        auto* profile = builder->createOptimizationProfile();
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::Dims4{1, dims.d[1], dims.d[2], dims.d[3]});
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
            nvinfer1::Dims4{config.opt_batch_size, dims.d[1], dims.d[2], dims.d[3]});
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
            nvinfer1::Dims4{config.max_batch_size, dims.d[1], dims.d[2], dims.d[3]});
        builder_config->addOptimizationProfile(profile);

        auto build_start = std::chrono::steady_clock::now();
        spdlog::info("Building engine (FP16={}, layers={}, opt_level={}, direct_io=ON)...",
                     config.precision == Precision::FP16 || config.precision == Precision::INT8,
                     network->getNbLayers(), config.builder_opt_level);

        builder_config->setProgressMonitor(nullptr);

        auto serialized = TrtUniquePtr<nvinfer1::IHostMemory>(
            builder->buildSerializedNetwork(*network, *builder_config));
        if (!serialized) throw std::runtime_error("Failed to build serialized engine");

        spdlog::info("Engine built in {:.1f}s",
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - build_start).count());

        if (timing_cache) {
            auto updated = TrtUniquePtr<nvinfer1::IHostMemory>(
                builder_config->getTimingCache()->serialize());
            if (updated) {
                std::ofstream tc_out(timing_cache_path, std::ios::binary);
                if (tc_out.is_open() && tc_out.write(static_cast<const char*>(updated->data()), updated->size()))
                    spdlog::info("Timing cache saved: {} ({} bytes)",
                                 timing_cache_path.string(), updated->size());
                else
                    spdlog::warn("Failed to write timing cache: {}", timing_cache_path.string());
            }
        }

        runtime_ = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
        if (!runtime_) throw std::runtime_error("Failed to create TRT runtime");

        engine_ = TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(serialized->data(), serialized->size()));
        if (!engine_) throw std::runtime_error("Failed to deserialize engine");
    }

    void TrtEngine::load_from_cache(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open())
            throw std::runtime_error("Failed to open engine file: " + path.string());

        auto size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> data(size);
        file.read(data.data(), size);

        auto& logger = TrtLogger::instance();
        runtime_ = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
        if (!runtime_) throw std::runtime_error("Failed to create TRT runtime");

        engine_ = TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(data.data(), data.size()));
        if (!engine_)
            throw std::runtime_error(
                "Failed to deserialize engine (cache may be stale -- delete .engine and rebuild)");
    }

    void TrtEngine::serialize_to_file(const std::filesystem::path& path) {
        if (path.has_parent_path())
            std::filesystem::create_directories(path.parent_path());

        auto serialized = TrtUniquePtr<nvinfer1::IHostMemory>(engine_->serialize());
        if (!serialized) throw std::runtime_error("Failed to serialize engine");

        std::ofstream file(path, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("Failed to create engine file: " + path.string());
        file.write(static_cast<const char*>(serialized->data()), serialized->size());
    }

    TrtUniquePtr<nvinfer1::IExecutionContext> TrtEngine::create_context() {
        auto ctx = TrtUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!ctx) throw std::runtime_error("Failed to create TRT execution context");

        if (l2_cache_bytes_ > 0)
            ctx->setPersistentCacheLimit(l2_cache_bytes_);

        return ctx;
    }

    const char* TrtEngine::input_name()  const noexcept { return input_name_.c_str(); }
    const char* TrtEngine::output_name() const noexcept { return output_name_.c_str(); }

    nvinfer1::Dims TrtEngine::input_dims()  const { return engine_->getTensorShape(input_name_.c_str()); }
    nvinfer1::Dims TrtEngine::output_dims() const { return engine_->getTensorShape(output_name_.c_str()); }

    static int safe_dim_cast(std::int64_t val, const char* name) {
        if (val < 0 || val > std::numeric_limits<int>::max())
            throw std::runtime_error(std::format("TRT dimension '{}' value {} out of int range", name, val));
        return static_cast<int>(val);
    }

    int TrtEngine::model_width()    const { return safe_dim_cast(input_dims().d[3], "model_width"); }
    int TrtEngine::model_height()   const { return safe_dim_cast(input_dims().d[2], "model_height"); }
    int TrtEngine::max_detections() const { return safe_dim_cast(output_dims().d[1], "max_detections"); }
    int TrtEngine::output_stride()  const { return safe_dim_cast(output_dims().d[2], "output_stride"); }

}

#pragma warning(pop)
