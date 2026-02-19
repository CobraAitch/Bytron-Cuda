#include "engine/trt_engine.hpp"
#include "engine/trt_logger.hpp"
#include "engine/calibrator.hpp"

// TRT 10.15 still uses kFP16/kINT8/setInt8Calibrator which are deprecated
#pragma warning(push)
#pragma warning(disable: 4996)

#include <NvOnnxParser.h>
#include <spdlog/spdlog.h>

#include <fstream>
#include <vector>
#include <stdexcept>
#include <format>
#include <chrono>

namespace bytron {

    TrtEngine::TrtEngine(const BuildConfig& config) {
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

        input_name_ = engine_->getIOTensorName(0);
        output_name_ = engine_->getIOTensorName(1);

        // Log tensor shapes for sanity checking
        spdlog::info("Input:  '{}' {}", input_name_, [&] {
            auto d = engine_->getTensorShape(input_name_.c_str());
            std::string s = "[";
            for (int i = 0; i < d.nbDims; ++i) {
                if (i > 0) s += ", ";
                s += std::to_string(d.d[i]);
            }
            return s + "]";
        }());

        spdlog::info("Output: '{}' {}", output_name_, [&] {
            auto d = engine_->getTensorShape(output_name_.c_str());
            std::string s = "[";
            for (int i = 0; i < d.nbDims; ++i) {
                if (i > 0) s += ", ";
                s += std::to_string(d.d[i]);
            }
            return s + "]";
        }());
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
            for (int i = 0; i < parser->getNbErrors(); ++i) {
                errors += std::format("\n  [{}] {}", i, parser->getError(i)->desc());
            }
            throw std::runtime_error("Failed to parse ONNX:" + errors);
        }

        auto builder_config = TrtUniquePtr<nvinfer1::IBuilderConfig>(
            builder->createBuilderConfig());
        if (!builder_config) throw std::runtime_error("Failed to create builder config");

        auto* input = network->getInput(0);
        auto dims = input->getDimensions();

        builder_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                            config.max_workspace_bytes);

        if (config.max_aux_streams > 0) {
            builder_config->setMaxAuxStreams(config.max_aux_streams);
        }

        // -- Timing cache: saves kernel profiling so rebuilds are near-instant --
        auto timing_cache_path = config.engine_cache_path;
        timing_cache_path.replace_extension(".timing");

        std::vector<char> timing_cache_data;
        if (std::filesystem::exists(timing_cache_path)) {
            std::ifstream tc_file(timing_cache_path, std::ios::binary | std::ios::ate);
            if (tc_file.is_open()) {
                auto tc_size = tc_file.tellg();
                tc_file.seekg(0, std::ios::beg);
                timing_cache_data.resize(tc_size);
                tc_file.read(timing_cache_data.data(), tc_size);
                spdlog::info("Loaded timing cache: {} ({} bytes)",
                             timing_cache_path.string(), static_cast<std::size_t>(tc_size));
            }
        }

        auto* timing_cache = builder_config->createTimingCache(
            timing_cache_data.data(), timing_cache_data.size());
        if (timing_cache) {
            builder_config->setTimingCache(*timing_cache, false);
        }

        // Precision flags
        if (config.precision == Precision::FP16 || config.precision == Precision::INT8) {
            builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }

        // INT8 calibrator (must outlive buildSerializedNetwork)
        std::unique_ptr<Int8EntropyCalibrator> calibrator;
        if (config.precision == Precision::INT8) {
            if (config.calibration_data_dir.empty() ||
                !std::filesystem::exists(config.calibration_data_dir)) {
                spdlog::warn("INT8 requested but no calibration dir -- falling back to FP16");
            } else {
                builder_config->setFlag(nvinfer1::BuilderFlag::kINT8);
                calibrator = std::make_unique<Int8EntropyCalibrator>(
                    config.calibration_data_dir,
                    config.calibration_batch_size,
                    static_cast<int>(dims.d[3]), static_cast<int>(dims.d[2]),
                    (config.engine_cache_path.parent_path() / "calibration.cache").string()
                );
                builder_config->setInt8Calibrator(calibrator.get());
                spdlog::info("INT8 calibration enabled with batch size {}", config.calibration_batch_size);
            }
        }

        // Optimization profile (explicit batch, single-image)
        auto* profile = builder->createOptimizationProfile();
        profile->setDimensions(input->getName(),
            nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::Dims4{1, dims.d[1], dims.d[2], dims.d[3]});
        profile->setDimensions(input->getName(),
            nvinfer1::OptProfileSelector::kOPT,
            nvinfer1::Dims4{config.opt_batch_size, dims.d[1], dims.d[2], dims.d[3]});
        profile->setDimensions(input->getName(),
            nvinfer1::OptProfileSelector::kMAX,
            nvinfer1::Dims4{config.max_batch_size, dims.d[1], dims.d[2], dims.d[3]});
        builder_config->addOptimizationProfile(profile);

        // -- Build engine --
        auto build_start = std::chrono::steady_clock::now();
        spdlog::info("Building engine (FP16={}, layers={})...",
                     config.precision == Precision::FP16 || config.precision == Precision::INT8,
                     network->getNbLayers());

        builder_config->setProgressMonitor(nullptr);

        auto serialized = TrtUniquePtr<nvinfer1::IHostMemory>(
            builder->buildSerializedNetwork(*network, *builder_config));
        if (!serialized) throw std::runtime_error("Failed to build serialized engine");

        auto build_elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - build_start).count();
        spdlog::info("Engine built in {:.1f}s", build_elapsed);

        // -- Persist timing cache for next build --
        if (timing_cache) {
            auto updated_cache = TrtUniquePtr<nvinfer1::IHostMemory>(
                builder_config->getTimingCache()->serialize());
            if (updated_cache) {
                std::ofstream tc_out(timing_cache_path, std::ios::binary);
                tc_out.write(static_cast<const char*>(updated_cache->data()),
                             updated_cache->size());
                spdlog::info("Timing cache saved: {} ({} bytes)",
                             timing_cache_path.string(), updated_cache->size());
            }
            delete timing_cache;
        }

        runtime_ = TrtUniquePtr<nvinfer1::IRuntime>(
            nvinfer1::createInferRuntime(logger));
        if (!runtime_) throw std::runtime_error("Failed to create TRT runtime");

        engine_ = TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(serialized->data(), serialized->size()));
        if (!engine_) throw std::runtime_error("Failed to deserialize engine");
    }

    void TrtEngine::load_from_cache(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open engine file: " + path.string());
        }

        auto size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> data(size);
        file.read(data.data(), size);

        auto& logger = TrtLogger::instance();
        runtime_ = TrtUniquePtr<nvinfer1::IRuntime>(
            nvinfer1::createInferRuntime(logger));
        if (!runtime_) throw std::runtime_error("Failed to create TRT runtime");

        engine_ = TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(data.data(), data.size()));
        if (!engine_) {
            throw std::runtime_error(
                "Failed to deserialize engine (cache may be stale -- delete .engine and rebuild)");
        }
    }

    void TrtEngine::serialize_to_file(const std::filesystem::path& path) {
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }

        auto serialized = TrtUniquePtr<nvinfer1::IHostMemory>(
            engine_->serialize());
        if (!serialized) throw std::runtime_error("Failed to serialize engine");

        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to create engine file: " + path.string());
        }
        file.write(static_cast<const char*>(serialized->data()), serialized->size());
    }

    TrtUniquePtr<nvinfer1::IExecutionContext> TrtEngine::create_context() {
        auto ctx = TrtUniquePtr<nvinfer1::IExecutionContext>(
            engine_->createExecutionContext());
        if (!ctx) throw std::runtime_error("Failed to create TRT execution context");
        return ctx;
    }

    const char* TrtEngine::input_name() const { return input_name_.c_str(); }
    const char* TrtEngine::output_name() const { return output_name_.c_str(); }

    nvinfer1::Dims TrtEngine::input_dims() const {
        return engine_->getTensorShape(input_name_.c_str());
    }

    nvinfer1::Dims TrtEngine::output_dims() const {
        return engine_->getTensorShape(output_name_.c_str());
    }

    int TrtEngine::model_width() const {
        auto dims = input_dims();
        return static_cast<int>(dims.d[3]); // NCHW: d[3] = W
    }

    int TrtEngine::model_height() const {
        auto dims = input_dims();
        return static_cast<int>(dims.d[2]); // NCHW: d[2] = H
    }

    int TrtEngine::max_detections() const {
        auto dims = output_dims();
        return static_cast<int>(dims.d[1]); // [1, 300, 6]: d[1] = 300
    }
}

#pragma warning(pop)
