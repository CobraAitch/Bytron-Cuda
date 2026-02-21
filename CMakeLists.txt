cmake_minimum_required(VERSION 3.28)

# ──────────────────────────── vcpkg toolchain ────────────────────────────
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE FILEPATH "vcpkg toolchain")
endif()

# ──────────────────────────── Project ────────────────────────────────────
project(Bytron
    VERSION   0.1.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerOptions)

# ──────────────────────────── CUDA ───────────────────────────────────────
# SM 89 = RTX 4070 Ti (Ada Lovelace)
# SM 120 = RTX 5070 (Blackwell) — requires CUDA 12.8+
# Only request SM 120 if the toolkit supports it.
set(BYTRON_CUDA_ARCHS "89")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
    list(APPEND BYTRON_CUDA_ARCHS "120")
endif()
set(CMAKE_CUDA_ARCHITECTURES ${BYTRON_CUDA_ARCHS})

# ──────────────────────────── Dependencies ───────────────────────────────
find_package(CUDAToolkit    REQUIRED)
find_package(TensorRT       REQUIRED)
find_package(Vulkan         REQUIRED)
find_package(glfw3          REQUIRED)
find_package(spdlog         REQUIRED)
find_package(nlohmann_json  REQUIRED)

# ──────────────────────────── Sources ────────────────────────────────────
set(BYTRON_SOURCES
    src/core/pipeline.cpp
    src/capture/desktop_capture.cpp
    src/preprocess/gpu_preprocessor.cpp
    src/engine/trt_engine.cpp
    src/engine/calibrator.cpp
    src/postprocess/postprocessor.cpp
    src/output/display_sink.cpp
    src/output/overlay_sink.cpp
    src/output/overlay_window.cpp
    src/output/vulkan_display.cpp
    src/util/config.cpp
    src/util/stb_image_impl.cpp
)

set(BYTRON_CUDA_SOURCES
    src/preprocess/preprocess_kernels.cu
    src/output/overlay_kernels.cu
)

# ──────────────────────────── Library target ─────────────────────────────
add_library(bytron_lib STATIC ${BYTRON_SOURCES} ${BYTRON_CUDA_SOURCES})

target_include_directories(bytron_lib
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(bytron_lib
    PUBLIC
        CUDA::cudart
        TensorRT::nvinfer
        TensorRT::nvonnxparser
        Vulkan::Vulkan
        glfw
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        d3d11
        dxgi
)

# ──────────────────────────── Executable ─────────────────────────────────
add_executable(bytron src/main.cpp)
target_link_libraries(bytron PRIVATE bytron_lib)

# ──────────────────────────── Copy runtime DLLs ─────────────────────────
# TensorRT DLLs live in a separate bin/ dir and aren't on PATH by default.
if(TensorRT_BIN_DIR AND EXISTS "${TensorRT_BIN_DIR}")
    file(GLOB _TRT_DLLS "${TensorRT_BIN_DIR}/*.dll")
    foreach(_dll ${_TRT_DLLS})
        cmake_path(GET _dll FILENAME _dll_name)
        add_custom_command(TARGET bytron POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll}" "$<TARGET_FILE_DIR:bytron>"
            COMMENT "Copying ${_dll_name}"
        )
    endforeach()
endif()
