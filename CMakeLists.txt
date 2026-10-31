cmake_minimum_required(VERSION 3.28)

# --- vcpkg toolchain --------------------------------------------------------
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE FILEPATH "vcpkg toolchain")
endif()

# --- Build options ----------------------------------------------------------
# Tests and benchmarks cover CPU code only. Turning BYTRON_BUILD_APP off drops
# the CUDA, TensorRT and Vulkan requirements, so they build on any Windows box.
option(BYTRON_BUILD_APP "Build the GPU pipeline and the bytron executable" ON)
option(BYTRON_BUILD_TESTS "Build unit tests" OFF)
option(BYTRON_BUILD_BENCHMARKS "Build CPU micro-benchmarks" OFF)

# --- Project ----------------------------------------------------------------
set(BYTRON_LANGUAGES CXX)
if(BYTRON_BUILD_APP)
    list(APPEND BYTRON_LANGUAGES CUDA)
endif()

project(Bytron
    VERSION   0.1.0
    LANGUAGES ${BYTRON_LANGUAGES}
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerOptions)

if(BYTRON_BUILD_APP)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # SM 89 = RTX 4070 Ti (Ada Lovelace)
    # SM 120 = RTX 5070 (Blackwell), requires CUDA 12.8+
    # Only request SM 120 if the toolkit supports it.
    set(BYTRON_CUDA_ARCHS "89")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
        list(APPEND BYTRON_CUDA_ARCHS "120")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES ${BYTRON_CUDA_ARCHS})
endif()

# --- Dependencies -----------------------------------------------------------
find_package(spdlog        REQUIRED)
find_package(nlohmann_json REQUIRED)

if(BYTRON_BUILD_APP)
    find_package(CUDAToolkit REQUIRED)
    find_package(TensorRT    REQUIRED)
    find_package(Vulkan      REQUIRED)
    find_package(glfw3       REQUIRED)
endif()

# --- Application ------------------------------------------------------------
if(BYTRON_BUILD_APP)
    set(BYTRON_SOURCES
        src/core/pipeline.cpp
        src/capture/capture_source.cpp
        src/capture/desktop_capture.cpp
        src/capture/graphics_capture.cpp
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

    # The C++/WinRT projection headers live in <SDK>/Include/<ver>/cppwinrt, which
    # is NOT on the default compiler include path. Locate it from the Windows SDK.
    get_filename_component(_win10_sdk_root
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots;KitsRoot10]"
        ABSOLUTE CACHE)
    if(NOT _win10_sdk_root OR NOT EXISTS "${_win10_sdk_root}")
        set(_win10_sdk_root "C:/Program Files (x86)/Windows Kits/10")
    endif()

    set(_sdk_ver "${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}")
    if(NOT _sdk_ver)
        set(_sdk_ver "10.0.26100.0")
    endif()

    set(BYTRON_CPPWINRT_DIR "${_win10_sdk_root}/Include/${_sdk_ver}/cppwinrt")
    if(NOT EXISTS "${BYTRON_CPPWINRT_DIR}/winrt/base.h")
        message(WARNING
            "C++/WinRT headers not found at ${BYTRON_CPPWINRT_DIR}; "
            "Windows.Graphics.Capture backend will fail to build.")
    endif()

    add_library(bytron_lib STATIC ${BYTRON_SOURCES} ${BYTRON_CUDA_SOURCES})

    target_include_directories(bytron_lib
        PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
        PRIVATE "${BYTRON_CPPWINRT_DIR}"
    )

    # Prevent <windows.h> from defining min/max macros that conflict with std::min/max.
    target_compile_definitions(bytron_lib PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)

    target_link_libraries(bytron_lib
        PUBLIC
            CUDA::cudart
            TensorRT::nvinfer
            Vulkan::Vulkan
            glfw
            spdlog::spdlog
        PRIVATE
            TensorRT::nvonnxparser
            nlohmann_json::nlohmann_json
            d3d11
            dxgi
            windowsapp   # WinRT activation for Windows.Graphics.Capture
    )

    add_executable(bytron src/main.cpp)
    target_link_libraries(bytron PRIVATE bytron_lib)

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
endif()

# --- Tests ------------------------------------------------------------------
# CPU-side units only (letterbox math, postprocess transform, config). Relevant
# sources are compiled in directly to keep the CUDA/TRT/Vulkan link chain out
# of the test binary, so `ctest` runs without a GPU.
if(BYTRON_BUILD_TESTS)
    find_package(GTest CONFIG REQUIRED)
    enable_testing()
    include(GoogleTest)

    add_executable(bytron_tests
        tests/test_letterbox.cpp
        tests/test_postprocessor.cpp
        tests/test_config.cpp
        src/postprocess/postprocessor.cpp
        src/util/config.cpp
    )
    target_include_directories(bytron_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_definitions(bytron_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_link_libraries(bytron_tests PRIVATE
        GTest::gtest GTest::gtest_main
        spdlog::spdlog
        nlohmann_json::nlohmann_json)

    gtest_discover_tests(bytron_tests)
endif()

# --- Benchmarks -------------------------------------------------------------
if(BYTRON_BUILD_BENCHMARKS)
    add_executable(bytron_bench
        benchmarks/bench_cpu.cpp
        src/postprocess/postprocessor.cpp)
    target_include_directories(bytron_bench PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_definitions(bytron_bench PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
