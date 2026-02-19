# FindTensorRT.cmake
# Finds TensorRT library and headers on Windows.
#
# Sets:
#   TensorRT_FOUND
#   TensorRT::nvinfer
#   TensorRT::nvonnxparser

# Search hints
set(_TRT_SEARCH_PATHS
    "$ENV{TENSORRT_ROOT}"
    "$ENV{TRT_ROOT}"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT"
    "C:/TensorRT"
)

# Find include directory
find_path(TensorRT_INCLUDE_DIR NvInfer.h
    HINTS ${_TRT_SEARCH_PATHS}
    PATH_SUFFIXES include
)

# Find libraries
find_library(TensorRT_NVINFER_LIBRARY
    NAMES nvinfer nvinfer_10
    HINTS ${_TRT_SEARCH_PATHS}
    PATH_SUFFIXES lib lib/x64
)

find_library(TensorRT_NVONNXPARSER_LIBRARY
    NAMES nvonnxparser nvonnxparser_10
    HINTS ${_TRT_SEARCH_PATHS}
    PATH_SUFFIXES lib lib/x64
)

# Determine version from header
if(TensorRT_INCLUDE_DIR)
    file(READ "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _TRT_VERSION_HEADER)
    string(REGEX MATCH "#define NV_TENSORRT_MAJOR ([0-9]+)" _ ${_TRT_VERSION_HEADER})
    set(TensorRT_VERSION_MAJOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "#define NV_TENSORRT_MINOR ([0-9]+)" _ ${_TRT_VERSION_HEADER})
    set(TensorRT_VERSION_MINOR ${CMAKE_MATCH_1})
    set(TensorRT_VERSION "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}")
    message(STATUS "TensorRT version: ${TensorRT_VERSION}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS
        TensorRT_INCLUDE_DIR
        TensorRT_NVINFER_LIBRARY
        TensorRT_NVONNXPARSER_LIBRARY
    VERSION_VAR TensorRT_VERSION
)

if(TensorRT_FOUND)
    # Locate the DLL directory (sibling "bin/" next to "lib/")
    get_filename_component(_TRT_LIB_DIR "${TensorRT_NVINFER_LIBRARY}" DIRECTORY)
    get_filename_component(_TRT_ROOT_DIR "${_TRT_LIB_DIR}" DIRECTORY)
    set(TensorRT_BIN_DIR "${_TRT_ROOT_DIR}/bin" CACHE PATH "TensorRT DLL directory")

    # nvinfer target
    if(NOT TARGET TensorRT::nvinfer)
        add_library(TensorRT::nvinfer SHARED IMPORTED)
        set_target_properties(TensorRT::nvinfer PROPERTIES
            IMPORTED_IMPLIB "${TensorRT_NVINFER_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
        )
    endif()

    # nvonnxparser target
    if(NOT TARGET TensorRT::nvonnxparser)
        add_library(TensorRT::nvonnxparser SHARED IMPORTED)
        set_target_properties(TensorRT::nvonnxparser PROPERTIES
            IMPORTED_IMPLIB "${TensorRT_NVONNXPARSER_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
        )
    endif()
endif()
