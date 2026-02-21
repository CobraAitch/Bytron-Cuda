# C++20 and CUDA compiler options for MSVC + NVCC
# Works with multi-config generators (Visual Studio) — no CMAKE_BUILD_TYPE checks.

if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    )

    # Release optimizations — use generator expressions for multi-config
    add_compile_options(
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/GL>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:/O2>
    )
    add_link_options(
        $<$<CONFIG:Release>:/LTCG>
    )
endif()

# CUDA-specific flags
if(CMAKE_CUDA_COMPILER)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>)

    # PTX-level optimization for Release builds.
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-Xptxas=-O3>)

    if(MSVC)
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
    endif()
endif()
