# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(SUPPORTED_GPUS gfx908 gfx90a gfx942 gfx950 gfx1201 gfx1250)

set(has_supported_gpu FALSE)
foreach(gpu IN LISTS GPU_TARGETS)
    # gfx1250-strict is the same ISA family as gfx1250; strip the suffix before the exact match.
    string(REGEX REPLACE "-strict$" "" gpu "${gpu}")
    if(gpu IN_LIST SUPPORTED_GPUS)
        set(has_supported_gpu TRUE)
        break()
    endif()
endforeach()

if(has_supported_gpu)
    # NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
    #       ... because they are auto-generated
    set(EXAMPLE_FLATMM_COMPILE_OPTIONS -Wno-undefined-func-template)
    set(EXAMPLE_MOE_FLATMM_COMPILE_OPTIONS)

    if(CK_USE_OCP_FP8)
        list(APPEND EXAMPLE_FLATMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
    endif()
    # [WIP diag, issue 509] the gtests build the same MX TDM instances with
    # -enable-noalias-to-md-conversion=0 instead and pass where these examples fail.
    # Empty means no -mllvm flag at all.
    set(CK_FLATMM_EXAMPLE_MLLVM "-greedy-reverse-local-assignment=1" CACHE STRING
        "Semicolon-separated -mllvm flags for the 18_flatmm examples")
    foreach(flag IN LISTS CK_FLATMM_EXAMPLE_MLLVM)
        list(APPEND EXAMPLE_FLATMM_COMPILE_OPTIONS "SHELL: -mllvm ${flag}")
    endforeach()
    add_executable(tile_example_flatmm_basic flatmm_basic.cpp)
    target_compile_options(tile_example_flatmm_basic PRIVATE ${EXAMPLE_FLATMM_COMPILE_OPTIONS})

    add_executable(tile_example_moe_flatmm moe_flatmm.cpp)
    target_compile_options(tile_example_moe_flatmm PRIVATE ${EXAMPLE_FLATMM_COMPILE_OPTIONS})

    add_executable(tile_example_grouped_flatmm grouped_flatmm.cpp)
    target_compile_options(tile_example_grouped_flatmm PRIVATE ${EXAMPLE_FLATMM_COMPILE_OPTIONS})

    if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx12")
        add_executable(tile_example_mixed_prec_flatmm mixed_prec/mixed_prec_flatmm.cpp)
        target_compile_options(tile_example_mixed_prec_flatmm PRIVATE ${EXAMPLE_FLATMM_COMPILE_OPTIONS})

        add_executable(tile_example_a16w4_moe_flatmm mixed_prec/a16w4_moe_flatmm.cpp)
        target_compile_options(tile_example_a16w4_moe_flatmm PRIVATE ${EXAMPLE_FLATMM_COMPILE_OPTIONS})
    endif()
    if (GPU_TARGETS MATCHES "gfx95" OR GPU_TARGETS MATCHES "gfx125")
        include(mxgemm/mx_flatmm_instance.cmake)
        mx_flatmm_instance_generate(EXAMPLE_MX_FLATMM_FILES)
        message(STATUS "Generated MX FlatMM kernel files: ${EXAMPLE_MX_FLATMM_FILES}")

        add_executable(tile_example_mx_flatmm mxgemm/mx_flatmm.cpp ${EXAMPLE_MX_FLATMM_FILES})
        target_include_directories(tile_example_mx_flatmm PRIVATE mxgemm)
        target_compile_options(tile_example_mx_flatmm PRIVATE ${EXAMPLE_FLATMM_COMPILE_OPTIONS})
    endif()
    if (GPU_TARGETS MATCHES "gfx125")
        add_executable(tile_example_mx_flatmm_mxgemm_data_cache_prefetch mxgemm/mx_flatmm_data_cache_prefetch.cpp)
        target_include_directories(tile_example_mx_flatmm_mxgemm_data_cache_prefetch PRIVATE mxgemm)
        target_compile_options(tile_example_mx_flatmm_mxgemm_data_cache_prefetch PRIVATE ${EXAMPLE_FLATMM_COMPILE_OPTIONS})
    endif()
endif()
