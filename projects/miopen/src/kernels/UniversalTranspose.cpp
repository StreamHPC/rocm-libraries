// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#endif

#include "float_types.h"
#include "miopen_cstdint.hpp"

using index_t = INDEX_TYPE;

extern "C" __global__ __launch_bounds__(256) void UniversalTranspose(const FLOAT* __restrict__ in,
                                                                     FLOAT* __restrict__ out,
                                                                     index_t lens_n,
                                                                     index_t lens_c,
                                                                     index_t lens_d,
                                                                     index_t lens_h,
                                                                     index_t lens_w,
                                                                     index_t in_strides_n,
                                                                     index_t in_strides_c,
                                                                     index_t in_strides_d,
                                                                     index_t in_strides_h,
                                                                     index_t in_strides_w,
                                                                     index_t out_strides_n,
                                                                     index_t out_strides_c,
                                                                     index_t out_strides_d,
                                                                     index_t out_strides_h,
                                                                     index_t out_strides_w)
{
    const index_t global_size = static_cast<index_t>(gridDim.x) * blockDim.x;
    const index_t global_id   = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const index_t lens_wh    = lens_w * lens_h;
    const index_t lens_whd   = lens_wh * lens_d;
    const index_t lens_whdc  = lens_whd * lens_c;
    const index_t lens_whdcn = lens_whdc * lens_n;

    for(index_t id = global_id; id < lens_whdcn; id += global_size)
    {
        const index_t n     = id / lens_whdc;
        const index_t rem_n = id - n * lens_whdc;
        const index_t c     = rem_n / lens_whd;
        const index_t rem_c = rem_n - c * lens_whd;
        const index_t d     = rem_c / lens_wh;
        const index_t rem_d = rem_c - d * lens_wh;
        const index_t h     = rem_d / lens_w;
        const index_t w     = rem_d - h * lens_w;

        const index_t in_id = n * in_strides_n + c * in_strides_c + d * in_strides_d +
                              h * in_strides_h + w * in_strides_w;

        const index_t out_id = n * out_strides_n + c * out_strides_c + d * out_strides_d +
                               h * out_strides_h + w * out_strides_w;

        out[out_id] = in[in_id];
    }
}
