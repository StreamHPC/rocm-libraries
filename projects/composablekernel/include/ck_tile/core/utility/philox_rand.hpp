// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"

namespace ck_tile {

// Reference: https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/philox.cuh
class philox
{
    public:
    CK_TILE_HOST_DEVICE philox(unsigned long long seed_, unsigned long long offset_)
    {
        seed.x   = static_cast<uint32_t>(seed_);
        seed.y   = static_cast<uint32_t>(seed_ >> 32);
        offset.x = static_cast<uint32_t>(offset_);
        offset.y = static_cast<uint32_t>(offset_ >> 32);
    }

    CK_TILE_HOST_DEVICE uint32x4_t get_philox_4x32(const unsigned long long subsequence) const
    {
        uint32x4_t counter_;
        counter_.x = offset.x;
        counter_.y = offset.y;
        counter_.z = static_cast<uint32_t>(subsequence);
        counter_.w = static_cast<uint32_t>(subsequence >> 32);

        uint32x2_t key_ = seed;
// 7-round philox
#pragma unroll
        for(int i = 0; i < 6; i++)
        {
            counter_ = philox_single_round(counter_, key_);
            key_.x += kPhilox10A;
            key_.y += kPhilox10B;
        }
        uint32x4_t output = philox_single_round(counter_, key_);
        return output;
    }

    CK_TILE_HOST_DEVICE void get_random_16x8(uint8_t* out,
                                             const unsigned long long subsequence) const
    {
        uint32x4_t tmp_ph;
        tmp_ph = get_philox_4x32(subsequence);

        __builtin_memcpy(out, &tmp_ph, sizeof(tmp_ph));
    }

    CK_TILE_HOST_DEVICE void get_random_8x8(uint8_t* out,
                                            const unsigned long long subsequence,
                                            const index_t idx0,
                                            const index_t idx1) const
    {
        uint32x4_t tmp_ph;
        tmp_ph = get_philox_4x32(subsequence);

        uint32_t out_tmp[2];
        out_tmp[0] = tmp_ph[idx0];
        out_tmp[1] = tmp_ph[idx1];
        __builtin_memcpy(out, &out_tmp, sizeof(out_tmp));
    }

    CK_TILE_HOST_DEVICE void
    get_random_4x8(uint8_t* out, const unsigned long long subsequence, const index_t idx) const
    {
        uint32x4_t tmp_ph;
        tmp_ph = get_philox_4x32(subsequence);

        uint32_t out_tmp;
        out_tmp = tmp_ph[idx];
        __builtin_memcpy(out, &out_tmp, sizeof(out_tmp));
    }

    private:
    uint32x2_t offset;
    uint32x2_t seed;

    CK_TILE_HOST_DEVICE uint32x2_t mulhilo32(const unsigned int a, const unsigned int b) const
    {
        unsigned long long tmp = static_cast<unsigned long long>(a) * b;
        uint32x2_t res;
        res.x = static_cast<uint32_t>(tmp);
        res.y = static_cast<uint32_t>(tmp >> 32);
        return res;
    }

    CK_TILE_HOST_DEVICE uint32x4_t philox_single_round(const uint32x4_t ctr,
                                                       const uint32x2_t key) const
    {
        uint32x2_t res0 = mulhilo32(kPhiloxSA, ctr.x);
        uint32x2_t res1 = mulhilo32(kPhiloxSB, ctr.z);
        uint32x4_t ret  = {res1.y ^ ctr.y ^ key.x, res1.x, res0.y ^ ctr.w ^ key.y, res0.x};
        return ret;
    }

    static const unsigned int kPhilox10A = 0x9E3779B9;
    static const unsigned int kPhilox10B = 0xBB67AE85;
    static const unsigned int kPhiloxSA  = 0xD2511F53;
    static const unsigned int kPhiloxSB  = 0xCD9E8D57;
};

} // namespace ck_tile
