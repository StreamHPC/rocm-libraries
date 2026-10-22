// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Repro (not for merge): comp_tdm / comp_tdm_v2, Row/Col/Row, power-of-two K_Tile. FP8 failed
// multi-iteration K loops on develop before #13808 (BF8 was only seen failing at K_Tile=192).
// INT8 (same sizeof as FP8, so same sub-tiles, LDS layout and padding) and FP16 are controls.
#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

#include <string>

// clang-format off
using KernelTypesCompTDMPow2Repro = ::testing::Types<
    // 128x128x128, 1 sub-tile
    std::tuple<Row, Col, Row, F8,   F8,   F32,   F16,   I128, I128, I128, I16, I16, Intrawave, CompTDMV1>,
    std::tuple<Row, Col, Row, F8,   F8,   F32,   F16,   I128, I128, I128, I16, I16, Intrawave, CompTDMV2>,
    std::tuple<Row, Col, Row, BF8,  BF8,  F32,   F16,   I128, I128, I128, I16, I16, Intrawave, CompTDMV1>,
    std::tuple<Row, Col, Row, BF8,  BF8,  F32,   F16,   I128, I128, I128, I16, I16, Intrawave, CompTDMV2>,
    std::tuple<Row, Col, Row, F16,  F16,  F32,   F16,   I128, I128, I128, I16, I16, Intrawave, CompTDMV1>,
    std::tuple<Row, Col, Row, F16,  F16,  F32,   F16,   I128, I128, I128, I16, I16, Intrawave, CompTDMV2>,
    std::tuple<Row, Col, Row, INT8, INT8, INT32, INT32, I128, I128, I128, I16, I16, Intrawave, CompTDMV2>,
    // 256x256x128, 2 sub-tiles (8-bit types)
    std::tuple<Row, Col, Row, F8,   F8,   F32,   F16,   I256, I256, I128, I16, I16, Intrawave, CompTDMV1>,
    std::tuple<Row, Col, Row, F8,   F8,   F32,   F16,   I256, I256, I128, I16, I16, Intrawave, CompTDMV2>,
    std::tuple<Row, Col, Row, INT8, INT8, INT32, INT32, I256, I256, I128, I16, I16, Intrawave, CompTDMV1>,
    std::tuple<Row, Col, Row, INT8, INT8, INT32, INT32, I256, I256, I128, I16, I16, Intrawave, CompTDMV2>,
    // 256x256x128 FP16: 4 sub-tiles, comparable with the earlier develop run only
    std::tuple<Row, Col, Row, F16,  F16,  F32,   F16,   I256, I256, I128, I16, I16, Intrawave, CompTDMV2>,
    // 256x256x64, 1 sub-tile: separates block tile size from the sub-tile path
    std::tuple<Row, Col, Row, F8,   F8,   F32,   F16,   I256, I256, I64,  I16, I16, Intrawave, CompTDMV1>,
    std::tuple<Row, Col, Row, F8,   F8,   F32,   F16,   I256, I256, I64,  I16, I16, Intrawave, CompTDMV2>
>;
// clang-format on

template <typename T>
class TestCkTileGemmPipelineCompTDMPow2Repro
    : public TestCkTileGemmPipelineWmmaBase<T, TestCkTileGemmPipelineCompTDMPow2Repro<T>>
{
    protected:
    // Regular (non-cluster) launch, which works on asicRevision=0, so the base-class TDM revision
    // skip does not apply. Run() does nothing for an unsupported combination, hence the assert.
    void SetUp() override
    {
        static_assert(TestCkTileGemmPipelineCompTDMPow2Repro<T>::check_data_type(),
                      "unsupported data type combination: this case would pass without running");
        this->k_batches_ = {1};
    }
};

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompTDMPow2Repro, KernelTypesCompTDMPow2Repro);

TYPED_TEST(TestCkTileGemmPipelineCompTDMPow2Repro, ExactTile)
{
    this->Run(TestFixture::M_Tile, TestFixture::N_Tile, TestFixture::K_Tile);
}

TYPED_TEST(TestCkTileGemmPipelineCompTDMPow2Repro, KLoops)
{
    for(const int k_loops : {2, 3, 4, 5, 6, 7, 8, 16})
    {
        SCOPED_TRACE("k_loops = " + std::to_string(k_loops));
        this->Run(2 * TestFixture::M_Tile, 2 * TestFixture::N_Tile, k_loops * TestFixture::K_Tile);
    }
}
