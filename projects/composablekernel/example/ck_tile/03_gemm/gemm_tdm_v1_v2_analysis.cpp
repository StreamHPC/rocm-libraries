// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Minimal FP16 RCR GEMM driver for TDM V1/V2 sanity checks.

#include "gemm_utils.hpp"
#include "run_gemm_example.inc"
#include "run_gemm_example_common.hpp"
#include "universal_gemm_invoker.hpp"

template <ck_tile::GemmPipeline Pipeline_, ck_tile::index_t MWave_, ck_tile::index_t NWave_>
struct TdmAnalysisConfig : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = MWave_;
    static constexpr ck_tile::index_t N_Warp = NWave_;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<ck_tile::half_t, M_Warp_Tile>();

    static constexpr bool kPadM = true;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;

    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr ck_tile::GemmPipeline Pipeline = Pipeline_;
    static constexpr ck_tile::DataCachePrefetchKind DataCachePrefetchA =
        ck_tile::DataCachePrefetchKind::None;
    static constexpr ck_tile::DataCachePrefetchKind DataCachePrefetchB =
        ck_tile::DataCachePrefetchKind::None;

    static constexpr ck_tile::index_t kClusterSizeM = 1;
    static constexpr ck_tile::index_t kClusterSizeN = 1;
};

using TdmV1Config = TdmAnalysisConfig<ck_tile::GemmPipeline::COMPUTE_TDM_V1, 2, 4>;
using TdmV2Config = TdmAnalysisConfig<ck_tile::GemmPipeline::COMPUTE_TDM_V2, 2, 2>;

template <typename Config>
int run_pipeline(ck_tile::ArgParser& arg_parser)
{
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;
    return run_gemm_example_with_layouts<Config,
                                         UniversalInvoker,
                                         ck_tile::half_t,
                                         ck_tile::half_t,
                                         ck_tile::half_t>(arg_parser, Row{}, Col{}, Row{});
}

int main(int argc, char* argv[])
{
    auto arg_parser = create_args();
    arg_parser.insert("pipeline", "v1", "TDM pipeline version: v1 or v2");
    const auto parsed = arg_parser.parse(argc, argv);
    if(!parsed)
        return -1;

    try
    {
        if(arg_parser.get_int("split_k") != 1)
            throw std::runtime_error("This analysis fixes split_k (K batch) at 1.");
        if(arg_parser.get_int("persistent") != 0)
            throw std::runtime_error("This analysis fixes persistent mode off.");

        const auto pipeline = arg_parser.get_str("pipeline");
        if(pipeline == "v1")
            return !run_pipeline<TdmV1Config>(arg_parser);
        if(pipeline == "v2")
            return !run_pipeline<TdmV2Config>(arg_parser);

        throw std::runtime_error("Unknown pipeline: " + pipeline);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }
}
