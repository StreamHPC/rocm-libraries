// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Analysis-only FP16 RCR GEMM driver for controlled and supplied-preset TDM comparisons.

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

using ControlledV1 = TdmAnalysisConfig<ck_tile::GemmPipeline::COMPUTE_TDM_V1, 2, 2>;
using ControlledV2 = TdmAnalysisConfig<ck_tile::GemmPipeline::COMPUTE_TDM_V2, 2, 2>;
using PresetV1     = TdmAnalysisConfig<ck_tile::GemmPipeline::COMPUTE_TDM_V1, 2, 4>;
using PresetV2     = ControlledV2;

template <typename Config>
struct AnalysisPipeline
{
    using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
    using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<Config::M_Tile, Config::N_Tile, Config::K_Tile>,
        ck_tile::sequence<Config::M_Warp, Config::N_Warp, Config::K_Warp>,
        ck_tile::sequence<Config::M_Warp_Tile, Config::N_Warp_Tile, Config::K_Warp_Tile>,
        Config::PermuteA,
        Config::PermuteB>;

    using Traits = ck_tile::TileGemmUniversalTraits<Config::kPadM,
                                                    Config::kPadN,
                                                    Config::kPadK,
                                                    Config::DoubleSmemBuffer,
                                                    ALayout,
                                                    BLayout,
                                                    CLayout,
                                                    Config::TransposeC,
                                                    Config::UseStructuredSparsity,
                                                    false,
                                                    Config::NumWaveGroups,
                                                    Config::Preshuffle,
                                                    16,
                                                    Config::DataCachePrefetchA,
                                                    Config::DataCachePrefetchB,
                                                    Config::Async>;

    using Problem = ck_tile::UniversalGemmPipelineProblem<ck_tile::half_t,
                                                          ck_tile::half_t,
                                                          float,
                                                          GemmShape,
                                                          Traits,
                                                          Config::Scheduler,
                                                          ck_tile::element_wise::PassThrough,
                                                          ck_tile::element_wise::PassThrough,
                                                          ck_tile::half_t,
                                                          ck_tile::half_t>;

    using Pipeline = typename PipelineTypeTraits<Config::Pipeline>::template GemmPipeline<Problem>;
};

template <typename Config>
void print_configuration(const std::string& variant)
{
    using Pipeline = typename AnalysisPipeline<Config>::Pipeline;
    // gfx1250 executes wave32. The fixed RCR problem selects one K sub-tile. Its two
    // double-buffered, padded 128x64 FP16 LDS allocations occupy 69,568 bytes. These
    // values are checked against code-object metadata by the offline workflow.
    constexpr auto wave_size          = 32;
    constexpr auto wave_count         = Config::M_Warp * Config::N_Warp * Config::K_Warp;
    constexpr auto sub_tile_num       = 1;
    constexpr auto pipeline_lds_bytes = 69568;

    std::cout << "analysis.variant=" << variant << '\n'
              << "analysis.datatypes=fp16,fp16,fp32_acc,fp16_out\n"
              << "analysis.layouts=RCR\n"
              << "analysis.tile=" << Config::M_Tile << 'x' << Config::N_Tile << 'x'
              << Config::K_Tile << '\n'
              << "analysis.warps=" << Config::M_Warp << 'x' << Config::N_Warp << 'x'
              << Config::K_Warp << '\n'
              << "analysis.block_size=" << wave_count * wave_size << '\n'
              << "analysis.sub_tile_num=" << sub_tile_num << '\n'
              << "analysis.pipeline_lds_bytes=" << pipeline_lds_bytes << '\n'
              << "analysis.pipeline=" << Pipeline::GetName() << '\n'
              << "analysis.cluster=1x1\n"
              << "analysis.data_cache_prefetch=none\n"
              << "analysis.k_batch=1" << std::endl;
}

template <typename Config>
int run_variant(ck_tile::ArgParser& arg_parser, const std::string& variant)
{
    print_configuration<Config>(variant);
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
    arg_parser.insert(
        "variant", "controlled-v1", "controlled-v1, controlled-v2, preset-v1, or preset-v2");
    const auto parsed = arg_parser.parse(argc, argv);
    if(!parsed)
        return -1;

    try
    {
        if(arg_parser.get_int("split_k") != 1)
            throw std::runtime_error("This analysis fixes split_k (K batch) at 1.");
        if(arg_parser.get_int("persistent") != 0)
            throw std::runtime_error("This analysis fixes persistent mode off.");

        const auto variant = arg_parser.get_str("variant");
        if(variant == "controlled-v1")
            return !run_variant<ControlledV1>(arg_parser, variant);
        if(variant == "controlled-v2")
            return !run_variant<ControlledV2>(arg_parser, variant);
        if(variant == "preset-v1")
            return !run_variant<PresetV1>(arg_parser, variant);
        if(variant == "preset-v2")
            return !run_variant<PresetV2>(arg_parser, variant);

        throw std::runtime_error("Unknown variant: " + variant);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }
}
