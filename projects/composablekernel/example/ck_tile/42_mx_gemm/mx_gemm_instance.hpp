// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/host.hpp"
#include "mx_gemm.hpp"
#include "ck_tile/ops/gemm_mx/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp"
#include "ck_tile/ops/gemm_mx/kernel/gemm_mx_kernel.hpp"
#include "ck_tile/ops/gemm_mx/pipeline/mx_gemm_wp_pipeline_agmem_bgmem_creg_v1.hpp"

template <typename Layout>
using is_row_major_t = ck_tile::bool_constant<
    std::is_same_v<ck_tile::remove_cvref_t<Layout>, ck_tile::tensor_layout::gemm::RowMajor>>;

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ScaleM,
          typename ScaleN,
          bool persistent,
          bool Splitk>
float mx_gemm_calc(const MXGemmHostArgs<ScaleM, ScaleN>& args, const ck_tile::stream_config& s)
{
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::
            sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>>;

    using MXGemmTraits = ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                                          GemmConfig::kPadN,
                                                          GemmConfig::kPadK,
                                                          GemmConfig::DoubleSmemBuffer,
                                                          ALayout,
                                                          BLayout,
                                                          CLayout,
                                                          GemmConfig::TransposeC,
                                                          GemmConfig::UseStructuredSparsity,
                                                          persistent,
                                                          GemmConfig::NumWaveGroups,
                                                          GemmConfig::Preshuffle>;

    using ComputeDataType = ADataType;
    static_assert(sizeof(ComputeDataType) >= sizeof(BDataType),
                  "mixed_prec_gemm requires ADataType is a wider type than BDataType");

    using MXPipelineProblem =
        ck_tile::MXGemmPipelineProblem<ADataType,
                                       BDataType,
                                       AccDataType,
                                       GemmShape,
                                       MXGemmTraits,
                                       GemmConfig::Scheduler,
                                       ck_tile::element_wise::PassThrough,
                                       ck_tile::element_wise::PassThrough,
                                       true,
                                       ck_tile::TailNumber::Full,
                                       ck_tile::amd_buffer_coherence_enum::coherence_default,
                                       false,
                                       ADataType,
                                       false,
                                       1,
                                       1,
                                       GemmConfig::TiledMMAPermuteN>;

    using MXGemmPipeline =
        std::conditional_t<GemmConfig::Preshuffle,
                           ck_tile::MXGemmPreshufflePipelineAGmemBGmemCRegV1<MXPipelineProblem>,
                           ck_tile::MXGemmPipelineAgBgCrCompAsync<MXPipelineProblem>>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                   GemmConfig::TileParitionerGroupNum,
                                                   GemmConfig::TileParitionerM01>;

    using GemmEpilogueProblem =
        ck_tile::CShuffleEpilogueProblem<ComputeDataType,
                                         ComputeDataType,
                                         ck_tile::tuple<>,
                                         AccDataType,
                                         CDataType,
                                         ck_tile::tuple<>,
                                         CLayout,
                                         ck_tile::element_wise::PassThrough,
                                         TilePartitioner::MPerBlock,
                                         TilePartitioner::NPerBlock,
                                         GemmConfig::M_Warp,
                                         GemmConfig::N_Warp,
                                         GemmConfig::M_Warp_Tile,
                                         GemmConfig::N_Warp_Tile,
                                         GemmConfig::K_Warp_Tile,
                                         MXPipelineProblem::TransposeC,
                                         GemmConfig::NumWaveGroups,
                                         false,
                                         1,
                                         GemmConfig::TiledMMAPermuteN,
                                         ck_tile::MXEpilogueTraits<GemmConfig>::BlockedXDLNPerWarp>;

    using GemmEpilogue = ck_tile::CShuffleEpilogue<GemmEpilogueProblem>;

    using Kernel = ck_tile::MXGemmKernel<TilePartitioner, MXGemmPipeline, GemmEpilogue>;

    auto kargs = Kernel::MakeKernelArgs(std::array<const void*, 1>{args.as_ptr},
                                        std::array<const void*, 1>{args.bs_ptr},
                                        std::array<const void*, 0>{},
                                        args.e_ptr,
                                        args.k_batch,
                                        args.M,
                                        args.N,
                                        args.K,
                                        std::array<ck_tile::index_t, 1>{args.stride_As},
                                        std::array<ck_tile::index_t, 1>{args.stride_Bs},
                                        std::array<ck_tile::index_t, 0>{},
                                        args.stride_E,
                                        args.scale_m,
                                        args.scale_n);

    const auto kernel = ck_tile::make_kernel<Kernel::kBlockPerCu>(
        Kernel{}, Kernel::GridSize(kargs), Kernel::BlockSize(), 0, kargs);

    return ck_tile::launch_kernel(s, kernel);
}
