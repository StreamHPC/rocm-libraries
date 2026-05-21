// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <sstream>
#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_mx/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

template <typename Layout>
static constexpr auto is_row_major(Layout)
{
    return ck_tile::bool_constant<
        std::is_same_v<ck_tile::remove_cvref_t<Layout>, ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename Tuple>
class TestCkTileGroupedGemm : public ::testing::Test
{
    protected:
    using ALayout     = std::tuple_element_t<0, Tuple>;
    using BLayout     = std::tuple_element_t<1, Tuple>;
    using CLayout     = std::tuple_element_t<2, Tuple>;
    using ADataType   = std::tuple_element_t<3, Tuple>;
    using BDataType   = std::tuple_element_t<4, Tuple>;
    using AccDataType = std::tuple_element_t<5, Tuple>;
    using CDataType   = std::tuple_element_t<6, Tuple>;
    using DsLayout    = ck_tile::tuple<>;
    using DsDataType  = ck_tile::tuple<>;
    using ScaleType   = ck_tile::e8m0_t;
    using ScaleM      = ck_tile::MXScalePointer<ScaleType, 1, 32>;
    using ScaleN      = ck_tile::MXScalePointer<ScaleType, 1, 32>;

    // Pack [MN, K/32] e8m0_t scales into [MN/MNPack, K/32/KPack] int32_t
    // Each int32_t contains MNPack * KPack e8m0_t values with byte layout matching
    // the GPU tile distribution: values are XdlMNThread apart in M and XdlKThread apart in K.
    //   byte[ik * MNPack + imn] = e8m0 at strided (mn, k) position
    // kLast=true for A scales (layout [M, K/32]), kLast=false for B scales (layout [K/32, N])
    template <ck_tile::index_t MNPack      = 2,
              ck_tile::index_t KPack       = 2,
              ck_tile::index_t XdlMNThread = 16,
              ck_tile::index_t XdlKThread  = 4>
    static auto packScalesMNxK(const ck_tile::HostTensor<ck_tile::e8m0_t>& src, bool kLast)
    {
        auto src_lengths                    = src.get_lengths();
        const ck_tile::index_t MN           = kLast ? src_lengths[0] : src_lengths[1];
        const ck_tile::index_t K_scale      = kLast ? src_lengths[1] : src_lengths[0];
        const ck_tile::index_t MN_packed    = MN / MNPack;
        const ck_tile::index_t K_packed     = K_scale / KPack;
        const ck_tile::index_t total_packed = MN_packed * K_packed;

        std::vector<int32_t> packed(total_packed);

        for(ck_tile::index_t packed_mn = 0; packed_mn < MN_packed; packed_mn++)
        {
            for(ck_tile::index_t packed_k = 0; packed_k < K_packed; packed_k++)
            {
                int32_t val               = 0;
                ck_tile::index_t mn_lane  = packed_mn % XdlMNThread;
                ck_tile::index_t mn_group = packed_mn / XdlMNThread;
                ck_tile::index_t k_lane   = packed_k % XdlKThread;
                ck_tile::index_t k_group  = packed_k / XdlKThread;
                for(ck_tile::index_t ik = 0; ik < KPack; ik++)
                {
                    for(ck_tile::index_t imn = 0; imn < MNPack; imn++)
                    {
                        ck_tile::index_t byteIdx = ik * MNPack + imn;
                        ck_tile::index_t orig_mn =
                            mn_group * XdlMNThread * MNPack + imn * XdlMNThread + mn_lane;
                        ck_tile::index_t orig_k =
                            k_group * XdlKThread * KPack + ik * XdlKThread + k_lane;

                        ck_tile::e8m0_t v = kLast ? src(orig_mn, orig_k) : src(orig_k, orig_mn);
                        val |= (static_cast<int32_t>(v.get()) << (byteIdx * 8));
                    }
                }
                packed[packed_mn * K_packed + packed_k] = val;
            }
        }
        return packed;
    }

    // Get the persistent value from ck_tile::bool_constant
    using PersistentType             = std::tuple_element_t<7, Tuple>;
    static constexpr bool Persistent = PersistentType::value;

    struct MxGemmConfig
    {
        static constexpr ck_tile::index_t M_Tile = 128;
        static constexpr ck_tile::index_t N_Tile = 128;
        static constexpr ck_tile::index_t K_Tile = 512;

        static constexpr ck_tile::index_t M_Warp = 1;
        static constexpr ck_tile::index_t N_Warp = 4;
        static constexpr ck_tile::index_t K_Warp = 1;

        static constexpr ck_tile::index_t M_Warp_Tile = 16;
        static constexpr ck_tile::index_t N_Warp_Tile = 16;
        static constexpr ck_tile::index_t K_Warp_Tile = 128;

        static constexpr bool kPadM = false;
        static constexpr bool kPadN = false;
        static constexpr bool kPadK = false;

        static constexpr bool TransposeC            = false;
        static constexpr bool UseStructuredSparsity = false;

        static constexpr int kBlockPerCu                = 1;
        static constexpr int TileParitionerGroupNum     = 8;
        static constexpr int TileParitionerM01          = 4;
        static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Intrawave;
        static constexpr ck_tile::index_t NumWaveGroups = 1;
        static constexpr bool DoubleSmemBuffer          = false;
        static constexpr bool Preshuffle                = false;

        static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
        static constexpr bool TiledMMAPermuteN = false;
    };

    // struct MXfp4_GemmConfig16 : MxGemmConfig
    // {
    //     static constexpr ck_tile::index_t M_Tile = 64;
    //     static constexpr ck_tile::index_t N_Tile = 64;
    //     static constexpr ck_tile::index_t K_Tile = 256;
    // };

    struct MXfp8_GemmConfig16 : MxGemmConfig
    {
        static constexpr ck_tile::index_t M_Tile = 64;
        static constexpr ck_tile::index_t N_Tile = 64;
        static constexpr ck_tile::index_t K_Tile = 256;
    };

    using grouped_gemm_kargs = ck_tile::MXGemmKernelArgs<>;
    std::size_t get_workspace_size(const std::vector<grouped_gemm_kargs>& gemm_descs)
    {
        return gemm_descs.size() * sizeof(ck_tile::MxGemmTransKernelArg<>);
    }

    template <typename GroupedGemKernelParam, typename ALayout, typename BLayout, typename CLayout>
    void invoke_grouped_gemm(const std::vector<grouped_gemm_kargs>& gemm_descs,
                             const ck_tile::stream_config& s,
                             void* kargs_ptr)
    {
        constexpr bool DoubleSmemBuffer = false;
        constexpr bool TransposeC       = false;

        constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        constexpr ck_tile::index_t TileParitionerM01      = 4;

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<GroupedGemKernelParam::M_Tile,
                                                     GroupedGemKernelParam::N_Tile,
                                                     GroupedGemKernelParam::K_Tile>,
                                   ck_tile::sequence<GroupedGemKernelParam::M_Warp,
                                                     GroupedGemKernelParam::N_Warp,
                                                     GroupedGemKernelParam::K_Warp>,
                                   ck_tile::sequence<GroupedGemKernelParam::M_Warp_Tile,
                                                     GroupedGemKernelParam::N_Warp_Tile,
                                                     GroupedGemKernelParam::K_Warp_Tile>>;
        using TilePartitioner = ck_tile::
            GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<GroupedGemKernelParam::kPadM,
                                                                     GroupedGemKernelParam::kPadN,
                                                                     GroupedGemKernelParam::kPadK,
                                                                     DoubleSmemBuffer,
                                                                     ALayout,
                                                                     BLayout,
                                                                     CLayout,
                                                                     TransposeC>;

        constexpr auto scheduler = ck_tile::GemmPipelineScheduler::Intrawave;
        using MXPipelineProblem  = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                         BDataType,
                                                                         AccDataType,
                                                                         GemmShape,
                                                                         GemmUniversalTraits,
                                                                         scheduler>;

        using MXGemmPipeline = ck_tile::MXGemmPipelineAgBgCrCompAsync<MXPipelineProblem>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             CLayout,
                                             ck_tile::element_wise::PassThrough,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             GroupedGemKernelParam::M_Warp,
                                             GroupedGemKernelParam::N_Warp,
                                             GroupedGemKernelParam::M_Warp_Tile,
                                             GroupedGemKernelParam::N_Warp_Tile,
                                             GroupedGemKernelParam::K_Warp_Tile,
                                             MXPipelineProblem::TransposeC>>;
        using Kernel = ck_tile::GroupedMXGemmKernel<TilePartitioner, MXGemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKargs(gemm_descs);
        EXPECT_TRUE(Kernel::IsSupportedArgument(kargs));

        // Use the filtered kargs (zero-dim groups are excluded by MakeKargs) to derive
        // the correct grid size and group count — not the raw gemm_descs vector.
        const dim3 blocks = Kernel::BlockSize();
        if(kargs.empty())
            return;

        const dim3 grids = dim3(kargs.back().block_end, 1, 1);

        ck_tile::hip_check_error(
            hipMemcpyWithStream(kargs_ptr,
                                kargs.data(),
                                kargs.size() * sizeof(ck_tile::MxGemmTransKernelArg<>),
                                hipMemcpyHostToDevice,
                                s.stream_id_));

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel: " << Kernel::GetName() << " with args:" << " grid: {"
                      << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {"
                      << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;
        }

        ck_tile::ignore =
            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<GroupedGemKernelParam::kBlockPerCu>(
                                       Kernel{},
                                       grids,
                                       blocks,
                                       0,
                                       ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                                       kargs.size()));
    }

    template <typename GroupedGemKernelParam, typename ALayout, typename BLayout, typename CLayout>
    void invoke_grouped_gemm_persistent(const ck_tile::stream_config& s,
                                        const ck_tile::index_t num_groups,
                                        void* kargs_ptr)
    {
        constexpr bool TransposeC       = false;
        constexpr bool DoubleSmemBuffer = false;

        constexpr int kBlockPerCu                         = 1;
        constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        constexpr ck_tile::index_t TileParitionerM01      = 4;

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<GroupedGemKernelParam::M_Tile,
                                                     GroupedGemKernelParam::N_Tile,
                                                     GroupedGemKernelParam::K_Tile>,
                                   ck_tile::sequence<GroupedGemKernelParam::M_Warp,
                                                     GroupedGemKernelParam::N_Warp,
                                                     GroupedGemKernelParam::K_Warp>,
                                   ck_tile::sequence<GroupedGemKernelParam::M_Warp_Tile,
                                                     GroupedGemKernelParam::N_Warp_Tile,
                                                     GroupedGemKernelParam::K_Warp_Tile>>;
        using TilePartitioner = ck_tile::
            GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

        using MXGemmTraits =
            ck_tile::PersistentTileGemmUniversalTraits<GroupedGemKernelParam::kPadM,
                                                       GroupedGemKernelParam::kPadN,
                                                       GroupedGemKernelParam::kPadK,
                                                       DoubleSmemBuffer,
                                                       ALayout,
                                                       BLayout,
                                                       CLayout,
                                                       TransposeC>;

        constexpr auto scheduler = ck_tile::GemmPipelineScheduler::Intrawave;

        // We create the GEMM pipeline without specifying hotloop or tailnumber.
        // These are automatically run inside the kernel based on the given input data.
        using MXPipelineProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                        BDataType,
                                                                        AccDataType,
                                                                        GemmShape,
                                                                        MXGemmTraits,
                                                                        scheduler>;

        using GemmPipeline = ck_tile::MXGemmPipelineAgBgCrCompAsync<MXPipelineProblem>;
        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             CLayout,
                                             ck_tile::element_wise::PassThrough,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             GroupedGemKernelParam::M_Warp,
                                             GroupedGemKernelParam::N_Warp,
                                             GroupedGemKernelParam::M_Warp_Tile,
                                             GroupedGemKernelParam::N_Warp_Tile,
                                             GroupedGemKernelParam::K_Warp_Tile,
                                             MXPipelineProblem::TransposeC>>;
        using Kernel = ck_tile::GroupedMXGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        const dim3 blocks = Kernel::BlockSize();
        const dim3 grids  = Kernel::MaxOccupancyGridSize(s);

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel: " << Kernel::GetName() << " with args:" << " grid: {"
                      << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {"
                      << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;
        }

        ck_tile::ignore =
            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<kBlockPerCu>(
                                       Kernel{},
                                       grids,
                                       blocks,
                                       0,
                                       ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                                       num_groups));
    }

    auto calculate_rtol_atol(const ck_tile::index_t K,
                             const ck_tile::index_t kbatch,
                             const float max_accumulated_value)
    {
        using ComputeType =
            std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
        // Calculate thresholds
        const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
            ck_tile::integer_divide_ceil(K, kbatch));
        const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
            max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
        // Calculate error due to split_k accumulation
        const auto rtol_split_k =
            ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
        const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
            max_accumulated_value, kbatch);
        // Use higher threshold
        return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
    }

    public:
    void Run(const std::vector<int>& Ms,
             const std::vector<int>& Ns,
             const std::vector<int>& Ks,
             std::vector<int>& stride_As,
             std::vector<int>& stride_Bs,
             std::vector<int>& stride_Cs,
             const int kbatch      = 1,
             const int group_count = 16,
             int seed              = 1234)
    {
        // using namespace ck_tile::literals;

        std::vector<ck_tile::HostTensor<ADataType>> a_m_k_tensors;
        std::vector<ck_tile::HostTensor<BDataType>> b_k_n_tensors;
        std::vector<ck_tile::HostTensor<CDataType>> c_m_n_tensors;

        std::vector<ck_tile::HostTensor<ScaleType>> scale_a_host_tensors;
        std::vector<ck_tile::HostTensor<ScaleType>> scale_b_host_tensors;

        a_m_k_tensors.reserve(group_count);
        b_k_n_tensors.reserve(group_count);
        c_m_n_tensors.reserve(group_count);

        scale_a_host_tensors.reserve(group_count);
        scale_b_host_tensors.reserve(group_count);

        std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_m_k_dev_buf;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> b_k_n_dev_buf;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> c_m_n_dev_buf;

        std::vector<std::unique_ptr<ck_tile::DeviceMem>> scale_a_dev_buf;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> scale_b_dev_buf;

        a_m_k_dev_buf.reserve(group_count);
        b_k_n_dev_buf.reserve(group_count);
        c_m_n_dev_buf.reserve(group_count);

        scale_a_dev_buf.reserve(group_count);
        scale_b_dev_buf.reserve(group_count);

        std::vector<grouped_gemm_kargs> gemm_descs;
        gemm_descs.reserve(group_count);

        for(int i = 0; i < group_count; ++i)
        {
            const ck_tile::index_t M = Ms[i];
            const ck_tile::index_t N = Ns[i];
            const ck_tile::index_t K = Ks[i];

            stride_As[i] = ck_tile::get_default_stride(M, K, stride_As[i], is_row_major(ALayout{}));
            stride_Bs[i] = ck_tile::get_default_stride(K, N, stride_Bs[i], is_row_major(BLayout{}));
            stride_Cs[i] = ck_tile::get_default_stride(M, N, stride_Cs[i], is_row_major(CLayout{}));

            a_m_k_tensors.push_back(ck_tile::HostTensor<ADataType>(
                ck_tile::host_tensor_descriptor(M, K, stride_As[i], is_row_major(ALayout{}))));
            b_k_n_tensors.push_back(ck_tile::HostTensor<BDataType>(
                ck_tile::host_tensor_descriptor(K, N, stride_Bs[i], is_row_major(BLayout{}))));
            c_m_n_tensors.push_back(ck_tile::HostTensor<CDataType>(
                ck_tile::host_tensor_descriptor(M, N, stride_Cs[i], is_row_major(CLayout{}))));

            const ck_tile::index_t scale_k_size = K / 32;
            const ck_tile::index_t stride_scale_a =
                ck_tile::get_default_stride(M, scale_k_size, 0, is_row_major(ALayout{}));
            const ck_tile::index_t stride_scale_b =
                ck_tile::get_default_stride(scale_k_size, N, 0, is_row_major(BLayout{}));

            scale_a_host_tensors.push_back(
                ck_tile::HostTensor<ScaleType>(ck_tile::host_tensor_descriptor(
                    M, scale_k_size, stride_scale_a, is_row_major(ALayout{}))));
            scale_b_host_tensors.push_back(
                ck_tile::HostTensor<ScaleType>(ck_tile::host_tensor_descriptor(
                    scale_k_size, N, stride_scale_b, is_row_major(BLayout{}))));

            // std::cout << "gemm[" << i << "]" << " a_m_k: " << a_m_k_tensors[i].mDesc
            //           << " b_k_n: " << b_k_n_tensors[i].mDesc
            //           << " c_m_n: " << c_m_n_tensors[i].mDesc << " KBatch: " << kbatch <<
            //           std::endl;

            ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k_tensors[i]);
            ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n_tensors[i]);
            ck_tile::FillUniformDistribution<ScaleType>{0.001f, 10.f, seed++}(
                scale_a_host_tensors[i]);
            ck_tile::FillUniformDistribution<ScaleType>{0.001f, 10.f, seed++}(
                scale_b_host_tensors[i]);

            using GemmConfig = MXfp8_GemmConfig16;

            // Compute effective XdlPack sizes based on GemmConfig tile dimensions
            constexpr ck_tile::index_t MPerXdl = GemmConfig::M_Warp_Tile;
            constexpr ck_tile::index_t NPerXdl = GemmConfig::N_Warp_Tile;
            constexpr ck_tile::index_t KPerXdl = GemmConfig::K_Warp_Tile;
            constexpr ck_tile::index_t MIterPerWarp =
                GemmConfig::M_Tile / (GemmConfig::M_Warp * MPerXdl);
            constexpr ck_tile::index_t NIterPerWarp =
                GemmConfig::N_Tile / (GemmConfig::N_Warp * NPerXdl);
            constexpr ck_tile::index_t KIterPerWarp = GemmConfig::K_Tile / KPerXdl;

            constexpr ck_tile::index_t MXdlPackEff =
                (MIterPerWarp >= 2 && MIterPerWarp % 2 == 0) ? 2 : 1;
            constexpr ck_tile::index_t NXdlPackEff =
                (NIterPerWarp >= 2 && NIterPerWarp % 2 == 0) ? 2 : 1;
            constexpr ck_tile::index_t KXdlPackEff =
                (KIterPerWarp >= 2 && KIterPerWarp % 2 == 0) ? 2 : 1;

            constexpr ck_tile::index_t XdlMNThread = GemmConfig::M_Warp_Tile;
            constexpr ck_tile::index_t XdlKThread  = 64 / XdlMNThread;

            // Pack scales into int32_t for GPU consumption
            auto scale_a_packed = packScalesMNxK<MXdlPackEff, KXdlPackEff, XdlMNThread, XdlKThread>(
                scale_a_host_tensors[i], true);
            auto scale_b_packed = packScalesMNxK<NXdlPackEff, KXdlPackEff, XdlMNThread, XdlKThread>(
                scale_b_host_tensors[i], false);

            a_m_k_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                a_m_k_tensors[i].get_element_space_size_in_bytes()));
            b_k_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                b_k_n_tensors[i].get_element_space_size_in_bytes()));
            c_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                c_m_n_tensors[i].get_element_space_size_in_bytes()));
            scale_a_dev_buf.push_back(
                std::make_unique<ck_tile::DeviceMem>(scale_a_packed.size() * sizeof(int32_t)));
            scale_b_dev_buf.push_back(
                std::make_unique<ck_tile::DeviceMem>(scale_b_packed.size() * sizeof(int32_t)));

            a_m_k_dev_buf[i]->ToDevice(a_m_k_tensors[i].data());
            b_k_n_dev_buf[i]->ToDevice(b_k_n_tensors[i].data());
            scale_a_dev_buf[i]->ToDevice(scale_a_packed.data());
            scale_b_dev_buf[i]->ToDevice(scale_b_packed.data());
            c_m_n_dev_buf[i]->SetZero();
            c_m_n_tensors[i].SetZero();

            const void* p_a = a_m_k_dev_buf[i]->GetDeviceBuffer();
            const void* p_b = b_k_n_dev_buf[i]->GetDeviceBuffer();
            void* p_c       = c_m_n_dev_buf[i]->GetDeviceBuffer();

            ck_tile::MXScalePointer<ck_tile::e8m0_t, 1, 32> scale_m(
                reinterpret_cast<ScaleType*>(scale_a_dev_buf[i]->GetDeviceBuffer()));
            ck_tile::MXScalePointer<ck_tile::e8m0_t, 1, 32> scale_n(
                reinterpret_cast<ScaleType*>(scale_b_dev_buf[i]->GetDeviceBuffer()));

            grouped_gemm_kargs tmp(std::array<const void*, 1>({p_a}),
                                   std::array<const void*, 1>({p_b}),
                                   std::array<const void*, 0>({/*arg.ds_ptr*/}),
                                   p_c,
                                   kbatch,
                                   M,
                                   N,
                                   K,
                                   std::array<ck_tile::index_t, 1>({stride_As[i]}),
                                   std::array<ck_tile::index_t, 1>({stride_Bs[i]}),
                                   std::array<ck_tile::index_t, 0>({/*stride_Ds*/}),
                                   stride_Cs[i],
                                   scale_m,
                                   scale_n);

            gemm_descs.push_back(std::move(tmp));
        }

        ck_tile::DeviceMem gemm_workspace;
        gemm_workspace.Realloc(get_workspace_size(gemm_descs));

        if constexpr(Persistent)
        {
            // Generate kernel arguments
            std::vector<ck_tile::MXGemmKernelArgs<>> kargs;
            void* kargs_ptr = gemm_workspace.GetDeviceBuffer();
            for(const auto& arg : gemm_descs)
            {
                kargs.emplace_back(ck_tile::MXGemmKernelArgs<>(arg.as_ptr,
                                                               arg.bs_ptr,
                                                               {/*arg.ds_ptr*/},
                                                               arg.e_ptr,
                                                               arg.k_batch,
                                                               arg.M,
                                                               arg.N,
                                                               arg.K,
                                                               arg.stride_As,
                                                               arg.stride_Bs,
                                                               {/*stride_Ds*/},
                                                               arg.stride_E,
                                                               arg.scale_m_ptr,
                                                               arg.scale_n_ptr));
            }
            const auto stream = ck_tile::stream_config{nullptr, false, 1};
            ck_tile::hip_check_error(
                hipMemcpyWithStream(kargs_ptr,
                                    kargs.data(),
                                    kargs.size() * sizeof(ck_tile::MxGemmTransKernelArg<>),
                                    hipMemcpyHostToDevice,
                                    stream.stream_id_));
            invoke_grouped_gemm_persistent<MXfp8_GemmConfig16, ALayout, BLayout, CLayout>(
                stream, group_count, kargs_ptr);
        }
        else
        {
            invoke_grouped_gemm<MXfp8_GemmConfig16, ALayout, BLayout, CLayout>(
                gemm_descs,
                ck_tile::stream_config{nullptr, false, 1},
                gemm_workspace.GetDeviceBuffer());
        }

        // Copy results back to host for validation
        for(int i = 0; i < group_count; i++)
        {
            c_m_n_dev_buf[i]->FromDevice(c_m_n_tensors[i].data());
        }

        bool pass{true};
        for(int i = 0; i < group_count; ++i)
        {
            // Groups with M=0 or N=0 produce no output — skip validation.
            // K=0 groups do produce output (all zeros) and are validated normally.
            if(Ms[i] == 0 || Ns[i] == 0)
                continue;

            ck_tile::HostTensor<CDataType> c_m_n_host_ref(ck_tile::host_tensor_descriptor(
                Ms[i], Ns[i], stride_Cs[i], is_row_major(CLayout{})));
            c_m_n_host_ref.SetZero();
            ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
                a_m_k_tensors[i], b_k_n_tensors[i], c_m_n_host_ref);
            // Use max absolute value (not algebraic max) to calibrate atol.
            // The absolute threshold in calculate_rtol_atol scales with this value,
            // so using the algebraic max (which may be a small positive number when
            // most outputs are negative) would produce a near-zero atol. Near-zero
            // reference elements then have no tolerance headroom for the ~1 ULP
            // error introduced by SplitK atomicAdd accumulation.
            const float max_accumulated_value = std::accumulate(
                c_m_n_host_ref.mData.begin(),
                c_m_n_host_ref.mData.end(),
                0.0f,
                [](float acc, auto v) { return std::max(acc, std::abs(static_cast<float>(v))); });
            const auto rtol_atol = calculate_rtol_atol(Ks[i], kbatch, max_accumulated_value);
            pass &= ck_tile::check_err(c_m_n_tensors[i],
                                       c_m_n_host_ref,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));
        }
        EXPECT_TRUE(pass);
    }
};
