// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gemm_utils.hpp"
#include "run_gemm_example.inc"
#include "universal_gemm_invoker.hpp"

#include "ck_tile/host.hpp"

#include "gtest/gtest.h"

#include <string>
#include <tuple>
#include <vector>

#ifdef CK_USE_GFX1250

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

template <typename PrecType, typename CDataType>
using KPaddingConfig =
    GemmConfigFixedVectorSize<GemmConfigComputeV3_WMMA<PrecType>,
                              1,
                              1,
                              static_cast<ck_tile::index_t>(16 / sizeof(CDataType))>;

template <typename Tuple>
class TestGemmUnalignedK : public ::testing::Test
{
    protected:
    using ADataType   = std::tuple_element_t<0, Tuple>;
    using BDataType   = std::tuple_element_t<1, Tuple>;
    using AccDataType = std::tuple_element_t<2, Tuple>;
    using CDataType   = std::tuple_element_t<3, Tuple>;

    void RunAndVerify(int K, int k_batch = 1)
    {
        constexpr int M = 128;
        constexpr int N = 128;

        const ck_tile::index_t stride_A = K;
        const ck_tile::index_t stride_B = K;
        const ck_tile::index_t stride_C = N;

        // Host tensors
        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_A, ck_tile::bool_constant<true>{}));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(K, N, stride_B, ck_tile::bool_constant<false>{}));
        ck_tile::HostTensor<CDataType> gpu_result(
            ck_tile::host_tensor_descriptor(M, N, stride_C, ck_tile::bool_constant<true>{}));

        ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5, 5, 11939}(a_m_k);
        ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5, 5, 11940}(b_k_n);

        // Device buffers
        ck_tile::DeviceMem a_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_buf(gpu_result.get_element_space_size_in_bytes());
        a_buf.ToDevice(a_m_k.data());
        b_buf.ToDevice(b_k_n.data());
        c_buf.SetZero();

        ck_tile::GemmHostArgs args = {a_buf.GetDeviceBuffer(),
                                      b_buf.GetDeviceBuffer(),
                                      c_buf.GetDeviceBuffer(),
                                      k_batch,
                                      M,
                                      N,
                                      K,
                                      stride_A,
                                      stride_B,
                                      stride_C};

        // Run UniversalInvoker::gemm()
        UniversalInvoker::gemm<KPaddingConfig<ADataType, CDataType>,
                               ADataType,
                               BDataType,
                               ck_tile::tuple<>,
                               AccDataType,
                               CDataType,
                               Row,
                               Col,
                               ck_tile::tuple<>,
                               Row,
                               /*Persistent=*/false,
                               ck_tile::element_wise::PassThrough>(
            args, ck_tile::stream_config{nullptr, false});

        c_buf.FromDevice(gpu_result.data());

        ck_tile::HostTensor<CDataType> host_reference(
            ck_tile::host_tensor_descriptor(M, N, stride_C, ck_tile::bool_constant<true>{}));
        host_reference.SetZero();
        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_k_n, host_reference);

        const float max_accumulated_value =
            *std::max_element(host_reference.mData.begin(), host_reference.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, k_batch, max_accumulated_value);

        // Compare both results
        EXPECT_TRUE(do_verify(gpu_result, host_reference, rtol_atol, "GPU"))
            << "K=" << K << ", k_batch=" << k_batch;
    }
};

using UnalignedKDataTypes =
    ::testing::Types<std::tuple<ck_tile::half_t, ck_tile::half_t, float, ck_tile::half_t>,
                     std::tuple<ck_tile::bf16_t, ck_tile::bf16_t, float, ck_tile::bf16_t>,
                     std::tuple<ck_tile::fp8_t, ck_tile::fp8_t, float, ck_tile::half_t>,
                     std::tuple<ck_tile::bf8_t, ck_tile::bf8_t, float, ck_tile::half_t>,
                     std::tuple<ck_tile::int8_t, ck_tile::int8_t, int32_t, int32_t>>;
TYPED_TEST_SUITE(TestGemmUnalignedK, UnalignedKDataTypes);

TYPED_TEST(TestGemmUnalignedK, SmallKAndTail)
{
    // Cover a K smaller than the normal vector width and a full tile plus a tail.
    constexpr int KTile = GemmConfigComputeV3_WMMA<typename TestFixture::ADataType>::K_Tile;
    for(int K : {5, KTile + 5})
    {
        this->RunAndVerify(K);
    }
}

TYPED_TEST(TestGemmUnalignedK, SplitK)
{
    // The first split is a full warp tile; the second contains the K tail.
    constexpr int K = GemmConfigComputeV3_WMMA<typename TestFixture::ADataType>::K_Warp_Tile + 5;
    this->RunAndVerify(K, /*k_batch=*/2);
}

// The typed tests above hand UniversalInvoker::gemm an already-resolved
// KPaddingConfig and so bypass the dispatcher. This drives the exact shared entry
// point tile_example_gemm_universal uses (run_gemm_example_with_layouts +
// UniversalInvoker + the stock GemmConfigComputeV3_WMMA), so the runtime
// k_pad_fallback_eligible check, the K-alignment decision, and the normal-vs-
// fallback config selection are all exercised end to end.
namespace {
template <typename PrecType>
int run_through_example_dispatch(int M, int N, int K, int split_k)
{
    auto arg_parser              = create_args();
    const std::string prec       = std::is_same_v<PrecType, ck_tile::bf16_t> ? "bf16" : "fp16";
    std::vector<std::string> tok = {"test",
                                    "-m=" + std::to_string(M),
                                    "-n=" + std::to_string(N),
                                    "-k=" + std::to_string(K),
                                    "-split_k=" + std::to_string(split_k),
                                    "-prec=" + prec,
                                    "-v=1",
                                    "-warmup=0",
                                    "-repeat=1",
                                    "-flush_cache=false"};
    std::vector<char*> argv;
    for(auto& t : tok)
        argv.push_back(t.data());
    arg_parser.parse(static_cast<int>(argv.size()), argv.data());

    // Returns pass (1) / fail (0); throws if IsSupportedArgument rejects the shape.
    return run_gemm_example_with_layouts<GemmConfigComputeV3_WMMA<PrecType>,
                                         UniversalInvoker,
                                         PrecType,
                                         PrecType,
                                         PrecType,
                                         Row,
                                         Col,
                                         Row>(arg_parser);
}
} // namespace

TEST(TestGemmUnalignedKDispatch, ThroughExampleEntryPoint)
{
    int rc = 0;
    // Unaligned K: the dispatcher must select the K-pad fallback (the stock
    // config would throw from IsSupportedArgument) and still verify.
    ASSERT_NO_THROW(rc = run_through_example_dispatch<ck_tile::bf16_t>(128, 128, 37, 1));
    EXPECT_EQ(rc, 1);
    ASSERT_NO_THROW(rc = run_through_example_dispatch<ck_tile::half_t>(128, 128, 37, 1));
    EXPECT_EQ(rc, 1);
    // Aligned K on the same path: the fallback must NOT fire; the stock config runs.
    ASSERT_NO_THROW(rc = run_through_example_dispatch<ck_tile::bf16_t>(128, 128, 64, 1));
    EXPECT_EQ(rc, 1);
}

#endif // CK_USE_GFX1250
