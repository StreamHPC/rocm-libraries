// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// EXPERIMENT (not for merge): gfx1250 hardware probe for the WMMA -> VALU write-after-read hazard
// on WMMA A/B sources. One wave runs, inside a single asm block (so the compiler cannot add hazard
// padding), optional back-to-back filler WMMAs, the target WMMA, exactly N v_nop, then a v_mov
// overwriting one source dword. The result is compared with a run without the overwrite.
// The compiler assumes 1 wait state is enough after v_wmma_f32_16x16x64_fp8_fp8 (latency 4) and 8
// after v_wmma_i32_16x16x64_iu8 (latency 16), see GCNHazardRecognizer.cpp, VALUWaitStates.
#include <hip/hip_runtime.h>

#include "ck_tile/core/config.hpp" // __gfx125__
#include "gtest/gtest.h"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using u32x8 = uint32_t __attribute__((ext_vector_type(8)));

constexpr int kLanes = 32;

// Which source dword the v_mov overwrites
enum Which : int
{
    kNone        = 0, // reference
    kAFirst      = 1, // v16
    kBFirst      = 2, // v24
    kBBeforeWmma = 3, // sanity: overwrite v24 before the WMMA, must change the result
    kALast       = 4, // v23
    kBLast       = 5, // v31
};

template <int Op, int Fillers, int Nops, int W>
__global__ void __launch_bounds__(kLanes)
    war_probe(const uint32_t* a, const uint32_t* b, uint32_t* c)
{
#if defined(__gfx125__)
    const uint32_t* pa = a + threadIdx.x * 8;
    const uint32_t* pb = b + threadIdx.x * 8;
    u32x8 d;
    asm volatile(
        // sources in fixed registers: A v[16:23], B v[24:31]
        "v_mov_b32 v16, %1\n\tv_mov_b32 v17, %2\n\tv_mov_b32 v18, %3\n\tv_mov_b32 v19, %4\n\t"
        "v_mov_b32 v20, %5\n\tv_mov_b32 v21, %6\n\tv_mov_b32 v22, %7\n\tv_mov_b32 v23, %8\n\t"
        "v_mov_b32 v24, %9\n\tv_mov_b32 v25, %10\n\tv_mov_b32 v26, %11\n\tv_mov_b32 v27, %12\n\t"
        "v_mov_b32 v28, %13\n\tv_mov_b32 v29, %14\n\tv_mov_b32 v30, %15\n\tv_mov_b32 v31, %16\n\t"
        ".rept 16\n\tv_nop\n\t.endr\n\t"
        ".if %c19 == 3\n\tv_mov_b32 v24, 0x38383838\n\t.rept 16\n\tv_nop\n\t.endr\n\t.endif\n\t"
        ".macro WMMA_PROBE dst\n\t"
        ".if %c17 == 0\n\tv_wmma_f32_16x16x64_fp8_fp8 \\dst, v[16:23], v[24:31], 0\n\t"
        ".else\n\tv_wmma_i32_16x16x64_iu8 \\dst, v[16:23], v[24:31], 0\n\t.endif\n\t"
        ".endm\n\t"
        ".if %c18 >= 1\n\tWMMA_PROBE v[32:39]\n\t.endif\n\t"
        ".if %c18 >= 2\n\tWMMA_PROBE v[40:47]\n\t.endif\n\t"
        ".if %c18 >= 3\n\tWMMA_PROBE v[48:55]\n\t.endif\n\t"
        ".if %c18 >= 4\n\tWMMA_PROBE v[56:63]\n\t.endif\n\t"
        "WMMA_PROBE %0\n\t"
        ".rept %c20\n\tv_nop\n\t.endr\n\t"
        ".if %c19 == 1\n\tv_mov_b32 v16, 0x38383838\n\t.endif\n\t"
        ".if %c19 == 2\n\tv_mov_b32 v24, 0x38383838\n\t.endif\n\t"
        ".if %c19 == 4\n\tv_mov_b32 v23, 0x38383838\n\t.endif\n\t"
        ".if %c19 == 5\n\tv_mov_b32 v31, 0x38383838\n\t.endif\n\t"
        // drain: nothing below reads the WMMA results before they are complete
        ".rept 64\n\tv_nop\n\t.endr\n\t"
        ".purgem WMMA_PROBE"
        : "=&v"(d)
        : "v"(pa[0]),
          "v"(pa[1]),
          "v"(pa[2]),
          "v"(pa[3]),
          "v"(pa[4]),
          "v"(pa[5]),
          "v"(pa[6]),
          "v"(pa[7]),
          "v"(pb[0]),
          "v"(pb[1]),
          "v"(pb[2]),
          "v"(pb[3]),
          "v"(pb[4]),
          "v"(pb[5]),
          "v"(pb[6]),
          "v"(pb[7]),
          "i"(Op),
          "i"(Fillers),
          "i"(W),
          "i"(Nops)
        : "v16",
          "v17",
          "v18",
          "v19",
          "v20",
          "v21",
          "v22",
          "v23",
          "v24",
          "v25",
          "v26",
          "v27",
          "v28",
          "v29",
          "v30",
          "v31",
          "v32",
          "v33",
          "v34",
          "v35",
          "v36",
          "v37",
          "v38",
          "v39",
          "v40",
          "v41",
          "v42",
          "v43",
          "v44",
          "v45",
          "v46",
          "v47",
          "v48",
          "v49",
          "v50",
          "v51",
          "v52",
          "v53",
          "v54",
          "v55",
          "v56",
          "v57",
          "v58",
          "v59",
          "v60",
          "v61",
          "v62",
          "v63");
    for(int i = 0; i < 8; ++i)
        c[threadIdx.x * 8 + i] = d[i];
#else
    static_cast<void>(a);
    static_cast<void>(b);
    static_cast<void>(c);
#endif
}

#define HIP_OK(expr) ASSERT_EQ((expr), hipSuccess) << #expr

constexpr int kNops[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16};

// One output per entry of kNops
template <int Op, int Fillers, int W, int... I>
std::vector<std::vector<uint32_t>>
run_row(const uint32_t* da, const uint32_t* db, uint32_t* dc, std::integer_sequence<int, I...>)
{
    std::vector<std::vector<uint32_t>> rows;
    (
        [&] {
            std::vector<uint32_t> out(kLanes * 8, 0xdeadbeef);
            war_probe<Op, Fillers, kNops[I], W><<<1, kLanes>>>(da, db, dc);
            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
            EXPECT_EQ(hipMemcpy(out.data(), dc, out.size() * 4, hipMemcpyDeviceToHost), hipSuccess);
            rows.push_back(out);
        }(),
        ...);
    return rows;
}

int mismatches(const std::vector<uint32_t>& x, const std::vector<uint32_t>& y)
{
    int n = 0;
    for(size_t i = 0; i < x.size(); ++i)
        n += x[i] != y[i];
    return n;
}

template <int Op, int Fillers>
void run_case(const uint32_t* da, const uint32_t* db, uint32_t* dc)
{
    using Seq         = std::make_integer_sequence<int, sizeof(kNops) / sizeof(kNops[0])>;
    const auto ref    = run_row<Op, Fillers, kNone>(da, db, dc, Seq{});
    const auto sanity = run_row<Op, Fillers, kBBeforeWmma>(da, db, dc, Seq{});

    const char* op_name = Op == 0 ? "fp8 16x16x64" : "iu8 16x16x64";
    // the reference must not depend on N, and the early overwrite must change the result
    for(size_t i = 0; i < ref.size(); ++i)
        EXPECT_EQ(mismatches(ref[i], ref[0]), 0) << op_name << " reference unstable";
    const int sanity_diff = mismatches(sanity[0], ref[0]);
    EXPECT_GT(sanity_diff, 0) << op_name << ": overwrite does not affect the result";

    // wait states the compiler requires before a VALU write of a WMMA source
    const int compiler_nops = Op == 0 ? 1 : 8;
    std::printf("\n%s, %d filler WMMA(s); wrong output dwords (of %d) vs v_nop count N "
                "(compiler pads %d; overwrite before the WMMA changes %d)\n",
                op_name,
                Fillers,
                kLanes * 8,
                compiler_nops,
                sanity_diff);
    std::printf("  %-8s", "N:");
    for(int n : kNops)
        std::printf("%5d", n);
    std::printf("\n");

    const auto print_row = [&](const char* name, auto w) {
        const auto rows = run_row<Op, Fillers, decltype(w)::value>(da, db, dc, Seq{});
        std::printf("  %-8s", name);
        for(size_t i = 0; i < rows.size(); ++i)
        {
            const int m = mismatches(rows[i], ref[0]);
            std::printf("%5d", m);
            if(kNops[i] >= compiler_nops)
                EXPECT_EQ(m, 0) << op_name << ", " << Fillers << " fillers, " << name
                                << ": wrong result with N = " << kNops[i]
                                << " >= the compiler's padding";
        }
        std::printf("\n");
    };
    print_row("A first", std::integral_constant<int, kAFirst>{});
    print_row("A last", std::integral_constant<int, kALast>{});
    print_row("B first", std::integral_constant<int, kBFirst>{});
    print_row("B last", std::integral_constant<int, kBLast>{});
}

} // namespace

TEST(WmmaSourceWarProbe, Gfx1250)
{
    // fp8 e4m3 values with exponent bits 0b0110..0b1000 (no NaN/Inf), random sign; any byte for iu8
    std::mt19937 rng(12345);
    std::vector<uint32_t> a(kLanes * 8), b(kLanes * 8);
    const auto fp8_byte = [&] {
        return static_cast<uint32_t>((rng() & 0x80) | (0x30 + rng() % 0x18));
    };
    for(auto* v : {&a, &b})
        for(auto& x : *v)
            x = fp8_byte() | fp8_byte() << 8 | fp8_byte() << 16 | fp8_byte() << 24;

    uint32_t *da, *db, *dc;
    HIP_OK(hipMalloc(&da, a.size() * 4));
    HIP_OK(hipMalloc(&db, b.size() * 4));
    HIP_OK(hipMalloc(&dc, a.size() * 4));
    HIP_OK(hipMemcpy(da, a.data(), a.size() * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(db, b.data(), b.size() * 4, hipMemcpyHostToDevice));

    run_case<0, 0>(da, db, dc);
    run_case<0, 4>(da, db, dc);
    run_case<1, 0>(da, db, dc);
    run_case<1, 4>(da, db, dc);

    HIP_OK(hipFree(da));
    HIP_OK(hipFree(db));
    HIP_OK(hipFree(dc));
}
