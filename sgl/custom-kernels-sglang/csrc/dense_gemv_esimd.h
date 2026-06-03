// ESIMD dense bf16/fp16 GEMV — entry points called from dense_gemv.sycl.
//
// Layout (matches PyTorch nn.Linear, M=1):
//   x   : [K]      T   row-major
//   W   : [N, K]   T   row-major (weight = [out_features, in_features])
//   out : [N]      T   row-major
//
// Returns false if (K, N) is not in the supported specialization table; the
// caller must fall back to the generic SYCL kernel.

#pragma once
#include <ATen/ATen.h>
#include <sycl/sycl.hpp>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

// scratch_fp32 is used only on K-split shapes (currently N=256 KS=2 and
// N=64 KS=4). Caller passes an at::Tensor& it owns; the launch fills it
// with [KS, N] fp32 partials and submits a final cast kernel that writes
// `out`. For non-K-split shapes the parameter is ignored.
bool launch_dense_gemv_esimd_bf16(
    sycl::queue& q,
    const c10::BFloat16* x, const c10::BFloat16* W, c10::BFloat16* out,
    int K, int N, at::Tensor& scratch_fp32);

bool launch_dense_gemv_esimd_fp16(
    sycl::queue& q,
    const c10::Half* x, const c10::Half* W, c10::Half* out,
    int K, int N, at::Tensor& scratch_fp32);
