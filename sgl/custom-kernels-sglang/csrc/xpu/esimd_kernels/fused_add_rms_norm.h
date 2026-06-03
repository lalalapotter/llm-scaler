/* fused_add_rms_norm.h — Fused residual add + RMSNorm (Gemma-style).
 *
 * For decode (bsz=1): residual[1,K] += hidden[1,K]; output[1,K] = rmsnorm(residual) * weight
 * Gemma convention: weight is pre-adjusted (w+1.0 already applied by caller).
 *
 * Single WG, 1 thread. K=2048 → 4 iterations with VL=512.
 * Two-pass: pass 1 = add + sum_sq; pass 2 = normalize + write output.
 * Residual updated in-place.
 *
 * Templated on activation dtype T ∈ {fp16, bf16}. Internal accumulation is fp32.
 */

#pragma once
#include "utils.h"

template <typename T>
struct FusedAddRmsNorm_kernel {
    T*       hidden_ptr;    // [1, K] — input, also used as output
    T*       residual_ptr;  // [1, K] — updated in-place
    const T* weight_ptr;    // [K] — Gemma norm weight (w+1.0)
    int K;
    float eps;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        constexpr int VL = 512;
        int n_chunks = K / VL;

        // Pass 1: residual += hidden, accumulate sum_sq
        float sum_sq = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> h = block_load<T, VL>(hidden_ptr + offset);
            simd<float, VL> r = block_load<T, VL>(residual_ptr + offset);
            simd<float, VL> added = h + r;

            // Write residual in-place
            block_store<T, VL>(residual_ptr + offset, simd<T, VL>(added));

            simd<float, VL> sq = added * added;
            sq.select<256,1>(0) += sq.select<256,1>(256);
            sq.select<128,1>(0) += sq.select<128,1>(128);
            sq.select<64,1>(0) += sq.select<64,1>(64);
            sq.select<32,1>(0) += sq.select<32,1>(32);
            sq.select<16,1>(0) += sq.select<16,1>(16);
            sq.select<8,1>(0) += sq.select<8,1>(8);
            sq.select<4,1>(0) += sq.select<4,1>(4);
            sq.select<2,1>(0) += sq.select<2,1>(2);
            sum_sq += (float)sq[0] + (float)sq[1];
        }

        float inv_rms = sycl::ext::intel::esimd::rsqrt(
            simd<float, 8>(sum_sq / (float)K + eps))[0];

        // Pass 2: normalize and write output (reuse hidden_ptr as output)
        for (int c = 0; c < n_chunks; c++) {
            int offset = c * VL;
            simd<float, VL> r = block_load<T, VL>(residual_ptr + offset);
            simd<float, VL> w = block_load<T, VL>(weight_ptr + offset);
            simd<float, VL> normed = r * inv_rms * w;
            block_store<T, VL>(hidden_ptr + offset, simd<T, VL>(normed));
        }
    }
};

template <typename T>
inline void fused_add_rms_norm_host_t(
    T* hidden_ptr, T* residual_ptr, const T* weight_ptr,
    int K, float eps, sycl::queue& q)
{
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(1, 1),
            FusedAddRmsNorm_kernel<T>{hidden_ptr, residual_ptr, weight_ptr, K, eps});
    });
}

// Backward-compat fp16 entry point (existing callers).
inline void fused_add_rms_norm_host(
    fp16* hidden_ptr, fp16* residual_ptr, const fp16* weight_ptr,
    int K, float eps, sycl::queue& q)
{
    fused_add_rms_norm_host_t<fp16>(hidden_ptr, residual_ptr, weight_ptr, K, eps, q);
}
