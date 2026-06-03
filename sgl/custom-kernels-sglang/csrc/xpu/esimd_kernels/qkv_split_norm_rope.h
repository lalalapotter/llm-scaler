#pragma once
#include "utils.h"
#include <cmath>

// Fused QKV Split + RMSNorm + RoPE kernel
// Adapted from qkv.split/qkv.split.h for project conventions.
//
// Operations per head:
//   Q heads: RMSNorm(weight+1.0, eps=1e-6) → RoPE(theta=10M)
//   K heads: RMSNorm(weight+1.0, eps=1e-6) → RoPE(theta=10M)
//   G heads (gating): copy only
//   V heads: copy only
//
// Work decomposition: 2D dispatch (totalHeads, nTokens), one WG per (head, token)
//   With gating:    totalHeads = 2*qHead + 2*kvHead  (Q,G interleaved)
//   Without gating: totalHeads = qHead + 2*kvHead
//
// Templated on activation dtype T (fp16 / bf16). Norm weights and cos_sin_cache
// share T. Compute is fp32 throughout (sum_sq, RoPE, sigmoid).

#define QKV_LOCATION_Q 0
#define QKV_LOCATION_G 1
#define QKV_LOCATION_K 2
#define QKV_LOCATION_V 3

template <typename T>
ESIMD_INLINE void qkv_split_norm_rope_kernel(
    uint8_t* qkvState,
    uint8_t* qState,
    uint8_t* gateState,
    uint8_t* kState,
    uint8_t* vState,
    uint8_t* normWq,
    uint8_t* normWk,
    uint32_t* ropePos,
    T* ropeCosSinCache,  // [max_pos, rotaryDim] T — first half cos, second half sin
    uint32_t ntoks,
    uint32_t hiddenDim,
    uint32_t headDim,
    uint32_t qHead,
    uint32_t kvHead,
    uint32_t rotaryDim,
    sycl::nd_item<2>& ndi) {

    constexpr float eps = 1e-6f;

    int32_t headIdx = ndi.get_group(0);
    int32_t tokIdx  = ndi.get_group(1);

    uint32_t outHead = headIdx;
    uint32_t hdimQkv  = headDim * (qHead + 2 * kvHead);
    uint32_t hdimQgkv = headDim * (2 * qHead + 2 * kvHead);
    uint32_t outputGate = (hiddenDim == hdimQgkv) ? 1 : 0;
    uint32_t whereAmI = QKV_LOCATION_Q;

    uint32_t inputOffset = tokIdx * hiddenDim + headIdx * headDim;
    uint32_t outputOffset;

    uint32_t i32RopeCoord = ropePos[tokIdx];

    simd<T, 256> activation;

    if (headDim != 256) {
        return;
    }

    // Load 256 elements (T) from QKV buffer
    activation = block_load<T, 256>((T*)qkvState + inputOffset);

    // Determine which output this head maps to
    if (outputGate == 1) {
        // With gating: heads are [Q0,G0, Q1,G1, ..., K0, K1, ..., V0, V1, ...]
        if ((uint32_t)headIdx < 2 * qHead) {
            whereAmI = headIdx & 0x1;  // 0=Q, 1=G
            outHead = headIdx >> 1;
        } else if ((uint32_t)headIdx < (2 * qHead + kvHead)) {
            whereAmI = QKV_LOCATION_K;
            outHead = headIdx - 2 * qHead;
        } else {
            whereAmI = QKV_LOCATION_V;
            outHead = headIdx - 2 * qHead - kvHead;
        }
    } else {
        // Without gating: heads are [Q0, Q1, ..., K0, K1, ..., V0, V1, ...]
        if ((uint32_t)headIdx < qHead) {
            whereAmI = QKV_LOCATION_Q;
            outHead = headIdx;
        } else if ((uint32_t)headIdx < (qHead + kvHead)) {
            whereAmI = QKV_LOCATION_K;
            outHead = headIdx - qHead;
        } else {
            whereAmI = QKV_LOCATION_V;
            outHead = headIdx - qHead - kvHead;
        }
    }

    if (whereAmI == QKV_LOCATION_Q) {
        // RMSNorm + partial RoPE for Q
        simd<T, 256> tRmsWeights;
        simd<float, 256> fp32RmsWeights;
        simd<float, 256> outputTemp;

        outputOffset = qHead * headDim * tokIdx + outHead * headDim;
        outputTemp = activation;
        simd<float, 256> outputSq = outputTemp * outputTemp;

        // RMSNorm: x * (weight + 1.0) / rms
        tRmsWeights = block_load<T, 256>((T*)normWq);
        float acc = sycl::ext::intel::esimd::detail::sum<float, float, 256>(outputSq) / (float)headDim;
        float scale = __ESIMD_NS::rsqrt(acc + eps);
        fp32RmsWeights = tRmsWeights + 1.0f;
        outputTemp = outputTemp * fp32RmsWeights;
        outputTemp.select<256, 1>(0) = outputTemp.select<256, 1>(0) * scale;

        // RoPE: read from rotary_emb.cos_sin_cache [max_pos, rotaryDim]
        // Layout: [cos(rotaryHalf), sin(rotaryHalf)] per row
        {
            // Row offset in T elements: position * rotaryDim
            uint32_t row_offset = i32RopeCoord * rotaryDim;
            if (rotaryDim == 64) {
                // Load 64 T: [cos(32), sin(32)]
                simd<T, 64> cs;
                cs.template select<32, 1>(0) = block_load<T, 32>(ropeCosSinCache + row_offset);
                cs.template select<32, 1>(32) = block_load<T, 32>(ropeCosSinCache + row_offset + 32);
                simd<float, 32> pcos = cs.template select<32, 1>(0);
                simd<float, 32> psin = cs.template select<32, 1>(32);

                simd<float, 32> x1 = outputTemp.select<32, 1>(0);
                simd<float, 32> x2 = outputTemp.select<32, 1>(32);
                outputTemp.select<32, 1>(0)  = x1 * pcos - x2 * psin;
                outputTemp.select<32, 1>(32) = x2 * pcos + x1 * psin;
            } else {
                // Full rotation: load 256 T
                simd<float, 128> fcos, fsin;
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    simd<T, 32> c = block_load<T, 32>(ropeCosSinCache + row_offset + 32 * kk);
                    fcos.template select<32, 1>(32 * kk) = c;
                }
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    simd<T, 32> s = block_load<T, 32>(ropeCosSinCache + row_offset + 128 + 32 * kk);
                    fsin.template select<32, 1>(32 * kk) = s;
                }

                simd<float, 128> x1 = outputTemp.select<128, 1>(0);
                simd<float, 128> x2 = outputTemp.select<128, 1>(128);
                outputTemp.select<128, 1>(0)   = x1 * fcos - x2 * fsin;
                outputTemp.select<128, 1>(128) = x2 * fcos + x1 * fsin;
            }
        }

        activation = outputTemp;
        block_store<T, 256>((T*)qState + outputOffset, activation);
    }
    else if (whereAmI == QKV_LOCATION_G) {
        // Gate: sigmoid(x) = 1 / (1 + e^(-x))
        outputOffset = qHead * headDim * tokIdx + outHead * headDim;
        simd<float, 256> gateTemp = activation;
        gateTemp = 1.0f / (1.0f + exp<float, 256>(-gateTemp));
        activation = gateTemp;
        block_store<T, 256>((T*)gateState + outputOffset, activation);
    }
    else if (whereAmI == QKV_LOCATION_K) {
        // RMSNorm + partial RoPE for K
        simd<T, 256> tRmsWeights;
        simd<float, 256> fp32RmsWeights;
        simd<float, 256> outputTemp;

        outputOffset = kvHead * headDim * tokIdx + outHead * headDim;
        outputTemp = activation;
        simd<float, 256> kOutputSq = outputTemp * outputTemp;

        tRmsWeights = block_load<T, 256>((T*)normWk);
        float acc = sycl::ext::intel::esimd::detail::sum<float, float, 256>(kOutputSq) / (float)headDim;
        float scale = __ESIMD_NS::rsqrt(acc + eps);
        fp32RmsWeights = tRmsWeights + 1.0f;
        outputTemp = outputTemp * fp32RmsWeights;
        outputTemp.select<256, 1>(0) = outputTemp.select<256, 1>(0) * scale;

        // RoPE: read from cos_sin_cache (same as Q)
        {
            uint32_t krow_offset = i32RopeCoord * rotaryDim;
            if (rotaryDim == 64) {
                simd<T, 64> kcs;
                kcs.template select<32, 1>(0) = block_load<T, 32>(ropeCosSinCache + krow_offset);
                kcs.template select<32, 1>(32) = block_load<T, 32>(ropeCosSinCache + krow_offset + 32);
                simd<float, 32> kpcos = kcs.template select<32, 1>(0);
                simd<float, 32> kpsin = kcs.template select<32, 1>(32);

                simd<float, 32> kx1 = outputTemp.select<32, 1>(0);
                simd<float, 32> kx2 = outputTemp.select<32, 1>(32);
                outputTemp.select<32, 1>(0)  = kx1 * kpcos - kx2 * kpsin;
                outputTemp.select<32, 1>(32) = kx2 * kpcos + kx1 * kpsin;
            } else {
                simd<float, 128> kfcos, kfsin;
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    simd<T, 32> kc = block_load<T, 32>(ropeCosSinCache + krow_offset + 32 * kk);
                    kfcos.template select<32, 1>(32 * kk) = kc;
                }
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    simd<T, 32> ks = block_load<T, 32>(ropeCosSinCache + krow_offset + 128 + 32 * kk);
                    kfsin.template select<32, 1>(32 * kk) = ks;
                }

                simd<float, 128> kx1 = outputTemp.select<128, 1>(0);
                simd<float, 128> kx2 = outputTemp.select<128, 1>(128);
                outputTemp.select<128, 1>(0)   = kx1 * kfcos - kx2 * kfsin;
                outputTemp.select<128, 1>(128) = kx2 * kfcos + kx1 * kfsin;
            }
        }

        activation = outputTemp;
        block_store<T, 256>((T*)kState + outputOffset, activation);
    }
    else if (whereAmI == QKV_LOCATION_V) {
        // V: copy only
        outputOffset = kvHead * headDim * tokIdx + outHead * headDim;
        block_store<T, 256>((T*)vState + outputOffset, activation);
    }
}

// Host dispatcher: submits the 2D kernel to the SYCL queue.
// Templated on activation dtype T (fp16 / bf16).
template <typename T>
inline void qkv_split_norm_rope_host_t(
    uint8_t* qkvState,
    uint8_t* qState,
    uint8_t* gateState,
    uint8_t* kState,
    uint8_t* vState,
    uint8_t* normWq,
    uint8_t* normWk,
    uint32_t* ropePos,
    T* ropeCosSinCache,
    uint32_t ntoks,
    uint32_t hiddenDim,
    uint32_t qHead,
    uint32_t kvHead,
    bool attnOutputGate,
    uint32_t rotaryDim,
    sycl::queue& q) {

    constexpr uint32_t headDim = 256;
    uint32_t totalHeads = attnOutputGate
        ? (2 * qHead + 2 * kvHead)
        : (qHead + 2 * kvHead);

    sycl::range<2> globalRange(totalHeads, ntoks);
    sycl::range<2> localRange(1, 1);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(globalRange, localRange),
            [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                qkv_split_norm_rope_kernel<T>(
                    qkvState, qState, gateState, kState, vState,
                    normWq, normWk, ropePos, ropeCosSinCache,
                    ntoks, hiddenDim, headDim, qHead, kvHead, rotaryDim, ndi);
            });
    });
}

// Backward-compat fp16 entry point.
inline void qkv_split_norm_rope_host(
    uint8_t* qkvState,
    uint8_t* qState,
    uint8_t* gateState,
    uint8_t* kState,
    uint8_t* vState,
    uint8_t* normWq,
    uint8_t* normWk,
    uint32_t* ropePos,
    fp16* ropeCosSinCache,
    uint32_t ntoks,
    uint32_t hiddenDim,
    uint32_t qHead,
    uint32_t kvHead,
    bool attnOutputGate,
    uint32_t rotaryDim,
    sycl::queue& q) {
    qkv_split_norm_rope_host_t<fp16>(
        qkvState, qState, gateState, kState, vState,
        normWq, normWk, ropePos, ropeCosSinCache,
        ntoks, hiddenDim, qHead, kvHead, attnOutputGate, rotaryDim, q);
}
