/* gdn_conv_fused_seq.h — Fused Conv1d + GDN ESIMD kernel for SEQUENTIAL qkvz layout.
 *
 * Templated on dtype T (sycl::half for fp16, sycl::ext::oneapi::bfloat16 for bf16)
 * and TS (state dtype, fp16/bf16/fp32). This sgl-tree variant reads conv_state
 * directly in sglang's native MambaPool layout (cache, conv_dim, W-1), avoiding
 * the (cache, W-1, conv_dim) transpose round-trip that the upstream vllm
 * version required.
 *
 * Sequential qkvz layout (per TP rank):
 *   [q(H*K) | k(H*K) | v(HV*V) | z(HV*V)]
 *
 * Sequential ba layout:
 *   [b_all(HV) | a_all(HV)]
 *
 * conv_state pool layout (sglang native):
 *   pool[cache_idx][d=0..conv_dim-1][t=0..W-2]
 *   where t=0 is the oldest history and t=W-2 is the most recent token.
 *
 * Per dim-chunk of 64 elements: each pool row is 3 contiguous tokens,
 * so 64*3=192 elements packed [d0t0,d0t1,d0t2, d1t0,..., d63t2]. We
 * issue one block_load<T,192> and demux with select<64,3>(t) to recover
 * each token's 64-wide row.
 *
 * This kernel handles only K=V=128, conv kernel size = 4 (so W-1 = 3).
 */

#include "utils.h"

namespace xmem = sycl::ext::intel::experimental::esimd;

/* ---- ESIMD scalar math helpers ---- */
ESIMD_INLINE float esimd_expf_seq(float x) {
    simd<float, 8> v(x);
    v = sycl::ext::intel::esimd::exp(v);
    return v[0];
}
ESIMD_INLINE float esimd_logf_seq(float x) {
    simd<float, 8> v(x);
    v = sycl::ext::intel::esimd::log(v);
    return v[0];
}
ESIMD_INLINE float esimd_sqrtf_seq(float x) {
    simd<float, 8> v(x);
    v = sycl::ext::intel::esimd::sqrt(v);
    return v[0];
}

/* ---- LSC load/store helpers (typed) ---- */
template <typename T>
ESIMD_INLINE simd<float, 64> lsc_load_state_64_seq(const T* ptr) {
    return xmem::lsc_block_load<T, 64,
        xmem::lsc_data_size::default_size,
        xmem::cache_hint::streaming, xmem::cache_hint::cached>(ptr);
}

template <typename T>
ESIMD_INLINE void lsc_store_state_64_seq(T* ptr, simd<float, 64> val) {
    xmem::lsc_block_store<T, 64,
        xmem::lsc_data_size::default_size,
        xmem::cache_hint::streaming, xmem::cache_hint::write_back>(
        ptr, simd<T, 64>(val));
}

/* ---- conv_state read in (conv_dim, W-1) layout ----
 * For a chunk of 64 dims starting at chunk_start, load 192 contiguous T
 * elements from cstate_base + chunk_start*3, then split into 3 rows of 64.
 * Returns each row promoted to float for f32-domain conv math.
 */
template <typename T>
ESIMD_INLINE void load_conv3_chunk_64_seq(
    const T* cstate_base, int chunk_start,
    simd<float, 64>& s0, simd<float, 64>& s1, simd<float, 64>& s2)
{
    constexpr int W_minus_1 = 3;
    simd<T, 192> blk = block_load<T, 192>(cstate_base + chunk_start * W_minus_1);
    simd<T, 64> r0 = blk.template select<64, W_minus_1>(0);
    simd<T, 64> r1 = blk.template select<64, W_minus_1>(1);
    simd<T, 64> r2 = blk.template select<64, W_minus_1>(2);
    s0 = r0;
    s1 = r1;
    s2 = r2;
}

/* ---- conv_state inline shift in (conv_dim, W-1) layout ----
 * Writes [s1, s2, x_new] interleaved as [d0t0=s1[d0], d0t1=s2[d0], d0t2=x[d0],
 * d1t0=s1[d1], ...] back to cstate_base + chunk_start*3.
 */
template <typename T>
ESIMD_INLINE void store_conv3_shift_64_seq(
    T* cstate_base, int chunk_start,
    simd<float, 64> s1, simd<float, 64> s2, simd<T, 64> x_new)
{
    constexpr int W_minus_1 = 3;
    simd<T, 192> out_blk;
    simd<T, 64> r0 = s1;
    simd<T, 64> r1 = s2;
    out_blk.template select<64, W_minus_1>(0) = r0;
    out_blk.template select<64, W_minus_1>(1) = r1;
    out_blk.template select<64, W_minus_1>(2) = x_new;
    block_store<T, 192>(cstate_base + chunk_start * W_minus_1, out_blk);
}

/* ---- Dot product 128 (split lo/hi 64) ---- */
ESIMD_INLINE float gdn_dot128_seq(simd<float, 64> a_lo, simd<float, 64> a_hi,
                                   simd<float, 64> b_lo, simd<float, 64> b_hi) {
    simd<float, 64> p_lo = a_lo * b_lo;
    simd<float, 64> p_hi = a_hi * b_hi;
    p_lo += p_hi;
    p_lo.template select<32,1>(0) += p_lo.template select<32,1>(32);
    p_lo.template select<16,1>(0) += p_lo.template select<16,1>(16);
    p_lo.template select<8,1>(0) += p_lo.template select<8,1>(8);
    p_lo.template select<4,1>(0) += p_lo.template select<4,1>(4);
    p_lo.template select<2,1>(0) += p_lo.template select<2,1>(2);
    return p_lo[0] + p_lo[1];
}

template <typename T>
ESIMD_INLINE float gdn_load_T_scalar_seq(const T* base, int64_t idx) {
    int64_t aligned = idx & ~15;
    int lane = (int)(idx & 15);
    simd<T, 16> chunk = block_load<T, 16>(base + aligned);
    simd<float, 16> chunk_f32 = chunk;
    return chunk_f32[lane];
}

/* ---- SLM layout per WG (byte offsets) ---- */
static constexpr int SLM_Q_LO_SEQ = 0;
static constexpr int SLM_Q_HI_SEQ = 256;
static constexpr int SLM_K_LO_SEQ = 512;
static constexpr int SLM_K_HI_SEQ = 768;
static constexpr int SLM_V_SEQ    = 1024;

/* ============================================================
 * KERNEL: reads from SEQUENTIAL qkvz layout [q|k|v|z].
 * conv_state in sglang native (cache, conv_dim, W-1) layout.
 * ============================================================ */
template<typename T, typename TS, int WG_SIZE>
ESIMD_INLINE void gdn_conv_fused_seq_kernel(
    const T* __restrict__ qkvz_ptr,
    int64_t qkvz_stride0,
    T* __restrict__ conv_state_ptr,
    const T* __restrict__ conv_weight_ptr,
    const T* __restrict__ conv_bias_ptr,
    const int* __restrict__ conv_state_indices_ptr,
    const T* __restrict__ A_log_ptr,
    const T* __restrict__ dt_bias_ptr,
    const T* __restrict__ ba_ptr,
    int64_t ba_stride0,
    TS* __restrict__ ssm_state_ptr,
    const int* __restrict__ ssm_state_indices_ptr,
    T* __restrict__ output_ptr,
    T* __restrict__ z_out_ptr,
    int N, int H, int HV, int gdn_K, int gdn_V,
    float attn_scale, int64_t conv_stride0, int64_t ssm_stride0,
    int inline_conv_shift,
    nd_item<3>& ndi)
{
    slm_init<2048>();

    const int seq_idx = ndi.get_group(0);
    const int hv = ndi.get_group(1);
    const int tid = ndi.get_local_id(2);

    const int heads_per_group = HV / H;
    const int i_h = hv / heads_per_group;

    const int num_v_threads = WG_SIZE - 4 * H;
    const bool double_v = (HV > num_v_threads / 2);

    const int conv_idx = conv_state_indices_ptr[seq_idx];
    const int ssm_idx = ssm_state_indices_ptr[seq_idx];

    const int q_base = 0;
    const int k_base = H * gdn_K;
    const int v_base = 2 * H * gdn_K;
    const int z_base = v_base + HV * gdn_V;

    int qkvz_offset = 0;
    int chunk_start = 0;
    int qkvz_offset_hi = 0;
    int chunk_start_hi = 0;

    if (tid < 2 * H) {
        int q_head = tid / 2;
        qkvz_offset = q_base + q_head * gdn_K + (tid & 1) * 64;
        chunk_start = qkvz_offset;
    } else if (tid < 4 * H) {
        int k_tid = tid - 2 * H;
        int k_head = k_tid / 2;
        qkvz_offset = k_base + k_head * gdn_K + (k_tid & 1) * 64;
        chunk_start = qkvz_offset;
    } else if (double_v) {
        int v_tid = tid - 4 * H;
        int v_hv = v_tid;
        qkvz_offset = v_base + v_hv * gdn_V;
        qkvz_offset_hi = qkvz_offset + 64;
        chunk_start = qkvz_offset;
        chunk_start_hi = chunk_start + 64;
    } else {
        int v_tid = tid - 4 * H;
        int v_hv = v_tid / 2;
        qkvz_offset = v_base + v_hv * gdn_V + (v_tid & 1) * 64;
        chunk_start = qkvz_offset;
    }

    const bool v_oob = (tid >= 4 * H) &&
        (double_v ? (tid - 4 * H >= HV) : ((tid - 4 * H) / 2 >= HV));
    if (v_oob) {
        qkvz_offset = v_base;
        qkvz_offset_hi = v_base + 64;
        chunk_start = v_base;
        chunk_start_hi = v_base + 64;
    }

    // ---- Phase 1: Conv1d ----
    const T* qkvz_row = qkvz_ptr + (int64_t)seq_idx * qkvz_stride0;
    T* cstate_base = conv_state_ptr + (int64_t)conv_idx * conv_stride0;

    simd<T, 64> x_T = block_load<T, 64>(qkvz_row + qkvz_offset);
    simd<float, 64> x_f32 = x_T;

    simd<float, 64> s0, s1, s2;
    load_conv3_chunk_64_seq<T>(cstate_base, chunk_start, s0, s1, s2);

    simd<T, 256> w_raw = block_load<T, 256>(conv_weight_ptr + (int64_t)chunk_start * 4);
    simd<float, 64> conv_result =
        s0 * w_raw.template select<64, 4>(0) + s1 * w_raw.template select<64, 4>(1) +
        s2 * w_raw.template select<64, 4>(2) + x_f32 * w_raw.template select<64, 4>(3) +
        (simd<float, 64>)block_load<T, 64>(conv_bias_ptr + chunk_start);

    {
        simd<float, 64> exp_neg = sycl::ext::intel::esimd::exp(-conv_result);
        conv_result = conv_result / (1.0f + exp_neg);
    }

    simd<T, 64> x_T_hi;
    simd<float, 64> s0_hi, s1_hi, s2_hi, conv_result_hi;

    if (double_v && tid >= 4 * H && !v_oob) {
        x_T_hi = block_load<T, 64>(qkvz_row + qkvz_offset_hi);
        simd<float, 64> x_f32_hi = x_T_hi;

        load_conv3_chunk_64_seq<T>(cstate_base, chunk_start_hi, s0_hi, s1_hi, s2_hi);

        simd<T, 256> w_raw_hi = block_load<T, 256>(
            conv_weight_ptr + (int64_t)chunk_start_hi * 4);
        conv_result_hi =
            s0_hi * w_raw_hi.template select<64, 4>(0) + s1_hi * w_raw_hi.template select<64, 4>(1) +
            s2_hi * w_raw_hi.template select<64, 4>(2) + x_f32_hi * w_raw_hi.template select<64, 4>(3) +
            (simd<float, 64>)block_load<T, 64>(conv_bias_ptr + chunk_start_hi);

        simd<float, 64> exp_neg_hi = sycl::ext::intel::esimd::exp(-conv_result_hi);
        conv_result_hi = conv_result_hi / (1.0f + exp_neg_hi);
    }

    // ---- Store q/k/v to SLM ----
    {
        const int q_tid_lo = 2 * i_h;
        if (tid == q_tid_lo)     slm_block_store<float, 64>(SLM_Q_LO_SEQ, conv_result);
        if (tid == q_tid_lo + 1) slm_block_store<float, 64>(SLM_Q_HI_SEQ, conv_result);

        const int k_tid_lo = 2 * H + 2 * i_h;
        if (tid == k_tid_lo)     slm_block_store<float, 64>(SLM_K_LO_SEQ, conv_result);
        if (tid == k_tid_lo + 1) slm_block_store<float, 64>(SLM_K_HI_SEQ, conv_result);

        if (double_v) {
            if (tid == 4 * H + hv) {
                slm_block_store<float, 64>(SLM_V_SEQ, conv_result);
                slm_block_store<float, 64>(SLM_V_SEQ + 256, conv_result_hi);
            }
        } else {
            const int v_tid_lo = 4 * H + 2 * hv;
            if (tid == v_tid_lo)     slm_block_store<float, 64>(SLM_V_SEQ, conv_result);
            if (tid == v_tid_lo + 1) slm_block_store<float, 64>(SLM_V_SEQ + 256, conv_result);
        }
    }

    barrier();

    // ---- Phase 2: GDN ----
    constexpr int VPT = 128 / WG_SIZE;

    if (ssm_idx >= 0) {
        simd<float, 64> q_lo = slm_block_load<float, 64>(SLM_Q_LO_SEQ);
        simd<float, 64> q_hi = slm_block_load<float, 64>(SLM_Q_HI_SEQ);
        simd<float, 64> k_lo = slm_block_load<float, 64>(SLM_K_LO_SEQ);
        simd<float, 64> k_hi = slm_block_load<float, 64>(SLM_K_HI_SEQ);

        float q_inv = 1.0f / esimd_sqrtf_seq(gdn_dot128_seq(q_lo, q_hi, q_lo, q_hi) + 1e-6f);
        float k_inv = 1.0f / esimd_sqrtf_seq(gdn_dot128_seq(k_lo, k_hi, k_lo, k_hi) + 1e-6f);
        q_lo *= q_inv * attn_scale; q_hi *= q_inv * attn_scale;
        k_lo *= k_inv; k_hi *= k_inv;

        const int vi0 = tid * VPT;
        simd<float, VPT> v_f32 = slm_block_load<float, VPT>(SLM_V_SEQ + vi0 * (int)sizeof(float));

        const float A_log_val = gdn_load_T_scalar_seq<T>(A_log_ptr, hv);
        const float dt_bias_val = gdn_load_T_scalar_seq<T>(dt_bias_ptr, hv);
        const float neg_exp_A = -esimd_expf_seq(A_log_val);

        const int b_col = hv;
        const int a_col = HV + hv;
        float a_val = gdn_load_T_scalar_seq<T>(ba_ptr, (int64_t)seq_idx * ba_stride0 + a_col);
        float b_val = gdn_load_T_scalar_seq<T>(ba_ptr, (int64_t)seq_idx * ba_stride0 + b_col);
        float x_gate = a_val + dt_bias_val;
        float sp = (x_gate > 20.0f) ? x_gate : esimd_logf_seq(1.0f + esimd_expf_seq(x_gate));
        float g = neg_exp_A * sp;
        float exp_g = esimd_expf_seq(g);
        float beta = 1.0f / (1.0f + esimd_expf_seq(-b_val));

        TS* sstate_base = ssm_state_ptr +
            (int64_t)ssm_idx * ssm_stride0 + (int64_t)hv * gdn_V * gdn_K;

        simd<float, VPT> o_acc;

        if constexpr (VPT == 4) {
            TS* sr0 = sstate_base + (int64_t)(vi0 + 0) * gdn_K;
            TS* sr1 = sstate_base + (int64_t)(vi0 + 1) * gdn_K;
            TS* sr2 = sstate_base + (int64_t)(vi0 + 2) * gdn_K;
            TS* sr3 = sstate_base + (int64_t)(vi0 + 3) * gdn_K;

            simd<float, 64> h0_lo = lsc_load_state_64_seq<TS>(sr0);
            simd<float, 64> h0_hi = lsc_load_state_64_seq<TS>(sr0 + 64);
            simd<float, 64> h1_lo = lsc_load_state_64_seq<TS>(sr1);
            simd<float, 64> h1_hi = lsc_load_state_64_seq<TS>(sr1 + 64);
            simd<float, 64> h2_lo = lsc_load_state_64_seq<TS>(sr2);
            simd<float, 64> h2_hi = lsc_load_state_64_seq<TS>(sr2 + 64);
            simd<float, 64> h3_lo = lsc_load_state_64_seq<TS>(sr3);
            simd<float, 64> h3_hi = lsc_load_state_64_seq<TS>(sr3 + 64);

            h0_lo *= exp_g; h0_hi *= exp_g;
            h1_lo *= exp_g; h1_hi *= exp_g;
            h2_lo *= exp_g; h2_hi *= exp_g;
            h3_lo *= exp_g; h3_hi *= exp_g;

            float kv0 = gdn_dot128_seq(h0_lo, h0_hi, k_lo, k_hi);
            float kv1 = gdn_dot128_seq(h1_lo, h1_hi, k_lo, k_hi);
            float kv2 = gdn_dot128_seq(h2_lo, h2_hi, k_lo, k_hi);
            float kv3 = gdn_dot128_seq(h3_lo, h3_hi, k_lo, k_hi);

            float d0 = (v_f32[0] - kv0) * beta;
            float d1 = (v_f32[1] - kv1) * beta;
            float d2 = (v_f32[2] - kv2) * beta;
            float d3 = (v_f32[3] - kv3) * beta;

            h0_lo += d0 * k_lo; h0_hi += d0 * k_hi;
            h1_lo += d1 * k_lo; h1_hi += d1 * k_hi;
            h2_lo += d2 * k_lo; h2_hi += d2 * k_hi;
            h3_lo += d3 * k_lo; h3_hi += d3 * k_hi;

            o_acc[0] = gdn_dot128_seq(h0_lo, h0_hi, q_lo, q_hi);
            o_acc[1] = gdn_dot128_seq(h1_lo, h1_hi, q_lo, q_hi);
            o_acc[2] = gdn_dot128_seq(h2_lo, h2_hi, q_lo, q_hi);
            o_acc[3] = gdn_dot128_seq(h3_lo, h3_hi, q_lo, q_hi);

            lsc_store_state_64_seq<TS>(sr0, h0_lo);
            lsc_store_state_64_seq<TS>(sr0 + 64, h0_hi);
            lsc_store_state_64_seq<TS>(sr1, h1_lo);
            lsc_store_state_64_seq<TS>(sr1 + 64, h1_hi);
            lsc_store_state_64_seq<TS>(sr2, h2_lo);
            lsc_store_state_64_seq<TS>(sr2 + 64, h2_hi);
            lsc_store_state_64_seq<TS>(sr3, h3_lo);
            lsc_store_state_64_seq<TS>(sr3 + 64, h3_hi);
        } else {
            // VPT == 2 (WG=64)
            TS* sr0 = sstate_base + (int64_t)(vi0 + 0) * gdn_K;
            TS* sr1 = sstate_base + (int64_t)(vi0 + 1) * gdn_K;

            simd<float, 64> h0_lo = lsc_load_state_64_seq<TS>(sr0);
            simd<float, 64> h0_hi = lsc_load_state_64_seq<TS>(sr0 + 64);
            simd<float, 64> h1_lo = lsc_load_state_64_seq<TS>(sr1);
            simd<float, 64> h1_hi = lsc_load_state_64_seq<TS>(sr1 + 64);

            h0_lo *= exp_g; h0_hi *= exp_g;
            h1_lo *= exp_g; h1_hi *= exp_g;

            float kv0 = gdn_dot128_seq(h0_lo, h0_hi, k_lo, k_hi);
            float kv1 = gdn_dot128_seq(h1_lo, h1_hi, k_lo, k_hi);

            float d0 = (v_f32[0] - kv0) * beta;
            float d1 = (v_f32[1] - kv1) * beta;

            h0_lo += d0 * k_lo; h0_hi += d0 * k_hi;
            h1_lo += d1 * k_lo; h1_hi += d1 * k_hi;

            o_acc[0] = gdn_dot128_seq(h0_lo, h0_hi, q_lo, q_hi);
            o_acc[1] = gdn_dot128_seq(h1_lo, h1_hi, q_lo, q_hi);

            lsc_store_state_64_seq<TS>(sr0, h0_lo);
            lsc_store_state_64_seq<TS>(sr0 + 64, h0_hi);
            lsc_store_state_64_seq<TS>(sr1, h1_lo);
            lsc_store_state_64_seq<TS>(sr1 + 64, h1_hi);
        }

        T* out = output_ptr + (int64_t)seq_idx * HV * gdn_V + (int64_t)hv * gdn_V + vi0;
        if constexpr (VPT >= 4) {
            xmem::lsc_block_store<T, VPT,
                xmem::lsc_data_size::default_size,
                xmem::cache_hint::streaming, xmem::cache_hint::write_back>(
                out, simd<T, VPT>(o_acc));
        } else {
            block_store<T, VPT>(out, simd<T, VPT>(o_acc));
        }
    } else {
        int vi0_z = tid * VPT;
        T* out = output_ptr + (int64_t)seq_idx * HV * gdn_V + (int64_t)hv * gdn_V + vi0_z;
        if constexpr (VPT >= 4) {
            xmem::lsc_block_store<T, VPT,
                xmem::lsc_data_size::default_size,
                xmem::cache_hint::streaming, xmem::cache_hint::write_back>(
                out, simd<T, VPT>(0.0f));
        } else {
            block_store<T, VPT>(out, simd<T, VPT>(0.0f));
        }
    }

    // ---- Phase 3: conv_state shift (inline path) ----
    if (inline_conv_shift && conv_idx >= 0 && hv == 0 && !v_oob) {
        store_conv3_shift_64_seq<T>(cstate_base, chunk_start, s1, s2, x_T);
        if (double_v && tid >= 4 * H) {
            store_conv3_shift_64_seq<T>(cstate_base, chunk_start_hi, s1_hi, s2_hi, x_T_hi);
        }
    }

    // ---- z extraction ----
    if (tid >= 4 * H && !v_oob) {
        int v_tid = tid - 4 * H;
        if (double_v) {
            int v_hv = v_tid;
            int z_off = z_base + v_hv * gdn_V;

            simd<T, 64> z_lo = block_load<T, 64>(qkvz_row + z_off);
            simd<T, 64> z_hi = block_load<T, 64>(qkvz_row + z_off + 64);

            T* z_dst = z_out_ptr + (int64_t)seq_idx * HV * gdn_V
                        + (int64_t)v_hv * gdn_V;
            block_store<T, 64>(z_dst, z_lo);
            block_store<T, 64>(z_dst + 64, z_hi);
        } else {
            int v_hv = v_tid / 2;
            int v_half = v_tid & 1;
            int z_qkvz_offset = z_base + v_hv * gdn_V + v_half * 64;

            simd<T, 64> z_data = block_load<T, 64>(qkvz_row + z_qkvz_offset);

            T* z_dst = z_out_ptr + (int64_t)seq_idx * HV * gdn_V
                        + (int64_t)v_hv * gdn_V + v_half * 64;
            block_store<T, 64>(z_dst, z_data);
        }
    }
}

/* ============================================================
 * KERNEL VARIANT for large H (Qwen3.5-35B-A3B at TP=1: H=16, HV=32).
 * Same conv_state layout as the main kernel; only computes conv1d
 * for q[i_h], k[i_h], v[hv].
 * ============================================================ */
template<typename T, typename TS>
ESIMD_INLINE void gdn_conv_fused_seq_kernel_large_h(
    const T* __restrict__ qkvz_ptr,
    int64_t qkvz_stride0,
    T* __restrict__ conv_state_ptr,
    const T* __restrict__ conv_weight_ptr,
    const T* __restrict__ conv_bias_ptr,
    const int* __restrict__ conv_state_indices_ptr,
    const T* __restrict__ A_log_ptr,
    const T* __restrict__ dt_bias_ptr,
    const T* __restrict__ ba_ptr,
    int64_t ba_stride0,
    TS* __restrict__ ssm_state_ptr,
    const int* __restrict__ ssm_state_indices_ptr,
    T* __restrict__ output_ptr,
    T* __restrict__ z_out_ptr,
    int N, int H, int HV, int gdn_K, int gdn_V,
    float attn_scale, int64_t conv_stride0, int64_t ssm_stride0,
    nd_item<3>& ndi)
{
    slm_init<2048>();

    constexpr int WG_SIZE = 64;
    const int seq_idx = ndi.get_group(0);
    const int hv = ndi.get_group(1);
    const int tid = ndi.get_local_id(2);

    const int heads_per_group = HV / H;
    const int i_h = hv / heads_per_group;

    const int conv_idx = conv_state_indices_ptr[seq_idx];
    const int ssm_idx = ssm_state_indices_ptr[seq_idx];

    const int q_base = 0;
    const int k_base = H * gdn_K;
    const int v_base = 2 * H * gdn_K;
    const int z_base = v_base + HV * gdn_V;

    const T* qkvz_row = qkvz_ptr + (int64_t)seq_idx * qkvz_stride0;
    T* cstate_base = conv_state_ptr + (int64_t)conv_idx * conv_stride0;

    // ---- Phase 1: Conv1d for q[i_h], k[i_h], v[hv] only ----
    simd<float, 64> conv_result(0.0f);

    if (tid < 6) {
        int chunk_start;
        if (tid < 2) {
            chunk_start = q_base + i_h * gdn_K + (tid & 1) * 64;
        } else if (tid < 4) {
            chunk_start = k_base + i_h * gdn_K + (tid & 1) * 64;
        } else {
            chunk_start = v_base + hv * gdn_V + (tid & 1) * 64;
        }

        simd<T, 64> x_T = block_load<T, 64>(qkvz_row + chunk_start);
        simd<float, 64> x_f32 = x_T;

        simd<float, 64> s0, s1, s2;
        load_conv3_chunk_64_seq<T>(cstate_base, chunk_start, s0, s1, s2);

        simd<T, 256> w_raw = block_load<T, 256>(conv_weight_ptr + (int64_t)chunk_start * 4);
        conv_result =
            s0 * w_raw.template select<64, 4>(0) + s1 * w_raw.template select<64, 4>(1) +
            s2 * w_raw.template select<64, 4>(2) + x_f32 * w_raw.template select<64, 4>(3) +
            (simd<float, 64>)block_load<T, 64>(conv_bias_ptr + chunk_start);

        simd<float, 64> exp_neg = sycl::ext::intel::esimd::exp(-conv_result);
        conv_result = conv_result / (1.0f + exp_neg);
    }

    // ---- Store q/k/v to SLM ----
    if (tid == 0) slm_block_store<float, 64>(SLM_Q_LO_SEQ, conv_result);
    if (tid == 1) slm_block_store<float, 64>(SLM_Q_HI_SEQ, conv_result);
    if (tid == 2) slm_block_store<float, 64>(SLM_K_LO_SEQ, conv_result);
    if (tid == 3) slm_block_store<float, 64>(SLM_K_HI_SEQ, conv_result);
    if (tid == 4) slm_block_store<float, 64>(SLM_V_SEQ, conv_result);
    if (tid == 5) slm_block_store<float, 64>(SLM_V_SEQ + 256, conv_result);

    barrier();

    // ---- Phase 2: GDN (all 64 threads, VPT=2) ----
    constexpr int VPT = 2;

    if (ssm_idx >= 0) {
        simd<float, 64> q_lo = slm_block_load<float, 64>(SLM_Q_LO_SEQ);
        simd<float, 64> q_hi = slm_block_load<float, 64>(SLM_Q_HI_SEQ);
        simd<float, 64> k_lo = slm_block_load<float, 64>(SLM_K_LO_SEQ);
        simd<float, 64> k_hi = slm_block_load<float, 64>(SLM_K_HI_SEQ);

        float q_inv = 1.0f / esimd_sqrtf_seq(gdn_dot128_seq(q_lo, q_hi, q_lo, q_hi) + 1e-6f);
        float k_inv = 1.0f / esimd_sqrtf_seq(gdn_dot128_seq(k_lo, k_hi, k_lo, k_hi) + 1e-6f);
        q_lo *= q_inv * attn_scale; q_hi *= q_inv * attn_scale;
        k_lo *= k_inv; k_hi *= k_inv;

        const int vi0 = tid * VPT;
        simd<float, VPT> v_f32 = slm_block_load<float, VPT>(SLM_V_SEQ + vi0 * (int)sizeof(float));

        const float A_log_val = gdn_load_T_scalar_seq<T>(A_log_ptr, hv);
        const float dt_bias_val = gdn_load_T_scalar_seq<T>(dt_bias_ptr, hv);
        const float neg_exp_A = -esimd_expf_seq(A_log_val);

        const int b_col = hv;
        const int a_col = HV + hv;
        float a_val = gdn_load_T_scalar_seq<T>(ba_ptr, (int64_t)seq_idx * ba_stride0 + a_col);
        float b_val = gdn_load_T_scalar_seq<T>(ba_ptr, (int64_t)seq_idx * ba_stride0 + b_col);
        float x_gate = a_val + dt_bias_val;
        float sp = (x_gate > 20.0f) ? x_gate : esimd_logf_seq(1.0f + esimd_expf_seq(x_gate));
        float g = neg_exp_A * sp;
        float exp_g = esimd_expf_seq(g);
        float beta = 1.0f / (1.0f + esimd_expf_seq(-b_val));

        TS* sstate_base = ssm_state_ptr +
            (int64_t)ssm_idx * ssm_stride0 + (int64_t)hv * gdn_V * gdn_K;

        TS* sr0 = sstate_base + (int64_t)(vi0 + 0) * gdn_K;
        TS* sr1 = sstate_base + (int64_t)(vi0 + 1) * gdn_K;

        simd<float, 64> h0_lo = lsc_load_state_64_seq<TS>(sr0);
        simd<float, 64> h0_hi = lsc_load_state_64_seq<TS>(sr0 + 64);
        simd<float, 64> h1_lo = lsc_load_state_64_seq<TS>(sr1);
        simd<float, 64> h1_hi = lsc_load_state_64_seq<TS>(sr1 + 64);

        h0_lo *= exp_g; h0_hi *= exp_g;
        h1_lo *= exp_g; h1_hi *= exp_g;

        float kv0 = gdn_dot128_seq(h0_lo, h0_hi, k_lo, k_hi);
        float kv1 = gdn_dot128_seq(h1_lo, h1_hi, k_lo, k_hi);

        float d0 = (v_f32[0] - kv0) * beta;
        float d1 = (v_f32[1] - kv1) * beta;

        h0_lo += d0 * k_lo; h0_hi += d0 * k_hi;
        h1_lo += d1 * k_lo; h1_hi += d1 * k_hi;

        simd<float, VPT> o_acc;
        o_acc[0] = gdn_dot128_seq(h0_lo, h0_hi, q_lo, q_hi);
        o_acc[1] = gdn_dot128_seq(h1_lo, h1_hi, q_lo, q_hi);

        lsc_store_state_64_seq<TS>(sr0, h0_lo);
        lsc_store_state_64_seq<TS>(sr0 + 64, h0_hi);
        lsc_store_state_64_seq<TS>(sr1, h1_lo);
        lsc_store_state_64_seq<TS>(sr1 + 64, h1_hi);

        T* out = output_ptr + (int64_t)seq_idx * HV * gdn_V + (int64_t)hv * gdn_V + vi0;
        block_store<T, VPT>(out, simd<T, VPT>(o_acc));
    } else {
        int vi0 = tid * VPT;
        T* out = output_ptr + (int64_t)seq_idx * HV * gdn_V + (int64_t)hv * gdn_V + vi0;
        block_store<T, VPT>(out, simd<T, VPT>(0.0f));
    }

    // ---- z extraction: tid 0-1 copy z[hv] lo/hi ----
    if (tid < 2) {
        int z_off = z_base + hv * gdn_V + tid * 64;
        simd<T, 64> z_data = block_load<T, 64>(qkvz_row + z_off);
        T* z_dst = z_out_ptr + (int64_t)seq_idx * HV * gdn_V + (int64_t)hv * gdn_V + tid * 64;
        block_store<T, 64>(z_dst, z_data);
    }
}

/* ============================================================
 * Multi-pass conv-state shift kernel for large H.
 * Reads new x from qkvz, plus the existing token rows from cstate
 * via 192-wide block_load, then writes back the shifted block.
 * ============================================================ */
template<typename T>
ESIMD_INLINE void conv_state_shift_seq_multipass_kernel(
    const T* __restrict__ qkvz_ptr,
    int64_t qkvz_stride0,
    T* __restrict__ conv_state_ptr,
    const int* __restrict__ conv_state_indices_ptr,
    int N, int H, int HV, int gdn_K, int gdn_V,
    int64_t conv_stride0,
    nd_item<3>& ndi)
{
    const int seq_idx = ndi.get_group(0);
    const int tid = ndi.get_local_id(2);

    const int conv_idx = conv_state_indices_ptr[seq_idx];
    if (conv_idx < 0) return;

    const int dim = 2 * H * gdn_K + HV * gdn_V;
    const int num_chunks = (dim + 63) / 64;

    const T* qkvz_row = qkvz_ptr + (int64_t)seq_idx * qkvz_stride0;
    T* cstate_base = conv_state_ptr + (int64_t)conv_idx * conv_stride0;

    for (int chunk_idx = tid; chunk_idx < num_chunks; chunk_idx += 64) {
        int offset = chunk_idx * 64;
        int remaining = dim - offset;
        if (remaining <= 0) break;

        simd<float, 64> s0, s1, s2;
        load_conv3_chunk_64_seq<T>(cstate_base, offset, s0, s1, s2);
        simd<T, 64> x_new = block_load<T, 64>(qkvz_row + offset);

        store_conv3_shift_64_seq<T>(cstate_base, offset, s1, s2, x_new);
    }
}

/* ============================================================
 * Standard conv-state shift kernel (one WG per seq, sized by H).
 * ============================================================ */
template<typename T, int WG_SIZE>
ESIMD_INLINE void conv_state_shift_seq_kernel(
    const T* __restrict__ qkvz_ptr,
    int64_t qkvz_stride0,
    T* __restrict__ conv_state_ptr,
    const int* __restrict__ conv_state_indices_ptr,
    int N, int H, int HV, int gdn_K, int gdn_V,
    int64_t conv_stride0,
    nd_item<3>& ndi)
{
    const int seq_idx = ndi.get_group(0);
    const int tid = ndi.get_local_id(2);

    const int conv_idx = conv_state_indices_ptr[seq_idx];
    if (conv_idx < 0) return;

    const int num_v_threads_s = WG_SIZE - 4 * H;
    const bool double_v = (HV > num_v_threads_s / 2);
    const int q_base = 0;
    const int k_base = H * gdn_K;
    const int v_base = 2 * H * gdn_K;

    int qkvz_offset = 0;
    int chunk_start = 0;
    int qkvz_offset_hi = 0;
    int chunk_start_hi = 0;

    if (tid < 2 * H) {
        int q_head = tid / 2;
        qkvz_offset = q_base + q_head * gdn_K + (tid & 1) * 64;
        chunk_start = qkvz_offset;
    } else if (tid < 4 * H) {
        int k_tid = tid - 2 * H;
        int k_head = k_tid / 2;
        qkvz_offset = k_base + k_head * gdn_K + (k_tid & 1) * 64;
        chunk_start = qkvz_offset;
    } else if (double_v) {
        int v_tid = tid - 4 * H;
        int v_hv = v_tid;
        qkvz_offset = v_base + v_hv * gdn_V;
        qkvz_offset_hi = qkvz_offset + 64;
        chunk_start = qkvz_offset;
        chunk_start_hi = chunk_start + 64;
    } else {
        int v_tid = tid - 4 * H;
        int v_hv = v_tid / 2;
        qkvz_offset = v_base + v_hv * gdn_V + (v_tid & 1) * 64;
        chunk_start = qkvz_offset;
    }

    const bool v_oob_s = (tid >= 4 * H) &&
        (double_v ? (tid - 4 * H >= HV) : ((tid - 4 * H) / 2 >= HV));
    if (v_oob_s) return;

    const T* qkvz_row = qkvz_ptr + (int64_t)seq_idx * qkvz_stride0;
    T* cstate_base = conv_state_ptr + (int64_t)conv_idx * conv_stride0;

    simd<float, 64> s0, s1, s2;
    load_conv3_chunk_64_seq<T>(cstate_base, chunk_start, s0, s1, s2);
    simd<T, 64> x_new = block_load<T, 64>(qkvz_row + qkvz_offset);
    store_conv3_shift_64_seq<T>(cstate_base, chunk_start, s1, s2, x_new);

    if (double_v && tid >= 4 * H) {
        simd<float, 64> s0_h, s1_h, s2_h;
        load_conv3_chunk_64_seq<T>(cstate_base, chunk_start_hi, s0_h, s1_h, s2_h);
        simd<T, 64> x_hi = block_load<T, 64>(qkvz_row + qkvz_offset_hi);
        store_conv3_shift_64_seq<T>(cstate_base, chunk_start_hi, s1_h, s2_h, x_hi);
    }
}

/* ============================================================
 * Templated dispatch helper.
 * ============================================================ */
template<typename T, typename TS, int WG_SIZE>
inline void gdn_conv_fused_seq_dispatch(
    const T* qkvz_ptr, int64_t qkvz_stride0,
    T* conv_state_ptr, const T* conv_weight_ptr,
    const T* conv_bias_ptr, const int* conv_state_indices_ptr,
    const T* A_log_ptr, const T* dt_bias_ptr,
    const T* ba_ptr, int64_t ba_stride0,
    TS* ssm_state_ptr, const int* ssm_state_indices_ptr,
    T* output_ptr, T* z_out_ptr,
    int N, int H, int HV, int K, int V, float scale,
    int64_t conv_stride0, int64_t ssm_stride0,
    sycl::queue& q)
{
    const int total_wgs = N * HV;
    const int inline_shift = (total_wgs <= WG_SIZE) ? 1 : 0;

    sycl::nd_range<3> Range(
        sycl::range<3>(N, HV, WG_SIZE),
        sycl::range<3>(1, 1, WG_SIZE));

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL {
            gdn_conv_fused_seq_kernel<T, TS, WG_SIZE>(
                qkvz_ptr, qkvz_stride0, conv_state_ptr,
                conv_weight_ptr, conv_bias_ptr, conv_state_indices_ptr,
                A_log_ptr, dt_bias_ptr, ba_ptr, ba_stride0,
                ssm_state_ptr, ssm_state_indices_ptr,
                output_ptr, z_out_ptr,
                N, H, HV, K, V, scale, conv_stride0, ssm_stride0,
                inline_shift, ndi);
        });
    });

    if (!inline_shift) {
        sycl::nd_range<3> ShiftRange(
            sycl::range<3>(N, 1, WG_SIZE),
            sycl::range<3>(1, 1, WG_SIZE));

        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(ShiftRange, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL {
                conv_state_shift_seq_kernel<T, WG_SIZE>(
                    qkvz_ptr, qkvz_stride0, conv_state_ptr,
                    conv_state_indices_ptr,
                    N, H, HV, K, V,
                    conv_stride0, ndi);
            });
        });
    }
}

/* ============================================================
 * Host Dispatcher — selects WG_SIZE=32, 64, or large-H path.
 * Templated on activation dtype T and state dtype TS independently.
 * ============================================================ */
template<typename T, typename TS>
inline void gdn_conv_fused_seq_host(
    const T* qkvz_ptr,
    int64_t qkvz_stride0,
    T* conv_state_ptr,
    const T* conv_weight_ptr,
    const T* conv_bias_ptr,
    const int* conv_state_indices_ptr,
    const T* A_log_ptr,
    const T* dt_bias_ptr,
    const T* ba_ptr,
    int64_t ba_stride0,
    TS* ssm_state_ptr,
    const int* ssm_state_indices_ptr,
    T* output_ptr,
    T* z_out_ptr,
    int N, int H, int HV, int K, int V,
    float scale,
    int64_t conv_stride0,
    int64_t ssm_stride0,
    sycl::queue& q)
{
    TORCH_CHECK(HV > 0 && HV % H == 0,
        "gdn_conv_fused_seq: HV (", HV, ") must be a positive multiple of H (", H, ")");
    TORCH_CHECK(K == 128 && V == 128,
        "gdn_conv_fused_seq: only K=128, V=128 supported, got K=", K, " V=", V);

    const int v_slots_32 = 32 - 4 * H;
    if (v_slots_32 > 0 && HV <= v_slots_32) {
        gdn_conv_fused_seq_dispatch<T, TS, 32>(
            qkvz_ptr, qkvz_stride0, conv_state_ptr,
            conv_weight_ptr, conv_bias_ptr, conv_state_indices_ptr,
            A_log_ptr, dt_bias_ptr, ba_ptr, ba_stride0,
            ssm_state_ptr, ssm_state_indices_ptr,
            output_ptr, z_out_ptr,
            N, H, HV, K, V, scale, conv_stride0, ssm_stride0, q);
    } else {
        const int v_slots_64 = 64 - 4 * H;
        if (v_slots_64 > 0 && HV <= v_slots_64) {
            gdn_conv_fused_seq_dispatch<T, TS, 64>(
                qkvz_ptr, qkvz_stride0, conv_state_ptr,
                conv_weight_ptr, conv_bias_ptr, conv_state_indices_ptr,
                A_log_ptr, dt_bias_ptr, ba_ptr, ba_stride0,
                ssm_state_ptr, ssm_state_indices_ptr,
                output_ptr, z_out_ptr,
                N, H, HV, K, V, scale, conv_stride0, ssm_stride0, q);
        } else {
            // Large-H path (Qwen3.5-35B-A3B at TP=1: H=16, HV=32)
            constexpr int WG = 64;
            TORCH_CHECK(V <= WG * 2,
                "gdn_conv_fused_seq: V (", V, ") too large for large-H kernel (max ", WG * 2, ")");

            sycl::nd_range<3> Range(
                sycl::range<3>(N, HV, WG),
                sycl::range<3>(1, 1, WG));

            q.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(Range, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL {
                    gdn_conv_fused_seq_kernel_large_h<T, TS>(
                        qkvz_ptr, qkvz_stride0, conv_state_ptr,
                        conv_weight_ptr, conv_bias_ptr, conv_state_indices_ptr,
                        A_log_ptr, dt_bias_ptr, ba_ptr, ba_stride0,
                        ssm_state_ptr, ssm_state_indices_ptr,
                        output_ptr, z_out_ptr,
                        N, H, HV, K, V, scale, conv_stride0, ssm_stride0,
                        ndi);
                });
            });

            sycl::nd_range<3> ShiftRange(
                sycl::range<3>(N, 1, WG),
                sycl::range<3>(1, 1, WG));

            q.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(ShiftRange, [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL {
                    conv_state_shift_seq_multipass_kernel<T>(
                        qkvz_ptr, qkvz_stride0, conv_state_ptr,
                        conv_state_indices_ptr,
                        N, H, HV, K, V,
                        conv_stride0, ndi);
                });
            });
        }
    }
}
