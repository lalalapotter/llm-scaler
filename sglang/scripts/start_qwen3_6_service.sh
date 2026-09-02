#!/usr/bin/env bash
# Launch SGLang for Qwen3.6-35B-A3B, online fp8 (e5m2), on Intel BMG at TP=2.
#
# Weights are quantized on the fly to fp8-e5m2 and the full ESIMD decode/prefill
# fast-path set is enabled. XPU graph is off (accuracy); the fused decode
# kernels below recover the host-dispatch cost that a graph would have hidden.
#
# Usage (inside the container):
#   scripts/start_qwen3_6_service.sh
#   MODEL_PATH=/models/Qwen3.6-35B-A3B PORT=30000 scripts/start_qwen3_6_service.sh
#
# See scripts/qwen3_6_common.sh for the shared device/mamba/ESIMD settings and
# for the full list of overridable variables.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./qwen3_6_common.sh
source "${SCRIPT_DIR}/qwen3_6_common.sh"

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-35B-A3B}"
# Static memory pool (weights + KV). Lower this if the cards are shared with
# another tenant, or if the KV pool needs to grow for longer contexts.
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.9}"

# --- online fp8 dtype ---------------------------------------------------
# The fused MoE-full decode kernel and the fp8 MoE router path below are
# implemented for e5m2 only; they silently fall back on e4m3.
export SGLANG_FP8_DTYPE="${SGLANG_FP8_DTYPE:-e5m2}"

# --- fp8-only ESIMD fast-paths ------------------------------------------
# These read fp8 weight/scale layouts, so they only fire on this path. The
# GGUF service enables its own SGL_XPU_GGUF_* equivalents instead.

# FP8 MoE silu routed kernel (replaces the triton fused_moe on XPU).
export SGL_XPU_ESIMD_MOE="${SGL_XPU_ESIMD_MOE:-1}"
# Full decode MoE fusion: router topk + routed + shared + gate -> 1 dispatch.
# Reads the native N-major w13 directly, so it needs no transposed weight copy
# and no extra device memory. This is the main decode TPOT lever.
export SGL_XPU_ESIMD_MOE_FULL="${SGL_XPU_ESIMD_MOE_FULL:-1}"
# FP8 MoE prefill (M-tiled DPAS).
export SGL_XPU_ESIMD_MOE_PREFILL="${SGL_XPU_ESIMD_MOE_PREFILL:-1}"
# GDN gated-RMSNorm as an ESIMD GEMV (decode).
export SGL_XPU_GDN_NORM_GEMV="${SGL_XPU_GDN_NORM_GEMV:-1}"
# Fuse GDN input_layernorm (resadd+rmsnorm) + in_proj (qkvz+ba) into one GEMV.
export SGL_XPU_GDN_RESADD_NORM="${SGL_XPU_GDN_RESADD_NORM:-1}"
# Fuse full-attention input_layernorm (resadd+rmsnorm) into qkv_proj (decode).
export SGL_XPU_FA_RESADD_NORM="${SGL_XPU_FA_RESADD_NORM:-1}"
# Superseded by SGL_XPU_GDN_RESADD_NORM, which fuses in_proj qkvz+ba together
# WITH input_layernorm and is strictly more fused. The standalone in_proj
# fusion caught only a couple of fallback cases per step and gave no e2e gain,
# so it stays OFF.
export SGL_XPU_GDN_INPROJ_FUSED2="${SGL_XPU_GDN_INPROJ_FUSED2:-0}"
# MoE router as an fp8 ESIMD GEMV instead of an fp16 aten::mm, saving one host
# launch per MoE layer. Quantizing the gate perturbs top-8 routing on a small
# fraction of tokens -- A/B with run_gsm8k.py before trusting it, and set to 0
# to fall back to the accurate fp16 gate.
export SGL_XPU_MOE_ROUTER_FP8="${SGL_XPU_MOE_ROUTER_FP8:-1}"

# --load-format layered_fp8: build on CPU, load the full bf16 checkpoint into
# host RAM, then move + quantize one module at a time onto the device. Peak
# device memory is the fp8 weights plus a single module's bf16, so a TP=2 split
# across two cards fits where the default loader would OOM on the full bf16.
# shellcheck disable=SC2086
exec python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --tp "${TP_SIZE}" \
    --dtype float16 \
    --quantization fp8 \
    --load-format layered_fp8 \
    --attention-backend intel_xpu \
    --trust-remote-code \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}" \
    --page-size "${PAGE_SIZE}" \
    --mamba-scheduler-strategy extra_buffer \
    --reasoning-parser qwen3 \
    --tool-call-parser "${TOOL_CALL_PARSER}" \
    --enable-cache-report \
    --enable-metrics \
    --host "${HOST}" \
    --port "${PORT}" \
    ${EXTRA_ARGS}
