# shellcheck shell=bash
#
# Shared runtime configuration for the Qwen3.6-35B-A3B services on Intel BMG.
#
# Sourced by start_qwen3_6_service.sh (online fp8) and
# start_qwen3_6_gguf_service.sh (GGUF Q4_K_M). Only holds settings that are
# genuinely format-agnostic — the fp8-only and GGUF-only fusion gates live in
# their respective launch scripts, because enabling the wrong family silently
# does nothing (the fast-path simply never fires) and makes the config
# misleading to read.
#
# Every value is `${VAR:-default}` so it can be overridden from the
# environment without editing this file.

# --- device selection ---------------------------------------------------
# TP=2 maps onto the two cards left visible by the mask, in mask order. The
# default exposes the first two BMG cards; override to pick a different pair
# (e.g. ZE_AFFINITY_MASK=2,3 for the last two on a 4-card box).
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0,1}"

# --- mamba state pool dtype ---------------------------------------------
# The pool defaults to bf16, but these services run --dtype float16. The
# triton causal_conv1d_update kernel rejects a dtype mismatch against the
# activations ("Mismatched type for col0 (bf16 vs fp16)"), so pin both to
# fp16. Required, not an optimisation.
export SGLANG_MAMBA_CONV_DTYPE="${SGLANG_MAMBA_CONV_DTYPE:-float16}"
export SGLANG_MAMBA_SSM_DTYPE="${SGLANG_MAMBA_SSM_DTYPE:-float16}"

# --- format-agnostic ESIMD fast-paths -----------------------------------
# All ESIMD/XPU gates use the SGL_XPU_* prefix. The newer SGLANG_XPU_* spelling
# is NOT accepted by the GDN/FA gates and fails silently, so keep SGL_XPU_*.

# Decode SDPA, split-K over a flat NHD KV layout. Mandatory for online perf.
export SGL_XPU_ESIMD_DECODE="${SGL_XPU_ESIMD_DECODE:-1}"
# Full-attention fused QKV split + RMSNorm + RoPE (Qwen3.5/3.6 layout).
export SGL_XPU_FA_ESIMD_QKV="${SGL_XPU_FA_ESIMD_QKV:-1}"
# GDN conv fused_seq on the linear-attention decode path.
export SGL_XPU_GDN_ESIMD="${SGL_XPU_GDN_ESIMD:-1}"
# GDN chunk_gated_delta_rule prefill (extend), M-tiled ESIMD. This is the TTFT
# lever: the triton GDN recurrence is the prefill bottleneck. Accepts an fp16
# ssm-state so it matches this model's fp16 mamba pool.
export SGL_XPU_GDN_EXTEND_ESIMD="${SGL_XPU_GDN_EXTEND_ESIMD:-1}"
# Prefill SDPA via DPAS/XMX (AOT-compiled, doubleGRF).
export SGL_XPU_PREFILL_DPAS="${SGL_XPU_PREFILL_DPAS:-1}"

# --- XPU Graph ----------------------------------------------------------
# Kept OFF. Graph capture is not accuracy-stable on this model, so decode runs
# eager; the fused decode kernels below recover the per-step host-dispatch cost
# that a graph would otherwise have hidden.
export SGL_XPU_ENABLE_GRAPH="${SGL_XPU_ENABLE_GRAPH:-0}"

# --- TP token broadcast -------------------------------------------------
# MUST stay 1 on XPU with TP>1. Each rank runs greedy argmax on its own logits;
# tiny cross-rank numeric differences at near-ties make ranks disagree on the
# sampled token, hence on EOS/finish. Decode step counts then drift apart and
# the next TP collective deadlocks (both ranks block in the D2H of
# next_token_ids while the GPUs spin at full clock). The broadcast runs on the
# CPU/gloo group, so it costs nothing on the XPU queue. Setting this to 0 hung
# long BFCL runs.
export SGLANG_XPU_TP_SYNC_TOKENS="${SGLANG_XPU_TP_SYNC_TOKENS:-1}"

# --- shared server arguments --------------------------------------------
# --page-size 64 + --mamba-scheduler-strategy extra_buffer: hybrid GDN
# scheduler tuning that keeps the radix prefix-cache usable on this model, so
# the radix cache is left ENABLED for prefill reuse. (extra_buffer snapshots
# intermediate SSM state on a page granularity; the no_buffer strategy would
# need page_size=1, which the intel_xpu decode kernel rejects.)
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-2}"
PAGE_SIZE="${PAGE_SIZE:-64}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-64}"

# Tool-call parser. The bundled chat_template.jinja instructs the model to emit
# the XML form -- <tool_call><function=NAME><parameter=P>v</parameter>...  --
# and renders prior tool calls the same way. That is what `qwen3_coder` parses;
# the `qwen` parser expects Hermes-style JSON and will not match this template.
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"

# Extra flags appended verbatim to the launch command, e.g.
#   EXTRA_ARGS="--chunked-prefill-size 8192" ./start_qwen3_6_service.sh
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Note on oneCCL: the image symlinks /opt/venv/lib/libccl.so.1 to the bundled
# oneCCL 2021.15, which libtorch_xpu.so picks up through its DT_RPATH. No
# LD_PRELOAD is needed or wanted here (preloading it into every process breaks
# plain /bin/bash, whose loader cannot resolve the oneAPI deps in /opt/venv/lib).
