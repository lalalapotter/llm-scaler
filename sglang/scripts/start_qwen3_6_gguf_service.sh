#!/usr/bin/env bash
# Launch SGLang for Qwen3.6-35B-A3B GGUF (Q4_K_M) on Intel BMG at TP=2.
#
# Same ESIMD decode/prefill fast-paths as the fp8 service, but the MoE and
# norm/projection fusions are the GGUF-native ones: the attention and GDN
# projections run as ESIMD q8_0 GEMVs, so the fp8 resadd-norm fusions would
# never fire and are deliberately not set here.
#
# Usage (inside the container):
#   scripts/start_qwen3_6_gguf_service.sh
#   MODEL_DIR=/models/Qwen3.6-35B-A3B-GGUF scripts/start_qwen3_6_gguf_service.sh
#   GGUF_FILE=/models/.../Qwen3.6-35B-A3B-UD-Q4_K_M.gguf scripts/start_qwen3_6_gguf_service.sh
#
# See scripts/qwen3_6_common.sh for the shared device/mamba/ESIMD settings and
# for the full list of overridable variables.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./qwen3_6_common.sh
source "${SCRIPT_DIR}/qwen3_6_common.sh"

MODEL_DIR="${MODEL_DIR:-/models/Qwen3.6-35B-A3B-GGUF}"
# Static memory pool (weights + KV). Q4_K_M weights are smaller than the fp8
# build, but the default is kept conservative; raise it to grow the KV pool.
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.7}"

# The GGUF file itself. Defaults to the largest .gguf in MODEL_DIR, which picks
# the real weights over any stray shard or projector file.
if [[ -z "${GGUF_FILE:-}" ]]; then
    GGUF_FILE="$(ls -S "${MODEL_DIR}"/*.gguf 2>/dev/null | head -1 || true)"
fi
if [[ -z "${GGUF_FILE:-}" || ! -f "${GGUF_FILE}" ]]; then
    echo "ERROR: no .gguf found (MODEL_DIR=${MODEL_DIR})." >&2
    echo "       Expected e.g. Qwen3.6-35B-A3B-UD-Q4_K_M.gguf. Fetch it with:" >&2
    echo "         huggingface-cli download unsloth/Qwen3.6-35B-A3B-GGUF \\" >&2
    echo "           Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --local-dir ${MODEL_DIR}" >&2
    echo "       Or point GGUF_FILE at the file directly." >&2
    exit 1
fi
echo "[start_qwen3_6_gguf] using GGUF: ${GGUF_FILE}"

# --- HF config / tokenizer ----------------------------------------------
# The GGUF architecture id (qwen35moe) is not known to transformers. Pointing
# at the matching HF directory (config.json + model.safetensors.index.json)
# lets sglang read the config and weight mapping directly instead. The XPU
# qwen35 GGUF loader REQUIRES this and raises a RuntimeError without it. The
# same directory supplies the tokenizer and the chat template.
TOKENIZER_PATH="${TOKENIZER_PATH:-/models/Qwen3.6-35B-A3B}"
if [[ ! -d "${TOKENIZER_PATH}" ]]; then
    echo "ERROR: HF config/tokenizer dir not found: ${TOKENIZER_PATH}" >&2
    echo "       Set TOKENIZER_PATH to the matching Qwen3.6-35B-A3B HF dir." >&2
    exit 1
fi
export SGLANG_GGUF_HF_CONFIG_DIR="${SGLANG_GGUF_HF_CONFIG_DIR:-${TOKENIZER_PATH}}"

# --- GGUF-only ESIMD fast-paths -----------------------------------------
# Full GGUF MoE fusion: router topk + routed Q4_K/Q5_K experts + Q8_0 shared
# expert -> one dispatch. Supersedes SGL_XPU_GGUF_MOE_SHARED whenever it fires
# and falls back safely otherwise. This is the main decode TPOT lever here.
export SGL_XPU_GGUF_MOE_FULL="${SGL_XPU_GGUF_MOE_FULL:-1}"
# Q8_0 shared-expert kernel, used on the steps the full fusion declines.
export SGL_XPU_GGUF_MOE_SHARED="${SGL_XPU_GGUF_MOE_SHARED:-1}"
# Not set here, but worth knowing:
#   SGL_XPU_GGUF_RESADD_NORM   folds GemmaRMSNorm(input_layernorm) + the q8_0
#                              in_proj/qkv GEMV + the fp16 in_proj_ba GEMV into
#                              one op. Defaults ON whenever GGUF_MOE_FULL=1;
#                              set to 0 for the unfused fallback.
#   SGL_XPU_GGUF_FUSE_MAX_M    largest decode batch that fusion handles
#                              (default 64; 1 restores single-token-only).

# The fp8 MoE kernels are format-agnostic entry points that dispatch on the
# loaded weight type, so they stay enabled for parity with the fp8 service.
export SGL_XPU_ESIMD_MOE="${SGL_XPU_ESIMD_MOE:-1}"
export SGL_XPU_ESIMD_MOE_PREFILL="${SGL_XPU_ESIMD_MOE_PREFILL:-1}"

# shellcheck disable=SC2086
exec python3 -m sglang.launch_server \
    --model-path "${GGUF_FILE}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --tp "${TP_SIZE}" \
    --dtype float16 \
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
