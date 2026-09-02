# SGLang on Intel BMG

End-to-end recipe for running Qwen3.6-35B-A3B on Intel Battlemage (BMG) GPUs
with the optimized ESIMD kernel fast-paths, at TP=2.

Two weight formats are supported, each with its own launch script:

| Format | Script | Notes |
|--------|--------|-------|
| Online fp8 (e5m2) | `scripts/start_qwen3_6_service.sh` | Quantizes the bf16 checkpoint on load. Needs the HF checkpoint. |
| GGUF Q4_K_M | `scripts/start_qwen3_6_gguf_service.sh` | Pre-quantized file. Still needs the HF dir for config/tokenizer. |

Both run decode **eager** — XPU graph capture is not accuracy-stable on this
model, so it is kept off and the fused decode kernels recover the per-step
host-dispatch cost a graph would otherwise have hidden.

## What's in here

```
sglang/
├── docker/
│   └── Dockerfile                       # builds the full image
├── scripts/
│   ├── build_image.sh                   # wrapper around `docker buildx build`
│   ├── qwen3_6_common.sh                # shared device / mamba / ESIMD settings
│   ├── start_qwen3_6_service.sh         # online fp8 (e5m2) server
│   ├── start_qwen3_6_gguf_service.sh    # GGUF Q4_K_M server
│   ├── run_gemma4_26b_moe.sh            # Gemma4-26B-A4B TP=2, e4m3/e5m2
│   ├── run_gemma4_31b.sh                # Gemma4-31B TP=2 online FP8
│   ├── run_gsm8k.py                     # standalone GSM8K accuracy harness
│   └── bfcl/                            # BFCL function-calling eval kit
├── patches/                             # sglang / sgl-kernel-xpu source patches
└── custom-esimd-kernels/                # merged ESIMD kernel package:
                                         #   decode attn, fp8 GEMM, fp8 MoE (silu + prefill),
                                         #   GGUF MoE fusions, fused QKV,
                                         #   GDN conv fused_seq, RMSNormGated
```

## Build

```bash
llm-scaler/sglang/scripts/build_image.sh
```

The script resolves `docker/Dockerfile` relative to itself, forwards
`http_proxy` / `https_proxy` from the environment, and bumps
`SGLANG_CACHEBUST` each run. Override the tag with `IMAGE_TAG=...`.

Cold builds take a while, dominated by the ESIMD AOT compile and the
sgl-kernel-xpu cmake build.

## Run

### Qwen3.6-35B-A3B — online fp8 (e5m2)

```bash
docker run --rm -it \
    --device=/dev/dri \
    --shm-size=16g \
    -v /home/intel/LLM/models/Qwen3.6-35B-A3B:/models/Qwen3.6-35B-A3B:ro \
    -p 30000:30000 \
    llm-scaler-sgl:bmg \
    /llm-scaler/sglang/scripts/start_qwen3_6_service.sh
```

### GGUF Q4_K_M

The GGUF architecture id (`qwen35moe`) is unknown to transformers, so the
loader reads the config and weight mapping from a matching HF directory
instead. **Both** mounts are required — the XPU GGUF loader raises a
`RuntimeError` without the HF config dir.

```bash
docker run --rm -it \
    --device=/dev/dri \
    --shm-size=16g \
    -v /home/intel/LLM/models/Qwen3.6-35B-A3B-GGUF:/models/Qwen3.6-35B-A3B-GGUF:ro \
    -v /home/intel/LLM/models/Qwen3.6-35B-A3B:/models/Qwen3.6-35B-A3B:ro \
    -p 30000:30000 \
    llm-scaler-sgl:bmg \
    /llm-scaler/sglang/scripts/start_qwen3_6_gguf_service.sh
```

The script picks the largest `*.gguf` in `MODEL_DIR`; set `GGUF_FILE` to choose
explicitly.

### Gemma4-26B-A4B

```bash
docker run --rm -it \
    --device=/dev/dri \
    --shm-size=32g \
    -v /path/to/gemma-4-26B-A4B-it:/models/gemma-4-26B-A4B-it:ro \
    -p 30000:30000 \
    --entrypoint /llm-scaler/sglang/scripts/run_gemma4_26b_moe.sh \
    llm-scaler-sgl:bmg
```

### Gemma4-31B

```bash
docker run --rm -it \
    --device=/dev/dri \
    --shm-size=32g \
    -v /path/to/gemma-4-31B-it:/models/gemma-4-31B-it:ro \
    -p 30000:30000 \
    --entrypoint /llm-scaler/sglang/scripts/run_gemma4_31b.sh \
    llm-scaler-sgl:bmg
```


### Common overrides

Every setting is `${VAR:-default}` and can be overridden from the environment:

| Variable | Default | Meaning |
|----------|---------|---------|
| `ZE_AFFINITY_MASK` | `0,1` | Which two cards TP=2 maps onto, in mask order |
| `PORT` / `HOST` | `30000` / `0.0.0.0` | Listen address |
| `TP_SIZE` | `2` | Tensor-parallel size |
| `MEM_FRACTION_STATIC` | `0.9` fp8 / `0.7` GGUF | Static pool (weights + KV) |
| `MODEL_PATH` | `/models/Qwen3.6-35B-A3B` | fp8 checkpoint dir |
| `MODEL_DIR` / `GGUF_FILE` | `/models/Qwen3.6-35B-A3B-GGUF` | GGUF location |
| `TOKENIZER_PATH` | `/models/Qwen3.6-35B-A3B` | HF config/tokenizer dir (GGUF) |
| `TOOL_CALL_PARSER` | `qwen3_coder` | See "Tool calling" below |
| `PAGE_SIZE` | `64` | KV page size; must stay aligned with the mamba scheduler strategy |
| `MAX_MAMBA_CACHE_SIZE` | `64` | Concurrent mamba state slots (caps effective concurrency) |
| `EXTRA_ARGS` | *(empty)* | Extra flags appended verbatim to the launch command |

```bash
# last two cards of a 4-card box, alternate port, extra server flag
ZE_AFFINITY_MASK=2,3 PORT=30012 EXTRA_ARGS="--chunked-prefill-size 8192" \
    scripts/start_qwen3_6_service.sh
```

## Fast-paths enabled

All ESIMD/XPU gates use the **`SGL_XPU_*`** prefix. The newer `SGLANG_XPU_*`
spelling is *not* accepted by the GDN/FA gates and fails silently.

### Shared (`qwen3_6_common.sh`, both formats)

| Env var | Path |
|---------|------|
| `SGL_XPU_ESIMD_DECODE` | Decode SDPA (split-K, flat NHD KV) |
| `SGL_XPU_FA_ESIMD_QKV` | Full-attention fused QKV + RMSNorm + RoPE |
| `SGL_XPU_GDN_ESIMD` | GDN conv fused_seq decode |
| `SGL_XPU_GDN_EXTEND_ESIMD` | GDN chunk_gated_delta_rule prefill (TTFT lever) |
| `SGL_XPU_PREFILL_DPAS` | Prefill SDPA via DPAS/XMX |
| `SGL_XPU_ENABLE_GRAPH` | XPU graph capture/replay — kept **0** |

### fp8-only (`start_qwen3_6_service.sh`)

| Env var | Path |
|---------|------|
| `SGL_XPU_ESIMD_MOE` | FP8 MoE silu routed kernel |
| `SGL_XPU_ESIMD_MOE_FULL` | Full decode MoE fusion (router+routed+shared+gate), e5m2, native N-major w13 |
| `SGL_XPU_ESIMD_MOE_PREFILL` | FP8 MoE prefill (M-tiled DPAS) |
| `SGL_XPU_GDN_NORM_GEMV` | GDN gated-RMSNorm as ESIMD GEMV (decode) |
| `SGL_XPU_GDN_RESADD_NORM` | Fuse GDN input_layernorm + in_proj (qkvz+ba) into one GEMV |
| `SGL_XPU_FA_RESADD_NORM` | Fuse FA input_layernorm into qkv_proj (decode) |
| `SGL_XPU_MOE_ROUTER_FP8` | MoE router as fp8 ESIMD GEMV (vs fp16 `aten::mm`) |
| `SGL_XPU_GDN_INPROJ_FUSED2` | Superseded by `GDN_RESADD_NORM` — kept **0** |

`SGL_XPU_ESIMD_MOE_FULL` and `SGL_XPU_MOE_ROUTER_FP8` require the online fp8
dtype to be **e5m2** (`SGLANG_FP8_DTYPE=e5m2`, which the script sets); they
silently fall back on e4m3. The e5m2 fused MoE kernel reads the native N-major
`w13` weight directly, so it needs no transposed copy and no extra device
memory. `SGL_XPU_MOE_ROUTER_FP8=1` quantizes the gate and perturbs top-8
routing on a fraction of tokens — A/B it with `run_gsm8k.py` before trusting
it, and set to `0` for the accurate fp16 gate.

### GGUF-only (`start_qwen3_6_gguf_service.sh`)

| Env var | Path |
|---------|------|
| `SGL_XPU_GGUF_MOE_FULL` | Full GGUF MoE fusion (router topk + Q4_K/Q5_K routed + Q8_0 shared → 1 dispatch) |
| `SGL_XPU_GGUF_MOE_SHARED` | Q8_0 shared-expert kernel (used when the full fusion declines) |
| `SGL_XPU_GGUF_RESADD_NORM` | Folds input_layernorm + q8_0 in_proj/qkv GEMV + fp16 in_proj_ba into one op. Defaults **on** whenever `GGUF_MOE_FULL=1` |
| `SGL_XPU_GGUF_FUSE_MAX_M` | Largest decode batch the fusion handles (default `64`; `1` = single-token only) |

The fp8 resadd-norm fusions are deliberately **not** set on the GGUF path: its
attention and GDN projections run as ESIMD q8_0 GEMVs, so those gates would
never fire and would only make the config misleading.

### Required for both

`SGLANG_MAMBA_CONV_DTYPE` / `SGLANG_MAMBA_SSM_DTYPE` are pinned to `float16`.
The mamba state pool defaults to bf16, but these services run `--dtype
float16`, and the triton `causal_conv1d_update` kernel rejects the mismatch
("Mismatched type for col0 (bf16 vs fp16)"). This is a correctness
requirement, not a tuning knob.

`SGLANG_XPU_TP_SYNC_TOKENS` **must stay 1** on XPU with TP>1. Each rank runs
greedy argmax on its own logits, and tiny cross-rank numeric differences at
near-ties make ranks disagree on the sampled token, hence on EOS/finish. The
decode step counts then drift apart and the next TP collective deadlocks (both
ranks block in the D2H of `next_token_ids` while the GPUs spin at full clock).
The broadcast runs on the CPU/gloo group, so it costs nothing on the XPU queue.
Setting it to `0` hung long BFCL runs.

## Scheduler / cache settings

Both scripts use `--page-size 64` with `--mamba-scheduler-strategy
extra_buffer`, which keeps the radix prefix-cache usable on this hybrid GDN
model, so the radix cache stays **enabled** for prefill reuse. `extra_buffer`
snapshots intermediate SSM state at page granularity; the `no_buffer` strategy
would require `page_size=1`, which the `intel_xpu` decode kernel rejects.

> **Known issue.** On sequences that cross a mamba snapshot boundary, the mamba
> radix cache can return a hit that overruns the true common prefix, so decode
> reads state that does not belong to the request. Output then depends on cache
> state rather than on the prompt alone, which shows up as run-to-run variation
> in long multi-turn evals (most visible above ~8k-token prompts). A proper fix
> — bounding snapshot creation so no intermediate snapshot can be mismatched —
> is pending. Meanwhile `EXTRA_ARGS="--disable-radix-cache"` bypasses the
> affected path entirely, at the cost of all prefill reuse; use it when
> bit-exact reproducibility matters more than TTFT.

## Tool calling

The bundled `chat_template.jinja` instructs the model to emit the **XML** form:

```
<tool_call>
<function=NAME>
<parameter=P>value</parameter>
</function>
</tool_call>
```

and renders prior tool calls the same way. That is what the `qwen3_coder`
parser consumes, so it is the default here. The `qwen` parser expects
Hermes-style JSON and will **not** match this template — do not use it with
this chat template.

Note that BFCL is not a guide on this point: it calls `/v1/completions` and
injects its own system prompt asking for Hermes JSON, bypassing both the chat
template and the parser.

## Accuracy checks

### GSM8K

`scripts/run_gsm8k.py` is a standalone harness (stdlib only) that hits the
running server's OpenAI-compatible endpoint with full sampling-parameter
control, then reports accuracy and classifies failures
(correct / wrong_answer / empty_output / runaway_len / error).

```bash
# non-thinking chat, greedy, 200 questions (cleanest kernel-debug signal)
python3 scripts/run_gsm8k.py \
    --base-url http://localhost:30000 \
    --num-questions 200 \
    --no-thinking \
    --temperature 0

# thinking mode with the Qwen3-recommended sampling params
python3 scripts/run_gsm8k.py \
    --base-url http://localhost:30000 \
    --num-questions 200 \
    --thinking \
    --temperature 0.6 --top-p 0.95 --top-k 20 --repetition-penalty 1.05
```

Key flags: `--thinking/--no-thinking` (explicitly sets `enable_thinking`),
`--chat-stop/--no-chat-stop` (adds `Question:` stops to prevent fake-question
continuation), `--api chat|completion`, plus the full sampling set
(`--temperature --top-p --top-k --min-p --repetition-penalty
--frequency-penalty --presence-penalty --max-tokens`). Outputs
`<prefix>_examples.jsonl` and `<prefix>_summary.txt`.

### BFCL (function calling)

`scripts/bfcl/` holds the Berkeley Function-Call Leaderboard kit. The harness
(steps 02–05) only needs an OpenAI-compatible `/v1` endpoint, so it works
against either launch script — start the server yourself and run step 04 with
`SKIP_START=1`. See `scripts/bfcl/README.md`.

`scripts/bfcl/01_start_server.sh` is kept as-is: it is an older single-tile
(TP=1, XPU graph on, no MoE fusion) reference that the kit was validated
against, not a performance config. For throughput use the launch scripts above.

Use `--num-threads 1` for any A/B comparison: batching changes decode numerics,
so results at different concurrency levels are not comparable.
