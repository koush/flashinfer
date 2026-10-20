# GLM.js fork

Based on upstream **`80a44189a`** (2026-10-20), on branch
`glm-upstream-20261020`. Required changes were manually ported to the new
SM120 execution/kernel layout. The previous fork remains on
`glm-upstream-20260911`, tip `217c4bcab`.

## Retained capabilities

| Previous commits | Port |
|---|---|
| `a664d0ff` | Split BF16 queries and prequantized FP8 NoPE queries with separate BF16 RoPE and FP32 scales, in decode and SG/MG/swapAB prefill. |
| `d4fa734f` | Dedicated one-/four-head V32 dispatch, bounded stores and true-head decode scratch. |
| `12440fb3`, `27fb57ba` | GLM H8 SG padded-head pruning and packed high/residual PV passes; candidate-major H8 QK and specialized H8 decode. |
| `db5f45cd`, `f87c9e6f` | PDL decode consumption: IO stages independent KV while math waits for Q. Launch and device-side wait are limited to grids of at most 64 blocks, matching the parent project's peer-progress constraint. |

The dense MLA context-parallel/custom-mask patch (`c807cd1c`) is retired.
GLM.js removed that inactive implementation before this migration. Its sparse
CP/cache/indexer operations are project-owned and remain in the parent.

The old direct shared-KV addressing patch (`217c4bca`) is superseded by upstream's
base-plus-offset shared layouts. `FLASHINFER_GLM_H8_PREFILL_DIRECT_KV` is retired;
the corresponding parent A/B tests were removed rather than testing a no-op.

## Native integration

The parent uses `execution::AttentionParams`, `AttentionMetadata` and
`resolve_attention()` to obtain an `ExecutionPlan`, then calls `dispatch_dsv32`,
`dispatch_decode` or `dispatch_prefill`. It retains its caller-owned split scratch
stride in `allocated_splits`, even when the planner selects a larger CPB.

`FLASHINFER_SPARSE_MLA_STANDALONE` builds the host resolver without its TVM exports.
Native validation uses framework-independent FlashInfer exceptions, and the TVM
candidate enumeration catches those exceptions. Decode dispatch returns CUDA
errors directly rather than depending on TVM. Python bindings and candidate
enumeration remain available in normal builds.

`AttentionParams` and `PrefillColdParams` have optional `q_rope_split` and
`q_scales` fields (null by default):

- Both null: upstream packed BF16 query layout.
- RoPE supplied: contiguous BF16 NoPE `[tokens, heads, 512]` and BF16 RoPE
  `[tokens, heads, 64]`.
- Both supplied: E4M3 NoPE bytes `[tokens, heads, 512]`, BF16 RoPE, and FP32
  dequantization scales `[tokens, heads, 4]`, one per 128 values. The raw `q`
  pointer retains its BF16 type but addresses bytes in this mode.

GLM's supplied query scales are power-of-two. SG/MG/decode online quantizers
also use power-of-two scales; upstream swapAB's online GLM quantizer uses
arbitrary FP32 scales. Compare each route against attention on its actual
quantized input, rather than requiring bitwise equality between those quantizers.

`FLASHINFER_GLM_H8_NATIVE=0` selects generic H8 decode.
`FLASHINFER_GLM_H8_PREFILL_SWAP_QK=0` selects conventional H8 QK while retaining
padded-head pruning and packed PV. Defaults enable both specializations.
Recreate captured graphs after changing either variable.

H8 ports preserve upstream's zero-row handling for masked cache reads, including
stale indices beyond the per-row length, and upstream's named pipeline barriers.

### DSV4 eight-head decode

The DSV4 H8 decode specialization zeros discarded probability rows before
quantization and omits their probability-scale atomics. This covers the parent
DeepSeek V4.1 TP8 single-/dual-cache decode path, which uses `ModelType::DSV4`.
`FLASHINFER_DSV4_H8_PRUNE=0` restores the original kernel for A/B measurements;
pruning is enabled by default. Recreate CUDA graphs (or restart the executor)
after changing this variable.

Alternating same-process graph-replay microbenchmarks measured a 1.0–1.6%
decode-plus-merge latency reduction for H8, 1/4/8/16 query rows, and 128, 512,
or 128+512 candidates. H16 controls were unchanged. These are warm-replay
measurements, not cold-cache or full-model throughput results. All 24 H8/H16
cases had bitwise-identical output/LSE, including sinks, invalid indices,
partial lengths and empty rows; Compute Sanitizer memcheck reported zero
errors. The parent's TP1/TP2 native-wrapper reference tests also pass.

## Validation

RTX PRO 6000 Blackwell, SM120:

- Parent `npm run build:all` passes (CUDA, addon, TypeScript).
- 136 targeted Node tests pass: plans, prefix/CP copies, parallel MLA, sparse
  planner, DeepSeek device and paged execution.
- Two Qwen3 batch-prefill/decode smoke tests pass. The additional multi-step
  CUDA-graph test fails its step-zero token comparison on both this checkout
  and the pre-migration native library.
- 114 native Python checks pass across sparse attention, H8 variants, FP8
  queries, cache writes, shared MLA operations, standard/hybrid attention and
  DeepSeek attention. Eight failures reproduce on the pre-migration baseline:
  three H64 prefill/decode tolerance checks, four indexer-flow checks, and the
  H8/H16 bitwise-LSE assertion (maximum difference `9.54e-7`).
- 92 upstream tests pass: DSV3.2 decode/prefill including heads 1/4, runtime
  heads, sinks, padded rows, strided indices, stale padding and GLM arbitrary
  FP32 KV scales.
- 165 upstream policy, execution-descriptor and LSE-base tests pass. Two
  subprocess import tests initially lacked the parent environment's PyTorch;
  they pass after making that dependency visible to validation subprocesses.
- Compute Sanitizer memcheck: H8 FP8 decode/prefill with changed-input graph
  replay and one-/four-head prequantized queries, four tests, zero errors.
  A broader initial sanitizer run exceeded its timeout; the targeted run completed.

Upstream Python validation used a temporary Python 3.12 environment with
`nvidia-cutlass-dsl[cu13]==4.7.1` and `FLASHINFER_DISABLE_VERSION_CHECK=1`. The
parent environment's older CuTe DSL cannot import latest upstream FlashInfer.
Tests JIT-built the current checkout; they do not validate an older cubin package.

These checks do not establish full-model output or throughput parity.
