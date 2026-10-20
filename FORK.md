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
| `db5f45cd`, `f87c9e6f` | PDL decode consumption: IO stages independent KV while math waits for Q. Launch and device-side wait are limited to grids of at most 64 blocks, matching the parent project's peer-progress constraint. Extended to `ModelType::DSV4` decode, counting allocated scratch splits in the grid bound. |

The DSV4 PDL extension enables programmatic stream serialization and waits in
the math warps immediately before Q staging; independent KV, index, and length
reads can overlap the preceding Q producer. Larger grids retain ordinary stream
ordering. Validation: parent `npm run build:all`, H8/H16 captured decode reference
checks at 1/4/8/16 rows, and memcheck with 64 allocated splits at 1/2 rows (64/128
blocks, exercising both sides of the limit) passed with zero memory errors.
Standalone attention replay does not measure producer/consumer overlap; the
end-to-end performance effect remains to be measured.

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

GLM H8 uses specialized decode and candidate-major prefill QK unconditionally,
including padded-head pruning and packed PV. Experimental environment overrides
have been removed; tests compare against FP32 attention and graph replay.

H8 ports preserve upstream's zero-row handling for masked cache reads, including
stale indices beyond the per-row length, and upstream's named pipeline barriers.

### DSV4 eight-head decode

The DSV4 H8 decode specialization zeros discarded probability rows before
quantization and omits their probability-scale atomics. This covers the parent
DeepSeek V4.1 TP8 single-/dual-cache decode path, which uses `ModelType::DSV4`.
Pruning is unconditional for DSV4 H8; its experimental environment override
has been removed.

Alternating same-process graph-replay microbenchmarks measured a 1.0–1.6%
decode-plus-merge latency reduction for H8, 1/4/8/16 query rows, and 128, 512,
or 128+512 candidates. H16 controls were unchanged. These are warm-replay
measurements, not cold-cache or full-model throughput results. All 24 H8/H16
cases had bitwise-identical output/LSE, including sinks, invalid indices,
partial lengths and empty rows; Compute Sanitizer memcheck reported zero
errors. The parent's TP1/TP2 native-wrapper reference tests also pass.

H8 decode uses four math warps plus one IO warp while
retaining the 64-candidate tile and split geometry. Each math warp computes
two QK candidate tiles. The execution planner supplies the specialized launch
dimensions and shared-memory size. The experimental four-warp environment
override has been removed. Before removing the A/B switches, alternating warm graph replays
measured approximately 3.0–3.7% lower H8 decode-plus-merge latency, with H16
controls unchanged. Reduction order changes: maximum observed output delta
was 0.000244141, and LSE comparisons passed with absolute tolerance 2e-6.
Both variants passed an independent FP32 attention reference (output atol
0.005, rtol 0.05), including sinks, masking and single/dual caches. All 24
H8/H16 cases passed eager-versus-graph checks; TP1/TP2 wrapper tests passed.
Targeted memcheck covered six H8 cases (1/16 rows, 128/512/128+512 candidates)
with zero errors after the broader sanitizer run exceeded its time limit.

Rejected experiments, reverted before the four-warp trial:
- Four-lane maximum reduction before scale atomics: bitwise-correct but
  2.7–3.4% slower than the probability-pruned baseline.
- Removing padded-head softmax shared stores/loads: bitwise-correct but
  0.3–1.0% slower than the probability-pruned baseline.
- Two math warps with the 64-candidate tile: reference checks passed but
  roughly 10–12% slower than the committed four-warp baseline.
- Four warps with a 32-candidate tile and doubled chunks per split: rejected
  after exceeding the existing A/B output tolerance (max delta 0.00268555
  in the first 128-candidate case).

### DSV4 H8 split-merge weights

For H8 with at least eight allocated splits, the merge reduction warp can
precompute each split weight in the existing shared LSE buffer. Output threads
reuse those weights instead of redundantly evaluating exponentials. Negative
weights mark invalid splits; valid weights that underflow to zero retain the
original load/accumulation behavior. No additional barrier or allocation is needed.
This is the normal H8 merge dispatch for at least eight allocated splits;
the experimental environment override has been removed.

Warm graph-replay measurements show approximately 7.1–7.4% lower total
decode-plus-merge latency for 512 and 128+512 candidates (1/4/8/16 query rows).
The two-split 128-candidate case retains the original merge; H16 is unchanged.
All 24 benchmark cases have bitwise-identical output/LSE, pass independent FP32
reference checks and eager/graph checks. Six additional H8 cases with 64 allocated
splits (including inactive splits) pass the same checks and Compute Sanitizer
memcheck with zero errors. TP1/TP2 native-wrapper tests pass.

## Validation

### Candidate-major DSV4 H8 QK

The H8 decode QK path computes a 16-candidate by 8-head MMA
tile instead of two padded 16-head by 8-candidate tiles. It swaps Q/K data
and UE8M0 scale operands, includes rotary QK, then uses warp shuffles to
restore the existing head-major softmax/PV register layout.
This is the normal DSV4 H8 decode path; the experimental environment override
has been removed.

An uncontended alternating graph-replay sweep measured 1.8–2.4% lower
H8 decode-plus-merge latency for 1/4/8/16 rows and 128/512/128+512 candidates.
H16 controls were unchanged. All 24 cases had bitwise-identical output/LSE
and passed independent FP32 attention and eager/graph checks. TP1/TP2 native
wrapper tests passed. Six further H8 cases with 64 allocated splits passed
Compute Sanitizer memcheck with zero errors. An earlier timing sweep was
discarded because another workload was active on the GPUs.

### 128-thread DSV4 H8 merge

The H8 merge uses 128 threads and four BF16 output dimensions
per thread (uint2 loads/stores), compared with 64 threads and eight dimensions
(uint4). This is the normal DSV4 H8 dispatch; the experimental environment
override has been removed.

Alternating warm graph replay measured 0.6–1.4% lower total decode-plus-merge
latency across the 12 H8 cases; H16 controls were flat. All 24 cases passed
bitwise output/LSE comparisons, independent FP32 reference and eager/graph
checks. Six additional cases with 64 allocated splits passed those checks
and Compute Sanitizer memcheck with zero errors. TP1/TP2 wrapper tests passed.

Rejected and reverted during this round (all bitwise-correct):
- H8 value-major PV with a register transpose each step: 2.6–3.2% slower.
- H8 value-major PV retaining transposed accumulators through scalar output
  stores: 2.7–7.8% slower.
- 32-thread merge: about 1% slower for two splits and 10–11% slower for
  eight/ten splits.

### Compact DSV4 H8 query staging

Candidate-major H8 QK only reads eight query heads. The compact staging
path quantizes/copies those eight heads without initializing the padded
eight heads. Shared layout and synchronization are unchanged. A compile-time
guard requires DSV4 H8 with candidate-major QK; other callers retain full staging.
This is the normal DSV4 H8 query staging path; the experimental environment
override has been removed.

Two alternating warm replay sweeps found a small repeatable gain: about
0.35–0.55% in most H8 cases, with the one-query/128-candidate case near 0.1%.
H16 controls were flat. All 24 cases passed bitwise output/LSE, FP32 reference,
and eager/graph comparisons. TP1/TP2 wrapper tests and all 15 GLM H8 tests pass.
Six H8 cases with 64 scratch splits passed memcheck with zero errors.

The preceding two-chunks-per-block experiment passed tolerance/reference
checks but was 30–36% slower and was reverted.

### DSV4 H8 prefill probability pruning

MG prefill zeros probabilities for padded H8 head rows and
skips their V-scale maximum atomics, for both single and dual caches.
This is the normal DSV4 H8 prefill path; the experimental override is removed.

Alternating warm graph replay measured 1.7–2.8% lower H8 prefill latency for
128/512 query rows and 128/512/128+512 candidates. H16 controls were flat.
All 12 benchmark cases had bitwise-identical output/LSE and passed independent
FP32 reference and eager/graph checks, with sinks, holes, partial lengths and
empty rows. Twelve further full-length cases (4/128 query rows, H8/H16) also
passed. Six forced-prefill H8 cases under memcheck reported zero errors.
TP1/TP2 native-wrapper tests and all 15 GLM H8 tests passed.

### Prefill query staging and QK experiments

DSV4 H8 BF16 prefill stages Q with uint4 vector loads/stores while retaining
zero-initialized padded heads. Short 128-candidate prefill improved by 1–3.6%
in alternating graph replay; dual-cache gains were small (up to 0.6%).
FP8 staging retains its original implementation. All 12 masked benchmark
cases and 12 full-length cases passed bitwise output/LSE and independent
reference/graph checks. Three forced-prefill cases passed memcheck with zero errors.
The experiment switch was removed before committing.

Rejected independently: candidate-major BF16/FP8 QK using four compute warps
and shared score redistribution was bitwise-correct but 0.4–4.1% slower.
Compact FP8 Q staging with explicit zero padding was flat to 0.4% slower.

### One-group DSV4 H8 prefill shared layout

DSV4 H8 MG prefill reserves one head group's Q, reductions, normalization,
and probability buffers rather than two. Launch/resource planning uses the
same layout. Shared memory falls from 97,936 to 79,696 bytes for BF16 QK
and 84,496 to 72,976 bytes for FP8 QK. This is a resource reduction; the
alternating 128/512-row benchmark was latency-neutral. All 12 masked and
12 full-length cases passed bitwise output/LSE, FP32 reference and graph
checks. Three H8 forced-prefill cases passed memcheck with zero errors.
The experimental switch is removed.

### Prefill warp/tile tuning result

A 32-candidate/four-math-warp variant was tested independently after the
one-group layout change, including adjusted synchronization, IO addressing,
and two rotary-output tiles per math warp. It exceeded the existing A/B
output tolerance (atol 0.002, rtol 0.025): 223 of 524,288 elements failed in
the first 128-row/128-candidate case; maximum absolute difference 0.00363159.
It was reverted rather than relaxing the numerical budget. The accepted
prefill path retains its 64-candidate/eight-math-warp geometry.

### Migration validation

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
