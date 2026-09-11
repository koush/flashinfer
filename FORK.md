# GLM.js fork

This fork is based on upstream **`82090eed`** (FlashInfer 0.7.0 development,
2026-09-11 checkout). The previous fork tip was **`db38ddac`**, whose upstream
merge base was **`f212ec82`**. The reapplication is on branch
`glm-upstream-20260911`; the original `main` history is retained.

## Retained patches

| Commit | Capability |
|---|---|
| `c807cd1c` | Interleaved context-parallel KV append and causal masking; full and causal-suffix custom masks for FA2 MLA; Python support through upstream's new backend-based wrapper. |
| `a664d0ff` | Separate BF16 RoPE queries and prequantized FP8 NoPE queries with FP32 scales, ported to decode and SG/MG/swapAB prefill. |
| `d4fa734f` | One-/four-head SG dispatch and true-head decode scratch, with bounded output/LSE stores. |

### Paged MLA

- `MLAPlan` retains upstream's `staged_int_workspace_bytes` output. Callers
  must copy those bytes from the pinned workspace to the device workspace.
  The extra CP world-size/rank arguments follow the stream.
- The custom-mask batch lookup adds one field to `MLAPlanInfo`: serialized
  plans contain **19 int64 values**, rather than upstream's 18. Rebuild
  consumers and regenerate plans when changing between upstream and this fork.
- CP world size zero means inactive. Active CP maps local slot `j` on rank
  `r` to global position `j * world_size + r`; the plan retains global KV lengths.
- Python custom masks use the FA2 backend only. Boolean masks are flattened
  request-by-request. Packed masks use little-endian bits and independently
  byte-aligned requests. `causal_custom_mask_kv_len` can extend the suffix mask
  into the existing cache for MTP trees.

### SM120 query inputs

The framework-independent C++ launchers have overloads accepting two additional
arguments after the upstream argument list:

```cpp
const bf16* Q_rope_split, const float* Q_scales
```

For GLM_NSA:

- Both null: upstream packed BF16 Q, `[tokens, heads, 576]`.
- RoPE supplied, scales null: dense BF16 NoPE `[tokens, heads, 512]` and
  separate BF16 RoPE `[tokens, heads, 64]`.
- Both supplied: dense E4M3 NoPE values `[tokens, heads, 512]`, separate BF16
  RoPE, and FP32 dequantization scales `[tokens, heads, 4]` (128 elements per
  scale). The raw Q argument keeps its BF16 pointer type for compatibility,
  but points to FP8 bytes in this mode. No re-quantization is performed.

The original packed-BF16 overloads remain available to upstream's Python
bindings. Prefill variant selection, row strides, and decode scratch arguments
use the current upstream interfaces. Other model families retain their packed
input paths.

Dedicated decode head counts 1/4/8/16/32/64/128 use true-head scratch. Other
runtime-H counts use upstream's HPB-aligned scratch. The small-head additions
remain necessary for GLM.js's compact allocations; merely using upstream's
runtime-H fallback would change their layout.

## Superseded changes

- The fork's subgroup-based online Q quantizer is replaced by upstream's
  vectorized, register-based quantizer.
- H8 weight-pass packing and padded-head compute pruning were initially
  omitted during the reapplication. Both are now restored for GLM_NSA H8 SG
  prefill; see below. The compact-store bounds remain a separate change.
- The old split-Q MG repair is incorporated into the new shared Q-loading
  paths, including the newly introduced swapAB register path.
- Upstream's prefill handshake and swapAB spectator-warp fixes are inherited
  from the base; no separate backports are needed.

## Restored H8 SG specialization

`prefill_mg_kernel.cuh` specializes the SG kernel when `MT == GLM_NSA` and
`NUM_HEADS == 8`. The unused upper eight heads are excluded from softmax,
scale atomics/reductions, persistent accumulator updates and output staging.
The XV weight matrix instead stores the high FP8 approximation in rows 0–7
and its low residual in rows 8–15. One m16 MMA pass computes both contributions,
which are folded into the eight real output heads. The upper scale slots cache
reciprocals for those real heads.

The specialization retains upstream's alternating IO handshake and all
producer/consumer synchronization needed for shared-memory reuse. Decode,
MG, swapAB and other model families do not use this specialization.

Validation: 35 native query/sparse-MLA tests; nine upstream prefill cases,
including attention sinks; and a targeted H8 CUDA memcheck run with zero errors.
H8 coverage includes lengths 0/1/63/65/129/256/2048, an independent quantized
attention reference, comparison with the unoptimized H16 SG kernel, and fresh-Q
graph replay. Packed passes change FP32 accumulation order: the H16 comparison
checks output NRMSE below 0.003 and exact LSE equality. No throughput claim is
made from these correctness checks.

A subsequent live smoke test overlapped a separately running server benchmark
and encountered an illegal memory access reported during an MTP draft step.
The executor was restored and healthy. Live testing was stopped to avoid
interfering with the benchmark; the mixed-load failure has not been isolated
or attributed to this specialization.

## Numerical compatibility

Upstream swapAB's online BF16 GLM Q quantizer uses arbitrary FP32 scales.
GLM.js's prequantizer uses power-of-two scales. Those inputs represent slightly
different quantized queries, so bitwise BF16/prequantized output equivalence is
not expected for swapAB. Validate against attention on the **actual supplied
FP8 values and scales**, and test graph replay against an independently
quantized eager input. SG/MG/decode retain the power-of-two online quantizer.

This migration does not establish bitwise model-output or throughput parity.
The previous GLM.js chunk-policy experiment was removed; tuning is paused.

## Validation

On SM120 (RTX PRO 6000 Blackwell):

- GLM.js `npm run build:all`.
- 35 native MLA/packed-KV tests, plus 22 FP8 quantization/input tests covering
  1/4/8/16/32/64/128 heads, independent attention references, and graph replay.
- 37 Node tests covering CP prefill, append, custom MTP masks and parallel MLA.
- 315 upstream-wrapper/custom-mask tests, including ragged packed-mask offsets.
- 20 selected upstream SM120 tests: GLM arbitrary scales, runtime heads,
  padded rows, strided indices/LSE, and CUDA graphs.
- 42 upstream DSV3.2 decode cases including the expanded small-head matrix.
- Targeted Compute Sanitizer memcheck on six prequantized-query cases:
  decode heads 1/4/64, prefill heads 1/4/64, including 64-head graph replay.
  Zero reported errors. An initial unfiltered sanitizer run exceeded its
  timeout; the completed runs filtered instrumentation to sparse-MLA kernels.
- Resident GLM.js MTP server: 192-token decode, four concurrent 4,034-token
  prompts, correct arithmetic answers, and prefix reuse.
- Ruff, the repository's mypy hook, and changed-line clang-format checks.

The Python tests used the editable checkout with
`FLASHINFER_DISABLE_VERSION_CHECK=1` because the environment's separately
installed cubin package is older. These tests JIT-build the relevant kernels;
they do not validate that older cubin package.

The tests and native adapters in the parent GLM.js repository are part of the
integration contract. Preserve the resident model loader during validation;
stop/restart only its executor when rebuilding or running standalone GPU tests.
