// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <type_traits>

#include "../arch/common.cuh"
#include "../common/lse.cuh"

namespace flashinfer::sparse_mla_sm120 {

template <int NUM_HEADS, int D_V_VAL, int BLOCK_THREADS, int DIMS_PER_THREAD,
          bool PRECOMPUTE_WEIGHTS = false>
static __global__ void __launch_bounds__(BLOCK_THREADS, 8)
    sparse_mla_decode_dsv4_merge_kernel(const bf16* __restrict__ mid_out,
                                        const float* __restrict__ mid_lse,
                                        bf16* __restrict__ output, float* __restrict__ out_lse,
                                        const float* __restrict__ attn_sink, int num_tokens,
                                        int num_splits, int num_heads, int mid_heads,
                                        size_t stride_out_lse, float lse_scale) {
  static_assert(BLOCK_THREADS % 32 == 0, "BLOCK_THREADS must be multiple of 32");
  static_assert(DIMS_PER_THREAD % 4 == 0, "DIMS_PER_THREAD must be multiple of 4");
  static_assert(BLOCK_THREADS * DIMS_PER_THREAD == D_V_VAL, "block must cover the full D_V row");
  constexpr int VEC_DIMS = DIMS_PER_THREAD < 8 ? 4 : 8;
  constexpr int VECS_PER_THREAD = DIMS_PER_THREAD / VEC_DIMS;
  using Packed = std::conditional_t<VEC_DIMS == 8, uint4, uint2>;
  const int q_heads = (NUM_HEADS == 0) ? num_heads : NUM_HEADS;
  const int m_heads = (NUM_HEADS == 0) ? mid_heads : NUM_HEADS;

  const int t_idx = blockIdx.x;
  const int h = blockIdx.y;
  if (t_idx >= num_tokens || h >= q_heads) return;
  const int tid = threadIdx.x;

  extern __shared__ __align__(16) unsigned char merge_smem_raw[];
  float* sm_lse = reinterpret_cast<float*>(merge_smem_raw);
  __shared__ float sm_gmax;
  __shared__ float sm_inv_gsum;
  __shared__ float sm_glse;

  const float* lse_ptr = mid_lse + (size_t)t_idx * m_heads * num_splits + (size_t)h * num_splits;

  for (int sp = tid; sp < num_splits; sp += BLOCK_THREADS) {
    sm_lse[sp] = lse_ptr[sp];
  }
  __syncthreads();

  if (tid < 32) {
    float local_max = -1e30f;
    for (int sp = tid; sp < num_splits; sp += 32) {
      local_max = fmaxf(local_max, sm_lse[sp]);
    }
#pragma unroll
    for (int s = 16; s >= 1; s >>= 1) {
      local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, s));
    }
    float gmax = (local_max > -1e29f) ? local_max : 0.f;
    float sink_log2 = 0.f;
    if (attn_sink != nullptr) {
      sink_log2 = __ldg(attn_sink + h) * LOG2E;
      gmax = local_max > -1e29f ? fmaxf(gmax, sink_log2) : sink_log2;
    }

    float local_sum = 0.f;
    for (int sp = tid; sp < num_splits; sp += 32) {
      float lse_sp = sm_lse[sp];
      if (lse_sp > -1e29f) local_sum += exp2f(lse_sp - gmax);
    }
#pragma unroll
    for (int s = 16; s >= 1; s >>= 1) {
      local_sum += __shfl_xor_sync(0xffffffff, local_sum, s);
    }
    if (tid == 0) {
      float total_sum = local_sum;
      if (attn_sink != nullptr) total_sum += exp2f(sink_log2 - gmax);
      sm_gmax = gmax;
      sm_inv_gsum = (total_sum > 0.f) ? (1.f / total_sum) : 0.f;
      sm_glse = (total_sum > 0.f) ? (log2f(total_sum) + gmax) : -INFINITY;
    }
    if constexpr (PRECOMPUTE_WEIGHTS) {
      // The reduction warp has consumed every LSE. Reuse the same shared
      // buffer for split weights before the existing publication barrier.
      for (int sp = tid; sp < num_splits; sp += 32) {
        const float lse_sp = sm_lse[sp];
        sm_lse[sp] = lse_sp > -1e29f ? exp2f(lse_sp - gmax) : -1.f;
      }
    }
  }
  __syncthreads();
  const float global_max = sm_gmax;
  const float inv_global_sum = sm_inv_gsum;

  const bf16* mid_base = mid_out + ((size_t)t_idx * m_heads + h) * (size_t)num_splits * D_V_VAL;
  bf16* out_ptr = output + ((size_t)t_idx * q_heads + h) * D_V_VAL;
  const int dim_base = tid * DIMS_PER_THREAD;

  float acc[DIMS_PER_THREAD];
#pragma unroll
  for (int d = 0; d < DIMS_PER_THREAD; d++) acc[d] = 0.f;

  for (int sp = 0; sp < num_splits; sp++) {
    float lse_sp = sm_lse[sp];
    if (PRECOMPUTE_WEIGHTS ? lse_sp < 0.f : lse_sp <= -1e29f) continue;
    const float weight = PRECOMPUTE_WEIGHTS ? lse_sp : exp2f(lse_sp - global_max);
    const bf16* row_base = mid_base + (size_t)sp * D_V_VAL + dim_base;
#pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; v++) {
      const Packed packed = *reinterpret_cast<const Packed*>(row_base + v * VEC_DIMS);
      const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
      for (int p = 0; p < VEC_DIMS / 2; p++) {
        const float2 f = __bfloat1622float2(pairs[p]);
        acc[v * VEC_DIMS + p * 2 + 0] += weight * f.x;
        acc[v * VEC_DIMS + p * 2 + 1] += weight * f.y;
      }
    }
  }

#pragma unroll
  for (int v = 0; v < VECS_PER_THREAD; v++) {
    Packed packed;
    __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
    for (int p = 0; p < VEC_DIMS / 2; p++) {
      pairs[p] = __floats2bfloat162_rn(acc[v * VEC_DIMS + p * 2 + 0] * inv_global_sum,
                                        acc[v * VEC_DIMS + p * 2 + 1] * inv_global_sum);
    }
    *reinterpret_cast<Packed*>(out_ptr + dim_base + v * VEC_DIMS) = packed;
  }
  if (out_lse != nullptr && tid == 0) {
    out_lse[(size_t)t_idx * stride_out_lse + h] = scale_output_lse(sm_glse, lse_scale);
  }
}

}  // namespace flashinfer::sparse_mla_sm120
