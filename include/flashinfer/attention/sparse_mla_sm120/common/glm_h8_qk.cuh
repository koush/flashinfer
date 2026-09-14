// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "../prefill_common.cuh"

namespace flashinfer::sparse_mla_sm120 {

// Four warps cover 64 candidates rather than eight warps computing padded
// sixteen-head tiles. The candidate-major score staging converts the swapped
// fragments back to SG's existing per-head softmax/PV ownership. It aliases W
// scratch before W is produced, preserving the KV pipeline and smem footprint.
template <int PAGE_BLOCK_SIZE>
__device__ __forceinline__ void glm_h8_prefill_qk(
    float* scores, float* maxima, const uint8_t* q, const float* qsc, const bf16* qr,
    const uint8_t* kv, const uint8_t* cache, const int32_t* indices,
    size_t stride_kv_block, int remaining, float sm_scale_log2e, int warp, int lane) {
  using KV = KVCacheTraits<ModelType::GLM_NSA>;
  const int gid = lane >> 2;
  const int tid = lane & 3;
  const int c0 = warp * 16 + gid;
  const int c1 = c0 + 8;
  const uint8_t* k0 = kv + c0 * KV::KV_SMEM_STRIDE;
  const uint8_t* k1 = kv + c1 * KV::KV_SMEM_STRIDE;
  const bf16* r0 = reinterpret_cast<const bf16*>(
      prefill_kv_entry_base<ModelType::GLM_NSA, PAGE_BLOCK_SIZE>(cache, indices[c0], stride_kv_block)
      + KV::KV_ROPE_GMEM_OFFSET);
  const bf16* r1 = reinterpret_cast<const bf16*>(
      prefill_kv_entry_base<ModelType::GLM_NSA, PAGE_BLOCK_SIZE>(cache, indices[c1], stride_kv_block)
      + KV::KV_ROPE_GMEM_OFFSET);
  uint32_t rope[4][4];
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
    const int d = ks * 16 + tid * 2;
    rope[ks][0] = *reinterpret_cast<const uint32_t*>(r0 + d);
    rope[ks][1] = *reinterpret_cast<const uint32_t*>(r1 + d);
    rope[ks][2] = *reinterpret_cast<const uint32_t*>(r0 + d + 8);
    rope[ks][3] = *reinterpret_cast<const uint32_t*>(r1 + d + 8);
  }
  float score[4] = {};
#pragma unroll
  for (int blk = 0; blk < KV::NUM_SCALES; ++blk) {
    const uint8_t sfb = fp32_to_ue8m0(qsc[gid * KV::NUM_SCALES + blk]);
    float acc[4] = {};
#pragma unroll
    for (int ks = 0; ks < KV::QUANT_TILE / 32; ++ks) {
      const int ko = blk * KV::QUANT_TILE + ks * 32;
      uint32_t a0, a1, a2, a3, b0, b1;
      ldmatrix_load_A_fp8(a0, a1, a2, a3, kv + warp * 16 * KV::KV_SMEM_STRIDE + ko,
                         KV::KV_SMEM_STRIDE, lane);
      ldmatrix_load_B_fp8(b0, b1, q + ko, KV::Q_NOPE_STRIDE, lane);
      auto r = mma_fp8_block_scaled_m16n8k32(a0, a1, a2, a3, b0, b1,
                                            acc[0], acc[1], acc[2], acc[3], 127, sfb);
      acc[0] = r.d0;
      acc[1] = r.d1;
      acc[2] = r.d2;
      acc[3] = r.d3;
    }
    const float sc0 = reinterpret_cast<const float*>(k0 + KV::D_NOPE)[blk];
    const float sc1 = reinterpret_cast<const float*>(k1 + KV::D_NOPE)[blk];
    score[0] += acc[0] * sc0;
    score[1] += acc[1] * sc0;
    score[2] += acc[2] * sc1;
    score[3] += acc[3] * sc1;
  }
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
    const bf16* p = qr + gid * KV::D_ROPE + ks * 16 + tid * 2;
    uint32_t b0 = *reinterpret_cast<const uint32_t*>(p);
    uint32_t b1 = *reinterpret_cast<const uint32_t*>(p + 8);
    auto r = mma_bf16_m16n8k16(rope[ks][0], rope[ks][1], rope[ks][2], rope[ks][3],
                              b0, b1, score[0], score[1], score[2], score[3]);
    score[0] = r.d0;
    score[1] = r.d1;
    score[2] = r.d2;
    score[3] = r.d3;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int c = i < 2 ? c0 : c1;
    score[i] = (c < remaining && indices[c] >= 0 ? score[i] : -1e30f) * sm_scale_log2e;
  }
  float lm0 = fmaxf(score[0], score[2]);
  float lm1 = fmaxf(score[1], score[3]);
#pragma unroll
  for (int s = 16; s >= 4; s >>= 1) {
    lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, s));
    lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, s));
  }
  if (gid == 0) {
    maxima[warp * HPB + tid * 2] = lm0;
    maxima[warp * HPB + tid * 2 + 1] = lm1;
  }
  scores[c0 * 8 + tid * 2] = score[0];
  scores[c0 * 8 + tid * 2 + 1] = score[1];
  scores[c1 * 8 + tid * 2] = score[2];
  scores[c1 * 8 + tid * 2 + 1] = score[3];
}

}  // namespace flashinfer::sparse_mla_sm120
