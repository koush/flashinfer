// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "decode_dsv3_2_kernel.cuh"

namespace flashinfer::sparse_mla_sm120 {

// Native GLM TP8: QK uses candidates as MMA rows and eight heads as columns.
// PV packs the high/residual probability passes into the two eight-row halves
// of one MMA, then adds them. Both stages fill the m16n8 tile. The pipeline and
// split scratch contract match the generic decoder; no query packing is needed.
constexpr int GLM_H8_THREADS = 160;

template <int PAGE_BLOCK_SIZE>
__global__ void __launch_bounds__(GLM_H8_THREADS) sparse_mla_decode_glm_h8_kernel(
    const bf16* __restrict__ Q, const uint8_t* __restrict__ KV_cache,
    const int32_t* __restrict__ indices, bf16* __restrict__ mid_out,
    float* __restrict__ mid_lse, const int* __restrict__ topk_length_ptr,
    int num_tokens, int num_heads, int topk, int num_splits, int chunks_per_block,
    float sm_scale, size_t stride_kv_block, size_t stride_indices_token, int stride_kv_row,
    const bf16* Q_rope_split = nullptr, const float* Q_scales = nullptr) {
  constexpr auto MT = ModelType::GLM_NSA;
  using KV = KVCacheTraits<MT>;
  static_assert(KV::D_NOPE == 512 && KV::D_ROPE == 64 && KV::NUM_SCALES == 4 &&
                KV::KV_SMEM_STRIDE == 528 && KV::KV_ROPE_GMEM_OFFSET == 528 &&
                KV::SCALE_FORMAT == ScaleFormat::ARBITRARY_FP32);
  constexpr int MATH_THREADS = 128;
  const int t = blockIdx.x;
  const int split = blockIdx.z;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int gid = lane >> 2;
  const int tid = lane & 3;
  int length = topk_length_ptr ? topk_length_ptr[t] : topk;
  length = max(0, min(length, topk));
  const int lo = split * chunks_per_block;
  const int hi = min(lo + chunks_per_block, (length + 63) / 64);
  if (lo >= hi) {
    if (threadIdx.x < 8) {
      mid_lse[((size_t)t * 8 + threadIdx.x) * num_splits + split] = -1e30f;
    }
    return;
  }
  extern __shared__ __align__(16) char raw[];
  // Reuse the generic double-buffer layout and Q loader. Q preparation still
  // reserves sixteen rows, but QK reads only eight; PV uses all sixteen rows
  // of W for high/residual values. This remains one shared-memory-bound CTA/SM.
  auto sm = DecodeDsv3_2Smem<MT>::init(raw);
  __shared__ float gmax[8], gsum[8], alpha[8];
  if (threadIdx.x < 8) {
    gmax[threadIdx.x] = -1e30f;
    gsum[threadIdx.x] = 0.f;
  }
  if (threadIdx.x == 0) {
    for (int b = 0; b < 2; ++b) {
      mbarrier_init(sm.mbar_full(b), 1);
      mbarrier_init(sm.mbar_empty(b), 1);
    }
  }
  __syncthreads();
  const int32_t* idx = indices + (size_t)t * stride_indices_token;
  if (warp == 4) {
    for (int chunk = lo; chunk < hi; ++chunk) {
      const int b = (chunk - lo) & 1;
      mbarrier_wait_parity(sm.mbar_empty(b), 1 ^ (((chunk - lo) / 2) & 1));
      if (lane == 0) {
        mbarrier_arrive_expect_tx(sm.mbar_full(b), 64 * (528 + 128));
      }
#pragma unroll
      for (int c = lane; c < 64; c += 32) {
        const int pos = chunk * 64 + c;
        const int slot = pos < length ? max(idx[pos], 0) : 0;
        const uint8_t* src = KV_cache + (size_t)(slot / PAGE_BLOCK_SIZE) * stride_kv_block
                            + (size_t)(slot % PAGE_BLOCK_SIZE) * stride_kv_row;
        cp_async_bulk_g2s(sm.kv_fp8(b) + c * 528, src, 528, sm.mbar_full(b));
        cp_async_bulk_g2s(sm.kv_rope(b) + c * 64, src + 528, 128, sm.mbar_full(b));
      }
    }
    return;
  }
  const size_t qhead = (size_t)t * 8;
  quantize_q_to_smem<MT, MATH_THREADS>(
      sm.q_fp8(), sm.q_sc(), sm.q_rope(), query_head_ptr<MT>(Q, qhead, Q_rope_split, Q_scales),
      8, Q_rope_split ? Q_rope_split + qhead * 64 : nullptr,
      Q_scales ? Q_scales + qhead * 4 : nullptr);
  float acc[4][4][2] = {};
  const int c0 = warp * 16 + gid;
  const int c1 = c0 + 8;
  const int h0 = tid * 2;
  const int h1 = h0 + 1;
  for (int chunk = lo; chunk < hi; ++chunk) {
    const int b = (chunk - lo) & 1;
    mbarrier_wait_parity(sm.mbar_full(b), ((chunk - lo) / 2) & 1);
    bar_sync_t<3, MATH_THREADS>();
    uint8_t* kv = sm.kv_fp8(b);
    auto scale = [&](int c, int vc) {
      return *reinterpret_cast<const float*>(kv + c * 528 + 512 + vc * 4);
    };
    float qk[4] = {};
#pragma unroll
    for (int vc = 0; vc < 4; ++vc) {
      float a[4] = {};
      const uint8_t sfb = fp32_to_ue8m0(sm.q_sc()[gid * 4 + vc]);
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const int ko = vc * 128 + ks * 32;
        uint32_t a0, a1, a2, a3, b0, b1;
        ldmatrix_load_A_fp8(a0, a1, a2, a3, kv + warp * 16 * 528 + ko, 528, lane);
        ldmatrix_load_B_fp8(b0, b1, sm.q_fp8() + ko, KV::Q_NOPE_STRIDE, lane);
        auto r = mma_fp8_block_scaled_m16n8k32(a0, a1, a2, a3, b0, b1,
                                               a[0], a[1], a[2], a[3], 127, sfb);
        a[0] = r.d0; a[1] = r.d1; a[2] = r.d2; a[3] = r.d3;
      }
      qk[0] += a[0] * scale(c0, vc);
      qk[1] += a[1] * scale(c0, vc);
      qk[2] += a[2] * scale(c1, vc);
      qk[3] += a[3] * scale(c1, vc);
    }
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      uint32_t a0, a1, a2, a3;
      ldmatrix_load_A_bf16(a0, a1, a2, a3, sm.kv_rope(b) + warp * 16 * 64 + ks * 16, 64, lane);
      const bf16* qr = sm.q_rope() + gid * 64 + ks * 16 + tid * 2;
      uint32_t b0 = *reinterpret_cast<const uint32_t*>(qr);
      uint32_t b1 = *reinterpret_cast<const uint32_t*>(qr + 8);
      auto r = mma_bf16_m16n8k16(a0, a1, a2, a3, b0, b1, qk[0], qk[1], qk[2], qk[3]);
      qk[0] = r.d0; qk[1] = r.d1; qk[2] = r.d2; qk[3] = r.d3;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int pos = chunk * 64 + (i < 2 ? c0 : c1);
      qk[i] = pos < length && idx[pos] >= 0 ? qk[i] * (sm_scale * LOG2E) : -1e30f;
    }
    float lm[2] = {fmaxf(qk[0], qk[2]), fmaxf(qk[1], qk[3])};
#pragma unroll
    for (int s = 16; s >= 4; s >>= 1) {
      lm[0] = fmaxf(lm[0], __shfl_xor_sync(0xffffffff, lm[0], s));
      lm[1] = fmaxf(lm[1], __shfl_xor_sync(0xffffffff, lm[1], s));
    }
    float ls[2] = {};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ls[i & 1] += qk[i] > -1e29f ? exp2f(qk[i] - lm[i & 1]) : 0.f;
    }
#pragma unroll
    for (int s = 16; s >= 4; s >>= 1) {
      ls[0] += __shfl_xor_sync(0xffffffff, ls[0], s);
      ls[1] += __shfl_xor_sync(0xffffffff, ls[1], s);
    }
    if (gid == 0) {
      sm.warp_max()[warp * 8 + h0] = lm[0];
      sm.warp_max()[warp * 8 + h1] = lm[1];
      sm.warp_sum()[warp * 8 + h0] = ls[0];
      sm.warp_sum()[warp * 8 + h1] = ls[1];
    }
    bar_sync_t<3, MATH_THREADS>();
    if (threadIdx.x < 8) {
      const int h = threadIdx.x;
      float m = gmax[h];
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        m = fmaxf(m, sm.warp_max()[w * 8 + h]);
      }
      alpha[h] = gmax[h] > -1e29f ? exp2f(gmax[h] - m) : 0.f;
      float sum = gsum[h] * alpha[h];
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        sum += sm.warp_sum()[w * 8 + h] * exp2f(sm.warp_max()[w * 8 + h] - m);
      }
      gmax[h] = m;
      gsum[h] = sum;
    }
    for (int i = threadIdx.x; i < 64; i += MATH_THREADS) {
      sm.w_head_sc()[i] = 0.f;
    }
    bar_sync_t<3, MATH_THREADS>();
    float p[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      p[i] = qk[i] > -1e29f ? exp2f(qk[i] - gmax[h0 + (i & 1)]) : 0.f;
    }
#pragma unroll
    for (int vc = 0; vc < 4; ++vc) {
      atomicMax(reinterpret_cast<int*>(sm.w_head_sc() + vc * 16 + h0),
                __float_as_int(fmaxf(fabsf(p[0] * scale(c0, vc)), fabsf(p[2] * scale(c1, vc)))));
      atomicMax(reinterpret_cast<int*>(sm.w_head_sc() + vc * 16 + h1),
                __float_as_int(fmaxf(fabsf(p[1] * scale(c0, vc)), fabsf(p[3] * scale(c1, vc)))));
    }
    bar_sync_t<3, MATH_THREADS>();
    if (threadIdx.x < 32) {
      const int i = (threadIdx.x / 8) * 16 + threadIdx.x % 8;
      sm.w_head_sc()[i] = fmaxf(sm.w_head_sc()[i], 1e-10f) / FP8_MAX;
    }
    bar_sync_t<3, MATH_THREADS>();
#pragma unroll
    for (int vc = 0; vc < 4; ++vc) {
      uint8_t* weights = sm.w_fp8(vc & 1);
      const float si0 = 1.f / sm.w_head_sc()[vc * 16 + h0];
      const float si1 = 1.f / sm.w_head_sc()[vc * 16 + h1];
      const float w0 = p[0] * scale(c0, vc) * si0;
      const float w1 = p[2] * scale(c1, vc) * si0;
      const float w2 = p[1] * scale(c0, vc) * si1;
      const float w3 = p[3] * scale(c1, vc) * si1;
#pragma unroll
      for (int pass = 0; pass < 2; ++pass) {
        auto w = quantize_weight_quad_for_pass<ScaleFormat::ARBITRARY_FP32>(w0, w1, w2, w3, pass);
        weights[(h0 + pass * 8) * 80 + c0] = w.h0_e0;
        weights[(h0 + pass * 8) * 80 + c1] = w.h0_e1;
        weights[(h1 + pass * 8) * 80 + c0] = w.h1_e0;
        weights[(h1 + pass * 8) * 80 + c1] = w.h1_e1;
      }
      bar_sync_t<3, MATH_THREADS>();
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        float xv[4] = {};
        const int dim = vc * 128 + (nt * 4 + warp) * 8;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
          uint32_t a0, a1, a2, a3, b0, b1;
          ldmatrix_load_A_fp8(a0, a1, a2, a3, weights + ks * 32, 80, lane);
          d2_load_b_fp8<528>(b0, b1, kv, ks * 32, dim, lane);
          auto r = mma_fp8_m16n8k32(a0, a1, a2, a3, b0, b1, xv[0], xv[1], xv[2], xv[3]);
          xv[0] = r.d0; xv[1] = r.d1; xv[2] = r.d2; xv[3] = r.d3;
        }
        const float sc = sm.w_head_sc()[vc * 16 + gid];
        acc[vc][nt][0] = acc[vc][nt][0] * alpha[gid] + (xv[0] + xv[2]) * sc;
        acc[vc][nt][1] = acc[vc][nt][1] * alpha[gid] + (xv[1] + xv[3]) * sc;
      }
    }
    bar_sync_t<3, MATH_THREADS>();
    if (threadIdx.x == 0) {
      mbarrier_arrive(sm.mbar_empty(b));
    }
  }
  const size_t out = ((size_t)t * 8 + gid) * num_splits + split;
  const float inv = gsum[gid] > 0.f ? 1.f / gsum[gid] : 0.f;
#pragma unroll
  for (int vc = 0; vc < 4; ++vc) {
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int dim = vc * 128 + (nt * 4 + warp) * 8 + tid * 2;
      mid_out[out * 512 + dim] = __float2bfloat16(acc[vc][nt][0] * inv);
      mid_out[out * 512 + dim + 1] = __float2bfloat16(acc[vc][nt][1] * inv);
    }
  }
  if (warp == 0 && tid == 0) {
    mid_lse[out] = gsum[gid] > 0.f ? gmax[gid] + log2f(gsum[gid]) : -1e30f;
  }
}

}  // namespace flashinfer::sparse_mla_sm120
