/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_MLA_PARAMS_CUH_
#define FLASHINFER_MLA_PARAMS_CUH_
#include <cuda.h>

#include "../fastdiv.cuh"
#include "../profiler.cuh"

namespace flashinfer {

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_>
struct MLAParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;

  DTypeQ* q_nope;
  DTypeQ* q_pe;
  DTypeKV* ckv;
  DTypeKV* kpe;
  DTypeO* partial_o;
  float* partial_lse;
  DTypeO* final_o;
  float* final_lse;

  IdType* q_indptr;
  IdType* kv_indptr;
  IdType* partial_indptr;
  IdType* merge_packed_offset_start;
  IdType* merge_packed_offset_end;
  IdType* merge_partial_packed_offset_start;
  IdType* merge_partial_packed_offset_end;
  IdType* merge_partial_stride;
  IdType* kv_indices;
  IdType* q_len;
  IdType* kv_len;
  IdType* q_start;
  IdType* kv_start;
  IdType* kv_end;
  IdType* work_indptr;

  PROFILER_PARAMS_DECL

  uint_fastdiv block_size;
  uint_fastdiv num_heads;

  uint32_t q_nope_stride_n;
  uint32_t q_nope_stride_h;
  uint32_t q_pe_stride_n;
  uint32_t q_pe_stride_h;
  uint32_t ckv_stride_page;
  uint32_t ckv_stride_n;
  uint32_t kpe_stride_page;
  uint32_t kpe_stride_n;
  uint32_t o_stride_n;
  uint32_t o_stride_h;

  float sm_scale;
  // Per-tensor symmetric quantization scales for the FP8 KV cache path.
  // Both default to 1.0 and have no effect on the BF16/FP16 KV path.
  // dequantized = ckv_scale * static_cast<float>(ckv_fp8); same for kpe.
  float ckv_scale = 1.f;
  float kpe_scale = 1.f;
  bool return_lse_base_on_e;

  // Context parallelism: causal mask correction for token-level interleaved KV shards.
  // When cp_world_size == 0, CP is inactive and the kernel uses original causal mask logic.
  // When cp_world_size > 0, global_kv_idx = kv_idx * cp_world_size + cp_rank replaces
  // kv_idx in causal comparisons, and cp_kv_len replaces kv_len.
  uint32_t cp_world_size;
  uint32_t cp_rank;

  // Custom attention mask (packed bitmask, little-endian bit order).
  // For MaskMode::kCustom:
  //   The mask has shape [qo_len, global_kv_len] flattened and bit-packed.
  //   Bit (q_idx * global_kv_len + kv_idx) is at byte offset / 8, bit position % 8.
  //   A set bit (1) means the position is allowed (not masked), 0 means masked out.
  // For MaskMode::kCausalCustom:
  //   The mask has shape [qo_len, mask_kv_len] flattened and bit-packed, covering
  //   the last mask_kv_len tokens of the KV sequence (which may include both existing
  //   KV cache tokens and new prefill tokens). Positions before (causal_kv_len -
  //   mask_kv_len) are always attended (causal is trivially satisfied).
  //   For kv positions >= prefix_len (= causal_kv_len - mask_kv_len), the mask is
  //   looked up at:
  //     offset = q_idx * mask_kv_len + (global_kv_idx - prefix_len)
  //   When maybe_mask_kv_len is nullptr, mask_kv_len defaults to qo_len (the mask
  //   covers only the prefill tokens, which is the original kCausalCustom behavior).
  //   When maybe_mask_kv_len is provided, mask_kv_len can be larger than qo_len,
  //   allowing the mask to extend into the most recent KV cache tokens.
  // maybe_custom_mask: pointer to packed uint8 bitmask array, or nullptr.
  // maybe_mask_indptr: pointer to int32/64 array of shape [batch_size + 1], or nullptr.
  //   maybe_mask_indptr[batch_idx] gives the byte offset into maybe_custom_mask for that request.
  // maybe_mask_kv_len: per-batch mask width for kCausalCustom, or nullptr.
  //   maybe_mask_kv_len[batch_idx] gives the number of KV columns in the mask for that request.
  //   Only used by kCausalCustom; ignored by kCustom (which uses causal_kv_len as width).
  // When nullptr, custom masking is not applied (only causal/no-mask is used based on MASK_MODE).
  uint8_t* maybe_custom_mask = nullptr;
  IdType* maybe_mask_indptr = nullptr;
  IdType* maybe_mask_kv_len = nullptr;

  // batch_indices: pointer to int32/64 array of shape [total_num_works].
  //   batch_indices[work_idx] gives the original batch index for this work item.
  //   Used for custom mask indptr lookup since work items may be reordered by the scheduler.
  IdType* batch_indices = nullptr;
};

};  // namespace flashinfer

#endif  // FLASHINFER_MLA_PARAMS_CUH_
