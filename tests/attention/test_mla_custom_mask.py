"""Test custom mask support for MLA paged attention (FA2 backend).

Validates MaskMode::kCustom and MaskMode::kCausalCustom by:
1. Comparing custom mask output against the well-tested causal/non-causal paths
2. Comparing custom mask output against a PyTorch reference implementation
3. Comparing kCausalCustom output against kCustom with equivalent full mask
4. Regression tests to ensure causal/non-causal paths are unaffected
"""

import math

import pytest
import torch

from flashinfer.mla import BatchMLAPagedAttentionWrapper
from flashinfer.quantization import segment_packbits

CKV_DIM = 512
KPE_DIM = 64


def attention_ref(batch_size, q, k, v, causal, sm_scale, custom_mask=None, prefix_len=None):
    """PyTorch reference for MLA attention.

    q: [batch_size * qo_len, num_heads, head_dim_qk]
    k: [batch_size * kv_len, num_heads, head_dim_qk]  (heads=1 for MLA, repeated)
    v: [batch_size * kv_len, num_heads, head_dim_vo]  (heads=1 for MLA, repeated)
    custom_mask: optional [batch_size, qo_len, kv_len] bool tensor (True=attend)
    prefix_len: optional int - when set with custom_mask,
        custom_mask is [batch_size, qo_len, kv_len - prefix_len] for suffix tokens only,
        and prefix positions are all-true (causal).
    """
    qo_len = q.shape[0] // batch_size
    kv_len = k.shape[0] // batch_size
    num_heads = q.shape[1]
    head_dim_qk = q.shape[2]

    q_4d = q.view(batch_size, qo_len, num_heads, head_dim_qk).float()
    k_4d = k.view(batch_size, kv_len, num_heads, head_dim_qk).float()
    v_4d = v.view(batch_size, kv_len, num_heads, v.shape[2]).float()

    logits = torch.einsum("bmhd,bnhd->bhmn", q_4d, k_4d) * sm_scale

    if custom_mask is not None and prefix_len is not None:
        # kCausalCustom: prefix is all-true, suffix uses custom_mask
        full_mask = torch.ones(batch_size, qo_len, kv_len, dtype=torch.bool, device=q.device)
        for i in range(batch_size):
            full_mask[i, :, prefix_len:] = custom_mask[i]
        mask = full_mask.unsqueeze(1)
        logits = logits.masked_fill(~mask, float("-inf"))
    elif custom_mask is not None:
        # custom_mask: [batch_size, qo_len, kv_len], True=attend
        mask = custom_mask.unsqueeze(1)  # [batch_size, 1, qo_len, kv_len]
        logits = logits.masked_fill(~mask, float("-inf"))
    elif causal:
        # causal mask: position i can attend to positions <= i
        # For MLA with qo_len <= kv_len: q position j attends to kv positions
        # 0..(kv_len - qo_len + j)
        causal_mask = (
            torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(1)
            >= torch.arange(0, kv_len, device=q.device).unsqueeze(0)
        )
        logits = logits.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    # else: no mask (non-causal), all positions attendable

    p = torch.softmax(logits, dim=-1)
    o = torch.einsum("bhmn,bnhd->bmhd", p, v_4d).reshape(
        batch_size * qo_len, num_heads, -1
    )
    return o.to(q.dtype)


def generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads):
    """Convert paged KV cache to standard K/V tensors for reference computation.

    ckv: [total_pages, page_size, ckv_dim]
    kpe: [total_pages, page_size, kpe_dim]
    Returns: k [batch_size*kv_len, num_heads, ckv_dim+kpe_dim], v [batch_size*kv_len, num_heads, ckv_dim]
    """
    pages_num = ckv.shape[0] // batch_size
    page_size = ckv.shape[1]
    k = ckv.view(batch_size, pages_num * page_size, CKV_DIM)[:, :kv_len, :].contiguous()
    k = torch.cat(
        [
            k,
            kpe.view(batch_size, pages_num * page_size, KPE_DIM)[:, :kv_len, :].contiguous(),
        ],
        dim=-1,
    )
    k = k.unsqueeze(2).expand(-1, -1, num_heads, -1).reshape(
        batch_size * kv_len, num_heads, -1
    )
    v = ckv.view(batch_size, pages_num * page_size, CKV_DIM)[:, :kv_len, :].contiguous()
    v = v.unsqueeze(2).expand(-1, -1, num_heads, -1).reshape(
        batch_size * kv_len, num_heads, -1
    )
    return k, v


@pytest.mark.parametrize("batch_size", [1, 3, 7])
@pytest.mark.parametrize("kv_len", [54, 97, 512])
@pytest.mark.parametrize("qo_len", [3, 17])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("page_size", [1])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_mla_custom_mask_causal_equivalent(batch_size, kv_len, qo_len, num_heads, page_size, dtype):
    """Custom mask with causal pattern must match causal=True output exactly.

    This is the primary correctness test: compare the custom mask path
    against the well-tested causal path. No PyTorch reference needed.
    """
    assert qo_len <= kv_len, "qo_len must be <= kv_len"
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    q_nope = torch.randn(batch_size * qo_len, num_heads, CKV_DIM, dtype=dtype, device=device)
    q_pe = torch.randn(batch_size * qo_len, num_heads, KPE_DIM, dtype=dtype, device=device)
    pages_num = math.ceil(kv_len / page_size)
    ckv = torch.randn(batch_size * pages_num, page_size, CKV_DIM, dtype=dtype, device=device)
    kpe = torch.randn(batch_size * pages_num, page_size, KPE_DIM, dtype=dtype, device=device)

    sm_scale = 1.0 / ((CKV_DIM + KPE_DIM) ** 0.5)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    q_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
    kv_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * pages_num
    kv_indices = torch.arange(0, batch_size * pages_num, device=device, dtype=torch.int32)
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    # Run causal=True
    wrapper_causal = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper_causal.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, True, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o_causal = wrapper_causal.run(q_nope, q_pe, ckv, kpe)

    # Run custom mask with causal pattern, causal=False
    custom_mask = torch.tril(
        torch.full((batch_size, qo_len, kv_len), True, device=device),
        diagonal=(kv_len - qo_len),
    )
    wrapper_custom = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper_custom.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, False, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o_custom = wrapper_custom.run(q_nope, q_pe, ckv, kpe, custom_mask=custom_mask)

    torch.testing.assert_close(o_custom, o_causal, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 3, 7])
@pytest.mark.parametrize("kv_len", [54, 97, 512])
@pytest.mark.parametrize("qo_len", [3, 17])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("page_size", [1])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_mla_custom_mask_all_ones_matches_noncausal(batch_size, kv_len, qo_len, num_heads, page_size, dtype):
    """All-ones custom mask must match causal=False output exactly."""
    assert qo_len <= kv_len, "qo_len must be <= kv_len"
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    q_nope = torch.randn(batch_size * qo_len, num_heads, CKV_DIM, dtype=dtype, device=device)
    q_pe = torch.randn(batch_size * qo_len, num_heads, KPE_DIM, dtype=dtype, device=device)
    pages_num = math.ceil(kv_len / page_size)
    ckv = torch.randn(batch_size * pages_num, page_size, CKV_DIM, dtype=dtype, device=device)
    kpe = torch.randn(batch_size * pages_num, page_size, KPE_DIM, dtype=dtype, device=device)

    sm_scale = 1.0 / ((CKV_DIM + KPE_DIM) ** 0.5)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    q_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
    kv_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * pages_num
    kv_indices = torch.arange(0, batch_size * pages_num, device=device, dtype=torch.int32)
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    # Run causal=False (no mask)
    wrapper_noncausal = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper_noncausal.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, False, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o_noncausal = wrapper_noncausal.run(q_nope, q_pe, ckv, kpe)

    # Run custom mask with all-ones (should be identical to non-causal)
    all_ones_mask = torch.ones(batch_size, qo_len, kv_len, dtype=torch.bool, device=device)
    wrapper_custom = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper_custom.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, False, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o_custom = wrapper_custom.run(q_nope, q_pe, ckv, kpe, custom_mask=all_ones_mask)

    torch.testing.assert_close(o_custom, o_noncausal, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 7])
@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("qo_len", [3, 17])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("page_size", [1])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_mla_custom_mask_sliding_window(batch_size, kv_len, qo_len, num_heads, page_size, dtype):
    """Sliding window custom mask vs PyTorch reference."""
    assert qo_len <= kv_len, "qo_len must be <= kv_len"
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    q_nope = torch.randn(batch_size * qo_len, num_heads, CKV_DIM, dtype=dtype, device=device)
    q_pe = torch.randn(batch_size * qo_len, num_heads, KPE_DIM, dtype=dtype, device=device)
    pages_num = math.ceil(kv_len / page_size)
    ckv = torch.randn(batch_size * pages_num, page_size, CKV_DIM, dtype=dtype, device=device)
    kpe = torch.randn(batch_size * pages_num, page_size, KPE_DIM, dtype=dtype, device=device)

    sm_scale = 1.0 / ((CKV_DIM + KPE_DIM) ** 0.5)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    q_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
    kv_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * pages_num
    kv_indices = torch.arange(0, batch_size * pages_num, device=device, dtype=torch.int32)
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    # Sliding window: each query attends to the last 32 kv positions
    window_size = 32
    sliding_mask = torch.zeros(batch_size, qo_len, kv_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        for q in range(qo_len):
            kv_pos_offset = kv_len - qo_len + q  # absolute kv position for this query
            start = max(0, kv_pos_offset - window_size + 1)
            end = kv_pos_offset + 1
            sliding_mask[i, q, start:end] = True

    wrapper = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, False, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o_custom = wrapper.run(q_nope, q_pe, ckv, kpe, custom_mask=sliding_mask)

    # PyTorch reference
    k, v = generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads)
    q_full = torch.cat([q_nope, q_pe], dim=-1)
    o_ref = attention_ref(batch_size, q_full, k, v, causal=False, sm_scale=sm_scale, custom_mask=sliding_mask)

    torch.testing.assert_close(o_custom, o_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 7])
@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("qo_len", [3, 17])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("page_size", [1])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_mla_no_mask_regression(batch_size, kv_len, qo_len, num_heads, causal, page_size, dtype):
    """Regression: causal/non-causal paths must produce same results after custom mask changes."""
    if causal and qo_len > kv_len:
        pytest.skip("qo_len > kv_len not supported for causal attention")
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    q_nope = torch.randn(batch_size * qo_len, num_heads, CKV_DIM, dtype=dtype, device=device)
    q_pe = torch.randn(batch_size * qo_len, num_heads, KPE_DIM, dtype=dtype, device=device)
    pages_num = math.ceil(kv_len / page_size)
    ckv = torch.randn(batch_size * pages_num, page_size, CKV_DIM, dtype=dtype, device=device)
    kpe = torch.randn(batch_size * pages_num, page_size, KPE_DIM, dtype=dtype, device=device)

    sm_scale = 1.0 / ((CKV_DIM + KPE_DIM) ** 0.5)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    q_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
    kv_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * pages_num
    kv_indices = torch.arange(0, batch_size * pages_num, device=device, dtype=torch.int32)
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    wrapper = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, causal, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o = wrapper.run(q_nope, q_pe, ckv, kpe)

    # PyTorch reference
    k, v = generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads)
    q_full = torch.cat([q_nope, q_pe], dim=-1)
    o_ref = attention_ref(batch_size, q_full, k, v, causal=causal, sm_scale=sm_scale)

    torch.testing.assert_close(o.flatten(), o_ref.flatten(), rtol=1e-3, atol=1e-3)


def _build_tree_mask(num_prefill_tokens, prefix_len):
    """Build a tree-structured mask for kCausalCustom testing.

    Each prefill token attends to all prefix tokens (causal) and its
    ancestors in the tree (binary tree: node i's parent is floor((i-1)/2)).

    Returns: [num_prefill_tokens, num_prefill_tokens] bool tensor
    """
    mask = torch.zeros(num_prefill_tokens, num_prefill_tokens, dtype=torch.bool)
    for q in range(num_prefill_tokens):
        for k in range(num_prefill_tokens):
            # Token q attends to token k if k is an ancestor of q (or q itself)
            cur = q
            while cur > 0:
                if cur == k:
                    mask[q, k] = True
                    break
                cur = (cur - 1) >> 1
            if k == 0:
                mask[q, k] = True  # root
            if k == q:
                mask[q, k] = True
    return mask


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("prefix_len", [32, 64, 200, 1000])
@pytest.mark.parametrize("num_prefill_tokens", [6, 14])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("page_size", [1])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_mla_causal_custom_mask_tree(batch_size, prefix_len, num_prefill_tokens, num_heads, page_size, dtype):
    """kCausalCustom with tree mask matches kCustom with equivalent full mask.

    This verifies that kCausalCustom produces identical results to kCustom when
    the full mask is: prefix=all-true, suffix=tree-structured. The key benefit
    is that kCausalCustom uses a smaller mask [qo_len, num_prefill_tokens] that
    doesn't depend on prefix_len.
    """
    kv_len = prefix_len + num_prefill_tokens
    qo_len = num_prefill_tokens
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    q_nope = torch.randn(batch_size * qo_len, num_heads, CKV_DIM, dtype=dtype, device=device)
    q_pe = torch.randn(batch_size * qo_len, num_heads, KPE_DIM, dtype=dtype, device=device)
    pages_num = math.ceil(kv_len / page_size)
    ckv = torch.randn(batch_size * pages_num, page_size, CKV_DIM, dtype=dtype, device=device)
    kpe = torch.randn(batch_size * pages_num, page_size, KPE_DIM, dtype=dtype, device=device)

    sm_scale = 1.0 / ((CKV_DIM + KPE_DIM) ** 0.5)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    q_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
    kv_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * pages_num
    kv_indices = torch.arange(0, batch_size * pages_num, device=device, dtype=torch.int32)
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    # Build tree mask for suffix tokens only
    tree_mask = _build_tree_mask(num_prefill_tokens, prefix_len)  # [num_prefill_tokens, num_prefill_tokens]

    # Build the equivalent full mask for kCustom: [batch_size, qo_len, kv_len]
    # Prefix positions are all-true, suffix positions follow tree mask
    full_mask = torch.zeros(batch_size, qo_len, kv_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        full_mask[i, :, :prefix_len] = True  # all prefix positions attended
        full_mask[i, :, prefix_len:] = tree_mask  # suffix follows tree

    # Run with kCustom (full mask)
    wrapper_custom = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper_custom.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, False, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o_custom = wrapper_custom.run(q_nope, q_pe, ckv, kpe, custom_mask=full_mask)

    # Pack the suffix-only mask for kCausalCustom: [batch_size, qo_len, num_prefill_tokens]
    suffix_mask = tree_mask.unsqueeze(0).expand(batch_size, -1, -1).to(device)
    # Pack using segment_packbits
    qo_indptr_host = q_indptr.cpu()
    mask_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cpu")
    for i in range(batch_size):
        mask_indptr[i + 1] = mask_indptr[i].item() + qo_len * num_prefill_tokens
    packed_mask, mask_indptr_gpu = segment_packbits(
        suffix_mask.contiguous().view(-1).to(device),
        mask_indptr.to(device),
        bitorder="little",
    )

    # Run with kCausalCustom
    wrapper_cc = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper_cc.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, True, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o_cc = wrapper_cc.run(q_nope, q_pe, ckv, kpe, causal_custom_mask=packed_mask)

    torch.testing.assert_close(o_cc, o_custom, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("prefix_len", [32, 64, 200, 1000])
@pytest.mark.parametrize("num_prefill_tokens", [3, 7])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("page_size", [1])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_mla_causal_custom_mask_all_ones_suffix(batch_size, prefix_len, num_prefill_tokens, num_heads, page_size, dtype):
    """kCausalCustom with all-ones suffix mask matches causal output.

    When the suffix mask is all-ones (every prefill token attends to every
    other prefill token), the result should match causal=True since prefix
    positions are always attended and suffix positions are all-attendable
    (which is a superset of causal).
    """
    kv_len = prefix_len + num_prefill_tokens
    qo_len = num_prefill_tokens
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    q_nope = torch.randn(batch_size * qo_len, num_heads, CKV_DIM, dtype=dtype, device=device)
    q_pe = torch.randn(batch_size * qo_len, num_heads, KPE_DIM, dtype=dtype, device=device)
    pages_num = math.ceil(kv_len / page_size)
    ckv = torch.randn(batch_size * pages_num, page_size, CKV_DIM, dtype=dtype, device=device)
    kpe = torch.randn(batch_size * pages_num, page_size, KPE_DIM, dtype=dtype, device=device)

    sm_scale = 1.0 / ((CKV_DIM + KPE_DIM) ** 0.5)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    q_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
    kv_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * pages_num
    kv_indices = torch.arange(0, batch_size * pages_num, device=device, dtype=torch.int32)
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    # Run causal=True
    wrapper_causal = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper_causal.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, True, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o_causal = wrapper_causal.run(q_nope, q_pe, ckv, kpe)

    # Build all-ones suffix mask
    suffix_mask = torch.ones(batch_size, qo_len, num_prefill_tokens, dtype=torch.bool, device=device)

    # Pack
    qo_indptr_host = q_indptr.cpu()
    mask_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cpu")
    for i in range(batch_size):
        mask_indptr[i + 1] = mask_indptr[i].item() + qo_len * num_prefill_tokens
    packed_mask, mask_indptr_gpu = segment_packbits(
        suffix_mask.contiguous().view(-1).to(device),
        mask_indptr.to(device),
        bitorder="little",
    )

    # Run kCausalCustom with all-ones suffix mask
    # Note: all-ones suffix is a superset of causal (which only allows causal ordering in suffix),
    # so results will differ from causal. Instead, compare against kCustom with full all-ones mask.
    full_mask = torch.ones(batch_size, qo_len, kv_len, dtype=torch.bool, device=device)
    wrapper_custom = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper_custom.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, False, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o_custom = wrapper_custom.run(q_nope, q_pe, ckv, kpe, custom_mask=full_mask)

    wrapper_cc = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper_cc.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, True, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o_cc = wrapper_cc.run(q_nope, q_pe, ckv, kpe, causal_custom_mask=packed_mask)

    torch.testing.assert_close(o_cc, o_custom, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("prefix_len", [32, 64, 200, 1000])
@pytest.mark.parametrize("num_prefill_tokens", [6, 14])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("page_size", [1])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_mla_causal_custom_mask_vs_reference(batch_size, prefix_len, num_prefill_tokens, num_heads, page_size, dtype):
    """kCausalCustom with tree mask matches PyTorch reference."""
    kv_len = prefix_len + num_prefill_tokens
    qo_len = num_prefill_tokens
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    q_nope = torch.randn(batch_size * qo_len, num_heads, CKV_DIM, dtype=dtype, device=device)
    q_pe = torch.randn(batch_size * qo_len, num_heads, KPE_DIM, dtype=dtype, device=device)
    pages_num = math.ceil(kv_len / page_size)
    ckv = torch.randn(batch_size * pages_num, page_size, CKV_DIM, dtype=dtype, device=device)
    kpe = torch.randn(batch_size * pages_num, page_size, KPE_DIM, dtype=dtype, device=device)

    sm_scale = 1.0 / ((CKV_DIM + KPE_DIM) ** 0.5)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    q_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
    kv_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * pages_num
    kv_indices = torch.arange(0, batch_size * pages_num, device=device, dtype=torch.int32)
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    # Build tree mask for suffix tokens
    tree_mask = _build_tree_mask(num_prefill_tokens, prefix_len)

    # Pack suffix mask
    suffix_mask = tree_mask.unsqueeze(0).expand(batch_size, -1, -1).to(device)
    qo_indptr_host = q_indptr.cpu()
    mask_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cpu")
    for i in range(batch_size):
        mask_indptr[i + 1] = mask_indptr[i].item() + qo_len * num_prefill_tokens
    packed_mask, mask_indptr_gpu = segment_packbits(
        suffix_mask.contiguous().view(-1).to(device),
        mask_indptr.to(device),
        bitorder="little",
    )

    # Run kCausalCustom
    wrapper_cc = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper_cc.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, True, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o_cc = wrapper_cc.run(q_nope, q_pe, ckv, kpe, causal_custom_mask=packed_mask)

    # PyTorch reference
    k, v = generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads)
    q_full = torch.cat([q_nope, q_pe], dim=-1)
    o_ref = attention_ref(batch_size, q_full, k, v, causal=False, sm_scale=sm_scale,
                          custom_mask=suffix_mask.cpu(), prefix_len=prefix_len)

    torch.testing.assert_close(o_cc, o_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("prefix_len", [32, 64, 200, 1000])
@pytest.mark.parametrize("num_prefill_tokens", [3, 7])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("page_size", [1])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_mla_causal_custom_mask_causal_suffix(batch_size, prefix_len, num_prefill_tokens, num_heads, page_size, dtype):
    """kCausalCustom with causal suffix mask matches causal output.

    When the suffix mask is a causal mask (each prefill token only attends to
    itself and prior prefill tokens), the result should exactly match causal=True
    since the prefix is always attended and the suffix follows causal ordering.
    """
    kv_len = prefix_len + num_prefill_tokens
    qo_len = num_prefill_tokens
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    q_nope = torch.randn(batch_size * qo_len, num_heads, CKV_DIM, dtype=dtype, device=device)
    q_pe = torch.randn(batch_size * qo_len, num_heads, KPE_DIM, dtype=dtype, device=device)
    pages_num = math.ceil(kv_len / page_size)
    ckv = torch.randn(batch_size * pages_num, page_size, CKV_DIM, dtype=dtype, device=device)
    kpe = torch.randn(batch_size * pages_num, page_size, KPE_DIM, dtype=dtype, device=device)

    sm_scale = 1.0 / ((CKV_DIM + KPE_DIM) ** 0.5)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    q_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
    kv_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * pages_num
    kv_indices = torch.arange(0, batch_size * pages_num, device=device, dtype=torch.int32)
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    # Run causal=True
    wrapper_causal = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper_causal.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, True, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o_causal = wrapper_causal.run(q_nope, q_pe, ckv, kpe)

    # Build causal suffix mask: lower-triangular [qo_len, num_prefill_tokens]
    causal_suffix_mask = torch.tril(
        torch.ones(qo_len, num_prefill_tokens, dtype=torch.bool, device=device)
    )
    suffix_mask = causal_suffix_mask.unsqueeze(0).expand(batch_size, -1, -1)

    # Pack
    qo_indptr_host = q_indptr.cpu()
    mask_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cpu")
    for i in range(batch_size):
        mask_indptr[i + 1] = mask_indptr[i].item() + qo_len * num_prefill_tokens
    packed_mask, mask_indptr_gpu = segment_packbits(
        suffix_mask.contiguous().view(-1).to(device),
        mask_indptr.to(device),
        bitorder="little",
    )

    # Run kCausalCustom with causal suffix mask
    wrapper_cc = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper_cc.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, True, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o_cc = wrapper_cc.run(q_nope, q_pe, ckv, kpe, causal_custom_mask=packed_mask)

    torch.testing.assert_close(o_cc, o_causal, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("qo_len", [1, 3, 7])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("page_size", [1])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_mla_causal_custom_mask_no_prefix(batch_size, qo_len, num_heads, page_size, dtype):
    """kCausalCustom with no prefix (prefix_len=0) matches causal output.

    When there is no prefix KV cache, prefix_len = kv_len - qo_len = 0.
    The CausalCustom mask reduces to a [qo_len, qo_len] lower-triangular
    mask with no prefix bypass, which should be equivalent to built-in causal.
    """
    kv_len = qo_len  # no prefix
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    q_nope = torch.randn(batch_size * qo_len, num_heads, CKV_DIM, dtype=dtype, device=device)
    q_pe = torch.randn(batch_size * qo_len, num_heads, KPE_DIM, dtype=dtype, device=device)
    pages_num = math.ceil(kv_len / page_size)
    ckv = torch.randn(batch_size * pages_num, page_size, CKV_DIM, dtype=dtype, device=device)
    kpe = torch.randn(batch_size * pages_num, page_size, KPE_DIM, dtype=dtype, device=device)

    sm_scale = 1.0 / ((CKV_DIM + KPE_DIM) ** 0.5)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    q_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
    kv_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * pages_num
    kv_indices = torch.arange(0, batch_size * pages_num, device=device, dtype=torch.int32)
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    # Run causal=True
    wrapper_causal = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper_causal.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, True, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o_causal = wrapper_causal.run(q_nope, q_pe, ckv, kpe)

    # Build causal suffix mask: lower-triangular [qo_len, qo_len]
    causal_suffix_mask = torch.tril(
        torch.ones(qo_len, qo_len, dtype=torch.bool, device=device)
    )
    suffix_mask = causal_suffix_mask.unsqueeze(0).expand(batch_size, -1, -1)

    # Pack
    mask_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cpu")
    for i in range(batch_size):
        mask_indptr[i + 1] = mask_indptr[i].item() + qo_len * qo_len
    packed_mask, mask_indptr_gpu = segment_packbits(
        suffix_mask.contiguous().view(-1).to(device),
        mask_indptr.to(device),
        bitorder="little",
    )

    # Run kCausalCustom with causal=True plan
    wrapper_cc = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper_cc.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, True, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o_cc = wrapper_cc.run(q_nope, q_pe, ckv, kpe, causal_custom_mask=packed_mask)

    torch.testing.assert_close(o_cc, o_causal, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Extended causal-custom mask tests (mask_kv_len > qo_len)
#
# These tests verify that when causal_custom_mask_kv_len is provided and
# larger than qo_len, the mask extends into the most recent KV cache tokens.
# Positions before (kv_len - mask_kv_len) are always attended (deep prefix),
# and positions within the last mask_kv_len tokens are subject to the custom
# mask.
# ---------------------------------------------------------------------------


def _pack_suffix_mask(suffix_mask, qo_len, mask_kv_len, batch_size, device):
    """Pack a [batch_size, qo_len, mask_kv_len] bool mask into bit-packed uint8.

    Returns (packed_mask, mask_indptr_gpu) ready for causal_custom_mask param.
    Note: segment_packbits expects element counts (not byte offsets) in indptr.
    """
    mask_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cpu")
    for i in range(batch_size):
        num_bits = qo_len * mask_kv_len
        mask_indptr[i + 1] = mask_indptr[i].item() + num_bits
    packed_mask, mask_indptr_gpu = segment_packbits(
        suffix_mask.contiguous().view(-1).to(device),
        mask_indptr.to(device),
        bitorder="little",
    )
    return packed_mask, mask_indptr_gpu


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("prefix_len", [32, 64, 200])
@pytest.mark.parametrize("num_prefill_tokens", [6, 14])
@pytest.mark.parametrize("extra_kv_cache", [4, 8, 16])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("page_size", [1])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_mla_causal_custom_mask_extended_tree(batch_size, prefix_len, num_prefill_tokens,
                                               extra_kv_cache, num_heads, page_size, dtype):
    """Extended kCausalCustom (mask_kv_len > qo_len) with tree mask matches kCustom.

    The mask covers the last mask_kv_len = qo_len + extra_kv_cache tokens of the
    KV sequence (extra_kv_cache KV cache tokens + all prefill tokens). Compared
    against kCustom with the equivalent full [qo_len, kv_len] mask.
    """
    mask_kv_len = num_prefill_tokens + extra_kv_cache
    kv_len = prefix_len + num_prefill_tokens
    qo_len = num_prefill_tokens
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    q_nope = torch.randn(batch_size * qo_len, num_heads, CKV_DIM, dtype=dtype, device=device)
    q_pe = torch.randn(batch_size * qo_len, num_heads, KPE_DIM, dtype=dtype, device=device)
    pages_num = math.ceil(kv_len / page_size)
    ckv = torch.randn(batch_size * pages_num, page_size, CKV_DIM, dtype=dtype, device=device)
    kpe = torch.randn(batch_size * pages_num, page_size, KPE_DIM, dtype=dtype, device=device)

    sm_scale = 1.0 / ((CKV_DIM + KPE_DIM) ** 0.5)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    q_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
    kv_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * pages_num
    kv_indices = torch.arange(0, batch_size * pages_num, device=device, dtype=torch.int32)
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    # Build tree mask for the mask region: [qo_len, mask_kv_len]
    # The mask region starts at position (kv_len - mask_kv_len) = prefix_len - extra_kv_cache
    # Within the mask, position j corresponds to absolute KV position (kv_len - mask_kv_len + j)
    # For the tree structure, we build a tree over the mask_kv_len columns.
    # Columns [0, extra_kv_cache) are KV cache positions (all-attend for causal compatibility).
    # Columns [extra_kv_cache, mask_kv_len) are prefill positions (tree-structured).
    tree_mask_prefill = _build_tree_mask(num_prefill_tokens, prefix_len)  # [num_prefill, num_prefill]
    extended_mask = torch.zeros(qo_len, mask_kv_len, dtype=torch.bool)
    # KV cache positions in mask region: all queries attend to them (causal)
    extended_mask[:, :extra_kv_cache] = True
    # Prefill positions: tree structure
    extended_mask[:, extra_kv_cache:] = tree_mask_prefill

    # Build equivalent full mask for kCustom: [batch_size, qo_len, kv_len]
    mask_start = kv_len - mask_kv_len  # = prefix_len - extra_kv_cache
    full_mask = torch.zeros(batch_size, qo_len, kv_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        full_mask[i, :, :mask_start] = True  # deep prefix: always attend
        full_mask[i, :, mask_start:] = extended_mask  # mask region

    # Run kCustom (full mask)
    wrapper_custom = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper_custom.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, False, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o_custom = wrapper_custom.run(q_nope, q_pe, ckv, kpe, custom_mask=full_mask)

    # Pack extended mask for kCausalCustom
    suffix_mask = extended_mask.unsqueeze(0).expand(batch_size, -1, -1).to(device)
    packed_mask, mask_indptr_gpu = _pack_suffix_mask(suffix_mask, qo_len, mask_kv_len, batch_size, device)
    mask_kv_len_tensor = torch.full((batch_size,), mask_kv_len, dtype=torch.int32, device=device)

    # Run kCausalCustom with extended mask
    wrapper_cc = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper_cc.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, True, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o_cc = wrapper_cc.run(q_nope, q_pe, ckv, kpe, causal_custom_mask=packed_mask,
                          causal_custom_mask_kv_len=mask_kv_len_tensor)

    torch.testing.assert_close(o_cc, o_custom, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("prefix_len", [32, 64])
@pytest.mark.parametrize("num_prefill_tokens", [3, 7])
@pytest.mark.parametrize("extra_kv_cache", [4, 16])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("page_size", [1])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_mla_causal_custom_mask_extended_vs_reference(batch_size, prefix_len, num_prefill_tokens,
                                                       extra_kv_cache, num_heads, page_size, dtype):
    """Extended kCausalCustom with tree mask matches PyTorch reference."""
    mask_kv_len = num_prefill_tokens + extra_kv_cache
    kv_len = prefix_len + num_prefill_tokens
    qo_len = num_prefill_tokens
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    q_nope = torch.randn(batch_size * qo_len, num_heads, CKV_DIM, dtype=dtype, device=device)
    q_pe = torch.randn(batch_size * qo_len, num_heads, KPE_DIM, dtype=dtype, device=device)
    pages_num = math.ceil(kv_len / page_size)
    ckv = torch.randn(batch_size * pages_num, page_size, CKV_DIM, dtype=dtype, device=device)
    kpe = torch.randn(batch_size * pages_num, page_size, KPE_DIM, dtype=dtype, device=device)

    sm_scale = 1.0 / ((CKV_DIM + KPE_DIM) ** 0.5)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    q_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
    kv_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * pages_num
    kv_indices = torch.arange(0, batch_size * pages_num, device=device, dtype=torch.int32)
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    # Build extended tree mask
    tree_mask_prefill = _build_tree_mask(num_prefill_tokens, prefix_len)
    extended_mask = torch.zeros(qo_len, mask_kv_len, dtype=torch.bool)
    extended_mask[:, :extra_kv_cache] = True
    extended_mask[:, extra_kv_cache:] = tree_mask_prefill

    # Pack
    suffix_mask = extended_mask.unsqueeze(0).expand(batch_size, -1, -1).to(device)
    packed_mask, mask_indptr_gpu = _pack_suffix_mask(suffix_mask, qo_len, mask_kv_len, batch_size, device)
    mask_kv_len_tensor = torch.full((batch_size,), mask_kv_len, dtype=torch.int32, device=device)

    # Run kCausalCustom
    wrapper_cc = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper_cc.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, True, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o_cc = wrapper_cc.run(q_nope, q_pe, ckv, kpe, causal_custom_mask=packed_mask,
                          causal_custom_mask_kv_len=mask_kv_len_tensor)

    # PyTorch reference: prefix_len here is the mask start position
    mask_start = kv_len - mask_kv_len
    k, v = generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads)
    q_full = torch.cat([q_nope, q_pe], dim=-1)
    o_ref = attention_ref(batch_size, q_full, k, v, causal=False, sm_scale=sm_scale,
                          custom_mask=suffix_mask.cpu(), prefix_len=mask_start)

    torch.testing.assert_close(o_cc, o_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("prefix_len", [32, 64, 200])
@pytest.mark.parametrize("num_prefill_tokens", [3, 7])
@pytest.mark.parametrize("extra_kv_cache", [4, 16])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("page_size", [1])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_mla_causal_custom_mask_extended_causal_equivalent(batch_size, prefix_len, num_prefill_tokens,
                                                            extra_kv_cache, num_heads, page_size, dtype):
    """Extended kCausalCustom with causal mask matches causal=True output.

    When the extended mask is a causal pattern of width mask_kv_len (query at
    position q attends to positions <= kv_len - qo_len + q), the result should
    match causal=True since the mask is a subset of the built-in causal mask.
    """
    mask_kv_len = num_prefill_tokens + extra_kv_cache
    kv_len = prefix_len + num_prefill_tokens
    qo_len = num_prefill_tokens
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    q_nope = torch.randn(batch_size * qo_len, num_heads, CKV_DIM, dtype=dtype, device=device)
    q_pe = torch.randn(batch_size * qo_len, num_heads, KPE_DIM, dtype=dtype, device=device)
    pages_num = math.ceil(kv_len / page_size)
    ckv = torch.randn(batch_size * pages_num, page_size, CKV_DIM, dtype=dtype, device=device)
    kpe = torch.randn(batch_size * pages_num, page_size, KPE_DIM, dtype=dtype, device=device)

    sm_scale = 1.0 / ((CKV_DIM + KPE_DIM) ** 0.5)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    q_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
    kv_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * pages_num
    kv_indices = torch.arange(0, batch_size * pages_num, device=device, dtype=torch.int32)
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    # Run causal=True
    wrapper_causal = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper_causal.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, True, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o_causal = wrapper_causal.run(q_nope, q_pe, ckv, kpe)

    # Build causal mask of width mask_kv_len: [qo_len, mask_kv_len]
    # The mask region covers positions [mask_start, kv_len) where mask_start = kv_len - mask_kv_len.
    # Query q (absolute position prefix_len + q) can attend to KV position j (absolute mask_start + j)
    # iff mask_start + j <= prefix_len + q, i.e., j <= q + extra_kv_cache.
    causal_mask = torch.zeros(qo_len, mask_kv_len, dtype=torch.bool, device=device)
    for q in range(qo_len):
        causal_mask[q, :q + extra_kv_cache + 1] = True

    suffix_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
    packed_mask, mask_indptr_gpu = _pack_suffix_mask(suffix_mask, qo_len, mask_kv_len, batch_size, device)
    mask_kv_len_tensor = torch.full((batch_size,), mask_kv_len, dtype=torch.int32, device=device)

    # Run kCausalCustom with causal extended mask
    wrapper_cc = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper_cc.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, True, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o_cc = wrapper_cc.run(q_nope, q_pe, ckv, kpe, causal_custom_mask=packed_mask,
                          causal_custom_mask_kv_len=mask_kv_len_tensor)

    torch.testing.assert_close(o_cc, o_causal, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("prefix_len", [32, 64])
@pytest.mark.parametrize("num_prefill_tokens", [3, 7])
@pytest.mark.parametrize("extra_kv_cache", [4, 16])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("page_size", [1])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_mla_causal_custom_mask_extended_all_ones(batch_size, prefix_len, num_prefill_tokens,
                                                   extra_kv_cache, num_heads, page_size, dtype):
    """Extended kCausalCustom with all-ones mask matches kCustom all-ones full mask."""
    mask_kv_len = num_prefill_tokens + extra_kv_cache
    kv_len = prefix_len + num_prefill_tokens
    qo_len = num_prefill_tokens
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    q_nope = torch.randn(batch_size * qo_len, num_heads, CKV_DIM, dtype=dtype, device=device)
    q_pe = torch.randn(batch_size * qo_len, num_heads, KPE_DIM, dtype=dtype, device=device)
    pages_num = math.ceil(kv_len / page_size)
    ckv = torch.randn(batch_size * pages_num, page_size, CKV_DIM, dtype=dtype, device=device)
    kpe = torch.randn(batch_size * pages_num, page_size, KPE_DIM, dtype=dtype, device=device)

    sm_scale = 1.0 / ((CKV_DIM + KPE_DIM) ** 0.5)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    q_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
    kv_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * pages_num
    kv_indices = torch.arange(0, batch_size * pages_num, device=device, dtype=torch.int32)
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    # kCustom with full all-ones mask
    full_mask = torch.ones(batch_size, qo_len, kv_len, dtype=torch.bool, device=device)
    wrapper_custom = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper_custom.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, False, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o_custom = wrapper_custom.run(q_nope, q_pe, ckv, kpe, custom_mask=full_mask)

    # kCausalCustom with all-ones extended mask
    suffix_mask = torch.ones(batch_size, qo_len, mask_kv_len, dtype=torch.bool, device=device)
    packed_mask, mask_indptr_gpu = _pack_suffix_mask(suffix_mask, qo_len, mask_kv_len, batch_size, device)
    mask_kv_len_tensor = torch.full((batch_size,), mask_kv_len, dtype=torch.int32, device=device)

    wrapper_cc = BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
    wrapper_cc.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, CKV_DIM, KPE_DIM,
        page_size, True, sm_scale,
        q_nope.dtype, ckv.dtype,
    )
    o_cc = wrapper_cc.run(q_nope, q_pe, ckv, kpe, causal_custom_mask=packed_mask,
                          causal_custom_mask_kv_len=mask_kv_len_tensor)

    torch.testing.assert_close(o_cc, o_custom, rtol=1e-3, atol=1e-3)
