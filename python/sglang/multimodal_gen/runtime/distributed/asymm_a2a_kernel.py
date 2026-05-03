"""Triton kernels for WAN SP asymmetric all2all (pull forward, push reverse).

Two persistent, peer-swizzled, TMA-descriptor-per-peer kernels:

  `asymm_pull_seq_to_heads`:  forward.  seq-sharded Q/K/V → head-sharded.
      src (peer, symm_mem):  [B, H_total, S_local, D]
      dst (local):           [B, H_local, WORLD * S_local, D]

  `asymm_push_heads_to_seq`:  reverse.  head-sharded attn_out → seq-sharded.
      src (local):           [B, H_local, WORLD * S_local, D]
      dst (peer, symm_mem):  [B, H_total, S_local, D]   (write at global-head row)

The reverse direction is push, not pull. A push lets the producer write
its local result straight into peers' recv buffers, so a single barrier
after the launch is enough to publish the data — no mid-iter sync between
"finish local attn" and "peer reads my output". Cross-rank head ownership
is disjoint, so different ranks write disjoint rows of each peer's
recv_symm — no row-level race.

Sync is the caller's job — sandwich each launch between `symm.barrier()`
so peers' writes are visible before loads and local writes finish before
any peer overwrites its own buffer.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_ALLOCATOR_REGISTERED = False


def _tma_allocator(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def ensure_tma_allocator() -> None:
    global _ALLOCATOR_REGISTERED
    if _ALLOCATOR_REGISTERED:
        return
    triton.set_allocator(_tma_allocator)
    _ALLOCATOR_REGISTERED = True


@triton.jit
def asymm_pull_seq_to_heads_kernel(
    peer_ptrs,  # *uint64, [WORLD]
    h_idxs_r,  # *int32, [H_LOCAL]
    recv_ptr,  # fp16/bf16 base pointer of [B, H_LOCAL, WORLD*S_LOCAL, D]
    B: tl.constexpr,
    H_TOTAL: tl.constexpr,
    S_LOCAL: tl.constexpr,
    D: tl.constexpr,
    H_LOCAL: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    S_BLOCK: tl.constexpr,
    NUM_SMS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    S_TILES: tl.constexpr = S_LOCAL // S_BLOCK
    T_PER_PEER: tl.constexpr = B * H_LOCAL * S_TILES

    recv_desc = tl.make_tensor_descriptor(
        recv_ptr,
        shape=[B, H_LOCAL, WORLD * S_LOCAL, D],
        strides=[H_LOCAL * WORLD * S_LOCAL * D, WORLD * S_LOCAL * D, D, 1],
        block_shape=[1, 1, S_BLOCK, D],
    )

    # Outer peer loop fully unrolled: descriptor built once per peer, not per tile.
    for peer_axis in tl.static_range(WORLD):
        peer = (peer_axis + RANK + 1) % WORLD  # swizzle: rank r starts at r+1
        src_ptr_u64 = tl.load(peer_ptrs + peer)
        src_ptr = src_ptr_u64.to(tl.pointer_type(DTYPE))
        src_desc = tl.make_tensor_descriptor(
            src_ptr,
            shape=[B, H_TOTAL, S_LOCAL, D],
            strides=[H_TOTAL * S_LOCAL * D, S_LOCAL * D, D, 1],
            block_shape=[1, 1, S_BLOCK, D],
        )

        # Inner: CTA strides through tiles within this peer.
        for tile in range(pid, T_PER_PEER, NUM_SMS):
            b = tile // (H_LOCAL * S_TILES)
            rem = tile % (H_LOCAL * S_TILES)
            lh = rem // S_TILES
            s_blk_in_peer = rem % S_TILES

            gh = tl.load(h_idxs_r + lh).to(tl.int32)
            data = src_desc.load([b, gh, s_blk_in_peer * S_BLOCK, 0])

            s_global = peer * S_LOCAL + s_blk_in_peer * S_BLOCK
            recv_desc.store([b, lh, s_global, 0], data)


_TRITON_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


def asymm_pull_seq_to_heads(
    peer_ptrs: torch.Tensor,  # [WORLD] uint64 on device
    h_idxs_r: torch.Tensor,  # [H_LOCAL] int32 on device
    recv: torch.Tensor,  # [B, H_LOCAL, WORLD*S_LOCAL, D]
    b: int,
    h_total: int,
    s_local: int,
    d: int,
    world_size: int,
    rank: int,
    s_block: int = 128,
    num_sms: int | None = None,
) -> None:
    """Launch the pull kernel. Caller must barrier before and after."""
    ensure_tma_allocator()

    if recv.dtype not in _TRITON_DTYPE:
        raise ValueError(f"Unsupported dtype {recv.dtype}")
    dtype = _TRITON_DTYPE[recv.dtype]

    h_local = h_idxs_r.numel()
    if s_local % s_block != 0:
        raise ValueError(f"s_local={s_local} not divisible by s_block={s_block}")

    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(recv.device).multi_processor_count

    grid = (num_sms,)
    asymm_pull_seq_to_heads_kernel[grid](
        peer_ptrs,
        h_idxs_r,
        recv,
        B=b,
        H_TOTAL=h_total,
        S_LOCAL=s_local,
        D=d,
        H_LOCAL=h_local,
        WORLD=world_size,
        RANK=rank,
        S_BLOCK=s_block,
        NUM_SMS=num_sms,
        DTYPE=dtype,
    )


@triton.jit
def asymm_push_heads_to_seq_kernel(
    peer_ptrs,  # *uint64, [WORLD] — peers' recv_symm base addresses
    h_idxs_r,  # *int32, [H_LOCAL] — this rank's global head indices
    src_ptr,  # local src [B, H_LOCAL, WORLD*S_LOCAL, D]
    B: tl.constexpr,
    H_TOTAL: tl.constexpr,
    S_LOCAL: tl.constexpr,
    D: tl.constexpr,
    H_LOCAL: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    S_BLOCK: tl.constexpr,
    NUM_SMS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    S_TILES: tl.constexpr = S_LOCAL // S_BLOCK
    T_PER_PEER: tl.constexpr = B * H_LOCAL * S_TILES

    src_desc = tl.make_tensor_descriptor(
        src_ptr,
        shape=[B, H_LOCAL, WORLD * S_LOCAL, D],
        strides=[H_LOCAL * WORLD * S_LOCAL * D, WORLD * S_LOCAL * D, D, 1],
        block_shape=[1, 1, S_BLOCK, D],
    )

    # Mirror image of the forward pull: same swizzle, descriptor-per-peer.
    # Each rank writes its H_LOCAL global-head rows into every peer's recv_symm,
    # at peer-local S range [0, S_LOCAL). Cross-rank heads are disjoint, so
    # different ranks' writes land on disjoint rows of the same recv_symm.
    for peer_axis in tl.static_range(WORLD):
        peer = (peer_axis + RANK + 1) % WORLD
        dst_ptr_u64 = tl.load(peer_ptrs + peer)
        dst_ptr = dst_ptr_u64.to(tl.pointer_type(DTYPE))
        dst_desc = tl.make_tensor_descriptor(
            dst_ptr,
            shape=[B, H_TOTAL, S_LOCAL, D],
            strides=[H_TOTAL * S_LOCAL * D, S_LOCAL * D, D, 1],
            block_shape=[1, 1, S_BLOCK, D],
        )

        for tile in range(pid, T_PER_PEER, NUM_SMS):
            b = tile // (H_LOCAL * S_TILES)
            rem = tile % (H_LOCAL * S_TILES)
            lh = rem // S_TILES
            s_blk_in_peer = rem % S_TILES

            gh = tl.load(h_idxs_r + lh).to(tl.int32)
            s_in_src = peer * S_LOCAL + s_blk_in_peer * S_BLOCK
            data = src_desc.load([b, lh, s_in_src, 0])
            dst_desc.store([b, gh, s_blk_in_peer * S_BLOCK, 0], data)


def asymm_push_heads_to_seq(
    peer_ptrs: torch.Tensor,  # [WORLD] uint64 on device
    h_idxs_r: torch.Tensor,  # [H_LOCAL] int32 on device
    src: torch.Tensor,  # [B, H_LOCAL, WORLD*S_LOCAL, D] (local)
    b: int,
    h_total: int,
    s_local: int,
    d: int,
    world_size: int,
    rank: int,
    s_block: int = 128,
    num_sms: int | None = None,
) -> None:
    """Launch the push kernel. Caller must barrier before and after."""
    ensure_tma_allocator()

    if src.dtype not in _TRITON_DTYPE:
        raise ValueError(f"Unsupported dtype {src.dtype}")
    dtype = _TRITON_DTYPE[src.dtype]

    h_local = h_idxs_r.numel()
    if s_local % s_block != 0:
        raise ValueError(f"s_local={s_local} not divisible by s_block={s_block}")

    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(src.device).multi_processor_count

    grid = (num_sms,)
    asymm_push_heads_to_seq_kernel[grid](
        peer_ptrs,
        h_idxs_r,
        src,
        B=b,
        H_TOTAL=h_total,
        S_LOCAL=s_local,
        D=d,
        H_LOCAL=h_local,
        WORLD=world_size,
        RANK=rank,
        S_BLOCK=s_block,
        NUM_SMS=num_sms,
        DTYPE=dtype,
    )
