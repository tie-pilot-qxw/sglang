"""Symmetric-memory wrapper for asymmetric all2all (forward pull + reverse push).

Forward (seq → heads) is pull. Reverse (heads → seq) is push. Push lets the
producer write its local attn_out straight into peers' `recv_symm` buffers;
combined with disjoint per-rank head ownership, no two ranks ever write the
same row, so a single barrier after the push is enough to publish — no
mid-iter sync between local-attn-finish and peer-reads.

Two-GPU N-rank simulation (`sim_world` / `sim_rank`):
  Real PG must have world_size=2. Each real GPU plays one `sim_rank` from a
  larger virtual world. The peer pointer table is built of length `sim_world`,
  with the self slot pointing at this GPU's symm buffer and every non-self slot
  aliasing the *other* real GPU's buffer. Kernels see WORLD=sim_world so all
  S/H stride math is faithful; what changes is bandwidth shape — instead of
  spreading (sim_world-1) flows across (sim_world-1) NVLinks, all (sim_world-1)
  per-rank flows funnel down the single real NVLink pair, so cross-rank
  throughput is roughly (sim_world-1)x slower than the real configuration.
  Latency / SM scheduling / sync surface area stay accurate.

Sync model for both directions: pre-barrier → kernel → post-barrier (each
defaulted to True for drop-in correctness; bench disables them and inserts a
single per-iter barrier when running the low-sync path).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm

from sglang.multimodal_gen.runtime.distributed.asymm_a2a_kernel import (
    asymm_pull_seq_to_heads,
    asymm_push_heads_to_seq,
)


class SymmAsymA2A:
    """Owns the per-rank symm buffers for Q/K/V (+ recv_symm for reverse) and
    dispatches the pull/push kernels.

    `buffer_shape` describes the per-rank shard in logical (unpadded) terms:
    `[B, H_total, S_local, D]`. Internally the S dim is rounded up to the next
    multiple of `s_block` (Triton TMA needs power-of-2 block shapes). Pad
    values are meaningless; the pull/push result's S dim is `WORLD * S_local_padded`
    and the caller trims down to the real `WORLD * S_local`.

    `s_block` must be a power of 2 (TMA block-shape constraint).

    Set `enable_reverse=True` to allocate `recv_symm` for `push_heads_to_seq`.

    Set `sim_world` + `sim_rank` to simulate a larger world on 2 real GPUs
    (real world_size must be 2). All kernel WORLD/RANK constexprs use the
    sim values; real `dist` PG calls (incl. `handle.barrier`) still operate
    on the real 2-rank group.
    """

    def __init__(
        self,
        group: dist.ProcessGroup | str,
        buffer_shape,
        dtype: torch.dtype,
        device: torch.device,
        s_block: int = 128,
        enable_reverse: bool = False,
        sim_world: Optional[int] = None,
        sim_rank: Optional[int] = None,
    ):
        if isinstance(group, dist.ProcessGroup):
            self.group_name = group.group_name
            self.group = group
        else:
            self.group_name = group
            self.group = dist.distributed_c10d._resolve_process_group(group)

        self.real_world_size = dist.get_world_size(self.group)
        self.real_rank = dist.get_rank(self.group)
        self.device = device
        self.dtype = dtype

        if (sim_world is None) != (sim_rank is None):
            raise ValueError("sim_world and sim_rank must be set together")
        if sim_world is not None:
            if self.real_world_size != 2:
                raise ValueError(
                    f"sim mode requires real world_size=2, got {self.real_world_size}"
                )
            if not (0 <= sim_rank < sim_world):
                raise ValueError(f"sim_rank={sim_rank} out of range [0, {sim_world})")
            self.sim_world = sim_world
            self.sim_rank = sim_rank
            self.world_size = sim_world
            self.rank = sim_rank
        else:
            self.sim_world = None
            self.sim_rank = None
            self.world_size = self.real_world_size
            self.rank = self.real_rank

        if s_block <= 0 or (s_block & (s_block - 1)) != 0:
            raise ValueError(f"s_block={s_block} must be a positive power of 2")
        self.s_block = s_block

        b, h_total, s_local, d = buffer_shape
        s_local_padded = ((s_local + s_block - 1) // s_block) * s_block
        if d & (d - 1) != 0:
            raise ValueError(f"d={d} must be a power of 2 for TMA block shape")
        self.b = b
        self.h_total = h_total
        self.s_local = s_local  # caller-facing logical size
        self.s_local_padded = s_local_padded
        self.d = d
        self.buffer_shape = (b, h_total, s_local_padded, d)

        self.q_symm = symm.empty(*self.buffer_shape, dtype=dtype, device=device)
        self.k_symm = symm.empty(*self.buffer_shape, dtype=dtype, device=device)
        self.v_symm = symm.empty(*self.buffer_shape, dtype=dtype, device=device)

        self.q_handle = symm.rendezvous(self.q_symm, self.group_name)
        self.k_handle = symm.rendezvous(self.k_symm, self.group_name)
        self.v_handle = symm.rendezvous(self.v_symm, self.group_name)

        self._symm: Dict[str, torch.Tensor] = {
            "q": self.q_symm,
            "k": self.k_symm,
            "v": self.v_symm,
        }
        self._handle = {
            "q": self.q_handle,
            "k": self.k_handle,
            "v": self.v_handle,
        }
        self._peer_ptrs: Dict[str, torch.Tensor] = {
            name: self._build_peer_ptrs(h) for name, h in self._handle.items()
        }

        self.recv_symm: Optional[torch.Tensor] = None
        self.recv_handle = None
        self.recv_peer_ptrs: Optional[torch.Tensor] = None
        if enable_reverse:
            self.recv_symm = symm.empty(*self.buffer_shape, dtype=dtype, device=device)
            self.recv_handle = symm.rendezvous(self.recv_symm, self.group_name)
            self.recv_peer_ptrs = self._build_peer_ptrs(self.recv_handle)

    def set_sim_rank(self, new_sim_rank: int) -> None:
        """Rebind peer_ptrs for a new sim_rank in 2-GPU sweep mode.

        Same physical buffers (no re-rendezvous), only the peer pointer table
        is rebuilt: slot `new_sim_rank` becomes self, every other slot points
        to the OTHER real GPU. Use to play multiple sim_ranks sequentially on
        one real GPU and measure each.
        """
        if self.sim_world is None:
            raise RuntimeError(
                "set_sim_rank requires sim mode (pass sim_world to ctor)"
            )
        if not (0 <= new_sim_rank < self.sim_world):
            raise ValueError(
                f"new_sim_rank={new_sim_rank} out of [0, {self.sim_world})"
            )
        self.sim_rank = new_sim_rank
        self.rank = new_sim_rank
        self._peer_ptrs = {
            name: self._build_peer_ptrs(h) for name, h in self._handle.items()
        }
        if self.recv_handle is not None:
            self.recv_peer_ptrs = self._build_peer_ptrs(self.recv_handle)

    def _build_peer_ptrs(self, handle) -> torch.Tensor:
        """Materialize the [WORLD] uint64 pointer table the kernels load.

        Real mode: just `handle.buffer_ptrs` (one entry per real rank).
        Sim mode: length is `sim_world`; self slot points at this GPU's buffer,
        every other slot aliases the OTHER real GPU's buffer.
        """
        real_ptrs = list(handle.buffer_ptrs)
        if self.sim_world is None:
            return torch.tensor(real_ptrs, dtype=torch.int64, device=self.device)
        # Sim mode: real world is 2.
        self_ptr = real_ptrs[self.real_rank]
        other_ptr = real_ptrs[1 - self.real_rank]
        ptrs = [
            self_ptr if p == self.sim_rank else other_ptr for p in range(self.sim_world)
        ]
        return torch.tensor(ptrs, dtype=torch.int64, device=self.device)

    def peer_ptrs(self, name: str) -> torch.Tensor:
        return self._peer_ptrs[name]

    def barrier(self, channel: int = 0) -> None:
        """Group-wide symm-mem barrier on the real PG. `handle.barrier()`
        syncs the whole process group regardless of which buffer's handle is
        used, so we pick `q_handle` arbitrarily.

        In sim mode this is still a 2-rank barrier (real PG size), not an
        sim_world-rank one — so the absolute barrier latency is optimistic
        relative to a real N-rank run. Document at call sites.
        """
        self.q_handle.barrier(channel=channel)

    def pull_seq_to_heads(
        self,
        name: str,
        h_idxs_r: torch.Tensor,
        num_sms: Optional[int] = None,
        out: Optional[torch.Tensor] = None,
        pre_barrier: bool = True,
        post_barrier: bool = True,
    ) -> torch.Tensor:
        """Pull this rank's heads (`h_idxs_r`, global head indices) from every
        peer's `{name}_symm` buffer. Returns `[B, H_local_r, WORLD*S_local, D]`.

        Pre-condition: caller has populated `self.{name}_symm` on every rank.
        """
        if name not in self._handle:
            raise KeyError(f"unknown buffer {name!r}; expected one of q/k/v")
        handle = self._handle[name]
        peer_ptrs = self._peer_ptrs[name]
        h_local = h_idxs_r.numel()

        if h_idxs_r.dtype != torch.int32:
            h_idxs_r = h_idxs_r.to(dtype=torch.int32)
        if h_idxs_r.device != self.device:
            h_idxs_r = h_idxs_r.to(self.device)

        if out is None:
            out = torch.empty(
                (self.b, h_local, self.world_size * self.s_local_padded, self.d),
                dtype=self.dtype,
                device=self.device,
            )

        if pre_barrier:
            handle.barrier(channel=0)

        asymm_pull_seq_to_heads(
            peer_ptrs=peer_ptrs,
            h_idxs_r=h_idxs_r,
            recv=out,
            b=self.b,
            h_total=self.h_total,
            s_local=self.s_local_padded,
            d=self.d,
            world_size=self.world_size,
            rank=self.rank,
            s_block=self.s_block,
            num_sms=num_sms,
        )

        if post_barrier:
            handle.barrier(channel=0)
        return out

    def push_heads_to_seq(
        self,
        src: torch.Tensor,
        h_idxs_r: torch.Tensor,
        num_sms: Optional[int] = None,
        pre_barrier: bool = False,
        post_barrier: bool = True,
    ) -> torch.Tensor:
        """Push this rank's local attn_out into every peer's `recv_symm` at
        the right global-head rows. Returns this rank's `recv_symm` view at
        `[B, H_total, S_local_padded, D]` (caller trims to real S_local).

        `src` shape: `[B, H_local_r, WORLD * S_local_padded, D]`. Caller is
        responsible for padding each per-peer S segment to `s_local_padded`
        (the kernel reads `S_LOCAL = s_local_padded` per-peer slabs).

        Default sync model is `pre_barrier=False, post_barrier=True` — push
        only needs *one* barrier after the launch to publish writes; pre-sync
        is unnecessary because the dst buffer was barriered by the previous
        iter's post_barrier (or by the caller's setup barrier on iter 0).
        """
        if self.recv_symm is None:
            raise RuntimeError(
                "push_heads_to_seq requires reverse setup; pass "
                "enable_reverse=True to SymmAsymA2A(...)"
            )

        h_local = h_idxs_r.numel()
        expected_shape = (
            self.b,
            h_local,
            self.world_size * self.s_local_padded,
            self.d,
        )
        if tuple(src.shape) != expected_shape:
            raise ValueError(
                f"src shape {tuple(src.shape)} != expected {expected_shape}"
            )
        if src.dtype != self.dtype:
            raise ValueError(f"src dtype {src.dtype} != {self.dtype}")

        if h_idxs_r.dtype != torch.int32:
            h_idxs_r = h_idxs_r.to(dtype=torch.int32)
        if h_idxs_r.device != self.device:
            h_idxs_r = h_idxs_r.to(self.device)

        if pre_barrier:
            self.recv_handle.barrier(channel=0)

        asymm_push_heads_to_seq(
            peer_ptrs=self.recv_peer_ptrs,
            h_idxs_r=h_idxs_r,
            src=src,
            b=self.b,
            h_total=self.h_total,
            s_local=self.s_local_padded,
            d=self.d,
            world_size=self.world_size,
            rank=self.rank,
            s_block=self.s_block,
            num_sms=num_sms,
        )

        if post_barrier:
            self.recv_handle.barrier(channel=0)
        return self.recv_symm
