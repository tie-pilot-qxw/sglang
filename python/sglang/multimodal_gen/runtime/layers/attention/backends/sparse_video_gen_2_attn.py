"""
Sparse Video Gen 2 (SAP) attention backend.

This is a baseline integration that wires the backend into the
attention framework.

Adapted from https://github.com/svg-project/Sparse-VideoGen/blob/main/svg/models/wan/attention.py
"""

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    from svg.kernels.triton.permute import (
        apply_inverse_permutation_triton,
        permute_tensor_by_labels_triton,
    )
    from svg.kmeans_utils import (
        batch_kmeans_Euclid,
        density_calculation,
        dynamic_block_sparse_fwd_flashinfer,
        identify_dynamic_map,
    )

    svg2_available = True
except ImportError:
    svg2_available = False

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SparseVideoGen2AttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SPARSE_VIDEO_GEN_2_ATTN

    @staticmethod
    def get_impl_cls() -> type["SparseVideoGen2AttentionImpl"]:
        return SparseVideoGen2AttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["SparseVideoGen2AttentionMetadata"]:
        return SparseVideoGen2AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["SparseVideoGen2AttentionMetadataBuilder"]:
        return SparseVideoGen2AttentionMetadataBuilder


@dataclass
class Svg2LayerCache:
    # centroids for kmeans clustering
    q_centroids: torch.Tensor | None = None
    k_centroids: torch.Tensor | None = None
    centroids_initialized: bool = False

    # gathered global centroids
    q_centroids_global: torch.Tensor | None = None
    k_centroids_global: torch.Tensor | None = None

    # density cache for head reordering (if head parallel is enabled, we need to do load balancing)
    density: torch.Tensor | None = None  # [num_heads]
    density_async_complete: torch.cuda.Event | None = None
    density_gpu: torch.Tensor | None = None

    # Flat head permutation (length = num_heads). Concatenation of per-rank
    # head bins under the current LPT placement. Defines the equivalent of
    # `head_perm[r*hpr:(r+1)*hpr]` for the legacy equal-cap path; for the
    # unequal LPT path the per-rank slice has length heads_per_rank[r].
    head_perm: torch.Tensor | None = None
    head_perm_inv: torch.Tensor | None = None
    # Per-rank head counts under the current LPT placement (sum == num_heads).
    # None until the first density-driven reassignment fires.
    heads_per_rank: list[int] | None = None
    # Flat int32 of bins[cur_rank] = global head ids this rank owns. Cached on
    # device for the asymm pull/push kernels (h_idxs_r). Re-derived whenever
    # head_perm/heads_per_rank change.
    h_idxs_r_dev: torch.Tensor | None = None


@dataclass
class Svg2Cache:
    planning_stream: torch.cuda.Stream
    layers: dict[int, Svg2LayerCache] = field(default_factory=dict)
    # Cached SymmAsymA2A wrapper, keyed by (b, h_total, s_local, d, dtype).
    # Allocation requires symm.empty + rendezvous (full PG barrier) so we
    # share one wrapper across all layers / steps with the same shape.
    symm_a2a: Any = None
    symm_a2a_key: tuple | None = None

    def get_layer(self, layer_idx: int) -> Svg2LayerCache:
        layer_cache = self.layers.get(layer_idx)
        if layer_cache is None:
            layer_cache = Svg2LayerCache()
            self.layers[layer_idx] = layer_cache
        return layer_cache

    def get_or_create_symm_a2a(
        self,
        group: "dist.ProcessGroup",
        b: int,
        h_total: int,
        s_local: int,
        d: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        from sglang.multimodal_gen.runtime.distributed.symm_asym_a2a import SymmAsymA2A

        key = (b, h_total, s_local, d, dtype)
        if self.symm_a2a is not None and self.symm_a2a_key == key:
            return self.symm_a2a
        # Drop the stale wrapper before allocating a new one (frees symm bufs).
        self.symm_a2a = None
        self.symm_a2a = SymmAsymA2A(
            group=group,
            buffer_shape=(b, h_total, s_local, d),
            dtype=dtype,
            device=device,
            enable_reverse=True,
        )
        self.symm_a2a_key = key
        return self.symm_a2a


@dataclass
class SparseVideoGen2AttentionMetadata(AttentionMetadata):
    current_timestep: int
    num_q_centroids: int
    num_k_centroids: int
    top_p_kmeans: float
    min_kc_ratio: float
    kmeans_iter_init: int
    kmeans_iter_step: int
    zero_step_kmeans_init: bool
    first_layers_fp: float
    first_times_fp: float
    context_length: int
    num_frame: int
    frame_size: int
    cache: Svg2Cache
    # SP head load balancing strategy. See ServerArgs.svg2_load_balance.
    # "off"           - no LB
    # "equal"         - LPT with equal heads/rank + symm a2a
    # "unequal_asymm" - LPT (variable heads/rank) + asymm pull/push
    load_balance: str = "off"
    prompt_length: int | None = None
    max_seqlen_q: int | None = None
    max_seqlen_k: int | None = None


def _even_heads_per_rank(num_heads: int, world_size: int) -> list[int]:
    """Distribute num_heads across world_size as evenly as possible.

    The first `num_heads % world_size` ranks get one extra head. Returned
    list always sums to num_heads, so it works for non-divisible counts.
    """
    base, extra = divmod(num_heads, world_size)
    return [base + (1 if r < extra else 0) for r in range(world_size)]


def _lpt_assignment_unequal(
    densities: list[float],
    world_size: int,
    min_heads_per_rank: int = 1,
) -> tuple[list[int], list[int]]:
    """Pure LPT (no equal-heads constraint), seeded with min_heads_per_rank
    round-robin assignments so no rank stays empty. Returns
    (flat_perm, heads_per_rank) where flat_perm is the concatenation of
    per-rank head bins (each bin sorted ascending) and heads_per_rank gives
    the per-rank count.

    Adapted from Sparse-VideoGen/scripts/wan/bench_wan_sp_all2all_attention.py
    `greedy_lpt_assignment`.
    """
    n = len(densities)
    if min_heads_per_rank * world_size > n:
        raise ValueError(
            f"min_heads_per_rank={min_heads_per_rank} * world_size={world_size} "
            f"> n_heads={n}"
        )
    loads = [0.0] * world_size
    bins: list[list[int]] = [[] for _ in range(world_size)]
    order = sorted(range(n), key=lambda i: -densities[i])

    # Seed: give each rank `min_heads_per_rank` heads round-robin from the top.
    idx = 0
    for _ in range(min_heads_per_rank):
        for r in range(world_size):
            head = order[idx]
            bins[r].append(head)
            loads[r] += densities[head]
            idx += 1

    # Remaining heads: pure LPT — least-loaded rank wins.
    for head in order[idx:]:
        best = min(range(world_size), key=lambda r: loads[r])
        bins[best].append(head)
        loads[best] += densities[head]

    for r in range(world_size):
        bins[r].sort()

    flat = [h for r in range(world_size) for h in bins[r]]
    return flat, [len(bins[r]) for r in range(world_size)]


def _require_kwarg(kwargs: dict[str, Any], name: str) -> Any:
    if name not in kwargs:
        raise ValueError(
            f"Missing required argument for SparseVideoGen2Attention: {name}"
        )
    return kwargs[name]


class SparseVideoGen2AttentionMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(  # type: ignore[override]
        self,
        current_timestep: int,
        raw_latent_shape: tuple[int, ...],
        patch_size: tuple[int, int, int],
        cache: Svg2Cache,
        num_q_centroids: int,
        num_k_centroids: int,
        top_p_kmeans: float,
        min_kc_ratio: float,
        kmeans_iter_init: int,
        kmeans_iter_step: int,
        zero_step_kmeans_init: bool,
        first_layers_fp: float,
        first_times_fp: float,
        context_length: int = 0,
        prompt_length: int | None = None,
        load_balance: str = "off",
        **kwargs: dict[str, Any],
    ) -> SparseVideoGen2AttentionMetadata:
        raw_shape = tuple(raw_latent_shape)
        if len(raw_shape) == 5:
            t, h, w = raw_shape[2:5]
        elif len(raw_shape) == 3:
            t, h, w = raw_shape
        else:
            raise ValueError(
                "raw_latent_shape must be (T, H, W) or (B, C, T, H, W) for SAP attention"
            )
        pt, ph, pw = patch_size
        if t % pt != 0 or h % ph != 0 or w % pw != 0:
            raise ValueError(
                "raw_latent_shape must be divisible by patch_size for SAP attention"
            )

        num_frame = t // pt
        frame_size = (h // ph) * (w // pw)

        return SparseVideoGen2AttentionMetadata(
            current_timestep=current_timestep,
            num_q_centroids=num_q_centroids,
            num_k_centroids=num_k_centroids,
            top_p_kmeans=top_p_kmeans,
            min_kc_ratio=min_kc_ratio,
            kmeans_iter_init=kmeans_iter_init,
            kmeans_iter_step=kmeans_iter_step,
            zero_step_kmeans_init=zero_step_kmeans_init,
            first_layers_fp=first_layers_fp,
            first_times_fp=first_times_fp,
            context_length=context_length,
            prompt_length=prompt_length,
            num_frame=num_frame,
            frame_size=frame_size,
            cache=cache,
            load_balance=load_balance,
        )


class SparseVideoGen2AttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        if causal:
            raise ValueError(
                "Sparse Video Gen 2 attention does not support causal attention"
            )
        if not svg2_available:
            raise ImportError(
                "Sparse Video Gen 2 attention backend requires svg package to be installed"
            )
        self.prefix = prefix
        self.num_heads = num_heads
        self.layer_idx = self._get_layer_idx(prefix)

    def _get_layer_idx(self, prefix: str) -> int:
        parts = prefix.split(".")
        if len(parts) < 3:
            raise ValueError(
                f"Invalid prefix for SparseVideoGen2AttentionImpl: {prefix}"
            )
        return int(parts[-3])

    def _use_sparse_attention(
        self, attn_metadata: SparseVideoGen2AttentionMetadata
    ) -> bool:
        if self.layer_idx < attn_metadata.first_layers_fp:
            return False
        if attn_metadata.current_timestep > attn_metadata.first_times_fp:
            return False
        return True

    def _compute_head_reorder_perm(
        self,
        density_cpu: torch.Tensor | None,
        num_heads: int,
        world_size: int,
        mode: str,
    ) -> tuple[list[int], list[int]]:
        """Returns (flat_perm, heads_per_rank).

        flat_perm: permutation of range(num_heads), concatenation of per-rank
        head bins. Per-rank slice `flat_perm[sum(hpr[:r]) : sum(hpr[:r+1])]`
        gives rank r's owned global head ids.

        heads_per_rank: list of length world_size with sum == num_heads.

        mode:
          "unequal_asymm" - unequal LPT (variable heads_per_rank).
          "equal" / other - equal-cap LPT + local search; heads_per_rank
                            uniform = num_heads // world_size.
        """
        if density_cpu is None:
            return list(range(num_heads)), _even_heads_per_rank(num_heads, world_size)

        if mode == "unequal_asymm":
            return _lpt_assignment_unequal(
                [float(density_cpu[i].item()) for i in range(num_heads)],
                world_size,
            )

        # weight: higher density => more work
        w = [float(density_cpu[i].item()) for i in range(num_heads)]

        # LPT order
        sorted_heads = sorted(range(num_heads), key=lambda i: (-w[i], i))

        # Equal-cap LPT requires divisibility — the routing layer
        # (sequence_model_parallel_all_to_all_4D) can't reshape otherwise.
        # For non-divisible num_heads use --svg2-load-balance unequal_asymm.
        if num_heads % world_size != 0:
            raise ValueError(
                f"--svg2-load-balance equal requires num_heads ({num_heads}) "
                f"divisible by sp world_size ({world_size}); use unequal_asymm "
                "for non-divisible head counts."
            )
        heads_per_rank = num_heads // world_size

        bins: list[list[int]] = [[] for _ in range(world_size)]
        sums = [0.0 for _ in range(world_size)]
        sizes = [0 for _ in range(world_size)]

        # ---- fix tie-bias: rotate scan start each call ----
        rr = getattr(self, "_head_lb_rr_cursor", 0) % world_size
        setattr(self, "_head_lb_rr_cursor", rr + 1)

        # ---- greedy pack (LPT + least-sum) ----
        for h in sorted_heads:
            best_r = None
            best_s = None
            for k in range(world_size):
                r = (rr + k) % world_size
                if sizes[r] >= heads_per_rank:
                    continue
                s = sums[r]
                if best_s is None or s < best_s - 1e-12:
                    best_s = s
                    best_r = r
            if best_r is None:
                break
            bins[best_r].append(h)
            sums[best_r] += w[h]
            sizes[best_r] += 1

        # safety: should always be full (equal-cap path requires divisibility,
        # asserted earlier, so _even_heads_per_rank returns uniform here).
        if sum(sizes) != num_heads:
            return list(range(num_heads)), _even_heads_per_rank(num_heads, world_size)

        def makespan() -> float:
            return max(sums)

        def argmax(a):
            m = max(a)
            return a.index(m)

        def argmin(a):
            m = min(a)
            return a.index(m)

        # ---- local improvement: few passes are enough for H~40, W<=8 ----
        for _ in range(6):
            improved = False
            T = makespan()
            rh = argmax(sums)
            rl = argmin(sums)

            # (A) 1-move: heavy -> light
            if bins[rh] and sizes[rl] < heads_per_rank:
                best_h = None
                best_T = T
                for h in bins[rh]:
                    s_h = sums[rh] - w[h]
                    s_l = sums[rl] + w[h]
                    new_T = max(
                        s_h,
                        s_l,
                        *(sums[r] for r in range(world_size) if r not in (rh, rl)),
                    )
                    if new_T < best_T - 1e-12:
                        best_T = new_T
                        best_h = h
                if best_h is not None:
                    bins[rh].remove(best_h)
                    bins[rl].append(best_h)
                    sums[rh] -= w[best_h]
                    sums[rl] += w[best_h]
                    sizes[rh] -= 1
                    sizes[rl] += 1
                    improved = True

            if improved:
                continue

            # (B) swap: heavy with someone else
            rh = argmax(sums)
            T = makespan()
            best = None  # (r2, h1, h2, newT)

            for r2 in range(world_size):
                if r2 == rh or not bins[r2]:
                    continue
                for h1 in bins[rh]:
                    for h2 in bins[r2]:
                        s1 = sums[rh] - w[h1] + w[h2]
                        s2 = sums[r2] - w[h2] + w[h1]
                        new_T = max(
                            s1,
                            s2,
                            *(sums[r] for r in range(world_size) if r not in (rh, r2)),
                        )
                        if new_T < T - 1e-12 and (
                            best is None or new_T < best[3] - 1e-12
                        ):
                            best = (r2, h1, h2, new_T)

            if best is not None:
                r2, h1, h2, _ = best
                bins[rh].remove(h1)
                bins[r2].remove(h2)
                bins[rh].append(h2)
                bins[r2].append(h1)
                sums[rh] += -w[h1] + w[h2]
                sums[r2] += -w[h2] + w[h1]
                improved = True

            if not improved:
                break

        if get_sequence_parallel_rank() == 0:
            print(f"densities: {list(density_cpu)}")
            print(f"sums: {list(sums)}")
            T = max(sums)
            avg = sum(sums) / world_size
            print("makespan", T, "avg", avg, "imb", T / avg)

        # keep stable head order within each rank (optional)
        for r in range(world_size):
            bins[r].sort()

        flat = [h for r in range(world_size) for h in bins[r]]
        return flat, [len(bins[r]) for r in range(world_size)]

    def _get_head_reorder_perm(
        self, attn_metadata: SparseVideoGen2AttentionMetadata, world_size: int
    ) -> torch.Tensor:
        layer_cache = attn_metadata.cache.get_layer(self.layer_idx)
        num_heads = self.num_heads
        cur_rank = get_sequence_parallel_rank()

        if layer_cache.density_async_complete is None:
            head_perm = torch.arange(
                num_heads, device=torch.device("cuda"), dtype=torch.long
            )
            heads_per_rank = _even_heads_per_rank(num_heads, world_size)
            ofs = sum(heads_per_rank[:cur_rank])
            layer_cache.head_perm = head_perm
            layer_cache.head_perm_inv = head_perm
            layer_cache.heads_per_rank = heads_per_rank
            layer_cache.h_idxs_r_dev = head_perm[
                ofs : ofs + heads_per_rank[cur_rank]
            ].to(dtype=torch.int32)
            return head_perm

        torch.cuda.current_stream().wait_event(layer_cache.density_async_complete)
        assert layer_cache.density is not None

        perm_list, heads_per_rank = self._compute_head_reorder_perm(
            layer_cache.density, num_heads, world_size, attn_metadata.load_balance
        )
        head_perm = torch.tensor(
            perm_list, device=torch.device("cuda"), dtype=torch.long
        )
        head_perm_inv = torch.empty_like(head_perm)
        head_perm_inv[head_perm] = torch.arange(num_heads, device=head_perm.device)
        layer_cache.head_perm = head_perm
        layer_cache.head_perm_inv = head_perm_inv
        layer_cache.heads_per_rank = heads_per_rank

        # This rank's owned global head ids = bins[cur_rank]. With unequal LPT
        # the per-rank slice has variable length; fall back to equal cumsum
        # under the symm path where heads_per_rank is uniform.
        ofs = sum(heads_per_rank[:cur_rank])
        head_ids = head_perm[ofs : ofs + heads_per_rank[cur_rank]]
        layer_cache.h_idxs_r_dev = head_ids.to(dtype=torch.int32)

        if (
            layer_cache.q_centroids_global is not None
            and layer_cache.k_centroids_global is not None
        ):
            q_clusters = layer_cache.q_centroids_global.size(1)
            q_dim = layer_cache.q_centroids_global.size(2)
            k_clusters = layer_cache.k_centroids_global.size(1)
            k_dim = layer_cache.k_centroids_global.size(2)

            q_global = layer_cache.q_centroids_global.view(
                -1, num_heads, q_clusters, q_dim
            )
            k_global = layer_cache.k_centroids_global.view(
                -1, num_heads, k_clusters, k_dim
            )
            layer_cache.q_centroids = q_global.index_select(1, head_ids).reshape(
                -1, q_clusters, q_dim
            )
            layer_cache.k_centroids = k_global.index_select(1, head_ids).reshape(
                -1, k_clusters, k_dim
            )

        return head_perm

    def preprocess_qkv_before_all_to_all(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: SparseVideoGen2AttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._use_sparse_attention(attn_metadata):
            return q, k, v
        world_size = get_sequence_parallel_world_size()
        head_perm = self._get_head_reorder_perm(attn_metadata, world_size)
        q = q.index_select(dim=2, index=head_perm)
        k = k.index_select(dim=2, index=head_perm)
        v = v.index_select(dim=2, index=head_perm)
        return q, k, v

    def postprocess_output_after_all_to_all(
        self, output: torch.Tensor, attn_metadata: SparseVideoGen2AttentionMetadata
    ) -> torch.Tensor:
        if not self._use_sparse_attention(attn_metadata):
            return output
        layer_cache = attn_metadata.cache.get_layer(self.layer_idx)
        assert (
            layer_cache.head_perm_inv is not None
        ), "Head permutation inverse not computed."
        return output.index_select(dim=2, index=layer_cache.head_perm_inv)

    def _launch_density_async(
        self,
        local_density: torch.Tensor,
        attn_metadata: SparseVideoGen2AttentionMetadata,
    ) -> None:
        layer_cache = attn_metadata.cache.get_layer(self.layer_idx)

        # this should after the first kmeans clustering
        assert layer_cache.q_centroids is not None
        assert layer_cache.k_centroids is not None

        world_size = get_sequence_parallel_world_size()
        if world_size <= 1:
            layer_cache.density = local_density.detach()
            return

        num_local_heads = local_density.numel()

        # Per-rank head counts that produced the local density / centroids.
        # heads_per_rank is set by the previous _get_head_reorder_perm call;
        # if absent (very first density gather) fall back to even split,
        # which handles non-divisible num_heads.
        if layer_cache.heads_per_rank is not None:
            heads_per_rank = layer_cache.heads_per_rank
            num_heads = sum(heads_per_rank)
        else:
            num_heads = self.num_heads
            heads_per_rank = _even_heads_per_rank(num_heads, world_size)
        max_hpr = max(heads_per_rank)

        if layer_cache.density_async_complete is None:
            layer_cache.density_async_complete = torch.cuda.Event()
        if layer_cache.density is None:
            layer_cache.density = torch.empty(
                [num_heads], dtype=local_density.dtype, device="cpu"
            ).pin_memory()

        planning_stream = attn_metadata.cache.planning_stream
        planning_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(planning_stream):
            # Build pos_of_head[gh] = r * max_hpr + lh: the position of global
            # head gh in the gathered+padded `[world_size * max_hpr, ...]` tensor.
            # In the equal-cap case (heads_per_rank uniform == num_local_heads,
            # max_hpr == num_local_heads, no padding) this reduces to the legacy
            # head_perm_inv. With unequal LPT the padded slots are never read
            # because pos_of_head only maps real heads.
            if layer_cache.head_perm is not None:
                head_perm_cpu = layer_cache.head_perm.tolist()
            else:
                head_perm_cpu = list(range(num_heads))
            pos_of_head_list = [0] * num_heads
            ofs = 0
            for r in range(world_size):
                hpr = heads_per_rank[r]
                for lh in range(hpr):
                    gh = head_perm_cpu[ofs + lh]
                    pos_of_head_list[gh] = r * max_hpr + lh
                ofs += hpr
            pos_of_head = torch.tensor(
                pos_of_head_list, device=local_density.device, dtype=torch.long
            )

            _, num_q_centroids, q_dim = layer_cache.q_centroids.size()
            _, num_k_centroids, k_dim = layer_cache.k_centroids.size()

            # reshape to [B, H_local_r, C, D]
            q_centroids = layer_cache.q_centroids.reshape(
                -1,
                num_local_heads,
                num_q_centroids,
                q_dim,
            ).contiguous()
            k_centroids = layer_cache.k_centroids.reshape(
                -1,
                num_local_heads,
                num_k_centroids,
                k_dim,
            ).contiguous()

            sp_group = get_sp_group().device_group

            # Pad local tensors to max_hpr along H so all_gather_into_tensor
            # (which requires uniform per-rank shape) works under variable LPT.
            def _pad_h(t: torch.Tensor, h_dim: int) -> torch.Tensor:
                pad_n = max_hpr - t.size(h_dim)
                if pad_n == 0:
                    return t.contiguous()
                pad_shape = list(t.shape)
                pad_shape[h_dim] = pad_n
                pad = t.new_zeros(pad_shape)
                return torch.cat([t, pad], dim=h_dim).contiguous()

            if num_local_heads < max_hpr:
                local_density_padded = torch.cat(
                    [local_density, local_density.new_zeros(max_hpr - num_local_heads)]
                ).contiguous()
            else:
                local_density_padded = local_density.contiguous()

            q_centroids_padded = _pad_h(q_centroids, 1)
            k_centroids_padded = _pad_h(k_centroids, 1)

            # gather density in (rank, padded local head) order, then index
            # back to global head id.
            density_gather = torch.empty(
                (world_size, max_hpr),
                device=local_density.device,
                dtype=local_density.dtype,
            )
            dist.all_gather_into_tensor(
                density_gather, local_density_padded, group=sp_group
            )
            density_all = density_gather.reshape(world_size * max_hpr)
            layer_cache.density_gpu = density_all.index_select(0, pos_of_head)

            # gather centroids and reorder to global head ids
            q_gather = torch.empty(
                (world_size,) + q_centroids_padded.shape,
                device=q_centroids_padded.device,
                dtype=q_centroids_padded.dtype,
            )
            dist.all_gather_into_tensor(q_gather, q_centroids_padded, group=sp_group)
            q_all = q_gather.permute(1, 0, 2, 3, 4).reshape(
                -1, world_size * max_hpr, num_q_centroids, q_dim
            )
            layer_cache.q_centroids_global = q_all.index_select(1, pos_of_head).reshape(
                -1, num_q_centroids, q_dim
            )

            k_gather = torch.empty(
                (world_size,) + k_centroids_padded.shape,
                device=k_centroids_padded.device,
                dtype=k_centroids_padded.dtype,
            )
            dist.all_gather_into_tensor(k_gather, k_centroids_padded, group=sp_group)
            k_all = k_gather.permute(1, 0, 2, 3, 4).reshape(
                -1, world_size * max_hpr, num_k_centroids, k_dim
            )
            layer_cache.k_centroids_global = k_all.index_select(1, pos_of_head).reshape(
                -1, num_k_centroids, k_dim
            )

            # copy back to cpu
            layer_cache.density.copy_(layer_cache.density_gpu, non_blocking=True)
            layer_cache.density_async_complete.record(planning_stream)

    def kmeans_init(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attn_metadata: SparseVideoGen2AttentionMetadata,
    ):
        cfg, num_heads, seq_len, dim = query.size()
        qlabels, qcentroids, qcluster_sizes, qiter = batch_kmeans_Euclid(
            query.reshape(cfg * num_heads, seq_len, dim),
            n_clusters=attn_metadata.num_q_centroids,
            max_iters=attn_metadata.kmeans_iter_init,
        )
        klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(
            key.reshape(cfg * num_heads, seq_len, dim),
            n_clusters=attn_metadata.num_k_centroids,
            max_iters=attn_metadata.kmeans_iter_init,
        )

        layer_cache = attn_metadata.cache.get_layer(self.layer_idx)
        layer_cache.q_centroids = qcentroids
        layer_cache.k_centroids = kcentroids

        return (
            qlabels,
            qcentroids,
            qcluster_sizes,
            qiter,
            klabels,
            kcentroids,
            kcluster_sizes,
            kiter,
        )

    def kmeans_step(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attn_metadata: SparseVideoGen2AttentionMetadata,
    ):
        cfg, num_heads, seq_len, dim = query.size()
        layer_cache = attn_metadata.cache.get_layer(self.layer_idx)
        qlabels, qcentroids, qcluster_sizes, qiter = batch_kmeans_Euclid(
            query.reshape(cfg * num_heads, seq_len, dim),
            n_clusters=attn_metadata.num_q_centroids,
            max_iters=attn_metadata.kmeans_iter_step,
            init_centroids=layer_cache.q_centroids,
        )
        klabels, kcentroids, kcluster_sizes, kiter = batch_kmeans_Euclid(
            key.reshape(cfg * num_heads, seq_len, dim),
            n_clusters=attn_metadata.num_k_centroids,
            max_iters=attn_metadata.kmeans_iter_step,
            init_centroids=layer_cache.k_centroids,
        )

        layer_cache.q_centroids = qcentroids
        layer_cache.k_centroids = kcentroids

        return (
            qlabels,
            qcentroids,
            qcluster_sizes,
            qiter,
            klabels,
            kcentroids,
            kcluster_sizes,
            kiter,
        )

    def kmeans_clustering(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attn_metadata: SparseVideoGen2AttentionMetadata,
    ):
        layer_cache = attn_metadata.cache.get_layer(self.layer_idx)
        if not layer_cache.centroids_initialized:
            (
                qlabels,
                qcentroids,
                qcluster_sizes,
                qiter,
                klabels,
                kcentroids,
                kcluster_sizes,
                kiter,
            ) = self.kmeans_init(query, key, attn_metadata)
            layer_cache.centroids_initialized = True
            logger.debug(
                "Centroids initialized at layer %s (init iters: %s).",
                self.layer_idx,
                attn_metadata.kmeans_iter_init,
            )
        else:
            (
                qlabels,
                qcentroids,
                qcluster_sizes,
                qiter,
                klabels,
                kcentroids,
                kcluster_sizes,
                kiter,
            ) = self.kmeans_step(query, key, attn_metadata)

        return (
            qlabels,
            qcentroids,
            qcluster_sizes,
            qiter,
            klabels,
            kcentroids,
            kcluster_sizes,
            kiter,
        )

    def semantic_aware_permutation(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: SparseVideoGen2AttentionMetadata,
    ):
        cfg, num_heads, seq_len, dim = query.size()

        # 1. Kmeans clustering
        (
            qlabels,
            qcentroids,
            qcluster_sizes,
            qiter,
            klabels,
            kcentroids,
            kcluster_sizes,
            kiter,
        ) = self.kmeans_clustering(query, key, attn_metadata)

        # 2. Identify dynamic map
        q_cluster_sizes = qcluster_sizes.view(
            cfg, num_heads, attn_metadata.num_q_centroids
        )
        k_cluster_sizes = kcluster_sizes.view(
            cfg, num_heads, attn_metadata.num_k_centroids
        )

        dynamic_map = identify_dynamic_map(
            qcentroids.view(cfg, num_heads, attn_metadata.num_q_centroids, dim),
            kcentroids.view(cfg, num_heads, attn_metadata.num_k_centroids, dim),
            q_cluster_sizes,
            k_cluster_sizes,
            attn_metadata.top_p_kmeans,
            attn_metadata.min_kc_ratio,
        )

        # 3. Permute the query, key, value
        q_permuted, q_sorted_indices = permute_tensor_by_labels_triton(
            query, qlabels, dim=2
        )
        k_permuted, k_sorted_indices = permute_tensor_by_labels_triton(
            key, klabels, dim=2
        )
        v_permuted, v_sorted_indices = permute_tensor_by_labels_triton(
            value, klabels, dim=2, sorted_indices=k_sorted_indices
        )

        return (
            q_permuted,
            k_permuted,
            v_permuted,
            dynamic_map,
            q_cluster_sizes,
            k_cluster_sizes,
            q_sorted_indices,
        )

    def _hunyuan_dynamic_map_post_processing(
        self,
        q_perm: torch.Tensor,
        k_perm: torch.Tensor,
        v_perm: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dyn_map: torch.Tensor,
        qc_sz_s: torch.Tensor,
        kc_sz_s: torch.Tensor,
        q_sorted_indices: torch.Tensor,
        video_length: int,
        context_length: int,
        prompt_length: int,
        unprompt_length: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # Place the permuted video tokens back and keep text tokens at the tail.
        query[:, :, :-context_length, :] = q_perm
        key[:, :, :-context_length, :] = k_perm
        value[:, :, :-context_length, :] = v_perm

        # Add prompt/unprompt clusters to the dynamic map.
        dyn_map = F.pad(dyn_map, (0, 2, 0, 2), value=0)
        dyn_map[:, :, -2, :-1] = True
        dyn_map[:, :, :-1, -2] = True
        dyn_map[:, :, -1, -1] = True

        qc_sz_s = F.pad(qc_sz_s, (0, 2), value=0)
        qc_sz_s[:, :, -2] = prompt_length
        qc_sz_s[:, :, -1] = unprompt_length
        kc_sz_s = F.pad(kc_sz_s, (0, 2), value=0)
        kc_sz_s[:, :, -2] = prompt_length
        kc_sz_s[:, :, -1] = unprompt_length

        q_sorted_indices = F.pad(q_sorted_indices, (0, context_length), value=0)
        q_sorted_indices[:, video_length:] = torch.arange(
            video_length,
            video_length + context_length,
            device=q_sorted_indices.device,
        )
        return query, key, value, dyn_map, qc_sz_s, kc_sz_s, q_sorted_indices

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: SparseVideoGen2AttentionMetadata,
    ) -> torch.Tensor:
        torch.backends.cuda.preferred_linalg_library(backend="magma")
        res = None
        # bshd -> bhsd
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        batch_size, num_heads, seq_len, dim = query.size()

        context_length, num_frame, frame_size = (
            attn_metadata.context_length,
            attn_metadata.num_frame,
            attn_metadata.frame_size,
        )
        prompt_length = attn_metadata.prompt_length
        if prompt_length is None:
            prompt_length = context_length

        assert (
            seq_len == context_length + num_frame * frame_size
        ), f"Query Shape: {seq_len} is not equivalent to {context_length} + {num_frame} * {frame_size}"

        if not self._use_sparse_attention(attn_metadata):
            if attn_metadata.zero_step_kmeans_init:
                video_length = attn_metadata.num_frame * attn_metadata.frame_size
                query_video = query[:, :, :video_length, :].contiguous()
                key_video = key[:, :, :video_length, :].contiguous()
                self.kmeans_clustering(query_video, key_video, attn_metadata)

            with sdpa_kernel(
                SDPBackend.CUDNN_ATTENTION
            ):  # not sure why we need to force cudnn here, but it's faster than flash attention
                output_hidden_states = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )

            res = output_hidden_states.reshape(
                batch_size, num_heads, seq_len, dim
            ).transpose(1, 2)
        else:
            if context_length > 0:
                video_length = num_frame * frame_size
                unprompt_length = max(context_length - prompt_length, 0)
                query_video = query[:, :, :video_length, :].contiguous()
                key_video = key[:, :, :video_length, :].contiguous()
                value_video = value[:, :, :video_length, :].contiguous()

                (
                    q_perm,
                    k_perm,
                    v_perm,
                    dyn_map,
                    qc_sz_s,
                    kc_sz_s,
                    q_sorted_indices,
                ) = self.semantic_aware_permutation(
                    query_video, key_video, value_video, attn_metadata
                )
                (
                    q_perm,
                    k_perm,
                    v_perm,
                    dyn_map,
                    qc_sz_s,
                    kc_sz_s,
                    q_sorted_indices,
                ) = self._hunyuan_dynamic_map_post_processing(
                    q_perm,
                    k_perm,
                    v_perm,
                    query,
                    key,
                    value,
                    dyn_map,
                    qc_sz_s,
                    kc_sz_s,
                    q_sorted_indices,
                    video_length,
                    context_length,
                    prompt_length,
                    unprompt_length,
                )
            else:
                (
                    q_perm,
                    k_perm,
                    v_perm,
                    dyn_map,
                    qc_sz_s,
                    kc_sz_s,
                    q_sorted_indices,
                ) = self.semantic_aware_permutation(query, key, value, attn_metadata)

            output_permuted = dynamic_block_sparse_fwd_flashinfer(
                q_perm, k_perm, v_perm, dyn_map, qc_sz_s, kc_sz_s, is_cpu=False
            )

            attn_output = apply_inverse_permutation_triton(
                output_permuted, q_sorted_indices, dim=2
            )

            # Density gather only matters when LB will use it next step.
            if (
                get_sequence_parallel_world_size() > 1
                and attn_metadata.load_balance != "off"
            ):
                densities = density_calculation(
                    dyn_map, qc_sz_s, kc_sz_s
                )  # [batch_size, num_heads]
                local_density = densities.mean(dim=0)  # [num_heads]
                self._launch_density_async(local_density, attn_metadata)

            res = attn_output.reshape(batch_size, num_heads, seq_len, dim).transpose(
                1, 2
            )

        torch.backends.cuda.preferred_linalg_library(
            backend="default"
        )  # reset to default
        return res.contiguous()

    def forward_distributed_asymm(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: SparseVideoGen2AttentionMetadata,
    ) -> torch.Tensor:
        """Asymm a2a path: variable heads-per-rank LPT + symm-mem pull/push.

        Replaces the symm path's
            preprocess → all_to_all(scatter=2,gather=1) × 3
            → forward → all_to_all(scatter=1,gather=2) → postprocess
        with
            head-LPT (variable hpr) → write to symm bufs → pull → forward
            → push → trim
        and skips the pre/post head permute (the kernel routes by h_idxs_r).

        Inputs are [B, S_local, H_total, D]; output matches.
        """
        world_size = get_sequence_parallel_world_size()
        if world_size == 1:
            return self.forward(q, k, v, attn_metadata)

        # 1. Plan: which heads does this rank own this step? Falls back to
        # contiguous when sparse window not active OR when no density yet.
        if self._use_sparse_attention(attn_metadata):
            self._get_head_reorder_perm(attn_metadata, world_size)
        layer_cache = attn_metadata.cache.get_layer(self.layer_idx)
        if layer_cache.h_idxs_r_dev is None:
            num_heads = self.num_heads
            heads_per_rank = _even_heads_per_rank(num_heads, world_size)
            cur_rank = get_sequence_parallel_rank()
            ofs = sum(heads_per_rank[:cur_rank])
            layer_cache.h_idxs_r_dev = torch.arange(
                ofs,
                ofs + heads_per_rank[cur_rank],
                device=q.device,
                dtype=torch.int32,
            )
            layer_cache.heads_per_rank = heads_per_rank

        h_idxs_r = layer_cache.h_idxs_r_dev
        heads_per_rank = layer_cache.heads_per_rank

        # 2. Allocate / fetch symm wrapper. Shape uses S_local from input.
        b, s_local, h_total, d = q.shape
        sp_group = get_sp_group().device_group
        symm_a2a = attn_metadata.cache.get_or_create_symm_a2a(
            sp_group, b, h_total, s_local, d, q.dtype, q.device
        )

        # 3. Write q/k/v into symm buffers in original head order
        # ([B, S_local, H, D] -> [B, H, S_local, D]).
        real_s = symm_a2a.s_local
        symm_a2a.q_symm[:, :, :real_s, :].copy_(q.transpose(1, 2).contiguous())
        symm_a2a.k_symm[:, :, :real_s, :].copy_(k.transpose(1, 2).contiguous())
        symm_a2a.v_symm[:, :, :real_s, :].copy_(v.transpose(1, 2).contiguous())

        # 4. Pull: each rank gathers its assigned heads × full seq.
        # Returns [B, H_local_r, WORLD * S_local_padded, D].
        q_loc = symm_a2a.pull_seq_to_heads(
            "q", h_idxs_r, pre_barrier=True, post_barrier=True
        )
        k_loc = symm_a2a.pull_seq_to_heads(
            "k", h_idxs_r, pre_barrier=False, post_barrier=False
        )
        v_loc = symm_a2a.pull_seq_to_heads(
            "v", h_idxs_r, pre_barrier=False, post_barrier=True
        )

        # Trim padded S back to real S_total = world_size * s_local.
        s_total = world_size * s_local
        if s_total != q_loc.size(2):
            q_loc = q_loc[:, :, :s_total, :].contiguous()
            k_loc = k_loc[:, :, :s_total, :].contiguous()
            v_loc = v_loc[:, :, :s_total, :].contiguous()

        # 5. Hand to attn_impl in [B, S, H, D] layout.
        q_loc = q_loc.transpose(1, 2).contiguous()
        k_loc = k_loc.transpose(1, 2).contiguous()
        v_loc = v_loc.transpose(1, 2).contiguous()
        out_loc = self.forward(q_loc, k_loc, v_loc, attn_metadata)
        # out_loc: [B, S_total, H_local_r, D]

        # 6. Push back: [B, H_local_r, S_total_padded, D] -> peers' recv_symm.
        out_h_first = out_loc.transpose(1, 2).contiguous()  # [B, H_local_r, S_total, D]
        s_padded_total = world_size * symm_a2a.s_local_padded
        if out_h_first.size(2) != s_padded_total:
            # Pad each per-peer S segment from s_local to s_local_padded so the
            # push kernel reads its expected stride. Per-peer slabs sit at
            # [p * s_local_padded : p * s_local_padded + s_local].
            pad_s = symm_a2a.s_local_padded - symm_a2a.s_local
            if pad_s > 0:
                h_local = out_h_first.size(1)
                src_padded = out_h_first.new_zeros(
                    b, h_local, world_size * symm_a2a.s_local_padded, d
                )
                for p in range(world_size):
                    src_padded[
                        :,
                        :,
                        p * symm_a2a.s_local_padded : p * symm_a2a.s_local_padded
                        + symm_a2a.s_local,
                        :,
                    ] = out_h_first[
                        :, :, p * symm_a2a.s_local : (p + 1) * symm_a2a.s_local, :
                    ]
                out_h_first = src_padded
        out_full = symm_a2a.push_heads_to_seq(
            out_h_first, h_idxs_r, pre_barrier=True, post_barrier=True
        )
        # out_full: [B, H_total, s_local_padded, D]; trim to s_local then bshd.
        if symm_a2a.s_local_padded != real_s:
            out_full = out_full[:, :, :real_s, :].contiguous()
        out = out_full.transpose(1, 2).contiguous()  # [B, S_local, H_total, D]
        return out
