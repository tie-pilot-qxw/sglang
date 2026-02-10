"""
Sparse Video Gen 2 (SAP) attention backend.

This is a baseline integration that wires the backend into the
attention framework.

Adapted from https://github.com/svg-project/Sparse-VideoGen/blob/main/svg/models/wan/attention.py
"""

import os
from dataclasses import dataclass, field
from typing import Any, List

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

    head_perm: torch.Tensor | None = None
    head_perm_inv: torch.Tensor | None = None


@dataclass
class Svg2Cache:
    planning_stream: torch.cuda.Stream
    layers: dict[int, Svg2LayerCache] = field(default_factory=dict)

    def get_layer(self, layer_idx: int) -> Svg2LayerCache:
        layer_cache = self.layers.get(layer_idx)
        if layer_cache is None:
            layer_cache = Svg2LayerCache()
            self.layers[layer_idx] = layer_cache
        return layer_cache


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
    prompt_length: int | None = None
    max_seqlen_q: int | None = None
    max_seqlen_k: int | None = None


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
        self, density_cpu: torch.Tensor | None, num_heads: int, world_size: int
    ) -> List[int]:
        if density_cpu is None:
            return list(range(num_heads))

        # weight: higher density => more work
        w = [float(density_cpu[i].item()) for i in range(num_heads)]

        # LPT order
        sorted_heads = sorted(range(num_heads), key=lambda i: (-w[i], i))

        heads_per_rank = (
            num_heads // world_size
        )  # guaranteed divisible per your assumption

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

        # safety: should always be full
        if sum(sizes) != num_heads:
            return list(range(num_heads))

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

        return [h for r in range(world_size) for h in bins[r]]

    def _get_head_reorder_perm(
        self, attn_metadata: SparseVideoGen2AttentionMetadata, world_size: int
    ) -> torch.Tensor:
        layer_cache = attn_metadata.cache.get_layer(self.layer_idx)
        num_heads = self.num_heads

        if layer_cache.density_async_complete is None:
            head_perm = torch.arange(
                num_heads, device=torch.device("cuda"), dtype=torch.long
            )
            layer_cache.head_perm = head_perm
            layer_cache.head_perm_inv = head_perm
            return head_perm

        torch.cuda.current_stream().wait_event(layer_cache.density_async_complete)
        assert layer_cache.density is not None

        perm_list = self._compute_head_reorder_perm(
            layer_cache.density, num_heads, world_size
        )
        head_perm = torch.tensor(
            perm_list, device=torch.device("cuda"), dtype=torch.long
        )
        head_perm_inv = torch.empty_like(head_perm)
        head_perm_inv[head_perm] = torch.arange(num_heads, device=head_perm.device)
        layer_cache.head_perm = head_perm
        layer_cache.head_perm_inv = head_perm_inv

        if (
            layer_cache.q_centroids_global is not None
            and layer_cache.k_centroids_global is not None
        ):
            num_local_heads = num_heads // world_size
            cur_rank = get_sequence_parallel_rank()
            head_ids = head_perm[
                cur_rank * num_local_heads : (cur_rank + 1) * num_local_heads
            ]
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
        num_heads = num_local_heads * world_size

        if layer_cache.density_async_complete is None:
            layer_cache.density_async_complete = torch.cuda.Event()
        if layer_cache.density is None:
            layer_cache.density = torch.empty(
                [num_heads], dtype=local_density.dtype, device="cpu"
            ).pin_memory()

        planning_stream = attn_metadata.cache.planning_stream
        planning_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(planning_stream):
            if layer_cache.head_perm_inv is not None:
                inv_old_perm = layer_cache.head_perm_inv
            else:
                inv_old_perm = torch.arange(
                    num_heads, device=local_density.device, dtype=torch.long
                )

            _, num_q_centroids, q_dim = layer_cache.q_centroids.size()
            _, num_k_centroids, k_dim = layer_cache.k_centroids.size()

            # reshape to [B, H, C, D]
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

            # gather density in old-perm order, then reorder to global head ids
            density_gather = torch.empty(
                (world_size,) + local_density.shape,
                device=local_density.device,
                dtype=local_density.dtype,
            )
            dist.all_gather_into_tensor(density_gather, local_density, group=sp_group)
            density_all = density_gather.reshape(num_heads)
            layer_cache.density_gpu = density_all.index_select(0, inv_old_perm)

            # gather centroids in old-perm order, then reorder to global head ids
            q_gather = torch.empty(
                (world_size,) + q_centroids.shape,
                device=q_centroids.device,
                dtype=q_centroids.dtype,
            )
            dist.all_gather_into_tensor(q_gather, q_centroids, group=sp_group)
            q_all = q_gather.permute(1, 0, 2, 3, 4).reshape(
                -1, num_heads, num_q_centroids, q_dim
            )
            layer_cache.q_centroids_global = q_all.index_select(
                1, inv_old_perm
            ).reshape(-1, num_q_centroids, q_dim)

            k_gather = torch.empty(
                (world_size,) + k_centroids.shape,
                device=k_centroids.device,
                dtype=k_centroids.dtype,
            )
            dist.all_gather_into_tensor(k_gather, k_centroids, group=sp_group)
            k_all = k_gather.permute(1, 0, 2, 3, 4).reshape(
                -1, num_heads, num_k_centroids, k_dim
            )
            layer_cache.k_centroids_global = k_all.index_select(
                1, inv_old_perm
            ).reshape(-1, num_k_centroids, k_dim)

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

            enable_loadbalance = os.getenv(
                "SGLANG_SVG2_LOADBALANCE", "1"
            ).lower() not in ("0", "false")
            # if multiple cards, collect density
            if get_sequence_parallel_world_size() > 1 and enable_loadbalance:
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
