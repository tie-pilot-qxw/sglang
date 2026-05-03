"""Synthetic-data cost profiler for SVG2 head load balancing.

Produces a JSON cost model with `mask_fit.slope_ms_per_unit` and
`attention_fit.slope_ms_per_unit` slopes so the LPT planner can weight
heads by predicted ms instead of bare density. The slopes correct for
the fixed mask-gen floor that pure-density LPT under-counts.

Adapted from Sparse-VideoGen/scripts/wan/profile_wan_maskgen_aware_cost.py.
The original needed a captured `attn_core_*.pt` file as input; this
version synthesizes random Q/K/V at multiple seq_len points so it can
auto-run on first inference without any pre-captured data.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


_DEFAULT_SEQ_LENS = [4096, 8192, 16384, 32768, 65536]
_DEFAULT_HEADS = 4
_DEFAULT_NUM_Q_CENTROIDS = 300
_DEFAULT_NUM_K_CENTROIDS = 1000
_DEFAULT_TOP_P = 0.9
_DEFAULT_MIN_KC_RATIO = 0.1
_DEFAULT_KMEANS_ITER = 2


def _cuda_timed(fn) -> tuple[Any, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, float(start.elapsed_time(end))


def _fit_affine(xs: list[float], ys: list[float]) -> dict:
    """Least-squares fit y = intercept + slope * x. Returns slope-only dict
    matching bench's shape."""
    if len(xs) < 2:
        return {
            "intercept_ms": 0.0,
            "slope_ms_per_unit": 0.0,
            "r2": 1.0,
            "num_samples": len(xs),
        }
    x = torch.tensor([[1.0, v] for v in xs], dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64).reshape(-1, 1)
    coeff = torch.linalg.lstsq(x, y).solution.reshape(-1)
    pred = (x @ coeff.reshape(-1, 1)).reshape(-1)
    y_mean = y.mean()
    ss_res = torch.sum((y.reshape(-1) - pred) ** 2)
    ss_tot = torch.sum((y.reshape(-1) - y_mean) ** 2)
    r2 = 1.0 if float(ss_tot) == 0.0 else 1.0 - float(ss_res / ss_tot)
    return {
        "intercept_ms": float(coeff[0]),
        "slope_ms_per_unit": float(coeff[1]),
        "r2": r2,
        "num_samples": len(xs),
    }


def profile_cost_model(
    num_heads: int = _DEFAULT_HEADS,
    head_dim: int = 128,
    seq_lens: list[int] | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
    num_q_centroids: int = _DEFAULT_NUM_Q_CENTROIDS,
    num_k_centroids: int = _DEFAULT_NUM_K_CENTROIDS,
    top_p: float = _DEFAULT_TOP_P,
    min_kc_ratio: float = _DEFAULT_MIN_KC_RATIO,
    kmeans_iter: int = _DEFAULT_KMEANS_ITER,
    warmup: int = 1,
    iters: int = 3,
) -> dict:
    """Sweep seq_len, time mask + attention kernels on synthetic Q/K/V,
    fit affine slopes. Returns a dict suitable to be written as JSON.
    """
    try:
        from svg.kernels.triton.permute import permute_tensor_by_labels_triton
        from svg.kmeans_utils import (
            batch_kmeans_Euclid,
            density_calculation,
            dynamic_block_sparse_fwd_flashinfer,
            identify_dynamic_map,
        )
    except ImportError as e:
        raise ImportError(
            "svg2 cost profiler requires the `svg` package. Install per "
            "https://github.com/svg-project/Sparse-VideoGen."
        ) from e

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError(f"profile_cost_model requires a CUDA device, got {device}")
    if seq_lens is None:
        seq_lens = list(_DEFAULT_SEQ_LENS)

    rows: list[dict] = []
    cfg = 1
    for seq_len in seq_lens:
        # Round seq_len up to a multiple of num_q_centroids so kmeans
        # converges; minor fudge that doesn't change the slope estimate.
        s = ((seq_len + num_q_centroids - 1) // num_q_centroids) * num_q_centroids
        torch.manual_seed(0)
        query = torch.randn(cfg, num_heads, s, head_dim, dtype=dtype, device=device)
        key = torch.randn(cfg, num_heads, s, head_dim, dtype=dtype, device=device)
        value = torch.randn(cfg, num_heads, s, head_dim, dtype=dtype, device=device)

        def _mask_step(q=query, k=key, val=value):
            qlabels, qcentroids, qcluster_sizes, _ = batch_kmeans_Euclid(
                q.view(cfg * num_heads, s, head_dim),
                n_clusters=num_q_centroids,
                max_iters=kmeans_iter,
            )
            klabels, kcentroids, kcluster_sizes, _ = batch_kmeans_Euclid(
                k.view(cfg * num_heads, s, head_dim),
                n_clusters=num_k_centroids,
                max_iters=kmeans_iter,
            )
            qcs = qcluster_sizes.view(cfg, num_heads, num_q_centroids)
            kcs = kcluster_sizes.view(cfg, num_heads, num_k_centroids)
            dyn_map = identify_dynamic_map(
                qcentroids.view(cfg, num_heads, num_q_centroids, head_dim),
                kcentroids.view(cfg, num_heads, num_k_centroids, head_dim),
                qcs,
                kcs,
                top_p,
                min_kc_ratio,
            )
            q_perm, _ = permute_tensor_by_labels_triton(q, qlabels, dim=2)
            k_perm, k_si = permute_tensor_by_labels_triton(k, klabels, dim=2)
            v_perm, _ = permute_tensor_by_labels_triton(
                val, klabels, dim=2, sorted_indices=k_si
            )
            return q_perm, k_perm, v_perm, dyn_map, qcs, kcs

        for _ in range(warmup):
            _mask_step()
            torch.cuda.synchronize(device)

        mask_times: list[float] = []
        outputs = None
        for _ in range(iters):
            outputs, ms = _cuda_timed(_mask_step)
            mask_times.append(ms)

        q_perm, k_perm, v_perm, dyn_map, qcs, kcs = outputs
        density = density_calculation(dyn_map, qcs, kcs).detach().float()
        density_mean = float(density.mean().item())

        def _attn_step(qp=q_perm, kp=k_perm, vp=v_perm, dm=dyn_map, qs=qcs, ks=kcs):
            return dynamic_block_sparse_fwd_flashinfer(
                qp, kp, vp, dm, qs, ks, is_cpu=False
            )

        for _ in range(warmup):
            _attn_step()
            torch.cuda.synchronize(device)

        attn_times: list[float] = []
        for _ in range(iters):
            _, ms = _cuda_timed(_attn_step)
            attn_times.append(ms)

        rows.append(
            {
                "heads": num_heads,
                "seq_len": s,
                "density_mean": density_mean,
                "mask_median_ms": statistics.median(mask_times),
                "attention_median_ms": statistics.median(attn_times),
                "mask_feature": num_heads * s,
                "attention_feature": num_heads * density_mean * s * s,
            }
        )
        del query, key, value, q_perm, k_perm, v_perm, dyn_map, qcs, kcs
        torch.cuda.empty_cache()

    mask_fit = _fit_affine(
        [r["mask_feature"] for r in rows],
        [r["mask_median_ms"] for r in rows],
    )
    attention_fit = _fit_affine(
        [r["attention_feature"] for r in rows],
        [r["attention_median_ms"] for r in rows],
    )
    return {
        "version": 1,
        "source": "auto-profile (synthetic Q/K/V)",
        "device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
        ),
        "dtype": str(dtype).split(".")[-1],
        "num_heads_profiled": num_heads,
        "head_dim": head_dim,
        "seq_lens_profiled": [r["seq_len"] for r in rows],
        "samples": rows,
        "mask_fit": mask_fit,
        "attention_fit": attention_fit,
    }


def load_or_profile_cost_model(
    path: str,
    dit_config: Any,
    seq_lens: list[int] | None = None,
) -> dict:
    """If path exists and parses as JSON, return its content. Otherwise run
    profile_cost_model with shapes derived from dit_config, write the
    result to path (creating parent dirs), and return it.

    Only rank 0 profiles + writes; other ranks block on a barrier and read
    the file. Avoids racing N profilers on the same machine.
    """
    p = Path(path)
    if p.exists():
        try:
            with p.open("r") as f:
                cm = json.load(f)
            logger.info("svg2 cost model: loaded cached %s", p)
            return cm
        except Exception as e:
            logger.warning(
                "svg2 cost model: failed to read cached %s (%s); reprofiling",
                p,
                e,
            )

    # Coordinate across ranks: only rank 0 actually profiles.
    rank = 0
    world_size = 1
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
    except Exception:
        pass

    if rank == 0:
        head_dim = int(getattr(dit_config, "attention_head_dim", 128))
        # Profile a small head count for speed; the slope fit is per-head.
        num_heads = min(
            int(getattr(dit_config, "num_attention_heads", _DEFAULT_HEADS)), 4
        )
        logger.info(
            "svg2 cost model: profiling at %s (heads=%d head_dim=%d). "
            "Takes ~30s; cached for subsequent runs.",
            p,
            num_heads,
            head_dim,
        )
        t0 = time.time()
        cm = profile_cost_model(
            num_heads=num_heads,
            head_dim=head_dim,
            seq_lens=seq_lens,
        )
        logger.info(
            "svg2 cost model: done in %.1fs (mask_slope=%.3e attn_slope=%.3e)",
            time.time() - t0,
            cm["mask_fit"]["slope_ms_per_unit"],
            cm["attention_fit"]["slope_ms_per_unit"],
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(cm, f, indent=2)
        os.replace(tmp, p)

    if world_size > 1:
        try:
            import torch.distributed as dist

            dist.barrier()
        except Exception:
            pass

    if rank != 0:
        with p.open("r") as f:
            cm = json.load(f)
    return cm
