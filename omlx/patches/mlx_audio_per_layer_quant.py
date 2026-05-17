# SPDX-License-Identifier: Apache-2.0
"""Make mlx_audio.utils.apply_quantization honour per-layer quantization
overrides ahead of any model-specific predicate.

Upstream mlx-audio's ``apply_quantization`` builds a class_predicate that
consults the model's ``model_quant_predicate`` BEFORE checking the per-layer
override block. For models that ship as mixed precision — e.g. Alkd's
``Qwen3-ASR-1.7B-audio8-text4-mlx``, where ``audio_tower.*`` is quantized to
8 bits and ``text_model.*`` to 4 bits via standard mlx-lm per-layer override
schema — Qwen3ASRModel.model_quant_predicate hard-codes
``not p.startswith("audio_tower")``, returning False for every audio layer.
The per-layer override fallback that follows is unreachable, so audio_tower
weights are loaded as fp16, dimensions don't match the packed safetensors,
and the first encoder matmul aborts with ``(B, T, packed_dim) × (in, out)``
shape mismatch.

The fix is one line of logic: check the per-layer override dict FIRST. If
the layer has an explicit override, use it; only fall back to the model-
specific predicate for layers without one. This preserves the original
behaviour for uniformly-quantized checkpoints (audio_tower.* not in the
override dict → predicate returns False → skip, as before).

Patch is idempotent. Safe to call multiple times.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_APPLIED = False


def apply_mlx_audio_per_layer_quant_patch() -> bool:
    """Replace ``mlx_audio.utils.apply_quantization`` with a per-layer-first
    variant. Returns True on first successful apply, False if already applied
    or mlx_audio is unavailable.
    """
    global _APPLIED
    if _APPLIED:
        return False
    try:
        from mlx_audio import utils as _mlxa_utils
        import mlx.nn as nn
    except ImportError:
        logger.debug("mlx_audio.utils not importable; skipping per-layer quant patch")
        return False

    def patched_apply_quantization(
        model,
        config: dict,
        weights: dict,
        model_quant_predicate=None,
    ) -> None:
        quantization = config.get("quantization", None)
        if quantization is None:
            quantization = config.get("quantization_config", None)
        if quantization is None:
            return
        group_size = quantization.get("group_size", 64)

        def get_class_predicate(p, m):
            if not hasattr(m, "to_quantized"):
                return False
            if hasattr(m, "weight") and m.weight.shape[-1] % group_size != 0:
                return False
            # Per-layer override has priority over model_quant_predicate so that
            # mixed-precision checkpoints (e.g. Qwen3-ASR audio8/text4) load with
            # the right bits per layer instead of being silently skipped by the
            # model's coarse audio_tower → False rule.
            if p in quantization:
                override = quantization[p]
                if isinstance(override, dict):
                    return override
            if model_quant_predicate is not None:
                pred_result = model_quant_predicate(p, m)
                if isinstance(pred_result, dict):
                    return pred_result
                if not pred_result:
                    return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            group_size=group_size,
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=get_class_predicate,
        )

    _mlxa_utils.apply_quantization = patched_apply_quantization
    _APPLIED = True
    logger.info(
        "mlx_audio.utils.apply_quantization patched: per-layer overrides "
        "now take priority over model_quant_predicate "
        "(fixes mixed-precision STT checkpoints like Qwen3-ASR audio8/text4)"
    )
    return True
