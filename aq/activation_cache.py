"""One-pass activation caching for the light tiers of method 1 (1A greedy,
1B layer-wise local reconstruction, 1C block-wise local reconstruction).

All three avoid ever running the full 32-layer model more than once per
calibration batch: activations are captured ONCE via forward hooks on the
pristine FP model, cached on CPU, and reused for every optimization step
afterward. This is what makes them dramatically cheaper than method 1D
(global KL), which reruns the whole model on every single step - measured
live: 1D cost 2.79s per optimization step per layer (a full-model forward
+ checkpointed backward), whereas a cached-activation local reconstruction
step is a single small matmul, no full-model pass at all.
"""
from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def capture_layer_input_activations(
    model, layers: dict, calibration_batches: list[dict], device: str
) -> dict[str, list[torch.Tensor]]:
    """Registers a forward pre-hook on every target nn.Linear in `layers`,
    runs each calibration batch through the model ONCE, and returns each
    layer's captured input tensor per batch (CPU, to keep this cheap cache
    from competing with the model for GPU memory - same rationale as
    aq.optimizer_core.compute_fp_reference_logits).
    """
    captured: dict[str, list[torch.Tensor]] = {name: [] for name in layers}
    handles = []

    def make_hook(name):
        def hook(_module, inputs):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            captured[name].append(x.detach().to("cpu"))

        return hook

    for name, module in layers.items():
        handles.append(module.register_forward_pre_hook(make_hook(name)))

    model.eval()
    try:
        for batch in calibration_batches:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return captured


@torch.no_grad()
def capture_block_input_activations(
    model, block_modules: dict[str, nn.Module], calibration_batches: list[dict], device: str
) -> dict[str, list[torch.Tensor]]:
    """Same idea as capture_layer_input_activations, but hooks whole
    transformer blocks (e.g. model.model.layers[i]) instead of individual
    Linear layers - used by method 1C's block-level reconstruction.
    """
    captured: dict[str, list[torch.Tensor]] = {name: [] for name in block_modules}
    handles = []

    def make_hook(name):
        def hook(_module, inputs):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            captured[name].append(x.detach().to("cpu"))

        return hook

    for name, module in block_modules.items():
        handles.append(module.register_forward_pre_hook(make_hook(name)))

    model.eval()
    try:
        for batch in calibration_batches:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return captured


def activation_sensitivity(cached_inputs: list[torch.Tensor]) -> torch.Tensor:
    """Per-input-feature mean squared activation (AWQ-style salience),
    computed from a layer's cached inputs. Shape: (in_features,). Used by
    method 1A as the "activation error increase" term in its greedy score -
    the same idea as if_awq_tier0's AWQQuantizerXL.get_activation_stats,
    reimplemented standalone here since that class also carries scale-search
    machinery method 1A has no use for.
    """
    total = None
    count = 0
    for x in cached_inputs:
        x_flat = x.reshape(-1, x.shape[-1]).float()
        sq_sum = x_flat.pow(2).sum(dim=0)
        total = sq_sum if total is None else total + sq_sum
        count += x_flat.shape[0]
    if total is None or count == 0:
        raise ValueError("cached_inputs is empty - nothing to compute sensitivity from")
    return total / count
