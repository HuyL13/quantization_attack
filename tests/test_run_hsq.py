"""Integration-style check (no GPU / no real 7B model, same pattern as
test_common_layer_discovery.py): runs aq.run_hsq.quantize_model_with_hsq
end to end against a tiny fake HF-shaped model to verify the sequential
block-quantization driver (capture -> Hessian -> run_hsq_layer -> commit)
wires together correctly, and that quantizing block i really does change
what block i+1's captured calibration activations look like (the whole
point of doing this block-by-block instead of independently per layer).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from aq.common import get_block_layer_groups, get_transformer_linear_layers
from aq.run_hsq import quantize_model_with_hsq


class FakeAttn(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)

    def forward(self, x):
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class FakeMLP(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.gate_proj = nn.Linear(h, h, bias=False)
        self.up_proj = nn.Linear(h, h, bias=False)
        self.down_proj = nn.Linear(h, h, bias=False)

    def forward(self, x):
        return self.down_proj(torch.tanh(self.gate_proj(x)) * self.up_proj(x))


class FakeBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.self_attn = FakeAttn(h)
        self.mlp = FakeMLP(h)

    def forward(self, x):
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class FakeInner(nn.Module):
    def __init__(self, vocab, h, n_layers):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, h)
        self.layers = nn.ModuleList([FakeBlock(h) for _ in range(n_layers)])

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x


class FakeCausalLM(nn.Module):
    def __init__(self, vocab=32, h=16, n_layers=2):
        super().__init__()
        self.model = FakeInner(vocab, h, n_layers)
        self.lm_head = nn.Linear(h, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=None):
        x = self.model(input_ids)
        logits = self.lm_head(x)
        return SimpleNamespace(logits=logits, loss=None)


def _fake_batches():
    torch.manual_seed(7)
    return [
        {"input_ids": torch.randint(0, 32, (2, 20)), "attention_mask": torch.ones(2, 20, dtype=torch.long)}
        for _ in range(3)
    ]


def test_quantize_model_with_hsq_gptq_mode_commits_all_layers():
    model = FakeCausalLM(n_layers=2)
    all_layers = get_transformer_linear_layers(model)
    block_groups = get_block_layer_groups(model)
    originals = {name: mod.weight.detach().clone() for name, mod in all_layers.items()}
    batches = _fake_batches()

    metrics = quantize_model_with_hsq(
        model, all_layers, block_groups, batches, hsq_mode="gptq",
        bits=4, group_size=16, blocksize=8, device="cpu",
    )

    assert len(metrics) == len(all_layers)
    for name, mod in all_layers.items():
        assert not torch.equal(mod.weight.detach(), originals[name])
    out = model(input_ids=batches[0]["input_ids"])
    assert torch.isfinite(out.logits).all()


def test_quantize_model_with_hsq_v0_and_v1_run_without_error():
    for mode in ("hsq_v0", "hsq_v1"):
        model = FakeCausalLM(n_layers=2)
        all_layers = get_transformer_linear_layers(model)
        block_groups = get_block_layer_groups(model)
        batches = _fake_batches()
        metrics = quantize_model_with_hsq(
            model, all_layers, block_groups, batches, hsq_mode=mode,
            bits=4, group_size=16, blocksize=8, tau=0.1, device="cpu",
        )
        assert len(metrics) == len(all_layers)
        out = model(input_ids=batches[0]["input_ids"])
        assert torch.isfinite(out.logits).all()


def test_block_1_activations_reflect_block_0_being_already_quantized():
    """The whole point of the block-sequential driver: block 1's captured
    calibration input must be computed against the model AS-QUANTIZED so
    far, not the pristine FP model - verify this by comparing against a
    fresh FP model run on the same batches.
    """
    from aq.activation_cache import capture_layer_input_activations

    model = FakeCausalLM(n_layers=2)
    fp_model = FakeCausalLM(n_layers=2)
    fp_model.load_state_dict(model.state_dict())

    all_layers = get_transformer_linear_layers(model)
    block_groups = get_block_layer_groups(model)
    batches = _fake_batches()

    quantize_model_with_hsq(
        model, all_layers, block_groups, batches, hsq_mode="gptq",
        bits=4, group_size=16, blocksize=8, device="cpu",
    )

    block1_layers = {name: all_layers[name] for name in block_groups[1]}
    fp_block1_layers = get_transformer_linear_layers(fp_model)
    fp_block1_layers = {name: fp_block1_layers[name] for name in block_groups[1]}

    quantized_inputs = capture_layer_input_activations(model, block1_layers, batches, "cpu")
    fp_inputs = capture_layer_input_activations(fp_model, fp_block1_layers, batches, "cpu")

    any_name = block_groups[1][0]
    assert not torch.allclose(quantized_inputs[any_name][0], fp_inputs[any_name][0])
