"""Integration-style check (still no GPU / no real 7B model): builds a tiny
fake HF-shaped model (model.model.layers[i].self_attn.{q,k,v,o}_proj +
.mlp.{gate,up,down}_proj) to verify aq.common's layer/block discovery and
aq.run_method's strategy dispatch wire together correctly end to end, since
the real if_awq_tier0/transformers dependencies for a live 7B run aren't
available in this environment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from aq.activation_cache import (
    capture_block_input_activations,
    capture_layer_input_activations,
    compute_layer_activation_sensitivity,
)
from aq.common import get_block_layer_groups, get_block_modules, get_transformer_linear_layers
from aq.optimizer_core import compute_fp_reference_logits
from aq.calibration_strategies import run_isolated, run_quantized_prefix
from aq.greedy_rounding import GreedyRoundingConfig, run_greedy_adversarial_rounding
from aq.local_reconstruction import run_blockwise_local_reconstruction, run_layerwise_local_reconstruction
from aq.quantizer import AdversarialQuantConfig
from aq.run_method import _apply_rtn4_baseline, _run_strategy


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
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return SimpleNamespace(logits=logits, loss=loss)


def _fake_batches():
    torch.manual_seed(7)
    return [{"input_ids": torch.randint(0, 32, (2, 5)), "attention_mask": torch.ones(2, 5, dtype=torch.long)}]


def test_get_transformer_linear_layers_finds_all_projections():
    model = FakeCausalLM(n_layers=2)
    layers = get_transformer_linear_layers(model)
    assert len(layers) == 2 * 7  # 4 attn + 3 mlp per block
    assert "model.layers.0.self_attn.q_proj" in layers
    assert "model.layers.1.mlp.down_proj" in layers


def test_get_block_layer_groups_groups_by_block():
    model = FakeCausalLM(n_layers=2)
    groups = get_block_layer_groups(model)
    assert len(groups) == 2
    assert len(groups[0]) == 7
    assert all(name.startswith("model.layers.0.") for name in groups[0])


def test_apply_rtn4_baseline_end_to_end():
    model = FakeCausalLM(n_layers=2)
    layers = get_transformer_linear_layers(model)
    originals = {name: mod.weight.detach().clone() for name, mod in layers.items()}
    _apply_rtn4_baseline(layers)
    for name, mod in layers.items():
        assert not torch.equal(mod.weight.detach(), originals[name])
    # forward pass must still run cleanly after every weight was overwritten
    batches = _fake_batches()
    out = model(input_ids=batches[0]["input_ids"])
    assert torch.isfinite(out.logits).all()


def test_run_strategy_dispatch_isolated_methods(monkeypatch):
    model = FakeCausalLM(n_layers=2)
    layers = get_transformer_linear_layers(model)
    order = list(layers.keys())
    block_groups = get_block_layer_groups(model)
    batches = _fake_batches()
    fp_ref = compute_fp_reference_logits(model, batches, device="cpu")
    cfg = AdversarialQuantConfig(bits=4, group_size=128, steps=2, lr=1e-2)

    for method_id in ["01d_global_kl", "02_adv_round_scale", "03_adv_codebook", "04_sensitivity_aware"]:
        # fresh model per method to avoid cross-method state leakage in this test
        m = FakeCausalLM(n_layers=2)
        l = get_transformer_linear_layers(m)
        o = list(l.keys())
        bg = get_block_layer_groups(m)
        b = _fake_batches()
        ref = compute_fp_reference_logits(m, b, device="cpu")
        cfg2 = AdversarialQuantConfig(
            bits=4,
            group_size=128,
            steps=2,
            lr=1e-2,
            optimize_scale=(method_id == "02_adv_round_scale"),
            use_codebook=(method_id == "03_adv_codebook"),
            n_codebook_centroids=8,
            use_sensitivity=(method_id == "04_sensitivity_aware"),
        )
        results = _run_strategy(method_id, m, l, o, bg, b, ref, cfg2, "cpu")
        assert set(results.keys()) == set(o)
        out = m(input_ids=b[0]["input_ids"])
        assert torch.isfinite(out.logits).all()


def test_run_strategy_dispatch_sequential_and_block_and_two_pass():
    for method_id in ["05_quantized_prefix", "06_block_wise", "08_two_pass_backward"]:
        m = FakeCausalLM(n_layers=2)
        l = get_transformer_linear_layers(m)
        o = list(l.keys())
        bg = get_block_layer_groups(m)
        b = _fake_batches()
        ref = compute_fp_reference_logits(m, b, device="cpu")
        cfg = AdversarialQuantConfig(bits=4, group_size=128, steps=2, lr=1e-2)
        results = _run_strategy(method_id, m, l, o, bg, b, ref, cfg, "cpu")
        assert set(results.keys()) == set(o)
        out = m(input_ids=b[0]["input_ids"])
        assert torch.isfinite(out.logits).all()


def test_run_strategy_dispatch_periodic_refresh():
    m = FakeCausalLM(n_layers=4)
    l = get_transformer_linear_layers(m)
    o = list(l.keys())
    bg = get_block_layer_groups(m)
    b = _fake_batches()
    ref = compute_fp_reference_logits(m, b, device="cpu")
    cfg = AdversarialQuantConfig(bits=4, group_size=128, steps=2, lr=1e-2)
    cfg.refresh_every_k_blocks = 2
    results = _run_strategy("07_periodic_refresh", m, l, o, bg, b, ref, cfg, "cpu")
    assert set(results.keys()) == set(o)


def test_method_1a_greedy_rounding_against_real_block_shaped_model():
    m = FakeCausalLM(n_layers=2)
    l = get_transformer_linear_layers(m)
    o = list(l.keys())
    b = _fake_batches()
    originals = {name: mod.weight.detach().clone() for name, mod in l.items()}

    sensitivity_by_layer = compute_layer_activation_sensitivity(m, l, b, device="cpu")
    cfg = GreedyRoundingConfig(bits=4, group_size=128, flip_fraction=0.2)
    results = run_greedy_adversarial_rounding(m, l, o, sensitivity_by_layer, cfg, device="cpu")

    assert set(results.keys()) == set(o)
    for name, mod in l.items():
        assert not torch.equal(mod.weight.detach(), originals[name])
    out = m(input_ids=b[0]["input_ids"])
    assert torch.isfinite(out.logits).all()


def test_method_1b_layerwise_local_against_real_block_shaped_model():
    m = FakeCausalLM(n_layers=2)
    l = get_transformer_linear_layers(m)
    o = list(l.keys())
    b = _fake_batches()

    cached = capture_layer_input_activations(m, l, b, device="cpu")
    cfg = AdversarialQuantConfig(bits=4, group_size=128, steps=2, lr=1e-2, batches_per_step=1)
    results = run_layerwise_local_reconstruction(l, o, cached, cfg, device="cpu")

    assert set(results.keys()) == set(o)
    out = m(input_ids=b[0]["input_ids"])
    assert torch.isfinite(out.logits).all()


def test_method_1c_blockwise_local_against_real_block_shaped_model():
    m = FakeCausalLM(n_layers=2)
    l = get_transformer_linear_layers(m)
    block_groups = get_block_layer_groups(m)
    block_modules = get_block_modules(m)
    b = _fake_batches()

    block_module_map = {str(i): mod for i, mod in enumerate(block_modules)}
    cached_by_idx = capture_block_input_activations(m, block_module_map, b, device="cpu")
    cached_block_inputs = [cached_by_idx[str(i)] for i in range(len(block_modules))]

    cfg = AdversarialQuantConfig(bits=4, group_size=128, steps=2, lr=1e-2, batches_per_step=1)
    results = run_blockwise_local_reconstruction(block_groups, l, block_modules, cached_block_inputs, cfg, device="cpu")

    assert set(results.keys()) == set(l.keys())
    out = m(input_ids=b[0]["input_ids"])
    assert torch.isfinite(out.logits).all()
