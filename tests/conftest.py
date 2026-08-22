import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("IF_AWQ_TIER0_ROOT", str(REPO_ROOT.parent / "if_awq_tier0"))


class TinyLM(nn.Module):
    """Minimal stand-in for a HF CausalLM: same call signature
    (input_ids, attention_mask, optional labels) and a `.logits` /
    `.loss` output, small enough to run instantly on CPU. Used to exercise
    optimize_layer / calibration_strategies / sensitivity without needing a
    real 7B checkpoint or a GPU.
    """

    def __init__(self, vocab: int = 32, hidden: int = 16, n_mid_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.mid_layers = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(n_mid_layers)]
        )
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=None):
        x = self.embed(input_ids)
        for layer in self.mid_layers:
            x = torch.tanh(layer(x))
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return SimpleNamespace(logits=logits, loss=loss)


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return TinyLM()


@pytest.fixture
def tiny_calibration_batches():
    torch.manual_seed(1)
    return [
        {"input_ids": torch.randint(0, 32, (2, 6)), "attention_mask": torch.ones(2, 6, dtype=torch.long)}
        for _ in range(2)
    ]
