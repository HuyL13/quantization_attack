import math
from types import SimpleNamespace

import pytest
import torch

from aq.ppl_eval import eval_ppl, load_corpus_ids


class FakeTokenizer:
    name_or_path = "fake-tokenizer"
    is_fast = True
    add_bos_token = False
    add_eos_token = False

    def __len__(self):
        return 32

    def __call__(self, text, return_tensors):
        assert return_tensors == "pt"
        ids = [ord(ch) % 32 for ch in text]
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))


class FakeModel:
    def __init__(self, losses):
        self.losses = list(losses)
        self.config = SimpleNamespace(use_cache=True)
        self.blocks = []
        self.training = True
        self.seen_batches = []

    def parameters(self):
        yield torch.zeros(1)

    def eval(self):
        self.training = False

    def __call__(self, batch, labels):
        self.seen_batches.append(batch.detach().cpu().tolist())
        assert labels is batch
        return SimpleNamespace(loss=torch.tensor(self.losses.pop(0)))


def test_load_corpus_ids_joins_wikitext2_raw_with_blank_lines(monkeypatch):
    def fake_load_dataset(name, config, split):
        assert name == "Salesforce/wikitext"
        assert config == "wikitext-2-raw-v1"
        assert split == "test"
        return {"text": ["alpha", "", "beta"]}

    monkeypatch.setattr("aq.ppl_eval.load_dataset", fake_load_dataset)

    enc = load_corpus_ids("wikitext2", FakeTokenizer(), seqlen=2048)

    expected_text = "alpha\n\n\n\nbeta"
    assert enc.tolist() == [[ord(ch) % 32 for ch in expected_text]]


def test_eval_ppl_uses_non_overlapping_blocks_and_gptq_loss_aggregation(monkeypatch):
    monkeypatch.setattr(
        "aq.ppl_eval.load_corpus_ids",
        lambda name, tokenizer, seqlen, cache_dir=None: torch.arange(1, 10).reshape(1, 9),
    )
    model = FakeModel([0.5, 1.0, 1.5])

    result = eval_ppl(model, FakeTokenizer(), ["wikitext2"], seqlen=3, verbose=False)

    expected = math.exp((0.5 * 3 + 1.0 * 3 + 1.5 * 3) / (3 * 3))
    assert result["wikitext2"] == pytest.approx(expected)
    assert model.seen_batches == [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]]
    assert model.config.use_cache is True
    assert model.training is False
