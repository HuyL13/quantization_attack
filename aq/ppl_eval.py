"""WikiText-2 perplexity evaluation using the GPTQ/AWQ block protocol."""
from __future__ import annotations

import gc
import hashlib
import pickle
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm


def _transformers_version() -> str:
    try:
        import transformers

        return transformers.__version__
    except Exception:
        return "unknown"


def load_corpus_ids(name: str, tokenizer, seqlen: int, cache_dir: Path | None = None) -> torch.Tensor:
    """Return corpus input_ids shaped [1, N].

    WikiText-2 intentionally keeps empty lines, joins with ``\n\n``, and
    tokenizes once over the full test corpus to match the GPTQ-style PPL
    protocol used by the local eval_ppl.py.
    """
    if name != "wikitext2":
        raise ValueError(f"Unknown dataset: {name}")

    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = "|".join(
            [
                name,
                str(getattr(tokenizer, "name_or_path", "tok")),
                type(tokenizer).__name__,
                f"fast={bool(getattr(tokenizer, 'is_fast', False))}",
                f"bos={getattr(tokenizer, 'add_bos_token', None)}",
                f"eos={getattr(tokenizer, 'add_eos_token', None)}",
                f"vocab={len(tokenizer)}",
                "seqlen=na",
                f"tfm={_transformers_version()}",
            ]
        )
        digest = hashlib.sha1(fingerprint.encode()).hexdigest()[:12]
        cache_file = cache_dir / f"{name}_{digest}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as fh:
                return pickle.load(fh)

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids

    if cache_file is not None:
        with open(cache_file, "wb") as fh:
            pickle.dump(enc, fh)
    return enc


@torch.no_grad()
def eval_ppl(
    model,
    tokenizer,
    testcases: list[str],
    seqlen: int = 2048,
    cache_dir: Path | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    results: dict[str, float] = {}

    prev_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False

    for name in testcases:
        enc = load_corpus_ids(name, tokenizer, seqlen, cache_dir)
        nsamples = enc.numel() // seqlen
        if nsamples == 0:
            if verbose:
                print(f"{name}: corpus shorter than seqlen, skipping")
            continue

        nlls = []
        for i in tqdm(range(nsamples), disable=not verbose, desc=name):
            batch = enc[:, i * seqlen : (i + 1) * seqlen].to(device)
            out = model(batch, labels=batch)
            nlls.append(out.loss.float() * seqlen)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()
        results[name] = ppl
        if verbose:
            print(f"{name}: PPL = {ppl:.4f}  ({nsamples} blocks x {seqlen} tokens)")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if prev_use_cache is not None:
        model.config.use_cache = prev_use_cache
    return results


def compute_wikitext2_ppl(
    model,
    tokenizer,
    device: str = "cuda",
    seqlen: int = 2048,
    cache_dir: str | Path | None = None,
) -> dict[str, float]:
    del device  # The model's parameter device is the source of truth.
    ppls = eval_ppl(
        model,
        tokenizer,
        ["wikitext2"],
        seqlen=seqlen,
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
        verbose=True,
    )
    return {"wikitext2_ppl": ppls["wikitext2"]}
