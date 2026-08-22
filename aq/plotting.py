"""Mandatory plots (plan section 14). Matplotlib only, headless (Agg) so this
works on a GPU box with no display server.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aq.decision_flow import METHOD_LABELS, MethodOutcome


def plot_kl_vs_distance_trace(trace_rows: list[dict], out_path: Path, title: str = "") -> None:
    if not trace_rows:
        return
    steps = [r["step"] for r in trace_rows]
    kl = [r["kl"] for r in trace_rows]
    dist = [r["distance"] for r in trace_rows]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(steps, kl, color="tab:blue", label="KL(M_FP, M_Q)")
    ax1.set_xlabel("optimization step")
    ax1.set_ylabel("KL divergence", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(steps, dist, color="tab:red", label="weight distance D(W, Q(W))")
    ax2.set_ylabel("relative weight distance", color="tab:red")
    plt.title(title or "KL vs weight distance over optimization")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_ppl_comparison(outcomes: list[MethodOutcome], out_path: Path) -> None:
    labels = [METHOD_LABELS.get(o.method_id, o.method_id) for o in outcomes]
    ppls = [o.ppl if o.ppl is not None else float("nan") for o in outcomes]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    ax.bar(labels, ppls, color="tab:orange")
    ax.set_ylabel("WikiText-2 PPL")
    ax.set_title("PPL by method (lower is closer to RTN4 baseline)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_watermark_fsr_comparison(outcomes: list[MethodOutcome], out_path: Path) -> None:
    evaluated = [o for o in outcomes if o.watermark_fsr_exact is not None]
    if not evaluated:
        return
    labels = [METHOD_LABELS.get(o.method_id, o.method_id) for o in evaluated]
    fsrs = [o.watermark_fsr_exact for o in evaluated]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    ax.bar(labels, fsrs, color="tab:green")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("watermark fsr_exact")
    ax.set_title("Watermark FSR by method (0 = watermark gone)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_rounding_flip_ratio_per_layer(layer_metrics_rows: list[dict], out_path: Path, title: str = "") -> None:
    if not layer_metrics_rows:
        return
    layers = [r["layer"] for r in layer_metrics_rows]
    flips = [float(r.get("rounding_flip_ratio_vs_rtn4", 0.0)) for r in layer_metrics_rows]
    fig, ax = plt.subplots(figsize=(max(6, len(layers) * 0.3), 4))
    ax.plot(range(len(layers)), flips, marker="o", markersize=2)
    ax.set_xlabel("layer index (depth order)")
    ax.set_ylabel("rounding flip ratio vs RTN4")
    ax.set_title(title or "Rounding flip ratio by depth")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
