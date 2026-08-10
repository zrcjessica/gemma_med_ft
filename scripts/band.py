# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Workspace-band location (jlens guide §6.3 step 4; paper Figs. 27-28).

Vendored from `medlens/band.py` in the lab's medlens repo
(`/gpfs/data/oermannlab/users/yeb04/jlens/code/medlens`, rev of 2026-08-06).
Kept byte-comparable to that source on purpose -- this is somebody else's
validated method and a divergence here is a divergence in the science, not a
refactor. Two deliberate deltas:

  * `effective_gamma` is inlined rather than imported from `medlens.jspace_patch`,
    so this file has no medlens dependency and runs in our `.venv-jlens`
    (jlens @ 581d398) as-is.
  * `jlens_vector_cka` / `jspace_dimensionality` take the token subsample as an
    argument as well, so the two can share one draw instead of rebuilding the
    (slow, whole-vocab) meaningful-token mask twice.

Metrics from the paper, computed per fitted layer:

* **CKA block structure** among J-lens vectors across layers (paper Fig. 27:
  features are the rows of ``(gamma * W_U) @ J_l``, final-norm weight folded
  into the unembedding). Linear CKA between layers over a subsample of
  meaningful vocabulary tokens exposes the sensory / workspace / motor block
  structure.
* **Next-token prediction accuracy** (Fig. 28a): fraction of positions where
  any top-k J-lens token matches the model's own top-1 prediction -- low in
  the band (readouts carry forward/abstract content), igniting in the motor
  layers.
* **Excess kurtosis** (Fig. 28b) of the lens-logit distribution over the
  vocabulary, averaged over positions of generic text -- near zero in early
  layers, rising once readouts carry real content.
* **Top-1 autocorrelation** (Fig. 28c): P(top-1 token identical at adjacent
  positions), reported as delta log-probability against a position-shuffled
  null -- high when the J-space carries content that persists across
  positions.
* **J-space dimensionality** (Fig. 28d): fraction of residual-stream
  dimensions needed to capture ``var_share`` of the variance across the
  J-lens vectors ``(gamma * W_U) @ J_l`` -- weights-only, like the CKA.

Where medlens draws the "generic text" for the text-statistics pass from a
wikitext-103 sample, we pass in our own frozen fit corpus instead
(`data/jlens/fit_corpus_v2.txt`) -- same intent (the distribution the lens was
fitted on), but it keeps the band probe on the same frozen instrument as the
rest of this project's trajectory work.

``locate_band`` combines kurtosis/autocorrelation/CKA into a suggested
contiguous [lo, hi] layer range; the final band is human-confirmed.
"""

from __future__ import annotations

import numpy as np
import torch

from jlens.lens import JacobianLens
from jlens.protocol import LensModel
from jlens.vis import _meaningful_token_mask


# ---------------------------------------------------------------------------
# Weights-space metrics: CKA and dimensionality of the J-lens vectors


@torch.no_grad()
def effective_gamma(model: LensModel) -> torch.Tensor:
    """Effective final-norm gain ``gamma_eff`` as a ``[d_model]`` fp32 tensor.

    ``final_norm(ones) - final_norm(zeros)`` equals the gain for every
    RMS-family norm (``weight`` for Llama/Qwen, ``1 + weight`` for Gemma).
    A classic LayerNorm maps any constant vector to its bias, making the
    difference ~0; fall back to the raw weight there (mean-subtraction is not
    representable as a diagonal gain anyway).

    Gemma 3 is the ``1 + weight`` case, so reading ``.weight`` directly would
    understate the gain on this whole model line.
    """
    norm = model._final_norm
    w = norm.weight
    ones = torch.ones_like(w)
    g = (norm(ones) - norm(torch.zeros_like(w))).float()
    if g.abs().max() < 1e-6:  # LayerNorm degenerate case
        return w.detach().float()
    return g


@torch.no_grad()
def unembed_token_sample(
    model: LensModel, *, n_tokens: int = 8192, seed: int = 0,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """``(gamma_eff * W_U)[tokens]`` for a seeded subsample of meaningful tokens.

    Slow the first time per tokenizer: the meaningful-token mask decodes the
    whole vocabulary one id at a time (262k ids on Gemma 3). It is cached inside
    jlens per tokenizer id, so draw once and pass the result to both consumers.
    """
    W = model._lm_head.weight  # [vocab, d_model]
    device = device or model.input_device
    try:
        gamma = effective_gamma(model).to(W.device)
    except AttributeError:  # no _final_norm on this adapter
        gamma = getattr(model._final_norm, "weight", None)
    W_eff = (W * gamma if gamma is not None else W).float()
    mask = _meaningful_token_mask(model.tokenizer, W.shape[0], torch.device("cpu"))
    candidates = mask.nonzero(as_tuple=True)[0]
    gen = torch.Generator().manual_seed(seed)
    token_ids = candidates[torch.randperm(len(candidates), generator=gen)[:n_tokens]]
    return W_eff[token_ids.to(W_eff.device)].to(device)  # [n_tokens, d]


@torch.no_grad()
def jlens_vector_cka(
    model: LensModel,
    lens: JacobianLens,
    *,
    n_tokens: int = 8192,
    seed: int = 0,
    device: str | torch.device | None = None,
    w_sub: torch.Tensor | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Pairwise linear CKA between layers' J-lens vector sets (paper Fig. 27).

    Features per layer: ``X_l = (gamma * W_U)[tokens] @ J_l`` with shape
    ``[n_tokens, d_model]`` -- token subsample drawn from the "meaningful"
    vocabulary mask. Linear CKA via d x d cross-covariances:
    ``||X_a^T X_b||_F^2 / (||X_a^T X_a||_F ||X_b^T X_b||_F)``.

    Cost is the quadratic-in-layers part of the band probe: ``n_layers^2 / 2``
    matmuls of ``[d, n_tokens] @ [n_tokens, d]``. At 27b (62 layers, d=5376,
    8192 tokens) that is ~450 TFLOP -- tens of minutes, which is still small
    next to the lens fit that produced ``lens``. Halve ``n_tokens`` if it isn't.

    Returns:
        ``(cka[n_layers, n_layers], layers)``.
    """
    layers = list(lens.source_layers)
    device = device or model.input_device
    W_sub = w_sub if w_sub is not None else unembed_token_sample(
        model, n_tokens=n_tokens, seed=seed, device=device)

    feats = []
    for layer in layers:
        J = lens.jacobians[layer].to(device)
        X = W_sub @ J  # [n_tokens, d]
        X = X - X.mean(0, keepdim=True)
        feats.append(X.half())

    n = len(layers)
    cka = np.eye(n)
    self_norm = [torch.linalg.matrix_norm(X.T.float() @ X.float()).item() for X in feats]
    for a in range(n):
        for b in range(a + 1, n):
            cross = torch.linalg.matrix_norm(feats[a].T.float() @ feats[b].float()) ** 2
            cka[a, b] = cka[b, a] = (cross / (self_norm[a] * self_norm[b])).item()
    return cka, layers


@torch.no_grad()
def jspace_dimensionality(
    model: LensModel,
    lens: JacobianLens,
    *,
    n_tokens: int = 8192,
    var_share: float = 0.90,
    seed: int = 0,
    device: str | torch.device | None = None,
    w_sub: torch.Tensor | None = None,
) -> np.ndarray:
    """Effective linear dimensionality of the J-space per layer (Fig. 28d).

    Per the paper: "the fraction of residual-stream dimensions needed to
    capture a given share of the variance across the J-lens vectors
    ``W_U J_l``". Computed from the singular spectrum of the same centered
    feature matrices the CKA uses; no text is involved.

    Returns:
        ``[n_layers]`` array of fractions in (0, 1].
    """
    layers = list(lens.source_layers)
    device = device or model.input_device
    W_sub = w_sub if w_sub is not None else unembed_token_sample(
        model, n_tokens=n_tokens, seed=seed, device=device)
    d_model = W_sub.shape[1]

    frac = np.zeros(len(layers))
    for i, layer in enumerate(layers):
        J = lens.jacobians[layer].to(device)
        X = W_sub @ J
        X = X - X.mean(0, keepdim=True)
        var = torch.linalg.svdvals(X.float()) ** 2
        cum = torch.cumsum(var, 0) / var.sum()
        n_dims = int(torch.searchsorted(cum, var_share).item()) + 1
        frac[i] = n_dims / d_model
    return frac


# ---------------------------------------------------------------------------
# Text-statistics pass: kurtosis + top-1 autocorrelation


@torch.no_grad()
def readout_text_stats(
    model: LensModel,
    lens: JacobianLens,
    prompts: list[str],
    *,
    max_seq_len: int = 128,
    skip_first: int = 16,
    n_shuffles: int = 200,
    seed: int = 0,
    acc_k: int = 10,
) -> dict[str, np.ndarray]:
    """Per-layer prediction accuracy, excess kurtosis, and top-1
    autocorrelation on generic text (paper Fig. 28a-c).

    Accuracy follows the paper: the fraction of positions at which any of the
    lens's top-``acc_k`` tokens matches the model's own top-1 prediction
    (``acc_topk``; ``acc_top1`` is the strict top-1 variant). Assumes the lens
    jacobians were preloaded to the model device.

    Returns dict with keys ``layers``, ``kurtosis``, ``autocorr``,
    ``autocorr_null``, ``acc_top1``, ``acc_topk``, ``acc_k``
    (all ``[n_layers]`` float arrays except ``layers`` and ``acc_k``).
    """
    layers = list(lens.source_layers)
    kurt_sums = np.zeros(len(layers))
    acc1_sums = np.zeros(len(layers))
    acck_sums = np.zeros(len(layers))
    top1_by_layer: list[list[torch.Tensor]] = [[] for _ in layers]
    n_positions = 0

    for prompt in prompts:
        lens_logits, model_logits, input_ids = lens.apply(
            model, prompt, layers=layers, max_seq_len=max_seq_len
        )
        valid = slice(skip_first, input_ids.shape[1] - 1)
        if input_ids.shape[1] - 1 <= skip_first:
            continue
        model_top1 = model_logits[valid].argmax(-1)  # [n_valid]
        for i, layer in enumerate(layers):
            logits = lens_logits[layer][valid]  # [n_valid, vocab] cpu fp32
            mean = logits.mean(-1, keepdim=True)
            var = logits.var(-1, keepdim=True)
            kurt = (((logits - mean) ** 4).mean(-1) / var.squeeze(-1) ** 2) - 3.0
            kurt_sums[i] += kurt.sum().item()
            top1 = logits.argmax(-1)
            top1_by_layer[i].append(top1)
            acc1_sums[i] += (top1 == model_top1).sum().item()
            topk = logits.topk(acc_k, dim=-1).indices  # [n_valid, k]
            acck_sums[i] += (topk == model_top1[:, None]).any(-1).sum().item()
        n_positions += logits.shape[0]

    rng = np.random.default_rng(seed)
    autocorr = np.zeros(len(layers))
    null = np.zeros(len(layers))
    for i in range(len(layers)):
        match, total = 0, 0
        null_match, null_total = 0, 0
        for seq in top1_by_layer[i]:
            arr = seq.numpy()
            match += (arr[1:] == arr[:-1]).sum()
            total += len(arr) - 1
            for _ in range(max(1, n_shuffles // len(top1_by_layer[i]))):
                sh = rng.permutation(arr)
                null_match += (sh[1:] == sh[:-1]).sum()
                null_total += len(sh) - 1
        autocorr[i] = match / max(total, 1)
        null[i] = null_match / max(null_total, 1)

    return {
        "layers": np.array(layers),
        "kurtosis": kurt_sums / max(n_positions, 1),
        "autocorr": autocorr,
        "autocorr_null": null,
        "acc_top1": acc1_sums / max(n_positions, 1),
        "acc_topk": acck_sums / max(n_positions, 1),
        "acc_k": np.array(acc_k),
    }


# ---------------------------------------------------------------------------
# Band selection


def locate_band(
    cka: np.ndarray,
    layers: list[int],
    kurtosis: np.ndarray,
    autocorr_excess: np.ndarray,
    *,
    kurt_frac: float = 0.25,
    auto_frac: float = 0.25,
) -> tuple[int, int]:
    """Suggest a contiguous workspace band [lo, hi] (inclusive, layer indices).

    Onset: first layer where BOTH excess kurtosis and autocorrelation excess
    exceed ``frac`` of their respective maxima and stay there for 2+ layers.
    Offset: end of the CKA block containing the onset -- found by sweeping the
    largest contiguous set of layers from the onset whose mean pairwise CKA
    stays above the matrix's 60th percentile.

    This is a suggestion generator; confirm against the diagnostic plot.
    """
    k_thr = kurt_frac * kurtosis.max()
    a_thr = auto_frac * autocorr_excess.max()
    onset_idx = 0
    for i in range(len(layers) - 1):
        if (
            kurtosis[i] > k_thr
            and autocorr_excess[i] > a_thr
            and kurtosis[i + 1] > k_thr
            and autocorr_excess[i + 1] > a_thr
        ):
            onset_idx = i
            break

    thr = np.percentile(cka[np.triu_indices_from(cka, 1)], 60)
    end_idx = onset_idx
    for j in range(onset_idx + 1, len(layers)):
        block = cka[onset_idx : j + 1, onset_idx : j + 1]
        if block[np.triu_indices_from(block, 1)].mean() < thr:
            break
        end_idx = j
    return layers[onset_idx], layers[end_idx]


def band_diagnostic_plot(
    cka: np.ndarray,
    layers: list[int],
    stats: dict[str, np.ndarray],
    band: tuple[int, int],
    out_path: str,
    *,
    dim_frac: np.ndarray | None = None,
    sens_band: tuple[int, int] | None = None,
    title: str | None = None,
) -> None:
    """CKA heatmap (paper Fig. 27) + the Fig. 28 panels available in ``stats``.

    ``band`` is shaded on every curve panel; ``sens_band``, when given, is
    shaded again on top (rendering as a darker inner region). ``title``
    overrides the default "suggested workspace band" heading -- use it when
    plotting a human-confirmed band. Falls back gracefully to the original
    three-panel layout when the accuracy/dimensionality metrics are absent.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve_panels = ["kurtosis", "autocorr"]
    if "acc_topk" in stats:
        curve_panels.insert(0, "accuracy")
    if dim_frac is not None:
        curve_panels.append("dimensionality")

    fig, axes = plt.subplots(1, 1 + len(curve_panels),
                             figsize=(4.6 * (1 + len(curve_panels)), 4.5))
    # Contrast-stretch the color scale: within/cross-block CKA differences can
    # be small (e.g. 0.76 vs 0.68), invisible on a fixed 0-1 range.
    off_diag = cka[np.triu_indices_from(cka, 1)]
    im = axes[0].imshow(cka, origin="lower", cmap="viridis",
                        vmin=off_diag.min(), vmax=1.0,
                        extent=[layers[0], layers[-1], layers[0], layers[-1]])
    axes[0].set_title("CKA among J-lens vectors")
    axes[0].set_xlabel("layer"); axes[0].set_ylabel("layer")
    fig.colorbar(im, ax=axes[0])

    ax_i = 1
    if "accuracy" in curve_panels:
        ax = axes[ax_i]; ax_i += 1
        k = int(stats.get("acc_k", 10))
        ax.plot(stats["layers"], stats["acc_topk"], marker="o", ms=3,
                label=f"any top-{k} = model top-1")
        ax.plot(stats["layers"], stats["acc_top1"], marker="o", ms=3,
                label="top-1 = model top-1")
        ax.set_title("next-token prediction accuracy")
        ax.set_xlabel("layer"); ax.set_ylim(0, 1); ax.legend()

    ax = axes[ax_i]; ax_i += 1
    ax.plot(stats["layers"], stats["kurtosis"], marker="o", ms=3)
    ax.set_title("excess kurtosis of readout")
    ax.set_xlabel("layer")

    ax = axes[ax_i]; ax_i += 1
    with np.errstate(divide="ignore"):
        delta_log = np.log(np.maximum(stats["autocorr"], 1e-9)) - np.log(
            np.maximum(stats["autocorr_null"], 1e-9))
    ax.plot(stats["layers"], delta_log, marker="o", ms=3)
    ax.axhline(0.0, ls="--", color="gray", lw=1)
    ax.set_title("top-1 autocorrelation (Δ log p vs shuffled null)")
    ax.set_xlabel("layer")

    if dim_frac is not None:
        ax = axes[ax_i]; ax_i += 1
        ax.plot(stats["layers"], dim_frac, marker="o", ms=3)
        ax.set_title("J-space dimensionality (frac dims, 90% var)")
        ax.set_xlabel("layer"); ax.set_ylim(0, 1)

    for ax in axes[1:]:
        ax.axvspan(band[0], band[1], alpha=0.15, color="tab:green")
        if sens_band is not None:
            ax.axvspan(sens_band[0], sens_band[1], alpha=0.15, color="tab:green")
    if title is None:
        title = f"suggested workspace band: layers {band[0]}-{band[1]}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
