"""Jacobian-lens trajectory probe.

Runs in the standalone `.venv-jlens` (transformers>=5.5 + jlens); imports nothing
from gemma_med. For each checkpoint on a training trajectory it:

  1. fits a fresh Jacobian lens on a FROZEN generic corpus (the lens is a
     function of the weights, so it must be refit per checkpoint);
  2. measures faithfulness/concordance on a FROZEN probe set -- per layer, does
     the lens's top-1 read-out match the model's actual next token, and the KL
     between the two distributions;
  3. measures how far J_l has drifted from the t=0 (base) lens.

t=0 is the untouched base model, prepended to the checkpoint list. Results are
appended to <out>/metrics.jsonl (one row per checkpoint) and, optionally, logged
to W&B with the training step as the x-axis.

Example:
  python scripts/probe_jlens.py \
    --base-model /path/to/gemma-3-270m-it/snapshots/<rev> \
    --run-dir outputs/270m/full_lr1e-5_<jobid> \
    --fit-corpus data/jlens/fit_corpus.txt \
    --probe-set data/jlens/probe_set.tsv \
    --out eval/jlens/270m_it
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
import transformers

import jlens

log = logging.getLogger("probe_jlens")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True, help="t=0 model + tokenizer source.")
    p.add_argument("--run-dir", default=None, help="Dir containing checkpoint-*/.")
    p.add_argument("--checkpoints", nargs="*", default=None,
                   help="Explicit checkpoint dirs (overrides --run-dir glob).")
    p.add_argument("--fit-corpus", required=True)
    p.add_argument("--probe-set", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fit-prompts", type=int, default=None, help="Cap corpus size.")
    p.add_argument("--dim-batch", type=int, default=16)
    p.add_argument("--fit-max-seq", type=int, default=128)
    p.add_argument("--apply-position", type=int, default=-1,
                   help="Token position read out on each probe prompt.")
    p.add_argument("--save-lenses", action="store_true")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run", default=None)
    return p.parse_args()


def discover_checkpoints(base_model: str, run_dir: str | None, explicit: list[str] | None):
    """[(step, path)] sorted by step, with the base model as step 0."""
    ckpts: list[tuple[int, str]] = [(0, base_model)]
    paths = explicit
    if paths is None and run_dir:
        paths = [str(p) for p in Path(run_dir).glob("checkpoint-*") if p.is_dir()]
    for path in paths or []:
        m = re.search(r"checkpoint-(\d+)", os.path.basename(path.rstrip("/")))
        step = int(m.group(1)) if m else 10**9
        ckpts.append((step, path))
    ckpts.sort(key=lambda x: x[0])
    return ckpts


def load_probes(path: str):
    probes = []
    for line in Path(path).read_text().splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        cat, _, prompt = line.partition("\t")
        probes.append((cat.strip(), prompt.strip()))
    return probes


def is_full_model(path: str) -> bool:
    d = Path(path)
    if (d / "adapter_config.json").exists() and not list(d.glob("model*.safetensors")) \
            and not (d / "pytorch_model.bin").exists():
        return False  # LoRA adapter only -- needs base+merge, out of Phase-0 scope
    return True


@torch.no_grad()
def concordance(lens, model, probes, position):
    """Per category, per source layer: mean top-1 agreement and mean KL(model||lens)."""
    agg = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))  # cat->layer->[agree,kl,n]
    for cat, prompt in probes:
        lens_logits, model_logits, _ = lens.apply(model, prompt, positions=[position])
        m = model_logits[0].float()
        m_top1 = int(m.argmax())
        logp = torch.log_softmax(m, dim=-1)
        p = logp.exp()
        for L, ll in lens_logits.items():
            q = ll[0].float()
            logq = torch.log_softmax(q, dim=-1)
            kl = float((p * (logp - logq)).sum())
            slot = agg[cat][L]
            slot[0] += float(int(q.argmax()) == m_top1)
            slot[1] += kl
            slot[2] += 1

    out = {}
    for cat, per_layer in agg.items():
        layers = sorted(per_layer)
        agree = [per_layer[L][0] / per_layer[L][2] for L in layers]
        kl = [per_layer[L][1] / per_layer[L][2] for L in layers]
        n = len(layers)
        fk = max(1, n // 5)
        cross = next((layers[i] / model.n_layers for i, a in enumerate(agree) if a >= 0.5), 1.0)
        out[cat] = {
            "layers": layers,
            "agree_curve": agree,
            "kl_curve": kl,
            "top1_final": sum(agree[-fk:]) / fk,
            "top1_auc": sum(agree) / n,
            "kl_final": sum(kl[-fk:]) / fk,
            "crossover_frac": cross,
        }
    return out


def jac_drift(base_jac, lens):
    rel, cos = [], []
    for L in lens.source_layers:
        Jt = lens.jacobians[L].detach().float().cpu().flatten()
        J0 = base_jac[L].flatten()
        n0 = torch.linalg.vector_norm(J0)
        rel.append(float(torch.linalg.vector_norm(Jt - J0) / (n0 + 1e-8)))
        nt = torch.linalg.vector_norm(Jt)
        cos.append(float(torch.dot(Jt, J0) / (nt * n0 + 1e-8)))
    return {
        "layers": list(lens.source_layers),
        "relfro_curve": rel,
        "cos_curve": cos,
        "relfro_mean": sum(rel) / len(rel),
        "cos_mean": sum(cos) / len(cos),
    }


def load_hf(path, dtype):
    """Load every trajectory point as the text-only Gemma3ForCausalLM.

    The 4b+ *base* checkpoints declare Gemma3ForConditionalGeneration, but
    train.py saves text-only Gemma3ForCausalLM (model_type=gemma3_text). Letting
    Auto* pick per-path would measure t=0 through the multimodal wrapper and
    every later point through the text model -- two different forwards on one
    trajectory, which silently corrupts drift (measured relative to the t=0 lens)
    and the concordance comparison. Forcing the text submodel keeps all points on
    one code path; the vision tower is irrelevant to a text-only lens and is
    dropped into unexpected_keys.
    """
    errs = []
    order = ["AutoModelForCausalLM", "AutoModelForImageTextToText"]
    try:
        cfg = transformers.AutoConfig.from_pretrained(path)
        if "Gemma3ForConditionalGeneration" in (cfg.architectures or []):
            order.insert(0, "Gemma3ForCausalLM")
    except Exception as e:
        errs.append(f"AutoConfig: {type(e).__name__}: {e}")
    for name in order:
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            return cls.from_pretrained(path, dtype=dtype)
        except Exception as e:
            errs.append(f"{name}: {type(e).__name__}: {e}")
    raise RuntimeError("all loaders failed:\n" + "\n".join(errs))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lenses").mkdir(exist_ok=True) if args.save_lenses else None
    metrics_path = out / "metrics.jsonl"
    metrics_path.write_text("")  # fresh trajectory

    tok = transformers.AutoTokenizer.from_pretrained(args.base_model)
    corpus = [l for l in Path(args.fit_corpus).read_text().splitlines() if l.strip()]
    if args.fit_prompts:
        corpus = corpus[: args.fit_prompts]
    probes = load_probes(args.probe_set)
    ckpts = discover_checkpoints(args.base_model, args.run_dir, args.checkpoints)
    log.info("fit_corpus=%d probes=%d checkpoints=%d", len(corpus), len(probes), len(ckpts))

    wb = None
    if args.wandb_project:
        import wandb
        wb = wandb.init(project=args.wandb_project, name=args.wandb_run,
                        config=vars(args), job_type="jlens-probe")

    base_jac = None
    for step, path in ckpts:
        if not is_full_model(path):
            log.warning("skip step %d: LoRA adapter only (%s)", step, path)
            continue
        log.info("=== step %d :: %s ===", step, path)
        hf = load_hf(path, dtype).to(args.device)
        model = jlens.from_hf(hf, tok)
        lens = jlens.fit(model, corpus, dim_batch=args.dim_batch, max_seq_len=args.fit_max_seq)

        conc = concordance(lens, model, probes, args.apply_position)
        if base_jac is None:
            base_jac = {L: lens.jacobians[L].detach().float().cpu() for L in lens.source_layers}
            drift = {"layers": list(lens.source_layers),
                     "relfro_curve": [0.0] * len(lens.source_layers),
                     "cos_curve": [1.0] * len(lens.source_layers),
                     "relfro_mean": 0.0, "cos_mean": 1.0}
        else:
            drift = jac_drift(base_jac, lens)

        row = {"step": step, "path": path, "n_layers": model.n_layers,
               "d_model": model.d_model, "concordance": conc, "drift": drift}
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

        if wb is not None:
            scalars = {"drift/relfro_mean": drift["relfro_mean"], "drift/cos_mean": drift["cos_mean"]}
            for cat, c in conc.items():
                for k in ("top1_final", "top1_auc", "kl_final", "crossover_frac"):
                    scalars[f"conc/{cat}/{k}"] = c[k]
            wb.log(scalars, step=step)

        if args.save_lenses:
            lens.save(str(out / "lenses" / f"lens_step{step}.pt"))

        for cat, c in conc.items():
            log.info("  [%s] top1_final=%.3f auc=%.3f kl_final=%.3f cross=%.2f",
                     cat, c["top1_final"], c["top1_auc"], c["kl_final"], c["crossover_frac"])
        log.info("  drift relfro_mean=%.4f cos_mean=%.4f", drift["relfro_mean"], drift["cos_mean"])

        del hf, model, lens
        torch.cuda.empty_cache()

    log.info("wrote %s", metrics_path)
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
