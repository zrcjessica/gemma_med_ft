# CLAUDE.md — gemma_med_ft

Project-specific context for Claude Code. The workspace-wide `~/CLAUDE.md`
(hardware, BigPurple cluster, hub-and-spoke git, Slurm partitions, Slack-notify
convention) still applies and is **not** repeated here.

---

## What this project is

A **training-dynamics / interpretability study**. We fine-tune Gemma 3 on medical
text, save checkpoints across the fine-tune, and measure how two things evolve as
medical fine-tuning proceeds:

1. **The Jacobian lens** ("j-space") — the average input→output Jacobian read-out
   from [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)
   (`jlens`). See `docs/RECIPE.md` and the memory note for the method.
2. **Faithfulness** — concordance between the lens read-out and the model's actual
   output (does the internal signal predict what the model says).

## Scope decisions
- **Objective = medical SFT, text modality first.** "Medical text only" means the
  text-only slice of MedGemma's recipe, which per the tech report is
  **post-training (SFT) only** — *not* continued pretraining on raw text. We use
  the existing `gemma_med/train.py` SFT path (completion-only loss, chat
  template). **Vision is deferred, not abandoned** — we start text-only for
  simplicity, and may fold in the multimodal (MedSigLIP) recipe after the initial
  text experiments, at which point the lens/faithfulness analysis would extend to
  image-conditioned generation too. `docs/RECIPE.md` is the authority on what
  MedGemma did and what we can/can't match.
- **Sweep across all sizes, both `-pt` and `-it`.** `270m / 1b / 4b / 12b / 27b`,
  each in pretrained (`-pt`) and instruction-tuned (`-it`) form. The pt-vs-it
  *starting point* is itself a study axis (how does the trajectory differ from a
  base vs an already-aligned model). This relaxes the original "-it only"
  decision.
- **Full fine-tune where affordable, LoRA only at 12b/27b — as a caveated
  sub-study.** A low-rank weight delta is a *different object* from a
  full-parameter trajectory, so LoRA sizes are not directly comparable to
  full-FT sizes for weight/Jacobian analysis. Prefer full-FT; treat LoRA sizes
  separately (you can merge the adapter per checkpoint to recover full weights).
- **KL is the workhorse faithfulness metric.** Top-1 argmax agreement is noisy at
  small scale / small probe sets; lens↔model KL is the sensitive signal. Report
  both, lean on KL.

## Models (all cached, offline)

Everything lives under the lab shared hub
`/gpfs/data/oermannlab/users/yeb04/hf/hub` (world-readable). `scripts/_resolve_model.sh`
resolves and logs the revision. Use `HF_HOME=/gpfs/data/oermannlab/users/yeb04/hf`
+ `HF_HUB_OFFLINE=1`.

- **Bases:** `gemma-3-{270m,1b,4b,12b,27b}` in `-pt` and `-it`.
  - Gotcha: 270m's base is named `gemma-3-270m` (no `-pt` suffix), so
    `resolve_gemma 270m pt` misses it — pass `BASE_MODEL` explicitly for that arm.
  - Gemma 3 at **4b+ is `Gemma3ForConditionalGeneration`** (a multimodal wrapper);
    1b/270m are plain `Gemma3ForCausalLM`. Both training and `jlens` handle this,
    but the 4b+ jlens path is only *claimed* verified below — confirm with the
    4b spike before trusting large-size lenses.
- **Eval ceilings (also cached):** `medgemma-4b-it`, `medgemma-27b-it`,
  `medgemma-27b-text-it` (the text-only endpoint we approximate).
- **Out of scope:** `gemma-4-31B` / `-it` (different generation, no MedGemma-4).

## Two environments (they conflict — keep them separate)

- **`.venv`** — training + eval. `transformers==4.53.2`, TRL, PEFT, DeepSpeed.
  Built by `scripts/build_env.sh`; eval env `.venv-eval` (vLLM) is separate again.
- **`.venv-jlens`** — the Jacobian-lens probe **only**. `jlens` requires
  `transformers>=5.5` (5.14.1, torch 2.6.0+cu124, + `wandb`), which is
  **hard-incompatible** with the training pin. Never merge these. Built by
  `scripts/build_jlens_env.sh`, which pins `jlens` to commit `581d398`. A
  read-only reference clone at the same commit lives at the sibling
  `~/jacobian-lens` on olab1 (source, `walkthrough.ipynb`, and the paper's curated
  prompt sets under `data/` + the `vis.py` lens visualization).

## Training conventions for the trajectory

Run via `scripts/train_med.sbatch`. Study-specific requirements:

- **Log-spaced checkpoints** (`gemma_med/ckpt_schedule.py`,
  `--ckpt-schedule log`, the default): saves at steps `{1,2,4,8,...}+final`.
  Representations move fastest early; uniform `save_steps` wastes checkpoints.
  The callback owns saving (`save_strategy="no"`); don't also set a uniform save.
- **t=0 is NOT saved** — it *is* the base model. The probe prepends the untouched
  base as the trajectory origin.
- **`--save-only-model`** to keep the many checkpoints affordable on disk.
- **Fixed seed + fixed data order** across every arm, so the pt/it and size
  trajectories are on the same clock and comparable.
- **`--attn eager`** (transformers strongly recommends it for Gemma 3 training).
- **Disk:** full bf16 checkpoints are ~2 B/param; at 27b × many checkpoints × 2
  arms this is TBs. For 12b/27b, prefer fitting the lens inline (or keep only a
  few weight checkpoints) rather than hoarding weights.

## The probe (`scripts/probe_jlens.py` / `.sbatch`, in `.venv-jlens`)

Self-contained — imports nothing from `gemma_med`. For each checkpoint (base +
`checkpoint-*`):

1. **Refit** the lens on a **frozen generic corpus** (`data/jlens/fit_corpus.txt`).
   The lens is a function of the weights, so it must be refit per checkpoint. Keep
   this corpus fixed across all checkpoints and sizes (the paper fits on generic
   web text, not medical — "fit on medical" is a later ablation).
2. **Concordance** on a **frozen probe set** (`data/jlens/probe_set.tsv`,
   `general` vs `medical` cues): per source layer, top-1 agreement + KL between
   lens and model. Watching medical concordance move *relative to the general
   control* is the point.
3. **Drift** of `J_l` vs the t=0 lens (relative-Frobenius + cosine, per layer).

Writes `metrics.jsonl` (one row per checkpoint) + optional W&B. Pair with the
**behavioral** trajectory (`gemma_med/evaluate.py`: MedQA/MedMCQA/PubMedQA) to
correlate the lens signal against what the model actually does; anchor against
the cached `medgemma-27b-text-it`.

Cross-size caveat: raw `J_l` is not comparable across sizes (different hidden
dims / tokenizers) — compare sizes only on derived scalars.

## Run commands

```bash
# Train one arm with log-spaced checkpoints (from .venv, via Slurm)
sbatch --gres=gpu:a100:2 --export=ALL,SIZE=1b,KIND=it,LR=1e-5 scripts/train_med.sbatch

# Probe a finished run's trajectory (from .venv-jlens, detached)
sbatch --export=ALL,SIZE=1b,KIND=it,RUN_DIR=outputs/1b/full_lr1e-5_<jobid> \
    scripts/probe_jlens.sbatch

# jlens <-> Gemma3 integration spike (also the 4b wrapper check)
SPIKE_MODEL=google/gemma-3-4b-it python scripts/spike_jlens_gemma.py
```

## Gotchas

- **Flaky A100 nodes**: some `a100_short` nodes hand out a GPU that reports free
  but rejects every allocation (`a100-4003/4029/4030` seen bad). Jobs preflight
  with a real allocation and requeue onto a fresh node, accumulating an exclude
  list. Keep that pattern in any GPU sbatch.
- **Run multi-minute probes via `sbatch` (detached), not interactive `srun`** —
  an `srun` tied to an SSH session dies when that session is backgrounded.
- **`data/` is gitignored** (large mixtures), with a scoped exception
  (`!/data/jlens/`) so the frozen fit/probe files ARE versioned. Don't commit
  `data/mix_*`.

## Status / phasing

- **Phase 0 — DONE, verified end-to-end.** Scaffold built; 270m-it produced
  checkpoints `{1,2,4,8,16,32,64}`; probe emitted a clean 8-point trajectory
  (drift rises then plateaus; lens↔model KL falls over training, medical below
  general). jlens verified on `gemma-3-1b`.
- **Phase 1 — next.** Confirm the 4b+ multimodal wrapper in jlens; first *real*
  trajectory (1b-it on `mix_default`, real LR/epochs); scale the fit corpus to a
  few hundred+ generic sequences for real runs.
- **Phase 2.** The pt+it × sizes sweep; 12b/27b via LoRA (caveated) or multi-node
  full-FT.
