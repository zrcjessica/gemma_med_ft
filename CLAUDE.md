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

## Environments (they conflict — keep them separate)

- **`.venv`** — training + eval. `transformers==4.53.2`, TRL, PEFT, DeepSpeed.
  Built by `scripts/build_env.sh`; eval env `.venv-eval` (vLLM) is separate again.
- **`.venv-judge`** — the LLM judge (`gemma_med.judge`) **only**: `anthropic` +
  `datasets`, no torch. Built by `scripts/build_judge_env.sh`, which pins
  Python 3.11 and `anthropic>=0.115` — on 3.8 the resolver silently caps at
  0.72, which has no `output_config`, so the judge would lose its schema
  constraint. Judging is a network post-process over files already on olab1, so
  run it here, not on BigPurple.
- **`.venv-jlens`** — the Jacobian-lens probe **only**. `jlens` requires
  `transformers>=5.5` (5.14.1, torch 2.6.0+cu124, + `wandb`), which is
  **hard-incompatible** with the training pin. Never merge these. Built by
  `scripts/build_jlens_env.sh`, which pins `jlens` to commit `581d398` and adds
  `datasets` (a lazy import inside `jlens.examples`). The same commit is checked
  out at the sibling `~/jacobian-lens` on olab1 — don't *edit* it, but do **call
  it**: `jlens` is installed in `.venv-jlens` from this SHA, so everything in
  `jlens/*.py` is importable, and it *is* the paper's method — a hand-rolled
  equivalent is a deviation we'd have to defend. `grep -rn` the package source
  before writing any helper; the surface is wider than the README:
  `examples.py` has `load_wikitext_prompts` (the fit corpus) and the curated
  `EXAMPLES`/`resolve_prompt`, `vis.py` has the slice visualization, `fitting.py`
  documents the estimator, and `data/evaluations/` + `data/experiments/` hold the
  paper's prompt sets. If a packaging split blocks the import, fix the packaging
  (check `uv pip install --dry-run` is additive) rather than reimplementing.

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

## Behavioral scoring is an LLM judge, not a regex

`evaluate.py` only **generates** — it writes `*_predictions.jsonl` (question,
options, gold, response) and no score. Scoring is `gemma_med.judge`, a
Claude-Haiku judge that returns `correct` / `incorrect` / `no_answer` per item
into `*_judgments.jsonl`; `analyze_traj.py` reads those. `answer_parsing.py` is
superseded and kept only to reproduce pre-judge numbers — never mix its scores
with judge scores in one figure.

Consequences to keep in mind:
- **`no_answer` is the successor to "unparsed"**, so acc_raw / acc_answered /
  answer_rate still separate lost knowledge from lost format compliance. Report
  all three; acc_raw alone is not interpretable mid-trajectory.
- **The judge matches option *content*, not the letter.** "A. <text of option
  C>" scores as a choice of C. This is more generous than the regex was, so
  judge numbers are not directly comparable to any figure made before the switch.
- **The judge is told the gold answer and forbidden to re-derive it**, so the
  metric stays benchmark accuracy rather than agreement between two models.
- Every judgment carries both the judge's verdict and the option it picked;
  `judge_self_disagreement` counts rows where those conflict. **It does not
  invalidate a run.** `_record` derives *both* `correct` and `answered` from
  `chosen`, never from `verdict`, so a mismatch changes no score — it is a
  quality signal only. Measured on `4b_it_full` (2026-07-30): 0.95% over 83,384
  rows, 0.2–0.5% at the ends. Of those 788 rows, 58% are `verdict=incorrect`
  while `chosen` *is* gold (a slip in the free-text field; the scored field is
  right), and 35% are the model explicitly rejecting every option — "Answer:
  None of the above" — which the judge records as `chosen=none` but labels
  `incorrect` rather than `no_answer`. Only **7 of 788 hit the token cap**, so
  this does *not* track truncation; an earlier note here claiming ~73% were
  truncated long CoT was wrong (`docs/BEHAVIORAL_EVAL.md` §9). Investigate the
  rows before discarding anything; escalate only if the disagreement is in
  `chosen`.
- Judging is **resumable and idempotent** (keyed on item index). Price a run
  first with `--estimate`.
- **Three providers.** `--provider anthropic` (Claude, `ANTHROPIC_API_KEY`) is
  the only one with a Batches API, hence the only half-price offline path;
  `--provider moonshot` (Kimi hosted API) is sync-only, which cancels most of
  its lower per-token price; `--provider local` is the **lab's self-hosted Kimi
  on BigPurple** — free and rate-limit-free, but it is somebody else's Slurm job
  and vanishes without notice. **`anthropic` is the judge of record as of
  2026-07-30** (`claude-haiku-4-5`, `--mode batch`): ~$32 per trajectory, ~11
  minutes wall-clock, runs on olab1. The two agree to within 1.3pp on identical
  items (45.0 vs 46.3 acc_raw on 300 medqa rows at 4b step-1024), so the
  freeze-the-judge rule is about reproducibility, not about either being wrong.
  `--mode batch` on the non-batching providers is rejected, not silently
  downgraded (an *unset* `--mode` resolves per provider — batch where there is a
  Batches API, sync where there isn't — so switching provider is one variable).
  Hosted Kimi is reached via `api.moonshot.ai/v1`, **not** its
  Anthropic-compatible endpoint: that one speaks OpenAI's `response_format`, not
  Anthropic's `output_config`, so pointing the `anthropic` SDK at it would drop
  the schema constraint without erroring.

### The self-hosted Kimi server (verified 2026-07-30)

`sbatch --export=ALL,TAG=<tag> scripts/judge_traj.sbatch` — ~10 judgments/s at
`CONC=32`, so a full 40-checkpoint trajectory (~238k items) is roughly 7h and
costs nothing. Four things about that server are not guessable and each one
silently breaks the judge:

- **The node moves.** It is somebody's Slurm job (`squeue -a | grep -i kimi`).
  The address that worked last week is a stranger's pretraining job today, and
  the symptom is every judgment failing to connect. `scripts/kimi_url.sh`
  discovers it; `judge_traj.sbatch` calls that. Never hardcode a node.
- **Only the head node answers.** A server on `sp-[0005,0008]` serves on
  `sp-0005`; `sp-0008` refuses.
- **Reachable from anywhere inside BigPurple** — login node and compute nodes
  both work — but **not from olab1**. So judging runs on BigPurple. Submit it
  (`cpu_short`, no GPU) rather than running on the login node: a full trajectory
  is hours of wall-clock.
- **It is a reasoning model, and the served id is not the HF name.** The lab
  server answers to `Barney`, not `Kimi-K2.7-Code`. Left alone it burns hundreds
  of reasoning tokens, returns `content: null`, and puts the text in a separate
  `reasoning` field — so the obvious `.choices[0].message.content` is `None`.
  `LocalJudge` sends `chat_template_kwargs: {"enable_thinking": false}` (115 →
  14 completion tokens, same verdict) and keeps a generous `max_tokens` in case
  a server build ignores it. `response_format: json_schema` *is* enforced;
  without it the model invents its own keys.

- **The judge is an instrument — freeze it.** Same discipline as the frozen fit
  corpus and probe set: one judge model for a whole trajectory, recorded in
  `judged_summary.json`. Two judges' scores are not comparable. **Switching
  providers is a per-run toggle, never a per-dir one**: `PROVIDER=<local|
  anthropic>` on `scripts/judge.sh` (olab1) and on both judge sbatch scripts,
  which share `scripts/_judge_provider.sh` for endpoint discovery, mode, and the
  batch submit→collect→sweep passes. Judging is resumable, so pointing the other
  provider at an already-judged dir would quietly fill only the missing rows —
  `judge.py` refuses unless `--allow-judge-switch`, and records
  `judge_mixed_with` in the summary when you insist.
- **So are the decode settings.** `temperature=0.0`, `max_new_tokens=1024`,
  `repetition_penalty=1.0` is the frozen config, now recorded in `summary.json`.
  `--repetition-penalty` / `--seed` exist only for the decode probe
  (`scripts/redecode_probe.sbatch`); its output lives in `eval/decode_probe/`
  and may not be mixed into trajectory figures.

## Unanswered ≠ truncated

`answer_rate` lags MedGemma by ~7pp on MedMCQA, and the cause is **verbatim
repetition loops under greedy decoding, not chain-of-thought running past the
token cap**. At 4b step-4882, 311 of 327 unanswered MedMCQA rows are loops and 8
are real truncation; the model derails at token p50=35 while answers, when they
come, land by p99=418 — so a bigger cap recovers nothing. MedGemma loops on
0.4% of rows, we loop on 13%. Bucket any eval dir with
`scripts/analyze_unanswered.py`; full write-up in `docs/BEHAVIORAL_EVAL.md` §8.

**It is the decoding, and it is fixable** (measured 2026-08-03 on step-512, 300
items/bench): `repetition_penalty=1.10` takes answer rate 85.7 → 98.7 (MedQA)
and 87.3 → 98.3 (MedMCQA), i.e. back to base/MedGemma levels; `temperature=0.7`
does the same. `acc_raw` gains +3.3/+6.0pp, `acc_answered` does **not** rise —
no capability appears, it was never missing. The frozen config is still
`t=0.0, rp=1.0`: adopting a new one means re-decoding *and* re-judging every arm.

## Run commands

```bash
# Train one arm with log-spaced checkpoints (from .venv, via Slurm)
sbatch --gres=gpu:a100:2 --export=ALL,SIZE=1b,KIND=it,LR=1e-5 scripts/train_med.sbatch

# Price a trajectory before judging it (always)
python -m gemma_med.judge --root eval/traj/1b_it_full_v3 --estimate

# Judge with Claude (default; ~$32, ~11 min per trajectory, on olab1).
# The key comes from .env automatically; scripts/judge.sh does the batch
# submit-all -> collect -> sync-sweep passes, all three of which are load-bearing.
scripts/judge.sh --root eval/traj/4b_it_full
scripts/judge.sh --pred-dir eval/gemma-3-1b-it-baseline

# Same thing with the lab's self-hosted Kimi (free; BigPurple compute node only)
scripts/kimi_url.sh                 # which node is serving right now
PROVIDER=local scripts/judge.sh --root eval/traj/4b_it_full     # on a compute node
sbatch --export=ALL,TAG=1b_it_full_v3 scripts/judge_traj.sbatch            # PROVIDER=local default
sbatch --export=ALL,TAG=1b_it_full_v3,PROVIDER=anthropic scripts/judge_traj.sbatch
sbatch --export=ALL,TAG=1b_it_full_v3,LIMIT=50 scripts/judge_traj.sbatch   # smoke test

# The raw module still takes the same flags if you need one pass at a time
python -m gemma_med.judge --root eval/traj/4b_it_full --mode batch --no-wait  # submit all
python -m gemma_med.judge --root eval/traj/4b_it_full --mode batch           # collect
python -m gemma_med.judge --root eval/traj/4b_it_full --mode sync            # sweep stragglers

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
