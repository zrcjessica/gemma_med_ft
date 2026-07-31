# Behavioral evaluation: from regex parsing to an LLM judge

**Status:** judge is the scorer of record as of 2026-07-30.
**Judge:** `claude-haiku-4-5` via the Anthropic Messages API, `--mode batch`.
**Superseded:** `gemma_med/answer_parsing.py` (regex). Kept only to reproduce
pre-judge figures. Never mix the two instruments in one plot.

---

## 1. Why the change

`evaluate.py` generates free text; something has to decide which option that text
committed to. The regex did this by pattern-matching a letter near the start of
the response. That works on a well-behaved instruction-tuned model and degrades
exactly where this study is most interesting — mid-trajectory, when medical SFT
is pulling the model away from clean answer formatting.

Three specific failures, all of which the judge fixes:

| Failure | Regex behaviour | Judge behaviour |
|---|---|---|
| Model restates an option's text under the wrong letter (`"A. <text of option C>"`) | scores it as A | scores it as C — **content wins over letter** |
| Model paraphrases the option instead of naming it | unparsed | scored |
| Model reasons at length and names the option late, or is truncated first | unparsed | `no_answer` only if it genuinely commits to nothing |

The consequence was not a rounding error. On `270m-it` the regex called
**half to ninety percent** of responses unparseable; the judge finds an answer
in a large share of those (§5).

## 2. Pipeline

Generation and scoring are separate, which is what makes re-scoring cheap:

```
gemma_med/evaluate.py   →  {bench}_predictions.jsonl   (question, options, gold, response)
gemma_med/judge.py      →  {bench}_judgments.jsonl     (idx, gold, chosen, verdict, correct, answered)
                        →  judged_summary.json         (metrics + which judge produced them)
scripts/analyze_traj.py →  joined.json                 (behavioral × jlens, per checkpoint)
```

`evaluate.py` emits **no score at all**. Judging is a network post-process over
files already on disk, so it runs on olab1, not BigPurple. It is idempotent and
resumable, keyed on benchmark item index — re-running only judges what is missing.

## 3. The judge contract

Three design choices carry most of the weight:

**The judge is told the gold answer and forbidden to re-derive it.** Without
this, the metric silently becomes *agreement between two models* rather than
benchmark accuracy. The system prompt says so explicitly: *"Do NOT use your own
medical knowledge to decide what the right answer is… Grade only against it, even
if you disagree."*

**The output is schema-constrained** via Anthropic's
`output_config: {format: {type: "json_schema", …}}`:

```json
{"chosen": "A|B|C|D|E|none", "verdict": "correct|incorrect|no_answer"}
```

(`yes|no|maybe|none` for PubMedQA.) This is load-bearing, not cosmetic — an
unconstrained judge that replies in prose counts as an *error*, which quietly
shrinks the sample rather than announcing itself.

**Scores derive from `chosen`, never from `verdict`.** In `_record`:

```python
"correct":  chosen != "none" and chosen == row["gold"],
"answered": chosen != "none",
```

The judge's own `verdict` string is recorded but never used arithmetically. This
is why the self-disagreement diagnostic (§8) is a quality signal and not a
correctness bug.

**The model's response is embedded untruncated.** Truncating to save money would
bias the single metric this study depends on most: a collapsed model that rambles
for 5000 characters and *then* names an option would be scored `no_answer`,
inflating apparent format collapse exactly where it is being measured.

## 4. Metrics

The three-way verdict preserves the split that raw accuracy destroys:

- **`acc_raw`** — `no_answer` counted wrong. Not interpretable mid-trajectory on
  its own.
- **`acc_answered`** — accuracy among responses that chose an option. *Medical
  signal.*
- **`answer_rate`** — fraction that chose an option. *Format compliance.*

`no_answer` is the direct successor to the regex's "unparsed", so every figure
built on parse-rate/parsed-accuracy has a judge-scored counterpart. Report all
three.

## 5. What the instrument change actually did — measured

The endpoints of all three arms plus all three baselines are judge-scored, and
the regex numbers still exist for the two older arms. So this is a direct
head-to-head on identical generations:

| Arm | Checkpoint | Bench | regex answer rate | judge answer rate | Δ rate | Δ acc_raw |
|---|---|---|---|---|---|---|
| `270m-it` | base (t=0) | medqa | 70.1 | 91.0 | **+20.8** | +6.0 |
| `270m-it` | base | medmcqa | 27.9 | 44.6 | **+16.7** | +11.0 |
| `270m-it` | base | pubmedqa | 6.6 | 10.2 | +3.6 | +2.6 |
| `270m-it` | step-4882 | medqa | 49.2 | 66.0 | **+16.8** | +5.9 |
| `270m-it` | step-4882 | medmcqa | 49.2 | 65.9 | **+16.7** | +6.4 |
| `270m-it` | step-4882 | pubmedqa | 5.0 | 29.2 | **+24.2** | **+22.0** |
| `1b-it` | base | medqa | 99.9 | 99.8 | −0.2 | +0.1 |
| `1b-it` | base | medmcqa | 99.8 | 99.8 | +0.0 | +0.1 |
| `1b-it` | base | pubmedqa | 100.0 | 100.0 | +0.0 | +0.0 |
| `1b-it` | step-4882 | medqa | 87.3 | 91.7 | +4.4 | +2.4 |
| `1b-it` | step-4882 | medmcqa | 82.8 | 91.4 | +8.6 | +3.9 |
| `1b-it` | step-4882 | pubmedqa | 96.8 | 98.0 | +1.2 | +0.8 |

Read this as a calibration curve for the old instrument:

- **On a clean-format model the two agree to ~0.1pp** (`1b-it` base, all three
  benchmarks). The regex was not broken in general.
- **The gap opens exactly where formatting degrades.** `1b-it` moves from 0.0pp
  at t=0 to +8.6pp after fine-tuning; `270m-it`, which never formats cleanly,
  runs +17 to +24pp throughout.
- **Therefore the regex systematically overstated format collapse**, and did so
  worst at small sizes and late in training — i.e. in a way correlated with the
  study's independent variable. That is the failure mode you cannot correct for
  after the fact, which is why the whole corpus is being re-judged rather than
  adjusted.
- `270m-it` step-4882 PubMedQA is the extreme case: acc_raw 2.4 → 24.4. The model
  was answering; the regex could not see it.

## 6. Results: `4b-it` full trajectory (judge-scored, 14 points including t=0)

| Step | drift | MedQA raw / ans / rate | MedMCQA raw / ans / rate | PubMedQA raw / ans / rate |
|---:|---:|---|---|---|
| 0 (base) | 0.000 | 52.9 / 53.1 / 99.5 | 46.6 / 46.8 / 99.7 | 58.2 / 58.2 / 100.0 |
| 2 | 0.032 | 51.5 / 51.7 / 99.6 | 46.9 / 47.1 / 99.6 | 58.6 / 58.6 / 100.0 |
| 4 | 0.026 | 51.5 / 51.6 / 99.8 | 46.8 / 46.9 / 99.7 | 59.2 / 59.2 / 100.0 |
| 8 | 0.027 | 51.8 / 51.9 / 99.7 | 47.2 / 47.3 / 99.8 | 59.6 / 59.6 / 100.0 |
| 16 | 0.133 | 51.9 / 52.4 / 99.1 | 47.1 / 47.8 / 98.6 | 61.4 / 61.4 / 100.0 |
| 32 | 0.331 | 47.9 / 54.7 / 87.6 | 43.6 / 49.1 / 88.7 | 62.4 / 62.4 / 100.0 |
| 64 | 0.457 | 46.3 / 51.7 / 89.6 | 41.4 / 47.7 / 86.8 | 66.6 / 66.9 / 99.6 |
| 128 | 0.475 | 45.4 / 51.7 / 87.9 | 41.3 / 45.4 / 91.0 | 64.8 / 64.8 / 100.0 |
| 256 | 0.490 | 45.6 / 50.6 / 90.3 | 41.8 / 46.2 / 90.4 | 68.6 / 68.7 / 99.8 |
| 512 | 0.531 | 43.0 / 51.4 / 83.8 | 41.2 / 46.2 / 89.2 | 73.4 / 73.7 / 99.6 |
| 1024 | 0.513 | 43.5 / 45.0 / 96.8 | 44.7 / 49.1 / 91.1 | 71.2 / 74.0 / 96.2 |
| 2048 | 0.508 | 49.4 / 56.1 / 88.1 | 47.9 / 52.4 / 91.5 | 73.4 / 73.8 / 99.4 |
| 4096 | 0.482 | 57.0 / 59.7 / 95.6 | 50.2 / 54.7 / 91.7 | 72.2 / 72.5 / 99.6 |
| **4882 (final)** | 0.479 | **56.4 / 58.4 / 96.5** | **50.8 / 55.1 / 92.2** | **72.4 / 72.8 / 99.4** |
| **Δ vs base** | | **+5.3** answered | **+8.3** answered | **+14.6** answered |

Two things this makes visible that raw accuracy alone would not:

**The dip is format, not knowledge.** Between steps 32 and 512, MedQA `acc_raw`
falls 52.9 → 43.0 while `acc_answered` moves 53.1 → 51.4 — essentially flat. The
model has not forgotten medicine; it has stopped reliably naming an option.
Answer rate recovers to 96.5% by the end and accuracy comes with it.

**Medical SFT works at 4b.** +5.3 / +8.3 / +14.6pp answered accuracy over the
untuned base. §7 places that against the `medgemma-4b-it` ceiling.

**Correlation against jlens drift** (14 points, Pearson):

| Bench | drift ~ answer_rate | drift ~ acc_answered |
|---|---:|---:|
| medqa | **−0.737** | +0.100 |
| medmcqa | **−0.912** | +0.373 |
| pubmedqa | −0.415 | +0.910 |

The lens tracks **format compliance**, not competence — strongly on the two MCQ
benchmarks. PubMedQA is the exception and is flagged rather than smoothed over.

## 7. Our fine-tuned 4b vs `medgemma-4b-it`

`medgemma-4b-it` is Google's medical fine-tune of the same base we started from,
so it is the natural ceiling: same architecture, same size, same starting weights,
a vastly larger and better-curated medical corpus. All three models below were
generated by the same `evaluate.py`, on the same held-out items, and scored by the
same frozen judge — so these numbers *are* directly comparable.

**Answered accuracy** (accuracy among responses that chose an option):

| Benchmark | `gemma-3-4b-it` (base) | **ours, step-4882** | `medgemma-4b-it` | ours − base | ours − medgemma | gap closed |
|---|---:|---:|---:|---:|---:|---:|
| MedQA | 53.12 | **58.42** | 66.56 | +5.30 | −8.14 | **39%** |
| MedMCQA | 46.79 | **55.13** | 56.14 | +8.34 | −1.01 | **89%** |
| PubMedQA¹ | 58.20 | **72.84** | 66.20 | +14.64 | **+6.64** | **183%** |

*"gap closed" = (ours − base) / (medgemma − base).*

**Raw accuracy and answer rate**, where the picture is less flattering:

| Benchmark | | base | ours | medgemma |
|---|---|---:|---:|---:|
| MedQA | acc_raw | 52.87 | 56.40 | 66.30 |
| | answer_rate | 99.53 | **96.54** | 99.61 |
| MedMCQA | acc_raw | 46.64 | 50.82 | 55.92 |
| | answer_rate | 99.69 | **92.18** | 99.59 |
| PubMedQA¹ | acc_raw | 58.20 | 72.40 | 66.20 |
| | answer_rate | 100.00 | 99.40 | 100.00 |

> ¹ **PubMedQA is not a fair anchor.** All models here are scored on our stricter
> held-out 500 (`docs/RECIPE.md`), which is not the split the published numbers
> use. Details below; anchor claims on MedQA and MedMCQA.

### What this says

**MedMCQA is close to solved: 89% of the gap, 1.0pp short.** This is the strongest
result — a single-arm SFT run on our mixture recovers nearly all of MedGemma's
advantage on the largest benchmark (4,183 items).

**MedQA is not: only 39% of the gap, 8.1pp short.** MedQA is the harder,
longer-vignette benchmark, and it is where scale and corpus quality still buy
MedGemma something we do not reproduce.

**PubMedQA nominally overshoots — do not quote it as a win.**¹ We score 6.6pp
*above* MedGemma, but the comparison is not sound: `docs/RECIPE.md` records that
we hold out the official 500 items by `pubid` while the paper reports on 1,000,
and MedGemma likely trained on its own PubMedQA test split. Both models are being
measured on our stricter split here, so the *relative* number is at least
apples-to-apples — but the benchmark is small (500 items) and its published
version is inflated. Anchor claims on MedQA and MedMCQA.

**We paid for it in format compliance, and MedGemma did not.** MedGemma answers
99.6% of the time on all three benchmarks — indistinguishable from the untuned
base. Our fine-tune drops to 96.5% / 92.2% / 99.4%. That ~7pp MedMCQA shortfall
is the entire reason our `acc_raw` gap (−5.1pp) is five times our `acc_answered`
gap (−1.0pp): **most of our remaining distance to MedGemma on MedMCQA is
formatting, not medicine.** It is also the behavioral signature of the jlens drift
in §6 — the same format degradation the lens correlates with at r = −0.91.

That gives a concrete next lever: MedGemma's recipe preserves answer formatting
that ours erodes. Recovering those ~7pp of answer rate would close the MedMCQA gap
outright and take a bite out of MedQA, without any new medical knowledge.

### Harness validation

Our numbers reproduce MedGemma's published Table 4 on the two benchmarks that are
comparable:

| | MedQA published / ours | MedMCQA published / ours | PubMedQA¹ published / ours |
|---|---|---|---|
| `gemma-3-4b-it` | 50.7 / 52.87 | 45.4 / 46.64 | 68.4 / 58.20 |
| `medgemma-4b-it` | 64.4 / 66.30 | 55.7 / 55.92 | 73.4 / 66.20 |

Landing within ~2pp on *both* models, on numbers we did not produce, is the
strongest end-to-end evidence that the prompt, generation, and judge pipeline are
sound. **Treat future MedQA/MedMCQA drift beyond ~2pp on these two models as a
harness regression, not a finding.** The PubMedQA shortfall is the split
difference described above, not a bug.

Regenerate this comparison with `scripts/compare_medgemma.py` →
`eval/medgemma_comparison.json`, which also carries the 270m and 1b arms.

## 8. Validity check: judge self-disagreement

`judge_self_disagreement` counts rows where the judge's
`verdict` string conflicts with what its `chosen` implies. Across the 4b
trajectory + baseline (83,384 items) it is **0.95% overall**, 0.2–0.5% at the
ends, peaking at **3.0%** (step-128 medmcqa).

This does **not** invalidate a run. Inspecting those rows, ~73% are one benign
shape: the model emits long chain-of-thought, hits the generation limit before
naming an option, and the judge sets `chosen=none` (correct) while labelling the
verdict `incorrect` rather than `no_answer`. Since §3 shows both `correct` and
`answered` derive from `chosen`, **zero metrics are affected**. It tracks how
rambly the *judged* model is, not how confused the judge is. Escalate only if the
disagreement appears in `chosen`.

## 9. Cost and wall-clock (measured)

| | |
|---|---|
| Judged 2026-07-30 | `4b_it_full` (13 checkpoints) + `gemma-3-4b-it-baseline` |
| Items | 83,384 (14 dirs × 5,956) |
| Wall-clock | **11 min 18 s** (15:55:33 → 16:06:51), batch mode |
| Cost | **~$34** (`--estimate`, extrapolated from a 25-row sample per dir; not billed-verified) |
| Per trajectory | ~$32 batch, ~$64 sync |
| Whole corpus (all 3 arms + baselines, 256k items) | ~$103 batch / ~$206 sync |

Cost is dominated by the judged model's **own output length**, which is embedded
untruncated. On `4b_it_full` medqa the mean response is ~2,100 chars at
step-2/step-128 but ~356 chars at step-1024 — per-item input tokens swing roughly
600 → 1,200. **Pricing one checkpoint and multiplying by N underestimates the
total.** Budget from a whole-corpus character count.

Prompt caching does not help: the shared SYSTEM prompt is 423 tokens, below
Haiku's 1024-token cache minimum.

## 10. Running it

```bash
# from .venv-judge, on olab1. Key in .env (gitignored).
export ANTHROPIC_API_KEY=...

# always price it first
.venv-judge/bin/python -m gemma_med.judge --root eval/traj/<tag> --estimate

# batch mode is two passes ON PURPOSE. Without --no-wait, the judge polls each
# batch to completion before creating the next, serialising ~42 batches.
.venv-judge/bin/python -m gemma_med.judge --root eval/traj/<tag> --mode batch --no-wait   # submit all
.venv-judge/bin/python -m gemma_med.judge --root eval/traj/<tag> --mode batch             # collect
.venv-judge/bin/python -m gemma_med.judge --root eval/traj/<tag> --mode sync              # sweep stragglers

# single dir / smoke test
.venv-judge/bin/python -m gemma_med.judge --pred-dir eval/gemma-3-4b-it-baseline --mode sync --limit 50

# then join against the lens
.venv-judge/bin/python scripts/analyze_traj.py \
    --traj-dir eval/traj/<tag> --lens-metrics eval/jlens/<tag>/metrics.jsonl \
    --baseline eval/gemma-3-4b-it-baseline --out eval/traj/<tag>/joined.json
```

The final `--mode sync` sweep is not optional. The judge exits 0 when a batch has
a handful of failed items, so the collect loop will not retry them — the 4b run
had 4 such items across three batches, picked up by the sweep.

**Providers.** `--provider` also accepts `moonshot` and `local`, but **only
`anthropic` is supported for this study.** It is the sole provider with a Batches
API — hence the only half-price offline path — and it is the judge of record; no
result in this document was produced by any other provider. `--mode batch` on the
non-batching providers is rejected, not silently downgraded.

## 11. Coverage

| Dir | Judged | Notes |
|---|---|---|
| `eval/traj/4b_it_full/` | ✅ all 13 checkpoints | complete trajectory |
| `eval/gemma-3-4b-it-baseline/` | ✅ | t=0 for the above |
| `eval/gemma-3-1b-it-baseline/` | ✅ | |
| `eval/gemma-3-270m-it-baseline/` | ✅ | |
| `eval/medgemma-4b-it/` | ✅ | ceiling reference — see §7 |
| `eval/traj/1b_it_full_v3/step-4882` | ✅ endpoint only | mid-trajectory still regex |
| `eval/traj/270m_it_full/step-4882` | ✅ endpoint only | mid-trajectory still regex |

**113,164 items judged**, all by `claude-haiku-4-5`, all provider `anthropic`.

**The open gap:** `270m_it_full` and `1b_it_full_v3` have judge-scored *endpoints*
but regex-scored *interiors*, so their `joined.json` still carries
`acc_parsed`/`parse_rate`. Until those two trajectories are fully re-judged
(~$32 and ~20 min each), **no cross-size behavioral claim is legitimate** — one
arm would be judge-scored and two regex-scored, and §5 shows the two instruments
diverge by up to 24pp precisely as a function of size and training step. The
slide deck quarantines the regex-scored arms on their own slide for this reason.

## 12. Standing rules

1. **The judge is an instrument — freeze it.** One judge model per trajectory,
   recorded in `judged_summary.json`. Same discipline as the frozen jlens fit
   corpus and probe set.
2. **Never mix regex and judge scores in one figure.** They are not comparable;
   §5 quantifies by how much.
3. **Never truncate the embedded model response** to reduce cost.
4. **Report all three metrics.** `acc_raw` alone is not interpretable
   mid-trajectory.
5. **Price with `--estimate` before every run.**
