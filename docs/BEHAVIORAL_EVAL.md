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
is why the self-disagreement diagnostic (§9) is a quality signal and not a
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

## 8. Why answer rate lags: repetition loops, not truncation

§7 ends on a lever — recover the ~7pp of MedMCQA answer rate that MedGemma keeps
and we lose. This section identifies what is actually consuming it, because the
obvious hypothesis is wrong and the two candidates call for opposite fixes.

The hypothesis was truncation: the model reasons at length, runs past
`--max-new-tokens 1024`, and never reaches its answer line. That story is
plausible, it was written into §9 as the explanation for the self-disagreement
rows, and it is false for every fine-tuned checkpoint we have. **The generations are not
unfinished reasoning; they are verbatim repetition loops** — the classic
pathology of greedy decoding.

### Method

`scripts/analyze_unanswered.py` joins predictions to judgments and buckets every
`answered=false` row. A row that ended at the token cap is a loop if the unique
8-gram share of its last 300 words is below 0.6 (digits normalised, because the
loops often present as an enumeration whose counter keeps climbing), and honest
truncation otherwise. Full table: `eval/unanswered_causes.json`.

### The result

MedMCQA, the benchmark carrying the shortfall. Every run below evaluates the
same 4,183 items; `unanswered` is the count of those scored `answered=false`,
and the remaining columns partition it:

| Run | unanswered | **loop** | real truncation | literal `<letter>` | other |
|---|---:|---:|---:|---:|---:|
| `gemma-3-4b-it` (base) | 13 | **0** | 6 | 0 | 7 |
| `medgemma-4b-it` | 17 | **14** | 2 | 0 | 1 |
| ours, step-32 | 472 | **378** | 46 | 17 | 31 |
| ours, step-512 | 453 | **437** | 3 | 5 | 8 |
| ours, step-4882 | 327 | **311** | 8 | 1 | 7 |
| `1b_it_full` step-4882 | 360 | **356** | 0 | 0 | 4 |
| `270m_it_full` step-4882 | 1,425 | **1,038** | 5 | 318 | 64 |

95% of our 4b's unanswered rows are loops. Eight are the thing we assumed all of
them were.

Two measurements make it decisive, both in tokens, both from the same script:

**The answer, when it comes, comes early.** For answered rows at step-4882, the
final `Answer:` line sits at token p50 = 54, p99 = 418, **max = 842** on MedMCQA
(p99 = 633, max = 790 on MedQA) — every one inside the 1024 cap, with room. The
cap is not binding on generations that behave.

**The loop starts long before any cap.** For unanswered rows, the repeated span
is first seen at token **p50 = 35, p90 = 104** (MedMCQA, step-4882). The model
derails inside the first ~35 tokens and then repeats for the remaining ~990.
Raising `--max-new-tokens` buys more of the same text.

What the loops look like — three real step-512 generations:

```
The thyroid cartilage is attached to the cricoid cartilage by the thyroepiglottic ligament.  (x40)
...not vulnerable to compression of the T39 nerve root. ...the T40 nerve root. ...the T41...
Congenital heart disease 2. Congenital cataracts 3. Cleft lip ... 7. Leukemia 8. Leukemia ... 182. Leukemia
```

### Three consequences

**The gap to MedGemma is a rate, not a mode.** MedGemma loops too — 14 of its 17
MedMCQA misses are loops. It just does it ~20× less often. Rows running to the
cap: base **0.2%**, `medgemma-4b-it` **0.4%**, ours at step-512 **21.4%**, at
step-4882 **13.4%**. Our SFT induces the pathology; MedGemma's post-training
does not.

**Answer rate undercounts the problem.** Looping also hits rows the judge scores
as *answered*: 243 of step-4882's MedMCQA rows end at the cap having already
named an option (237 of those 243 are looping). Fine-tuning teaches the model to
commit first — median answered MedQA output falls from 468 tokens at t=0 to 18
at step-4882 — so a row survives whenever the answer lands before the derail.
**`answer_rate` is measuring whether the loop started before or after the answer**,
which is a noisier quantity than "did the model know the answer".

**The base model's few failures really were truncation.** All 6 of
`gemma-3-4b-it`'s MedQA misses are honest long CoT, and its answered rows commit
at p99 = 716 / max = 973 — genuinely close to the cap. So the cap is near-binding
for the *verbose base* and irrelevant for the *looping fine-tune*. Don't
generalise one arm's failure mode to the others; that is the mistake this section
corrects.

### Secondary cause: the prompt's own placeholder

`MCQ_SUFFIX` asks the model to end with `'Answer: <letter>'`, and the model
sometimes emits that string verbatim rather than filling it in. Small and
transient for 4b (113 MedMCQA rows at step-64, gone by step-512), but **318 of
270m's 1,425 misses — 22%**. Rewording the suffix would fix it and would also
break comparability with every generation produced so far (§13, rule 1 applies to
the prompt as much as to the judge). Left frozen deliberately; noted here so the
270m numbers are read with it in mind.

### It is the decoding — measured

Our eval decodes greedily (`temperature=0.0`) with no repetition penalty, which
is exactly where degenerate repetition is expected to live. So: is the shortfall
a property of *the model* or of *how we sample it*?

`scripts/redecode_probe.sbatch` answers it. One checkpoint (step-512, the worst
point), a fixed 300-item slice of MedQA and MedMCQA, four decode settings, model
and prompt and items and judge held constant. Judged 2026-08-03 by the same
frozen `claude-haiku-4-5`, $2.16, sync mode.

| Arm | MedQA rate / raw / answered | MedMCQA rate / raw / answered | loops (of 300+300) |
|---|---|---|---:|
| **control** `t=0.0 rp=1.00` | **85.7** / 47.7 / 55.6 | **87.3** / 40.3 / 46.2 | 41 + 37 |
| `t=0.0 rp=1.05` | 91.7 / 51.0 / 55.6 | 93.7 / 45.7 / 48.8 | 23 + 16 |
| `t=0.0 rp=1.10` | **98.7** / 51.0 / 51.7 | **98.3** / 46.3 / 47.1 | 3 + 3 |
| `t=0.7 rp=1.00` | **99.0** / 54.3 / 54.9 | **98.7** / 43.3 / 43.9 | 3 + 1 |

**A repetition penalty of 1.10 removes the shortfall.** Answer rate goes
85.7 → 98.7 and 87.3 → 98.3, i.e. back to where the untuned base and
`medgemma-4b-it` sit (99.5%+), and the loop count collapses by ~92%. Sampling at
`t=0.7` does the same thing by the same mechanism. The control arm reproduces the
trajectory row on this slice, so the comparison is internally anchored.

**The rows it recovers are worth about what the model is generally worth, not
more.** `acc_raw` rises +3.3pp (MedQA) and +6.0pp (MedMCQA) — that is the format
loss being handed back. `acc_answered` does *not* rise; on MedQA it falls
55.6 → 51.7, because the recovered items are the ones the model was least sure
of. **No medical capability appears. It was never missing** — which is precisely
what §6 claimed from `acc_answered` staying flat through the dip, now confirmed
by a second, independent route.

Caveats, in order of how likely they are to bite:

- **One checkpoint, 300 items per benchmark.** Enough to identify the mechanism,
  not enough to restate §6 or §7. Re-deciding the study's decode config means
  re-decoding every arm.
- **A repetition penalty is not free.** It taxes legitimate repetition, and
  medical answers repeat terminology by nature. The 3.9pp `acc_answered` drop on
  MedQA at `rp=1.10` is the visible edge of that; `rp=1.05` halves the loops
  while leaving `acc_answered` untouched, and may be the better operating point.
- **`t=0.7` buys the same fix at the cost of determinism.** Seeded here, but
  sampling adds run-to-run variance to every number in the study. If the config
  ever changes, greedy + a penalty is the smaller intervention.

**What this does to §7's conclusion.** Our remaining MedMCQA distance to MedGemma
was attributed there to format compliance rather than medicine. That now has a
mechanism and a demonstrated fix: it is degenerate repetition under greedy
decoding, and it is ours, not the model's. The honest statement of §7's gap is
that at matched decoding our step-512 checkpoint answers as reliably as MedGemma
does — the accuracy gap that remains after that is the real one.

```bash
sbatch --array=0-3 scripts/redecode_probe.sbatch              # BigPurple, ~6 min/arm on 1 A100
.venv-judge/bin/python -m gemma_med.judge --pred-dir eval/decode_probe/4b_step512/<arm> --mode sync
.venv-probe/bin/python scripts/analyze_unanswered.py eval/decode_probe/4b_step512/*
```

`--root` only globs `step-*`, so the arms are judged one `--pred-dir` at a time.
Array tasks each get their own `VLLM_CACHE_ROOT`: co-scheduled vLLM processes
otherwise race on the shared torch-compile cache and die with "corrupted
compilation artifact" (it killed 3 of 4 on the first submission).

> **Decode settings are part of the instrument.** `temperature=0.0`,
> `max_new_tokens=1024`, `repetition_penalty=1.0` **remains** the frozen
> configuration and the only one any published number may use — the result above
> is a reason to consider changing it, not a licence to have changed it.
> `evaluate.py` now records the config in `summary.json` for exactly this reason.
> The probe lives in its own `eval/decode_probe/` tree and its numbers may not be
> mixed into §6 or §7 — same rule as regex-vs-judge (§13). Changing the config
> means re-decoding **and re-judging every arm**, on the same argument §5 makes
> about the regex: a change that lands unevenly across sizes and training steps
> cannot be corrected for after the fact.

Predictions written from this commit on also carry `finish_reason` and
`n_gen_tokens`, so "did this response hit the cap?" is a field lookup. Older
dirs, including everything above, are handled by re-tokenising.

## 9. Validity check: judge self-disagreement

`judge_self_disagreement` counts rows where the judge's
`verdict` string conflicts with what its `chosen` implies. Across the 4b
trajectory + baseline (83,384 items) it is **0.95% overall**, 0.2–0.5% at the
ends, peaking at **3.0%** (step-128 medmcqa).

This does **not** invalidate a run: §3 shows both `correct` and `answered` derive
from `chosen`, so a wrong `verdict` string changes **zero metrics**. Escalate only
if the disagreement appears in `chosen`.

All 788 rows, by shape:

| Shape | n | % |
|---|---:|---:|
| `verdict=incorrect` while `chosen` **is** the gold option | 460 | 58% |
| `verdict=incorrect` while `chosen=none` | 274 | 35% |
| `verdict=correct` while `chosen` is **not** the gold option | 54 | 7% |

- The **274** are dominated (84%) by one clean shape: the model explicitly
  rejects every option — *"…therefore there is no correct answer among the given
  options. Answer: None of the above"*. The judge records `chosen=none`, which is
  right, and then calls the verdict `incorrect` rather than `no_answer`. Both
  labels are defensible for a response that committed to something not on the
  list.
- The **460** are a slip in the free-text `verdict` field only: in 92% the
  model's own letter agrees with the judge's `chosen`, and `chosen` equals gold,
  so the scored field is right and the string is wrong. The likeliest cause is
  the judge reacting to faulty *reasoning* behind a correct *choice*, which §3's
  "do not re-derive the medicine" instruction is meant to suppress. Not pinned
  down; it is 0.55% of rows and affects nothing.

**Earlier revisions of this section attributed ~73% of these to long
chain-of-thought hitting the generation limit. That is wrong** — only **7 of 788
(1%)** ended at the token cap. Truncation is not what this diagnostic tracks;
§8 covers what actually happens at the cap.

## 10. Cost and wall-clock (measured)

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

## 11. Running it

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
non-batching providers is rejected, not silently downgraded; an *unset* `--mode`
resolves per provider (batch where there is a Batches API, sync where there is
not), so the provider is the only variable that has to change.

Switching is one env var, and the wrappers absorb the rest — endpoint discovery
for the self-hosted server, the mode, and the three batch passes — via
`scripts/_judge_provider.sh`:

```bash
scripts/judge.sh --root eval/traj/<tag>                  # anthropic (default here), on olab1
PROVIDER=local scripts/judge.sh --root eval/traj/<tag>   # lab Kimi, compute node only
sbatch --export=ALL,TAG=<tag>[,PROVIDER=anthropic] scripts/judge_traj.sbatch
```

**The toggle is per run, not per directory.** Judging is resumable and keyed by
item index, so re-running an already-judged dir under the other provider would
fill in only the rows the first judge missed and then label the whole file with
the second judge — §5's instrument-divergence numbers are what that costs.
`judge.py` refuses the switch unless you pass `--allow-judge-switch`, which
records `judge_mixed_with` in `judged_summary.json` so the mixing is at least
visible downstream. `PROVIDER=anthropic` reads `ANTHROPIC_API_KEY` from repo-root
`.env` if it is not exported; `.env` is gitignored, so it is not on BigPurple
unless you copied it there.

## 12. Coverage

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

## 13. Standing rules

1. **The judge is an instrument — freeze it.** One judge model per trajectory,
   recorded in `judged_summary.json`. Same discipline as the frozen jlens fit
   corpus and probe set.
2. **Never mix regex and judge scores in one figure.** They are not comparable;
   §5 quantifies by how much.
3. **Never truncate the embedded model response** to reduce cost.
4. **Report all three metrics.** `acc_raw` alone is not interpretable
   mid-trajectory.
5. **Price with `--estimate` before every run.**
