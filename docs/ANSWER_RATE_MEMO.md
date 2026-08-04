# Why our fine-tuned models "refuse to answer" — and why they don't

**Date:** 2026-08-03 · **Arm:** `4b_it_full` (with 1b/270m endpoints) ·
**Detail:** `docs/BEHAVIORAL_EVAL.md` §8 · **Data:** `eval/unanswered_causes.json`,
`eval/decode_probe/4b_step512/`

---

## Motivation

Switching from the regex parser to an LLM judge raised answer rate everywhere,
but our fine-tuned models still trail `medgemma-4b-it` by ~7pp on MedMCQA. Since
`acc_raw` counts an unanswered row as wrong, that shortfall was five times larger
than our actual accuracy gap — most of our remaining distance to MedGemma was
formatting, not medicine.

The standing explanation was truncation: the model reasons at length and runs
past `--max-new-tokens 1024` before naming an option. That story implies an easy
fix (raise the cap) and a benign reading (the model knew, we cut it off). It is
wrong.

## What is actually happening

Bucketing every judged `answered=false` row by cause
(`scripts/analyze_unanswered.py`), MedMCQA, n = 4,183:

| Run | unanswered | **repetition loop** | real truncation |
|---|---:|---:|---:|
| `gemma-3-4b-it` (base) | 13 | **0** | 6 |
| `medgemma-4b-it` | 17 | **14** | 2 |
| ours, step-512 | 453 | **437** | 3 |
| ours, step-4882 | 327 | **311** | 8 |

**95% are verbatim repetition loops** — the model repeats one sentence until the
token cap. Two measurements rule truncation out:

- **Answers arrive early.** Among answered rows, the final `Answer:` line sits at
  token p99 = 418, max = 842 — inside the 1024 cap with room.
- **Loops start earlier still.** The repeated span first appears at token
  p50 = 35, p90 = 104. The model derails in the first ~35 tokens and repeats for
  the remaining ~990. A bigger cap buys more of the same text.

MedGemma exhibits the same failure ~20× less often (0.4% of rows run to the cap
vs our 13%). The gap is a rate, not a different mode.

## The cause is greedy decoding, and it is fixable

Repetition is the classic pathology of greedy sampling, and our eval decodes at
`temperature=0.0` with no repetition penalty. Re-decoding step-512 on a fixed
300-item slice under four settings, everything else held constant:

| Arm | MedQA answer rate | MedMCQA answer rate |
|---|---:|---:|
| control `t=0.0 rp=1.00` | 85.7 | 87.3 |
| `t=0.0 rp=1.05` | 91.7 | 93.7 |
| **`t=0.0 rp=1.10`** | **98.7** | **98.3** |
| `t=0.7 rp=1.00` | 99.0 | 98.7 |

A repetition penalty of 1.10 restores answer rate to base/MedGemma levels and
removes ~92% of the loops.

## Implications

**No capability was ever missing.** `acc_raw` gains +3.3pp (MedQA) / +6.0pp
(MedMCQA), but `acc_answered` does *not* rise — on MedQA it falls 55.6 → 51.7,
the recovered items being the ones the model was least sure of. The fix hands
back format loss and nothing else. This independently confirms §6's claim, made
from `acc_answered` staying flat through the mid-trajectory dip, that the dip is
format rather than knowledge.

**§7's MedGemma gap needs restating.** At matched decoding our step-512
checkpoint answers as reliably as MedGemma does. The accuracy gap that survives
that is the real one; the formatting component was ours, not the model's.

**`answer_rate` is a noisier instrument than it looks.** Loops also hit rows the
judge scores as *answered* — 243 of step-4882's MedMCQA rows end at the cap
having already named an option. Fine-tuning teaches the model to commit first
(median answered MedQA output falls 468 → 18 tokens), so a row survives whenever
the answer lands before the derail. `answer_rate` measures whether the loop
started before or after the answer.

**The jlens correlation is unaffected but re-reads.** The lens tracks
`answer_rate` at r = −0.91 on MedMCQA (§6). That correlation is now against a
*decoding* pathology the fine-tune induces, not against lost format knowledge —
still a real property of the weights, but a narrower claim than "the lens tracks
format compliance".

**Secondary, small but real:** 22% of 270m's misses are the model emitting the
prompt's literal `<letter>` placeholder. Fixable by rewording `MCQ_SUFFIX`, at
the cost of comparability with every generation produced so far.

## Open decision

The frozen decode config is unchanged: `temperature=0.0`, `max_new_tokens=1024`,
`repetition_penalty=1.0`. Adopting a new one means re-decoding **and re-judging
every arm** (~$100 of judging plus GPU time) — a change that lands unevenly
across sizes and training steps cannot be corrected for after the fact, the same
argument §5 makes about the regex.

`rp=1.05` may be the better operating point than `rp=1.10`: it halves the loops
while leaving `acc_answered` untouched, whereas 1.10 costs 3.9pp on MedQA by
taxing the legitimate repetition of medical terminology. `t=0.7` fixes it too but
sacrifices determinism.
