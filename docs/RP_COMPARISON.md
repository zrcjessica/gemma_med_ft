# rp = 1.05 vs rp = 1.10

Measured 2026-08-10. Paired per-item comparison of the two repetition-penalty
arms across all five `-it` full-FT trajectories (`270m/1b/4b/12b/27b`), every
log-spaced checkpoint, all three benchmarks. 198 paired benchmark cells plus six
paired anchors; 786,069 judged responses. Everything except `repetition_penalty`
is held: `temperature=0.0`, `max_new_tokens=1024`, `seed=0`, judge `local`/`Barney`
at `temperature=0.0, seed=0`. No dir carries `judge_mixed_with`.

Report with charts: https://claude.ai/code/artifact/916ffba2-b74d-4cba-a69c-4c09d449d4a2

## Verdict

**Keep rp = 1.10.** It suppresses repetition loops ~2.4x better at every size,
and the margin grows as models get smaller — where the trajectory work is most
fragile. Its capability cost is real but ≤0.9pp and confined to two benchmarks.

## 1. Answer rate: rp=1.10 wins 13/15 cells, never loses

McNemar on paired items, pooled over each trajectory. Δ is `1.05 − 1.10`.

| size | medqa | medmcqa | pubmedqa |
|---|---|---|---|
| 270m | −4.16 | −4.18 | −13.37 |
| 1b   | −4.21 | −1.09 | −1.23 |
| 4b   | −1.79 | −1.37 | −0.09 (ns) |
| 12b  | −0.71 | −0.62 | 0.00 |
| 27b  | −0.53 | −0.37 | 0.00 |

All significant at p < 1e-5 except the three marked. The two zeros are exact ties
(PubMedQA saturates at 100% answer rate for both settings at 12b/27b).

## 2. The mechanism is loops, and only loops

`scripts/analyze_unanswered.py` over ~393k rows per arm:

| cause | rp=1.05 | rp=1.10 | change |
|---|---|---|---|
| loop | 9,703 | 4,026 | **−58%** |
| no_choice | 16,858 | 15,479 | −8% |
| placeholder | 1,043 | 845 | −19% |
| long_cot (real truncation) | 550 | 671 | **+22%** |
| judge_miss | 124 | 137 | +10% |
| **total unanswered** | **28,278 (7.19%)** | **21,158 (5.38%)** | **−25%** |

Real truncation is flat-to-worse at rp=1.10 — the penalty is not buying a token
budget, it is only killing loops. This confirms §8 of `BEHAVIORAL_EVAL.md` at
full scale and across all sizes.

Looping is a **fine-tuning artifact**: near zero at the start of every
trajectory, emerging around step 32 and persisting. On MedMCQA at the final
checkpoint, loop rate is 11.6% (270m) / 3.8% (1b) / 2.1% (4b) / 0.4% (12b) /
0.02% (27b) at rp=1.05, against 4.0 / 0.9 / 0.2 / 0.07 / 0.00 at rp=1.10.

At 12b and 27b, `no_choice` — finishing cleanly without committing — is now the
dominant unanswered cause at both settings (875 of 964 unanswered 27b MedMCQA
rows). Repetition penalty cannot touch that bucket.

## 3. The terminology tax is real, and ~1/6 the recorded size

Restricting to items **both** arms committed to removes the loop effect and
leaves only capability:

| bench | n both | acc 1.05 | acc 1.10 | Δ | p |
|---|---|---|---|---|---|
| medmcqa | 251,637 | 48.34 | 47.73 | **+0.61** | 2.1e-12 |
| pubmedqa | 28,904 | 64.56 | 63.66 | **+0.90** | 2.4e-05 |
| medqa | 78,193 | 52.03 | 51.82 | +0.21 | 0.22 (ns) |

**The 3.9pp figure in `CLAUDE.md` does not replicate.** That note records
"1.10 costs 3.9pp of acc_answered on MedQA," from a 300-item probe at 4b
step-512. On the full 1,273-item set at that exact checkpoint:

| 4b step-512 medqa | acc_raw | acc_answered | answer_rate |
|---|---|---|---|
| rp=1.05 | 48.63 | 52.37 | 92.85 |
| rp=1.10 | 51.45 | 52.57 | 97.88 |
| Δ | −2.82 | **−0.20** | −5.03 |

Direction reversed, magnitude 20x smaller; the per-item conditional test on 4b
MedQA gives −0.07pp at p=0.87. The tax exists on MedMCQA and PubMedQA, not on
MedQA. **`CLAUDE.md` should be corrected** so the `acc_answered` step at the
changeover is not later read against a number that was sampling noise.

## 4. Net acc_raw flips sign with size

rp=1.10 is worth up to **7.3pp** of raw accuracy at 270m (PubMedQA), ties at
1b/4b, and loses slightly at 12b/27b MedMCQA (+0.33 / +0.43 for rp=1.05, the
latter at p=0.019) — where there are no loops left to suppress and only the tax
shows. The untuned anchors agree: identical within a point at 1b+, dramatically
split at 270m (baseline PubMedQA answer rate 9.2% at rp=1.05 vs 21.8% at 1.10).

## Caveats

- **`270m_it_full_rp11/step-4096` has 1,150 of 1,273 MedQA judgments.** All
  numbers here are computed on the *intersection* of the two arms so they are
  unaffected, but that cell's own `judged_summary.json` is unreliable and should
  be re-judged before it enters a figure.
- Buckets use the default 0.6 unique-8-gram loop threshold. The direction of §2
  is not sensitive to it; the exact −58% is.
- **This says nothing about the lens.** The jlens fit and lens↔model concordance
  are teacher-forced — `probe_jlens.py` has no sampling parameters. One jlens
  trajectory per size serves both arms; no rp-specific lens exists or is needed.
- Both arms share a judge, so absolute accuracies inherit its bias; the deltas do
  not, since every pair is the same item scored by the same judge.

## Reproduce

```bash
# unanswered-cause buckets for both arms (runs in .venv-probe, ~4 min)
.venv-probe/bin/python scripts/analyze_unanswered.py --json /tmp/unans.json \
    eval/traj/*_rp105/step-* eval/traj/*_rp11/step-* eval/*-rp105 eval/*-rp11
```

The paired McNemar tests read `*_judgments.jsonl` from both arms and intersect on
`idx`; `correct` and `answered` are taken per row, never from the summaries.
