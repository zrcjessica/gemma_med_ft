# Fine-tuning data

Every text corpus the Gemma 3 arms are SFT'd on in this repo. Defined in
`gemma_med/mixture.py` (registry + named mixtures), normalized to chat
`messages` by `gemma_med/data.py`, materialized to disk by
`scripts/build_data.py`.

`data/` is gitignored, so mixtures are built on a login node (needs internet)
and saved to `data/mix_<name>/{train,val,mixture_meta.json}`; training then runs
with `HF_HUB_OFFLINE=1`.

Text modality only — vision (MedSigLIP) is deferred, so nothing here is
image-conditioned.

---

## 1. Sources

### Tier 1 — named in MedGemma Table 1 (`role="tier1"`)

| key | HF id | config / split | Table 1 N | license |
|---|---|---|---|---|
| `medqa` | `GBaker/MedQA-USMLE-4-options` | — / train | 9,275 | CC-BY-4.0 |
| `medmcqa` | `openlifescienceai/medmcqa` | — / train | 182,806 | Apache-2.0 |
| `pubmedqa` | `qiaojin/PubMedQA` | `pqa_labeled` / train | 1,000 | MIT |
| `medexpqa` | `HiTZ/MedExpQA` | `en` / train | 434 | CC-BY-4.0 |
| `afrimedqa` | `intronhealth/afrimedqa_v2` | — / train | 1,003 | CC-BY-SA-4.0 |
| `liveqa` | `truehealth/liveqa` | — / train | 634 | unstated |

`afrimedqa` is **gated on the hub** despite its license. `build_mixture()` warns
and continues without it rather than failing the build, so every mixture built
so far is missing it (recorded as `sources_missing` in `mixture_meta.json`). It
is ~1k of ~305k rows. To include it: request access at
`huggingface.co/datasets/intronhealth/afrimedqa_v2` and set `HF_TOKEN`.

Paper-vs-ours discrepancies on tier 1 (all explained in `docs/RECIPE.md`):

- **MedQA 9,275 vs our 10,176** — Google filtered ~9% of the public train split
  by an unstated method; we can't match it.
- **PubMedQA 1,000 vs our 500** — the hub ships `pqa_labeled` as one 1,000-row
  split, of which the official benchmark test set is 500. We hold those out by
  `pubid`. Table 1's "1,000" implies MedGemma trained on its own test split.
- **LiveQA 634 vs our 420** — the paper counts question/answer *pairs*; our
  dedup keeps one row per question.

### Synthetic proxies (`role="synthetic_proxy"`)

MedGemma's 200k synthetic question set was never released, and its distillation
used teacher *logits* that no public API exposes. We substitute response-level
SFT on public chain-of-thought corpora. **This is the largest fidelity gap in
the project.**

| key | HF id | what it is |
|---|---|---|
| `medical_o1` | `FreedomIntelligence/medical-o1-reasoning-SFT` (`en`) | GPT-4o CoT, verifier-checked (HuatuoGPT-o1). The `en` config is 19,704 rows — *not* the ~90k often quoted, which sums en/zh/en_mix/zh_mix |
| `medreason` | `UCSC-VLAA/MedReason` | ~33k knowledge-graph-grounded reasoning chains; different failure mode from pure CoT |
| `reasonmed` | `lingshu-medical-mllm/ReasonMed` | 1,111,555 rows, the largest available; subsampled via `MixtureSpec.cap` |

Both `medreason` and `reasonmed` are **derived from MedQA/MedMCQA** — MedReason
even carries provenance in a `dataset_name` column — which is why
decontamination and cross-source dedup are mandatory, not optional.

### General-instruction replay (`role="replay"`)

| key | HF id | why |
|---|---|---|
| `aloe_general` | `HPAI-BSC/Aloe-Beta-General-Collection` | 316,741 raw multi-turn rows. Counterweight to the alignment tax the report measures (MMLU Pro 43.6 → 39.1 at 4B, 67.5 → 60.2 at 27B) |

### Eval-only — never trained on (`role="eval_only"`)

| key | HF id | why excluded |
|---|---|---|
| `healthsearchqa` | `katielink/healthsearchqa` | Table 1 lists 3,375, but every public release is *questions only, no reference answers* → unusable for SFT. `build_mixture()` raises if you include it. Public row counts also conflict (3,173 vs ~4,576) |

---

## 2. Named mixtures

| name | sources | caps | purpose |
|---|---|---|---|
| `default` | all 6 tier-1 + `medical_o1` + `medreason` + `reasonmed` + `aloe_general` | `reasonmed`: 80,000 · `aloe_general`: 40,000 | **The one the real trajectories use.** Sized to mirror the paper's ~1:1 real-QA-to-synthetic ratio, plus ~10% replay |
| `paper_only` | the 6 tier-1 sources | — | Closest literal match to Table 1, minus the synthetic set. Fidelity reference point, and the natural ablation against `default` for the replay question |
| `small` | `medqa`, `pubmedqa`, `medexpqa`, `medical_o1` | `medical_o1`: 5,000 | 1b LR sweep and pipeline smoke tests |

Seed 42 throughout (shuffle, cap subsampling, train/val split), so all arms see
the same data in the same order.

---

## 3. What `default` actually produced

**312,335 train / 3,155 val** (1% val fraction), after decontamination,
cross-source dedup, and capping:

```
source               raw   -contam    -dupe     final     paper
medqa             10,178         0        2    10,176      9275
medmcqa          182,822        13   30,838   151,971    182806
pubmedqa             500         0        0       500      1000
medexpqa             434         0       10       424       434
liveqa               633         0      213       420       634
medical_o1        19,704         0       36    19,668         -
medreason         32,682         2   20,349    12,331         -
reasonmed      1,111,555         0  940,536    80,000         -
aloe_general     316,741         0   46,070    40,000         -
afrimedqa            SKIPPED (gated — no access)
```

The `final` column sums to 315,490, which is exactly the 312,335 train + 3,155
val split. Composition: ~48.2% MedMCQA, ~25.4% ReasonMed,
~12.7% Aloe general replay, ~6.2% medical-o1, ~3.9% MedReason, ~3.2% MedQA, and
~0.4% combined from PubMedQA/MedExpQA/LiveQA.

Three measured facts worth carrying around:

- **ReasonMed is ~85% redundant.** 940,536 of 1,111,555 rows duplicate questions
  already in MedQA/MedMCQA/MedReason; only ~171k are novel (we cap at 80k).
  Treating its 1.1M headline as 1.1M of new signal would be a mistake.
- **MedReason loses 62%** to the same effect — it is largely MedMCQA re-reasoned.
- **MedMCQA has ~31k internal duplicate questions** (~17% of its train split).
  That's the dataset's own redundancy, not eval leakage.

Eval contamination itself was small — 13 rows in MedMCQA, 2 in MedReason — so
the decontamination pass mostly buys deduplication. But the leakage was real and
non-zero, and you can't know that without running the check.

Both finished trajectory runs (`270m_it_full`, `1b_it_full_v3`) end at step
4,882, consistent with this build at effective batch 128 × 2 epochs.

---

## 4. Filtering pipeline (`gemma_med/data.py:build_mixture`)

Applied in this order, per source, in `spec.sources` order:

1. **Load + normalize** to chat `messages` via the source's loader. Each loader
   was written against the dataset's actual hub schema (read off with
   `scripts/inspect_schemas.py`) — they differ more than you'd expect and are
   not interchangeable: MedQA nests options in a dict, MedExpQA keys them
   `"1".."5"` with a 1-indexed `correct_option`, MedMCQA uses flat `opa`..`opd`
   with a 0-indexed `cop`.
2. **PubMedQA holdout** (inside its loader, not the generic pass): the official
   500-question PQA-L test set is fetched by `pubid` from the pubmedqa GitHub
   ground-truth file and dropped. It is deliberately absent from `EVAL_SPLITS`
   below — blocking it by question hash would also nuke the 500 rows we're
   supposed to train on.
3. **Decontamination** against the splits we report on, by normalized question
   hash (NFKC → lowercase → strip non-alphanumerics → SHA-1):
   - `GBaker/MedQA-USMLE-4-options[test]`
   - `openlifescienceai/medmcqa[validation]` (MedMCQA's test split is unlabeled)

   `--no-decontaminate` exists for ablations only; results from it are inflated
   by construction.
4. **Cross-source dedup** on the same question hash, **earlier sources win** — so
   tier-1 data outranks the reasoning corpora derived from it. Done by explicit
   index selection, not a stateful `.filter()` predicate, which isn't safe under
   the `datasets` cache.
5. **Cap** (`spec.cap`), applied after dedup, via `shuffle(seed).select(range(cap))`.
6. **Concatenate + shuffle** (seed 42), then a 1% train/val split.

`mixture_meta.json` records the mixture name, sources, caps, decontamination
flag, seed, row counts, per-source counts, and `sources_missing` — the record of
what the mixture *actually is*, not what it was meant to be.

---

## 5. Format the model sees

Every source is normalized to a single-turn (or multi-turn, for replay) chat
record with `messages`, `source`, and `qhash` columns.

- **MCQ sources** (`medqa`, `medmcqa`, `medexpqa`, `medreason`) get a fixed
  prefix `"Answer the following multiple-choice question."` followed by the
  question and lettered options. Target is `"<letter>. <option text>"`.
- **Rationale prepending**, where one exists in the source, giving
  `"<rationale>\n\nAnswer: <answer>"` — the closest public analogue to a teacher
  rationale:
  - `medmcqa` → its human-written `exp`
  - `medexpqa` → `full_answer_no_ref`
  - `medical_o1` → `Complex_CoT`
  - `medreason` → `reasoning`
  - `pubmedqa` → `long_answer`
- **PubMedQA** is abstract-conditioned: the contexts are concatenated into the
  prompt, and the target is `"<long_answer>\n\nAnswer: <yes|no|maybe>"`.
- **`liveqa`** is free-text consumer health Q&A, used as-is.
- **`aloe_general`** is multi-turn, remapped `human/gpt/system` →
  `user/assistant/system`, then filtered to what Gemma's template can render:
  must start on a user turn, must contain an assistant turn, no standalone
  system turn (Gemma's template requires strict user/model alternation).

Training config that touches data (`gemma_med/train.py`, `scripts/train_med.sbatch`):

- `--max-seq-len 2048`
- `--epochs 2`, LR 1e-5 cosine, effective batch ~128
- `assistant_only_loss=True` — the report's "cross-entropy on completion only".
  Requires the `{% generation %}` chat-template variant during training
  (`gemma_med/chat.py:ensure_chat_template(for_training=True)`); the stock
  template is restored before saving so inference/vLLM get the official one.
- Fixed seed and fixed data order across every arm, so the pt/it and size
  trajectories are on the same clock and comparable.

---

## 6. Not fine-tuning data

Also under `data/`, but these are probe/eval inputs — the model is never trained
on them.

**Jacobian-lens inputs** (`data/jlens/`, versioned in git via a scoped
`.gitignore` exception):

| file | what |
|---|---|
| `fit_corpus_v2.txt` | 1,000 WikiText-103 sequences (`Salesforce/wikitext`, `wikitext-103-raw-v1`, train) via `jlens.examples.load_wikitext_prompts`. The frozen **generic** corpus the lens is refit on per checkpoint — deliberately not medical; "fit on medical" is a later ablation |
| `probe_set_v2.tsv` | 193 frozen concordance prompts: 50 general + 50 medical hand-authored style-matched one-hop cloze pairs, plus 93 multihop items from jlens's `lens-eval-multihop.json` |
| `fit_corpus.txt`, `probe_set.tsv` | v1 (40 / 30 rows), Phase-0 scale |

**Behavioral eval** (`gemma_med/evaluate.py`) — all excluded from training by the
decontamination and holdout above:

| benchmark | split |
|---|---|
| MedQA | `GBaker/MedQA-USMLE-4-options[test]` |
| MedMCQA | `openlifescienceai/medmcqa[validation]` |
| PubMedQA | the held-out official 500 from `pqa_labeled` |

---

## 7. Rebuilding

```bash
# On a login node (needs internet)
python scripts/build_data.py --mixture default --out data/mix_default

# Then train offline against it
sbatch --gres=gpu:a100:2 --export=ALL,SIZE=1b,KIND=it,LR=1e-5 scripts/train_med.sbatch
```

`DATA_DIR` defaults to `data/mix_default`; override it to point at
`mix_paper_only` / `mix_small`.

See `docs/RECIPE.md` for what MedGemma actually did, what we can't reproduce,
and the target numbers.
