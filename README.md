# gemma-med-ft

Fine-tune Gemma 3 (1b/4b/12b/27b) on a public approximation of MedGemma's
medical text data, on BigPurple via Slurm.

**Read [`docs/RECIPE.md`](docs/RECIPE.md) first.** It documents what MedGemma
actually did (per [arXiv:2507.05201](https://arxiv.org/abs/2507.05201)), what we
reproduce, and — importantly — the three things we cannot.

The short version: MedGemma-27B-text is Gemma 3 plus **post-training only** ("the
text-only version of MedGemma 27B leveraged the post-training stage alone"), and
its medical text data is almost entirely public. So an SFT reproduction is
faithful in kind. What we can't match is the unreleased teacher's *logits*, its
200k synthetic questions, and HealthSearchQA's answers.

## Layout

```
gemma_med/
  chat.py       Gemma 3 chat template (+ {% generation %} variant for TRL masking)
  mixture.py    Which datasets, how many rows, and why — keyed to the report's Table 1
  data.py       Loaders (written against real hub schemas), dedup, decontamination
  train.py      SFT entry point (full FT or LoRA)
  evaluate.py   MedQA / MedMCQA / PubMedQA via vLLM
scripts/
  build_env.sh        training venv
  build_eval_env.sh   separate vLLM venv (vLLM pins torch; can't share)
  fetch_models.sh     pull gemma-3-*-it
  build_data.py       materialize a mixture to disk
  train_med.sbatch    Slurm entry point
  inspect_schemas.py  print real dataset schemas (run before touching a loader)
tests/          chat-template and answer-parsing tests
```

## Setup (BigPurple)

```bash
cd /gpfs/data/oermannlab/users/zhouj14/gemma_med_ft
srun -p a100_dev --gres=gpu:a100:1 -c 16 --mem=96G -t 3:00:00 bash scripts/build_env.sh
sbatch -p cpu_short -c 8 --mem=48G -t 4:00:00 --wrap "bash scripts/build_eval_env.sh"
```

**No model download needed.** Every Gemma 3 size — 270m / 1b / 4b / 12b / 27b, in
both `-pt` and `-it` — is already cached and world-readable under the lab's shared
hub at `/gpfs/data/oermannlab/users/yeb04/hf/hub`. `scripts/_resolve_model.sh`
finds them and logs the revision each run used. `scripts/fetch_models.sh` exists
only for pinning your own copy.

Set `KIND=pt` to train from the pretrained bases instead of `-it` (see
RECIPE.md for why `-it` is the default).

## Build the data

Run on a **login node** — it needs internet. Training jobs then run with
`HF_HUB_OFFLINE=1`.

```bash
source .venv/bin/activate
python scripts/build_data.py --mixture default --out data/mix_default
```

Mixtures (`gemma_med/mixture.py`): `small` (smoke tests / LR sweep),
`paper_only` (Table 1 text sources only), `default` (Table 1 + synthetic proxy +
general replay).

## Train

```bash
sbatch -p a100_short --gres=gpu:a100:2 --export=ALL,SIZE=1b,LR=1e-5 scripts/train_med.sbatch
sbatch -p a100_short --gres=gpu:a100:4 --export=ALL,SIZE=4b,LR=1e-5 scripts/train_med.sbatch
sbatch -p oermannlab --gres=gpu:a100:8 --export=ALL,SIZE=27b,LORA=1 scripts/train_med.sbatch
```

Full fine-tuning at 1b/4b; LoRA defaults on at 12b/27b. Slack notifications fire
on success and failure via `$SLACK_WEBHOOK_URL`; runs log to W&B project
`gemma-med-ft`.

Attention defaults to `eager` — transformers strongly recommends it over
`flash_attention_2` for Gemma 3 training. Pass `ATTN=flash_attention_2` to trade
that for speed once you've confirmed the loss curves agree.

**Partition note.** `oermannlab`'s 24 A100s are frequently fully allocated (we've
seen estimated starts a day out); `a100_short` usually has capacity. Some
`a100_short` nodes hand out a GPU that reports itself free but rejects every
allocation — the job preflights for this and requeues onto another node, up to 5
times, so you generally don't need to care.

### LR sweep

The report publishes no LR for this stage (see RECIPE.md), so pick one:

```bash
bash scripts/sweep_lr.sh data/mix_default    # 5e-6 / 1e-5 / 2e-5 at 1b
```

## Evaluate

Always evaluate the base `gemma-3-*-it` too — that's the number to beat, and
MedGemma's published score is the ceiling (see RECIPE.md).

```bash
source .venv-eval/bin/activate
python -m gemma_med.evaluate --model-path outputs/1b/full_lr1e-5_*/final \
    --out eval/1b_med --label gemma3-1b-med
```

Check `unparsed_pct` in the summary: a high value means the score reflects format
compliance, not medical knowledge.

## Tests

```bash
GEMMA_TOKENIZER=$HF_HOME/hub/models--google--gemma-3-1b-it/snapshots/*/ \
  python -m pytest tests/ -q
```
