# Workspace-band probe (`band.py` / `probe_band.py`)

Recreates the lab's `medlens/band.py` analysis — the jlens paper's Fig. 27/28
"workspace band" location — over our medical-SFT trajectories, so the band can be
read as a function of **size × training step** rather than for one static model.

Source: `/gpfs/data/oermannlab/users/yeb04/jlens/code/medlens/medlens/band.py`
(rev of 2026-08-06), driven there by `medlens/runm2.py` for a single model.

---

## What it measures

Per fitted layer, on a lens that has been refit for that checkpoint:

| metric | paper | what moves it |
|---|---|---|
| CKA among J-lens vectors | Fig. 27 | block structure: sensory / workspace / motor |
| next-token prediction accuracy | Fig. 28a | low in the band, ignites in the motor layers |
| excess kurtosis of the readout | Fig. 28b | ~0 early, rises once readouts carry content |
| top-1 autocorrelation vs shuffled null | Fig. 28c | high where J-space content persists across positions |
| J-space dimensionality | Fig. 28d | weights-only, like the CKA |

`locate_band` folds kurtosis + autocorrelation + CKA into a suggested contiguous
`[lo, hi]`. **It is a suggestion generator, not an answer** — the upstream
docstring says to confirm against the diagnostic plot, and we record the raw
curves alongside the band so that confirmation is possible after the fact.

## Two deliberate deltas from the vendored source

1. `effective_gamma` is **inlined** rather than imported from
   `medlens.jspace_patch`, so `scripts/band.py` has no medlens dependency and
   runs in our `.venv-jlens` (jlens @ `581d398`) unmodified. Gemma 3 is the
   `1 + weight` RMSNorm case, so reading `.weight` directly would understate the
   final-norm gain on this whole model line — that is what `effective_gamma`
   exists to fix, and it is why the function had to come along.
2. The CKA/dimensionality token draw is **shared** between the two consumers
   instead of being rebuilt twice. Building it decodes all 262k Gemma 3 vocab ids
   one at a time.

Everything else is kept byte-comparable to the source on purpose: it is somebody
else's validated method, and a divergence here is a divergence in the science.

The text-statistics pass reads generic text from our frozen
`data/jlens/fit_corpus_v2.txt` rather than medlens's wikitext sample. Same intent
(the distribution the lens was fitted on), but it keeps the band probe on the
same frozen instrument as the rest of the trajectory work.

---

## The cost model, and why it is what it is

**The lens fit dominates; the band metrics are noise next to it.** Measured in
`fanout_jlens.sh`: ~3.5 h/ckpt at 4b on an H100, ~15 h at 12b, ~47 h at 27b
(cost ~ `d_model × params`). The band pass on an *existing* lens is minutes —
tens of minutes at 27b, where the CKA is `n_layers²/2` matmuls at `d=5376`.

**The per-checkpoint lenses from the original j-lens fanout do not exist.**
`probe_jlens.py` has `--save-lenses` (`:314`), but `probe_jlens.sbatch:107` only
passes it when `SAVE_LENSES` is set in the environment, and neither
`fanout_jlens.sh:68` nor `fanout_jlens_4b.sh:41` includes it in `--export`. Every
worker fitted a lens, wrote its `metrics.jsonl` row, and dropped it. Searched
`/gpfs/data/oermannlab/users/zhouj14`, `/gpfs/scratch/zhouj14`,
`/gpfs/home/zhouj14` — the only survivors are three t=0 base lenses
(`jlens_out/{4b,12b,27b}_it/.base_lens.pt`), saved by a different code path
(`probe_jlens.py:287`, unconditional at step 0, preserved by `--keep-base-lens`).
The resumable `.fit_ckpt/step*.pt` files are no help; they are unlinked as soon
as their row is durable.

So **any step axis requires refitting**, and that is why every band worker runs
`--save-lens`. A saved lens is `n_layers × d_model² × 2` bytes at fp16:

| arm | per ckpt | × 14 ckpts |
|---|---|---|
| 270m | 15 MB | 0.2 GB |
| 1b | 69 MB | 1.0 GB |
| 4b | 446 MB | 6.2 GB |
| 12b | 1.4 GB | 20 GB |
| 27b | 3.6 GB | 50 GB |
| | | **~77 GB** |

Small against the 1.5 TB the checkpoints already occupy, and *very* small against
the ~860 GPU-h the full grid costs. Watch the shared `data_oermannlab` fileset
quota, not your own — "Disk quota exceeded" there is usually not you.

**Reusing the three surviving t=0 lenses is safe.** They were fitted on
`fit_corpus_v2.txt`: the base-only run logged `fit_corpus=1000 probes=193`, and
v2 is 1000/193 lines while v1 is 40/30. That is the same corpus every refit uses,
so t=0 and the later steps stay on one instrument. **Never mix corpora on one
trajectory** — the lens is a function of the weights *and* the fit distribution.

---

## Running it

```bash
# The whole grid: 5 sizes x every checkpoint, plus t=0. Submits a BASE_ONLY
# probe per arm and a dependent (afterok) fanout of one job per checkpoint.
scripts/run_band_grid.sh
DRY_RUN=1 scripts/run_band_grid.sh          # print what would be submitted
ARMS="270m 1b" scripts/run_band_grid.sh     # a subset

# One arm by hand, in two stages (the fanout must be seeded from the t=0 row):
sbatch --export=ALL,SIZE=4b,BASE_ONLY=1,OUT=$BAND_OUT/4b_it,SAVE_LENS=1,\
LENS=$JLENS_OUT/4b_it/.base_lens.pt scripts/probe_band.sbatch
SIZE=4b RUN_DIR=outputs/4b/full_lr1e-5_25911616 scripts/fanout_band.sh

# Merge, then plot
python scripts/merge_band_fanout.py --fanout-dir $BAND_OUT/4b_it_fanout \
    --expect-run-dir outputs/4b/full_lr1e-5_25911616 \
    --out eval/band/4b_it/band.jsonl
python scripts/plot_band.py --arm 270m --arm 1b --arm 4b --arm 12b --arm 27b \
    eval/band/{270m,1b,4b,12b,27b}_it/band.jsonl --out figures/band
```

The `afterok` dependency is the safety net: if a base probe fails, its fanout is
killed rather than launching 13 GPU-hogging workers against a broken code path.

### Gotchas that will bite

- **4b+ must use the pre-converted text base** (`data/text_bases/<size>_it`).
  transformers 5.14.1 does *not* remap the multimodal base's nested keys onto
  `Gemma3ForCausalLM` — it silently random-initializes, which would make t=0
  meaningless. `probe_band.py` reuses `probe_jlens.load_hf`, which treats any
  missing weight as fatal, and `fanout_band.sh` sets `BASE_MODEL` for 4b+.
- **27b needs `DIM_BATCH=8`** or the fit OOMs on top of 55 GB of weights, and
  `CKA_TOKENS=4096` to keep the CKA in tens of minutes.
- **No V100 nodes** (`gpu4_*`): compute capability 7.0 has no bf16.
- Jacobians are moved on-device before the text pass. Skipping that makes
  `jlens.lens.readout` transfer 115 MB × 62 layers per prompt at 27b, turning a
  minutes-long pass into an hours-long one. Same numbers, wildly different clock.

---

## Reading the figures

`scripts/plot_band.py` writes four:

| file | question it answers |
|---|---|
| `band_ribbon.png` | where does the band sit, and does it move over the trajectory? |
| `band_edges.png` | onset / offset / width, all arms on one axis |
| `band_profiles.png` | the depth profiles behind the band, metric × arm |
| `band_delta.png` | change from that arm's own t=0 — "did SFT move it, and where" |

**Everything is plotted against relative depth (`layer / n_layers`), never layer
index.** Arms differ in depth (18 layers at 270m, 62 at 27b), so a raw index
means a different thing in every panel. Same rule the j-lens work already
follows: compare derived scalars across sizes, never raw `J_l`.

Size keeps the validated ordinal blue ramp it wears everywhere else in the deck
(`plot_arms.ARM_COLOR`) — categorical slots 1–3 already mean
general/medical/drift. Step is a *second* sequential context, so per the one-hue
rule it takes t=0 in slot-2 orange and later steps on a grey ramp; blue keeps
meaning size. The delta panels are diverging (blue↔red, neutral grey midpoint,
symmetric limits shared across a row) because their zero is meaningful.

t=0 is drawn in every figure. A finetuned-only panel cannot show that something
moved.
