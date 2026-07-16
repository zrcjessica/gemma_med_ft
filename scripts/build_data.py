"""Build a mixture and save it to disk for training.

Run on a login node (has internet) before submitting training jobs, so the
training job itself can run with HF_HUB_OFFLINE=1 and not depend on the hub.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemma_med.data import build_mixture  # noqa: E402
from gemma_med.mixture import MIXTURES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixture", default="default", choices=list(MIXTURES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--no-decontaminate", action="store_true",
                    help="Disable eval-set filtering. Only for ablations -- results will be inflated.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    spec = MIXTURES[args.mixture]
    ds = build_mixture(spec, decontaminate=not args.no_decontaminate)

    split = ds.train_test_split(test_size=args.val_frac, seed=args.seed)
    out = Path(args.out)
    split["train"].save_to_disk(out / "train")
    split["test"].save_to_disk(out / "val")

    meta = {
        "mixture": args.mixture,
        "sources": spec.sources,
        "cap": spec.cap,
        "decontaminated": not args.no_decontaminate,
        "seed": args.seed,
        "n_train": len(split["train"]),
        "n_val": len(split["test"]),
        "counts_by_source": {
            s: sum(1 for x in split["train"]["source"] if x == s) for s in set(split["train"]["source"])
        },
        # Requested but absent (e.g. gated) -- the record of what this mixture
        # actually is, not what it was meant to be.
        "sources_missing": sorted(set(spec.sources) - set(split["train"]["source"])),
    }
    (out / "mixture_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {len(split['train']):,} train / {len(split['test']):,} val -> {out}")
    print(json.dumps(meta["counts_by_source"], indent=2))


if __name__ == "__main__":
    main()
