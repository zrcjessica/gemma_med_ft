"""Print the real schema + a sample row for each source in the mixture.

Guessing these schemas is how loaders silently produce garbage, so we read them
off the hub before writing any normalization code.
"""

import sys
import traceback

from datasets import get_dataset_config_names, load_dataset

sys.path.insert(0, ".")
from gemma_med.mixture import ALL_SOURCES  # noqa: E402


def probe(src):
    print(f"\n{'=' * 70}\n{src.key}  <-  {src.hf_id}  (config={src.config}, role={src.role})")
    try:
        configs = get_dataset_config_names(src.hf_id)
        print(f"  configs: {configs[:10]}")
    except Exception as e:
        print(f"  configs: <{type(e).__name__}: {e}>")
        configs = []

    cfg = src.config
    if cfg is None and configs and configs != ["default"]:
        cfg = configs[0]
        print(f"  (no config pinned; probing '{cfg}')")

    try:
        ds = load_dataset(src.hf_id, cfg, split=src.split, streaming=True)
        row = next(iter(ds))
    except Exception:
        print("  LOAD FAILED:")
        traceback.print_exc(limit=1)
        return

    print(f"  columns: {list(row.keys())}")
    for k, v in row.items():
        s = repr(v)
        print(f"    - {k}: {s[:200]}{' ...' if len(s) > 200 else ''}")


if __name__ == "__main__":
    keys = sys.argv[1:] or list(ALL_SOURCES)
    for k in keys:
        probe(ALL_SOURCES[k])
