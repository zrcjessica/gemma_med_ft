"""Normalize each source to chat `messages` and build the training mixture.

Every loader below was written against the dataset's *actual* schema, read off
the hub with scripts/inspect_schemas.py. The schemas differ more than you'd
expect (MedQA nests options in a dict, MedExpQA keys them "1".."5" and stores
the correct option as a 1-indexed int, MedMCQA uses flat opa/opd + a 0-indexed
`cop`), so these are not interchangeable.

Decontamination is not optional here. MedReason and ReasonMed are *built from*
MedMCQA/MedQA items -- MedReason even carries the provenance in `dataset_name`
-- so blending them naively leaks eval questions into training and inflates the
headline numbers we are trying to compare against MedGemma.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata

from datasets import Dataset, concatenate_datasets, load_dataset

from .mixture import ALL_SOURCES, MixtureSpec, Source

log = logging.getLogger(__name__)

MCQ_INSTRUCTION = "Answer the following multiple-choice question."


def _norm_text(s: str) -> str:
    """Aggressive normalization for dedup/contamination matching only."""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def qhash(s: str) -> str:
    return hashlib.sha1(_norm_text(s).encode()).hexdigest()


def _mcq_prompt(question: str, options: dict[str, str]) -> str:
    lines = [f"{k}. {v}" for k, v in sorted(options.items())]
    return f"{MCQ_INSTRUCTION}\n\n{question.strip()}\n\n" + "\n".join(lines)


def _chat(prompt: str, response: str, source: str, question: str, **extra) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "source": source,
        # Hash of the *question* only -- what we dedup and decontaminate on.
        "qhash": qhash(question),
        **extra,
    }


# --- Tier 1: datasets named in MedGemma Table 1 --------------------------------


def load_medqa(src: Source) -> Dataset:
    ds = load_dataset(src.hf_id, split=src.split)

    def fn(r):
        opts = r["options"]
        letter = r["answer_idx"]
        return _chat(
            _mcq_prompt(r["question"], opts),
            f"{letter}. {opts[letter]}",
            src.key,
            r["question"],
        )

    return ds.map(fn, remove_columns=ds.column_names, desc="medqa")


def load_medmcqa(src: Source) -> Dataset:
    ds = load_dataset(src.hf_id, split=src.split)

    def fn(r):
        opts = {"A": r["opa"], "B": r["opb"], "C": r["opc"], "D": r["opd"]}
        letter = "ABCD"[r["cop"]]  # cop is 0-indexed
        # `exp` is a human-written explanation; use it as the rationale when
        # present, which is the closest public analogue to a teacher rationale.
        exp = (r.get("exp") or "").strip()
        answer = f"{letter}. {opts[letter]}"
        response = f"{exp}\n\nAnswer: {answer}" if exp else answer
        return _chat(_mcq_prompt(r["question"], opts), response, src.key, r["question"])

    ds = ds.filter(lambda r: r["cop"] is not None and 0 <= r["cop"] <= 3)
    return ds.map(fn, remove_columns=ds.column_names, desc="medmcqa")


PUBMEDQA_TEST_GT_URL = (
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json"
)
_pqa_test_ids: set[str] | None = None


def pubmedqa_test_ids() -> set[str]:
    """The official 500-question PQA-L test split, keyed by pubid.

    PubMedQA is both a training source (Table 1: 1,000) and a benchmark, and the
    hub only ships `pqa_labeled` as one 1,000-row `train` split. Without the
    official split we'd either train on our own test set or -- as the first run
    of this pipeline actually did -- decontaminate the source to zero. Published
    PubMedQA numbers all use this split, so inventing our own would make our
    results incomparable.
    """
    global _pqa_test_ids
    if _pqa_test_ids is None:
        import requests

        r = requests.get(PUBMEDQA_TEST_GT_URL, timeout=30)
        r.raise_for_status()
        _pqa_test_ids = set(r.json().keys())
        if len(_pqa_test_ids) != 500:
            raise ValueError(f"expected 500 PubMedQA test ids, got {len(_pqa_test_ids)}")
    return _pqa_test_ids


def load_pubmedqa(src: Source) -> Dataset:
    ds = load_dataset(src.hf_id, src.config, split=src.split)
    test_ids = pubmedqa_test_ids()
    ds = ds.filter(lambda r: str(r["pubid"]) not in test_ids, desc="pubmedqa:holdout")

    def fn(r):
        ctx = "\n".join(r["context"]["contexts"])
        prompt = (
            "Given the following abstract, answer the question with yes, no, or maybe.\n\n"
            f"{ctx}\n\nQuestion: {r['question'].strip()}"
        )
        response = f"{r['long_answer'].strip()}\n\nAnswer: {r['final_decision']}"
        return _chat(prompt, response, src.key, r["question"])

    return ds.map(fn, remove_columns=ds.column_names, desc="pubmedqa")


def load_medexpqa(src: Source) -> Dataset:
    ds = load_dataset(src.hf_id, src.config, split=src.split)

    def fn(r):
        opts_raw = {k: v for k, v in r["options"].items() if v}
        # Keys are "1".."5"; remap to letters for a consistent prompt format.
        keys = sorted(opts_raw, key=int)
        opts = {"ABCDE"[i]: opts_raw[k] for i, k in enumerate(keys)}
        letter = "ABCDE"[keys.index(str(r["correct_option"]))]
        rationale = (r.get("full_answer_no_ref") or "").strip()
        answer = f"{letter}. {opts[letter]}"
        response = f"{rationale}\n\nAnswer: {answer}" if rationale else answer
        return _chat(_mcq_prompt(r["full_question"], opts), response, src.key, r["full_question"])

    ds = ds.filter(lambda r: r["correct_option"] is not None and str(r["correct_option"]) in r["options"])
    return ds.map(fn, remove_columns=ds.column_names, desc="medexpqa")


def load_afrimedqa(src: Source) -> Dataset:
    # Gated on the hub: requires an approved access request + HF_TOKEN.
    ds = load_dataset(src.hf_id, split=src.split)
    cols = set(ds.column_names)
    q_col = next((c for c in ("question", "question_clean", "sample_question") if c in cols), None)
    a_col = next((c for c in ("answer", "answer_rationale", "correct_answer") if c in cols), None)
    if not q_col or not a_col:
        raise ValueError(f"afrimedqa: unexpected schema {sorted(cols)}")

    def fn(r):
        return _chat(str(r[q_col]).strip(), str(r[a_col]).strip(), src.key, str(r[q_col]))

    ds = ds.filter(lambda r: r[q_col] and r[a_col])
    return ds.map(fn, remove_columns=ds.column_names, desc="afrimedqa")


def load_liveqa(src: Source) -> Dataset:
    ds = load_dataset(src.hf_id, split=src.split)

    def fn(r):
        return _chat(r["message"].strip(), r["answer"].strip(), src.key, r["message"])

    ds = ds.filter(lambda r: r["message"] and r["answer"])
    return ds.map(fn, remove_columns=ds.column_names, desc="liveqa")


# --- Synthetic-set proxies ----------------------------------------------------


def load_medical_o1(src: Source) -> Dataset:
    ds = load_dataset(src.hf_id, src.config, split=src.split)

    def fn(r):
        cot = (r.get("Complex_CoT") or "").strip()
        resp = r["Response"].strip()
        return _chat(r["Question"].strip(), f"{cot}\n\n{resp}" if cot else resp, src.key, r["Question"])

    return ds.map(fn, remove_columns=ds.column_names, desc="medical_o1")


def load_medreason(src: Source) -> Dataset:
    ds = load_dataset(src.hf_id, split=src.split)

    def fn(r):
        q = r["question"].strip()
        opts = (r.get("options") or "").strip()
        prompt = f"{MCQ_INSTRUCTION}\n\n{q}\n\n{opts}" if opts else q
        reasoning = (r.get("reasoning") or "").strip()
        answer = r["answer"].strip()
        response = f"{reasoning}\n\nAnswer: {answer}" if reasoning else answer
        # Keep provenance: these rows are derived from medqa/medmcqa and must be
        # decontaminated against those eval splits.
        return _chat(prompt, response, src.key, q, provenance=r.get("dataset_name") or "")

    return ds.map(fn, remove_columns=ds.column_names, desc="medreason")


def load_reasonmed(src: Source) -> Dataset:
    ds = load_dataset(src.hf_id, split=src.split)

    def fn(r):
        prompt = r["instruction"].strip()
        if (r.get("input") or "").strip():
            prompt = f"{prompt}\n\n{r['input'].strip()}"
        return _chat(prompt, r["output"].strip(), src.key, prompt)

    return ds.map(fn, remove_columns=ds.column_names, desc="reasonmed")


# --- General-instruction replay ------------------------------------------------


def load_aloe_general(src: Source) -> Dataset:
    ds = load_dataset(src.hf_id, split=src.split)
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}

    def fn(r):
        msgs = []
        for turn in r["conversations"]:
            role = role_map.get(turn["from"])
            if role is None:
                continue
            msgs.append({"role": role, "content": turn["value"]})
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        return {"messages": msgs, "source": src.key, "qhash": qhash(first_user)}

    def usable(r):
        roles = [role_map.get(t["from"]) for t in r["conversations"]]
        # Gemma's template has no standalone system turn and requires strict
        # user/model alternation; drop anything that won't render.
        return roles[:1] == ["user"] and "assistant" in roles

    ds = ds.filter(usable, desc="aloe_general:filter")
    return ds.map(fn, remove_columns=ds.column_names, desc="aloe_general")


LOADERS = {
    "medqa": load_medqa,
    "medmcqa": load_medmcqa,
    "pubmedqa": load_pubmedqa,
    "medexpqa": load_medexpqa,
    "afrimedqa": load_afrimedqa,
    "liveqa": load_liveqa,
    "medical_o1": load_medical_o1,
    "medreason": load_medreason,
    "reasonmed": load_reasonmed,
    "aloe_general": load_aloe_general,
}


# --- Contamination ------------------------------------------------------------

# Splits we report on. Anything whose question matches one of these is dropped
# from training.
# PubMedQA is deliberately absent: it is held out by pubid in load_pubmedqa()
# against the official 500-question test split. Blocking it by question hash
# here would also nuke the 500 rows we are supposed to train on.
EVAL_SPLITS: list[tuple[str, str | None, str]] = [
    ("GBaker/MedQA-USMLE-4-options", None, "test"),
    ("openlifescienceai/medmcqa", None, "validation"),  # MedMCQA test is unlabeled
]


def eval_question_hashes() -> set[str]:
    hashes: set[str] = set()
    for hf_id, cfg, split in EVAL_SPLITS:
        ds = load_dataset(hf_id, cfg, split=split)
        col = "question"
        hashes.update(qhash(q) for q in ds[col])
        log.info("decontam: +%d hashes from %s[%s]", len(ds), hf_id, split)
    return hashes


def build_mixture(spec: MixtureSpec, decontaminate: bool = True) -> Dataset:
    blocked = eval_question_hashes() if decontaminate else set()
    log.info("decontam: %d blocked eval question hashes", len(blocked))

    parts, stats = [], []
    seen: set[str] = set()

    for key in spec.sources:
        src = ALL_SOURCES[key]
        if src.role == "eval_only":
            raise ValueError(f"{key} is eval-only (no reference answers) and cannot be trained on")

        ds = LOADERS[key](src)
        n0 = len(ds)

        if decontaminate:
            ds = ds.filter(lambda r: r["qhash"] not in blocked, desc=f"{key}:decontam")
        n1 = len(ds)

        # Cross-source dedup: earlier sources win, so Table 1 data outranks the
        # reasoning corpora derived from it. Done by explicit index selection --
        # a stateful predicate inside .filter() is not safe under the datasets
        # cache, which may reuse or reorder execution.
        keep = []
        for i, h in enumerate(ds["qhash"]):
            if h not in seen:
                seen.add(h)
                keep.append(i)
        ds = ds.select(keep)
        n2 = len(ds)

        cap = spec.cap.get(key)
        if cap and n2 > cap:
            ds = ds.shuffle(seed=spec.seed).select(range(cap))

        stats.append((key, n0, n0 - n1, n1 - n2, len(ds), src.paper_n))
        parts.append(ds.select_columns(["messages", "source", "qhash"]))

    print(f"\n{'source':<14} {'raw':>9} {'-contam':>9} {'-dupe':>8} {'final':>9} {'paper':>9}")
    print("-" * 64)
    for key, n0, ncon, ndup, nf, pn in stats:
        print(f"{key:<14} {n0:>9,} {ncon:>9,} {ndup:>8,} {nf:>9,} {str(pn or '-'):>9}")
    total = sum(s[4] for s in stats)
    print("-" * 64)
    print(f"{'TOTAL':<14} {'':>9} {'':>9} {'':>8} {total:>9,}")

    return concatenate_datasets(parts).shuffle(seed=spec.seed)
