"""Integration tests for the dataset loaders. Require network:

    RUN_DATA_TESTS=1 python -m pytest tests/test_loaders.py -q

The risk these cover: every source indexes its options differently (MedMCQA's
`cop` is 0-indexed, MedExpQA's `correct_option` is 1-indexed over string keys,
MedQA ships an explicit letter). An off-by-one here trains the model on wrong
answers and never raises -- you'd only see it as a mediocre eval score.

So each test pins the gold letter back to the gold *answer text* from the raw
row, which is independent of our indexing arithmetic.
"""

import os
import re

import pytest
from datasets import load_dataset

from gemma_med.data import load_medexpqa, load_medmcqa, load_medqa, load_pubmedqa
from gemma_med.mixture import ALL_SOURCES

pytestmark = pytest.mark.skipif(not os.environ.get("RUN_DATA_TESTS"), reason="set RUN_DATA_TESTS=1")


def _answer_letter(response: str) -> str:
    m = re.search(r"(?:Answer:\s*)?([A-E])\.", response.strip().splitlines()[-1])
    assert m, f"no answer letter in: {response[-200:]!r}"
    return m.group(1)


def test_medqa_letter_matches_gold_text():
    raw = load_dataset("GBaker/MedQA-USMLE-4-options", split="train[:20]")
    ds = load_medqa(ALL_SOURCES["medqa"])
    for i in range(10):
        letter = _answer_letter(ds[i]["messages"][1]["content"])
        assert letter == raw[i]["answer_idx"]
        # The letter must name the option holding the gold answer text.
        assert raw[i]["options"][letter] == raw[i]["answer"]


def test_medmcqa_cop_is_zero_indexed():
    raw = load_dataset("openlifescienceai/medmcqa", split="train[:50]")
    ds = load_medmcqa(ALL_SOURCES["medmcqa"])
    checked = 0
    for i in range(20):
        row = raw[i]
        if row["cop"] is None or not 0 <= row["cop"] <= 3:
            continue
        letter = _answer_letter(ds[checked]["messages"][1]["content"])
        expected_text = [row["opa"], row["opb"], row["opc"], row["opd"]][row["cop"]]
        prompt = ds[checked]["messages"][0]["content"]
        assert f"{letter}. {expected_text}" in prompt
        checked += 1
    assert checked > 0


def test_medexpqa_correct_option_is_one_indexed():
    ds = load_medexpqa(ALL_SOURCES["medexpqa"])
    raw = load_dataset("HiTZ/MedExpQA", "en", split="train")
    raw = [r for r in raw if r["correct_option"] is not None and str(r["correct_option"]) in r["options"]]
    for i in range(10):
        letter = _answer_letter(ds[i]["messages"][1]["content"])
        gold_text = raw[i]["options"][str(raw[i]["correct_option"])]
        assert f"{letter}. {gold_text}" in ds[i]["messages"][0]["content"]


def test_pubmedqa_holds_out_official_test_split():
    """The 1,000-row labeled set must yield exactly the 500 non-test rows."""
    ds = load_pubmedqa(ALL_SOURCES["pubmedqa"])
    assert len(ds) == 500
    for i in range(5):
        assert ds[i]["messages"][1]["content"].strip().endswith(("yes", "no", "maybe"))


def test_every_row_is_user_then_assistant():
    ds = load_medqa(ALL_SOURCES["medqa"])
    for i in range(5):
        roles = [m["role"] for m in ds[i]["messages"]]
        assert roles == ["user", "assistant"]
