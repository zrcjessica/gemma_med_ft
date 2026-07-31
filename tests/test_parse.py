"""Tests for the superseded regex answer parser.

Scoring now goes through gemma_med.judge; answer_parsing.py is kept only to
reproduce pre-judge numbers. These tests pin its behaviour so that reproduction
stays faithful -- they are not a check on the current scoring path.
"""

import pytest

from gemma_med.answer_parsing import parse_answer


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Answer: C", "C"),
        ("answer: c", "C"),
        ("Answer: (B)", "B"),
        ("Answer - D", "D"),
        ("The answer is A", None),  # not our required format; falls through
        ("Reasoning...\n\nAnswer: D. Nitrofurantoin", "D"),
        # CoT that mentions other options first -- must take the last.
        ("Could be Answer: A, but no.\nActually Answer: C", "C"),
        ("blah\nC", "C"),      # bare letter on final line
        ("blah\n(C).", "C"),
        ("no answer here", None),
        ("", None),
        ("Answer: **C**", "C"),          # markdown bold
        ("Answer: **D**. Nitrofurantoin", "D"),
    ],
)
def test_mcq(text, expected):
    assert parse_answer(text, "mcq") == expected


@pytest.mark.parametrize(
    "text",
    [
        "Answer: about 5 mg",              # would parse as A
        "Answer: definitely C",            # would parse as D
        "Answer: Eosinophils are elevated",  # would parse as E
        "Answer: Category B",              # would parse as C
        "Answer: correct option follows",  # would parse as C
    ],
)
def test_mcq_rejects_leading_letter_of_next_word(text):
    """Never silently grab the first letter of the following word.

    Returning None marks the row unparsed, which is visible in unparsed_pct;
    returning a confidently wrong letter is not.
    """
    assert parse_answer(text, "mcq") is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Answer: yes", "yes"),
        ("ANSWER: NO", "no"),
        ("Answer: maybe", "maybe"),
        ("Long reasoning.\n\nAnswer: yes", "yes"),
        ("Answer: no\nActually, Answer: yes", "yes"),
        ("yes", "yes"),
        ("nonsense", None),
        ("Answer: **yes**", "yes"),
        ("Answer: nothing conclusive", None),  # must not match the "no" in "nothing"
        ("Answer: noteworthy findings", None),
    ],
)
def test_yn(text, expected):
    assert parse_answer(text, "yn") == expected


def test_mcq_does_not_match_yn_words():
    assert parse_answer("Answer: yes", "mcq") is None
