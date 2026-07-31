"""SUPERSEDED. The scoring path is now `gemma_med.judge` (an LLM judge).

Nothing in the pipeline imports this any more. It is kept only so numbers
produced before the switch can be reproduced -- do not wire it back into
evaluate.py or analyze_traj.py, and do not mix regex scores with judge scores in
one figure.

Why it went: the parser could only recognise answer shapes we had anticipated,
and medical SFT kept inventing new ones. Each surprise silently mis-scored a
whole trajectory until someone noticed and patched the regex (see the
_LABELLED_RE note below for the worst case). A judge reads the response the way
a human grader would, so the score stops being a function of our regex.
"""

from __future__ import annotations

import re

_MCQ_RE = re.compile(r"answer\s*[:\-]*\s*\**\(?\s*([A-E])\s*\)?\**(?![A-Za-z])", re.I)
_YN_RE = re.compile(r"answer\s*[:\-]*\s*\**\s*\b(yes|no|maybe)\b", re.I)

# A bare letter alone on the line: "A", "(A)", "A."
_BARE_RE = re.compile(r"\(?([A-E])\)?\.?", re.I)

# The letter followed by the option text: "A. Inhibition of proteasome".
# Medical SFT moves the model to this format -- it answers by restating the
# chosen option rather than emitting a bare letter. Scoring that as unparsed
# understates format compliance badly: at 4b step 1024 it accounted for 94% of
# all "unparsed" MedQA responses, turning a 95% parse rate into a reported 15%
# and making a healthy model look collapsed.
_LABELLED_RE = re.compile(r"\(?([A-E])\)?[\.\):]\s+\S", re.I)


def parse_answer(text: str, kind: str) -> str | None:
    """Extract the answer, or None if the response does not contain one.

    Tried in order of decreasing explicitness: the requested 'Answer: X' form
    anywhere in the response (last match wins, so CoT before it is fine), then
    the final non-empty line as a bare letter, then that line as a labelled
    option. Only the final line is considered for the fallbacks -- earlier lines
    may restate the option list, where a leading letter is not a choice.
    """
    rx = _MCQ_RE if kind == "mcq" else _YN_RE
    matches = rx.findall(text)
    if matches:
        return matches[-1].upper() if kind == "mcq" else matches[-1].lower()

    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return None
    last = lines[-1]

    if kind == "mcq":
        if _BARE_RE.fullmatch(last):
            return re.sub(r"[^A-E]", "", last.upper())
        m = _LABELLED_RE.match(last)
        if m:
            return m.group(1).upper()
    elif last.lower().strip(".") in {"yes", "no", "maybe"}:
        return last.lower().strip(".")
    return None
