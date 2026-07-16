"""Chat-template tests. Run against a real Gemma 3 tokenizer:

    GEMMA_TOKENIZER=/path/to/gemma-3-1b-it python -m pytest tests/test_chat.py -q

These guard the two ways completion-only training fails silently: the training
template drifting from the inference template, and the assistant mask covering
the wrong tokens (which trains the model on the prompt and looks fine in the
loss curve).
"""

import os

import pytest
from transformers import AutoTokenizer

from gemma_med.chat import (
    GEMMA3_CHAT_TEMPLATE,
    GEMMA3_CHAT_TEMPLATE_TRAIN,
    ensure_chat_template,
)

TOKENIZER = os.environ.get("GEMMA_TOKENIZER")
pytestmark = pytest.mark.skipif(not TOKENIZER, reason="set GEMMA_TOKENIZER")

MSGS = [
    {"role": "user", "content": "What causes anemia?"},
    {"role": "assistant", "content": "Iron deficiency is the most common cause."},
    {"role": "user", "content": "How is it treated?"},
    {"role": "assistant", "content": "Oral iron supplementation."},
]


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TOKENIZER)


def test_templates_render_identically(tok):
    """The {% generation %} markers must not change a single byte of output."""
    tok.chat_template = GEMMA3_CHAT_TEMPLATE
    stock = tok.apply_chat_template(MSGS, tokenize=False)
    tok.chat_template = GEMMA3_CHAT_TEMPLATE_TRAIN
    train = tok.apply_chat_template(MSGS, tokenize=False)
    assert stock == train


def test_matches_official_template_if_present(tok):
    """If the checkpoint ships a template (-it), ours must match it byte for byte."""
    official = AutoTokenizer.from_pretrained(TOKENIZER).chat_template
    if not official:
        pytest.skip("checkpoint has no chat_template (-pt base)")
    ref = AutoTokenizer.from_pretrained(TOKENIZER)
    expected = ref.apply_chat_template(MSGS, tokenize=False)
    tok.chat_template = GEMMA3_CHAT_TEMPLATE
    assert tok.apply_chat_template(MSGS, tokenize=False) == expected


def test_assistant_mask_covers_only_assistant_turns(tok):
    ensure_chat_template(tok, for_training=True)
    enc = tok.apply_chat_template(
        MSGS, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
    )
    mask = enc["assistant_masks"]
    assert sum(mask) > 0, "no assistant tokens marked -- {% generation %} not honored"

    ids = enc["input_ids"]
    kept = tok.decode([i for i, m in zip(ids, mask) if m])
    dropped = tok.decode([i for i, m in zip(ids, mask) if not m])

    # Assistant content is trained on; user content never is.
    assert "Iron deficiency" in kept
    assert "Oral iron supplementation" in kept
    assert "What causes anemia" not in kept
    assert "How is it treated" not in kept
    assert "What causes anemia" in dropped

    # The model must learn to stop.
    assert "<end_of_turn>" in kept


def test_generation_prompt_appended(tok):
    ensure_chat_template(tok, for_training=False)
    out = tok.apply_chat_template(MSGS[:1], tokenize=False, add_generation_prompt=True)
    assert out.endswith("<start_of_turn>model\n")
