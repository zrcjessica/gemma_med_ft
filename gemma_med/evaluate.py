"""Benchmark a checkpoint on the MedGemma text benchmarks via vLLM.

Runs in the `.venv-eval` environment (vLLM pins torch and can't share the
training env). Benchmarks and splits mirror docs/RECIPE.md so numbers are
comparable to the report's Table 4 and to the base gemma-3-*-it baseline.

This module only *generates*. Scoring is a separate post-process
(`gemma_med.judge`, an LLM judge) over the saved `*_predictions.jsonl`, for the
same reason the regex parser was split out before it: generations are expensive
and scoring is not, so scoring must be re-runnable without touching the GPU.
Each row therefore carries everything the judge needs -- question, option texts,
gold -- rather than just the generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from .chat import GEMMA3_CHAT_TEMPLATE
from .data import pubmedqa_test_ids

MCQ_INSTRUCTION = "Answer the following multiple-choice question."
# Ask for a terminal, machine-checkable answer line. Free-form CoT before it is
# fine and expected. The judge reads the whole response, not just this line, but
# asking for it keeps the task identical to how the benchmarks were scored
# before -- the prompt is part of the measurement and must not drift.
MCQ_SUFFIX = "\n\nThink step by step, then end your reply with 'Answer: <letter>'."
YN_SUFFIX = "\n\nThink step by step, then end your reply with 'Answer: yes', 'Answer: no', or 'Answer: maybe'."

YN_OPTIONS = {"yes": "yes", "no": "no", "maybe": "maybe"}


def build_prompt(item: dict) -> str:
    """The exact text shown to the model under test. Frozen -- see MCQ_SUFFIX."""
    if item["kind"] == "mcq":
        lines = [f"{k}. {v}" for k, v in sorted(item["options"].items())]
        return f"{MCQ_INSTRUCTION}\n\n{item['question'].strip()}\n\n" + "\n".join(lines) + MCQ_SUFFIX
    return (
        "Given the following abstract, answer the question with yes, no, or maybe.\n\n"
        f"{item['context']}\n\nQuestion: {item['question'].strip()}{YN_SUFFIX}"
    )


def load_medqa_test():
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    return [
        {"question": r["question"], "options": dict(r["options"]), "gold": r["answer_idx"], "kind": "mcq"}
        for r in ds
    ]


def load_medmcqa_val():
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    out = []
    for r in ds:
        if r["cop"] is None or not 0 <= r["cop"] <= 3:
            continue
        opts = {"A": r["opa"], "B": r["opb"], "C": r["opc"], "D": r["opd"]}
        out.append({"question": r["question"], "options": opts, "gold": "ABCD"[r["cop"]], "kind": "mcq"})
    return out


def load_pubmedqa_test():
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    test_ids = pubmedqa_test_ids()
    out = []
    for r in ds:
        if str(r["pubid"]) not in test_ids:
            continue
        out.append(
            {
                "question": r["question"],
                "context": "\n".join(r["context"]["contexts"]),
                "options": dict(YN_OPTIONS),
                "gold": r["final_decision"],
                "kind": "yn",
            }
        )
    return out


BENCHMARKS = {
    "medqa": load_medqa_test,
    "medmcqa": load_medmcqa_val,
    "pubmedqa": load_pubmedqa_test,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Subsample for smoke tests.")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    # Greedy decoding sends the fine-tuned models into verbatim repetition loops
    # that run to --max-new-tokens without ever naming an option (docs/BEHAVIORAL_EVAL.md
    # §8). These two exist to test that; 1.0 / 0.0 is the frozen default and the
    # only setting any published number may use.
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model_path)
    if not tok.chat_template:
        tok.chat_template = GEMMA3_CHAT_TEMPLATE

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    decode = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
    }

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    results = {}

    for name in args.benchmarks:
        items = BENCHMARKS[name]()
        if args.limit:
            items = items[: args.limit]

        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": build_prompt(it)}], tokenize=False, add_generation_prompt=True
            )
            for it in items
        ]
        gens = llm.generate(prompts, sampling)

        # `idx` is the item's position in the benchmark and is the join key the
        # judge uses. It is written explicitly rather than left implicit in line
        # order so a partially-rewritten file can still be joined safely.
        with open(out_path / f"{name}_predictions.jsonl", "w") as f:
            for i, (it, g) in enumerate(zip(items, gens)):
                row = {
                    "idx": i,
                    "kind": it["kind"],
                    "gold": it["gold"],
                    "question": it["question"],
                    "options": it["options"],
                    "output": g.outputs[0].text,
                    # Saved so "did this response hit the cap?" is a field lookup
                    # rather than a re-tokenisation of the whole corpus.
                    "n_gen_tokens": len(g.outputs[0].token_ids),
                    "finish_reason": g.outputs[0].finish_reason,
                }
                f.write(json.dumps(row) + "\n")

        n_cap = sum(1 for g in gens if g.outputs[0].finish_reason == "length")
        results[name] = {"n": len(items), "n_at_token_cap": n_cap}
        print(f"{name:<10} generated n={len(items)}  at_cap={n_cap}  (unscored -- run gemma_med.judge)")

    summary = {
        "model": args.model_path,
        "label": args.label or Path(args.model_path).name,
        "scored": False,
        # Part of the instrument: two runs with different decode settings are no
        # more comparable than two runs with different judges.
        "decode": decode,
        "results": results,
    }
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
