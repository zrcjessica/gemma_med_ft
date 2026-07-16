"""Benchmark a checkpoint on the MedGemma text benchmarks via vLLM.

Runs in the `.venv-eval` environment (vLLM pins torch and can't share the
training env). Benchmarks and splits mirror docs/RECIPE.md so numbers are
comparable to the report's Table 4 and to the base gemma-3-*-it baseline.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

from .chat import GEMMA3_CHAT_TEMPLATE
from .data import pubmedqa_test_ids

MCQ_INSTRUCTION = "Answer the following multiple-choice question."
# Ask for a terminal, machine-checkable answer line. Free-form CoT before it is
# fine and expected -- we parse the last match, not the first.
MCQ_SUFFIX = "\n\nThink step by step, then end your reply with 'Answer: <letter>'."
YN_SUFFIX = "\n\nThink step by step, then end your reply with 'Answer: yes', 'Answer: no', or 'Answer: maybe'."


def _mcq_prompt(question: str, options: dict[str, str]) -> str:
    lines = [f"{k}. {v}" for k, v in sorted(options.items())]
    return f"{MCQ_INSTRUCTION}\n\n{question.strip()}\n\n" + "\n".join(lines) + MCQ_SUFFIX


def load_medqa_test():
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    return [
        {"prompt": _mcq_prompt(r["question"], r["options"]), "gold": r["answer_idx"], "kind": "mcq"}
        for r in ds
    ]


def load_medmcqa_val():
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    out = []
    for r in ds:
        if r["cop"] is None or not 0 <= r["cop"] <= 3:
            continue
        opts = {"A": r["opa"], "B": r["opb"], "C": r["opc"], "D": r["opd"]}
        out.append({"prompt": _mcq_prompt(r["question"], opts), "gold": "ABCD"[r["cop"]], "kind": "mcq"})
    return out


def load_pubmedqa_test():
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    test_ids = pubmedqa_test_ids()
    out = []
    for r in ds:
        if str(r["pubid"]) not in test_ids:
            continue
        ctx = "\n".join(r["context"]["contexts"])
        prompt = (
            "Given the following abstract, answer the question with yes, no, or maybe.\n\n"
            f"{ctx}\n\nQuestion: {r['question'].strip()}{YN_SUFFIX}"
        )
        out.append({"prompt": prompt, "gold": r["final_decision"], "kind": "yn"})
    return out


BENCHMARKS = {
    "medqa": load_medqa_test,
    "medmcqa": load_medmcqa_val,
    "pubmedqa": load_pubmedqa_test,
}

# The option letter must stand alone. Without the trailing lookahead these
# regexes happily match the first letter of the *next word*: "Answer: definitely
# C" parsed as D, "Answer: Category B" as C, "Answer: about 5mg" as A. That
# doesn't crash -- it silently reports a wrong accuracy. Same for yes/no, where
# an unanchored alternation matches the "no" inside "nothing".
# `\**` tolerates markdown bold, which instruction-tuned models emit constantly.
_MCQ_RE = re.compile(r"answer\s*[:\-]*\s*\**\(?\s*([A-E])\s*\)?\**(?![A-Za-z])", re.I)
_YN_RE = re.compile(r"answer\s*[:\-]*\s*\**\s*\b(yes|no|maybe)\b", re.I)


def parse_answer(text: str, kind: str) -> str | None:
    rx = _MCQ_RE if kind == "mcq" else _YN_RE
    matches = rx.findall(text)
    if matches:
        return matches[-1].upper() if kind == "mcq" else matches[-1].lower()
    # Fallback: a bare letter/word on the final non-empty line.
    for line in reversed([l.strip() for l in text.strip().splitlines() if l.strip()]):
        if kind == "mcq" and re.fullmatch(r"\(?([A-E])\)?\.?", line):
            return re.sub(r"[^A-E]", "", line).upper()
        if kind == "yn" and line.lower().strip(".") in {"yes", "no", "maybe"}:
            return line.lower().strip(".")
        break
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Subsample for smoke tests.")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
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
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    results = {}

    for name in args.benchmarks:
        items = BENCHMARKS[name]()
        if args.limit:
            items = items[: args.limit]

        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": it["prompt"]}], tokenize=False, add_generation_prompt=True
            )
            for it in items
        ]
        gens = llm.generate(prompts, sampling)

        correct = unparsed = 0
        rows = []
        for it, g in zip(items, gens):
            text = g.outputs[0].text
            pred = parse_answer(text, it["kind"])
            if pred is None:
                unparsed += 1
            ok = pred == it["gold"]
            correct += ok
            rows.append({"gold": it["gold"], "pred": pred, "correct": ok, "output": text})

        acc = 100.0 * correct / len(items)
        results[name] = {
            "accuracy": round(acc, 2),
            "n": len(items),
            "unparsed": unparsed,
            "unparsed_pct": round(100.0 * unparsed / len(items), 2),
        }
        with open(out_path / f"{name}_predictions.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        # A high unparsed rate means the accuracy number is measuring format
        # compliance, not medical knowledge. Surface it next to the score.
        print(f"{name:<10} acc={acc:5.2f}  n={len(items):<6} unparsed={unparsed} ({results[name]['unparsed_pct']}%)")

    summary = {"model": args.model_path, "label": args.label or Path(args.model_path).name, "results": results}
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
