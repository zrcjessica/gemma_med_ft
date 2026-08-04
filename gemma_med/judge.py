"""LLM-judge scoring for the behavioral benchmarks.

Replaces the regex answer parser (`answer_parsing.py`) as the scoring path. The
regex could only recognise answers in the shapes we anticipated, and medical SFT
kept inventing new ones -- the last such surprise turned a 95% compliant model
into a reported 15% until the parser was patched. A judge reads the response the
way a human grader would, so the score stops being a function of our regex.

Scoring is a *post-process* over `*_predictions.jsonl`, never part of the eval
run: generations are expensive, judgments are cheap and re-runnable. Nothing
here imports vLLM or torch.

Three-way verdict, deliberately:

  correct    -- committed to the gold option
  incorrect  -- committed to a different option
  no_answer  -- committed to no option at all

`no_answer` is the LLM analogue of the old "unparsed", so the
acc_raw / acc_parsed / answer_rate split that separates lost knowledge from lost
format compliance survives intact. A binary verdict would collapse the two.

Two judgment calls worth knowing about before trusting the numbers:

1. **Content beats letter.** A response like "A. <verbatim text of option C>"
   is scored as a choice of C. Fine-tuned models re-letter the option list
   constantly, and the letter is a formatting artifact while the text is the
   thing that shows whether the model knew the answer. This is more generous
   than the regex, which took the letter.
2. **The judge never re-decides the medicine.** It is given the gold option and
   told to grade against it, not against its own opinion. Otherwise we would be
   measuring agreement between two models rather than benchmark accuracy.

The judge also returns which option it thought was chosen. `correct` is
recomputed from `chosen` in code and never read off `verdict`, so a judge that
contradicts itself cannot corrupt the score -- but the contradiction rate is
reported as `judge_self_disagreement` because it is the cheapest available
signal of judge quality. Measured 1/300 (0.3%) for Kimi K2.7-Code on 4b
step-1024. Low single digits is noise; several percent means the judge is
struggling with the task and the prompt or model needs revisiting.

Three providers:

  anthropic  Claude API (ANTHROPIC_API_KEY). The only one with a Batches API,
             hence the only one with a half-price offline path. Default.
  moonshot   Kimi via Moonshot's hosted API (MOONSHOT_API_KEY). Sync only, so
             its lower per-token price does not beat batched Claude.
  local      A self-hosted vLLM server -- the lab's Kimi on BigPurple. Free,
             no rate limit, and by far the cheapest option, but reachable only
             from inside BigPurple and pinned to a server that moves (see
             LocalJudge and scripts/kimi_url.sh).

Usage. Price it first, then judge:

    python -m gemma_med.judge --root eval/traj/<tag> --estimate
    python -m gemma_med.judge --root eval/traj/<tag>
    python -m gemma_med.judge --root eval/traj/<tag> --provider moonshot --mode sync

    # Self-hosted Kimi, from a compute node (submit, do not run on the login node):
    sbatch --export=ALL,TAG=<tag> scripts/judge_traj.sbatch          # PROVIDER=local, the default
    sbatch --export=ALL,TAG=<tag>,PROVIDER=anthropic scripts/judge_traj.sbatch

Whichever you pick, freeze it: the judge is part of the measurement apparatus,
like the frozen fit corpus and probe set on the lens side. Scores from two judges
are not comparable, so a trajectory must be judged end to end by one model, and
`judged_summary.json` records which one. That record is also enforced: pointing a
different provider/model at a directory that already holds judgments is refused
unless `--allow-judge-switch`, because the resulting file would be half one
judge's verdicts and half the other's with only the second one named.

Over a whole trajectory that is one batch per (checkpoint, benchmark), each
polled to completion in turn -- fine overnight, slow if you are watching. To
fan out instead, submit everything first and collect later; the batch ids are
persisted per directory, so the second invocation resumes rather than resubmits:

    python -m gemma_med.judge --root eval/traj/<tag> --no-wait   # submit all
    python -m gemma_med.judge --root eval/traj/<tag>             # collect all

Both are resumable and idempotent: judged items are keyed by benchmark index and
skipped, so an interrupted run is re-run, never re-paid.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

BENCHMARKS = ("medqa", "medmcqa", "pubmedqa")

SYSTEM = """\
You are grading a language model's answer to a medical exam question.

You are given the question, the answer options, which option is correct, and the \
model's full response. The response may contain reasoning before its answer.

Your job: decide which single option the model committed to as its final answer, \
then grade that choice against the correct option you were given.

Rules:
- Only the model's FINAL answer counts. If it weighs several options and then \
settles on one, grade the one it settles on.
- Match by option CONTENT, not by letter. Models frequently restate an option's \
text under the wrong letter (e.g. writing "A. <the text of option C>"). That is a \
choice of C. When the letter and the restated text disagree, the text wins.
- A paraphrase counts. The model does not have to quote an option verbatim; if it \
clearly names or describes exactly one option, that is its choice.
- Use "none" only when the response commits to no option at all: a refusal, empty \
or truncated output, reasoning that never reaches a conclusion, an answer that is \
not one of the listed options, or two contradictory final answers. Do NOT use \
"none" merely because the formatting is unusual.
- Do NOT use your own medical knowledge to decide what the right answer is. The \
correct option is given to you. Grade only against it, even if you disagree.

Return the option the model chose, and the verdict:
  correct   -- it chose the correct option
  incorrect -- it chose a different option
  no_answer -- it chose no option ("none")\
"""


def verdict_schema(kind: str) -> dict:
    """The bare JSON Schema. Providers wrap it differently; the shape is shared.

    Constraining the output is load-bearing, not a nicety: an unconstrained judge
    that answers in prose gets counted as an error, and one that answers in
    almost-JSON gets counted as an error too. Both quietly shrink the sample.
    """
    choices = ["A", "B", "C", "D", "E"] if kind == "mcq" else ["yes", "no", "maybe"]
    return {
        "type": "object",
        "properties": {
            "chosen": {
                "type": "string",
                "enum": choices + ["none"],
                "description": "The option the model committed to, or 'none'.",
            },
            "verdict": {"type": "string", "enum": ["correct", "incorrect", "no_answer"]},
        },
        "required": ["chosen", "verdict"],
        "additionalProperties": False,
    }


def build_user_message(row: dict) -> str:
    """What the judge sees.

    Note what is *absent*: PubMedQA's abstract. The judge is told the gold label
    and forbidden from re-deriving it, so the abstract is several hundred tokens
    of pure cost. The MCQ question stem is kept -- it is needed when a model
    answers by describing the clinical picture instead of naming an option.
    """
    opts = row["options"]
    lines = [f"{k}. {v}" for k, v in sorted(opts.items())]
    gold = row["gold"]
    return (
        f"Question:\n{row['question'].strip()}\n\n"
        f"Options:\n" + "\n".join(lines) + "\n\n"
        f"Correct option: {gold}. {opts.get(gold, gold)}\n\n"
        f"Model response:\n<<<\n{row['output']}\n>>>"
    )


# ---------------------------------------------------------------- providers
#
# Two wire formats for the same request. They are NOT interchangeable at the
# endpoint level: Moonshot also publishes an Anthropic-compatible endpoint
# (api.moonshot.ai/anthropic) that the `anthropic` SDK will talk to with only a
# base_url change, but it speaks OpenAI's `response_format` for structured
# output, not Anthropic's `output_config`. Pointing this module's Anthropic path
# at it would drop the schema constraint silently, so Moonshot goes through the
# OpenAI-compatible endpoint where `strict` schemas are actually honoured.


def request_params(row: dict, model: str) -> dict:
    """Anthropic Messages API."""
    return {
        "model": model,
        "max_tokens": 64,
        "system": SYSTEM,
        "output_config": {"format": {"type": "json_schema", "schema": verdict_schema(row["kind"])}},
        "messages": [{"role": "user", "content": build_user_message(row)}],
    }


class AnthropicJudge:
    supports_batch = True

    def __init__(self, model):
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic(max_retries=8)

    def judge(self, row):
        resp = self.client.messages.create(**request_params(row, self.model))
        if resp.stop_reason == "refusal":
            raise RuntimeError("judge refused")
        return json.loads(next(b.text for b in resp.content if b.type == "text"))

    def count_input(self, row):
        p = request_params(row, self.model)
        r = self.client.messages.count_tokens(model=self.model, system=p["system"], messages=p["messages"])
        return r.input_tokens, True


class OpenAICompatJudge:
    """Any OpenAI-compatible chat-completions endpoint.

    No Batches API in this family, so there is no half-price offline path and
    `--mode batch` is rejected rather than silently ignored.
    """

    supports_batch = False
    BASE_URL = "https://api.moonshot.ai/v1"
    KEY_ENV = "MOONSHOT_API_KEY"
    MAX_TOKENS = 64
    EXTRA_BODY: dict = {}

    def __init__(self, model, base_url=None):
        from openai import OpenAI

        self.model = model
        self.base_url = base_url or os.environ.get("JUDGE_BASE_URL") or self.BASE_URL
        self.client = OpenAI(
            api_key=os.environ.get(self.KEY_ENV) or "dummy",
            base_url=self.base_url,
            max_retries=8,
        )

    def params(self, row):
        p = {
            "model": self.model,
            "max_tokens": self.MAX_TOKENS,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_user_message(row)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "strict": True, "schema": verdict_schema(row["kind"])},
            },
        }
        if self.EXTRA_BODY:
            p["extra_body"] = dict(self.EXTRA_BODY)
        return p

    def judge(self, row):
        msg = self.client.chat.completions.create(**self.params(row)).choices[0].message
        content = msg.content
        if not content:
            # A reasoning model that spent the whole budget thinking returns
            # content=None with the text in `reasoning`. Say so, rather than
            # letting `json.loads(None)` surface as an inscrutable TypeError.
            n = len(getattr(msg, "reasoning", None) or "")
            raise RuntimeError(
                f"empty content (reasoning={n} chars) -- raise MAX_TOKENS or disable thinking"
            )
        return json.loads(content)

    def count_input(self, row):
        p = self.params(row)
        try:
            # cast_to must be a *parameterized* mapping: the OpenAI SDK unpacks
            # get_args(type_) and a bare `dict` raises before the request is even
            # made -- which the except below would swallow, silently pinning this
            # to the heuristic forever.
            raw = self.client.post(
                "/tokenizers/estimate-token-count",
                body={"model": self.model, "messages": p["messages"]},
                cast_to=Dict[str, Any],
            )
            n = _find_key(raw, "total_tokens")
            if n is not None:
                return n, True
        except Exception:  # noqa: BLE001 -- estimation must never block judging
            pass
        # Fall back to a character heuristic and say so, rather than quoting a
        # made-up number with the same confidence as a real token count.
        chars = sum(len(m["content"]) for m in p["messages"])
        return chars / 3.5, False


class MoonshotJudge(OpenAICompatJudge):
    """Kimi via Moonshot's hosted API (paid)."""


class LocalJudge(OpenAICompatJudge):
    """A self-hosted vLLM server -- on BigPurple, the lab's Kimi K2.7-Code.

    Verified against vllm-0.25.1 (tp8-pp2) on 2026-07-30. Three things about that
    server drove the settings below, none of them guessable:

    - The served model id is whatever `--served-model-name` was set to, NOT the
      HuggingFace name. On the lab server it is "Barney"; `Kimi-K2.7-Code` 404s.
    - It is a **reasoning** model. Left alone it emits hundreds of reasoning
      tokens per call, returns `content: null`, and puts the text in a separate
      `reasoning` field -- so the obvious `resp.choices[0].message.content` is
      None. `enable_thinking: false` took one probe from 115 completion tokens to
      14 with the same verdict. The judge is a reading-comprehension task, not a
      reasoning one, so the thinking buys nothing and costs shared GPU time.
    - `response_format: json_schema` IS enforced. Without it the same prompt came
      back as {"chosen_option": "B. heparin", "correct": true} -- right answer,
      wrong keys. The schema is load-bearing, not decoration.

    MAX_TOKENS stays generous anyway: if a server build ignores enable_thinking,
    a small cap truncates mid-reasoning and every row fails. Unused budget is free.
    """

    BASE_URL = "http://sp-0005:8000/v1"
    KEY_ENV = "JUDGE_API_KEY"  # vLLM ignores it; "dummy" is fine
    MAX_TOKENS = 1024
    EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


def _find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], (int, float)):
            return obj[key]
        for v in obj.values():
            found = _find_key(v, key)
            if found is not None:
                return found
    return None


PROVIDERS = {
    "anthropic": (AnthropicJudge, "claude-haiku-4-5", "ANTHROPIC_API_KEY"),
    "moonshot": (MoonshotJudge, "kimi-k2.5", "MOONSHOT_API_KEY"),
    "local": (LocalJudge, "Barney", None),
}


# ---------------------------------------------------------------- loading


def _reload_benchmark(name: str) -> list[dict]:
    """Recover question/options for runs scored before evaluate.py saved them.

    The 40-odd trajectory dirs that already exist only stored gold + output. The
    loaders are deterministic and order-preserving, so item i of the benchmark is
    row i of the file; `load_rows` verifies that by checking the gold labels
    agree and refuses to judge if they don't.
    """
    from .evaluate import BENCHMARKS as LOADERS

    return LOADERS[name]()


def load_rows(pred_path: Path, bench: str, limit: int | None) -> list[dict]:
    rows = [json.loads(l) for l in open(pred_path)]
    if limit:
        rows = rows[:limit]
    if rows and "question" in rows[0]:
        for i, r in enumerate(rows):
            r.setdefault("idx", i)
        return rows

    items = _reload_benchmark(bench)
    if len(items) < len(rows):
        sys.exit(f"{pred_path}: {len(rows)} predictions but benchmark has {len(items)} items")
    out = []
    for i, r in enumerate(rows):
        it = items[i]
        if it["gold"] != r["gold"]:
            sys.exit(
                f"{pred_path}: index join failed at row {i} "
                f"(file gold={r['gold']!r}, benchmark gold={it['gold']!r}). "
                "The saved generations cannot be matched to their questions; re-run the eval."
            )
        out.append(
            {
                "idx": i,
                "kind": it["kind"],
                "gold": it["gold"],
                "question": it["question"],
                "options": it["options"],
                "output": r["output"],
            }
        )
    return out


def prior_judge(d: Path) -> str | None:
    """provider/model that last wrote into this dir, or None if it is fresh."""
    sp = d / "judged_summary.json"
    if not sp.exists():
        return None
    try:
        s = json.loads(sp.read_text())
    except json.JSONDecodeError:
        return None
    p, m = s.get("judge_provider"), s.get("judge_model")
    return f"{p or '?'}/{m or '?'}" if (p or m) else None


def check_judge(d: Path, args) -> str | None:
    """Refuse to append a second judge's verdicts to a directory.

    The two providers are a toggle by design, but the toggle is per *run*, not
    per row: `judged_summary.json` names one judge for the whole dir, so a dir
    judged half by Claude and half by Kimi reports scores from an instrument that
    does not exist. Judgments are keyed by index and resumable, which is exactly
    what makes the mistake easy -- a re-run with the other provider silently
    fills in only the rows the first one missed.
    """
    was = prior_judge(d)
    now = f"{args.provider}/{args.model}"
    if was is None or was == now or args.estimate:
        return None
    if not args.allow_judge_switch:
        sys.exit(
            f"{d}: already judged by {was}, but this run is {now}.\n"
            "Two judges' scores are not comparable and judging is resumable, so this "
            "would leave one file holding both. Either re-run with the original judge, "
            f"or delete {d.name}/*_judgments.jsonl and re-judge the dir from scratch, "
            "or pass --allow-judge-switch if you really mean to mix them."
        )
    print(f"  WARNING: {d.name} was judged by {was}; mixing in {now}", file=sys.stderr)
    return was


def done_idxs(out_path: Path) -> set[int]:
    """Judged rows, so an interrupted run resumes instead of re-paying."""
    if not out_path.exists():
        return set()
    seen = set()
    for line in open(out_path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # truncated final line from a killed run
        if r.get("verdict"):
            seen.add(r["idx"])
    return seen


# ---------------------------------------------------------------- judging


def _record(row: dict, parsed: dict) -> dict:
    """One judgment. `correct` is recomputed from `chosen`, not taken on faith."""
    chosen = parsed["chosen"]
    return {
        "idx": row["idx"],
        "gold": row["gold"],
        "chosen": chosen,
        "verdict": parsed["verdict"],
        "correct": chosen != "none" and chosen == row["gold"],
        "answered": chosen != "none",
    }


def judge_sync(judge, rows, out_path, concurrency):
    lock = threading.Lock()
    f = open(out_path, "a")
    n_err = 0

    def one(row):
        nonlocal n_err
        try:
            rec = _record(row, judge.judge(row))
        except Exception as e:  # noqa: BLE001 -- one bad row must not kill 6k others
            with lock:
                n_err += 1
                if n_err <= 5:
                    print(f"  idx {row['idx']}: {type(e).__name__}: {e}", file=sys.stderr)
            return
        with lock:
            f.write(json.dumps(rec) + "\n")
            f.flush()

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(one, rows))
    f.close()
    return n_err


def judge_batch(judge, rows, out_path, state_path, poll):
    """Message Batches API: half price, which is the whole point at this scale.

    The batch id is persisted before the first poll so a killed process resumes
    the existing batch rather than paying for it twice.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client, model = judge.client, judge.model
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    batch_id = state.get("batch_id")

    if batch_id is None:
        batch = client.messages.batches.create(
            requests=[
                Request(
                    custom_id=str(r["idx"]),
                    params=MessageCreateParamsNonStreaming(**request_params(r, model)),
                )
                for r in rows
            ]
        )
        batch_id = batch.id
        state_path.write_text(json.dumps({"batch_id": batch_id, "n": len(rows)}))
        print(f"  submitted batch {batch_id} ({len(rows)} requests)")
    else:
        print(f"  resuming batch {batch_id}")

    if not poll:
        return -1

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        c = batch.request_counts
        print(f"  {batch.processing_status}: {c.succeeded} ok / {c.processing} pending", flush=True)
        time.sleep(30)

    by_idx = {r["idx"]: r for r in rows}
    n_err = 0
    with open(out_path, "a") as f:
        # Results come back in arbitrary order -- key on custom_id, never position.
        for res in client.messages.batches.results(batch_id):
            # A batch resumed after a partial write returns results for rows
            # already on disk; they are not in `todo`, and they are not missing.
            row = by_idx.get(int(res.custom_id))
            if row is None:
                continue
            if res.result.type != "succeeded":
                n_err += 1
                continue
            try:
                text = next(b.text for b in res.result.message.content if b.type == "text")
                f.write(json.dumps(_record(row, json.loads(text))) + "\n")
            except Exception:  # noqa: BLE001
                n_err += 1
    state_path.unlink(missing_ok=True)
    return n_err


# ---------------------------------------------------------------- summary


def summarize(out_path: Path, n_total: int, keep: set[int] | None = None) -> dict:
    recs = [json.loads(l) for l in open(out_path)] if out_path.exists() else []
    # A resumed run appends; a retried idx can appear twice. Last write wins.
    rows = list({r["idx"]: r for r in recs}.values())
    # Under --limit the file may hold more rows than were asked for; summarising
    # all of them reports against a set the caller did not request.
    if keep is not None:
        rows = [r for r in rows if r["idx"] in keep]
    n = len(rows)
    answered = [r for r in rows if r["answered"]]
    correct = [r for r in answered if r["correct"]]
    # The judge's own verdict vs. the verdict implied by the option it picked.
    # `correct` above comes from `chosen`, so a mismatch does not change any
    # score -- it is a quality signal. Low single-digit percent is normal.
    disagree = sum(
        1
        for r in rows
        if r["verdict"] != ("no_answer" if not r["answered"] else "correct" if r["correct"] else "incorrect")
    )
    return {
        "n": n,
        "n_expected": n_total,
        "acc_raw": round(100.0 * len(correct) / n, 2) if n else None,
        "acc_answered": round(100.0 * len(correct) / len(answered), 2) if answered else None,
        "answer_rate": round(100.0 * len(answered) / n, 2) if n else None,
        "judge_self_disagreement": disagree,
        "judge_self_disagreement_pct": round(100.0 * disagree / n, 2) if n else None,
    }


# ---------------------------------------------------------------- cost


# List prices in USD per million tokens, (input, output). These WILL go stale --
# they are here to size a run, not to bill one. Confirm on the provider console
# before treating any number this prints as a budget.
PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "kimi-k2.5": (0.60, 3.00),
    "kimi-k2.6": (0.60, 3.00),
    "kimi-k3": (3.00, 15.00),
}


def estimate(judge, rows, sample=25):
    """Priced before spending. Extrapolates a token count over a sample."""
    step = max(1, len(rows) // sample)
    picks = rows[::step][:sample]
    tot, exact = 0.0, True
    for r in picks:
        n_tok, is_exact = judge.count_input(r)
        tot += n_tok
        exact &= is_exact
    mean_in = tot / len(picks)
    n = len(rows)
    p_in, p_out = PRICES.get(judge.model, (None, None))
    out = {
        "n": n,
        "mean_input_tokens": round(mean_in),
        "token_count": "exact" if exact else "approximate (char heuristic)",
    }
    if p_in is None:
        out["usd"] = f"unknown -- no list price recorded for {judge.model}"
        return out
    # ~15 output tokens per item: the constrained JSON verdict and nothing else.
    sync = (mean_in * n / 1e6) * p_in + (15 * n / 1e6) * p_out
    out["usd_sync"] = round(sync, 2)
    if judge.supports_batch:
        out["usd_batch"] = round(sync / 2, 2)
    return out


# ---------------------------------------------------------------- driver


def judge_dir(judge, d: Path, args) -> dict:
    out = {}
    for bench in args.benchmarks:
        pred = d / f"{bench}_predictions.jsonl"
        if not pred.exists():
            continue
        rows = load_rows(pred, bench, args.limit)
        jpath = d / f"{bench}_judgments.jsonl"
        state = d / f".{bench}_batch.json"

        if args.estimate:
            print(f"{d.name}/{bench}: {estimate(judge, rows)}")
            continue

        done = done_idxs(jpath)  # once, not once per row
        todo = [r for r in rows if r["idx"] not in done]
        if todo:
            print(f"{d.name}/{bench}: judging {len(todo)}/{len(rows)}", flush=True)
            if args.mode == "batch":
                n_err = judge_batch(judge, todo, jpath, state, not args.no_wait)
            else:
                n_err = judge_sync(judge, todo, jpath, args.concurrency)
            if n_err > 0:
                print(f"  {n_err} failed (re-run to retry)", file=sys.stderr)
            if n_err < 0:
                continue  # submitted, not waited on
        s = summarize(jpath, len(rows), {r['idx'] for r in rows})
        out[bench] = s
        print(
            f"{d.name:<12} {bench:<9} acc={s['acc_raw']}  answered={s['acc_answered']}"
            f"  answer_rate={s['answer_rate']}%  n={s['n']}/{s['n_expected']}"
            + (
                f"  self-disagree={s['judge_self_disagreement']} ({s['judge_self_disagreement_pct']}%)"
                if s["judge_self_disagreement"]
                else ""
            )
        )
    return out


def load_dotenv(path: Path | None = None) -> None:
    """Read repo-root `.env` into the environment, without overriding it.

    Only so that toggling `--provider anthropic` does not also mean remembering
    to `export ANTHROPIC_API_KEY` in whatever shell (or Slurm job) this is. A
    real exported value always wins, so `.env` cannot silently swap a key out
    from under a run that set one deliberately.
    """
    p = path or Path(__file__).resolve().parent.parent / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pred-dir", help="A single eval dir containing *_predictions.jsonl")
    src.add_argument("--root", help="A trajectory dir; judges every step-*/ under it")
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    ap.add_argument(
        "--provider",
        default=os.environ.get("JUDGE_PROVIDER", "anthropic"),
        choices=list(PROVIDERS),
        help="anthropic (Claude API, batchable) | moonshot (Kimi hosted API) | "
        "local (self-hosted vLLM, e.g. the lab Kimi on BigPurple). "
        "Defaults to $JUDGE_PROVIDER, else anthropic.",
    )
    ap.add_argument(
        "--base-url",
        default=None,
        help="Override the endpoint (or set JUDGE_BASE_URL). The self-hosted server is a "
        "Slurm job, so its node changes on every restart -- discover it with "
        "scripts/kimi_url.sh rather than hardcoding.",
    )
    ap.add_argument("--model", default=None, help="Defaults to the provider's judge model")
    ap.add_argument(
        "--mode",
        default=None,
        choices=["batch", "sync"],
        help="batch is half price. Defaults to batch on providers that have a Batches "
        "API and sync on those that do not, so switching provider is one variable.",
    )
    ap.add_argument("--concurrency", type=int, default=8, help="sync mode only")
    ap.add_argument(
        "--no-wait",
        action="store_true",
        help="batch mode: submit every batch and exit. Re-run without it to collect; "
        "batch ids are persisted so nothing is resubmitted.",
    )
    ap.add_argument("--limit", type=int, default=None, help="Judge only the first N items (smoke test)")
    ap.add_argument("--estimate", action="store_true", help="Price the run with count_tokens; judge nothing")
    ap.add_argument(
        "--allow-judge-switch",
        action="store_true",
        help="Judge a dir that was already judged by a different provider/model. Off by "
        "default: the result is one file holding two judges' verdicts.",
    )
    args = ap.parse_args()

    load_dotenv()
    cls, default_model, key_env = PROVIDERS[args.provider]
    args.model = args.model or default_model

    # Config errors first, before we touch credentials or import an SDK. Refuse
    # rather than silently downgrade: a batch run that quietly became a sync run
    # costs double and nothing in the output would say so.
    if args.mode == "batch" and not cls.supports_batch and not args.estimate:
        sys.exit(
            f"--provider {args.provider} publishes no Batches API; re-run with --mode sync "
            "(there is no half-price offline path there)."
        )
    # An *explicit* --mode batch on a sync-only provider is an error above, never
    # a silent downgrade. Resolving the unset default per provider is a different
    # thing: nothing was requested, so nothing is being overridden.
    if args.mode is None:
        args.mode = "batch" if cls.supports_batch else "sync"

    # key_env is None for a self-hosted server, which authenticates nothing.
    have_key = (
        key_env is None
        or os.environ.get(key_env)
        or (args.provider == "anthropic" and os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    )
    if not have_key:
        sys.exit(f"{key_env} is not set.")
    try:
        judge = cls(args.model, args.base_url) if cls is not AnthropicJudge else cls(args.model)
    except ImportError as e:
        sys.exit(f"{e} -- run scripts/build_judge_env.sh and use .venv-judge/bin/python")

    # Always name the judge, on every provider: with a toggle in play, "which
    # instrument produced these numbers" must be answerable from the log alone.
    where = f" @ {judge.base_url}" if hasattr(judge, "base_url") else ""
    print(f"judge: {args.provider} {args.model}{where} ({args.mode})")

    if hasattr(judge, "base_url"):
        try:
            judge.client.models.list()
        except Exception as e:  # noqa: BLE001
            sys.exit(
                f"cannot reach {judge.base_url}: {type(e).__name__}: {e}\n"
                "A self-hosted server is reachable only from inside BigPurple (not olab1), and "
                "its node changes whenever the server job restarts -- run scripts/kimi_url.sh "
                "for the current address, or submit via scripts/judge_traj.sbatch."
            )

    if args.pred_dir:
        dirs = [Path(args.pred_dir)]
    else:
        root = Path(args.root)
        dirs = sorted(root.glob("step-*"), key=lambda p: int(re.search(r"step-(\d+)", p.name).group(1)))
    if not dirs:
        sys.exit("no eval dirs found")

    for d in dirs:
        mixed = check_judge(d, args)
        res = judge_dir(judge, d, args)
        if res and not args.estimate:
            summary = {"judge_provider": args.provider, "judge_model": args.model, "results": res}
            if mixed:
                # A dir that survived --allow-judge-switch must say so, or the
                # only trace of the mixing is a shell history nobody kept.
                summary["judge_mixed_with"] = mixed
            (d / "judged_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
