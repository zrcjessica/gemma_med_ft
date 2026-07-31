"""Does the self-hosted Kimi server give gemma_med.judge what it needs?

The first probe found the server is vLLM serving a *reasoning* model: every
response came back with `content: null` and the text in a separate `reasoning`
field, having burned the whole token budget thinking. Both of the judge's
assumptions -- read `.content`, cap at 64 tokens -- are wrong against it.

This checks the two things that decide the judge's implementation:
  1. With enough headroom for the reasoning to finish, does `content` arrive and
     does `response_format: json_schema` actually constrain it?
  2. Can the reasoning be turned off? A judge that thinks for 400 tokens per item
     costs ~250k x 400 tokens of someone else's GPU time for no gain -- the task
     is "which option did this text pick", not a reasoning problem.

Run via scripts/probe_kimi_endpoint.sbatch's discovery, or:
    python3 scripts/probe_kimi_schema.py http://sp-0005:8000/v1 Barney
"""

import json
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://sp-0005:8000/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Barney"

SCHEMA = {
    "type": "object",
    "properties": {
        "chosen": {"type": "string", "enum": ["A", "B", "none"]},
        "verdict": {"type": "string", "enum": ["correct", "incorrect", "no_answer"]},
    },
    "required": ["chosen", "verdict"],
    "additionalProperties": False,
}
MESSAGES = [
    {"role": "system", "content": "Grade the answer. Reply with the JSON object only."},
    {
        "role": "user",
        "content": "Options:\nA. aspirin\nB. heparin\n\nCorrect option: B. heparin\n\n"
        "Model response:\n<<<\nI would give heparin.\n>>>\n\nWhich option did it choose, and is it correct?",
    },
]
FMT = {"type": "json_schema", "json_schema": {"name": "verdict", "strict": True, "schema": SCHEMA}}

CASES = [
    ("schema, 1500 tok", {"max_tokens": 1500, "response_format": FMT}),
    ("schema, 1500 tok, enable_thinking=false", {"max_tokens": 1500, "response_format": FMT,
                                                 "chat_template_kwargs": {"enable_thinking": False}}),
    ("schema, 1500 tok, reasoning_effort=low", {"max_tokens": 1500, "response_format": FMT,
                                                "reasoning_effort": "low"}),
    ("no schema, 1500 tok", {"max_tokens": 1500}),
]


def call(extra):
    body = {"model": MODEL, "messages": MESSAGES, **extra}
    req = urllib.request.Request(
        f"{URL}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


for label, extra in CASES:
    print(f"\n=== {label} ===")
    try:
        d = call(extra)
    except Exception as e:
        body = getattr(e, "read", lambda: b"")()
        print(f"  ERROR {type(e).__name__}: {e}  {body[:300]}")
        continue
    msg = d["choices"][0]["message"]
    content, reasoning = msg.get("content"), msg.get("reasoning") or ""
    print(f"  finish_reason : {d['choices'][0]['finish_reason']}")
    print(f"  completion_tok: {d['usage']['completion_tokens']}")
    print(f"  reasoning len : {len(reasoning)} chars")
    print(f"  content       : {content!r}"[:400])
    if content:
        try:
            parsed = json.loads(content)
            ok = set(parsed) == {"chosen", "verdict"} and parsed["chosen"] in ("A", "B", "none")
            print(f"  -> JSON parses, schema-conformant: {ok}  {parsed}")
        except json.JSONDecodeError as e:
            print(f"  -> NOT valid JSON ({e}) -- schema not enforced")
