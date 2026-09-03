#!/usr/bin/env python3
"""Record real Azure replies into the fixture file so the suite can stay offline.

This is the only script in a project that is allowed to cost money, and it is
never run by CI. Give it a JSONL file of message lists, one call per line:

    {"messages": [{"role": "user", "content": "..."}]}

Then: LLM_PROVIDER=azure python scripts/record_fixtures.py prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.adapter import AzureProvider, Message  # noqa: E402
from llm.stub import record  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompts", type=Path, help="JSONL, one {'messages': [...]} per line")
    parser.add_argument("--fixtures", type=Path, default=None)
    args = parser.parse_args()

    if os.getenv("LLM_PROVIDER", "").lower() != "azure":
        return int(bool(print("Refusing to run without LLM_PROVIDER=azure set explicitly.")))

    provider = AzureProvider()
    written = 0
    for n, line in enumerate(args.prompts.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        messages = [Message(**m) for m in json.loads(line)["messages"]]
        reply = provider.complete(messages)
        key = record(messages, reply.text, args.fixtures)
        print(f"line {n}: recorded {key} ({reply.total_tokens} tokens)")
        written += 1

    print(f"\n{written} fixture(s) written. Commit them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
