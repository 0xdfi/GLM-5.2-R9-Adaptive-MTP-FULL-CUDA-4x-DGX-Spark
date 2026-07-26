#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Smoke-test a running GLM-5.2 R9 endpoint.

Checks, in order:

  1. GET  /health              -> HTTP 200
  2. GET  /v1/models           -> the served model, at the expected max_model_len
  3. POST /v1/chat/completions -> one deterministic generation

The generation is deterministic on purpose: temperature 0, top_p 1, a pinned
seed, a fixed short prompt, and an exact expected string. A smoke test that
accepts any output is not a smoke test.

Standard library only -- no requests, no openai client. Run it from anywhere
that can reach the endpoint.

    python3 scripts/smoke-openai.py --base-url http://<HEAD_NODE_IP>:<API_PORT>

Exit codes: 0 all checks passed; 1 a check failed; 2 usage/connection error.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

EXPECTED_MAX_MODEL_LEN = 520_000
SENTINEL = "R9-SMOKE-OK"


def _request(url: str, timeout: float, payload: dict | None = None):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"connection error for {url}: {exc}") from exc


class Checks:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def ok(self, msg: str) -> None:
        print(f"  PASS  {msg}")
        self.passed += 1

    def bad(self, msg: str) -> None:
        print(f"  FAIL  {msg}")
        self.failed += 1

    def note(self, msg: str) -> None:
        print(f"  ....  {msg}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True,
                    help="e.g. http://<HEAD_NODE_IP>:<API_PORT> (private cluster address)")
    ap.add_argument("--model", default="glm-5.2", help="served model name")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="per-request timeout in seconds (default 180)")
    ap.add_argument("--expect-max-model-len", type=int, default=EXPECTED_MAX_MODEL_LEN,
                    help=f"expected max_model_len (default {EXPECTED_MAX_MODEL_LEN})")
    ap.add_argument("--skip-generation", action="store_true",
                    help="run only /health and /v1/models")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    c = Checks()

    print("== 1. /health ==")
    status, _ = _request(f"{base}/health", args.timeout)
    c.ok("/health returned 200") if status == 200 else c.bad(f"/health returned {status}")

    print()
    print("== 2. /v1/models ==")
    status, body = _request(f"{base}/v1/models", args.timeout)
    if status != 200:
        c.bad(f"/v1/models returned {status}")
    else:
        try:
            models = json.loads(body)["data"]
        except (ValueError, KeyError, TypeError) as exc:
            c.bad(f"/v1/models returned unparseable JSON: {exc}")
            models = []
        ids = [m.get("id") for m in models]
        if args.model in ids:
            c.ok(f"served model '{args.model}' present")
            entry = next(m for m in models if m.get("id") == args.model)
            actual = entry.get("max_model_len")
            if actual == args.expect_max_model_len:
                c.ok(f"max_model_len = {actual}")
            else:
                c.bad(f"max_model_len: expected {args.expect_max_model_len}, got {actual}")
        else:
            c.bad(f"served model '{args.model}' not in {ids}")

    if not args.skip_generation:
        print()
        print("== 3. deterministic generation ==")
        payload = {
            "model": args.model,
            "messages": [
                {"role": "user",
                 "content": f"Reply with exactly this and nothing else: {SENTINEL}"},
            ],
            "temperature": 0,
            "top_p": 1,
            "seed": 20260726,
            "max_tokens": 32,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        status, body = _request(f"{base}/v1/chat/completions", args.timeout, payload)
        if status != 200:
            c.bad(f"/v1/chat/completions returned {status}: {body[:400]!r}")
        else:
            try:
                out = json.loads(body)
                text = out["choices"][0]["message"]["content"] or ""
                usage = out.get("usage", {})
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                c.bad(f"unparseable completion: {exc}")
                text, usage = "", {}
            if SENTINEL in text:
                c.ok(f"generation returned the sentinel ({text.strip()[:80]!r})")
            else:
                c.bad(f"generation did not contain '{SENTINEL}': {text.strip()[:200]!r}")
            if usage:
                c.note(f"usage: prompt={usage.get('prompt_tokens')} "
                       f"completion={usage.get('completion_tokens')}")

    print()
    print("== result ==")
    print(f"  {c.passed} passed, {c.failed} failed")
    if c.failed:
        return 1
    print("  OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
