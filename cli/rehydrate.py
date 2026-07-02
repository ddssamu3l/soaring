#!/usr/bin/env python3
"""rehydrate.py -- print the durable session state so a fresh or post-compaction
context can resume from disk instead of guessing.

Two uses:
  * manually:  python3 cli/rehydrate.py         (read the board anytime)
  * SessionStart hook (`--hook`): its stdout is injected into the new context, so
    after compaction the session reopens already knowing where it is.

The design principle: conversation is scratch, disk is truth. This is the read
side of that -- task_list.json + progress.txt + git history, distilled.
Pure stdlib; runs under any python3.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
STATE = REPO / "task_list.json"
PROGRESS = REPO / "progress.txt"


def _tail(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return lines[-n:]


def _git_log(n: int) -> list[str]:
    r = subprocess.run(
        ["git", "log", "--oneline", f"-{n}"], cwd=REPO, capture_output=True, text=True
    )
    return r.stdout.strip().splitlines() if r.returncode == 0 else []


def _pickable_next(tasks: list[dict[str, Any]]) -> dict[str, Any] | None:
    status = {t["id"]: t["status"] for t in tasks}
    for t in tasks:
        if t["status"] == "pending" and all(status.get(d) == "done" for d in t["deps"]):
            return t
    return None


def build() -> str:
    out = ["## Session state (rehydrated from disk — resume from here)"]

    if STATE.exists():
        tasks = json.loads(STATE.read_text()).get("tasks", [])
        by = {
            s: [t for t in tasks if t["status"] == s]
            for s in ("done", "active", "blocked", "pending")
        }
        out.append(
            f"\nTasks: {len(by['done'])} done, {len(by['active'])} active, "
            f"{len(by['blocked'])} blocked, {len(by['pending'])} pending."
        )
        if by["active"]:
            for t in by["active"]:
                out.append(f"ACTIVE → {t['id']}: {t['title']}")
        else:
            nxt = _pickable_next(tasks)
            out.append(
                f"NEXT → {nxt['id']}: {nxt['title']}" if nxt else "NEXT → (nothing pickable)"
            )
        for t in by["blocked"]:
            out.append(f"BLOCKED → {t['id']}: {t['title']}  ({t['notes']})")
    else:
        out.append("\n(no task_list.json yet)")

    prog = _tail(PROGRESS, 8)
    if prog:
        out.append("\nRecent progress (progress.txt):")
        out += [f"  {ln}" for ln in prog]

    commits = _git_log(8)
    if commits:
        out.append("\nRecent commits:")
        out += [f"  {ln}" for ln in commits]

    out.append(
        "\nResume an ACTIVE task if one exists; if none, propose the NEXT task and await "
        "the user's go before starting it. Mutate state ONLY via cli/task.py."
    )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="rehydrate.py", description="Print durable session state from disk."
    )
    ap.add_argument(
        "--hook",
        action="store_true",
        help="SessionStart-hook mode: emit JSON additionalContext for reliable injection",
    )
    args = ap.parse_args()
    board = build()
    if args.hook:
        # A SessionStart hook's raw stdout is NOT reliably injected across Claude Code
        # versions; the documented `additionalContext` form is. Emit that JSON so the
        # board actually lands in the new session's context.
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": board,
                    }
                }
            )
        )
    else:
        print(board)
    return 0


if __name__ == "__main__":
    sys.exit(main())
