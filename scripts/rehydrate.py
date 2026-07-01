#!/usr/bin/env python3
"""rehydrate.py -- print the durable session state so a fresh or post-compaction
context can resume from disk instead of guessing.

Two uses:
  * manually:  python3 scripts/rehydrate.py         (read the board anytime)
  * SessionStart hook (`--hook`): its stdout is injected into the new context, so
    after compaction the session reopens already knowing where it is.

The design principle: conversation is scratch, disk is truth. This is the read
side of that -- feature_list.json + progress.txt + git history, distilled.
Pure stdlib; runs under any python3.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
STATE = REPO / "feature_list.json"
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
        out.append("\n(no feature_list.json yet)")

    prog = _tail(PROGRESS, 8)
    if prog:
        out.append("\nRecent progress (progress.txt):")
        out += [f"  {ln}" for ln in prog]

    commits = _git_log(8)
    if commits:
        out.append("\nRecent commits:")
        out += [f"  {ln}" for ln in commits]

    out.append("\nResume the ACTIVE/NEXT task. Mutate state ONLY via scripts/task.py.")
    return "\n".join(out)


def main() -> int:
    print(build())
    return 0


if __name__ == "__main__":
    sys.exit(main())
