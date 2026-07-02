#!/usr/bin/env python3
"""guard_state.py -- PreToolUse hook: block direct Edit/Write of the durable-state files.

The durable-state files may only change through their CLI, which enforces the invariants:
  * task_list.json -> scripts/task.py  (valid JSON, one active task, deps, real commit)
  * progress.txt   -> scripts/log.py   (append-only, auto-stamped date + task + sha)
This hook denies any Edit/Write/MultiEdit tool call whose target is one of them, so the CLI
is the only door in. It only intercepts the *tool* calls, not the CLIs' own writes (those
run in a subprocess), so task.py/log.py keep working. Break-glass for a human: ALLOW_STATE_EDIT=1.

Wire in .claude/settings.json under PreToolUse (matcher "Edit|Write|MultiEdit").
Reads the hook JSON on stdin. Exit 2 = block (stderr is shown to Claude); 0 = allow.
Pure stdlib; runs under any python3.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# the guarded state file (resolved, absolute) -> the CLI allowed to change it.
# Keyed by the REPO's actual file, not a bare name, so we lock only this repo's
# state — an unrelated file that happens to be named progress.txt elsewhere
# (e.g. /tmp, an additional working dir) is NOT blocked.
GUARDED = {
    (REPO / "task_list.json").resolve(): "python3 scripts/task.py",
    (REPO / "progress.txt").resolve(): "python3 scripts/log.py",
}


def main() -> int:
    if os.environ.get("ALLOW_STATE_EDIT") == "1":
        return 0
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # can't parse the hook input -> don't interfere
    file_path = (payload.get("tool_input") or {}).get("file_path", "")
    if not file_path:
        return 0
    target = Path(file_path).resolve()
    cli = GUARDED.get(target)
    if cli:
        print(
            f"Blocked: change {target.name} only via `{cli}` (it enforces the state "
            "invariants). Human break-glass: ALLOW_STATE_EDIT=1.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
