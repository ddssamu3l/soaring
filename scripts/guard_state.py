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

# file name -> the CLI that is allowed to change it
GUARDED = {
    "task_list.json": "python3 scripts/task.py",
    "progress.txt": "python3 scripts/log.py",
}


def main() -> int:
    if os.environ.get("ALLOW_STATE_EDIT") == "1":
        return 0
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # can't parse the hook input -> don't interfere
    file_path = (payload.get("tool_input") or {}).get("file_path", "")
    name = Path(file_path).name if file_path else ""
    if name in GUARDED:
        print(
            f"Blocked: change {name} only via `{GUARDED[name]}` (it enforces the state "
            "invariants). Human break-glass: ALLOW_STATE_EDIT=1.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
