#!/usr/bin/env python3
"""guard_state.py -- PreToolUse hook: block direct Edit/Write of feature_list.json.

The state file may only change through scripts/task.py, which enforces the
invariants (valid JSON, one active task, deps satisfied, real commit for done).
This hook denies any Edit/Write/MultiEdit tool call whose target is that file, so
the CLI is the only door in. Break-glass for a human: ALLOW_STATE_EDIT=1.

Wire in .claude/settings.json under PreToolUse (matcher "Edit|Write|MultiEdit").
Reads the hook JSON on stdin. Exit 2 = block (stderr is shown to Claude); 0 = allow.
Pure stdlib; runs under any python3.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

GUARDED = "feature_list.json"


def main() -> int:
    if os.environ.get("ALLOW_STATE_EDIT") == "1":
        return 0
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # can't parse the hook input -> don't interfere
    file_path = (payload.get("tool_input") or {}).get("file_path", "")
    if file_path and Path(file_path).name == GUARDED:
        print(
            f"Blocked: change {GUARDED} only via `python3 scripts/task.py` "
            "(it enforces the state invariants). Human break-glass: ALLOW_STATE_EDIT=1.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
