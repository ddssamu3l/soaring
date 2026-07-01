#!/usr/bin/env python3
"""hook_probe.py -- a throwaway diagnostic to confirm what this Claude Code
version actually sends hooks, and whether SessionStart stdout is injected.

It does two things:
  1. Appends the raw stdin payload (the hook JSON) to /tmp/soaring-hook-probe.jsonl
     -> proves the event fired and shows its exact schema/field names.
  2. Prints a distinctive marker to stdout
     -> if the marker shows up at the TOP of a fresh session's context, then
        SessionStart stdout-injection works (so rehydrate.py's auto-inject works too).

Wire it TEMPORARILY via .claude/settings.local.json (personal, gitignored), start a
new session, then read the /tmp file and look for the marker. Delete the wiring after.
Pure stdlib.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

LOG = Path("/tmp/soaring-hook-probe.jsonl")
MARKER = ">>> HOOK_PROBE: if you can see this line in context, SessionStart stdout is injected <<<"


def main() -> int:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        payload = {"_unparsed_stdin": raw}
    record = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "payload": payload}
    with open(LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(MARKER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
