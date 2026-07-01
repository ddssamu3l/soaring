#!/usr/bin/env python3
"""log.py -- append a stamped line to progress.txt (the narrative log).

Auto-prepends the date, the current active task id, and the short git sha, so
every entry is uniform without the caller having to remember the format.
Append-only by nature; low corruption risk (it's text), so no hard guard.

Usage:  python3 scripts/log.py "t2 started — plan: 2-layer MLP, MSE, keystone plot"
Pure stdlib; runs under any python3.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROGRESS = REPO / "progress.txt"
STATE = REPO / "feature_list.json"


def _active() -> str:
    if not STATE.exists():
        return "-"
    try:
        for t in json.loads(STATE.read_text()).get("tasks", []):
            if t["status"] == "active":
                return str(t["id"])
    except (json.JSONDecodeError, OSError, KeyError):
        pass
    return "-"


def _sha() -> str:
    r = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True
    )
    return r.stdout.strip() if r.returncode == 0 else "-------"


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="log.py", description="Append a stamped line to progress.txt."
    )
    ap.add_argument("message", nargs="+", help="the progress note to append")
    args = ap.parse_args()
    msg = " ".join(args.message)
    day = datetime.date.today().isoformat()
    line = f"{day}  [{_active()} {_sha()}]  {msg}\n"
    with open(PROGRESS, "a") as f:
        f.write(line)
    print(line.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
