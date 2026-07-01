#!/usr/bin/env python3
"""task.py -- the ONLY sanctioned way to mutate feature_list.json.

Free-editing structured state with an LLM eventually corrupts it: malformed
JSON, two tasks marked active, deps skipped, a task "done" with no real commit.
So we do the same thing we do for code -- don't trust the agent to hold the
invariant, build a tool that enforces it. A PreToolUse hook (guard_state.py)
blocks direct Edit/Write of feature_list.json, so this CLI is the only door in.

Pure stdlib on purpose: hooks and land.py call it, and it must never depend on
the project venv.

Commands:
    task.py add   --title T [--deps a,b] [--files "a.py;b.py"] [--notes N]
    task.py start <id>              # -> active   (refuses if another is active)
    task.py done  <id> --commit SHA # -> done     (SHA must exist in git)
    task.py block <id> --reason R   # -> blocked
    task.py list                    # status board
    task.py next                    # the next pickable task
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
STATE = REPO / "feature_list.json"


def _load() -> dict[str, Any]:
    if not STATE.exists():
        return {"tasks": []}
    return json.loads(STATE.read_text())


def _save(data: dict[str, Any]) -> None:
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, STATE)  # atomic: a crash mid-write never corrupts the file


def _find(data: dict[str, Any], tid: str) -> dict[str, Any] | None:
    return next((t for t in data["tasks"] if t["id"] == tid), None)


def _status(data: dict[str, Any], tid: str) -> str:
    t = _find(data, tid)
    return t["status"] if t else "missing"


def _next_id(data: dict[str, Any]) -> str:
    existing = {t["id"] for t in data["tasks"]}
    n = 1
    while f"t{n}" in existing:
        n += 1
    return f"t{n}"


def _git_has(sha: str) -> bool:
    return (
        subprocess.run(["git", "cat-file", "-e", sha], cwd=REPO, capture_output=True).returncode
        == 0
    )


def _err(msg: str) -> int:
    print(f"task: {msg}", file=sys.stderr)
    return 1


def cmd_add(a: argparse.Namespace) -> int:
    data = _load()
    deps = [d for d in (a.deps.split(",") if a.deps else []) if d]
    for d in deps:
        if not _find(data, d):
            return _err(f"dep {d!r} does not exist")
    files = [f for f in (a.files.split(";") if a.files else []) if f]
    tid = _next_id(data)
    data["tasks"].append(
        {
            "id": tid,
            "title": a.title,
            "status": "pending",
            "deps": deps,
            "files": files,
            "commit": None,
            "notes": a.notes or "",
        }
    )
    _save(data)
    print(f"added {tid}: {a.title}")
    return 0


def cmd_start(a: argparse.Namespace) -> int:
    data = _load()
    t = _find(data, a.id)
    if not t:
        return _err(f"{a.id} not found")
    active = [x["id"] for x in data["tasks"] if x["status"] == "active"]
    if active and active != [a.id]:
        return _err(f"{active[0]} is already active — single-writer. done/block it first.")
    if t["status"] not in ("pending", "blocked"):
        return _err(f"{a.id} is {t['status']}, cannot start")
    unmet = [d for d in t["deps"] if _status(data, d) != "done"]
    if unmet:
        return _err(f"deps not done: {unmet}")
    t["status"] = "active"
    _save(data)
    print(f"started {a.id}: {t['title']}")
    return 0


def cmd_done(a: argparse.Namespace) -> int:
    data = _load()
    t = _find(data, a.id)
    if not t:
        return _err(f"{a.id} not found")
    if t["status"] != "active":
        return _err(f"{a.id} is {t['status']}, only an active task can be marked done")
    if not _git_has(a.commit):
        return _err(f"commit {a.commit!r} not in git — a real landed commit is required")
    t["status"] = "done"
    t["commit"] = a.commit
    _save(data)
    print(f"done {a.id} @ {a.commit[:8]}")
    return 0


def cmd_block(a: argparse.Namespace) -> int:
    data = _load()
    t = _find(data, a.id)
    if not t:
        return _err(f"{a.id} not found")
    t["status"] = "blocked"
    t["notes"] = (t["notes"] + f" | BLOCKED: {a.reason}").strip(" |")
    _save(data)
    print(f"blocked {a.id}: {a.reason}")
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    data = _load()
    if not data["tasks"]:
        print("(no tasks)")
        return 0
    for t in data["tasks"]:
        dep = f"  deps={t['deps']}" if t["deps"] else ""
        commit = f"  @{t['commit'][:8]}" if t["commit"] else ""
        print(f"  [{t['status']:7}] {t['id']}: {t['title']}{dep}{commit}")
    return 0


def cmd_next(_: argparse.Namespace) -> int:
    data = _load()
    active = _find_by_status(data, "active")
    if active:
        print(f"active: {active['id']}: {active['title']}")
        return 0
    for t in data["tasks"]:
        if t["status"] == "pending" and all(_status(data, d) == "done" for d in t["deps"]):
            print(f"next: {t['id']}: {t['title']}")
            return 0
    print("(nothing pickable — all done, or remaining tasks are blocked/waiting on deps)")
    return 0


def _find_by_status(data: dict[str, Any], status: str) -> dict[str, Any] | None:
    return next((t for t in data["tasks"] if t["status"] == status), None)


def main() -> int:
    ap = argparse.ArgumentParser(prog="task.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add")
    p_add.add_argument("--title", required=True)
    p_add.add_argument("--deps", default="")
    p_add.add_argument("--files", default="")
    p_add.add_argument("--notes", default="")
    p_add.set_defaults(fn=cmd_add)

    p_start = sub.add_parser("start")
    p_start.add_argument("id")
    p_start.set_defaults(fn=cmd_start)

    p_done = sub.add_parser("done")
    p_done.add_argument("id")
    p_done.add_argument("--commit", required=True)
    p_done.set_defaults(fn=cmd_done)

    p_block = sub.add_parser("block")
    p_block.add_argument("id")
    p_block.add_argument("--reason", required=True)
    p_block.set_defaults(fn=cmd_block)

    sub.add_parser("list").set_defaults(fn=cmd_list)
    sub.add_parser("next").set_defaults(fn=cmd_next)

    args = ap.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
