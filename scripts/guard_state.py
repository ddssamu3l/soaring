#!/usr/bin/env python3
"""guard_state.py -- PreToolUse hook: block hand-edits that skip the workflow.

Two things are guarded:
  1. task_list.json / progress.txt -- ALWAYS, in ANY checkout (primary or a task's
     worktree). Change them only via their CLI, which enforces the invariants:
       task_list.json -> cli/task.py  (valid JSON, one active task, deps, real commit)
       progress.txt   -> cli/log.py   (append-only, auto-stamped date + task + sha)
     Break-glass for a human: ALLOW_STATE_EDIT=1.
  2. Every OTHER tracked, non-gitignored file -- ONLY when the CURRENT checkout is
     the PRIMARY one, not a linked task worktree. Real changes belong on a task's
     dedicated worktree (`task.py begin tN`), committed there, landed via `land.py`
     -- primary should stay clean and ready to land at all times. A linked worktree
     is unaffected: that's where coding is supposed to happen. Caught for real: a
     tracked-doc edit landed straight in primary once, self-caught before commit
     and redone properly -- that worked because it was noticed, not because
     anything stopped it. Break-glass for a human: ALLOW_MAIN_EDIT=1.
Gitignored files (CLAUDE.md, .venv, data/, ...) are never guarded by either rule --
they're untracked, no gate applies, hand-editing them anywhere is fine.

This hook only intercepts the *tool* calls, not a CLI's own writes (those run in a
subprocess), so task.py/log.py/land.py all keep working normally.

Wire in .claude/settings.json under PreToolUse (matcher "Edit|Write|MultiEdit").
Reads the hook JSON on stdin. Exit 2 = block (stderr is shown to Claude); 0 = allow.
Pure stdlib (uses subprocess for git, no third-party deps); runs under any python3.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# the guarded state file (resolved, absolute) -> the CLI allowed to change it.
# Keyed by the REPO's actual file, not a bare name, so we lock only this repo's
# state — an unrelated file that happens to be named progress.txt elsewhere
# (e.g. /tmp, an additional working dir) is NOT blocked.
GUARDED = {
    (REPO / "task_list.json").resolve(): "python3 cli/task.py",
    (REPO / "progress.txt").resolve(): "python3 cli/log.py",
}


def _git(args: list[str]) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def _is_primary_checkout() -> bool:
    """True iff REPO is the PRIMARY checkout, not a linked (task) worktree -- a
    linked worktree's `.git` is a file pointing elsewhere, so its --git-dir differs
    from the shared --git-common-dir; the primary's are identical."""
    return _git(["rev-parse", "--git-dir"]) == _git(["rev-parse", "--git-common-dir"])


def _is_gitignored(path: Path) -> bool:
    return subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=REPO).returncode == 0


def main() -> int:
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
        if os.environ.get("ALLOW_STATE_EDIT") == "1":
            return 0
        print(
            f"Blocked: change {target.name} only via `{cli}` (it enforces the state "
            "invariants). Human break-glass: ALLOW_STATE_EDIT=1.",
            file=sys.stderr,
        )
        return 2

    if os.environ.get("ALLOW_MAIN_EDIT") == "1":
        return 0
    try:
        target.relative_to(REPO)
    except ValueError:
        return 0  # outside this repo entirely -> not ours to guard
    if not _is_primary_checkout():
        return 0  # a task's worktree -- coding here is the whole point
    if _is_gitignored(target):
        return 0  # untracked local file (e.g. CLAUDE.md) -- no gate applies

    print(
        f"Blocked: {target.relative_to(REPO)} is a tracked file in the PRIMARY "
        "checkout. Real changes go through a task's dedicated worktree "
        "(`task.py begin tN`), committed there, landed via `land.py`. "
        "Human break-glass: ALLOW_MAIN_EDIT=1.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
