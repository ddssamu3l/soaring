#!/usr/bin/env python3
"""
land.py -- the serialized merge queue. The third pillar: many agents work in
isolated worktrees; landing to main goes through here, ONE AT A TIME.

A file lock guarantees serialization: if two agents call `land` at once, the
second blocks until the first finishes, so main only ever moves one merge at a
time. Each landing:

    1. acquire the queue lock            (only one landing in flight)
    2. require primary worktree on main, clean
    3. merge the feature branch into main  (conflicts surface HERE, not silently)
    4. run check_all on the merged result  ("main moved under me" is caught now)
    5. run the AI review on the merge
    6. all green  -> keep the merge (+ best-effort push)
       any red    -> roll main back, report why; the agent fixes and re-lands

Usage (run from the primary worktree, on a clean main):
    python scripts/land.py feature/dataset-logger

Isolate during work (worktrees), serialize at integration (this). To make a
worktree for an agent:
    git worktree add ../soaring-<name> -b feature/<name>
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from pathlib import Path

import check_all  # sibling in scripts/ (on sys.path); shares the policy logic

REPO = Path(__file__).resolve().parent.parent
LOCK = REPO / ".git" / "queue.lock"
PY = REPO / ".venv" / "bin" / "python"


def git(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=check)


def out(args: list[str]) -> str:
    return git(args).stdout.strip()


def fail(msg: str) -> int:
    print(f"\033[31mland: {msg}\033[0m", file=sys.stderr)
    return 1


def land(branch: str) -> int:
    # 2. preconditions -----------------------------------------------------
    if out(["rev-parse", "--abbrev-ref", "HEAD"]) != "main":
        return fail("primary worktree must be on `main` to land.")
    if git(["status", "--porcelain"]).stdout.strip():
        return fail("main worktree is dirty — commit/stash first (won't stomp your work).")
    if git(["rev-parse", "--verify", branch], check=False).returncode != 0:
        return fail(f"branch `{branch}` does not exist.")

    before = out(["rev-parse", "HEAD"])

    # (best-effort) catch up to the remote so we integrate on top of latest main
    if git(["remote"], check=False).stdout.strip():
        git(["fetch", "origin", "main"], check=False)
        git(["merge", "--ff-only", "origin/main"], check=False)
        before = out(["rev-parse", "HEAD"])

    # 3. merge — conflicts surface here ------------------------------------
    print(f"merging {branch} → main …")
    merge = git(["merge", "--no-ff", "--no-edit", branch], check=False)
    if merge.returncode != 0:
        git(["merge", "--abort"], check=False)
        return fail(f"merge conflict with `{branch}`. Rebase it on main and retry.\n{merge.stdout}")

    def rollback(reason: str) -> int:
        git(["reset", "--hard", before], check=False)
        return fail(f"{reason}\nmain rolled back to {before[:8]}. Fix on `{branch}`, then re-land.")

    # 4. deterministic gate on the merged result ---------------------------
    print("\n── check_all on merged main ──")
    if subprocess.run([str(PY), "scripts/check_all.py"], cwd=REPO).returncode != 0:
        return rollback("check_all FAILED on the merged result.")

    # 4b. re-enforce the commit-time policy on the MERGE DELTA. check_all's
    #     coupling/exempt gates key off the staged set, which is empty here — so a
    #     worktree commit that bypassed the hook (--no-verify) would slip through.
    #     Re-check against `before...HEAD` to close that hole.
    delta = {
        ln.strip()
        for ln in git(["diff", f"{before}...HEAD", "--name-only"]).stdout.splitlines()
        if ln.strip()
    }
    coup = check_all.coupling_violations(delta)
    if coup and os.environ.get("ALLOW_NO_TEST_UPDATE") != "1":
        return rollback(
            "test-coupling FAILED on the merge (edited code, untouched tests):\n  "
            + "\n  ".join(coup)
        )
    added_ex = check_all.exemptions_added(delta, before)
    if added_ex and os.environ.get("ALLOW_EXEMPT") != "1":
        return rollback(
            "merge adds test exemptions without sign-off (re-land with ALLOW_EXEMPT=1):\n  "
            + "\n  ".join(sorted(added_ex))
        )

    # 5. AI review of exactly what this landing adds -----------------------
    print("\n── AI review ──")
    if subprocess.run([str(PY), "scripts/review.py", "--base", before], cwd=REPO).returncode != 0:
        return rollback("review BLOCKED the merge.")

    # 6. success -----------------------------------------------------------
    if git(["remote"], check=False).stdout.strip():
        push = git(["push", "origin", "main"], check=False)
        if push.returncode != 0:
            print(f"⚠️  landed locally but push failed:\n{push.stderr.strip()}")
    print(f"\n\033[32mLANDED\033[0m {branch} → main  ({out(['rev-parse', '--short', 'HEAD'])})")
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        return fail("usage: python scripts/land.py <branch>")
    branch = sys.argv[1]

    LOCK.parent.mkdir(exist_ok=True)
    with open(LOCK, "w") as lockf:
        print("acquiring merge-queue lock …")
        fcntl.flock(lockf, fcntl.LOCK_EX)  # blocks → serializes concurrent lands
        print("lock acquired.\n")
        return land(branch)


if __name__ == "__main__":
    sys.exit(main())
