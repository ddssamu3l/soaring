#!/usr/bin/env python3
"""
land.py -- the serialized merge queue: the only path from a feature branch to main.

Work is sequential, branch-per-task in ONE checkout (no worktrees while single-writer).
A file lock still serializes landings, so if two ever run at once the second blocks
until the first finishes and main only moves one merge at a time. Each landing:

    1. acquire the queue lock              (only one landing in flight)
    2. require the checkout on `main`, clean
    3. merge the feature branch --no-ff    (conflicts surface HERE, not silently)
       (the pre-merge-commit hook refuses merges into main unless LAND_ACTIVE=1,
        which we set below — so this is the ONLY sanctioned door to main)
    4. run check_all on the merged result  ("main moved under me" is caught now)
    4b. re-check test-coupling + exempt-guard on the merge delta (catches a commit
        that bypassed the pre-commit hook with --no-verify)
    5. run the AI review on exactly what this landing adds
    6. all green -> keep the merge, best-effort push, and mark the task done
       any red   -> roll main back, report why; fix on the branch and re-land

The task is marked done automatically: the branch name `feature/<taskid>-<slug>`
binds the task, so `land feature/t1-dataset` runs `task.py done t1` on success
(override with --task tN). Best-effort -- a bookkeeping mismatch never undoes a merge.

Usage (run from the checkout, on a clean main):
    python scripts/land.py feature/t1-dataset      # derives + marks t1 done
    python scripts/land.py my-branch --task t1      # explicit task binding

The loop: task.py start -> git checkout -b feature/<taskid>-<slug> -> commit
(pre-commit runs check_all) -> git checkout main -> land.py.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import subprocess
import sys
from pathlib import Path

import check_all  # sibling in scripts/ (on sys.path); shares the policy logic

REPO = Path(__file__).resolve().parent.parent


def _common_git_dir() -> Path:
    """The SHARED .git dir (identical across all linked worktrees). In a linked
    worktree REPO/.git is a *file* pointing here, not a directory — so we must derive
    the real common dir rather than assume REPO/.git. Putting the queue lock here also
    makes it shared, so concurrent lands from different worktrees actually serialize
    (the whole point of the flock)."""
    r = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"], cwd=REPO, capture_output=True, text=True
    )
    p = Path(r.stdout.strip())
    return p if p.is_absolute() else (REPO / p)


LOCK = _common_git_dir() / "queue.lock"
PY = REPO / ".venv" / "bin" / "python"


def _task_from_branch(branch: str) -> str | None:
    """Derive the task_list task id from the branch name. Convention:
    `feature/<taskid>-<slug>` (e.g. feature/t1-dataset -> t1). This is what ties a
    task to a branch — no separate mapping to remember."""
    m = re.search(r"(?:^|/)(t\d+)(?:-|$)", branch)
    return m.group(1) if m else None


def git(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=check)


def out(args: list[str]) -> str:
    return git(args).stdout.strip()


def fail(msg: str) -> int:
    print(f"\033[31mland: {msg}\033[0m", file=sys.stderr)
    return 1


def land(branch: str, task: str | None = None) -> int:
    # task binding: explicit --task wins; else derive it from the branch name.
    task = task or _task_from_branch(branch)
    if task:
        print(f"task binding: {task} (will be marked done on success)")

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

    # 6b. mark the task done programmatically — the state update is a SIDE EFFECT
    #     of landing, not a thing the agent has to remember. Best-effort: a
    #     bookkeeping mismatch (e.g. the task wasn't `start`ed) must NEVER undo a
    #     good merge, so we warn instead of failing.
    if task:
        merged = out(["rev-parse", "HEAD"])
        r = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "task.py"), "done", task, "--commit", merged],
            cwd=REPO,
        )
        if r.returncode != 0:
            print(f"⚠️  merged, but couldn't mark {task} done — fix via scripts/task.py.")

    print(f"\n\033[32mLANDED\033[0m {branch} → main  ({out(['rev-parse', '--short', 'HEAD'])})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="land.py")
    ap.add_argument("branch", help="feature branch to land into main")
    ap.add_argument(
        "--task",
        help="task_list.json task id to mark done on a successful land (e.g. t2)",
    )
    args = ap.parse_args()

    # Authorize our own merge past the pre-merge-commit guard (.githooks/pre-merge-commit),
    # which refuses any un-sanctioned `git merge` into main. Set process-wide so every
    # child git call (the --no-ff merge in particular) inherits it.
    os.environ["LAND_ACTIVE"] = "1"

    LOCK.parent.mkdir(exist_ok=True)
    with open(LOCK, "w") as lockf:
        print("acquiring merge-queue lock …")
        fcntl.flock(lockf, fcntl.LOCK_EX)  # blocks → serializes concurrent lands
        print("lock acquired.\n")
        return land(args.branch, args.task)


if __name__ == "__main__":
    sys.exit(main())
