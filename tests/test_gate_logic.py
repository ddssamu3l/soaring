"""Regression guards for the environment gate itself.

The gate is the one thing agents can't route around, so the gate's own logic gets
tests. These lock in the three fixes from the worktree/merge-queue audit:

  1. the pre-commit hook must resolve its interpreter from the SHARED repo, not a
     worktree-relative `.venv/bin/python` that doesn't exist in a linked worktree;
  2. coupling logic is reusable (so land.py can re-check the merge delta);
  3. review.py caps `claude` with a timeout (so a hung review can't hold the lock).
"""

from __future__ import annotations

from pathlib import Path

import check_all

REPO = Path(__file__).resolve().parent.parent


# --- coupling logic (shared by the commit gate + land.py) ----------------
def test_coupling_flags_source_without_its_test() -> None:
    # glider_sim.py has tests/test_glider_sim.py, but it isn't in the changed set.
    assert check_all.coupling_violations({"glider_sim.py"})


def test_coupling_passes_when_test_touched() -> None:
    assert not check_all.coupling_violations({"glider_sim.py", "tests/test_glider_sim.py"})


def test_coupling_ignores_exempt_source() -> None:
    # fly.py is exempt in .test-exempt → never a coupling offender.
    assert not check_all.coupling_violations({"fly.py"})


def test_coupling_ignores_nonpython_and_test_files() -> None:
    assert not check_all.coupling_violations({"README.md", "tests/test_glider_sim.py"})


# --- Fix 1: the hook must survive linked worktrees -----------------------
def test_hook_resolves_interpreter_from_shared_repo() -> None:
    hook = (REPO / ".githooks" / "pre-commit").read_text()
    # the worktree-breaking pattern must be gone ...
    assert "exec .venv/bin/python" not in hook, "hook uses a worktree-relative venv path"
    # ... and the interpreter must be resolved via the shared git dir.
    assert "git-common-dir" in hook, "hook no longer resolves the shared repo root"


# --- Fix 3: review must bound the reviewer call --------------------------
def test_review_caps_claude_with_a_timeout() -> None:
    review = (REPO / "cli" / "review.py").read_text()
    assert "timeout=CLAUDE_TIMEOUT" in review, "review.py must pass a timeout to claude"


# --- Merge guard: land.py must be the ONLY door to main ------------------
def test_pre_merge_commit_hook_guards_main() -> None:
    # A raw `git merge` into main bypasses the judge; the hook must refuse it unless
    # the merge came from land.py (LAND_ACTIVE) or a human break-glass.
    hook_path = REPO / ".githooks" / "pre-merge-commit"
    assert hook_path.exists(), "pre-merge-commit hook missing — main un-guarded against raw merges"
    hook = hook_path.read_text()
    assert 'BRANCH" = "main"' in hook, "hook must only guard merges into main"
    assert "LAND_ACTIVE" in hook, "hook must recognize land.py's sentinel"
    assert "exit 1" in hook, "hook must actually refuse (non-zero exit) an un-sanctioned merge"


def test_land_sets_the_merge_sentinel() -> None:
    # land.py's own merge has to pass the pre-merge-commit guard, so it must set the
    # sentinel the hook checks for.
    land = (REPO / "cli" / "land.py").read_text()
    assert 'LAND_ACTIVE"] = "1"' in land, "land.py must set LAND_ACTIVE so its merge passes"


# --- land.py must survive running from a task's dedicated worktree -------
# (worktree-per-task is the standing default; land.py itself runs from the
# primary checkout, but a bare `REPO / ".venv"` broke the moment ANYONE ran it
# from a fresh worktree with no .venv of its own — caught landing t10.)
def test_land_resolves_interpreter_from_shared_repo() -> None:
    land = (REPO / "cli" / "land.py").read_text()
    assert 'PY = REPO / ".venv"' not in land, "PY must not be worktree-relative"
    assert "_common_git_dir().parent" in land, "PY must resolve via the shared repo root"


def test_land_rolls_back_on_an_unexpected_crash() -> None:
    # A merge that lands on main but never gets gated (because land.py crashed, not
    # because a gate failed) is worse than a loud failure — the post-merge gate/review
    # steps must be wrapped so ANY exception still triggers rollback().
    land = (REPO / "cli" / "land.py").read_text()
    assert "except Exception" in land, "post-merge steps must catch unexpected crashes too"


# --- land.py must commit its own done-marking, not leave it dirty --------
# (task.py's `done` command only rewrites task_list.json on disk, it never commits.
# Left as-is, every land ends with a dirty working tree that someone has to remember
# to commit — caught when t10's done-mark got stranded uncommitted inside a
# concurrent session's WIP checkout instead of ever reaching a commit.)
def test_land_commits_the_done_mark_itself() -> None:
    land = (REPO / "cli" / "land.py").read_text()
    assert "task: mark {task} done" in land, "land.py must commit task_list.json's done-mark"
