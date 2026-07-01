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
    review = (REPO / "scripts" / "review.py").read_text()
    assert "timeout=CLAUDE_TIMEOUT" in review, "review.py must pass a timeout to claude"
