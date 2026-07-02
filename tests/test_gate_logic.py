"""Regression guards for the environment gate itself.

The gate is the one thing agents can't route around, so the gate's own logic gets
tests. These lock in the three fixes from the worktree/merge-queue audit:

  1. the pre-commit hook must resolve its interpreter from the SHARED repo, not a
     worktree-relative `.venv/bin/python` that doesn't exist in a linked worktree;
  2. coupling logic is reusable (so land.py can re-check the merge delta);
  3. review.py caps `claude` with a timeout (so a hung review can't hold the lock).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import check_all
import pytest

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


# --- land.py must authorize its own done-mark commit -----------------------
# (the pre-commit hook refuses any direct `git commit` on main unless
# ALLOW_MAIN_COMMIT=1. land.py's done-mark commit runs ON main, so without this it
# is silently refused -- best-effort means the land still reports LANDED, so the
# failure is invisible unless you go looking. Caught landing the fix above: the very
# first real land under it hit exactly this and reproduced the bug it was fixing.)
def test_land_authorizes_its_own_done_mark_commit() -> None:
    land = (REPO / "cli" / "land.py").read_text()
    assert 'ALLOW_MAIN_COMMIT"] = "1"' in land, "land.py must self-authorize its main commit"


# --- land.py reconciles a dirty task_list.json/progress.txt instead of refusing --
# (task.py's start/block/notes/done and log.py's appends all rewrite these two files
# on disk without committing -- that's land.py's job. Hard-refusing on ANY dirty file
# forced a manual discard/reset before every land, easy to forget -- hit for real
# landing t18/t19. These two files are the only ones this scaffold's own tools ever
# leave dirty, so folding them into a commit here is no riskier than the existing
# done-mark commit. Static assertions -- `_reconcile_dirty_state` shells real git
# against the actual REPO constant, so it can't be exercised live in a test without
# risking the exact corruption class documented for task.py's own git-touching tests.)
def test_land_reconciles_dirty_task_state_before_landing() -> None:
    land = (REPO / "cli" / "land.py").read_text()
    assert 'RECONCILABLE = {"task_list.json", "progress.txt"}' in land
    assert "_reconcile_dirty_state" in land, "land.py must reconcile, not just refuse"


def test_land_still_refuses_other_dirty_files() -> None:
    land = (REPO / "cli" / "land.py").read_text()
    assert "dirty <= RECONCILABLE" in land, "only task_list.json/progress.txt may auto-reconcile"


# --- ...but never auto-commits a DELETED state file (t20 review follow-up) -------
def test_land_reconcile_refuses_a_deleted_state_file() -> None:
    land = (REPO / "cli" / "land.py").read_text()
    assert '"D" in ln[:2]' in land, "a deleted task_list.json/progress.txt must not auto-reconcile"


# --- task.py add must allocate ids off main, not a stale local copy --------
# (worktree-per-task means every worktree forks its own snapshot of task_list.json;
# `_next_id()` used to read that local copy, so two worktrees adding a task around
# the same time independently computed the same next id -- hit for real: two live
# sessions both landed a task called t11. `add` must instead resolve the next id
# from main's actual tip, so a second worktree's call sees a first worktree's
# addition even though its OWN local file never changed.
#
# Static assertions, not a live git-executing integration test: a scratch-repo
# version of this test proved to be a real hazard -- worktree/subprocess
# interaction under pytest repeatedly resolved onto THIS repo's own main instead
# of the isolated scratch repo, corrupting it several times despite three rounds
# of hardening (-C pinning, a toplevel guard, --detach worktrees). Not worth the
# risk for a `cli/` file that's exempt from needing a test at all -- this checks
# the same properties by reading the source, matching every other test in this
# file that touches land.py/task.py.)
def test_task_add_resolves_ids_from_main_not_local_copy() -> None:
    task = (REPO / "cli" / "task.py").read_text()
    assert "_load_from_main" in task, "add must read task_list.json off main's tip"
    assert '"git", "show"' in task, "id allocation must read via git show, not the local file"


def test_task_add_serializes_on_lands_lock() -> None:
    task = (REPO / "cli" / "task.py").read_text()
    land = (REPO / "cli" / "land.py").read_text()
    assert 'LOCK = _common_git_dir() / "queue.lock"' in task
    assert 'LOCK = _common_git_dir() / "queue.lock"' in land, "add and land must share one lock"
    assert "fcntl.flock(lockf, fcntl.LOCK_EX)" in task


def test_task_add_commits_via_plumbing_not_git_commit() -> None:
    # commit-tree/update-ref never run hooks, so this needs no ALLOW_MAIN_COMMIT and
    # never touches the calling checkout's HEAD -- unlike a plain `git commit`.
    task = (REPO / "cli" / "task.py").read_text()
    assert '"commit-tree"' in task
    assert '"update-ref", "refs/heads/main"' in task
    assert '"git", "commit"' not in task, "add must not use a plain `git commit` on main"


def test_task_add_does_not_push() -> None:
    task = (REPO / "cli" / "task.py").read_text()
    assert '"push"' not in task, "add must not publish -- that stays land.py's job"


# --- task.py list hides done tasks by default (user preference, 2026-07-02) -----
# Functional (not just static) since it's cheap and safe here: `list` never touches
# git, only reads task_list.json, so a copy of task.py in a scratch dir with a fake
# state file is fully isolated -- no worktree/git plumbing involved, unlike `add`.
def test_task_list_hides_done_by_default(tmp_path: Path) -> None:
    cli_dir = tmp_path / "cli"
    cli_dir.mkdir()
    shutil.copy(REPO / "cli" / "task.py", cli_dir / "task.py")
    state = {
        "tasks": [
            {
                "id": "t1",
                "title": "done one",
                "status": "done",
                "deps": [],
                "files": [],
                "commit": "abc123",
                "notes": "",
            },
            {
                "id": "t2",
                "title": "pending one",
                "status": "pending",
                "deps": [],
                "files": [],
                "commit": None,
                "notes": "",
            },
        ]
    }
    (tmp_path / "task_list.json").write_text(json.dumps(state))

    plain = subprocess.run(
        ["python3", "cli/task.py", "list"], cwd=tmp_path, capture_output=True, text=True
    )
    full = subprocess.run(
        ["python3", "cli/task.py", "list", "--full"], cwd=tmp_path, capture_output=True, text=True
    )
    assert "t1" not in plain.stdout, "done task t1 must be hidden by default"
    assert "t2" in plain.stdout
    assert "t1" in full.stdout, "--full must still show done tasks"
    assert "t2" in full.stdout


# --- task.py notes -- edit a task's notes without touching status (2026-07-02) ---
# Functional and safe for the same reason as `list` above: notes editing only reads
# and writes task_list.json, no git plumbing involved.
def _write_state(tmp_path: Path) -> None:
    cli_dir = tmp_path / "cli"
    cli_dir.mkdir()
    shutil.copy(REPO / "cli" / "task.py", cli_dir / "task.py")
    state = {
        "tasks": [
            {
                "id": "t1",
                "title": "some task",
                "status": "pending",
                "deps": [],
                "files": [],
                "commit": None,
                "notes": "original note",
            },
        ]
    }
    (tmp_path / "task_list.json").write_text(json.dumps(state))


def test_task_notes_set_replaces_outright(tmp_path: Path) -> None:
    _write_state(tmp_path)
    subprocess.run(
        ["python3", "cli/task.py", "notes", "t1", "--set", "replaced"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    data = json.loads((tmp_path / "task_list.json").read_text())
    assert data["tasks"][0]["notes"] == "replaced"


def test_task_notes_append_preserves_existing(tmp_path: Path) -> None:
    _write_state(tmp_path)
    subprocess.run(
        ["python3", "cli/task.py", "notes", "t1", "--append", "extra context"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    data = json.loads((tmp_path / "task_list.json").read_text())
    assert data["tasks"][0]["notes"] == "original note | extra context"


def test_task_notes_does_not_touch_status(tmp_path: Path) -> None:
    _write_state(tmp_path)
    subprocess.run(
        ["python3", "cli/task.py", "notes", "t1", "--set", "x"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    data = json.loads((tmp_path / "task_list.json").read_text())
    assert data["tasks"][0]["status"] == "pending"


def test_task_notes_missing_id_errors(tmp_path: Path) -> None:
    _write_state(tmp_path)
    r = subprocess.run(
        ["python3", "cli/task.py", "notes", "t99", "--set", "x"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0


# --- task.py done accepts pending, not just active (2026-07-02) -----------------
# `active` lives only in the primary checkout's uncommitted working tree, so it's
# easy to lose between `start` and land.py's post-merge done-mark (hit for real
# landing t18). `done` shells out to `git cat-file` to verify the commit exists --
# a real subprocess git call in a test previously inherited GIT_DIR from the
# pre-commit hook's own environment and corrupted the shared repo for real (hit
# for real writing this test the first time: primary's core.bare flipped true,
# a stray commit landed on this very branch). Fix: never shell out to git at
# all here -- import task.py as a module, monkeypatch `_git_has`, and call
# `cmd_done` in-process. Zero subprocesses, zero risk.
def _load_task_module() -> Any:
    spec = importlib.util.spec_from_file_location("task_cli_under_test", REPO / "cli" / "task.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _done_ns(tid: str) -> argparse.Namespace:
    return argparse.Namespace(id=tid, commit="deadbeef")


def _state(status: str) -> dict[str, Any]:
    return {
        "tasks": [
            {
                "id": "t1",
                "title": "some task",
                "status": status,
                "deps": [],
                "files": [],
                "commit": None,
                "notes": "",
            },
        ]
    }


def test_task_done_accepts_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_cli = _load_task_module()
    monkeypatch.setattr(task_cli, "STATE", tmp_path / "task_list.json")
    monkeypatch.setattr(task_cli, "_git_has", lambda sha: True)
    task_cli.STATE.write_text(json.dumps(_state("pending")))
    assert task_cli.cmd_done(_done_ns("t1")) == 0
    assert json.loads(task_cli.STATE.read_text())["tasks"][0]["status"] == "done"


def test_task_done_still_accepts_active(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_cli = _load_task_module()
    monkeypatch.setattr(task_cli, "STATE", tmp_path / "task_list.json")
    monkeypatch.setattr(task_cli, "_git_has", lambda sha: True)
    task_cli.STATE.write_text(json.dumps(_state("active")))
    assert task_cli.cmd_done(_done_ns("t1")) == 0


def test_task_done_rejects_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_cli = _load_task_module()
    monkeypatch.setattr(task_cli, "STATE", tmp_path / "task_list.json")
    monkeypatch.setattr(task_cli, "_git_has", lambda sha: True)
    task_cli.STATE.write_text(json.dumps(_state("blocked")))
    assert task_cli.cmd_done(_done_ns("t1")) != 0


def test_task_done_rejects_already_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_cli = _load_task_module()
    monkeypatch.setattr(task_cli, "STATE", tmp_path / "task_list.json")
    monkeypatch.setattr(task_cli, "_git_has", lambda sha: True)
    task_cli.STATE.write_text(json.dumps(_state("done")))
    assert task_cli.cmd_done(_done_ns("t1")) != 0


# --- task.py begin -- start + derive a slug + git worktree add, one step (2026-07-02) --
# `subprocess.run` is monkeypatched to a recorder rather than allowed to run for real:
# `git worktree add` against the actual REPO would create a real worktree as a test
# side effect -- not the corruption-class hazard from earlier (no GIT_DIR inheritance
# risk, it's a real intended operation, not an isolated-tmp-dir one), but still not
# something a test run should do to the real repo. Verifying the exact command is
# built correctly is the same coverage without the side effect.
def _begin_ns(tid: str) -> argparse.Namespace:
    return argparse.Namespace(id=tid)


def test_task_slug_derivation() -> None:
    task_cli = _load_task_module()
    assert task_cli._slug("Fix Some Thing!!") == "fix-some-thing"
    assert task_cli._slug("  leading/trailing -- spaces  ") == "leading-trailing-spaces"
    assert task_cli._slug("") == "task"
    assert not task_cli._slug("x" * 100).endswith("-")


def test_task_begin_starts_then_creates_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_cli = _load_task_module()
    monkeypatch.setattr(task_cli, "STATE", tmp_path / "task_list.json")
    task_cli.STATE.write_text(json.dumps(_state("pending")))

    calls = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(task_cli.subprocess, "run", fake_run)

    assert task_cli.cmd_begin(_begin_ns("t1")) == 0
    assert json.loads(task_cli.STATE.read_text())["tasks"][0]["status"] == "active"
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[:3] == ["git", "worktree", "add"]
    assert cmd[3] == "../soaring-t1"
    assert cmd[4] == "-b"
    assert cmd[5] == "feature/t1-some-task"
    assert cmd[6] == "main"


def test_task_begin_refuses_if_start_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "blocked" tasks CAN be resumed (start accepts pending or blocked) -- use
    # "done" instead, the actual case start refuses.
    task_cli = _load_task_module()
    monkeypatch.setattr(task_cli, "STATE", tmp_path / "task_list.json")
    task_cli.STATE.write_text(json.dumps(_state("done")))

    calls: list[list[str]] = []
    monkeypatch.setattr(task_cli.subprocess, "run", lambda cmd, **kw: calls.append(cmd))

    assert task_cli.cmd_begin(_begin_ns("t1")) != 0
    assert not calls, "must not touch git if start itself refused"
