#!/usr/bin/env python3
"""task.py -- the ONLY sanctioned way to mutate task_list.json.

Free-editing structured state with an LLM eventually corrupts it: malformed
JSON, two tasks marked active, deps skipped, a task "done" with no real commit.
So we do the same thing we do for code -- don't trust the agent to hold the
invariant, build a tool that enforces it. A PreToolUse hook (guard_state.py)
blocks direct Edit/Write of task_list.json, so this CLI is the only door in.

Pure stdlib on purpose: hooks and land.py call it, and it must never depend on
the project venv.

`add` allocates ids from `main`'s actual tip, not this checkout's local copy --
worktree-per-task means every worktree forks its own stale snapshot, so two
sessions adding a task around the same time used to compute the same next id
(hit for real: two sessions both got t11). `add` now serializes on the SAME
lock `land.py` uses and commits the claim straight onto `refs/heads/main` (git
plumbing, no `git commit` -- doesn't touch this checkout's HEAD, doesn't need
`ALLOW_MAIN_COMMIT`, doesn't push). Falls back to a local-only add if `main`
can't be resolved (fresh/bootstrap repo) so `add` never hard-fails on that.

`done` accepts an `active` OR `pending` task -- `active` lives only in the local
checkout's uncommitted working tree, so it's easy to lose between `start` and
`land.py`'s post-merge done-mark (hit for real landing t18). land.py's
branch-name-derived task binding is the real proof a task landed, not that
fragile flag. `blocked`/`done` still refuse (no silent unblock/double-mark).

`begin` is `start` + a derived-from-the-title slug + `git worktree add`, one step
instead of three -- a subprocess can't `cd` the calling shell, so it prints the
`cd` to run next rather than actually landing you there. `start` alone still
exists for the rare off-convention case (no worktree wanted yet).

Commands:
    task.py add   --title T [--deps a,b] [--files "a.py;b.py"] [--notes N]
    task.py start <id>              # -> active   (refuses if another is active)
    task.py begin <id>              # start + create ../soaring-<id> worktree, prints `cd`
    task.py done  <id> --commit SHA # -> done     (from active OR pending; SHA must exist in git)
    task.py block <id> --reason R   # -> blocked
    task.py notes <id> (--set T | --append T)  # edit notes without touching status
    task.py list                    # status board
    task.py next                    # the next pickable task
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
STATE = REPO / "task_list.json"


def _common_git_dir() -> Path:
    """The SHARED .git dir across all linked worktrees -- duplicated from land.py
    (not imported) so task.py stays dependency-free of check_all/the venv."""
    r = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"], cwd=REPO, capture_output=True, text=True
    )
    p = Path(r.stdout.strip())
    return p if p.is_absolute() else (REPO / p)


LOCK = _common_git_dir() / "queue.lock"  # same lock land.py uses -- one main-mutation at a time


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


def _next_pickable(data: dict[str, Any]) -> dict[str, Any] | None:
    """The next pickable task: the first pending task whose deps are all done.
    The SINGLE source of the pickability rule — both `list` (its `← next` marker)
    and `next` read it, so the board and `task.py next` can never diverge."""
    done_ids = {t["id"] for t in data["tasks"] if t["status"] == "done"}
    return next(
        (t for t in data["tasks"] if t["status"] == "pending" and set(t["deps"]) <= done_ids),
        None,
    )


def _git_has(sha: str) -> bool:
    return (
        subprocess.run(["git", "cat-file", "-e", sha], cwd=REPO, capture_output=True).returncode
        == 0
    )


def _err(msg: str) -> int:
    print(f"task: {msg}", file=sys.stderr)
    return 1


def _load_from_main() -> tuple[dict[str, Any], str] | None:
    """task_list.json off main's actual tip -- the canonical source for id
    allocation, not this checkout's possibly-stale local copy. None if `main`
    can't be resolved (fresh/bootstrap repo); caller falls back to `_load()`."""
    sha = subprocess.run(
        ["git", "rev-parse", "main"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    if not sha:
        return None
    show = subprocess.run(
        ["git", "show", f"{sha}:task_list.json"], cwd=REPO, capture_output=True, text=True
    )
    if show.returncode != 0:
        return None
    return json.loads(show.stdout), sha


def _commit_onto_main(data: dict[str, Any], main_sha: str, message: str) -> str | None:
    """Commit `data` as task_list.json directly onto refs/heads/main via plumbing
    (hash-object / read-tree / write-tree / commit-tree), never touching this
    checkout's HEAD or working tree. Plumbing doesn't run hooks, so this needs
    no ALLOW_MAIN_COMMIT -- it's a structured state update through its own
    sanctioned CLI, the same reasoning task_list.json's edit-lock already rests
    on. Returns the new sha, or None on any failure (best-effort: the caller
    falls back to a local-only add rather than hard-failing)."""
    blob = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=REPO,
        input=json.dumps(data, indent=2) + "\n",
        capture_output=True,
        text=True,
    )
    if blob.returncode != 0:
        return None
    blob_sha = blob.stdout.strip()

    with tempfile.TemporaryDirectory() as tmp:
        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = str(Path(tmp) / "index")
        if subprocess.run(["git", "read-tree", main_sha], cwd=REPO, env=env).returncode != 0:
            return None
        upd = subprocess.run(
            ["git", "update-index", "--add", "--cacheinfo", f"100644,{blob_sha},task_list.json"],
            cwd=REPO,
            env=env,
        )
        if upd.returncode != 0:
            return None
        tree = subprocess.run(
            ["git", "write-tree"], cwd=REPO, env=env, capture_output=True, text=True
        )
        if tree.returncode != 0:
            return None
        tree_sha = tree.stdout.strip()

    commit = subprocess.run(
        ["git", "commit-tree", tree_sha, "-p", main_sha, "-m", message],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if commit.returncode != 0:
        return None
    new_sha = commit.stdout.strip()

    # Atomic compare-and-swap: refuses if main moved since main_sha was read. Shouldn't
    # happen -- we hold LOCK, which land.py also acquires -- but never clobber blind.
    cas = subprocess.run(["git", "update-ref", "refs/heads/main", new_sha, main_sha], cwd=REPO)
    return new_sha if cas.returncode == 0 else None


def cmd_add(a: argparse.Namespace) -> int:
    deps = [d for d in (a.deps.split(",") if a.deps else []) if d]
    files = [f for f in (a.files.split(";") if a.files else []) if f]

    LOCK.parent.mkdir(exist_ok=True)
    with open(LOCK, "w") as lockf:
        # Serializes with land.py (same lock) and any concurrent `add`, so id
        # allocation can never race between "read the next id" and "claim it".
        fcntl.flock(lockf, fcntl.LOCK_EX)

        canonical = _load_from_main()
        ref_data = canonical[0] if canonical else _load()
        for d in deps:
            if not _find(ref_data, d):
                return _err(f"dep {d!r} does not exist")
        tid = _next_id(ref_data)
        entry = {
            "id": tid,
            "title": a.title,
            "status": "pending",
            "deps": deps,
            "files": files,
            "commit": None,
            "notes": a.notes or "",
        }

        if canonical:
            main_data, main_sha = canonical
            main_data["tasks"].append(entry)
            if _commit_onto_main(main_data, main_sha, f"task: add {tid} — {a.title}") is None:
                print(
                    f"⚠️  couldn't commit {tid}'s id claim onto main "
                    "(added locally only) — sync manually if this recurs.",
                    file=sys.stderr,
                )

    # Sync THIS checkout's own view: keep whatever local state already diverged (e.g.
    # this worktree's own task mid-flight) and layer the new entry on top of it.
    local_data = _load()
    local_data["tasks"].append(entry)
    _save(local_data)
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


def _slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return (s[:40].rstrip("-")) or "task"


def cmd_begin(a: argparse.Namespace) -> int:
    """`start` + derive-a-slug + `git worktree add`, in one step -- a subprocess
    can't `cd` the parent shell, so this can't teleport you into the worktree, but
    it collapses the previous 3 hand-typed steps (start, hand-derive a slug, git
    worktree add) down to "run this, then cd"."""
    rc = cmd_start(a)
    if rc != 0:
        return rc
    t = _find(_load(), a.id)
    assert t is not None  # cmd_start just verified it exists and is now active
    branch = f"feature/{a.id}-{_slug(t['title'])}"
    worktree = f"../soaring-{a.id}"
    wt = subprocess.run(["git", "worktree", "add", worktree, "-b", branch, "main"], cwd=REPO)
    if wt.returncode != 0:
        return _err(
            f"claimed {a.id} but worktree creation failed -- create it by hand: "
            f"git worktree add {worktree} -b {branch} main"
        )
    print(f"\nnext: cd {worktree}")
    return 0


def cmd_done(a: argparse.Namespace) -> int:
    data = _load()
    t = _find(data, a.id)
    if not t:
        return _err(f"{a.id} not found")
    # `active` is the common case, but under worktree-per-task it's an ephemeral
    # local flag (never committed) -- ordinary git operations on the primary
    # checkout (a stray `git checkout`, a dirty-tree reset before landing) can
    # wipe it between `start` and `land.py`'s post-merge done-mark. Accepting
    # `pending` too closes that gap: land.py's branch-name-derived task binding
    # is the real proof this task legitimately landed, not the fragile flag.
    # `blocked`/`done` still rejected -- no silent unblock, no double-mark.
    if t["status"] not in ("active", "pending"):
        return _err(f"{a.id} is {t['status']}, cannot be marked done")
    if not _git_has(a.commit):
        return _err(f"commit {a.commit!r} not in git — a real landed commit is required")
    t["status"] = "done"
    t["commit"] = a.commit
    _save(data)
    print(f"done {a.id} @ {a.commit[:8]}")
    return 0


def cmd_notes(a: argparse.Namespace) -> int:
    data = _load()
    t = _find(data, a.id)
    if not t:
        return _err(f"{a.id} not found")
    t["notes"] = a.set if a.set is not None else (t["notes"] + f" | {a.append}").strip(" |")
    _save(data)
    print(f"notes {a.id}: {t['notes']}")
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


_ST_GLYPH = {"done": "✓", "active": "▶", "blocked": "✗", "pending": "○"}
_ST_CODE = {"done": "32", "active": "1;36", "blocked": "31", "pending": "2"}


def cmd_list(a: argparse.Namespace) -> int:
    data = _load()
    tasks = data["tasks"]
    if not tasks:
        print("(no tasks)")
        return 0

    full = getattr(a, "full", False)
    color = sys.stdout.isatty()  # tint for humans; stay plain in agent/piped output

    def paint(s: str, code: str) -> str:
        return f"\033[{code}m{s}\033[0m" if color else s

    ind = " " * 8
    wrapw = max(40, min(shutil.get_terminal_size((100, 24)).columns, 100)) - len(ind)

    def wrap(text: str) -> str:
        return textwrap.fill(text, width=wrapw, initial_indent=ind, subsequent_indent=ind)

    total = len(tasks)
    counts = {s: sum(1 for t in tasks if t["status"] == s) for s in _ST_CODE}
    done_ids = {t["id"] for t in tasks if t["status"] == "done"}
    nxt = _next_pickable(data)  # same rule as `task.py next` — one source, no drift
    next_id = nxt["id"] if nxt else None

    # header: a progress bar + status counts, each tinted by its status colour
    barw = 12
    filled = round(counts["done"] / total * barw)
    bar = paint("█" * filled, "32") + paint("░" * (barw - filled), "2")
    summary = "  ".join(
        paint(f"{counts[s]} {s}", _ST_CODE[s]) for s in ("done", "active", "blocked", "pending")
    )
    print(f"\n{paint('soaring — task board', '1')}")
    print(f"  {bar}  {counts['done']}/{total}    {summary}\n")

    if full and data.get("note"):
        frame = textwrap.fill(
            data["note"], width=wrapw + len(ind), initial_indent="  ", subsequent_indent="  "
        )
        print(paint(frame, "2"), end="\n\n")

    # done tasks only grow the list as the roadmap progresses; hide them from the
    # default view so it stays a "what's left" board, not a full history (git +
    # progress.txt already are that). -v/--full brings them back.
    shown = tasks if full else [t for t in tasks if t["status"] != "done"]
    if not shown:
        print(paint("  (everything done — pass -v/--full to see it)", "2"))
        print()
        return 0

    width = min(max(len(t["title"]) for t in shown), 52)
    for t in shown:
        st = t["status"]
        title = t["title"]
        title = (title[: width - 1] + "…") if len(title) > width else title.ljust(width)
        unmet = [d for d in t["deps"] if d not in done_ids]
        dim = False
        if st == "done":
            note = paint(f"@{t['commit'][:8]}" if t["commit"] else "done", "2")
        elif st == "active":
            note = paint("← ACTIVE", "1;36")
        elif st == "blocked":
            reason = t["notes"].split("BLOCKED:", 1)[-1].strip() if "BLOCKED:" in t["notes"] else ""
            note = paint("BLOCKED" + (f": {reason}" if reason else ""), "31")
        elif t["id"] == next_id:
            note = paint("← next", "1;32")
        elif unmet:
            note, dim = paint("needs " + ", ".join(unmet), "2"), True
        else:
            note = paint("ready", "2")
        gly = paint(_ST_GLYPH[st], _ST_CODE[st])
        tid = paint(t["id"].rjust(3), "1")
        cell = paint(title, "2") if dim else (paint(title, "1;36") if st == "active" else title)
        print(f"  {gly} {tid}  {cell}  {note}")

        if full:
            meta = []
            if t["deps"]:
                meta.append("deps: " + ", ".join(t["deps"]))
            meta.append("files: " + (", ".join(t["files"]) if t["files"] else "—"))
            print(paint(ind + "   ·   ".join(meta), "2"))
            if t["notes"]:
                print(paint(wrap(t["notes"]), "2"))
            print()

    # compact view: surface just the active task's plan; --full already shows every note
    active = next((t for t in tasks if t["status"] == "active"), None)
    if not full and active and active["notes"]:
        plan = active["notes"]
        plan = (plan[:75] + "…") if len(plan) > 76 else plan
        print(f"\n  {paint('↳ ' + active['id'] + ' plan:', '1;36')} {paint(plan, '2')}")
    print()
    return 0


def cmd_next(_: argparse.Namespace) -> int:
    data = _load()
    active = _find_by_status(data, "active")
    if active:
        print(f"active: {active['id']}: {active['title']}")
        return 0
    t = _next_pickable(data)
    if t:
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

    p_begin = sub.add_parser("begin")
    p_begin.add_argument("id")
    p_begin.set_defaults(fn=cmd_begin)

    p_done = sub.add_parser("done")
    p_done.add_argument("id")
    p_done.add_argument("--commit", required=True)
    p_done.set_defaults(fn=cmd_done)

    p_block = sub.add_parser("block")
    p_block.add_argument("id")
    p_block.add_argument("--reason", required=True)
    p_block.set_defaults(fn=cmd_block)

    p_notes = sub.add_parser("notes")
    p_notes.add_argument("id")
    g_notes = p_notes.add_mutually_exclusive_group(required=True)
    g_notes.add_argument("--set", help="replace the notes outright")
    g_notes.add_argument("--append", help="append, pipe-separated (same style as block's reason)")
    p_notes.set_defaults(fn=cmd_notes)

    p_list = sub.add_parser("list")
    p_list.add_argument(
        "-v",
        "--full",
        action="store_true",
        help="expand each task with its notes, files, deps + the roadmap framing",
    )
    p_list.set_defaults(fn=cmd_list)
    sub.add_parser("next").set_defaults(fn=cmd_next)

    args = ap.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
