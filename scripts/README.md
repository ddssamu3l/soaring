# Agentic toolchain — how to use it

The scaffold that lets agents work on this repo safely. Two layers:

- **Gate + merge** — make bad code un-landable (deterministic gate → AI review → serialized merge).
- **Durable state** — survive compaction / fresh sessions by keeping truth on disk, not in the context window.

> Full human reference for the toolchain. The condensed, always-loaded version lives
> in `.claude/rules/agentic-workflow.md`; this is the deeper dive.

**Quick menu:** `python3 cli/menu.py` prints every CLI, the automatic hooks, and the
escape hatches at a glance — the fastest way to see what you can run.

---

## Durable session state (the anti-amnesia layer)

Truth lives in three places: **`task_list.json`** (structured work order),
**`progress.txt`** (append-only narrative log), and **git history**. Conversation
is scratch; these are canonical. A session can die at any point and the next one
rebuilds full state from them.

### `task.py` — the ONLY way to change `task_list.json`

Never hand-edit `task_list.json`. A PreToolUse hook (`guard_state.py`) blocks
direct Edit/Write of it, so this CLI is the only door. It enforces the invariants
an LLM would otherwise corrupt: valid JSON, **exactly one `active` task**
(single-writer), deps satisfied before start, and a **real landed commit required
to mark done**.

```bash
python3 cli/task.py add --title "Log a dataset" --deps t1,t2 --files "data_gen.py" --notes "..."
python3 cli/task.py start t2          # -> active   (refuses if another task is active)
python3 cli/task.py done  t2 --commit <sha>   # -> done (sha must exist in git)
python3 cli/task.py block t2 --reason "sim API changed"
python3 cli/task.py list              # status board — progress bar, deps, next-up (colorized on a TTY)
python3 cli/task.py list --full       # + each task's notes, files, and the roadmap framing
python3 cli/task.py next              # the next pickable task
```

Break-glass for a human to hand-edit anyway: `ALLOW_STATE_EDIT=1`.

### `log.py` — append to the narrative log

```bash
python3 cli/log.py "t2 started — plan: 2-layer MLP, MSE, keystone plot is the deliverable"
```

Auto-stamps date + active task + git sha. Append-only. Like `task_list.json`, the file is
edit-locked by the `guard_state.py` PreToolUse hook — direct Edit/Write is blocked, so
`log.py` is the door (break-glass: `ALLOW_STATE_EDIT=1`). The hook only catches the *tool*,
not this CLI's own write.

### `rehydrate.py` — resume from disk

```bash
python3 cli/rehydrate.py     # print the board: active/next task, recent progress, recent commits
```

Also wired as a **SessionStart hook** (`.claude/settings.json`) so its output is
injected at the top of every new/compacted session — the session reopens already
knowing where it is. **The standing convention: at session start, resume the
ACTIVE (or NEXT) task; mutate state only via `task.py`.**

---

## Gate + merge (the make-bad-code-un-landable layer)

### `check_all.py` — the deterministic gate (runs on every commit via the hook)

11 gates: format, lint, mypy --strict, pytest, **test-presence** (every non-exempt
`.py` needs `tests/test_<name>.py`), **test-coupling** (edit a file → touch its
test), **exempt-guard** (adding a `.test-exempt` entry needs `ALLOW_EXEMPT=1`),
**doc-coupling** (a `scripts/*.py` or `cli/*.py` change must touch a doc), **docs-generated**
(the generated CLI reference isn't stale), no-todos, file-size. Run it directly anytime:

```bash
.venv/bin/python scripts/check_all.py
```

**Conventions to avoid needless gate failures:** write the test file alongside the
code; run `ruff format` before committing; no `TODO/FIXME/XXX`; keep files < 600
lines; new non-exempt module ⇒ add `tests/test_<name>.py` (or exempt it, which
needs a human).

### `land.py` — serialized merge queue (gate + review + state update, atomic)

Merges a feature branch into main behind a lock: re-runs `check_all` on the merged
result, re-checks coupling/exempt on the merge delta, runs the AI review, and
**rolls main back on any failure — including an unexpected crash**, not just a red
gate (a merge that lands ungated because the tool crashed is worse than a loud
failure). It resolves its own interpreter (and `.venv`) from the SHARED repo root, not
its own worktree, since a task's dedicated worktree (the standing default) has no
`.venv` of its own.

On a green land it also **commits and pushes the task's done-mark itself**, inside
the same lock — `task.py done` only rewrites `task_list.json` on disk, so `land.py`
has to be the one to commit that edit, or every land leaves main's working tree dirty
for whoever's checkout happens to be primary next.

**`land.py` is the ONLY door to main — enforced, not asked.** A raw `git merge` into
main would bypass the judge *and* `check_all` (the pre-commit hook doesn't fire on
merges). A `pre-merge-commit` hook refuses any merge into main unless it came from
`land.py` (which sets `LAND_ACTIVE=1`); the repo also sets `merge.ff = false` so a
fast-forward can't slip past unhooked. So "merge ⇒ judged" is a git guarantee, not a
convention. Human break-glass for a deliberate manual merge: `ALLOW_DIRECT_MERGE=1`.

**The task↔branch binding is the branch name.** Name branches
`feature/<taskid>-<slug>`; `land.py` parses the task id out of it and marks that task
done on a green merge — no flag to remember. It's tied to the **branch name**.

```bash
python3 cli/land.py feature/t1-dataset        # derives t1 from the branch, marks it done
python3 cli/land.py my-branch --task t1        # override when the branch can't follow the convention
```

On success it merges, pushes (best-effort), and runs `task.py done t1 --commit <sha>`.
A bookkeeping mismatch warns but never undoes a good merge.

### `review.py` — the AI reviewer (invoked by `land.py`; can run standalone)

Risk-tiered panel (correctness / ml-integrity / quality / **docs**) + a coordinator
that biases toward approval. The **ml-integrity** reviewer guards the sensor firewall
and silent-ML failures; the **docs** reviewer fires whenever a `scripts/*.py` or `cli/*.py` changes and
checks the docs actually match the code. Break-glass: `BREAK_GLASS=1`.

**Cost / offline note.** The judge shells out to the `claude` CLI, so it uses your Claude
**subscription** (effectively $0 marginal — the "t1–t7 ~$0" framing is about *compute*, not
tooling) and needs `claude` on PATH + network. That adds no new constraint on landing, which
already needs network to *push*. If `claude` is unavailable and you still must land,
`BREAK_GLASS=1` skips the judge (human override) — otherwise a failed review rolls the merge
back.

```bash
python3 cli/review.py --base main     # review main..HEAD
```

---

## The standard task loop (what an agent does)

```bash
python3 cli/task.py next                       # see what's up
python3 cli/task.py start t1                    # claim it (single-writer)
git worktree add ../soaring-t1 -b feature/t1-dataset main   # dedicated checkout per task (standing default)
cd ../soaring-t1
python3 cli/log.py "t1 started — plan: ..."     # note the plan
# ... write code + its test; commit (pre-commit hook runs check_all) ...
cd -                                            # back to the primary checkout, on main
python3 cli/land.py feature/t1-dataset --task t1   # gate + review + merge + cleanup + mark done
```

Landing a task's worktree isn't something you switch into and out of by hand for
`land.py` itself — `land.py` always runs from the **primary checkout** (the one that
stays on `main`), and after a successful land it best-effort removes the branch's
worktree + the merged local branch, so the task's checkout is disposable once its work
is on main.

## What you can rely on to exist

- Direct commits to `main` are refused by the pre-commit hook — work goes on a
  `feature/<taskid>-<slug>` branch; main advances only via `land.py` (override: `ALLOW_MAIN_COMMIT=1`).
- Direct `git merge` into `main` is refused by the pre-merge-commit hook — the judge
  can't be skipped by merging around `land.py` (override: `ALLOW_DIRECT_MERGE=1`).
- Every commit (yours or an agent's) is gated by `check_all` via the pre-commit hook.
- Every land re-gates the merge and runs the AI review before main moves.
- `task_list.json` can only change through `task.py` (the hook blocks the bypass).
- A fresh/compacted session is auto-rehydrated from disk at SessionStart.
