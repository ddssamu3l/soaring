# Agentic workflow — the scaffold + how to use every piece (USE THIS)

This is a **tracked** rule (loads every session like CLAUDE.md, but versioned so a
`check_all` gate keeps it in sync with the scripts). Full reference: `scripts/README.md`.

**Mental model.** State lives on disk, NOT the conversation — `task_list.json` (work
order) + `progress.txt` (narrative log) + git history. Conversation is scratch; a session
can die anytime and the next resumes from disk. **Single-writer: exactly ONE `active` task
at a time — one branch, one task.** Every task gets its **own dedicated worktree**
(`../soaring-<taskid>`, standing default — see "Parallelism" below): this makes a task's
checkout disposable and never collides with any other session's WIP, whether or not
another task happens to be active at the same time.
Layers: deterministic gate (`check_all`) → AI reviewer (`review`) → serialized merge queue
(`land`) → durable state (`task`/`log`/`rehydrate`).

**Session start.** A SessionStart hook auto-runs `rehydrate.py` → injects the board
(active/next task, recent progress, commits). **If a task is ACTIVE, resume it** — a prior
session died mid-work; picking it up is the whole point of durable state. **If NONE is
active, surface the next task and wait for the user's go before `start`ing it** — claiming
new work is a direction call (theirs); resuming interrupted work is not.

**The task loop (do exactly this):**
```bash
python3 cli/task.py next                     # what's up
python3 cli/task.py start t1                  # claim it (refuses if another active, or deps unmet)
git worktree add ../soaring-t1 -b feature/t1-dataset main   # dedicated checkout; branch NAME carries the task id
cd ../soaring-t1
python3 cli/log.py "t1 started — plan: ..."
#   ... write the code AND its test; git commit  (pre-commit hook runs check_all) ...
cd -                                          # back to the primary checkout, on main
python3 cli/land.py feature/t1-dataset        # serialize + gate + review + merge + cleanup; marks t1 done
python3 cli/log.py "t1 done — <outcome / decision>"
```

**Commit freely; ask before publishing.** Feature-branch commits are pre-authorized —
local, reversible, they never touch main — so commit as you work (that's what trips the
pre-commit gate). The ask-gate is `land.py`: it moves main **and pushes to the public repo**,
so run it **only on the user's explicit greenlight** (they read the diff first). Any other
`git push` needs an ask too. Rule of thumb: everything up to `land.py` is yours; publishing
is the user's call.

**When to use each command** (exact signatures are in the generated block below):
- `task.py next` / `list` — next pickable / full board (start here).
- `task.py start tN` — claim a task (enforces single-writer + deps).
- `task.py add` — append a pending task when new work surfaces.
- `task.py block tN` — mark blocked when stuck.
- `task.py done tN` — rarely by hand; `land.py` does it on a green merge.
- `log.py "msg"` — stamped note at every start / finish / decision.
- `rehydrate.py` — print the board to catch up mid-session.
- `check_all.py` — run the deterministic gate to pre-check before committing.
- `review.py --base main` — standalone AI review (`land` runs it for you).
- `land.py <branch>` — land a finished branch (gate + review + merge + rollback + mark-done).
- `menu.py` — print the human cheat-sheet: every CLI, the automatic hooks, and the escape hatches.

**Task↔branch binding = the BRANCH NAME.** `feature/<taskid>-<slug>` → `land.py` derives
`tN` and marks it done on a green merge (no flag to remember). `--task tN` overrides for
off-convention branches. Tied to the branch name.

**Parallelism (worktree-per-task is the standing default).** Every task, whether or not
another task happens to be active, gets its **own** checkout: `git worktree add
../soaring-<taskid> -b feature/<taskid>-<slug> main`. This is what makes two Claude sessions
safe to run at the same time without a special "are we parallel?" judgment call — a task's
worktree can never collide with any other session's WIP, staged index, or branch switches.
`task_list.json`'s single-writer rule (one `active` task) is still per-checkout, not a
global lock — two unrelated tasks CAN both be `active` on their own worktree simultaneously;
the actual serialization point is `land.py`'s file lock, which forces one merge into `main`
at a time and reconciles any `task_list.json` diffs at that point (see `scripts/README.md`).
`land.py` always runs from the **primary checkout**, on `main` — never from a task's
worktree — and on a successful land it best-effort removes that branch's worktree + the
now-merged local branch (step 7), so a task's checkout is disposable once its work lands.
Your harness will also offer worktrees (`Agent isolation:worktree`, ultracode): fine for
**read-only** fan-out (explore / research / review that never commits here); **never** for
agents that write code — those get their own task worktree via the flow above, same as any
other task.

**What's enforced (work WITH it, don't fight it):**
- **No direct commits to `main`** — the pre-commit hook refuses them; real work goes on a
  `feature/<taskid>-<slug>` branch and reaches main only via `land.py`. (Override for
  deliberate/bootstrap commits: `ALLOW_MAIN_COMMIT=1`.)
- **`check_all` on every commit** (pre-commit hook), gates include: format, lint, mypy
  --strict, pytest, test-presence, test-coupling, exempt-guard, no-todos, file-size,
  doc-coupling, docs-generated. Avoid needless failures: write the test file alongside the
  code, `ruff format` first, no `TODO/FIXME/XXX`, files < 600 lines, new non-exempt `.py` ⇒
  `tests/test_<name>.py`, and **regenerate docs after changing a CLI** (`gen_docs.py`).
- **`task_list.json` and `progress.txt` are edit-locked** — a PreToolUse hook blocks direct
  Edit/Write. Change them ONLY via their CLI: `task.py` (enforces one-active-task, deps,
  real-commit-for-done) and `log.py` (append-only, auto-stamped). Break-glass: `ALLOW_STATE_EDIT=1`.
- **`land.py`** serializes via a file lock, re-gates the *merged* result + the merge delta,
  runs the AI review, and **rolls main back on ANY failure**.
- **No direct `git merge` into `main`** — a `pre-merge-commit` hook refuses it (main moves
  only via `land.py`, which sets `LAND_ACTIVE=1`; the repo sets `merge.ff=false` so a
  fast-forward can't dodge the hook). This makes "every merge is judged" a git guarantee,
  not a convention. (Override for a deliberate manual merge: `ALLOW_DIRECT_MERGE=1`.)
- **AI reviewer** (risk-tiered panel via `land`): `ml-integrity` guards the **sensor
  firewall** (a model must NEVER see a Thermal's true `x0,y0,w_peak,radius` — only
  `sense()`) + silent-ML; a `docs` lens fires on any `scripts/*.py` or `cli/*.py` change and checks the
  docs actually match the code (coupling proves a doc was touched; this proves it's right).

**Escape hatches (human break-glass — agents do NOT self-exempt):**
`ALLOW_EXEMPT=1` add a `.test-exempt` entry · `ALLOW_NO_TEST_UPDATE=1` edit a source without
its test · `ALLOW_STATE_EDIT=1` hand-edit `task_list.json` or `progress.txt` · `ALLOW_NO_DOC_UPDATE=1`
change a script without touching docs · `ALLOW_MAIN_COMMIT=1` commit directly to main ·
`ALLOW_DIRECT_MERGE=1` merge into main without `land.py` · `BREAK_GLASS=1` force-approve
the AI review.

**Guarantees you can rely on:** every commit is gated; every land re-gates + AI-reviews
before main moves; **main cannot move any other way** — a direct commit *or* a raw `git merge`
into main is refused by a hook, so the judge is un-skippable; state changes only through their
CLIs (`task_list.json` via `task.py`, `progress.txt` via `log.py`); a fresh/compacted session is
auto-rehydrated from disk; the CLI reference below can't drift (it's generated from the
scripts' own `--help`).

## Exact CLI signatures
<!-- BEGIN cli-reference (generated by cli/gen_docs.py -- DO NOT EDIT) -->
```text
usage: gen_docs.py [-h] [--check]

usage: land.py [-h] [--task TASK] branch

usage: log.py [-h] message [message ...]

usage: menu.py [-h]

usage: rehydrate.py [-h] [--hook]

usage: review.py [-h] [--base BASE] [--staged]

usage: task.py [-h] {add,start,done,block,list,next} ...

usage: task.py add [-h] --title TITLE [--deps DEPS] [--files FILES] [--notes NOTES]

usage: task.py start [-h] id

usage: task.py done [-h] --commit COMMIT id

usage: task.py block [-h] --reason REASON id

usage: task.py list [-h] [-v]

usage: task.py next [-h]
```
<!-- END cli-reference -->
Regenerate after changing any CLI: `python3 cli/gen_docs.py` (the `docs-generated`
gate fails if this block is stale).
