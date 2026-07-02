#!/usr/bin/env python3
"""menu.py -- the human cheat-sheet for the agentic toolchain.

Run `python3 cli/menu.py` to see every CLI you can use in this repo, what each does,
the one command you'll reach for most, plus the hooks that fire on their own and the
break-glass env vars. This is the FRIENDLY overview; the authoritative CLI signatures are
in the generated block in .claude/rules/agentic-workflow.md (regenerate with gen_docs.py),
and the deep dive is scripts/README.md. Pure stdlib; runs under any python3.
"""

from __future__ import annotations

import argparse
import sys

BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
DIM = "\033[2m"
RESET = "\033[0m"

# (command you type, one-line purpose)
SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "SESSION",
        [
            (
                "python3 cli/rehydrate.py",
                "print the board — active/next task, recent progress, recent commits",
            ),
        ],
    ),
    (
        "WORK ORDER  (state — the ONLY sanctioned way to touch task_list.json)",
        [
            ("python3 cli/task.py next", "show the next pickable task"),
            (
                "python3 cli/task.py list",
                "status board — progress bar, deps, next-up  (add --full for notes + framing)",
            ),
            ("python3 cli/task.py start t1", "claim a task — single-writer; checks deps"),
            (
                'python3 cli/task.py add --title "…" --deps t1 --files "x.py"',
                "append a new pending task",
            ),
            ('python3 cli/task.py block t1 --reason "…"', "mark a task blocked"),
            (
                "python3 cli/task.py done t1 --commit <sha>",
                "mark done — rare by hand; land.py does it for you",
            ),
            ('python3 cli/log.py "t1 started — plan: …"', "stamped note to progress.txt"),
        ],
    ),
    (
        "GATE + MERGE  (make bad code un-landable)",
        [
            (
                ".venv/bin/python scripts/check_all.py",
                "run the deterministic gate (11 checks) — pre-check before committing",
            ),
            ("python3 cli/review.py --base main", "run the AI judge standalone"),
            (
                "python3 cli/land.py feature/t1-dataset",
                "merge queue: gate → judge → merge → rollback-on-fail → mark task done",
            ),
            (
                "python3 cli/gen_docs.py",
                "regenerate the CLI reference after changing any CLI",
            ),
        ],
    ),
]

# hooks that run on their own — you never call these directly
AUTOMATIC: list[tuple[str, str]] = [
    ("pre-commit", "runs check_all on every commit; refuses a direct commit to main"),
    ("pre-merge-commit", "refuses a raw `git merge` into main — land.py is the only door"),
    (
        "PreToolUse guard",
        "blocks direct Edit/Write of task_list.json + progress.txt (use their CLIs)",
    ),
    ("SessionStart", "runs rehydrate.py so a new/compacted session reopens knowing the board"),
]

# break-glass overrides — human decisions; an agent does NOT self-exempt
HATCHES: list[tuple[str, str]] = [
    ("ALLOW_MAIN_COMMIT=1", "commit directly to main"),
    ("ALLOW_DIRECT_MERGE=1", "git merge into main without land.py"),
    ("ALLOW_EXEMPT=1", "add a .test-exempt entry"),
    ("ALLOW_NO_TEST_UPDATE=1", "edit a source without touching its test"),
    ("ALLOW_NO_DOC_UPDATE=1", "change a script without touching docs"),
    ("ALLOW_STATE_EDIT=1", "hand-edit task_list.json or progress.txt"),
    ("BREAK_GLASS=1", "force-approve the AI review"),
]

FLOW: list[str] = [
    "python3 cli/task.py start t1               # claim it (single-writer)",
    "git checkout -b feature/t1-dataset             # branch name carries the task id",
    'python3 cli/log.py "t1 started — plan: …"',
    "#   … write code AND its test; git commit  (pre-commit runs check_all) …",
    "git checkout main",
    "python3 cli/land.py feature/t1-dataset     # on your go: gate + judge + merge",
]


def _hdr(title: str) -> str:
    return f"\n{BOLD}{GREEN}{title}{RESET}"


def render() -> str:
    out: list[str] = []
    out.append(f"\n{BOLD}soaring — toolchain menu{RESET}  {DIM}(python3 cli/menu.py){RESET}")

    for title, rows in SECTIONS:
        out.append(_hdr(title))
        for cmd, purpose in rows:
            out.append(f"  {CYAN}{cmd}{RESET}")
            out.append(f"      {DIM}{purpose}{RESET}")

    out.append(_hdr("AUTOMATIC  (hooks — fire on their own; you don't call them)"))
    for name, purpose in AUTOMATIC:
        out.append(f"  {CYAN}{name}{RESET}  {DIM}— {purpose}{RESET}")

    out.append(_hdr("ESCAPE HATCHES  (human break-glass — agents do NOT self-exempt)"))
    for var, purpose in HATCHES:
        out.append(f"  {CYAN}{var}{RESET}  {DIM}— {purpose}{RESET}")

    out.append(_hdr("COMMON FLOW  (how one task runs)"))
    for line in FLOW:
        out.append(f"  {line}")

    out.append(
        f"\n{DIM}Full signatures: .claude/rules/agentic-workflow.md  ·  "
        f"deep dive: scripts/README.md{RESET}"
    )
    return "\n".join(out)


def main() -> int:
    argparse.ArgumentParser(
        prog="menu.py", description="Print the human cheat-sheet for the agentic toolchain."
    ).parse_args()
    print(render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
