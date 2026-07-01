#!/usr/bin/env python3
"""
check_all.py -- the deterministic gate every commit must pass.

This is the "agents as an environment problem" core: a single, fast, NON-AI
command that hard-codes what "shippable" means here. The pre-commit hook runs it,
so neither you nor any agent can land code that fails it. Run it directly too:

    python scripts/check_all.py

Exit 0 only if ALL gates pass; non-zero (and a clear report) otherwise.

Test policy is FILE-BASED, not percentage-based:
  * every source .py must have a matching tests/test_<name>.py  (presence)
  * editing a source file without touching its test is blocked   (coupling)
  * exceptions live in .test-exempt, and ADDING one needs a human (guard)

RATCHET -- add next, when the project earns each:
  * duplicate-code detection  -> when agents start adding modules (step 1+)
  * AGENTS.md validation      -> if/when a committed agent-doc references files
  * CI parity check           -> the moment a CI workflow exists
  * sensor-firewall gate      -> when the data-gen module lands: assert the MLP
                                 input builder never imports Thermal internals
  * mutation-score floor      -> the real test-QUALITY gate (mutmut); file-presence
                                 proves a test EXISTS, mutation proves it BITES
  * bump mypy/ruff rule sets  -> tighten as the codebase grows
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXEMPT_FILE = REPO / ".test-exempt"
TESTS_DIR = REPO / "tests"
MAX_FILE_LINES = 600
# docs that must track the scripts (keeps the workflow doc in sync with the mechanics)
DOC_FILES = {".claude/rules/agentic-workflow.md", "scripts/README.md"}
SKIP_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
TODO_MARKERS = ("TODO", "FIXME", "XXX")

# this file legitimately contains the marker words (it defines them), so the
# no-todos gate skips its own definition site.
_SELF = Path(__file__).resolve()


def _bin(name: str) -> str:
    """Prefer the tool next to the current interpreter (the venv), so the hook
    uses the same toolchain regardless of PATH."""
    cand = Path(sys.executable).parent / name
    return str(cand) if cand.exists() else name


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr)


def _source_files() -> list[Path]:
    return [p for p in REPO.rglob("*.py") if not any(part in SKIP_DIRS for part in p.parts)]


# --- test-policy helpers -------------------------------------------------
def _exempt_entries(text: str | None = None) -> set[str]:
    """Parse .test-exempt. Entries are repo-relative posix paths; a trailing '/'
    means a directory prefix. Inline '#' comments and blank lines are ignored."""
    if text is None:
        text = EXEMPT_FILE.read_text() if EXEMPT_FILE.exists() else ""
    entries = set()
    for line in text.splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            entries.add(entry)
    return entries


def _is_exempt(rel: str, entries: set[str]) -> bool:
    for e in entries:
        if e.endswith("/") and rel.startswith(e):
            return True
        if rel == e:
            return True
    return False


def _find_test(stem: str) -> str | None:
    """Return the repo-relative path of tests/**/test_<stem>.py, or None."""
    if not TESTS_DIR.exists():
        return None
    for p in TESTS_DIR.rglob(f"test_{stem}.py"):
        return p.relative_to(REPO).as_posix()
    return None


def _needs_test() -> list[Path]:
    """Source files that must have a matching test (not tests, not exempt)."""
    entries = _exempt_entries()
    out = []
    for p in _source_files():
        rel = p.relative_to(REPO).as_posix()
        if rel.startswith("tests/") or _is_exempt(rel, entries):
            continue
        out.append(p)
    return out


def _staged_files() -> list[str]:
    code, o = _run(["git", "diff", "--cached", "--name-only"])
    return [ln.strip() for ln in o.splitlines() if ln.strip()] if code == 0 else []


# --- reusable policy logic (shared by the commit-time gates AND land.py) ---
# The gates below run at commit time over the *staged* set; land.py runs the same
# two functions over the *merge delta* so the policy holds even if a worktree
# commit slipped past the hook (e.g. `--no-verify`). One source of truth.
def coupling_violations(changed: set[str]) -> list[str]:
    """Source files in `changed` whose test file exists but wasn't also changed.
    (A missing-entirely test is the presence gate's job, not this one.)"""
    entries = _exempt_entries()
    offenders = []
    for rel in changed:
        if not rel.endswith(".py") or rel.startswith("tests/") or _is_exempt(rel, entries):
            continue
        test = _find_test(Path(rel).stem)
        if test and test not in changed:
            offenders.append(f"{rel}  (its {test} was not touched)")
    return offenders


def exemptions_added(changed: set[str], base_ref: str) -> set[str]:
    """Exemption entries present now but absent at `base_ref` — i.e. this change
    ADDS them. Empty if .test-exempt isn't in `changed`."""
    if ".test-exempt" not in changed:
        return set()
    code, base_text = _run(["git", "show", f"{base_ref}:.test-exempt"])
    base = _exempt_entries(base_text if code == 0 else "")
    return _exempt_entries() - base


# --- gates: each returns (ok, detail) ------------------------------------
def gate_format() -> tuple[bool, str]:
    code, out = _run([_bin("ruff"), "format", "--check", "."])
    return code == 0, out.strip()


def gate_lint() -> tuple[bool, str]:
    code, out = _run([_bin("ruff"), "check", "."])
    return code == 0, out.strip()


def gate_types() -> tuple[bool, str]:
    code, out = _run([_bin("mypy")])  # files/strictness come from pyproject.toml
    return code == 0, out.strip()


def gate_tests() -> tuple[bool, str]:
    code, out = _run([_bin("pytest")])
    if code != 0:
        return False, out.strip() or "tests failed"
    last = [ln for ln in out.splitlines() if ln.strip()]
    return True, last[-1] if last else "tests passed"


def gate_test_presence() -> tuple[bool, str]:
    """Every non-exempt source file must have a matching test file."""
    missing = [
        f"{p.relative_to(REPO).as_posix()}  ->  needs tests/test_{p.stem}.py"
        for p in _needs_test()
        if _find_test(p.stem) is None
    ]
    if missing:
        head = "source files with no test file (add one, or exempt in .test-exempt):\n  "
        return False, head + "\n  ".join(missing)
    return True, f"{len(_needs_test())} source file(s) each have a test"


def gate_test_coupling() -> tuple[bool, str]:
    """A source file changed in this commit must have its test file changed too.
    Only runs on staged changes (commit time). Override: ALLOW_NO_TEST_UPDATE=1."""
    staged = set(_staged_files())
    if not staged:
        return True, "no staged changes — coupling check skipped"
    offenders = coupling_violations(staged)
    if offenders and os.environ.get("ALLOW_NO_TEST_UPDATE") != "1":
        head = "code changed but its tests weren't (set ALLOW_NO_TEST_UPDATE=1 to override):\n  "
        return False, head + "\n  ".join(offenders)
    return True, "edited files have their tests touched"


def gate_exemption_guard() -> tuple[bool, str]:
    """Adding a NEW exemption is a human decision. If this commit adds entries to
    .test-exempt, block unless ALLOW_EXEMPT=1. (Answers: an agent cannot silently
    exempt itself out of writing tests.)"""
    added = exemptions_added(set(_staged_files()), "HEAD")
    if added and os.environ.get("ALLOW_EXEMPT") != "1":
        head = "new test exemptions need human approval (re-run with ALLOW_EXEMPT=1):\n  "
        return False, head + "\n  ".join(sorted(added))
    return True, "exemption changes approved" if added else "exemption list unchanged"


def gate_doc_coupling() -> tuple[bool, str]:
    """A change to any scripts/*.py must also touch a workflow doc, so the docs stay
    in sync with the mechanics. Commit-time (staged). Override: ALLOW_NO_DOC_UPDATE=1."""
    staged = set(_staged_files())
    if not staged:
        return True, "no staged changes — doc-coupling skipped"
    scripts_changed = sorted(f for f in staged if f.startswith("scripts/") and f.endswith(".py"))
    if scripts_changed and not (staged & DOC_FILES):
        if os.environ.get("ALLOW_NO_DOC_UPDATE") == "1":
            return True, "scripts changed, docs untouched — allowed via override"
        head = "scripts changed but no doc touched (update a doc, or ALLOW_NO_DOC_UPDATE=1):\n  "
        return False, head + "\n  ".join(scripts_changed)
    return True, "docs tracked with script changes"


def gate_docs_generated() -> tuple[bool, str]:
    """The generated CLI reference must match the scripts' actual --help (no drift)."""
    code, out = _run([sys.executable, "scripts/gen_docs.py", "--check"])
    return code == 0, out.strip().splitlines()[-1] if out.strip() else "cli-reference current"


def gate_no_todos() -> tuple[bool, str]:
    hits = []
    for p in _source_files():
        if p.resolve() == _SELF:
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if any(m in line for m in TODO_MARKERS):
                hits.append(f"{p.relative_to(REPO)}:{i}: {line.strip()[:70]}")
    if hits:
        return False, "unfinished markers (agents may not punt work):\n  " + "\n  ".join(hits)
    return True, "no TODO/FIXME/XXX"


def gate_file_size() -> tuple[bool, str]:
    big = [
        f"{p.relative_to(REPO)}: {len(p.read_text().splitlines())} lines (> {MAX_FILE_LINES})"
        for p in _source_files()
        if len(p.read_text().splitlines()) > MAX_FILE_LINES
    ]
    if big:
        head = "files too large (split them; huge files wreck agent context):\n  "
        return False, head + "\n  ".join(big)
    return True, f"all files <= {MAX_FILE_LINES} lines"


GATES = [
    ("format        (ruff format)", gate_format),
    ("lint          (ruff check)", gate_lint),
    ("types         (mypy --strict)", gate_types),
    ("tests         (pytest)", gate_tests),
    ("test-presence (every file tested)", gate_test_presence),
    ("test-coupling (edits touch tests)", gate_test_coupling),
    ("exempt-guard  (no self-exempting)", gate_exemption_guard),
    ("doc-coupling  (scripts→docs synced)", gate_doc_coupling),
    ("docs-generated(cli ref not stale)", gate_docs_generated),
    ("no-todos", gate_no_todos),
    ("file-size", gate_file_size),
]


def main() -> int:
    print("running check-all …\n")
    results = []
    for name, fn in GATES:
        ok, detail = fn()
        mark = "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"
        note = ""
        if ok and detail:
            note = f"  — {detail.splitlines()[0]}"
        print(f"  {mark}  {name}{note}")
        if not ok:
            for line in detail.splitlines():
                print(f"        {line}")
        results.append(ok)

    failed = results.count(False)
    print()
    if failed:
        print(f"\033[31mcheck-all FAILED\033[0m — {failed} gate(s) red. Fix, then commit.")
        return 1
    print("\033[32mcheck-all passed\033[0m — safe to commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
