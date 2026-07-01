#!/usr/bin/env python3
"""
review.py -- the AI reviewer gate (the Cloudflare side of the factory).

check_all.py catches what a machine can decide deterministically. This catches
the semantic stuff a machine can't: logic bugs, silent-ML-failure smells, and the
soaring-specific sensor-firewall invariant. It is risk-tiered (don't send the
dream team to review a typo), runs scoped sub-reviewers, then fuses their findings
with a coordinator that dedupes, drops nitpicks, and biases toward approval.

Usage:
    python scripts/review.py --base main          # review main..HEAD
    python scripts/review.py --staged             # review staged changes
    BREAK_GLASS=1 python scripts/review.py --base main   # force approve

Exit 0 = APPROVE, 2 = BLOCK, 1 = infra error (claude missing, etc.).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAX_DIFF_CHARS = 60_000  # cap context sent per reviewer; truncation is logged, never silent
CLAUDE_TIMEOUT = 300  # s/reviewer; land holds the global merge lock during review, so a
#                       hung `claude` would deadlock the whole queue. Fail loud instead.

# --- reviewer roles: what to look for AND what to IGNORE (Cloudflare technique) ---
ROLES = {
    "correctness": (
        "You are a CORRECTNESS reviewer for a physics/ML sim. Look ONLY for: real "
        "logic bugs, wrong math/physics, off-by-one, incorrect units, broken control "
        "flow, mishandled edge cases. IGNORE: style, formatting, import order, naming, "
        "type annotations, test coverage — deterministic tools already enforce those. "
        "Do NOT nitpick. A clean diff should be APPROVED."
    ),
    "ml-integrity": (
        "You are an ML-INTEGRITY reviewer for a JEPA world-model project. This project "
        "fails SILENTLY, so look ONLY for: (1) SENSOR-FIREWALL violations — any code that "
        "lets a model/data-builder see a Thermal's true fields (x0, y0, w_peak, radius) "
        "instead of only sense()'s local vario; (2) silent-failure smells — data leakage "
        "train/test, a loss that can look fine while learning nothing, latent collapse not "
        "guarded, metrics computed on the wrong tensor; (3) fake/inert results presented as "
        "real. IGNORE style/coverage/formatting. If none present, APPROVE."
    ),
    "quality": (
        "You are a CODE-QUALITY reviewer. Look ONLY for: duplicated logic that should reuse "
        "an existing function, a function/file doing too much, or a genuinely confusing "
        "abstraction that will hurt future agents. IGNORE: subjective style, naming bikesheds, "
        "formatting. Only flag things with real maintenance cost. Default to APPROVE."
    ),
}


def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout


def get_diff(base: str | None, staged: bool) -> str:
    if staged:
        return _git(["diff", "--cached"])
    if base:
        return _git(["diff", f"{base}...HEAD"])
    return _git(["diff", "HEAD"])


def risk_tier(diff: str) -> list[str]:
    """Pick reviewers by blast radius (Cloudflare's risk-tiered compute)."""
    added = sum(1 for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    files = len(re.findall(r"^\+\+\+ ", diff, re.MULTILINE))
    if added < 10 and files < 2:
        return ["correctness"]  # trivial → one reviewer
    if added < 100:
        return ["correctness", "ml-integrity"]  # small → two
    return ["correctness", "ml-integrity", "quality"]  # full pipeline


def call_claude(prompt: str) -> str:
    try:
        proc = subprocess.run(
            ["claude", "-p", prompt],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=CLAUDE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"claude timed out after {CLAUDE_TIMEOUT}s") from None
    if proc.returncode != 0:
        raise RuntimeError(f"claude failed: {proc.stderr.strip()[:300]}")
    return proc.stdout.strip()


def review_one(role: str, diff: str) -> str:
    truncated = diff[:MAX_DIFF_CHARS]
    note = "" if len(diff) <= MAX_DIFF_CHARS else "\n\n[NOTE: diff truncated for length]"
    prompt = (
        f"{ROLES[role]}\n\n"
        "Review this git diff. Output a short list of findings (or 'none'), each as "
        "`SEVERITY: <critical|warning> - <file>: <what and why>`. Do not restate the "
        f"diff. End with exactly one line: `VERDICT: APPROVE` or `VERDICT: BLOCK`.\n\n"
        f"```diff\n{truncated}\n```{note}"
    )
    return call_claude(prompt)


def coordinate(sub_reviews: dict[str, str], diff: str) -> tuple[bool, str]:
    """Fuse sub-reviews into one verdict. Single reviewer → use it directly."""
    if len(sub_reviews) == 1:
        text = next(iter(sub_reviews.values()))
        return ("VERDICT: BLOCK" not in text.upper()), text

    joined = "\n\n".join(f"### {r} reviewer\n{txt}" for r, txt in sub_reviews.items())
    prompt = (
        "You are the COORDINATOR of a code review. Below are findings from specialized "
        "reviewers. Produce ONE fused verdict: dedupe overlapping findings, DROP nitpicks "
        "and speculative/low-value comments, and bias toward APPROVAL — only BLOCK for a "
        "genuine critical problem (a real bug, a sensor-firewall breach, or a fake result). "
        "A warning on otherwise-clean code should APPROVE-with-comments, not block. "
        "List the surviving findings, then end with exactly one line: "
        "`VERDICT: APPROVE` or `VERDICT: BLOCK`.\n\n"
        f"{joined}"
    )
    out = call_claude(prompt)
    return ("VERDICT: BLOCK" not in out.upper()), out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="review <base>...HEAD (e.g. main)")
    ap.add_argument("--staged", action="store_true", help="review staged changes")
    args = ap.parse_args()

    if os.environ.get("BREAK_GLASS") == "1":
        print("⚠️  BREAK_GLASS=1 — review bypassed, forcing APPROVE.")
        return 0

    if not (Path(sys.executable).parent / "claude").exists() and not _which("claude"):
        print("review: `claude` CLI not found — cannot run AI review.", file=sys.stderr)
        return 1

    diff = get_diff(args.base, args.staged)
    if not diff.strip():
        print("review: empty diff, nothing to review → APPROVE.")
        return 0

    roles = risk_tier(diff)
    print(f"reviewing diff  ({len(diff.splitlines())} lines)  →  reviewers: {', '.join(roles)}\n")

    try:
        subs = {role: review_one(role, diff) for role in roles}
        approved, report = coordinate(subs, diff)
    except RuntimeError as e:
        print(f"review: {e}", file=sys.stderr)
        return 1

    print(report)
    print()
    if approved:
        print("\033[32mREVIEW: APPROVED\033[0m")
        return 0
    print("\033[31mREVIEW: BLOCKED\033[0m  (set BREAK_GLASS=1 to override)")
    return 2


def _which(name: str) -> bool:
    return any((Path(d) / name).exists() for d in os.environ.get("PATH", "").split(":") if d)


if __name__ == "__main__":
    sys.exit(main())
