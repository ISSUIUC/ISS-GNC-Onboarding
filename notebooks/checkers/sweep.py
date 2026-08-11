#!/usr/bin/env python
"""Grade every exercise's own reference solution — the repo's regression test.

The fastest way to know a checker is sane is to feed it the answer the lead
wrote for that exercise: if the reference can't pass, no student can. This
sweeps all six notebooks and prints one row per exercise.

    uv run python notebooks/checkers/sweep.py           # everything
    uv run python notebooks/checkers/sweep.py kalman    # notebooks matching "kalman"

Exits non-zero when something fails that isn't in KNOWN_FAILURES below, so it
works as a pre-commit / CI gate after editing a notebook or a checker.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine import grader  # noqa: E402
from engine.parser import parse_notebook  # noqa: E402

# Teaching order, so the table reads the way the modules are taken.
NOTEBOOKS = [
    "Introduction.ipynb",
    "Vectors.ipynb",
    "Linear_Algebra.ipynb",
    "basic_filters.ipynb",
    "kalman_filter.ipynb",
    "extended_kalman_filter.ipynb",
]

# Authoring gaps that predate the checker work: these four exercises ship a
# SOLUTION region that doesn't satisfy their own checker. They're reported as
# KNOWN, not as regressions, so a red row here always means something new.
# Fix the notebook's SOLUTION region and delete the entry.
KNOWN_FAILURES = {
    "introduction-ex1": "solution keeps the stub's 'Hello GNC Team!' message",
    "introduction-ex2": "solution never assigns MIDAS_color",
    "introduction-ex6": "solution still prints, the checker requires silence",
    "vectors-ex1": "solution never defines `p`",
}


def main() -> int:
    wanted = [a.lower() for a in sys.argv[1:]]
    notebooks = [n for n in NOTEBOOKS
                 if not wanted or any(w in n.lower() for w in wanted)]

    regressions: list[str] = []
    known_hit: list[str] = []

    for name in notebooks:
        module = parse_notebook(ROOT / "notebooks" / name)
        print(f"\n{module.id}  ({name})")
        for exercise in module.exercises:
            try:
                result = grader.grade(module, exercise, exercise.reference_code)
                passed, score = result.passed, result.score
                failed = [c for c in result.checks if not c.ok]
            except Exception as exc:  # a checker that dies takes the sweep with it
                passed, score, failed = False, 0.0, []
                print(f"  ERR   {exercise.id:32s} {type(exc).__name__}: {exc}")

            if passed:
                print(f"  ok    {exercise.id:32s} {score:.2f}  "
                      f"{'graded' if exercise.graded else 'practice'}")
                continue

            known = KNOWN_FAILURES.get(exercise.id)
            (known_hit if known else regressions).append(exercise.id)
            print(f"  {'KNOWN' if known else 'FAIL '} {exercise.id:32s} {score:.2f}"
                  + (f"  ({known})" if known else ""))
            for check in failed:
                print(f"          - {check.label}: {check.message[:90]}")

    print()
    if known_hit:
        print(f"{len(known_hit)} known authoring gap(s): {', '.join(known_hit)}")
    if regressions:
        print(f"FAILED — {len(regressions)} regression(s): {', '.join(regressions)}")
        return 1
    print("All reference solutions pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
