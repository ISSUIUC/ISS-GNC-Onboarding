"""Grade a student submission against an exercise's reference solution.

Flow for a graded exercise:
    1. Run  setup + reference solution   -> expected variables + expected stdout
       (cached; only recomputed when the notebook changes).
    2. Run  setup + student submission   -> their variables + their stdout.
    3. Compare per the exercise's checks  -> CheckResults + an overall score.

All code execution happens in `runner.py` in a separate process with a
wall-clock timeout, so a student's infinite loop can't take down the server.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import feedback
from .feedback import CheckResult
from .parser import Exercise, Module

RUNNER = str(Path(__file__).with_name("runner.py"))
WALL_TIMEOUT = 12  # seconds; a hard stop on top of the runner's CPU rlimit

_reference_cache: dict[str, "RunResult"] = {}


@dataclass
class RunResult:
    ok: bool
    stdout: str = ""
    error: str | None = None
    vars: dict = field(default_factory=dict)  # name -> encoded value
    missing: list[str] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)  # from a behaviour checker
    figures: list[str] = field(default_factory=list)  # matplotlib PNGs, as data: URIs


@dataclass
class GradeResult:
    passed: bool
    score: float  # 0.0 - 1.0
    checks: list[CheckResult]
    stdout: str = ""
    error: str | None = None
    figures: list[str] = field(default_factory=list)


def execute(setup: list[str], code: str, extract: list[str], checker: str = "") -> RunResult:
    """Run `code` (after best-effort `setup`) in the sandboxed subprocess."""
    job = {"setup": setup, "code": code, "extract": extract, "checker": checker}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(job, fh)
        job_path = fh.name
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    try:
        proc = subprocess.run(
            [sys.executable, RUNNER, job_path],
            capture_output=True,
            text=True,
            timeout=WALL_TIMEOUT,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return RunResult(ok=False, error="Timed out — check for an infinite loop.")
    finally:
        Path(job_path).unlink(missing_ok=True)

    if proc.returncode != 0 and not proc.stdout:
        # Process died hard (e.g. killed by the CPU/memory limit).
        detail = proc.stderr.strip() or "process exited abnormally"
        return RunResult(ok=False, error=f"Execution failed: {detail}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return RunResult(ok=False, error="Internal grader error (bad runner output).")
    return RunResult(
        ok=data["ok"],
        stdout=data["stdout"],
        error=data["error"],
        vars=data["vars"],
        missing=data["missing"],
        checks=data.get("checks", []),
        figures=data.get("figures", []),
    )


def _extract_names(exercise: Exercise) -> list[str]:
    names: list[str] = []
    for check in exercise.checks:
        for name, _tol in check.variables:
            if name not in names:
                names.append(name)
    return names


def _reference(module: Module, exercise: Exercise) -> RunResult:
    setup = module.setup_code(exercise.cell_index)
    extract = _extract_names(exercise)
    key = hashlib.sha1(
        json.dumps([setup, exercise.reference_code, extract]).encode()
    ).hexdigest()
    if key not in _reference_cache:
        _reference_cache[key] = execute(setup, exercise.reference_code, extract)
    return _reference_cache[key]


def run_only(module: Module, exercise: Exercise, student_code: str) -> RunResult:
    """Just execute the student's code and return output (the "Run" button)."""
    return run_cell(module, exercise.cell_index, student_code)


def run_cell(module: Module, cell_index: int, code: str) -> RunResult:
    """Run one cell with the notebook state that precedes it.

    Backs "Run" on both exercises and worked-example cells — an example is
    re-runnable and editable, so a student can poke at it the way they would in
    Jupyter, and nothing about it is graded or saved.
    """
    return execute(module.setup_code(cell_index), code, [])


def grade(module: Module, exercise: Exercise, student_code: str) -> GradeResult:
    if not exercise.graded:
        # Practice cell: running it successfully is the whole bar.
        run = run_only(module, exercise, student_code)
        return GradeResult(
            passed=run.ok,
            score=1.0 if run.ok else 0.0,
            checks=[],
            stdout=run.stdout,
            error=run.error,
            figures=run.figures,
        )

    reference = _reference(module, exercise)
    if not reference.ok:
        return GradeResult(
            passed=False,
            score=0.0,
            checks=[CheckResult("Reference solution", ok=False,
                                message="The lead's reference solution failed to run — "
                                        "this exercise is misconfigured.")],
            error=reference.error,
        )

    setup = module.setup_code(exercise.cell_index)
    student = execute(setup, student_code, _extract_names(exercise), exercise.checker_code)

    checks: list[CheckResult] = []
    has_behaviour = any(c.kind == "behaviour" for c in exercise.checks)
    if not student.ok and not has_behaviour:
        # Their code raised — report it and fail every check. (A behaviour
        # checker reports the crash itself, in its own words, so skip this.)
        checks.append(CheckResult("Runs without error", ok=False,
                                  message="Your code raised an exception (see output)."))

    for check in exercise.checks:
        if check.kind == "vars":
            for name, tol in check.variables:
                if name in student.missing or name not in student.vars:
                    checks.append(CheckResult(
                        f"Variable `{name}`", ok=False,
                        message=f"`{name}` was never defined."))
                    continue
                ref_val = feedback.decode(reference.vars[name])
                got_val = feedback.decode(student.vars[name])
                res = feedback.compare(f"Variable `{name}`", ref_val, got_val,
                                       tol, exercise.reveal)
                checks.append(res)
        elif check.kind == "output":
            checks.append(feedback.compare_output(reference.stdout, student.stdout,
                                                  exercise.reveal))
        elif check.kind == "output_contains":
            checks.extend(feedback.output_contains(student.stdout, check.substrings))
        elif check.kind == "behaviour":
            checks.extend(feedback.behaviour_checks(student.checks, exercise.reveal))

    graded_checks = [c for c in checks if c.label != "Runs without error"] or checks
    passed_count = sum(1 for c in graded_checks if c.ok)
    score = passed_count / len(graded_checks) if graded_checks else 0.0
    passed = all(c.ok for c in checks) and student.ok

    return GradeResult(
        passed=passed,
        score=score,
        checks=checks,
        stdout=student.stdout,
        error=student.error,
        figures=student.figures,
    )
