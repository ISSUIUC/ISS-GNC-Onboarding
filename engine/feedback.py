"""Decode values from the runner and compare student answers to the reference.

This is the "grading logic" — a small, readable reimplementation of the kind of
checks PrairieLearn's Python autograder does (`check_scalar`, array closeness,
etc.). Everything returns a `CheckResult` so the UI can show friendly feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = 1e-6


@dataclass
class CheckResult:
    label: str
    ok: bool
    message: str = ""
    expected: str | None = None  # shown only when the exercise allows reveal
    got: str | None = None


class _Opaque:
    """A value we could only carry across as its repr() string."""

    def __init__(self, text: str):
        self.text = text

    def __eq__(self, other):
        return isinstance(other, _Opaque) and other.text == self.text


def decode(enc: dict):
    """Inverse of runner._encode."""
    t = enc.get("t")
    if t == "ndarray":
        return np.array(enc["data"], dtype=enc["dtype"])
    if t in ("int", "float", "bool", "str"):
        return enc["v"]
    if t == "list":
        items = [decode(x) for x in enc["v"]]
        return tuple(items) if enc.get("tuple") else items
    if t == "dict":
        return {k: decode(v) for k, v in enc["v"].items()}
    if t == "set":
        return set(enc["v"])
    if t == "none":
        return None
    if t == "repr":
        return _Opaque(enc["v"])
    return _Opaque(repr(enc))


def _fmt(value) -> str:
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=6, separator=", ")
    if isinstance(value, _Opaque):
        return value.text
    if isinstance(value, float):
        return f"{value:.6g}"
    return repr(value)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _close(a, b, atol: float, rtol: float) -> bool:
    return bool(np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True))


def compare(label: str, ref, got, atol: float | None, reveal: bool) -> CheckResult:
    """Dispatch comparison on the *reference* value's type."""
    a = DEFAULT_ATOL if atol is None else atol
    r = DEFAULT_RTOL if atol is None else 0.0

    def result(ok: bool, message: str = "") -> CheckResult:
        return CheckResult(
            label=label,
            ok=ok,
            message=message,
            expected=_fmt(ref) if (reveal and not ok) else None,
            got=_fmt(got) if not ok else None,
        )

    # Numbers / booleans -----------------------------------------------------
    if isinstance(ref, bool):
        return result(isinstance(got, bool) and got == ref)
    if _is_number(ref):
        if not _is_number(got) and not isinstance(got, (np.integer, np.floating)):
            return result(False, f"expected a number, got {type(got).__name__}")
        return result(_close(float(ref), float(got), a, r))

    # numpy arrays -----------------------------------------------------------
    if isinstance(ref, np.ndarray):
        got_arr = np.asarray(got) if not isinstance(got, np.ndarray) else got
        if got_arr.shape != ref.shape:
            return result(False, f"wrong shape: expected {ref.shape}, got {got_arr.shape}")
        try:
            return result(bool(np.allclose(got_arr, ref, atol=a, rtol=r, equal_nan=True)))
        except (TypeError, ValueError):
            return result(False, "values are not numerically comparable")

    # strings ----------------------------------------------------------------
    if isinstance(ref, str):
        return result(isinstance(got, str) and got == ref)

    # lists / tuples ---------------------------------------------------------
    if isinstance(ref, (list, tuple)):
        if not isinstance(got, (list, tuple)) or len(got) != len(ref):
            n = len(got) if isinstance(got, (list, tuple)) else "?"
            return result(False, f"expected {len(ref)} items, got {n}")
        ok = all(
            compare(label, rv, gv, atol, reveal=False).ok for rv, gv in zip(ref, got)
        )
        return result(ok)

    # dicts ------------------------------------------------------------------
    if isinstance(ref, dict):
        if not isinstance(got, dict) or set(got) != set(ref):
            return result(False, "keys do not match")
        ok = all(
            compare(label, ref[k], got.get(k), atol, reveal=False).ok for k in ref
        )
        return result(ok)

    # sets / everything else -------------------------------------------------
    return result(got == ref)


def normalize_output(text: str) -> str:
    """Compare stdout leniently: trim trailing spaces and blank edges."""
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def compare_output(ref_out: str, got_out: str, reveal: bool) -> CheckResult:
    ref_n, got_n = normalize_output(ref_out), normalize_output(got_out)
    ok = ref_n == got_n
    return CheckResult(
        label="Printed output",
        ok=ok,
        expected=ref_out.rstrip() if (reveal and not ok) else None,
        got=got_out.rstrip() if not ok else None,
    )


def output_contains(got_out: str, substrings: list[str]) -> list[CheckResult]:
    results = []
    for sub in substrings:
        results.append(
            CheckResult(
                label=f'Prints "{sub}"',
                ok=sub in (got_out or ""),
            )
        )
    return results
