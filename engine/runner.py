"""Sandboxed executor — runs in a *subprocess*, never in the web server.

Reads a job (JSON) from the file named in argv[1], executes it, and prints a
single JSON result object to stdout. The job has:

    {
      "setup":   ["<code cell>", ...],   # notebook state, run best-effort
      "code":    "<code to run/grade>",  # student submission or reference
      "extract": ["var1", "var2"]        # variable names to capture
    }

Result:

    {
      "ok": bool,                # did `code` run without raising?
      "stdout": "...",           # captured prints from `code`
      "error": "traceback|null",
      "vars": {name: <encoded>}, # captured values for names that existed
      "missing": ["name", ...]   # requested names that were never defined
    }

Isolation is deliberately lightweight: a CPU-time + address-space rlimit plus a
wall-clock timeout enforced by the parent. This is an internal onboarding tool
for trusted teammates, not a hostile-code sandbox — see README security notes.
"""

from __future__ import annotations

import io
import json
import sys
import traceback
from contextlib import redirect_stdout


def _apply_limits() -> None:
    try:
        import resource
    except ImportError:
        return  # non-POSIX; parent's wall-clock timeout still applies
    cpu_seconds = 10
    mem_bytes = 1024 * 1024 * 1024  # 1 GiB
    for res, limit in ((resource.RLIMIT_CPU, cpu_seconds), (resource.RLIMIT_AS, mem_bytes)):
        try:
            soft, hard = resource.getrlimit(res)
            new_hard = hard if hard != resource.RLIM_INFINITY and hard < limit else limit
            resource.setrlimit(res, (limit, new_hard))
        except (ValueError, OSError):
            pass  # some platforms (macOS RLIMIT_AS) refuse; not fatal


def _encode(v):
    """Turn a runtime value into a JSON-safe, comparison-friendly form."""
    import numpy as np

    if isinstance(v, np.ndarray):
        return {"t": "ndarray", "data": v.tolist(), "dtype": str(v.dtype)}
    if isinstance(v, np.bool_):
        return {"t": "bool", "v": bool(v)}
    if isinstance(v, np.integer):
        return {"t": "int", "v": int(v)}
    if isinstance(v, np.floating):
        return {"t": "float", "v": float(v)}
    if isinstance(v, bool):
        return {"t": "bool", "v": v}
    if isinstance(v, int):
        return {"t": "int", "v": v}
    if isinstance(v, float):
        return {"t": "float", "v": v}
    if isinstance(v, str):
        return {"t": "str", "v": v}
    if isinstance(v, (list, tuple)):
        return {"t": "list", "tuple": isinstance(v, tuple), "v": [_encode(x) for x in v]}
    if isinstance(v, dict):
        return {"t": "dict", "v": {str(k): _encode(val) for k, val in v.items()}}
    if isinstance(v, set):
        return {"t": "set", "v": sorted(repr(x) for x in v)}
    if v is None:
        return {"t": "none"}
    return {"t": "repr", "v": repr(v)}


def main() -> None:
    _apply_limits()
    job = json.loads(open(sys.argv[1], encoding="utf-8").read())

    namespace: dict = {"__name__": "__student__"}

    # 1. Rebuild notebook state (best-effort: a failure in one setup cell,
    #    e.g. an example that referenced a variable from an ungraded exercise,
    #    must not block grading of the current one).
    with redirect_stdout(io.StringIO()):
        for cell in job.get("setup", []):
            try:
                exec(compile(cell, "<setup>", "exec"), namespace)
            except Exception:
                pass

    # 2. Run the code under test, capturing its output and any error.
    out = io.StringIO()
    error = None
    ok = True
    try:
        with redirect_stdout(out):
            exec(compile(job.get("code", ""), "<submission>", "exec"), namespace)
    except SystemExit:
        pass
    except Exception:
        ok = False
        error = _format_error()

    # 3. Extract requested variables.
    variables = {}
    missing = []
    for name in job.get("extract", []):
        if name in namespace:
            try:
                variables[name] = _encode(namespace[name])
            except Exception:
                variables[name] = {"t": "repr", "v": "<unencodable>"}
        else:
            missing.append(name)

    result = {
        "ok": ok,
        "stdout": out.getvalue(),
        "error": error,
        "vars": variables,
        "missing": missing,
    }
    sys.stdout.write(json.dumps(result))


def _format_error() -> str:
    """A student-friendly traceback: drop our exec frames, keep theirs."""
    exc_type, exc, tb = sys.exc_info()
    frames = traceback.extract_tb(tb)
    student_frames = [f for f in frames if f.filename == "<submission>"]
    lines = ["Traceback (most recent call last):\n"]
    lines += traceback.format_list(student_frames or frames)
    lines += traceback.format_exception_only(exc_type, exc)
    return "".join(lines).rstrip()


if __name__ == "__main__":
    main()
