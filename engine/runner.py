"""Sandboxed executor — runs in a *subprocess*, never in the web server.

Reads a job (JSON) from the file named in argv[1], executes it, and prints a
single JSON result object to stdout. The job has:

    {
      "setup":   ["<code cell>", ...],   # notebook state, run best-effort
      "code":    "<code to run/grade>",  # student submission or reference
      "extract": ["var1", "var2"],       # variable names to capture
      "checker": "def check(ctx): ..."   # optional behaviour checker (see below)
    }

Result:

    {
      "ok": bool,                # did `code` run without raising?
      "stdout": "...",           # captured prints from `code`
      "error": "traceback|null",
      "vars": {name: <encoded>}, # captured values for names that existed
      "missing": ["name", ...],  # requested names that were never defined
      "checks": [{"label", "ok", "message", "expected", "got"}, ...],
      "figures": ["data:image/png;base64,...", ...]   # matplotlib output
    }

A *behaviour checker* is lead-authored code defining `check(ctx)`. It runs here,
right after the submission, so it can see everything about it — the printed
output, the live namespace, the AST — rather than one extracted value. That is
what makes open-ended exercises ("simulate a rocket, your numbers are up to
you") gradeable. See `CheckContext` for the API it gets.

Isolation is deliberately lightweight: a CPU-time + address-space rlimit plus a
wall-clock timeout enforced by the parent. This is an internal onboarding tool
for trusted teammates, not a hostile-code sandbox — see README security notes.
"""

from __future__ import annotations

import ast
import base64
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


MAX_FIGURES = 8


def _close_figures() -> None:
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None:
        try:
            plt.close("all")
        except Exception:
            pass


def _capture_figures() -> list[str]:
    """Any matplotlib figures the code left open, as PNG data: URIs.

    The subprocess runs under the Agg backend, so `plt.show()` is a no-op and
    the figures just stay open — rendering them here is the only way a plot
    ever reaches the page.
    """
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return []
    images: list[str] = []
    try:
        for num in plt.get_fignums()[:MAX_FIGURES]:
            buf = io.BytesIO()
            plt.figure(num).savefig(buf, format="png", dpi=110, bbox_inches="tight")
            images.append(
                "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
            )
    except Exception:
        pass  # a broken figure must not sink an otherwise good run
    _close_figures()
    return images


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


class CheckContext:
    """Everything a behaviour checker can see about one submission.

    Passed to the lead's `check(ctx)`. Attributes:

        ctx.ok        did the submission run without raising?
        ctx.error     its traceback, if it did raise
        ctx.source    the submitted code, as text
        ctx.stdout    everything it printed
        ctx.lines     stdout as stripped, non-blank lines
        ctx.env       the namespace it left behind
        ctx.tree      its AST (empty module if the code doesn't parse)
        ctx.assigned  names the submission itself assigns (not inherited setup)

    Methods:

        ctx.require(label, condition, message="", expected=None, got=None)
                      record one pass/fail row; returns the bool so you can
                      `if not ctx.require(...): return` to stop early
        ctx.uses(kind) / ctx.count(kind)
                      look for "while" / "for" / "if" / "def" / "print" / ...
        ctx.defined(*names) -> the first of `names` the submission assigned
        ctx.get(*names, default=None) -> that name's value
    """

    NODES = {
        "while": ast.While,
        "for": ast.For,
        "if": ast.If,
        "def": ast.FunctionDef,
        "class": ast.ClassDef,
        "try": ast.Try,
        "fstring": ast.JoinedStr,
        "list": ast.List,
        "dict": ast.Dict,
        "import": (ast.Import, ast.ImportFrom),
        "comprehension": (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
    }

    def __init__(self, source: str, stdout: str, env: dict, ok: bool, error: str | None):
        self.source = source
        self.stdout = stdout
        self.lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
        self.env = env
        self.ok = ok
        self.error = error
        self.results: list[dict] = []
        try:
            self.tree = ast.parse(source)
        except SyntaxError:
            self.tree = ast.Module(body=[], type_ignores=[])
        self.assigned = _assigned_names(self.tree)

    def require(self, label, condition, message="", expected=None, got=None) -> bool:
        ok = bool(condition)
        self.results.append({
            "label": str(label),
            "ok": ok,
            "message": "" if ok else str(message),
            "expected": None if ok else _text(expected),
            "got": None if ok else _text(got),
        })
        return ok

    def count(self, kind: str) -> int:
        if kind == "print":
            return sum(
                1 for n in ast.walk(self.tree)
                if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "print"
            )
        node = self.NODES.get(kind)
        if node is None:
            return 0
        return sum(1 for n in ast.walk(self.tree) if isinstance(n, node))

    def uses(self, kind: str) -> bool:
        return self.count(kind) > 0

    def defined(self, *names: str) -> str | None:
        """First of `names` the submission assigns *and* leaves in the namespace.

        Requiring the assignment to be in the submission matters: earlier
        notebook cells are replayed as setup, so plain `name in env` would
        happily match a variable the student never wrote.
        """
        for name in names:
            if name in self.assigned and name in self.env:
                return name
        return None

    def get(self, *names: str, default=None):
        name = self.defined(*names)
        return self.env[name] if name else default


def _text(value) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else repr(value)


def _assigned_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()

    def bind(target):
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                bind(elt)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bind(target)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For)):
            bind(node.target)
        elif isinstance(node, ast.NamedExpr):
            bind(node.target)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _run_checker(checker: str, ctx: CheckContext) -> list[dict]:
    """Execute the lead's checker against `ctx`, in its own namespace."""
    def crashed(detail: str) -> list[dict]:
        return ctx.results + [{
            "label": "Automatic checks",
            "ok": False,
            "message": f"This exercise's checker {detail}. Tell the GNC lead.",
            "expected": None,
            "got": None,
        }]

    namespace: dict = {"__name__": "__checker__"}
    try:
        with redirect_stdout(io.StringIO()):  # a checker's prints aren't the student's
            exec(compile(checker, "<checker>", "exec"), namespace)
            check = namespace.get("check")
            if not callable(check):
                return crashed("defines no check(ctx) function")
            check(ctx)
    except Exception as exc:
        return crashed(f"crashed ({type(exc).__name__}: {exc})")
    return ctx.results


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
    # Earlier cells plot too; drop theirs so only this cell's figures show.
    _close_figures()

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
    figures = _capture_figures()

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

    # 4. Run the exercise's behaviour checker, if it has one, over what just
    #    happened (output + namespace + source).
    checks: list[dict] = []
    checker = job.get("checker") or ""
    if checker.strip():
        ctx = CheckContext(job.get("code", ""), out.getvalue(), namespace, ok, error)
        checks = _run_checker(checker, ctx)

    result = {
        "ok": ok,
        "stdout": out.getvalue(),
        "error": error,
        "vars": variables,
        "missing": missing,
        "checks": checks,
        "figures": figures,
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
