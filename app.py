"""ISS GNC Onboarding — the web app ("mini-PrairieLearn").

Run it:
    uv run flask --app app run --debug     # or: uv run python app.py

Then open http://127.0.0.1:5000 . Modules are auto-discovered from notebooks/;
add or edit an .ipynb and refresh (restart in non-debug mode).
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for

from engine import discover_modules, grader
from engine.progress import Progress
from engine.renderer import pygments_css, render_blocks

BASE = Path(__file__).parent
NOTEBOOKS = BASE / "notebooks"

app = Flask(__name__)

_modules: dict = {}


def load_modules() -> None:
    global _modules
    _modules = {m.id: m for m in discover_modules(NOTEBOOKS)}


load_modules()
progress = Progress(BASE / "data" / "progress.json")


def _ordered_modules() -> list:
    return list(_modules.values())


def _student() -> str:
    return (request.cookies.get("student") or "").strip() or "anonymous"


def _gating() -> dict:
    """Per-module {done, total, complete, unlocked, blocked_by} for this student."""
    return progress.module_counts(_student(), _ordered_modules())


def _locked(module_id: str) -> str | None:
    """Title of the module blocking `module_id`, or None if it's open."""
    status = _gating().get(module_id)
    if status is None or status["unlocked"]:
        return None
    return status["blocked_by"] or "the previous module"


def _lock_message(blocker: str) -> str:
    return f"This module is locked. Finish “{blocker}” first."


@app.context_processor
def inject_globals() -> dict:
    return {"pygments_css": pygments_css()}


@app.route("/")
def index():
    modules = _ordered_modules()
    counts = _gating()
    # ?locked=<id> comes from a redirect off a module the student can't open yet.
    requested = _modules.get(request.args.get("locked", ""))
    notice = None
    if requested is not None and not counts[requested.id]["unlocked"]:
        notice = {
            "module": requested.title,
            "blocker": counts[requested.id]["blocked_by"],
        }
    return render_template(
        "index.html", modules=modules, counts=counts, student=_student(), locked_notice=notice
    )


@app.route("/m/<module_id>")
def module_page(module_id: str):
    module = _modules.get(module_id)
    if module is None:
        abort(404)
    if _locked(module_id):
        # Direct-URL access to a module the student hasn't unlocked yet.
        return redirect(url_for("index", locked=module_id))
    blocks = render_blocks(module)
    done = progress.completed(_student(), module_id)
    return render_template(
        "module.html", module=module, blocks=blocks, done=done, student=_student()
    )


def _lookup():
    data = request.get_json(force=True, silent=True) or {}
    module = _modules.get(data.get("module"))
    exercise = module.exercise(data.get("exercise")) if module else None
    if module is None or exercise is None:
        abort(404)
    return module, exercise, data


@app.post("/run")
def run():
    module, exercise, data = _lookup()
    if (blocker := _locked(module.id)) is not None:
        return jsonify(ok=False, stdout="", error=_lock_message(blocker)), 403
    result = grader.run_only(module, exercise, data.get("code", ""))
    return jsonify(ok=result.ok, stdout=result.stdout, error=result.error)


@app.post("/grade")
def grade():
    module, exercise, data = _lookup()
    if (blocker := _locked(module.id)) is not None:
        return jsonify(
            graded=exercise.graded,
            passed=False,
            score=0.0,
            stdout="",
            error=_lock_message(blocker),
            checks=[],
        ), 403
    result = grader.grade(module, exercise, data.get("code", ""))
    if result.passed:
        progress.mark(_student(), module.id, exercise.id, result.score)
    return jsonify(
        graded=exercise.graded,
        passed=result.passed,
        score=result.score,
        stdout=result.stdout,
        error=result.error,
        checks=[
            {
                "label": c.label,
                "ok": c.ok,
                "message": c.message,
                "expected": c.expected,
                "got": c.got,
            }
            for c in result.checks
        ],
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
