"""ISS GNC Onboarding — the web app ("mini-PrairieLearn").

Run it:
    uv run flask --app app run --debug     # or: uv run python app.py

Then open http://127.0.0.1:5000 . Modules are auto-discovered from notebooks/;
add or edit an .ipynb and refresh (restart in non-debug mode).
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request

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


@app.context_processor
def inject_globals() -> dict:
    return {"pygments_css": pygments_css()}


@app.route("/")
def index():
    modules = _ordered_modules()
    counts = progress.module_counts(_student(), modules)
    return render_template(
        "index.html", modules=modules, counts=counts, student=_student()
    )


@app.route("/m/<module_id>")
def module_page(module_id: str):
    module = _modules.get(module_id)
    if module is None:
        abort(404)
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
    result = grader.run_only(module, exercise, data.get("code", ""))
    return jsonify(ok=result.ok, stdout=result.stdout, error=result.error)


@app.post("/grade")
def grade():
    module, exercise, data = _lookup()
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
