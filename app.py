"""ISS GNC Onboarding — the web app ("mini-PrairieLearn").

Run it:
    uv run flask --app app run --debug     # or: uv run python app.py

Then open http://127.0.0.1:5000 . Modules are auto-discovered from notebooks/;
add or edit an .ipynb and refresh (restart in non-debug mode).
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path

from flask import (Flask, abort, jsonify, redirect, render_template, request,
                   send_from_directory, url_for)

from engine import discover_modules, grader
from engine.progress import Progress
from engine.renderer import pygments_css, render_blocks

BASE = Path(__file__).parent
NOTEBOOKS = BASE / "notebooks"

# Shared code that unlocks every module (for instructors demoing a later module,
# or a student who needs to skip ahead). Override with GNC_ADMIN_CODE if you want
# something less guessable. This is convenience gating, not security: progress is
# still tracked per student, and anyone with the code can unlock everything.
ADMIN_CODE = os.environ.get("GNC_ADMIN_CODE", "gnc-admin")
ADMIN_COOKIE = "gnc_admin"

app = Flask(__name__)

_modules: dict = {}


def load_modules() -> None:
    global _modules
    _modules = {m.id: m for m in discover_modules(NOTEBOOKS)}


load_modules()
progress = Progress(BASE / "data" / "progress.json")


# Teaching order. discover_modules() sorts by filename, which reads
# extended_kalman_filter before kalman_filter — backwards. Anything not listed
# here falls to the end alphabetically, so a new notebook still shows up. This
# order also drives the sequential unlock in _gating().
MODULE_ORDER = [
    "introduction",
    "linear_algebra",
    "vectors",
    "basic_filters",
    "kalman_filter",
    "extended_kalman_filter",
]


def _ordered_modules() -> list:
    rank = {module_id: i for i, module_id in enumerate(MODULE_ORDER)}
    return sorted(_modules.values(), key=lambda m: (rank.get(m.id, len(rank)), m.id))


def _student() -> str:
    return (request.cookies.get("student") or "").strip() or "anonymous"


def _is_admin() -> bool:
    """True if this browser has entered the admin code (cookie set by /admin)."""
    return hmac.compare_digest(request.cookies.get(ADMIN_COOKIE, ""), ADMIN_CODE)


def _gating() -> dict:
    """Per-module {done, total, complete, unlocked, blocked_by} for this student."""
    return progress.module_counts(_student(), _ordered_modules(), unlock_all=_is_admin())


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
    return {"pygments_css": pygments_css(), "admin": _is_admin()}


@app.post("/admin")
def admin():
    """Turn the unlock-everything override on (with the code) or off.

    Body: {"code": "..."} to enable, {"off": true} to disable.
    """
    data = request.get_json(force=True, silent=True) or {}
    if data.get("off"):
        resp = jsonify(admin=False)
        resp.delete_cookie(ADMIN_COOKIE, path="/")
        return resp
    code = (data.get("code") or "").strip()
    if not hmac.compare_digest(code, ADMIN_CODE):
        return jsonify(admin=False, error="Incorrect admin code."), 403
    resp = jsonify(admin=True)
    resp.set_cookie(ADMIN_COOKIE, ADMIN_CODE, max_age=60 * 60 * 12, path="/", samesite="Lax")
    return resp


@app.post("/reset")
def reset():
    """Wipe saved progress.

    Body: {"all": true} to empty progress.json entirely, otherwise just this
    browser's student is cleared. The UI confirms before calling either way.
    """
    data = request.get_json(force=True, silent=True) or {}
    if data.get("all"):
        progress.reset()
        return jsonify(ok=True, scope="all")
    progress.reset(_student())
    return jsonify(ok=True, scope=_student())


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


# Notebook images are plain relative links (`![](foo.png)`), so the browser
# requests them alongside the module page, under /m/. Serve those from
# notebooks/ instead of reading them as a module id.
NOTEBOOK_ASSETS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}


@app.route("/m/<module_id>")
def module_page(module_id: str):
    if Path(module_id).suffix.lower() in NOTEBOOK_ASSETS:
        return send_from_directory(NOTEBOOKS, module_id)
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
