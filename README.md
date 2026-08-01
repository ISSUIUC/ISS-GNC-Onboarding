# ISS GNC Onboarding Engine

A tiny, self-contained **"mini-PrairieLearn"** for the Illinois Space Society
Avionics **GNC** subteam. The GNC lead writes ordinary Jupyter notebooks; this
engine turns each one into an interactive, **auto-graded** onboarding module in
the browser — no PrairieLearn account, no Docker, no external service.

```
notebooks/*.ipynb  ──parser──▶  Module  ──renderer──▶  interactive web page
                                   │
                                   └──grader──▶  per-exercise pass/fail + feedback
```

Everything is plain Python + Flask so the team can read and extend it.

---

## Quick start

```bash
uv sync              # install deps into .venv (uses the pinned Python)
uv run python app.py # start the server
```

Open <http://127.0.0.1:5000>, type your name once, and work through the modules.

> First run downloads Flask/numpy/etc. via `uv`. The page pulls MathJax and
> CodeMirror from a CDN, so you need internet the first time you load it.

## How it works

| Piece | File | Job |
|-------|------|-----|
| Parser | `engine/parser.py` | Read an `.ipynb`; split teaching cells from exercises; strip solutions to make the student version. |
| Renderer | `engine/renderer.py` | Markdown → HTML with LaTeX (MathJax) and Pygments-highlighted code. |
| Runner | `engine/runner.py` | Execute student code in a **separate process** with a timeout, capture output + variables. |
| Grader | `engine/grader.py`, `engine/feedback.py` | Compare the student's variables/output to the reference solution with tolerances. |
| Progress | `engine/progress.py` | Per-student completion, saved to `data/progress.json`. |
| Web app | `app.py`, `templates/`, `static/` | Serve modules, run/grade endpoints. |

An exercise with no reference solution still works — it renders as a **practice**
cell the student can run. Add a solution + `#% check` later to make it graded.
See **[AUTHORING.md](AUTHORING.md)** for the (small) notebook convention.

## Adding a module

Drop a new `.ipynb` into `notebooks/` and restart the server. Modules are
listed in filename order. Mark exercises and (optionally) add graders following
[AUTHORING.md](AUTHORING.md).

## Security note

Student code runs in a subprocess with a CPU/memory limit and a wall-clock
timeout — enough to stop an accidental infinite loop. It is **not** a hardened
sandbox against hostile code. This is an internal tool for trusted teammates;
run it locally or on a trusted network, not as a public service.

---

*hi! please read the Introduction module first.*
*Save yourselves, don't end up like him:*

![Alt Text](Divij.png)
