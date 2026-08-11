# `notebooks/checkers/` — the behaviour checkers

Lead-authored Python that grades an exercise by inspecting what the submission
*did* — its printed output, its namespace, its AST — instead of diffing one
variable. The API (`check(ctx)` / `ctx.require(...)`) is documented in
AUTHORING.md → *Grading open-ended challenges*; `rocket_flight.md` is the worked
write-up of a full-size one.

## What grades what

One module per notebook, named after it, auto-discovered. `check_exN` binds to
the Nth exercise cell with no directive in the notebook; a `#% checker: name`
directive in the cell pins the binding by name instead — preferred for the final
challenges, since inserting a cell above one can't then re-point it.

| Notebook | Checker | Exercises | Bound by |
|---|---|---|---|
| `Introduction.ipynb` | `introduction.py` | ex1, ex2 | `check_ex1`, `check_ex2` |
| | | ex6 | `#% checker: silence_the_ifs` |
| | `rocket_flight.py` | ex7 | `#% checker: rocket_flight.py` |
| `Vectors.ipynb` | `vectors.py` | ex1 | `check_ex1` |
| `Linear_Algebra.ipynb` | `linear_algebra.py` | ex1, ex2 | `check_ex1`, `check_ex2` |
| | | ex3 | `#% checker: matrix_boss` |
| `basic_filters.ipynb` | `basic_filters.py` | ex1–ex3 | `check_ex1`…`check_ex3` |
| | | ex4 | `#% checker: filter_the_altitude` |
| `kalman_filter.ipynb` | `kalman_filter.py` | ex1, ex2 | `check_ex1`, `check_ex2` |
| | | ex3 | `#% checker: tune_the_filter` |
| `extended_kalman_filter.ipynb` | `extended_kalman_filter.py` | ex1 | `check_ex1` |
| | | ex2 | `#% checker: bearing_measurement` |

`introduction-ex3/4/5` and `vectors-ex2` aren't here — they have a reference
solution to diff against, so a plain `#% check:` / `#% check_output` directive
grades them and a behaviour checker would be the wrong tool.

## The one constraint that shapes these files

A checker's **source is read and `exec`'d standalone** inside the grading
subprocess (`engine/runner.py`). So:

- **Checker files cannot import each other.** Each module is self-contained;
  `import numpy as np`, `ast`, `re`, `math` are fine. Where two modules hold a
  near-identical helper, that duplication is deliberate, not an oversight.
- That's also why it's one module *per notebook* rather than one per exercise —
  a notebook's exercises share helpers (recomputing a filter from the student's
  own parameter, locating their arrays, detecting plots), and per-exercise files
  would have to copy them. A file of its own is reserved for a checker big
  enough to need nothing shared, which is what `rocket_flight.py` is.
- **Figures are already closed** when a checker runs — the runner captures and
  closes them first. `plt.get_fignums()` is empty; detect plotting by walking
  `ctx.tree` for the calls instead.
- `ctx.defined` / `ctx.get` only match names the *submission* assigns, so the
  replayed setup cells can't produce a false positive.
- A checker that raises replaces all student feedback with one "Automatic checks
  … crashed" row, so guard against `None`, wrong types and odd shapes throughout.

## Testing a change

```
uv run python notebooks/checkers/sweep.py           # all six notebooks
uv run python notebooks/checkers/sweep.py kalman    # just the Kalman ones
```

Grades every exercise's own reference solution and exits non-zero on a
regression. Four pre-existing authoring gaps are listed in the script's
`KNOWN_FAILURES` and reported as known rather than failing the run.

A checker that fails correct work is worse than one that's slightly weak — so
when you tighten something, also feed it two or three *correct* submissions
written differently (other variable names, other valid parameter values,
printed instead of assigned) and confirm they still pass.
