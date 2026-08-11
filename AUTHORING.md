# Authoring modules (for the GNC lead)

You write a **normal Jupyter notebook**. The engine turns it into a module. This
page is the entire convention — it's short on purpose.

## The mental model

Each notebook = one module. Cells become, in order:

| Cell | Becomes |
|------|---------|
| Markdown | Teaching content (rendered with LaTeX — `$...$` and `\begin{bmatrix}...` both work). |
| Plain code cell | A read-only **worked example** (its saved output is shown too). It also becomes shared *setup* for later exercises — imports and variables carry forward, just like running the notebook top to bottom. |
| **Exercise** code cell | An **interactive editor** the student fills in. |

A code cell is treated as an **exercise** when any of these is true:

- it has a `#% exercise` directive, **or**
- the markdown cell just above it says "Your Turn" or "Challenge", **or**
- it contains a `# write ... here` placeholder.

So the notebooks you already write are *mostly* exercises-ready. To make an
exercise **auto-graded**, add a solution and say what to check.

## Making an exercise graded

Add two things to the exercise cell:

1. A `#% check:` directive listing the variable(s) to grade.
2. A `### BEGIN SOLUTION ... ### END SOLUTION` block with the correct answer.

```python
# Compute the products and store them in sol1..sol4
#% exercise
#% check: sol1, sol2, sol3, sol4
### BEGIN SOLUTION
sol1 = 15 * 8
sol2 = 144 / 12
sol3 = 2 ** 10
sol4 = 10 % 12
### END SOLUTION
```

The student sees only the comment. Behind the scenes the engine runs your
solution to get the expected values, runs the student's code, and compares.

### Grading printed output instead of variables

```python
#% exercise
#% check_output
### BEGIN SOLUTION
print(5 > 3, 5 < 3, 5 == 2, 5 > 1)   # prints: True False False True
### END SOLUTION
```

Or require certain phrases to appear:

```python
#% check_output_contains: Liftoff!, Low Fuel!, Engine Cutoff!
```

That one's a low bar on purpose — three bare `print`s satisfy it. When a task is
open-ended enough that you'd want more, reach for a behaviour checker (below).

### "Fix this code" exercises

Show the student some starter/buggy code with a `### BEGIN STUB` block. It
appears **only** in the editor; your solution replaces it when grading:

```python
#% exercise
#% check_output
### BEGIN STUB
print(5 > 3, 5 == 5, 5 != 2, 5 <= 1)   # student edits this line
### END STUB
### BEGIN SOLUTION
print(5 > 3, 5 < 3, 5 == 2, 5 > 1)
### END SOLUTION
```

## Grading open-ended challenges (behaviour checkers)

Some tasks have no single right answer — *"simulate a rocket; pick your own
climb rate and tank size"*. Checking variables or exact output can't grade those,
and `check_output_contains` is too weak (three bare `print`s pass it). For these,
write a **behaviour checker**: your own Python that inspects the submission and
reports named checks.

### One checker file per notebook (the usual way)

Most checkers are five lines, and a notebook has several. Put them all in
**`notebooks/checkers/<notebook>.py`** — one function per exercise, named after
it — and the cells need **no directive at all**:

```python
# notebooks/checkers/introduction.py
def check_ex1(ctx):                     # grades introduction-ex1
    ctx.require("The message is your own", ctx.stdout.strip() != "Hello GNC Team!",
                "Change the text inside the quotation marks.")

def check_ex6(ctx):                     # grades introduction-ex6
    ctx.require("Nothing is printed", not ctx.stdout.strip(), "Your code still prints.")
```

The file name matches the notebook (`Introduction.ipynb` → `introduction.py`),
and the function name matches the exercise id: `introduction-ex6` → `check_ex6`,
and a cell with `#% id: rocket-gains` → `check_rocket_gains`. Anything else in
the file (constants, helpers) is just shared code — it's an ordinary Python
module you can import in a test.

**The `exN` number is positional** — it's the Nth exercise cell from the top of
the notebook, so inserting an exercise above one shifts every number below it.
When you'd rather not think about that, name the checker in the cell:

```python
#% exercise
#% checker: silence_the_ifs      # -> check_silence_the_ifs() in introduction.py
```

Now the cell says which function grades it, reordering can't re-point it, and
the name can be as descriptive as you like. (`#% id:` also pins the binding, but
it's the key student progress is stored under — renaming one resets who's
completed that exercise. `#% checker:` has no such cost.) The `check_` prefix is
optional in the directive: `#% checker: check_silence_the_ifs` works too.

This is what upgrades a *practice* cell into a graded one: as soon as a
`check_exN` exists, that exercise stops passing on "it ran" alone.

### One checker file per exercise

When a single checker gets big, give it its own file and point the cell at it:

```python
#% exercise
#% checker: rocket_flight.py
### BEGIN SOLUTION
...                     # still worth having: it's run as setup for later cells
### END SOLUTION
```

`#% checker: shared.py:check_gains` picks one function out of a file that holds
several — handy when two notebooks share a checker.

Or write it inline for something short (the region is invisible to the student
and never runs as their code):

```python
### BEGIN CHECKER
def check(ctx):
    ctx.require("Uses a loop", ctx.uses("while"), "A row of prints isn't a simulation.")
    ctx.require("Counts down to zero", ctx.get("fuel") == 0)
### END CHECKER
```

Your checker defines `check(ctx)` and calls `ctx.require(...)` once per thing you
want to verify — each one becomes a green/red row in the student's feedback, and
the score is the fraction that pass. It runs in the same sandboxed subprocess,
immediately after the submission, so it sees everything:

| | |
|---|---|
| `ctx.ok` / `ctx.error` | did their code run; the traceback if not |
| `ctx.stdout` / `ctx.lines` | everything they printed (`lines` = stripped, non-blank) |
| `ctx.env` | the namespace they left behind |
| `ctx.source` / `ctx.tree` | their code as text / as an AST |
| `ctx.assigned` | names *their* code assigns (not inherited from setup cells) |
| `ctx.uses(kind)` / `ctx.count(kind)` | `"while"`, `"for"`, `"if"`, `"def"`, `"print"`, `"fstring"`, … |
| `ctx.defined(*names)` | the first of `names` they actually assigned, else `None` |
| `ctx.get(*names, default=None)` | that variable's value |

```python
ctx.require(label, condition, message="", expected=None, got=None)  # -> bool
```

`message`, `expected` and `got` are only shown when the check fails. `require`
returns the condition, so you can bail out early instead of piling on confusing
follow-up failures:

```python
if not ctx.require("Prints a flight log", ctx.lines, "Nothing was printed."):
    return
```

When an exercise has more than one of these, the most specific wins:
`#% checker:` file → inline `### BEGIN CHECKER` → the notebook's checker module.

`notebooks/checkers/rocket_flight.py` is the worked example (written up in
`notebooks/checkers/rocket_flight.md`): it reads the
altitude and fuel numbers back out of the student's printed log and re-flies the
mission from them, checking the climb matches their own velocity, the fuel burn
is steady and lands on zero, and each event fires at the right second. Two habits
from it worth copying — judge ambiguous cases in the student's favour, and say
in the prompt whatever your checker relies on (there, that the numbers are
labelled).

Guardrails: a checker that crashes, a `#% checker:` filename that doesn't exist,
or a checker file with a syntax error shows the student one failed "Automatic
checks" row telling them to report it — it never silently marks the exercise
ungraded.

## All directives

Directives are comment lines starting with `#%`. They never appear to the
student and never run.

| Directive | Meaning |
|-----------|---------|
| `#% exercise` | Force this cell to be an exercise. |
| `#% check: a, b~0.001, c` | Grade these variables. `~tol` sets an absolute tolerance (default is a relative `1e-6`). |
| `#% check_output` | Require stdout to match the reference (whitespace-lenient). |
| `#% check_output_contains: x, y` | Require these substrings in stdout. |
| `#% checker: rocket_flight.py` | Grade with a behaviour checker file from `notebooks/checkers/`. Add `:function` to pick one out of a multi-checker file. |
| `#% checker: silence_the_ifs` | No `.py` = a function in this notebook's own checker module (`check_` prefix optional). Use it to bind a checker by name instead of by cell position. |
| `#% points: 2` | Weight for this exercise (default 1). |
| `#% title: Dot products` | Override the exercise title. |
| `#% id: vectors-dot` | Stable id (default `<module>-ex<N>`). Set this if you reorder cells and want progress to stick. |
| `#% reveal: false` | Hide the expected value in feedback (default shows it, which is friendlier for onboarding). |

### Code regions

| Marker | Appears in student editor | Runs when grading |
|--------|:--:|:--:|
| `### BEGIN SOLUTION` … `### END SOLUTION` | no | yes |
| `### BEGIN STUB` … `### END STUB` | yes | no |
| `### BEGIN CHECKER` … `### END CHECKER` | no | as the checker, after the submission |
| everything else in the cell | yes | yes |

## What gets checked, by type

The comparison adapts to the reference value's type:

- **numbers** → close within tolerance (so `12` matches `12.0`);
- **numpy arrays** → shape match + `np.allclose`;
- **lists / tuples / dicts** → compared element-by-element;
- **strings / sets** → exact.

## Tips

- Variables you check must be ones the student can reasonably name. Tell them
  the names in the prompt (e.g. *"store the magnitude in `mag`"*).
- Keep each exercise's needed inputs defined either in an earlier example cell
  or inside the exercise's own starter code.
- Preview your work: `uv run python app.py`, then open the module. Editing a
  notebook needs a server restart to re-read it.
- An exercise with no solution and no checker is still useful — it's a runnable
  practice cell, marked complete once it runs without an error. Add a
  `check_exN` to the notebook's checker module whenever you're ready for it to
  be graded properly; the cell itself doesn't change. (As of now every exercise
  in every notebook is graded, so there are none left in this state.)
- **Regression test after any edit**: `uv run python notebooks/checkers/sweep.py`
  grades every exercise's own reference solution and exits non-zero on a
  regression. If the reference can't pass, no student can. Add `kalman` (or any
  substring) to sweep one notebook. `notebooks/checkers/README.md` is the index
  of which checker grades what, and the constraints they're written under.
- Checker modules are plain Python — `uv run python -m pytest` over them, or
  just import one and call `check(ctx)` with a fake ctx, if a checker gets
  hairy enough to be worth a test.
