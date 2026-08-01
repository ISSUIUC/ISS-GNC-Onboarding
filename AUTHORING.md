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

Or require certain phrases to appear (great for open-ended challenges):

```python
#% check_output_contains: Liftoff!, Low Fuel!, Engine Cutoff!
```

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

## All directives

Directives are comment lines starting with `#%`. They never appear to the
student and never run.

| Directive | Meaning |
|-----------|---------|
| `#% exercise` | Force this cell to be an exercise. |
| `#% check: a, b~0.001, c` | Grade these variables. `~tol` sets an absolute tolerance (default is a relative `1e-6`). |
| `#% check_output` | Require stdout to match the reference (whitespace-lenient). |
| `#% check_output_contains: x, y` | Require these substrings in stdout. |
| `#% points: 2` | Weight for this exercise (default 1). |
| `#% title: Dot products` | Override the exercise title. |
| `#% id: vectors-dot` | Stable id (default `<module>-ex<N>`). Set this if you reorder cells and want progress to stick. |
| `#% reveal: false` | Hide the expected value in feedback (default shows it, which is friendlier for onboarding). |

### Code regions

| Marker | Appears in student editor | Runs when grading |
|--------|:--:|:--:|
| `### BEGIN SOLUTION` … `### END SOLUTION` | no | yes |
| `### BEGIN STUB` … `### END STUB` | yes | no |
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
- An exercise with no solution is still useful — it's a runnable practice cell.
  Add grading whenever you're ready.
