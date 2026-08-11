# `rocket_flight.py` — the rocket-flight checker

Grades the **Final Challenge** of the Introduction module (`introduction-ex7`),
attached to the exercise cell with:

```python
#% checker: rocket_flight.py
```

See AUTHORING.md → *Grading open-ended challenges* for the checker API itself.
This page is about what this particular checker decides, and why.

## Why it isn't just `check_output_contains`

The challenge is deliberately open-ended — the student picks their own climb
rate, tank size, burn rate and print formatting — so there's no single reference
answer to diff against. The original grading was:

```python
#% check_output_contains: Liftoff!, Low Fuel!, Engine Cutoff!
```

which three bare `print` statements satisfy. Nothing checked that a rocket was
ever simulated.

So instead of comparing values, this checker **reads the flight log back out of
the student's printed output and re-flies the mission from it**, then verifies
the simulation is self-consistent: the altitude climbs by *their own* velocity,
the fuel burn is steady and lands on empty, and each event fires on the right
second. Every number is theirs; only the physics has to hold together.

## What it checks

25 checks, each a row in the student's feedback. The score is the fraction that
pass, so partial work gets partial credit.

**Does it run** — `Runs without an error`, `Prints a flight log`. Both bail out
early: no point reporting fifteen failures about a log that was never printed.

**Uses the tools the challenge teaches**
| Check | Fails when |
|---|---|
| Uses a `while` loop to fly the rocket | no `while` in the AST |
| Uses `if` statements for the flight events | fewer than 2 `if`s (low fuel + cutoff) |
| The log is printed from inside the loop | no `print` call inside the `while` body |
| Stores the starting altitude / vertical velocity / fuel in a variable | no recognised name assigned (3 rows) |

Together these are the anti-hard-code net: a transcript of `print` statements
that *looks* like a flight log fails all of them.

**Reads the telemetry back** — `Prints the altitude every second` and `Prints the
fuel remaining every second` (≥3 readings each; both bail out early, with a
message showing the expected `print(f"Altitude: {altitude} m")` form), plus
`Reports altitude and fuel together each second` (equal counts).

**The physics holds together**
| Check | Fails when |
|---|---|
| The rocket climbs every second | altitude isn't strictly increasing |
| It climbs by the vertical velocity each second | the climb per second isn't constant |
| Altitude gain matches the velocity variable | the constant climb ≠ their own `velocity` |
| Fuel drops every second | fuel isn't strictly decreasing |
| Fuel burns at a steady rate | the burn per second isn't constant |
| Fuel never goes negative | any reading below 0 |
| The tank ends up empty | the last reading isn't exactly 0 |

*Altitude gain matches the velocity variable* is the one that catches
`altitude += 7` alongside `velocity = 50`. It's skipped (not failed) if the
velocity variable doesn't hold a number.

**The events fire at the right moment**
| Check | Fails when |
|---|---|
| Announces "Liftoff!" before the first second | missing, or printed after the first telemetry line |
| Warns "Low Fuel!" when the tank drops below 20% | never printed |
| The warning only appears below 20% | warned while the tank was still above threshold |
| The warning appears as soon as the fuel is low | warned, but later than the first low second |
| Announces "Engine Cutoff!" when the fuel is gone | never printed |
| Cutoff is announced exactly once | printed 0 or 2+ times |
| Cutoff happens at zero fuel | announced on a second where fuel isn't 0 |

The 20% threshold is **their** 20%: `_tank_size` reads the first number they
assigned to the fuel variable, falling back to reconstructing it from the log
(first reading + one burn). A student with `fuel = 60` is checked against 12.

## Design decisions worth knowing

**Ambiguity goes to the student.** A `Low Fuel!` line sitting between two
telemetry lines could belong to either second — students print the warning
before or after that second's numbers, both reasonable. `_steps_around` returns
*both* neighbouring seconds and the check passes if either qualifies. Being
generous by one second is the right direction to be wrong; it still catches an
unconditional warning, whose very first occurrence has no low neighbour at all.

**Variable names vs. log labels are separate lists.** `ALTITUDE_VARS` is generous
(`altitude`, `alt`, `height`, `h`, `y`, …) because `ctx.defined` only matches
names *the student's own code assigns* — replayed setup cells from earlier in the
notebook can't produce a false positive. `ALTITUDE_LABELS`, used to parse their
printed output, is stricter: `h` would happily match `Time: 1 h`.

**Event lines are excluded from number parsing**, so `Low Fuel! 10% left` can't
sneak in as an extra fuel reading and break the steady-burn check.

**The checker depends on labelled output, so the prompt promises it.** The
markdown cell above the exercise tells students the grader reads their log and
that numbers must be labelled like the worked example. A checker that relies on
something the prompt doesn't state is a trap, not a check.

## Tuning it

Everything adjustable is a constant at the top of the file:

- `ALTITUDE_VARS` / `VELOCITY_VARS` / `FUEL_VARS` — accepted variable names. **This
  is the one place the checker can be wrong in the strict direction**: a student
  whose naming falls outside these lists fails the three "stores it in a
  variable" rows despite correct code. The failure message names the convention;
  widen the tuple if it comes up.
- `ALTITUDE_LABELS` / `FUEL_LABELS` — accepted labels in the printed log.
- `LOW_FUEL_FRACTION` — currently `0.20`, matching the prompt.
- `EVENTS` — the regexes for the three announcements. Already lenient about
  case and spacing (`Engine cut-off`, `LIFTOFF!` both match).
- `TOL` — float comparison tolerance, `1e-6`.

## Verifying a change

The checker is plain importable Python with no engine imports, but the quickest
real test is to grade sample submissions through the engine:

```python
from engine import grader
from engine.parser import parse_notebook

module = parse_notebook("notebooks/Introduction.ipynb")
ex = module.exercise("introduction-ex7")

result = grader.grade(module, ex, SUBMISSION)          # ex.reference_code to smoke-test
print(result.passed, result.score)
for c in result.checks:
    print("PASS" if c.ok else "FAIL", c.label, c.message)
```

Behaviour confirmed against these submissions:

| Submission | Result |
|---|---|
| The exercise's own reference solution | passes, 25/25 |
| Different names, numbers and print format (`h`, `climb_rate`, one line per second) | passes, 25/25 |
| Three bare `print`s | 0.22 — every structural check fails |
| Hard-coded log, no loop | 0.71 — loop/variable checks fail, uneven burn caught |
| `Low Fuel!` printed unconditionally every second | 0.92 |
| `Low Fuel!` printed too late (`if fuel < 5`) | 0.96 |
| Fuel burns past zero into negatives | 0.84 |
| `altitude += 7` with `velocity = 50` | 0.96 |
| No liftoff, unlabelled numbers | 0.78 |
| Raises an exception | 0.00, reports the error |
| Infinite loop | 0.00, reports that it didn't finish |

Re-run these after any edit — the two "should pass" rows are the ones that
matter most. A checker that fails correct work is worse than a weak one.
