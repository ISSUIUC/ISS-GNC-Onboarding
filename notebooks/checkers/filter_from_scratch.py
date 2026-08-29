"""Behaviour checkers for filter_from_scratch.ipynb.

The last module takes the training wheels off: the student writes a whole
filter — `__init__`, `predict`, `update` — and runs it through the SILSIM cell
at the bottom. There is deliberately no reference answer to diff against. An
alpha-beta filter, a complementary filter and a full Kalman filter are all
correct here, every one of them with different tuning attributes, so anything
that graded a particular design would be grading the wrong thing.

What these three grade instead is that a filter was actually *written*: the
class carries the student's own parameters rather than the one bare
`self.state` the cell ships with, `predict` propagates the state out of the
filter's own attributes, and `update` genuinely depends on the measurement it
is handed. Each of the two step functions is then run once against a stand-in
filter, and the six numbers that come back have to be finite and not the zeros
the stub left behind.

Two constraints shape the file.

**There is no end-to-end check, on purpose.** Grading replays *reference*
solutions for the cells above the one being graded (`Module.setup_code`), never
what the student wrote two cells up — so a checker on the SILSIM cell would be
flying the reference filter and would pass no matter what the student did. The
SILSIM cell is therefore left as a plain runnable cell: on the page it runs
with the student's own live edits (`run_cell_with_live_setup`), which is where
they see whether their filter actually flies. The notebook's stated RMS goal
isn't graded either — `static/SAWA_Decimate.csv` is time plus three
accelerations with no truth trajectory to take an error against.

**The stand-in `self`.** For the same replay reason, the `filter_scratch` in
the namespace while `predict` is being graded was built by the reference
`__init__` and knows nothing about the student's attributes. So the step
function is handed a `_Stand` that invents any attribute it reaches for, and a
probe that still won't run is dropped rather than reported — see `_probe`. That
is the deliberate soft spot in this file: a step function whose shapes are
wrong loses its dynamic row instead of failing it. The student sees that in the
SILSIM cell, which is the honest place for it.
"""

import ast

import numpy as np


# --- the shapes the notebook fixes ----------------------------------------
STATE_SIZE = 6        # [position_x, velocity_x, position_y, velocity_y, position_z, velocity_z]
MEASUREMENT_SIZE = 4  # [t, ax, ay, az]

# Positions at the origin, a different velocity on each axis. Propagating this
# has to move *something*, whatever model the student chose.
SEED_STATE = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0])

# Two samples a tenth of a second apart that differ in every acceleration
# channel, so an `update` that reads its measurement can't land twice in the
# same place.
SAMPLE_A = np.array([1.000, 12.0, -3.0, 4.0])
SAMPLE_B = np.array([1.095, -8.0, 9.0, -6.0])

# What a wrongly-guessed attribute raises. These are never reported against the
# student: we can't tell our bad guess from their bug, so we say nothing.
FORGIVEN = (AttributeError, TypeError, ValueError, KeyError, IndexError)

# Filled in for a required constructor argument we know nothing about. Small,
# positive and non-integral: a plausible gain, and it can't be mistaken for an
# index or a count that happens to work.
FILLER_ARG = 0.1

# Attributes whose *name* says what shape they are. Anything else becomes a
# scalar, which is what most tuning knobs are.
VECTOR6_NAMES = {"state", "x", "x_hat", "xhat", "estimate", "state_estimate", "mu"}
VECTOR3_NAMES = {"accel", "acceleration", "accel_est", "a_est", "bias", "offset",
                 "gravity_vector", "g_vec", "position", "velocity", "pos", "vel"}
MATRIX_NAMES = {"p", "q", "r", "f", "h", "k", "a", "b", "phi", "cov", "covariance",
                "process_noise", "measurement_noise", "gain_matrix"}

# Closer than this to zero on every element and the state *is* the stub's zeros.
ZERO_EPS = 1e-12
# Two runs whose numbers agree this closely are the same run.
SAME_RTOL = 1e-9
SAME_ATOL = 1e-12


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _ran(ctx, hint="Fix the error shown in the output above, then check again."):
    """Every checker starts here: nothing else is meaningful if the cell crashed."""
    return ctx.require("Runs without an error", ctx.ok, hint)


def _function(tree, name):
    """The `def name(...)` node in the submission, at any nesting depth."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _class(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _attr_target(target):
    """The attribute name in `self.x = ...`, `self.x[i] = ...`, `self.x += ...`."""
    while isinstance(target, (ast.Subscript, ast.Starred)):
        target = target.value
    if (isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name) and target.value.id == "self"):
        return target.attr
    return None


def _self_writes(node):
    """Attributes the function assigns on `self`.

    `self.state, self.P = predicted, cov` counts as writing both, so the stack
    rather than a plain loop: an unpacking target holds more targets inside it.
    """
    names = set()
    pending = []
    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            pending.extend(child.targets)
        elif isinstance(child, (ast.AugAssign, ast.AnnAssign, ast.For)):
            pending.append(child.target)
    while pending:
        target = pending.pop()
        if isinstance(target, (ast.Tuple, ast.List)):
            pending.extend(target.elts)
            continue
        name = _attr_target(target)
        if name is not None:
            names.add(name)
    return names


def _self_reads(node):
    """Attributes the function reads off `self`.

    `self.state[1::2] = ...` counts as a read of `state`, which is right: the
    slice on the left still has to fetch the array first.
    """
    return {
        child.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute)
        and isinstance(child.value, ast.Name) and child.value.id == "self"
        and isinstance(child.ctx, ast.Load)
    }


def _reads_name(node, name):
    """Does the function body actually use the local `name`?"""
    return any(isinstance(child, ast.Name) and child.id == name
               and isinstance(child.ctx, ast.Load)
               for child in ast.walk(node))


def _parameters(node):
    """Positional parameter names, `self` first."""
    return [arg.arg for arg in node.args.posonlyargs + node.args.args]


def _state_of(value):
    """`value` as a float array if it could be a six-number state, else None."""
    try:
        array = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    return array if array.size == STATE_SIZE else None


def _fmt(array):
    return "[" + ", ".join(f"{v:.4g}" for v in array) + "]"


# ---------------------------------------------------------------------------
# Running one step function on its own
# ---------------------------------------------------------------------------

def _named_shape(name):
    """Invent an attribute from what its name suggests it is."""
    key = name.lower().lstrip("_")
    if key in VECTOR6_NAMES:
        return np.zeros(STATE_SIZE)
    if key in VECTOR3_NAMES:
        return np.zeros(3)
    if key in MATRIX_NAMES:
        return np.eye(STATE_SIZE)
    return FILLER_ARG


def _scalar_shape(_name):
    return FILLER_ARG


def _vector_shape(_name):
    return np.zeros(STATE_SIZE)


def _matrix_shape(_name):
    return np.eye(STATE_SIZE)


# Tried in order until one lets the function run. The first handles a filter
# that mixes scalars and matrices (`self.dt` and `self.F`); the rest are for
# names we simply don't recognise.
POLICIES = (_named_shape, _scalar_shape, _vector_shape, _matrix_shape)


class _Stand:
    """A stand-in filter, so one step function can be run without the others.

    Grading `predict` replays the *reference* `__init__` above it, so the real
    object knows nothing about the student's own attributes. This one invents
    whatever is asked for and remembers it, which is enough to walk their code
    through a single step.
    """

    def __init__(self, invent, state=None):
        object.__setattr__(self, "_invent", invent)
        self.state = SEED_STATE.copy() if state is None else np.array(state, dtype=float)

    def __getattr__(self, name):
        if name.startswith("__"):  # never fake a dunder; that confuses numpy
            raise AttributeError(name)
        value = object.__getattribute__(self, "_invent")(name)
        object.__setattr__(self, name, value)
        return value


def _snapshot(stand):
    """The numeric attributes a stand-in is left holding, for comparing runs."""
    out = {}
    for name, value in vars(stand).items():
        if name.startswith("_"):
            continue
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            continue
        if array.size:
            out[name] = array
    return out


def _differs(before, after):
    if set(before) != set(after):
        return True
    for name, value in after.items():
        other = before[name]
        if other.shape != value.shape:
            return True
        if not np.allclose(other, value, rtol=SAME_RTOL, atol=SAME_ATOL, equal_nan=True):
            return True
    return False


def _probe(func, measurement, state=None):
    """Run one step function against a stand-in filter.

    Returns `(stand, blame)`. `blame` is None when it ran; a string when the
    student's code raised something a wrong guess of ours can't explain; and
    False when no policy got it running, which is never graded — see the module
    docstring.
    """
    for policy in POLICIES:
        stand = _Stand(policy, state)
        try:
            func(stand, np.array(measurement, dtype=float))
        except FORGIVEN:
            continue  # probably our invented attribute, not their code
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
        return stand, None
    return None, False


def _construct(cls):
    """Build the student's filter, filling in any argument they made required."""
    import inspect

    attempts = [()]
    try:
        parameters = list(inspect.signature(cls.__init__).parameters.values())[1:]
        required = [p for p in parameters
                    if p.default is p.empty
                    and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        if required:
            attempts.append(tuple(FILLER_ARG for _ in required))
    except (TypeError, ValueError):
        pass
    for args in attempts:
        try:
            return cls(*args), None
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
    return None, failure


def _step(ctx, name):
    """The function this cell was supposed to write, from the namespace.

    Prefers the module-level `def` the student wrote here; falls back to what
    the binding line hung on the class, in case they only wrote it as a method.
    """
    func = ctx.get(name)
    if func is None:
        cls = ctx.env.get("filter_scratch")
        func = getattr(cls, name, None) if isinstance(cls, type) else None
    return func if callable(func) else None


# ---------------------------------------------------------------------------
# Initialize Step
# ---------------------------------------------------------------------------

def check_initialize_step(ctx):
    """"Set your initial state, coefficients, covariance, etc." """
    if not _ran(ctx):
        return

    cls = ctx.get("filter_scratch")
    if not ctx.require(
            "Defines the `filter_scratch` class", isinstance(cls, type),
            "Keep the `class filter_scratch:` line — the SILSIM cell at the bottom "
            "builds the filter by that name."):
        return

    node = _class(ctx.tree, "filter_scratch")
    init = _function(node, "__init__") if node is not None else None
    if not ctx.require(
            "Defines `__init__`", init is not None,
            "The filter needs an `__init__(self, ...)` to set itself up in."):
        return

    stored = _self_writes(init)
    ctx.require(
        "Your own parameters are stored on the filter",
        bool(stored - {"state"}),
        "The cell ships with `self.state` and nothing else. Whatever your filter "
        "is tuned by — a gain, a time step, a covariance — has to be set here as "
        "`self.name = value`, so `predict` and `update` can reach it later.",
        expected="e.g. self.alpha = 0.1, self.dt = 0.095",
        got=", ".join(f"self.{name}" for name in sorted(stored)) or "nothing")

    filter_object, failure = _construct(cls)
    if not ctx.require(
            "The filter can be created", filter_object is not None,
            "The SILSIM cell calls `filter_scratch(...)`, and building one raised "
            "an error. Give every argument in the header a default (`alpha=0.1`) "
            "so it can be built without them.",
            got=failure):
        return

    state = _state_of(getattr(filter_object, "state", None))
    ctx.require(
        "`self.state` starts as six numbers",
        state is not None and np.all(np.isfinite(state)),
        "The SILSIM cell plots `state[0]` through `state[5]` as "
        "[position_x, velocity_x, position_y, velocity_y, position_z, velocity_z], "
        "so `self.state` has to be six finite numbers before the first predict.",
        expected=f"{STATE_SIZE} numbers",
        got=_fmt(state) if state is not None else repr(getattr(filter_object, "state", None)))


# ---------------------------------------------------------------------------
# Predict Step
# ---------------------------------------------------------------------------

def check_predict_step(ctx):
    """"Compute a prediction of the state based on the previous state." """
    if not _ran(ctx):
        return

    node = _function(ctx.tree, "predict")
    if not ctx.require(
            "Defines `predict`", node is not None,
            "Keep the `def predict(self, measurement):` header — the SILSIM cell "
            "calls `filter.predict(...)` once per sample."):
        return

    parameters = _parameters(node)
    ctx.require(
        "Takes `(self, measurement)`", len(parameters) >= 2,
        "SILSIM calls `filter.predict(input_data[i])`, so predict needs `self` and "
        "one more parameter for the measurement.",
        expected="def predict(self, measurement)",
        got=f"def predict({', '.join(parameters)})")

    ctx.require(
        "Predicts from the filter's own attributes", bool(_self_reads(node)),
        "The cell ships with `self.state = np.zeros(6)`, which throws the state "
        "away every step. A prediction has to *read* something off the filter — "
        "the previous `self.state`, your time step, your acceleration estimate — "
        "and carry it forward.")

    ctx.require(
        "Writes the predicted state back", "state" in _self_writes(node),
        "Nothing downstream can see the prediction unless it lands in "
        "`self.state`: that's what SILSIM records and plots.")

    cls = ctx.env.get("filter_scratch")
    ctx.require(
        "Bound to the filter class",
        isinstance(cls, type) and getattr(cls, "predict", None) is ctx.env.get("predict"),
        "Keep the `filter_scratch.predict = predict` line at the bottom of the "
        "cell — without it the filter has no predict step.")

    func = _step(ctx, "predict")
    if func is None or len(parameters) < 2:
        return

    stand, blame = _probe(func, SAMPLE_A)
    if blame:
        ctx.require("Runs on one sample", False,
                    "Predicting a single measurement raised an error.", got=blame)
        return
    if stand is None:
        return  # our stand-in couldn't match their attributes; not their problem

    state = _state_of(getattr(stand, "state", None))
    if not ctx.require(
            "Leaves six finite numbers in `self.state`",
            state is not None and np.all(np.isfinite(state)),
            "After one predict, `self.state` has to still be the six numbers "
            "SILSIM plots — no NaNs, no infinities, no change of length.",
            expected=f"{STATE_SIZE} finite numbers",
            got=_fmt(state) if state is not None else repr(getattr(stand, "state", None))):
        return

    ctx.require(
        "The prediction moves the state on",
        bool(np.any(np.abs(state) > ZERO_EPS)),
        "Started from a state with velocity on all three axes and got back all "
        "zeros — that's the `self.state = np.zeros(6)` the cell ships with, not a "
        "prediction. Propagate the previous state instead of replacing it.",
        expected="the seeded state carried forward",
        got=_fmt(state))


# ---------------------------------------------------------------------------
# Update Step
# ---------------------------------------------------------------------------

def check_update_step(ctx):
    """"Use the measurement to update your prediction." """
    if not _ran(ctx):
        return

    node = _function(ctx.tree, "update")
    if not ctx.require(
            "Defines `update`", node is not None,
            "Keep the `def update(self, measurement):` header — the SILSIM cell "
            "calls `filter.update(...)` after every predict."):
        return

    parameters = _parameters(node)
    ctx.require(
        "Takes `(self, measurement)`", len(parameters) >= 2,
        "SILSIM calls `filter.update(input_data[i])`, so update needs `self` and "
        "one more parameter for the measurement.",
        expected="def update(self, measurement)",
        got=f"def update({', '.join(parameters)})")

    if len(parameters) >= 2:
        ctx.require(
            "Uses the measurement", _reads_name(node, parameters[1]),
            f"`{parameters[1]}` is never read. The update step is the *only* place "
            "the filter hears from the sensor: it's [t, ax, ay, az], so the "
            "accelerations are elements 1 to 3.")

    ctx.require(
        "Writes the corrected state back", "state" in _self_writes(node),
        "The correction has to land in `self.state` — that's what SILSIM records "
        "and what the next predict propagates.")

    cls = ctx.env.get("filter_scratch")
    ctx.require(
        "Bound to the filter class",
        isinstance(cls, type) and getattr(cls, "update", None) is ctx.env.get("update"),
        "Keep the `filter_scratch.update = update` line at the bottom of the cell "
        "— without it the filter has no update step.")

    func = _step(ctx, "update")
    if func is None or len(parameters) < 2:
        return

    first, blame = _probe(func, SAMPLE_A)
    if blame:
        ctx.require("Runs on one sample", False,
                    "Updating with a single measurement raised an error.", got=blame)
        return
    if first is None:
        return  # our stand-in couldn't match their attributes; not their problem

    state = _state_of(getattr(first, "state", None))
    if not ctx.require(
            "Leaves six finite numbers in `self.state`",
            state is not None and np.all(np.isfinite(state)),
            "After one update, `self.state` has to still be the six numbers SILSIM "
            "plots — no NaNs, no infinities, no change of length.",
            expected=f"{STATE_SIZE} finite numbers",
            got=_fmt(state) if state is not None else repr(getattr(first, "state", None))):
        return

    ctx.require(
        "The update keeps the state alive",
        bool(np.any(np.abs(state) > ZERO_EPS)),
        "Started from a state with velocity on all three axes and got back all "
        "zeros — that's the `self.state = np.zeros(6)` the cell ships with. "
        "Correct the predicted state instead of replacing it.",
        expected="the predicted state, corrected",
        got=_fmt(state))

    # Same starting point, a different sample: a filter that listens has to end
    # up somewhere else. This catches an `update` that names `measurement` in a
    # comment or a throwaway line but never really uses it.
    second, blame = _probe(func, SAMPLE_B)
    if blame or second is None:
        return
    ctx.require(
        "A different measurement gives a different result",
        _differs(_snapshot(first), _snapshot(second)),
        "Two very different acceleration readings left the filter in exactly the "
        "same place, so the measurement isn't reaching the state. Element 0 of the "
        "measurement is the timestamp — the accelerations you want are 1, 2 and 3.",
        expected="two readings, two results",
        got=_fmt(state))
