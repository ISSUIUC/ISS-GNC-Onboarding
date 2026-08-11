"""Behaviour checkers for kalman_filter.ipynb.

Three exercises. The first two hand the student a working filter and ask them
to turn the knobs — the measurement uncertainty R and the model uncertainty Q —
and watch what the estimate does; the third asks them to write the
position/velocity filter themselves and then tune that. Either way "try a
larger measurement uncertainty" has as many right answers as there are numbers,
so there is nothing to diff against. What these checkers do instead is re-run
the filter themselves, from the student's own numbers, and insist that what is
sitting in the student's variables is what those numbers actually produce. That
is the difference between a student who tuned the filter and a student who
edited the plot. Where re-running can't settle it — exercise 2 leaves enough
free that a state could legitimately start anywhere — the lists are made to
agree with each other instead: the residual kept at each step has to be the
measurement minus the position the step before predicts, which is true of every
real run of that filter and of no column copied off the truth.

The thresholds are the part worth explaining. This notebook deliberately
invites extreme tunings ("make both really small", "make the velocity
uncertainty larger"), and an extreme tuning is a legitimate thing to *try* —
so the accuracy checks are gated rather than absolute. In exercise 1 the
estimate is only asked to beat the raw measurements when the steady gain those
two numbers settle to is somewhere near a sensible balance; with the values the
cell ships with, the 1-D filter is in fact *worse* than the raw data, which is
the lesson of the cell, and it would be perverse to fail a student for finding
that out. Exercise 2 is gated against itself: each accuracy row is only asked
for when this same filter, re-run here with the student's own Q and R, clears
it too — "making position uncertainty larger" genuinely does wreck the velocity
estimate once Q[0][0] is a few hundred, and that is the experiment working, not
a mistake. The final challenge leans the other way: its rocket is accelerating
while the filter still models constant velocity, so the velocity estimate has
to lag, and the band it is judged against was measured off a good reference
solution rather than guessed at.

Shared helpers do most of the work — pulling a series out of a namespace that
may hold a list, a ragged list, or a numpy array of the wrong shape; reading a
matrix or a call's arguments back off the AST; deciding whether a plot was
drawn, including one drawn inside a helper the student wrote. All of them
return None rather than raising, because a checker that raises replaces every
row of feedback with one "the checker crashed" apology. They are also, on
purpose, generous about surface: the two numbers of exercise 1 are found
whether they were named, passed positionally, left at a default or typed
straight into the call, and the series they produce are found whether the
student kept the cell's names or their own.

Exercise 3 is pinned with `#% checker: tune_the_filter` in the cell itself, so
inserting an exercise above it can never re-point it.
"""

import ast
import re

import numpy as np


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Exercise 1 — the two numbers the cell ships with. "At least one of these
# changed" is the whole evidence that the student did the exercise, so if you
# edit the cell's defaults, edit these too.
SHIPPED_MEASUREMENT_UNCERTAINTY = 35.0 ** 2
SHIPPED_MODEL_UNCERTAINTY = 5.0
# The other two arguments in the cell's call. Only used when the student's own
# call can't be read off the AST (e.g. they inlined an expression we can't
# evaluate); normally we take whatever they actually passed.
INITIAL_UNCERTAINTY = 100.0
# The cell's filter takes its five arguments in this order. Used to name the
# *positional* arguments of a call to a function we can't otherwise find.
SIMPLE_FILTER_PARAMS = ("measurements", "initial_estimate", "initial_uncertainty",
                        "measurement_uncertainty", "model_uncertainty")

# Exercise 2 — the process noise matrix the cell ships with, same idea, plus
# the two numbers its call passes, for re-running the filter ourselves.
SHIPPED_Q = ((1.0, 0.0), (0.0, 3.0))
SHIPPED_MEASUREMENT_STD = 35.0
SHIPPED_DT = 0.1
MATRIX_FILTER_PARAMS = ("measurements", "dt", "measurement_std")

# How close a recomputed series has to be to the student's to count as "this
# came out of the filter". Loose enough for float noise, far too tight for a
# number that was typed in by hand.
MATCH_RTOL = 1e-6
MATCH_ATOL = 1e-8

# Exercise 1 only asks the estimate to beat the raw measurements when the gain
# the student's R and Q settle to lands in this band. Outside it the tuning is
# an experiment the prompt invited, not a mistake, and the check is skipped
# entirely. Widen the band if you want the accuracy row to appear more often.
TRACKING_GAIN_BAND = (0.10, 0.80)

# Exercise 2: the constant-velocity model is *correct* for that data, so any
# sane Q beats the raw sensor and lands the velocity on the true climb rate.
# Measured across every tuning the prompt suggests, the worst case is a ratio of
# 0.66 and a velocity error of 1.3 m/s, so these are generous.
EX2_TRACKING_RATIO = 1.0          # estimate RMS / raw RMS must be below this
EX2_VELOCITY_TOLERANCE = 10.0     # m/s, on the mean over the last quarter
SETTLE_FRACTION = 0.05            # last step's change in P, as a fraction of P
# Both of those rows are *gated*: they are only asked for when the same filter,
# re-run here with the student's own Q, clears them too. "Making position
# uncertainty larger" is something the cell invites, and past about Q[0][0]=1000
# it genuinely ruins the velocity estimate — that is the experiment working, not
# the student failing it.

# How closely the residuals have to agree with the states they were collected
# beside, as a fraction of the largest measurement. A real filter agrees to
# floating-point noise; anything typed in by hand is out by metres.
RESIDUAL_TOLERANCE = 1e-6
# How far the settled uncertainty may sit from the one the student's own Q and
# R imply, as a ratio either way. A filter that really ran lands on it exactly;
# the bar is loose because P also has to survive whatever they did to the
# initial guess, and only wide misses mean anything.
SETTLED_P_TOLERANCE = 1.25

# Exercise 3: the rocket accelerates at 2.5 m/s^2 and the filter models constant
# velocity, so the velocity estimate *must* lag — these numbers come from
# running good tunings over the challenge data, not from theory. A well-tuned
# filter gets the position RMS down to ~0.35 of the raw sensor and ends within
# ~4 m/s of the true velocity; a filter with far too small a Q sits at 1.0-1.4
# and 20-30 m/s out. The bar is set between the two, closer to the bad end.
EX3_TRACKING_RATIO = 0.80         # estimate RMS / raw RMS must be below this
EX3_VELOCITY_TOLERANCE = 15.0     # m/s, on the mean over the last quarter
EX3_VELOCITY_FOUND = 45.0         # further out than this and it isn't a
                                  # velocity estimate at all, it's some other array

# The fraction of the run treated as "once the filter has settled".
LATE_FRACTION = 0.25

# Accepted variable names. `ctx.defined` only matches names the *submission*
# assigns, so short ones are safe: a leftover from a replayed setup cell can
# never satisfy them.
MEASUREMENT_UNCERTAINTY_VARS = (
    "MEASUREMENT_UNCERTAINTY", "measurement_uncertainty", "MEASUREMENT_VARIANCE",
    "measurement_variance", "MEASUREMENT_NOISE", "measurement_noise_var",
    "sensor_uncertainty", "R", "r")
MODEL_UNCERTAINTY_VARS = (
    "MODEL_UNCERTAINTY", "model_uncertainty", "MODEL_VARIANCE", "model_variance",
    "PROCESS_NOISE", "process_noise", "model_noise", "Q", "q")
INITIAL_ESTIMATE_VARS = (
    "initial_estimate", "initial_state", "initial", "x0", "start")
INITIAL_UNCERTAINTY_VARS = (
    "initial_uncertainty", "initial_P", "initial_p", "P0", "p0", "P_initial")
MEASUREMENT_STD_VARS = (
    "measurement_std", "sensor_std", "measurement_sigma", "sigma", "std")

ESTIMATE_VARS = (
    "kalman_position", "kalman_estimate", "kalman_estimates", "kalman_altitude",
    "estimated_position", "estimates", "estimate", "filtered_position",
    "position_estimate", "kalman_pos", "positions")
GAIN_VARS = (
    "kalman_gain", "kalman_gains", "gains", "gain", "K", "k", "kalman_K")
STATE_VARS = (
    "states", "state", "state_history", "xs", "x_history", "kalman_states")
RESIDUAL_VARS = ("residuals", "residual", "innovations", "innovation", "ys")
UNCERTAINTY_VARS = (
    "uncertainty_P", "uncertainty", "uncertainties", "covariances", "Ps",
    "P_history", "covariance_history")
POSITION_ESTIMATE_VARS = (
    "estimated_position", "estimated_altitude", "kalman_position",
    "filtered_position", "position_estimate", "pos_est", "est_position")
VELOCITY_ESTIMATE_VARS = (
    "estimated_velocity", "kalman_velocity", "filtered_velocity",
    "velocity_estimate", "vel_est", "est_velocity")

# The data the notebook hands the student. Never treat one of these as their
# own estimate, however well it happens to fit the truth.
GIVEN_DATA_VARS = (
    "time", "dt", "true_position", "true_velocity", "true_acceleration",
    "measured_position", "measurement_noise", "measurements")

PLOT_ATTRS = {
    "plot", "scatter", "step", "errorbar", "fill_between", "stairs",
    "semilogx", "semilogy", "loglog", "axhline", "axvline", "hlines", "bar",
}
INVERSE_ATTRS = {"inv", "pinv", "solve", "lstsq"}
MATMUL_ATTRS = {"dot", "matmul", "vdot", "inner"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ran(ctx, hint="Fix the error shown in the output above, then check again."):
    """Every exercise starts here: nothing else means anything if it crashed."""
    return ctx.require("Runs without an error", ctx.ok, hint)


def _number(value):
    """`value` as a finite float, or None if it isn't a plain number."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(-1)[0]  # a 1-element array is still one number
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _array(value):
    """`value` as a float ndarray, or None if it can't be one.

    Lists, lists of arrays and numpy arrays all arrive here; strings, dicts,
    functions and ragged lists all leave as None.
    """
    if value is None or isinstance(value, (str, bytes, dict, set)):
        return None
    try:
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if out.ndim == 0 or out.size == 0:
        return None
    return out


def _series(value, length=None):
    """`value` as a 1-D float series (of `length`, if given), or None."""
    out = _array(value)
    if out is None:
        return None
    if out.ndim == 2 and 1 in out.shape:
        out = out.reshape(-1)  # a column vector is still one value per step
    if out.ndim != 1:
        return None
    if length is not None and out.size != length:
        return None
    return out


def _matrix(value):
    """`value` as a 2-D float array, or None."""
    out = _array(value)
    if out is None or out.ndim != 2:
        return None
    return out


def _rms(a, b):
    """Root-mean-square difference between two series, or None."""
    if a is None or b is None or a.shape != b.shape:
        return None
    try:
        out = float(np.sqrt(np.mean((a - b) ** 2)))
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _same(a, b):
    """Are these two series numerically the same run?"""
    if a is None or b is None or a.shape != b.shape:
        return False
    try:
        return bool(np.allclose(a, b, rtol=MATCH_RTOL, atol=MATCH_ATOL))
    except Exception:
        return False


def _from_env(ctx, name, length=None):
    """A series the *notebook* defined (setup cells count here, unlike ctx.get)."""
    return _series(ctx.env.get(name), length)


def _late(values):
    """The tail of a run — where a Kalman filter is supposed to have settled."""
    if values is None or values.size == 0:
        return values
    start = int(values.size * (1.0 - LATE_FRACTION))
    return values[min(start, values.size - 1):]


def _shape(value):
    """`np.shape`, but a value that isn't there says so rather than showing `()`."""
    if value is None:
        return "missing"
    try:
        return str(np.shape(value))
    except Exception:
        return "?"


def _fmt(value, digits=2):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return repr(value)


# --- AST helpers -----------------------------------------------------------

def _func_name(node):
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _calls_named(node, names):
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call) and _func_name(inner) in names:
            return True
    return False


def _has_matmul(node):
    for inner in ast.walk(node):
        if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.MatMult):
            return True
    return _calls_named(node, MATMUL_ATTRS)


def _has_transpose(node):
    for inner in ast.walk(node):
        if isinstance(inner, ast.Attribute) and inner.attr == "T":
            return True
    return _calls_named(node, {"transpose"})


def _has_op(node, op):
    return any(isinstance(inner, ast.BinOp) and isinstance(inner.op, op)
               for inner in ast.walk(node))


def _assign_count(node):
    return sum(1 for inner in ast.walk(node) if isinstance(inner, ast.Assign))


def _filter_loops(tree):
    """Loops that could plausibly be a predict/update loop, one group each.

    Anything that assigns something in the body qualifies — the point is to
    skip `for i in range(3): print(i)`, not to insist on a shape. A loop whose
    body calls a function defined in the same cell brings that function into
    its group: `for z in zs: x, P = step(x, P, z)` is still a filter loop, and
    the matrix arithmetic inside `step` is still the student's own.
    """
    functions = {node.name: node for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    groups = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.While)):
            continue
        group = [node]
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                called = functions.get(inner.func.id)
                if called is not None and called not in group:
                    group.append(called)
        if _assign_count(node) >= 1 or len(group) > 1:
            groups.append(group)
    return groups


def _in_loops(loops, *tests):
    """Do one loop and the helpers it calls, between them, do all of `tests`?"""
    return any(all(any(test(node) for node in group) for test in tests)
               for group in loops)


def _assigned_node(tree, names, first=False):
    """The last (or first) expression assigned to any of `names`, as a node.

    An assignment *inside* a function wins over one at the top of the cell:
    exercise 2's Q lives inside the filter, and a Q set underneath it — after
    the filter has already run — is a line that changes nothing. `first=True`
    asks for the other end, which is what the initial uncertainty needs: `P` is
    assigned once before the loop and then again on every step of it.
    """
    def scan(root):
        best, best_line = None, None
        for node in ast.walk(root):
            if not isinstance(node, ast.Assign):
                continue
            line = getattr(node, "lineno", 0)
            for target in node.targets:
                if not isinstance(target, ast.Name) or target.id not in names:
                    continue
                if best_line is None or (line < best_line if first else line >= best_line):
                    best, best_line = node.value, line
        return best

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found = scan(node)
            if found is not None:
                return found
    return scan(tree)


def _literals(tree):
    """Every `name = <literal>` in the cell, including the ones inside functions.

    `Q = scale * np.array(...)` is a perfectly good way to write the process
    noise, and `scale` is a local that never reaches `ctx.env` — so collect the
    simple constants ourselves before trying to evaluate anything.
    """
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                try:
                    found[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError, MemoryError):
                    pass
    return found


def _eval_node(ctx, node):
    """Evaluate an expression from the submission, in the submission's namespace.

    Used to read back values that only exist inside a function (the `Q` of
    exercise 2 is a local), where there is nothing in `ctx.env` to look at.
    Anything that doesn't evaluate — a name that was a parameter, a call we
    can't reproduce — comes back as None, and the caller then skips that check
    rather than failing it.
    """
    if node is None:
        return None
    namespace = dict(ctx.env)
    namespace.update(_literals(ctx.tree))
    try:
        return eval(compile(ast.Expression(node), "<checker>", "eval"), namespace)
    except Exception:
        return None


def _unparse(node):
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _signature(ctx, node):
    """(parameter names, {name: default}) for the function a call refers to.

    Wanted so that a *positional* call can be read like a keyword one — a
    student who writes `simple_kalman_filter(measured_position, z0, 100, 400,
    20)` has changed the two numbers just as surely as one who names them — and
    so that an argument they left at its default is still a number we know.
    """
    func = node.func
    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
    if not name:
        return [], {}
    for inner in ast.walk(ctx.tree):
        if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)) and inner.name == name:
            names = [arg.arg for arg in inner.args.args]
            values = [_eval_node(ctx, default) for default in inner.args.defaults]
            defaults = dict(zip(names[len(names) - len(values):], values))
            for keyword, default in zip(inner.args.kwonlyargs, inner.args.kw_defaults):
                defaults[keyword.arg] = _eval_node(ctx, default)
            return names, defaults
    function = ctx.env.get(name)
    code = getattr(function, "__code__", None)
    if code is None:
        return [], {}
    names = list(code.co_varnames[:code.co_argcount])
    values = list(getattr(function, "__defaults__", None) or ())
    return names, dict(zip(names[len(names) - len(values):], values))


def _from_call(arguments, names):
    """The first of `names` this call has a value for."""
    for name in names:
        if arguments.get(name) is not None:
            return arguments[name]
    return None


def _find_call(ctx, needle, targets=()):
    """The call that ran the filter, as best the cell lets us tell.

    First choice is the call whose result was unpacked into the names we are
    about to read back — `kalman_position, kalman_gain = anything(...)` is the
    filter call whatever the student called their function. Failing that, the
    first call whose own name mentions the filter.
    """
    wanted = set(targets)
    for node in ast.walk(ctx.tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        assigned = {inner.id for target in node.targets
                    for inner in ast.walk(target) if isinstance(inner, ast.Name)}
        if assigned & wanted:
            return node.value
    calls = [node for node in ast.walk(ctx.tree)
             if isinstance(node, ast.Call) and needle in _func_name(node).lower()]
    calls.sort(key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)))
    return calls[0] if calls else None


def _call_args(ctx, node, fallback=()):
    """The arguments of a call, by name.

    Keyword arguments come back under their own names. Positional ones are
    matched up with the callee's parameter names, and — when there are exactly
    as many of them as the cell's own filter takes — with `fallback` as well,
    so that a student who typed the numbers straight into the call is read the
    same way as one who named them. A value we can't evaluate comes back as
    None, and the caller falls back to the cell's own default.
    """
    if node is None:
        return {}
    parameters, defaults = _signature(ctx, node)
    values = []
    for argument in node.args:
        if isinstance(argument, ast.Starred):  # f(*args) still passed five numbers
            unpacked = _eval_node(ctx, argument.value)
            values.extend(unpacked if isinstance(unpacked, (list, tuple)) else [None])
        else:
            values.append(_eval_node(ctx, argument))
    orders = [parameters]
    if len(values) == len(fallback):
        orders.insert(0, list(fallback))  # the callee's own names win ties
    out = {}
    for names in orders:
        for index, value in enumerate(values):
            if index < len(names):
                out[names[index]] = value
    for keyword in node.keywords:
        if keyword.arg:
            out[keyword.arg] = _eval_node(ctx, keyword.value)
    for name, default in defaults.items():
        out.setdefault(name, default)  # an argument left alone is still a number
    return out


def _call_sites(tree):
    """For each function defined here, what its callers pass for each parameter.

    Only used to make sense of plotting done inside a helper — see
    `_plot_calls`.
    """
    functions = {node.name: node for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    table = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        function = functions.get(node.func.id)
        if function is None:
            continue
        parameters = [arg.arg for arg in function.args.args]
        seen = table.setdefault(function.name, {})
        for index, argument in enumerate(node.args):
            if index < len(parameters):
                seen.setdefault(parameters[index], []).append(_unparse(argument))
        for keyword in node.keywords:
            if keyword.arg:
                seen.setdefault(keyword.arg, []).append(_unparse(keyword.value))
    return table


def _plot_nodes(node, holder=None):
    """Every plotting call, paired with the function it sits inside (or None)."""
    found = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.extend(_plot_nodes(child, child.name))
            continue
        if isinstance(child, ast.Call) and _func_name(child) in PLOT_ATTRS:
            found.append((holder, child))
        found.extend(_plot_nodes(child, holder))
    return found


def _plot_calls(tree):
    """Every plotting call in the cell, as source text.

    A call inside a helper carries its caller's arguments along with it:
    `def show(t, y): plt.plot(t, y)`, called as `show(time,
    estimated_position)`, really does plot the estimate, and the text of the
    plotting line on its own doesn't say so.
    """
    aliases = _call_sites(tree)
    texts = []
    for holder, node in _plot_nodes(tree):
        text = _unparse(node)
        passed = aliases.get(holder, {})
        extra = [option for parameter, options in passed.items()
                 if re.search(r"\b%s\b" % re.escape(parameter), text)
                 for option in options]
        texts.append(" ".join([text] + extra))
    return texts


def _mentions(texts, names):
    """Does any of these plotting calls name one of `names`?"""
    for name in names:
        if not name:
            continue
        pattern = re.compile(r"\b%s\b" % re.escape(name))
        if any(pattern.search(text) for text in texts):
            return True
    return False


def _plotted_series(tree):
    """How many curves the cell draws. `plot(t, a, t, b)` is two, not one."""
    return sum(max(1, len(node.args) // 2) for _holder, node in _plot_nodes(tree))


def _drew_a_figure(ctx, needed):
    """Is the cell's plot still there?

    Counting the curves is the whole test — insisting on a `plt.figure` as well
    would fail a student who tidied the plotting down to two bare `plt.plot`
    lines, which is not what any of these exercises is about.
    """
    return _plotted_series(ctx.tree) >= needed


def _aliases(ctx, series, length):
    """The submission's own names for a series it was handed rather than made.

    `z = measured_position` then `plt.plot(time, z)` is still plotting the
    measurements, and the plot checks should say so.
    """
    if series is None:
        return []
    return [name for name in sorted(ctx.assigned)
            if name in ctx.env and _same(_series(ctx.env[name], length), series)]


def _candidate_series(ctx, length, given):
    """Every series the submission itself produced, one entry per column.

    Anything numerically identical to the data the challenge handed over is
    dropped: the truth is sitting right there in the cell, and an "estimate"
    that is a copy of it hasn't estimated anything.
    """
    found = []
    for name in sorted(ctx.assigned):
        if name in GIVEN_DATA_VARS or name not in ctx.env:
            continue
        array = _array(ctx.env[name])
        if array is None:
            continue
        if array.ndim == 1 and array.size == length:
            columns = [array]
        elif array.ndim == 2 and array.shape[0] == length and 1 <= array.shape[1] <= 6:
            columns = [array[:, i] for i in range(array.shape[1])]
        elif array.ndim == 2 and array.shape[1] == length and 1 <= array.shape[0] <= 6:
            columns = [array[i, :] for i in range(array.shape[0])]
        else:
            continue
        for column in columns:
            if not np.all(np.isfinite(column)):
                continue
            if any(_same(column, series) for series in given):
                continue
            found.append((name, column))
    return found


def _closest(candidates, target):
    """The candidate series nearest a target, with the names that alias it."""
    best = None
    for name, column in candidates:
        error = _rms(column, target)
        if error is not None and (best is None or error < best[0]):
            best = (error, name, column)
    if best is None:
        return None, None, set()
    _, name, column = best
    names = {other for other, series in candidates if _same(series, column)}
    names.add(name)
    return name, column, names


# --- the notebook's own filters, re-implemented ----------------------------

def _simple_kalman(measurements, initial_estimate, initial_uncertainty,
                   measurement_uncertainty, model_uncertainty):
    """The 1-D filter from the worked example, so we can re-derive the answer.

    A checker file is exec'd on its own and can't import the notebook, and
    re-running the student's copy of the function would only prove it agrees
    with itself. These are the same eight lines as the cell above the exercise.
    """
    estimate = float(initial_estimate)
    P = float(initial_uncertainty)
    estimates, gains = [], []
    for z in measurements:
        predicted_P = P + model_uncertainty
        gain = predicted_P / (predicted_P + measurement_uncertainty)
        estimate = estimate + gain * (float(z) - estimate)
        P = (1.0 - gain) * predicted_P
        estimates.append(estimate)
        gains.append(gain)
    return np.array(estimates), np.array(gains)


def _matrix_kalman(measurements, dt, measurement_std, Q, initial_uncertainty=100.0):
    """The position/velocity filter from the worked example, re-run here.

    exercise 2 hands the student that filter and asks them to change one matrix
    inside it, so the honest question about the result is not "is it good" but
    "is it as good as this Q allows" — which needs the same filter, driven by
    the same numbers, to compare against. Returns (states, flattened P), or
    None if those numbers can't drive a filter at all.
    """
    try:
        Q = np.asarray(Q, dtype=float).reshape(2, 2)
        F = np.array([[1.0, float(dt)], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[float(measurement_std) ** 2]])
        identity = np.eye(2)
        x = np.array([[float(measurements[0])], [0.0]])
        P = np.asarray(initial_uncertainty, dtype=float)
        P = np.eye(2) * float(P) if P.ndim == 0 else P.reshape(2, 2)
        states, uncertainties = [], []
        for z in measurements:
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
            y = np.array([[float(z)]]) - H @ x_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            x = x_pred + K @ y
            P = (identity - K @ H) @ P_pred
            states.append(x.flatten())
            uncertainties.append(P.flatten())
        history = np.array(states)
        spread = np.array(uncertainties)
    except Exception:
        return None
    if not (np.all(np.isfinite(history)) and np.all(np.isfinite(spread))):
        return None
    return history, spread


def _true_climb_rate(ctx, fallback=25.0):
    """The rocket's actual velocity, read off the truth the notebook generated."""
    time = _from_env(ctx, "time")
    truth = _from_env(ctx, "true_position")
    if time is None or truth is None or time.shape != truth.shape or time.size < 2:
        return fallback
    try:
        slope = float(np.polyfit(time, truth, 1)[0])
    except Exception:
        return fallback
    return slope if np.isfinite(slope) else fallback


# ---------------------------------------------------------------------------
# Exercise 1 — tune R and Q on the 1-D filter
# ---------------------------------------------------------------------------

def check_ex1(ctx):
    """"Change `measurement_uncertainty` and `model_uncertainty`."

    Two numbers, then re-run the filter and look at the picture. So: are they
    numbers, did at least one of them move, and is the curve on the plot really
    what those two numbers produce? The accuracy of the result is the last
    thing we look at, and only when the tuning is near enough to sensible that
    "better than the raw data" is a fair thing to ask for.
    """
    if not _ran(ctx):
        return

    # Named variables first, then whatever the call itself passes — a student
    # who typed the two numbers straight into the call has still changed them.
    arguments = _call_args(ctx, _find_call(ctx, "kalman", ESTIMATE_VARS + GAIN_VARS),
                           SIMPLE_FILTER_PARAMS)
    measurement = _number(ctx.get(
        *MEASUREMENT_UNCERTAINTY_VARS,
        default=_from_call(arguments, MEASUREMENT_UNCERTAINTY_VARS)))
    model = _number(ctx.get(
        *MODEL_UNCERTAINTY_VARS, default=_from_call(arguments, MODEL_UNCERTAINTY_VARS)))

    if not ctx.require(
            "The measurement uncertainty is a number", measurement is not None,
            "Set MEASUREMENT_UNCERTAINTY to a single number before the filter runs "
            "— it is R, the variance of the sensor noise, so it is a squared "
            "distance like 35**2, not an array.",
            got=repr(ctx.get(
                *MEASUREMENT_UNCERTAINTY_VARS,
                default=_from_call(arguments, MEASUREMENT_UNCERTAINTY_VARS)))):
        return
    if not ctx.require(
            "The model uncertainty is a number", model is not None,
            "Set MODEL_UNCERTAINTY to a single number before the filter runs — it "
            "is Q, how much the filter is allowed to distrust its own physics "
            "each step.",
            got=repr(ctx.get(
                *MODEL_UNCERTAINTY_VARS,
                default=_from_call(arguments, MODEL_UNCERTAINTY_VARS)))):
        return

    if not ctx.require(
            "The measurement uncertainty is positive", measurement > 0,
            "R is a variance, so it can only be positive. At zero the filter "
            "believes the sensor completely and the estimate is just the noisy "
            "measurement back again.",
            got=_fmt(measurement)):
        return
    if not ctx.require(
            "The model uncertainty is not negative", model >= 0,
            "Q is a variance too. A negative one means less than no uncertainty, "
            "which the maths has no way to interpret — use 0 if you want the "
            "filter to trust the model completely.",
            got=_fmt(model)):
        return

    ctx.require(
        "At least one of the two uncertainties was changed",
        (abs(measurement - SHIPPED_MEASUREMENT_UNCERTAINTY) > 1e-9
         or abs(model - SHIPPED_MODEL_UNCERTAINTY) > 1e-9),
        "Both numbers are still the ones the cell came with. Change at least one "
        "and re-run — that is the exercise: a bigger R leans on the model, a "
        "bigger Q leans on the sensor.",
        expected=f"something other than R={SHIPPED_MEASUREMENT_UNCERTAINTY:g}, "
                 f"Q={SHIPPED_MODEL_UNCERTAINTY:g}",
        got=f"R={measurement:g}, Q={model:g}")

    data = _from_env(ctx, "measured_position")
    if data is None:
        return  # the notebook's data cell didn't run; nothing to re-derive from
    length = data.size

    # Re-derive the run from their own numbers. If they rewrote the filter in
    # this cell, judge them against their own version rather than ours.
    initial = _number(_from_call(arguments, INITIAL_ESTIMATE_VARS))
    if initial is None:
        initial = float(data[0])
    initial_p = _number(_from_call(arguments, INITIAL_UNCERTAINTY_VARS))
    if initial_p is None:
        initial_p = INITIAL_UNCERTAINTY
    if "simple_kalman_filter" in ctx.assigned and callable(ctx.env.get("simple_kalman_filter")):
        try:
            expected_estimate, expected_gain = ctx.env["simple_kalman_filter"](
                data, initial, initial_p, measurement, model)
            expected_estimate = _series(expected_estimate, length)
            expected_gain = _series(expected_gain, length)
        except Exception:
            expected_estimate = expected_gain = None
    else:
        expected_estimate, expected_gain = _simple_kalman(
            data, initial, initial_p, measurement, model)
    if expected_estimate is None or expected_gain is None:
        return

    # The cell's own names first. A student who tried three tunings in a loop
    # leaves the last one in whatever names the loop used, so fall back to the
    # series in their namespace that looks most like a filter run.
    truth = _from_env(ctx, "true_position", length)
    given = [series for series in (_from_env(ctx, "time", length), truth, data)
             if series is not None]
    candidates = _candidate_series(ctx, length, given)
    estimate = _series(ctx.get(*ESTIMATE_VARS), length)
    if estimate is None:
        _, estimate, _ = _closest(candidates, expected_estimate)
    gains = _series(ctx.get(*GAIN_VARS), length)
    if gains is None:
        _, gains, _ = _closest(candidates, expected_gain)
    if not ctx.require(
            "The filter is still run over the measurements",
            estimate is not None and gains is not None,
            "Keep the call to simple_kalman_filter and keep both of the things it "
            "hands back — kalman_position and kalman_gain — one value per "
            "measurement.",
            expected=f"two series of {length} values",
            got=f"estimate: {_shape(ctx.get(*ESTIMATE_VARS))}, "
                f"gain: {_shape(ctx.get(*GAIN_VARS))}"):
        return

    ctx.require(
        "The estimate is the filter's own output",
        not (_same(estimate, truth) or _same(estimate, data)),
        "That estimate is an exact copy of a series the notebook already had — "
        "true_position is the answer key the dashed line is drawn from, and "
        "measured_position is the raw sensor. Run simple_kalman_filter over the "
        "measurements and plot what it hands back.",
        expected="whatever the filter returned",
        got="a copy of true_position" if _same(estimate, truth)
            else "a copy of measured_position")

    ctx.require(
        "The estimate is what those two numbers produce",
        _same(estimate, expected_estimate),
        "Your kalman_position isn't the filter run with your R and Q. Re-run the "
        "cell so the estimate is recomputed — changing the numbers only matters "
        "if the filter sees them.",
        expected=f"starts {expected_estimate[0]:.2f}, ends {expected_estimate[-1]:.2f}",
        got=f"starts {estimate[0]:.2f}, ends {estimate[-1]:.2f}")
    ctx.require(
        "The Kalman gain is what those two numbers produce",
        _same(gains, expected_gain),
        "Your kalman_gain isn't the gain series those two numbers give. It comes "
        "out of the filter alongside the estimate — don't set it by hand.",
        expected=f"settles at {expected_gain[-1]:.4f}",
        got=f"settles at {gains[-1]:.4f}")

    ctx.require(
        "The Kalman gain stays between 0 and 1",
        bool(np.all(gains > 0.0) and np.all(gains < 1.0)),
        "K = P / (P + R) is a blend: 0 means ignore the sensor, 1 means ignore "
        "the model, and nothing outside that is meaningful. A gain outside the "
        "range means R or Q went somewhere the maths can't follow.",
        expected="0 < K < 1",
        got=f"lowest {gains.min():.4f}, highest {gains.max():.4f}")

    settled = float(expected_gain[-1])
    low, high = TRACKING_GAIN_BAND
    if truth is not None and low <= settled <= high:
        raw = _rms(data, truth)
        filtered = _rms(estimate, truth)
        if raw is not None and filtered is not None:
            ctx.require(
                "The estimate follows the rocket better than the raw sensor",
                filtered < raw,
                "With a gain in this range the filter should be beating the raw "
                "measurements, and it isn't. Look at the plot: if the line lags "
                "behind the dots, the model — not the sensor — is what's wrong, "
                "so raise MODEL_UNCERTAINTY.",
                expected=f"below {raw:.1f} m RMS (the raw measurements)",
                got=f"{filtered:.1f} m RMS")

    plots = _plot_calls(ctx.tree)
    ctx.require(
        "The plot is still drawn", _drew_a_figure(ctx, 2),
        "Keep the plotting code: the whole point of changing R and Q is seeing "
        "what it does to the curve.",
        expected="the measurements and the estimate, on one figure",
        got=f"{len(plots)} plotting call(s)")


# ---------------------------------------------------------------------------
# Exercise 2 — tune Q inside the position/velocity filter
# ---------------------------------------------------------------------------

def _ex2_estimates(ctx, length):
    """The position and velocity series, however the student kept them."""
    states = _array(ctx.get(*STATE_VARS))
    if states is not None and states.ndim == 2 and states.shape[0] == length \
            and states.shape[1] >= 2:
        return states[:, 0], states[:, 1]
    return (_series(ctx.get(*POSITION_ESTIMATE_VARS), length),
            _series(ctx.get(*VELOCITY_ESTIMATE_VARS), length))


def check_ex2(ctx):
    """"Change the values inside `Q` and rerun the cells."

    The whole filter is sitting in the cell, so most of this is an anti-deletion
    net: the predict/update skeleton has to survive, the three return series have
    to survive, and the plot has to survive. The graded part is Q itself — that
    it moved, that it is still a covariance (2x2, symmetric, no negative
    variances), and that the states and residuals in the student's lists came
    out of the same filter run rather than off the truth the cell plots against.

    The two accuracy rows are gated: each one is only asked for when this same
    filter, re-run here with the student's own Q, clears it too. "Making
    position uncertainty larger" is one of the things the cell tells them to
    try, and a big enough Q[0][0] really does wreck the velocity estimate —
    that is the experiment working, and failing them for running it would be
    perverse.
    """
    if not _ran(ctx):
        return

    if not ctx.require(
            "The filter function is still defined in this cell", ctx.uses("def"),
            "Keep the whole function in the cell — you only need to change the "
            "numbers inside Q."):
        return

    loops = _filter_loops(ctx.tree)
    ctx.require(
        "The predict step still propagates the state and its uncertainty",
        _in_loops(loops, _has_matmul) and _has_transpose(ctx.tree),
        "x_pred = F @ x and P_pred = F @ P @ F.T + Q are the predict step — "
        "without them the filter has no model to trust.")
    ctx.require(
        "The update step still computes a Kalman gain from the covariances",
        _in_loops(loops, lambda node: _calls_named(node, INVERSE_ATTRS)),
        "K = P_pred @ H.T @ np.linalg.inv(S) is what decides how much of the "
        "measurement to believe. It has to be computed each step, from S.")
    ctx.require(
        "The state is still corrected by the residual",
        _in_loops(loops, lambda node: _has_op(node, ast.Sub),
                  lambda node: _has_op(node, ast.Add)),
        "The residual y = z - H @ x_pred, and then x = x_pred + K @ y, are the "
        "update step: the prediction plus a share of how wrong it was.")

    # A Q we can't evaluate at all (built out of a function parameter, say) is
    # judged in the student's favour: the row passes and the value checks are
    # skipped. A Q that evaluates to something which simply isn't a matrix is a
    # different thing, and is caught below.
    node = _assigned_node(ctx.tree, {"Q", "q"})
    value = _eval_node(ctx, node)
    matrix = _matrix(value)
    shipped = np.array(SHIPPED_Q, dtype=float)
    if value is not None and matrix is None:
        ctx.require(
            "Q is a 2x2 matrix", False,
            "Q has to stay a 2x2 array. A single number or a flat list of two "
            "broadcasts into the covariance arithmetic without complaining and "
            "quietly stops being a covariance — write both rows out, e.g. "
            "Q = np.array([[0.05, 0], [0, 0.5]]).",
            expected="(2, 2)", got=_shape(value))
    if ctx.require(
            "The filter sets a process noise Q", node is not None,
            "Keep Q as a matrix written out in the function, e.g. "
            "Q = np.array([[0.05, 0], [0, 0.5]]) — that is the line this "
            "exercise is about.") and matrix is not None:
        ctx.require(
            "Q is not the matrix the cell shipped with",
            not (matrix.shape == shipped.shape and np.allclose(matrix, shipped)),
            "Q is still [[1, 0], [0, 3]]. Change the numbers inside it and re-run "
            "— the top-left entry is how much slack you give the position, the "
            "bottom-right how much you give the velocity.",
            got=str(matrix.tolist()))
        if ctx.require(
                "Q is a 2x2 matrix", matrix.shape == (2, 2),
                "The state is [position, velocity], so its process noise is a "
                "2x2 covariance — one row and one column per state.",
                expected="(2, 2)", got=str(matrix.shape)):
            ctx.require(
                "Q is symmetric", bool(np.allclose(matrix, matrix.T, atol=1e-9)),
                "A covariance matrix is symmetric: the covariance of position "
                "with velocity is the same number as velocity with position.",
                got=str(matrix.tolist()))
            determinant = float(np.linalg.det(matrix))
            ctx.require(
                "Q is a covariance the filter can use",
                bool(matrix[0, 0] >= 0 and matrix[1, 1] >= 0 and determinant >= -1e-9),
                "The diagonal of Q holds variances, and a variance is a squared "
                "quantity — it cannot be negative, and neither can the "
                "determinant. Making an entry small tells the filter to trust "
                "the model; making it negative tells it nothing at all.",
                got=f"diagonal ({matrix[0, 0]:g}, {matrix[1, 1]:g}), "
                    f"determinant {determinant:g}")

    data = _from_env(ctx, "measured_position")
    if data is None:
        return
    length = data.size

    position, velocity = _ex2_estimates(ctx, length)
    residuals = _series(ctx.get(*RESIDUAL_VARS), length)
    uncertainty = _array(ctx.get(*UNCERTAINTY_VARS))
    uncertainty_ok = (uncertainty is not None and uncertainty.ndim == 2
                      and uncertainty.shape[0] == length and uncertainty.shape[1] >= 4)
    if not ctx.require(
            "The filter still returns a state, a residual and an uncertainty "
            "for every step",
            position is not None and velocity is not None
            and residuals is not None and uncertainty_ok,
            "Keep all three lists and keep appending to them every step: states "
            "(position and velocity), residuals, and the flattened P. The plot "
            "underneath reads all three.",
            expected=f"states ({length}, 2), residuals ({length},), "
                     f"uncertainty_P ({length}, 4)",
            got=f"states {_shape(ctx.get(*STATE_VARS))}, "
                f"residuals {_shape(ctx.get(*RESIDUAL_VARS))}, "
                f"uncertainty {_shape(ctx.get(*UNCERTAINTY_VARS))}"):
        return

    # The residual is defined by the two lists either side of it: y = z - H @
    # x_pred, and x_pred is F @ (the state from the step before). Every real run
    # of this filter satisfies that to floating-point noise, whatever Q, R or P
    # the student chose — and a state list copied off true_position, however
    # carefully, does not.
    timestep = _number(ctx.env.get("dt")) or SHIPPED_DT
    predicted = position[:-1] + timestep * velocity[:-1]
    slip = _rms(residuals[1:], data[1:] - predicted)
    ctx.require(
        "Each residual is the measurement minus the filter's own prediction",
        slip is not None and slip <= RESIDUAL_TOLERANCE * max(1.0, float(np.max(np.abs(data)))),
        "The three lists have to come from one run of the filter: the residual "
        "kept at each step is that step's measurement minus the position the "
        "previous state predicts, y = z - H @ F @ x. Yours don't line up, so "
        "the states and the residuals were not produced side by side.",
        expected="z - (position + velocity * dt) from the step before",
        got=f"{slip:.3g} m out on average" if slip is not None else "not comparable")

    # Both accuracy rows are gated on the same filter, re-run with the student's
    # own Q and R, being able to clear them — see the module docstring.
    measurement_std = _number(_from_call(
        _call_args(ctx, _find_call(ctx, "kalman", STATE_VARS + RESIDUAL_VARS),
                   MATRIX_FILTER_PARAMS), MEASUREMENT_STD_VARS))
    if measurement_std is None:
        measurement_std = SHIPPED_MEASUREMENT_STD
    initial_P = _matrix(_eval_node(ctx, _assigned_node(ctx.tree, {"P"}, first=True)))
    reference = None
    if matrix is not None and initial_P is not None:
        reference = _matrix_kalman(data, timestep, measurement_std, matrix, initial_P)
    reference_states, reference_P = reference if reference is not None else (None, None)

    truth = _from_env(ctx, "true_position", length)
    if truth is not None:
        raw = _rms(data, truth)
        filtered = _rms(position, truth)
        reachable = raw
        if reference_states is not None:
            best = _rms(reference_states[:, 0], truth)
            reachable = raw if best is None else max(raw, best)
        if raw is not None and filtered is not None and reachable <= raw:
            ctx.require(
                "The position estimate follows the rocket better than the raw sensor",
                filtered < raw * EX2_TRACKING_RATIO,
                "This rocket really does climb at a constant rate, so the "
                "constant-velocity model is a good one and almost any sensible Q "
                "beats the raw dots. If yours doesn't, check that Q is small "
                "relative to R rather than the other way round.",
                expected=f"below {raw:.1f} m RMS (the raw measurements)",
                got=f"{filtered:.1f} m RMS")

    climb = _true_climb_rate(ctx)
    late = _late(velocity)
    settles = True
    if reference_states is not None:
        reference_late = _late(reference_states[:, 1])
        settles = bool(reference_late is not None and reference_late.size
                       and abs(float(reference_late.mean()) - climb) <= EX2_VELOCITY_TOLERANCE)
    if settles and late is not None and late.size:
        ctx.require(
            "The velocity estimate settles near the rocket's true climb rate",
            abs(float(late.mean()) - climb) <= EX2_VELOCITY_TOLERANCE,
            "The sensor never measures velocity — the filter infers it from how "
            "position changes, and it should converge on the real climb rate. "
            "Wandering off means Q is telling the filter to distrust the model "
            "far more than the data deserves.",
            expected=f"about {climb:.1f} m/s by the end of the run",
            got=f"{float(late.mean()):.1f} m/s on average over the last quarter")

    position_p = uncertainty[:, 0]
    ctx.require(
        "The position uncertainty stays positive and finite",
        bool(np.all(np.isfinite(position_p)) and np.all(position_p > 0)),
        "P is a covariance: its diagonal is a variance and can never be "
        "negative or NaN. If it went there, Q did — a covariance matrix has to "
        "be symmetric with a non-negative diagonal.",
        got=f"lowest {np.nanmin(position_p):g}")
    if position_p.size > 2 and np.all(np.isfinite(position_p)):
        scale = float(np.mean(np.abs(position_p))) or 1.0
        step = abs(float(position_p[-1] - position_p[-2]))
        ctx.require(
            "The uncertainty settles instead of drifting",
            step <= SETTLE_FRACTION * scale,
            "Predict adds Q and grows the uncertainty; the update shrinks it "
            "again. After a while the two balance and P stops moving — that "
            "steady value is the filter telling you how well it can ever do.",
            expected="P barely changing by the last step",
            got=f"still moving {step:g} per step")
    if reference_P is not None and position_p.size and np.all(np.isfinite(position_p)):
        expected_p = float(reference_P[-1, 0])
        got_p = float(position_p[-1])
        ratio = max(got_p, expected_p) / min(got_p, expected_p) if min(got_p, expected_p) > 0 \
            else float("inf")
        ctx.require(
            "The uncertainty settles where Q and R say it should",
            ratio <= SETTLED_P_TOLERANCE,
            "P never looks at the measurements — Q, R and the model fix it "
            "completely, so for a given Q there is exactly one value it can "
            "settle at. Yours settled somewhere else, which means the numbers "
            "being plotted didn't come out of the Q now written in the "
            "function. Re-run the cell so the filter actually uses it.",
            expected=f"about {expected_p:.1f} m^2 with this Q and R",
            got=f"{got_p:.1f} m^2")

    plots = _plot_calls(ctx.tree)
    ctx.require(
        "The three-panel plot is still drawn", _drew_a_figure(ctx, 4),
        "Keep the three panels — position, velocity and uncertainty. Watching "
        "all three move together is how you tell what Q actually did.",
        expected="position, velocity and uncertainty panels",
        got=f"{len(plots)} plotting call(s)")


# ---------------------------------------------------------------------------
# Exercise 3 — the final challenge, pinned as `#% checker: tune_the_filter`
# ---------------------------------------------------------------------------

def check_tune_the_filter(ctx):
    """The final challenge: write the position/velocity filter and tune it.

    The student writes the filter themselves this time, so the first thing to
    establish is that they did — a predict/update loop with matrix arithmetic
    and a gain computed from the covariances, not a moving average and not a
    curve copied off the truth that happens to be printed in the same cell.

    After that it is accuracy, with one large caveat: the truth here
    *accelerates* (20 + 8t + 1.25t^2) while the filter models constant
    velocity. That mismatch is the lesson — the position estimate can still get
    comfortably inside the raw noise, but the velocity estimate is guaranteed
    to lag, permanently. Both bands were measured by running good tunings over
    this data rather than reasoned about, and the velocity message says out
    loud that the lag is expected.
    """
    if not _ran(ctx):
        return

    time = _from_env(ctx, "time")
    truth = _from_env(ctx, "true_position")
    true_velocity = _from_env(ctx, "true_velocity")
    measured = _from_env(ctx, "measured_position")
    length = time.size if time is not None else 0
    if not ctx.require(
            "The challenge data is still in place",
            length > 0 and all(series is not None and series.size == length
                               for series in (truth, true_velocity, measured)),
            "Keep the block at the top of the cell that builds time, "
            "true_position, true_velocity and measured_position — your filter is "
            "judged against that truth, so it has to still be there.",
            got=f"time {_shape(ctx.env.get('time'))}, "
                f"measured_position {_shape(ctx.env.get('measured_position'))}"):
        return

    loops = _filter_loops(ctx.tree)
    ctx.require(
        "The filter runs as a loop over the measurements", bool(loops),
        "Step through the measurements one at a time — `for z in "
        "measured_position:` — predicting and then updating on each one. A "
        "Kalman filter is a recursion, not a formula applied to the whole "
        "array at once.")
    ctx.require(
        "The loop does the predict/update matrix arithmetic",
        _in_loops(loops, lambda node: _has_matmul(node) or _has_transpose(node)),
        "Build the matrices — F, H, P, Q, R — and use them: x_pred = F @ x, "
        "P_pred = F @ P @ F.T + Q. Smoothing the measurements some other way "
        "isn't a Kalman filter, however good the plot looks.")
    ctx.require(
        "The Kalman gain is computed from the covariances, not picked",
        _in_loops(loops, lambda node: _calls_named(node, INVERSE_ATTRS)
                  or _has_op(node, ast.Div)),
        "K = P_pred @ H.T @ np.linalg.inv(S) — the gain has to come out of the "
        "uncertainties each step. A fixed number you chose yourself is an alpha "
        "filter, which is where this module started.")

    given = [series for series in (time, truth, true_velocity, measured)
             if series is not None]
    candidates = _candidate_series(ctx, length, given)
    _, position, position_names = _closest(candidates, truth)
    if not ctx.require(
            "The filter produces an altitude estimate for every measurement",
            position is not None,
            "Collect the position half of the state as you go and keep it in its "
            "own variable — one value per measurement, e.g. estimated_position = "
            "states[:, 0]. It has to be your filter's own output; a copy of "
            "true_position doesn't count.",
            expected=f"a series of {length} values"):
        return

    raw = _rms(measured, truth)
    filtered = _rms(position, truth)
    if raw is not None and filtered is not None:
        ctx.require(
            "The altitude estimate is clearly closer to the truth than the sensor",
            filtered < EX3_TRACKING_RATIO * raw,
            "This is the bar that tuning actually has to clear. The rocket is "
            "accelerating and your model says constant velocity, so if Q is too "
            "small the estimate falls behind and ends up no better than the raw "
            "dots — open up the velocity entry of Q until the line stops lagging.",
            expected=f"below {EX3_TRACKING_RATIO * raw:.1f} m RMS "
                     f"(the raw sensor is {raw:.1f} m)",
            got=f"{filtered:.1f} m RMS")

    _, velocity, velocity_names = _closest(candidates, true_velocity)
    late_error = None
    if velocity is not None:
        late = _late(velocity)
        late_truth = _late(true_velocity)
        if late is not None and late.size and late_truth is not None:
            late_error = abs(float(late.mean()) - float(late_truth.mean()))
    if ctx.require(
            "The filter estimates velocity as well as position",
            late_error is not None and late_error <= EX3_VELOCITY_FOUND,
            "The state is [position, velocity]: keep the second half too, e.g. "
            "estimated_velocity = states[:, 1]. The sensor never measures it — "
            "the filter infers it from how the position moves, which is the "
            "whole reason for the matrix form.",
            expected="a velocity series somewhere near the truth"):
        ctx.require(
            "The velocity estimate ends up near the rocket's true velocity",
            late_error <= EX3_VELOCITY_TOLERANCE,
            "A constant-velocity filter can never quite catch a rocket that is "
            f"speeding up — expect it to sit a few m/s low, and anything within "
            f"{EX3_VELOCITY_TOLERANCE:.0f} m/s by the end of the run is fine "
            "here. Much further out than that and Q is too tight for the "
            "velocity to keep up at all.",
            expected=f"within {EX3_VELOCITY_TOLERANCE:.0f} m/s of "
                     f"{float(_late(true_velocity).mean()):.1f} m/s",
            got=f"{late_error:.1f} m/s out on average over the last quarter")

    plots = _plot_calls(ctx.tree)
    measured_names = ["measured_position"] + _aliases(ctx, measured, length)
    truth_names = ["true_position"] + _aliases(ctx, truth, length)
    true_velocity_names = ["true_velocity"] + _aliases(ctx, true_velocity, length)
    ctx.require(
        "Plots the measured altitude against the true altitude",
        _mentions(plots, measured_names) and _mentions(plots, truth_names),
        "Draw the noisy measurements and the true altitude on the same axes "
        "first — the estimate only means something next to what it is trying to "
        "recover and what it had to work with.",
        got=f"{len(plots)} plotting call(s)")
    ctx.require(
        "Plots the filter's altitude estimate alongside them",
        _mentions(plots, sorted(position_names)),
        "Add your estimated altitude to that plot. Tuning is a visual job: you "
        "are looking for a line that is smooth but still keeps up.",
        expected="your estimate plotted with the measurements and the truth")
    ctx.require(
        "Plots the estimated velocity against the true velocity",
        _mentions(plots, sorted(velocity_names))
        and (_mentions(plots, true_velocity_names) or _calls_named(ctx.tree, {"axhline"})),
        "Plot the estimated velocity and true_velocity together, on their own "
        "axes. The gap between them is the price of modelling an accelerating "
        "rocket as a constant-velocity one — worth seeing.",
        expected="estimated and true velocity, on one figure")
