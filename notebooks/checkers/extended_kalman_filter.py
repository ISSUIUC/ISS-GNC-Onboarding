"""Behaviour checkers for extended_kalman_filter.ipynb.

Two exercises, both of them open-ended. The "Your Turn" cell hands the student
a working EKF and asks them to move the radar and turn the measurement noise
up; the final challenge asks them to bolt a second, differently-scaled
measurement onto it. Neither has a single right answer, so nothing here diffs
variables — it re-derives what the student's own code should have produced and
checks that against what it did.

Four ideas do most of the work, and they are shared between the two:

*Verify a Jacobian by differentiating.* A Jacobian is the one thing in an EKF
that is easy to get subtly wrong and impossible to spot in a plot, so we call
the student's own H at a few points on the trajectory and compare it against a
central finite difference of the student's own h. That is honest — it doesn't
care how they wrote either function, what they named them, whether their state
arrives as a column or a flat array, or which order they put the rows in — and
it catches the one mistake that matters.

*Keep our own copy of the filter.* `_reference_ekf` is the notebook's range-only
EKF, written out again here. A checker file is exec'd on its own so it can't
import anything from the notebook, and it shouldn't lean on the copy sitting in
`ctx.env` either: that one is precisely the code the student may have just
broken. Having our own gives us a yardstick — "what should this filter have
done with *your* settings" — which is what lets the tracking checks pass a
student who deliberately cranked the noise up to watch the estimate fall apart,
while still failing one whose filter is wrong.

*Read the call, not the variable.* "Make the radar closer" can be written three
ways: reassign `d`, invent `radar_distance`, or type the number straight into
the call. So `_last_filter_call` finds the call to the EKF in their source and
evaluates its arguments in the namespace they left behind, which understands
all three. Where reading the source can't settle a question, behaviour does:
whether the cell was retuned at all is decided by running the shipped setup
ourselves and seeing whether their estimate is the one the worked example
already drew, which is right about a loop over three distances, a helper that
takes them as arguments, and a `measurement_std` that was assigned and then
never passed to anything.

*Find the run by shape, not by name.* Which array holds the history being
graded has no single spelling either. The names the prompt suggests come first,
then names that read like a record of states, then — at exactly the right
length — anything else; a stack of (2, 1) columns counts, and so does a bare
series of altitudes. Two kinds of array are dropped rather than graded: ones we
can already identify as the student's own measurements, and ones that were only
ever copied out of an earlier cell, that being the one way to produce a
perfect-looking answer without running a filter at all.

Tolerances are the constants below. Widen them, don't weaken a check: judging a
genuinely ambiguous case in the student's favour is fine, marking correct work
wrong is not.
"""

import ast

import numpy as np

# --- what the notebook's setup cell ships with -----------------------------
# Used to spot a cell that was run unchanged, and to reconstruct the truth if
# the student has overwritten one of the setup variables.
SHIPPED_D = 20000.0                 # radar distance in the worked example
SHIPPED_MEASUREMENT_STD = 15.0      # measurement_std it passes to the filter
SHIPPED_RANGE_NOISE = 25.0          # sigma of the noise on `measured_range`
POSITION, VELOCITY = 0, 1           # the state is [position, velocity] throughout
TRUE_START = 10.0                   # rocket altitude at t = 0, m
TRUE_VELOCITY = 450.0               # its constant climb rate, m/s
DEFAULT_DT = 0.1

# --- names we accept -------------------------------------------------------
# `ctx.defined` only matches names the *submission* assigns, so short ones are
# safe: a leftover from a replayed setup cell can't collide with them.
DISTANCE_VARS = ("d", "radar_distance", "radar_d", "radar_range", "distance",
                 "d_new", "new_d", "d_radar", "range_to_pad", "baseline", "D")
STD_VARS = ("measurement_std", "meas_std", "measurement_sigma", "sensor_std",
            "range_std", "sigma", "std", "noise_std")
MEASUREMENT_VARS = ("measured_range", "measurements", "measured", "z",
                    "ranges", "measured_ranges", "range_measurements",
                    "noisy_range", "measured_r", "new_measurements")
STATE_VARS = ("states_both", "states_rb", "states_range_bearing", "states_bearing",
              "states_full", "states_combined", "states_new", "states2", "states",
              "state_history", "x_history", "history", "estimates", "filtered_states")
# ...and, failing those, any name that reads like a record of states.
STATE_HINTS = ("state", "estimate", "_est", "est_", "hist", "track", "traj",
               "filtered", "result", "output", "xs")
# A run kept as bare altitudes has to be recognised by name alone — a cell is
# full of 1-D arrays of the right length, and most of them are not the answer.
POSITION_HINTS = ("pos", "alt", "height", "state", "estimate", "est_", "_est",
                  "filtered", "track", "hist", "xs")

EX1_H_NAMES = ("h", "h_range", "measurement", "measurement_function", "hx")
EX1_JACOBIAN_NAMES = ("H_jacobian", "H_jac", "jacobian", "H", "Hjacobian")
EX1_FILTER_NAMES = ("extended_kalman_filter", "ekf", "EKF", "kalman_filter",
                    "extended_kf", "run_ekf")

EX2_H_NAMES = ("h_rb", "h2", "h_2", "h_range_bearing", "h_full", "h_both", "h",
               "measurement", "measurement_function", "measurement_model", "hx")
EX2_JACOBIAN_NAMES = ("H_jacobian_rb", "H_jacobian2", "H_jacobian_2",
                      "H_jacobian_range_bearing", "H_jacobian_full",
                      "H_jacobian", "H_rb", "H2", "H_full", "H",
                      "jacobian", "jacobian_rb", "measurement_jacobian",
                      "H_jac", "Hjac", "jac")
R_VARS = ("R", "R2", "R_rb", "R_matrix", "measurement_noise", "meas_noise",
          "measurement_covariance", "sensor_noise", "noise_matrix")

# Anything with one of these in its name is a filter, not a measurement model —
# skipped when we go hunting for h and H so we never try to drive a 200-step
# loop with a two-element array.
FILTER_HINTS = ("kalman", "ekf", "filter", "predict", "update", "plot", "run")
ARCTAN_NAMES = ("arctan", "arctan2", "atan", "atan2")
PLOT_CALLS = ("plot", "scatter", "step", "errorbar")

# --- how the models get probed ---------------------------------------------
# Three points spread up the real trajectory (the rocket climbs 10 m -> ~9 km).
TEST_POSITIONS = (500.0, 3000.0, 7000.0)
TEST_VELOCITY = 450.0
FD_STEP_POSITION = 1.0              # m; the central difference step
FD_STEP_VELOCITY = 1.0              # m/s
MODEL_RTOL = 1e-6                   # how exactly h must reproduce sqrt(d^2+h^2)
BEARING_ROW_TOL = 1e-6              # rad, same idea for the bearing row
# The Jacobian is compared entry by entry. Its two rows differ by four orders
# of magnitude (m/m against rad/m), so the tolerance has to be relative; the
# absolute floor only exists to let an exact zero be an exact zero.
JACOBIAN_RTOL = 1e-2
JACOBIAN_ATOL = 1e-9

# --- how good the estimates have to be -------------------------------------
# Position error is judged over the back half of the flight: the front half is
# the filter hauling itself in from a deliberately bad initial guess, which the
# notebook makes a point of.
TRACKING_SLACK = 2.0                # times what our own EKF manages, same settings
POSITION_RMS_FLOOR = 400.0          # m, out of a ~9 km climb
VELOCITY_BAND = 75.0                # m/s either side of the true 450
# No filter fed noisy ranges reproduces the truth: with the shipped Q the
# estimate jitters by metres even from a perfect sensor. An error below this
# means the "estimate" was copied from `true_position`, not filtered.
IMPOSSIBLE_RMS = 0.01               # m
# The bearing has to actually reach the update. If a run comes out identical
# to the range-only one to within this, it never did.
BEARING_EFFECT_TOL = 1.0            # m
# Same idea for "did you change the radar at all": a cell that changed nothing
# draws exactly the plot the worked example above already drew.
SETTINGS_EFFECT_TOL = 1.0           # m
# Range measurements have to belong to the radar distance they were filtered
# with. The mismatch is measured against the scatter of the data itself, so a
# student who also turned the sensor noise up doesn't get failed for it.
RANGE_MATCH_TOL = 250.0             # m, ten times the shipped sensor noise
RANGE_MATCH_SIGMAS = 6.0
# A bearing series is recognised by shape, not by exactness — a student may add
# any noise they like to it, or none.
BEARING_MATCH_TOL = 0.08            # rad, about 4.5 degrees
BASELINE_RANGE_STD = 15.0           # what the range-only yardstick run trusts
COMPARISON_SLACK = 1.5              # adding a measurement must not make it worse
COMPARISON_FLOOR = 50.0             # m, below which the two runs are a tie


# ---------------------------------------------------------------------------
# Reading values out of a submission, defensively
# ---------------------------------------------------------------------------

def _ran(ctx, hint="Fix the error shown in the output above, then check again."):
    """Every exercise starts here: nothing else means anything if it crashed."""
    return ctx.require("Runs without an error", ctx.ok, hint)


def _numeric_array(value, ndim=None):
    """`value` as a finite float array, or None if it isn't one.

    The single door every student value comes through, so that a string, a
    ragged list, a dict, a function or a column of NaNs turns into None here
    instead of an exception fifteen lines later.
    """
    if value is None or isinstance(value, (str, bytes, dict, set)):
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    if ndim is not None and arr.ndim != ndim:
        return None
    return arr


def _number(value):
    """`value` as a plain float, or None. Booleans are not numbers here."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(-1)[0]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _pair(value):
    """`value` as exactly two floats — (2,), (2, 1), [a, b] all count."""
    arr = _numeric_array(value)
    if arr is None or arr.size != 2:
        return None
    return arr.reshape(-1)


def _as_matrix(value, rows):
    """`value` read as a Jacobian with `rows` rows, or None.

    A flat result of the right length is read as a single column — the
    charitable reading of `np.array([dr_dh, dtheta_dh])`, someone who only
    differentiated with respect to position. For a one-row model a flat result
    is that row instead, since there is nothing else it could be.
    """
    arr = _numeric_array(value)
    if arr is None:
        return None
    if arr.ndim == 1:
        if rows == 1:
            return arr.reshape(1, -1)
        return arr.reshape(rows, 1) if arr.shape[0] == rows else None
    if arr.ndim == 2 and arr.shape[0] == rows:
        return arr
    return None


def _called_name(node):
    """`np.arctan(...)` -> "arctan", `foo(...)` -> "foo"."""
    func = node.func
    return getattr(func, "id", None) or getattr(func, "attr", None)


def _first_name(target):
    """The first plain name bound by an assignment target."""
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            name = _first_name(element)
            if name:
                return name
    return None


def _const(node):
    """A literal number written in the source, or None."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _const(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
        return _number(node.value)
    return None


def _arg(call, index, keyword):
    """One argument of a call, given either positionally or by name."""
    if call is None:
        return None
    for kw in call.keywords:
        if kw.arg == keyword:
            return kw.value
    if index < len(call.args):
        return call.args[index]
    return None


def _value_of(ctx, node):
    """Evaluate a source expression in the namespace the submission left.

    This is how `extended_kalman_filter(close_range, dt, 4000, ...)` tells us
    the radar distance even though nothing was ever assigned to `d`. Names
    hold their final value, so it is only meaningful for the last call — which
    is all we ask it for.
    """
    if node is None:
        return None
    try:
        expression = ast.Expression(body=node)
        ast.fix_missing_locations(expression)
        return eval(compile(expression, "<checker>", "eval"), dict(ctx.env))
    except Exception:
        return None


def _true_position(ctx):
    """The rocket's real altitude, from the setup cell or rebuilt if it's gone."""
    time = _numeric_array(ctx.env.get("time"), ndim=1)
    if time is None:
        time = np.arange(0.0, 20.0, DEFAULT_DT)
    truth = _numeric_array(ctx.env.get("true_position"), ndim=1)
    if truth is None or truth.shape != time.shape:
        truth = TRUE_START + TRUE_VELOCITY * time
    return truth


def _model_name(ctx, names):
    """The first of `names` that is callable, theirs preferred over inherited.

    The exercise-1 cell ships with `h`, `H_jacobian` and the filter already in
    it, and a student may reasonably delete the copies and retune the ones
    defined in the worked example above instead. So we look at what the
    submission assigned first — that is the version we want to probe if it
    exists — and fall back to the notebook's own. Shadowing one of them with
    something that isn't callable still comes back as "missing", which is the
    honest answer.
    """
    for source in (ctx.assigned, ctx.env):
        for name in names:
            if name in source and callable(ctx.env.get(name)):
                return name
    return None


# ---------------------------------------------------------------------------
# Finding the student's models and driving them
# ---------------------------------------------------------------------------

def _candidates(ctx, preferred, hints=(), exclude=()):
    """Callables the submission defined, most likely first.

    The prompt asks for particular names, but a checker that only knows one
    spelling is a checker that fails correct work — so the suggested names come
    first, then anything whose name hints at the right job (`sensor_jac`), then
    the rest. Filters are skipped throughout: they want a whole measurement
    series, and feeding one a two-element state would only waste time.
    """
    named, hinted, rest = [], [], []
    for name in list(preferred) + sorted(ctx.assigned):
        if (name in named or name in hinted or name in rest
                or name in exclude or name not in ctx.assigned
                or not callable(ctx.env.get(name))):
            continue
        if any(hint in name.lower() for hint in FILTER_HINTS):
            continue
        if name in preferred:
            named.append(name)
        elif any(hint in name.lower() for hint in hints):
            hinted.append(name)
        else:
            rest.append(name)
    return [(n, ctx.env[n]) for n in named + hinted + rest]


def _adapter(function, d, accept):
    """A uniform `g(position, velocity)` wrapper around a student's model.

    They may take the state as the notebook's (2, 1) column or as a flat
    (2,) array, and may take `d` as an argument or close over the module-level
    one. We settle which by trying, once, then stick to it — a finite
    difference only means anything if every call is made the same way.
    """
    for column in (True, False):
        for pass_d in (True, False):
            def call(position, velocity, column=column, pass_d=pass_d):
                x = (np.array([[float(position)], [float(velocity)]]) if column
                     else np.array([float(position), float(velocity)]))
                return function(x, d) if pass_d else function(x)

            try:
                probes = [call(px, TEST_VELOCITY) for px in TEST_POSITIONS]
            except Exception:
                continue
            if all(accept(probe) for probe in probes):
                return call
    return None


def _find_model(ctx, preferred, accept, d, prefer=None, hints=(), exclude=()):
    """The student's measurement model (or Jacobian), and how to call it.

    `prefer` is an optional second opinion: when several functions have the
    right shape, the one that also behaves correctly wins, so a stray helper
    can't shadow the real thing. Returns (name, callable) or (None, None).
    """
    fallback = (None, None)
    for name, function in _candidates(ctx, preferred, hints, exclude):
        call = _adapter(function, d, accept)
        if call is None:
            continue
        if prefer is None or prefer(call):
            return name, call
        if fallback[0] is None:
            fallback = (name, call)
    return fallback


def _matches_range_bearing(call, d):
    """Does this model return the slant range and the bearing?

    Either order, radians or degrees: both are self-consistent choices that
    the student's own Jacobian and R can be written around, and the filter
    works out the same. What we won't accept is a model that isn't measuring
    those two things at all.
    """
    for swapped in (False, True):
        for scale in (1.0, np.pi / 180.0):
            if all(_row_pair_ok(call, px, d, swapped, scale) for px in TEST_POSITIONS):
                return True
    return False


def _row_pair_ok(call, position, d, swapped, scale):
    try:
        pair = _pair(call(position, TEST_VELOCITY))
    except Exception:
        return False
    if pair is None:
        return False
    slant, bearing = (pair[1], pair[0]) if swapped else (pair[0], pair[1])
    want_slant = float(np.sqrt(d ** 2 + position ** 2))
    want_bearing = float(np.arctan2(position, d))
    return (abs(slant - want_slant) <= MODEL_RTOL * max(1.0, abs(want_slant))
            and abs(bearing * scale - want_bearing) <= BEARING_ROW_TOL)


def _fd_jacobian(call, position, velocity):
    """Central finite difference of a measurement model, as a (rows, 2) array."""
    columns = []
    for dp, dv, step in ((FD_STEP_POSITION, 0.0, FD_STEP_POSITION),
                         (0.0, FD_STEP_VELOCITY, FD_STEP_VELOCITY)):
        plus = _numeric_array(call(position + dp, velocity + dv))
        minus = _numeric_array(call(position - dp, velocity - dv))
        if plus is None or minus is None or plus.shape != minus.shape:
            return None
        columns.append((plus.reshape(-1) - minus.reshape(-1)) / (2.0 * step))
    return np.column_stack(columns)


def _jacobian_agrees(theirs, expected):
    """Entry-by-entry agreement, over however many columns they gave us."""
    if theirs is None or expected is None:
        return False
    width = min(theirs.shape[1], expected.shape[1])
    if width < 1 or theirs.shape[0] != expected.shape[0]:
        return False
    mine = expected[:, :width]
    return bool(np.all(np.abs(theirs[:, :width] - mine)
                       <= JACOBIAN_ATOL + JACOBIAN_RTOL * np.abs(mine)))


# ---------------------------------------------------------------------------
# The filter itself, and what it should have produced
# ---------------------------------------------------------------------------

def _reference_ekf(measurements, dt, d, measurement_std):
    """The notebook's range-only EKF, written out again so we own a copy.

    Returns the (steps, 2) state history, or None if the settings make it
    meaningless (a zero-variance sensor, a radar on top of the pad).
    """
    measurements = _numeric_array(measurements, ndim=1)
    dt = _number(dt)
    d = _number(d)
    std = _number(measurement_std)
    if measurements is None or dt is None or d is None or std is None:
        return None
    if abs(std) < 1e-9 or abs(d) < 1e-9:
        return None
    x = np.array([[250.0], [0.0]])
    P = np.array([[500.0, 0.0], [0.0, 500.0]])
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = np.array([[1.0, 0.0], [0.0, 3.0]])
    R = np.array([[std ** 2]])
    identity = np.eye(2)
    states = []
    try:
        for z in measurements:
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
            position = x_pred[0, 0]
            slant = np.sqrt(d ** 2 + position ** 2)
            H = np.array([[position / slant, 0.0]])
            y = float(z) - slant
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            x = x_pred + K * y
            P = (identity - K @ H) @ P_pred
            states.append(x.flatten())
    except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
        return None
    out = np.array(states)
    return out if np.all(np.isfinite(out)) else None


def _filter_names(ctx):
    """Names in the submission that look like the EKF entry point."""
    names = set(EX1_FILTER_NAMES)
    for name in ctx.assigned:
        if not callable(ctx.env.get(name)):
            continue
        if any(hint in name.lower() for hint in ("kalman", "ekf")):
            names.add(name)
    return names


def _filter_calls(ctx):
    """Every call to the EKF in the submission, in source order."""
    names = _filter_names(ctx)
    calls = [node for node in ast.walk(ctx.tree)
             if isinstance(node, ast.Call) and _called_name(node) in names]
    calls.sort(key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))
    return calls


def _last_filter_call(ctx):
    """(call, name the state history went into) for the last EKF call."""
    calls = _filter_calls(ctx)
    if not calls:
        return None, None
    targets = {}
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            targets[id(node.value)] = _first_name(node.targets[0])
    last = calls[-1]
    return last, targets.get(id(last))


def _settings(ctx):
    """(measurements, d, measurement_std, state variable) actually filtered with."""
    call, states_name = _last_filter_call(ctx)
    measurements = _numeric_array(_value_of(ctx, _arg(call, 0, "measurements")), ndim=1)
    d = _number(_value_of(ctx, _arg(call, 2, "d")))
    std = _number(_value_of(ctx, _arg(call, 3, "measurement_std")))

    if measurements is None:
        measurements = _numeric_array(ctx.get(*MEASUREMENT_VARS), ndim=1)
    if measurements is None:
        measurements = _numeric_array(ctx.env.get("measured_range"), ndim=1)
    if d is None:
        d = _number(ctx.get(*DISTANCE_VARS))
    if d is None:
        d = _number(ctx.env.get("d"))
    if std is None:
        std = _number(ctx.get(*STD_VARS))
    if std is None:
        std = SHIPPED_MEASUREMENT_STD
    return measurements, d, std, states_name


def _tried_settings(ctx):
    """Every radar distance and sensor sigma we can read off the source.

    Only the last call's *variables* can be trusted once the cell has run, so
    earlier calls contribute the numbers typed straight into them — enough to
    notice a student who worked through three radar distances in a row.
    """
    distances, stds = [], []
    for call in _filter_calls(ctx):
        distance = _const(_arg(call, 2, "d"))
        if distance is not None:
            distances.append(distance)
        std = _const(_arg(call, 3, "measurement_std"))
        if std is not None:
            stds.append(std)
    for node in ast.walk(ctx.tree):
        if not isinstance(node, ast.Assign):
            continue
        name = _first_name(node.targets[0])
        value = _const(node.value)
        if name is None or value is None:
            continue
        if name in DISTANCE_VARS:
            distances.append(value)
        elif name in STD_VARS:
            stds.append(value)
    call, _ = _last_filter_call(ctx)
    resolved_d = _number(_value_of(ctx, _arg(call, 2, "d")))
    resolved_std = _number(_value_of(ctx, _arg(call, 3, "measurement_std")))
    if resolved_d is not None:
        distances.append(resolved_d)
    if resolved_std is not None:
        stds.append(resolved_std)
    return distances, stds


def _history_array(value):
    """`value` as a (steps, 2) array, or None.

    Appending `x` rather than `x.flatten()` leaves a stack of (2, 1) columns,
    which is a (steps, 2, 1) array — the same history, one axis fatter — so
    that axis is flattened away before anything else looks at it.
    """
    arr = _numeric_array(value)
    if arr is None:
        return None
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    elif arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    if arr.ndim != 2:
        return None
    if arr.shape[0] == 2 and arr.shape[1] != 2:
        arr = arr.T                          # they stacked it the other way up
    if arr.shape[1] != 2 or arr.shape[0] < 2:
        return None
    return arr


def _state_histories(ctx, preferred_name=None, length=None, reject=()):
    """Every (steps, 2) state history the submission left behind, best first.

    Name order comes first — the ones we suggested, then anything that reads
    like a record of states — because shape alone doesn't separate a state
    history from a stack of measurements: `np.column_stack([ranges, bearings])`
    is (steps, 2) too. But a name nobody guessed is still a state history, so
    anything else comes last rather than not at all — only at exactly the right
    length, though, since an unrecognised name is worth taking seriously only
    when the shape insists, and R is a 2x2 array too. The series we already
    know to be measurements are dropped throughout, and so is an array that was
    only ever copied out of an earlier cell.
    """
    named = [preferred_name] if preferred_name else []
    named += [n for n in STATE_VARS if n not in named]
    named += [n for n in sorted(ctx.assigned)
              if n not in named and any(hint in n.lower() for hint in STATE_HINTS)]
    order = [(n, True) for n in named]
    order += [(n, False) for n in sorted(ctx.assigned) if n not in named]
    exact, loose = [], []
    for name, recognised in order:
        if not name or name not in ctx.assigned or name not in ctx.env:
            continue
        arr = _history_array(ctx.env[name])
        if arr is None:
            continue
        if not recognised and arr.shape[0] != (length or 0):
            continue                         # only the shape vouches for this one
        if any(_same_series(arr[:, column], known)
               for column in range(2) for known in reject):
            continue                         # this is their measurements, not states
        if _copied_from_setup(ctx, name):
            continue                         # this is the cell above's answer
        (exact if length is None or arr.shape[0] == length else loose).append((name, arr))
    return exact + loose


def _passthrough_source(node):
    """The name an assignment is a whole-array copy of, if that's all it is.

    `states_both = states`, `= states.copy()`, `= np.array(states)`, `= states[:]`
    all come back as "states". Anything that does arithmetic, indexes a row, or
    calls a filter does not — those are results, not copies.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Subscript):
        index = node.slice
        if (isinstance(index, ast.Slice) and index.lower is None
                and index.upper is None and index.step is None):
            return _passthrough_source(node.value)
        return None
    if isinstance(node, ast.Call):
        name = _called_name(node)
        if name in ("array", "asarray", "copy", "deepcopy", "vstack") and node.args:
            return _passthrough_source(node.args[0])
        if name == "copy" and isinstance(node.func, ast.Attribute):
            return _passthrough_source(node.func.value)
    return None


def _copied_from_setup(ctx, name):
    """The earlier-cell variable `name` was copied from, if it was.

    Reusing the state history the notebook already computed is the one way to
    produce a perfect-looking answer without running a filter, and it is the
    only thing this catches: every assignment to the name has to be a bare copy
    of something the submission never computed itself.
    """
    sources = []
    for node in ast.walk(ctx.tree):
        if not isinstance(node, ast.Assign):
            continue
        if _first_name(node.targets[0]) != name:
            continue
        origin = _passthrough_source(node.value)
        if origin is None or origin in ctx.assigned:
            return None
        sources.append(origin)
    return sources[-1] if sources else None


def _same_series(a, b):
    """Are these the same 1-D series? Anything else is not comparable."""
    if b is None or np.ndim(b) != 1 or np.ndim(a) != 1 or len(a) != len(b):
        return False
    return bool(np.allclose(a, b, rtol=0, atol=1e-9))


def _state_history(ctx, preferred_name=None, length=None, reject=()):
    """The (steps, 2) state history the submission produced, and its name."""
    found = _state_histories(ctx, preferred_name, length, reject)
    return found[0] if found else (None, None)


def _back_half_rms(estimate, truth):
    """RMS error over the second half of the flight, or None."""
    n = min(len(estimate), len(truth))
    if n < 4:
        return None
    half = n // 2
    error = np.asarray(estimate[half:n], dtype=float) - np.asarray(truth[half:n], dtype=float)
    if not np.all(np.isfinite(error)):
        return None
    return float(np.sqrt(np.mean(error ** 2)))


def _tail_mean(values, fraction=0.1):
    """Mean of the last tenth — "where the estimate ended up"."""
    n = max(1, int(len(values) * fraction))
    tail = np.asarray(values[-n:], dtype=float)
    return float(np.mean(tail)) if np.all(np.isfinite(tail)) else None


def _is_slant_range(call, d):
    """Does this h still return sqrt(d**2 + position**2)?"""
    if call is None:
        return False
    try:
        for position in TEST_POSITIONS:
            want = float(np.sqrt(d ** 2 + position ** 2))
            got = _number(call(position, TEST_VELOCITY))
            if got is None or abs(got - want) > MODEL_RTOL * want:
                return False
    except Exception:
        return False
    return True


def _is_slant_derivative(call, d):
    """Does this H still return [[position / sqrt(d**2 + position**2), 0]]?"""
    if call is None:
        return False
    try:
        for position in TEST_POSITIONS:
            want = np.array([[position / np.sqrt(d ** 2 + position ** 2), 0.0]])
            if not _jacobian_agrees(_as_matrix(call(position, TEST_VELOCITY), 1), want):
                return False
    except Exception:
        return False
    return True


def _retuned(ctx, states, dt):
    """Did the cell actually move the radar or change how much it is trusted?

    Settled by running the shipped setup ourselves and looking at the answer,
    not by reading numbers out of the source: a cell that changed nothing
    produces, to the metre, the plot the worked example above already produced.
    That is the only way to be right about both a loop over three distances
    (where the numbers never appear as an argument) and a new `measurement_std`
    that was assigned but then never passed to anything.

    Regenerating the measurements counts as a change too — a noisier sensor is
    part of the radar setup — and if we can't reproduce the shipped run at all
    we fall back on whatever numbers the source does show.
    """
    shipped = _reference_ekf(ctx.env.get("measured_range"), dt,
                             SHIPPED_D, SHIPPED_MEASUREMENT_STD)
    if states is not None and shipped is not None:
        n = min(len(states), len(shipped))
        if n >= 4:
            moved = float(np.max(np.abs(states[:n, POSITION] - shipped[:n, POSITION])))
            return moved > SETTINGS_EFFECT_TOL or bool(ctx.defined(*MEASUREMENT_VARS))
    distances, stds = _tried_settings(ctx)
    return (any(abs(v - SHIPPED_D) > 1e-9 for v in distances)
            or any(abs(v - SHIPPED_MEASUREMENT_STD) > 1e-9 for v in stds))


# ---------------------------------------------------------------------------
# Exercise 1 — move the radar, turn the noise up
# ---------------------------------------------------------------------------

def check_ex1(ctx):
    """"Try a closer radar, a farther radar, and a bigger measurement_std."

    The tuning is the student's, so the checks split in two. The filter itself
    has to still be the filter — h is still the slant range, H is still its
    derivative, one state comes out per measurement. The *results* are only
    ever judged against what our own EKF manages on their settings, because
    watching the estimate fall apart when you stop trusting the sensor is the
    whole point of the exercise, not a failure.
    """
    if not _ran(ctx):
        return

    truth = _true_position(ctx)
    dt = _number(ctx.env.get("dt")) or DEFAULT_DT
    measurements, d, std, states_name = _settings(ctx)

    h_name = _model_name(ctx, EX1_H_NAMES)
    j_name = _model_name(ctx, EX1_JACOBIAN_NAMES)
    f_name = _model_name(ctx, EX1_FILTER_NAMES)
    models = h_name and j_name and f_name
    if not ctx.require("The cell runs the EKF over a set of measurements",
                       models and _filter_calls(ctx),
                       "This cell is the worked example again, yours to poke at: change "
                       "the radar setup and call `extended_kalman_filter` on it. Trimming "
                       "the copies of `h` and `H_jacobian` out and using the ones defined "
                       "above is fine — running nothing, or reading the answer off an "
                       "earlier cell, is not.",
                       expected="a call to extended_kalman_filter(...)",
                       got=("no call to the filter in this cell" if models
                            else "h, H_jacobian or the filter is missing")):
        return

    probe_d = d if d else SHIPPED_D
    call_h = _adapter(ctx.env[h_name], probe_d, lambda v: _number(v) is not None)
    call_j = _adapter(ctx.env[j_name], probe_d, lambda v: _as_matrix(v, 1) is not None)

    ctx.require("`h(x, d)` still returns the slant range",
                _is_slant_range(call_h, probe_d),
                "h maps a state to what the radar should read, and that is the slant "
                "range sqrt(d**2 + position**2) — the altitude on its own is what the "
                "plain Kalman filter got to use.",
                expected=f"h([[{TEST_POSITIONS[1]:g}], [450]], {probe_d:g}) = "
                         f"{np.sqrt(probe_d ** 2 + TEST_POSITIONS[1] ** 2):.3f}",
                got=_describe(call_h, TEST_POSITIONS[1]))

    ctx.require("The Jacobian is still dh/dx",
                _is_slant_derivative(call_j, probe_d),
                "H has to be the derivative of h at the predicted state: "
                "d/dposition of sqrt(d**2 + position**2) is position/sqrt(d**2 + "
                "position**2), and h doesn't depend on velocity at all, so the second "
                "entry is 0. Get this wrong and the filter still runs — it just "
                "corrects by the wrong amount every step.",
                expected=f"[[{TEST_POSITIONS[1] / np.sqrt(probe_d ** 2 + TEST_POSITIONS[1] ** 2):.6f}, 0]]"
                         f" at position {TEST_POSITIONS[1]:g}",
                got=_describe(call_j, TEST_POSITIONS[1]))

    # --- did they actually change anything? --------------------------------
    _, states = _state_history(
        ctx, states_name, len(measurements) if measurements is not None else None,
        reject=[measurements] if measurements is not None else ())
    ctx.require("The radar setup is not the one the cell came with",
                _retuned(ctx, states, dt),
                "This is still the worked example above, number for number, so the "
                "plots come out identical. Bring the radar in (a smaller d), push it "
                "out (a bigger d), and raise `measurement_std` — then compare the "
                "three plots each time.",
                expected="a different d, or a different measurement_std",
                got=f"d = {SHIPPED_D:g}, measurement_std = {SHIPPED_MEASUREMENT_STD:g}")

    # --- the measurements have to belong to that radar ---------------------
    if measurements is not None and d is not None and len(measurements) >= 4:
        n = min(len(measurements), len(truth))
        difference = measurements[:n] - np.sqrt(d ** 2 + truth[:n] ** 2)
        mismatch = float(np.mean(np.abs(difference)))
        allowed = max(RANGE_MATCH_TOL, RANGE_MATCH_SIGMAS * float(np.std(difference)))
        ctx.require("The measurements match the radar you filtered with",
                    mismatch <= allowed,
                    f"The filter is being told the radar sits {d:g} m from the pad, but "
                    "the ranges you fed it were generated for a station somewhere else. "
                    "h(x) and z then describe two different geometries, and the filter "
                    "will happily converge on a rocket that never flew. Generate the "
                    "measurements from the same d you filter with:\n"
                    "    measured_range = np.sqrt(d**2 + true_position**2) "
                    f"+ np.random.normal(0, {SHIPPED_RANGE_NOISE:g}, len(time))",
                    expected=f"ranges near {np.sqrt(d ** 2 + truth[0] ** 2):.0f} m at t = 0",
                    got=f"{measurements[0]:.0f} m at t = 0 "
                        f"(off by {mismatch:.0f} m on average)")

    # --- and the filter has to have worked ---------------------------------
    if not ctx.require("The filter returns one state per measurement",
                       states is not None and measurements is not None
                       and states.shape == (len(measurements), 2),
                       "`extended_kalman_filter` appends one [position, velocity] row "
                       "per measurement, so the state history should be as long as the "
                       "series you handed it. Keep its result — the plots below read "
                       "`states`.",
                       expected=(f"({len(measurements)}, 2)"
                                 if measurements is not None else "(steps, 2)"),
                       got=str(states.shape) if states is not None else "no state history"):
        return

    error = _back_half_rms(states[:, POSITION], truth)
    expected_states = _reference_ekf(measurements, dt, d, std)
    expected_error = (_back_half_rms(expected_states[:, POSITION], truth)
                      if expected_states is not None else None)
    allowed = POSITION_RMS_FLOOR
    if expected_error is not None:
        allowed = max(allowed, TRACKING_SLACK * expected_error)
    ctx.require("The position estimate converges on the rocket",
                error is not None and IMPOSSIBLE_RMS < error <= allowed,
                _tracking_message(error),
                expected=f"back-half RMS error between {IMPOSSIBLE_RMS:g} m "
                         f"and {allowed:.0f} m",
                got="no usable estimate" if error is None else f"{error:.3f} m")

    velocity = _tail_mean(states[:, VELOCITY])
    band = VELOCITY_BAND
    if expected_states is not None:
        reference_velocity = _tail_mean(expected_states[:, VELOCITY])
        if reference_velocity is not None:
            band = max(band, TRACKING_SLACK * abs(reference_velocity - TRUE_VELOCITY))
    ctx.require(f"The velocity estimate settles near {TRUE_VELOCITY:g} m/s",
                velocity is not None and abs(velocity - TRUE_VELOCITY) <= band,
                "The rocket climbs at a constant 450 m/s, and the filter should end up "
                "saying so even though nothing ever measures velocity — it gets there "
                "from the way position and velocity are coupled in P. Ending somewhere "
                "else means the update is fighting the model rather than feeding it.",
                expected=f"{TRUE_VELOCITY:g} +/- {band:.0f} m/s",
                got="no usable estimate" if velocity is None else f"{velocity:.1f} m/s")


def _tracking_message(error):
    """Two very different failures share this row, so they get two messages."""
    if error is not None and error <= IMPOSSIBLE_RMS:
        return ("This estimate is `true_position` itself, to the millimetre. The filter "
                "only ever sees noisy slant ranges, so it cannot reproduce the truth "
                "exactly — whatever is in your state history did not come out of the "
                "filter. Plot what `extended_kalman_filter` actually returned.")
    return ("Over the second half of the flight the estimate should have settled onto "
            "the true altitude, and this one hasn't — by more than your choice of d "
            "and measurement_std can explain. Check that the update step still uses "
            "h(x_pred, d) and H_jacobian(x_pred, d), and that the measurements you "
            "passed came from the same radar.")


def _describe(call, position):
    """What a model returned at one test point, for a feedback row."""
    if call is None:
        return "could not be called with a state"
    try:
        value = call(position, TEST_VELOCITY)
    except Exception as exc:
        return f"raised {type(exc).__name__} at position {position:g}"
    arr = _numeric_array(value)
    return f"{np.array2string(arr, precision=6)} at position {position:g}" \
        if arr is not None else f"{value!r} at position {position:g}"


# ---------------------------------------------------------------------------
# Exercise 2 — add a bearing measurement  (#% checker: bearing_measurement)
# ---------------------------------------------------------------------------

def check_bearing_measurement(ctx):
    """"Extend z to [range, bearing], extend h and H, re-run, compare."

    The measurement model is now the interesting part, so most of these checks
    are about it: two outputs, the right two outputs, and a Jacobian that is
    genuinely the derivative of whatever h they wrote — differentiated by hand
    here rather than pattern-matched, which is the only way to check a Jacobian
    that doesn't care how it was written.

    We look for their functions by name first and by behaviour second, and if
    we still can't find them we grade the arrays they produced instead of
    failing the whole cell: a correct filter written in an unusual shape should
    still get most of the way.
    """
    if not _ran(ctx):
        return

    truth = _true_position(ctx)
    dt = _number(ctx.env.get("dt")) or DEFAULT_DT
    d = _number(ctx.env.get("d")) or SHIPPED_D
    measured_range = _numeric_array(ctx.env.get("measured_range"), ndim=1)

    # --- the new measurement -----------------------------------------------
    ctx.require("The bearing comes out of an arctangent",
                any(isinstance(node, ast.Call) and _called_name(node) in ARCTAN_NAMES
                    for node in ast.walk(ctx.tree)),
                "The bearing to the rocket is theta = arctan(h / d) — an angle from a "
                "ratio of two lengths. Build the measurements with `np.arctan` (or "
                "`np.arctan2(true_position, d)`, which behaves better once the rocket "
                "is overhead).")

    bearing_name, bearings = _bearing_series(ctx, truth, d)
    ctx.require("The bearing measurements track arctan(h / d)", bearing_name,
                "I couldn't find a series in this cell that looks like the bearing over "
                "time. Compute it from the trajectory you already have and keep it in "
                "an array of its own:\n"
                "    true_bearing = np.arctan(true_position / d)\n"
                "    measured_bearing = true_bearing + np.random.normal(0, 0.001, len(time))",
                expected=f"{len(truth)} angles running 0 -> "
                         f"{np.arctan2(truth[-1], d):.3f} rad",
                got="nothing in this cell matches")

    # --- the model and its Jacobian ----------------------------------------
    h_name, call_h = _find_model(ctx, EX2_H_NAMES, lambda v: _pair(v) is not None, d,
                                 prefer=lambda call: _matches_range_bearing(call, d))
    ctx.require("The measurement function returns two numbers", call_h,
                "h(x) has to predict everything the radar reports, so it now returns "
                "a 2-vector — np.array([[range], [bearing]]) — instead of a single "
                "range. Give it a name of its own (`h_rb`) so the range-only filter "
                "above keeps working.",
                expected="h_rb([[3000], [450]], d) -> two numbers",
                got=_describe(call_h, TEST_POSITIONS[1]) if call_h
                    else "no two-output measurement function found")

    ctx.require("Those two numbers are the range and the bearing",
                call_h is not None and _matches_range_bearing(call_h, d),
                "One row should be the slant range sqrt(d**2 + position**2), the other "
                "the bearing arctan(position / d) — the two things this radar actually "
                "reports, in the same order as your measurement vector z.",
                expected=f"[{np.sqrt(d ** 2 + TEST_POSITIONS[1] ** 2):.1f}, "
                         f"{np.arctan2(TEST_POSITIONS[1], d):.5f}] "
                         f"at position {TEST_POSITIONS[1]:g}",
                got=_describe(call_h, TEST_POSITIONS[1]))

    _, call_j = _find_model(ctx, EX2_JACOBIAN_NAMES,
                            lambda v: _as_matrix(v, 2) is not None, d,
                            prefer=lambda call: _jacobian_is_derivative(call, call_h, d),
                            hints=("jac", "deriv", "grad"),
                            exclude=(h_name,) if h_name else ())
    rows = _jacobian_rows(call_j)
    ctx.require("The Jacobian has one row per measurement", rows == 2,
                "H is the derivative of h, so it grows a row when h does: two "
                "measurements over two state variables makes it 2x2. The velocity "
                "column stays zero — neither the range nor the bearing depends on how "
                "fast the rocket is going.",
                expected="a 2x2 matrix",
                got=f"{rows} row(s)" if rows else "no 2-row Jacobian found")

    ctx.require("The Jacobian is the derivative of your own h(x)",
                _jacobian_is_derivative(call_j, call_h, d),
                "I differentiated your h numerically and compared it against your H, "
                "and they disagree. dr/dposition is position/sqrt(d**2 + position**2); "
                "dtheta/dposition is d/(d**2 + position**2) — a much smaller number, "
                "which is the point: it tells the filter how much a metre of altitude "
                "moves each reading. Both velocity entries are 0.",
                expected="H(x) = dh/dx at the same state",
                got=_jacobian_mismatch(call_j, call_h, d))

    ctx.require("The measurement noise `R` is 2x2", _r_is_2x2(ctx),
                "R holds one variance per measurement, so it is 2x2 now: "
                "np.array([[range_std**2, 0], [0, bearing_std**2]]). Mind the units — "
                "metres squared against radians squared, numbers that are nowhere near "
                "each other. Keep it called `R` so this check can find it.")

    # --- the run, and whether it helped ------------------------------------
    ctx.require("The filter loops over the measurements",
                ctx.uses("for") or ctx.uses("while"),
                "The EKF has to walk the measurements one at a time, predicting and "
                "then correcting — there is no way to do a filter in one shot.")

    baseline = _reference_ekf(measured_range, dt, d, BASELINE_RANGE_STD)
    baseline_error = (_back_half_rms(baseline[:, POSITION], truth)
                      if baseline is not None else None)
    known = [series for series in (measured_range, bearings) if series is not None]
    histories = _state_histories(ctx, None, len(truth), known)
    histories += _position_histories(ctx, len(truth), known + [truth])
    _, states = _pick_history(histories, baseline)
    if not ctx.require("It produces one filtered state per measurement",
                       states is not None and abs(len(states) - len(truth)) <= 1,
                       _history_message(ctx),
                       expected=f"{len(truth)} states",
                       got=f"{len(states)}" if states is not None
                           else "no state history found"):
        return

    error = _back_half_rms(states[:, POSITION], truth)
    allowed = COMPARISON_FLOOR
    if baseline_error is not None:
        allowed = max(allowed, COMPARISON_SLACK * baseline_error)
    ctx.require("The estimate is at least as good as range-only",
                error is not None and IMPOSSIBLE_RMS < error <= allowed,
                _comparison_message(error),
                expected=f"back-half RMS error under {allowed:.0f} m"
                         + (f" (range only manages {baseline_error:.0f} m)"
                            if baseline_error is not None else ""),
                got="no usable estimate" if error is None else f"{error:.3f} m")

    ctx.require("The bearing actually reaches the update",
                _bearing_changed_anything(states, baseline),
                "Your range+bearing run comes out identical to the range-only one, to "
                "the metre — so the bearing is being generated and then never used. In "
                "the loop, z, y = z - h(x_pred), H and R all have to be the two-row "
                "versions; if any one of them is still the range-only shape, the second "
                "measurement quietly falls out of the arithmetic.",
                expected="an estimate the bearing has moved",
                got="the same numbers the range-only filter produces")

    ctx.require("Both position estimates are plotted together",
                _plot_calls(ctx) >= 2,
                "Draw the range-only and range+bearing estimates on the same axes "
                "against `true_position`. The interesting part is the first second or "
                "two, where the range alone barely notices the rocket climbing.",
                expected="at least two series on a plot",
                got=f"{_plot_calls(ctx)} plotted")


def _history_message(ctx):
    """Missing because they never ran a filter, or because they reused one?"""
    for name in sorted(ctx.assigned):
        origin = _copied_from_setup(ctx, name)
        if origin and _history_array(ctx.env.get(origin)) is not None:
            return (f"`{name}` is `{origin}` from the cell above, copied across — that "
                    "is the range-only run you are supposed to be comparing against, "
                    "not a new one. Run your two-measurement filter over the bearings "
                    "you just built and keep what it returns.")
    return ("Collect [position, velocity] once per timestep and keep the history — "
            "you need it both to plot and to compare against the range-only run.")


def _comparison_message(error):
    if error is not None and error <= IMPOSSIBLE_RMS:
        return ("This estimate is `true_position` itself, to the millimetre. Every "
                "measurement the filter gets is noisy, so no amount of fusing gets you "
                "the truth exactly — whatever is in your state history did not come out "
                "of a filter.")
    return ("Adding a measurement cannot make a Kalman filter worse, so if it has, "
            "something is telling the filter to trust the wrong thing. The usual "
            "culprit is R: a bearing variance written in degrees, or copied from the "
            "range, makes the filter chase an angle it should be ignoring. Check the "
            "units, and check that h, H and z put range and bearing in the same order.")


def _bearing_changed_anything(states, baseline):
    """Did fusing the bearing move the answer at all?

    Only decisive when their range-only settings match the yardstick run we
    built; when they don't, the two differ for that reason alone and this
    passes. It exists for one specific submission: the one that builds a
    beautiful two-row h and then filters the range on its own.
    """
    if states is None or baseline is None:
        return True
    n = min(len(states), len(baseline))
    if n < 4:
        return True
    return bool(np.max(np.abs(states[:n, POSITION] - baseline[:n, POSITION]))
                > BEARING_EFFECT_TOL)


def _position_histories(ctx, length, reject):
    """Runs kept as a bare position series, for when no (steps, 2) one exists.

    The prompt asks for the state history, but a student who kept only the
    altitudes has still filtered — everything ex2 asks of the history is asked
    of its position column. Only names that say so count: every cell is full of
    1-D arrays this long, and the residuals are one of them. The velocity
    column is filled with NaN rather than invented, and these come last, after
    every real state history.
    """
    found = []
    for name in sorted(ctx.assigned):
        if name not in ctx.env or name.lower().startswith("true"):
            continue
        if not any(hint in name.lower() for hint in POSITION_HINTS):
            continue
        series = _numeric_array(ctx.env[name], ndim=1)
        if series is None or series.shape[0] != length:
            continue
        if any(_same_series(series, known) for known in reject):
            continue                         # the measurements, or the truth itself
        if _copied_from_setup(ctx, name):
            continue
        found.append((name, np.column_stack([series, np.full(length, np.nan)])))
    return found


def _pick_history(histories, baseline):
    """The run that used the bearing, out of the histories they kept.

    Most students keep both runs so they can plot them together, and the
    range-only one is easy to recognise: it is the one our own range-only
    filter reproduces almost exactly. Grading that one would let a broken
    range+bearing filter through, so it goes to the back of the queue.
    """
    if not histories:
        return None, None
    if baseline is None:
        return histories[0]
    fresh = []
    for name, states in histories:
        n = min(len(states), len(baseline))
        if n > 4 and np.max(np.abs(states[:n, POSITION]
                                   - baseline[:n, POSITION])) < BEARING_EFFECT_TOL:
            continue                         # this one *is* the range-only run
        fresh.append((name, states))
    return fresh[0] if fresh else histories[0]


def _bearing_series(ctx, truth, d):
    """(name, series) of an array in the submission shaped like arctan(h / d).

    Recognised by shape rather than exactness: how much noise they put on it —
    if any — is their call, and the prompt doesn't ask for a particular amount.
    Degrees are accepted as readily as radians; a student who works in degrees
    just writes their R and their Jacobian to match.

    The series comes back alongside the name because the name may well be a
    two-column `measured_z`, and everything downstream wants the one column
    that is actually the bearing.
    """
    expected = np.arctan2(truth, d)
    for name in sorted(ctx.assigned):
        if name not in ctx.env:
            continue
        arr = _numeric_array(ctx.env[name])
        if arr is None or arr.ndim > 2:
            continue
        for series in _series_of(arr, len(expected)):
            for scale in (1.0, np.pi / 180.0):
                if np.mean(np.abs(series * scale - expected)) <= BEARING_MATCH_TOL:
                    return name, np.asarray(series, dtype=float)
    return None, None


def _series_of(arr, length):
    """Every 1-D series of `length` hiding in an array, columns included."""
    if arr.ndim == 1 and arr.shape[0] == length:
        yield arr
    elif arr.ndim == 2:
        if arr.shape[0] == length and arr.shape[1] <= 4:
            for column in range(arr.shape[1]):
                yield arr[:, column]
        if arr.shape[1] == length and arr.shape[0] <= 4:
            for row in range(arr.shape[0]):
                yield arr[row, :]


def _jacobian_rows(call):
    """How many rows the student's Jacobian actually has."""
    if call is None:
        return 0
    try:
        arr = _numeric_array(call(TEST_POSITIONS[1], TEST_VELOCITY))
    except Exception:
        return 0
    if arr is None:
        return 0
    return int(arr.shape[0]) if arr.ndim == 2 else 1


def _reference_models(call_h, d):
    """Models to differentiate against: theirs first, then the textbook one.

    If we couldn't find their h at all we still want to say something useful
    about their H, so we fall back on the canonical (range, bearing) model in
    both row orders rather than reporting nothing.
    """
    if call_h is not None:
        return [call_h]

    def canonical(position, velocity, swapped=False):
        slant = np.sqrt(d ** 2 + position ** 2)
        bearing = np.arctan2(position, d)
        return np.array([bearing, slant]) if swapped else np.array([slant, bearing])

    return [canonical, lambda p, v: canonical(p, v, swapped=True)]


def _jacobian_is_derivative(call_j, call_h, d):
    if call_j is None:
        return False
    for model in _reference_models(call_h, d):
        if all(_agrees_at(call_j, model, px) for px in TEST_POSITIONS):
            return True
    return False


def _agrees_at(call_j, model, position):
    try:
        theirs = _as_matrix(call_j(position, TEST_VELOCITY), 2)
        expected = _fd_jacobian(model, position, TEST_VELOCITY)
    except Exception:
        return False
    return _jacobian_agrees(theirs, expected)


def _jacobian_mismatch(call_j, call_h, d):
    """One concrete disagreement, for the feedback row."""
    if call_j is None:
        return "no 2-row Jacobian found"
    position = TEST_POSITIONS[1]
    try:
        theirs = _as_matrix(call_j(position, TEST_VELOCITY), 2)
    except Exception as exc:
        return f"raised {type(exc).__name__} at position {position:g}"
    if theirs is None:
        return _describe(call_j, position)
    model = _reference_models(call_h, d)[0]
    try:
        expected = _fd_jacobian(model, position, TEST_VELOCITY)
    except Exception:
        expected = None
    if expected is None:
        return f"{np.array2string(theirs, precision=8)} at position {position:g}"
    return (f"{np.array2string(theirs, precision=8)}, but differentiating your h gives "
            f"{np.array2string(expected, precision=8)} at position {position:g}")


def _r_is_2x2(ctx):
    """Is the measurement-noise matrix 2x2?

    R usually lives inside the filter function, so it never reaches `ctx.env`.
    Three ways of finding out, in order of how much they can be trusted: the
    value itself if it is a module-level variable; the expression it was built
    from, evaluated in the namespace they left behind; and failing both — the
    pieces are function arguments, say — the shape of the expression read
    straight off the source. Anything that builds a 2x2 counts: a literal with
    two rows of two, np.eye(2), np.zeros((2, 2)), np.diag of a pair, and those
    scaled or added.
    """
    for name in R_VARS:
        if name in ctx.assigned and name in ctx.env:
            arr = _numeric_array(ctx.env[name], ndim=2)
            if arr is not None and arr.shape == (2, 2):
                return True
    for node in ast.walk(ctx.tree):
        if not isinstance(node, ast.Assign):
            continue
        if _first_name(node.targets[0]) not in R_VARS:
            continue
        arr = _numeric_array(_value_of(ctx, node.value), ndim=2)
        if arr is not None and arr.shape == (2, 2):
            return True
        if _matrix_shape(node.value) == (2, 2):
            return True
    return False


def _matrix_shape(node):
    """(rows, columns) of a matrix expression written in the source, or None."""
    if isinstance(node, ast.UnaryOp):
        return _matrix_shape(node.operand)
    if isinstance(node, ast.BinOp):
        return _matrix_shape(node.left) or _matrix_shape(node.right)
    if isinstance(node, (ast.List, ast.Tuple)):
        widths = {_sequence_length(row) for row in node.elts}
        return (len(node.elts), widths.pop()) if len(widths) == 1 and None not in widths \
            else None
    if not isinstance(node, ast.Call) or not node.args:
        return None
    name = _called_name(node)
    if name in ("eye", "identity"):
        size = _const(node.args[0])
        return (int(size), int(size)) if size else None
    if name == "diag":
        size = _sequence_length(node.args[0])
        return (size, size) if size else None
    if name in ("zeros", "ones", "empty", "full"):
        shape = _sequence_length(node.args[0], as_shape=True)
        return shape if shape and len(shape) == 2 else None
    if name in ("array", "asarray", "matrix"):
        return _matrix_shape(node.args[0])
    return None


def _sequence_length(node, as_shape=False):
    """How many items a sequence expression holds, seeing through wrappers.

    `[a, b]`, `np.array([a, b])` and `np.array([a, b]) ** 2` are all two long;
    with `as_shape` a tuple of literals comes back as the tuple itself, which
    is how `np.zeros((2, 2))` gets read.
    """
    if isinstance(node, ast.UnaryOp):
        return _sequence_length(node.operand, as_shape)
    if isinstance(node, ast.BinOp):
        return (_sequence_length(node.left, as_shape)
                or _sequence_length(node.right, as_shape))
    if isinstance(node, (ast.List, ast.Tuple)):
        if as_shape:
            sizes = [_const(element) for element in node.elts]
            return tuple(int(s) for s in sizes) if all(s for s in sizes) else None
        return len(node.elts)
    if isinstance(node, ast.Call) and node.args and _called_name(node) in (
            "array", "asarray", "matrix", "square"):
        return _sequence_length(node.args[0], as_shape)
    return None


def _plot_calls(ctx):
    """How many series the submission draws (`plt.plot`, `ax1.scatter`, ...)."""
    return sum(1 for node in ast.walk(ctx.tree)
               if isinstance(node, ast.Call) and _called_name(node) in PLOT_CALLS)
