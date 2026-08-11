"""Behaviour checkers for basic_filters.ipynb.

Every exercise in this notebook has the same shape: turn one knob, re-run a
filter the notebook already wrote for you, and look at what changed. That is
also what makes them easy to fake — leave the shipped number where it is, or
hard-code a smooth-looking array, and the picture still looks plausible. So
each checker here re-runs the filter itself, from the student's own gain, and
compares the numbers. The three filters are re-implemented in this file rather
than called back out of `ctx.env`, because a submission that redefines
`moving_average` would otherwise be graded against its own redefinition.

The knob *is* the exercise in the first three cells ("Change the window_size
below"), so "this isn't the value the cell shipped with" is a graded row. Two of
the shipped values are also one of the suggestions the prompt lists, which is a
horrible thing to be told you got wrong without explanation — those messages say
so, and name another value to try.

Everything else stays lenient. Any recognisable variable name is accepted, print
formatting is ignored, and the final challenge finds the student's three
filtered signals *by their values*, under whatever names they chose. Nothing
here grades estimate quality: exercise 3's prompt says outright that you do not
need perfect gains, so a deliberately awful gamma passes. What is graded is that
the filter genuinely ran again with the student's own numbers, and that the
result still reaches the plot. A filter written out by hand counts everywhere
the notebook's helper does: a moving average may be the trailing window or the
centred convolution, and a low-pass may be seeded with the first measurement or
with zero.

Two things make finding "the student's own numbers" harder than it looks, and
`_own` and `_agreeing_pair` are the answers to them. First, earlier cells are
replayed as setup, so the previous exercise's `filtered` is sitting in the
namespace — and a helper function with a local variable of the same name is
enough to make `ctx.assigned` claim the cell wrote it. Only *module-level*
bindings count here, for that reason. Second, several names can plausibly be the
knob (the notebook's own moving average keeps its slice in a variable called
`window`), so instead of trusting the first one we look for the gain and the
signal that agree with each other — whichever gain, re-run through the filter,
reproduces whichever signal the cell left behind. That is the pair the student
meant.
"""

import ast
import math

import numpy as np


# --- what the cells ship with --------------------------------------------
# An unchanged value means the exercise wasn't done. Keep these in step with
# the notebook if the starting values there ever change.
EX1_DEFAULT_WINDOW = 3
EX2_DEFAULT_ALPHA = 0.2
EX3_DEFAULT_GAINS = (0.3, 0.05, 0.001)

# --- what the final challenge asks for -----------------------------------
CHALLENGE_WINDOW = 5
CHALLENGE_ALPHA = 0.2
MIN_PLOTTED_SERIES = 3  # three filtered signals, the measurements make four

DT = 0.1  # the data's sample spacing — the student's gains are the variable, not this

# How close an array has to be to count as "this came out of that filter with
# those gains". Loose enough that `x += alpha * (z - x)` and
# `alpha * z + (1 - alpha) * x` — the same recursion written two ways, and they
# differ in the last couple of bits — agree; tight enough that a different gain
# never does. Widen if a legitimate implementation starts falling outside.
MATCH_RTOL = 1e-6
MATCH_ATOL = 1e-6
CHANGED_EPS = 1e-9  # closer than this to the shipped value and it *is* the shipped value

# Gains accepted without complaint. Deliberately wider than the usual 0..1: the
# exercise 3 prompt asks what happens when a gain is *too large*, and failing a
# student for answering the question it asked would be perverse.
GAIN_RANGE = (0.0, 2.0)

# Accepted variable names. `_own` only matches names the submission itself binds
# at module level, so the replayed setup cells above can't produce a false
# positive and short shorthands are safe. The name the cell ships with leads.
WINDOW_VARS = ("number_of_values_avg", "window_size", "window", "windowsize",
               "window_length", "num_values", "number_of_values", "n_values",
               "avg_window", "samples", "num_samples", "n", "N", "k")
ALPHA_VARS = ("alpha", "ALPHA", "alpha_value", "alpha_gain", "a", "A", "gain")
FILTERED_VARS = ("filtered", "filtered_altitude", "filtered_data", "filtered_signal",
                 "smoothed", "smoothed_altitude", "moving_average_altitude",
                 "low_pass_altitude", "averaged", "avg_altitude", "estimate",
                 "result", "output", "y")

BETA_VARS = ("BETA", "beta", "beta_value", "beta_gain", "b", "B")
GAMMA_VARS = ("GAMMA", "gamma", "gamma_value", "gamma_gain", "g", "G")
POSITION_VARS = ("position_estimate", "positions", "position_estimates", "position",
                 "pos_estimate", "x_estimate", "x_est", "pos", "x", "p")
VELOCITY_VARS = ("velocity_estimate", "velocities", "velocity_estimates", "velocity",
                 "vel_estimate", "v_estimate", "v_est", "vel", "v")
ACCEL_VARS = ("acceleration_estimate", "accelerations", "acceleration_estimates",
              "acceleration", "accel_estimate", "a_estimate", "a_est", "accel", "acc",
              "a")

# Data the notebook hands the student — never one of their filtered signals.
GIVEN_SIGNALS = ("time", "true_altitude", "measured_altitude", "noise",
                 "measured_acceleration", "true_accel")

# Anything that draws a series. Attribute calls (`plt.plot`, `ax1.scatter`) and
# bare ones (`from matplotlib.pyplot import plot`) both count.
PLOT_FUNCTIONS = ("plot", "scatter", "step", "stairs", "errorbar", "fill_between",
                  "semilogx", "semilogy", "loglog", "bar", "plot_date")

# Filters called inline inside a plot call, e.g. plt.plot(t, moving_average(z, 5)).
FILTER_FUNCTIONS = ("moving_average", "low_pass_filter", "alpha_beta_gamma_filter")


# ---------------------------------------------------------------------------
# Reading values back out of a submission, defensively. Anything at all can be
# sitting in the namespace under the name we asked for, so every one of these
# returns None rather than raising — a checker that crashes replaces the whole
# feedback list with one "tell the GNC lead" row.
# ---------------------------------------------------------------------------

def _ran(ctx, hint="Fix the error shown in the output above, then check again."):
    """Every exercise starts here: nothing else is meaningful if it crashed."""
    return ctx.require("Runs without an error", ctx.ok, hint)


def _bound_by(node, names):
    """Collect the names bound at this level, not counting a `def`'s insides."""
    def bind(target):
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                bind(element)

    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(child.name)
            # ...except what it declares `global`, which is the cell's after all.
            for inner in ast.walk(child):
                if isinstance(inner, ast.Global):
                    names.update(inner.names)
            continue
        if isinstance(child, ast.Lambda):
            continue
        if isinstance(child, ast.Assign):
            for target in child.targets:
                bind(target)
        elif isinstance(child, (ast.AugAssign, ast.AnnAssign, ast.For, ast.AsyncFor,
                                ast.NamedExpr)):
            bind(child.target)
        _bound_by(child, names)


def _own(ctx):
    """Names this cell itself put in the namespace, at module level.

    `ctx.assigned` is nearly this, but it also counts names bound *inside* a
    `def` — and the earlier cells are replayed as setup, so a student who writes
    their own filter with a local list called `filtered` would otherwise be
    graded against the previous exercise's answer sitting in `ctx.env` under
    that name. Only bindings that really are this cell's count.
    """
    names = set()
    _bound_by(ctx.tree, names)
    return {name for name in names if name in ctx.env}


def _defined_as(ctx, names, kind):
    """First of `names` bound to something `kind` accepts; else the first bound.

    Falling back matters: when nothing of the right type is there, the row should
    still name the variable the student actually wrote, so the feedback can show
    what was in it.
    """
    candidates = [name for name in names if name in _own(ctx)]
    for name in candidates:
        if kind(ctx.env.get(name)) is not None:
            return name
    return candidates[0] if candidates else None


def _number(value):
    """`value` as a plain finite float, or None if it isn't one."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _whole(value):
    """`value` as a positive whole number of samples, or None."""
    number = _number(value)
    if number is None or number < 1 or not float(number).is_integer():
        return None
    return int(number)


def _array(value):
    """`value` as a 1-D float array, or None if it isn't a signal at all."""
    if value is None or isinstance(value, (str, bytes, dict, set)):
        return None
    try:
        array = np.asarray(value, dtype=float)
    except Exception:
        return None  # ragged lists, objects, functions, figures...
    if array.ndim != 1 or array.size == 0:
        return None
    return array


def _changed(value, default):
    return value is not None and abs(value - default) > CHANGED_EPS


def _short(value, limit=60):
    """A value as feedback-sized text — a 150-point array is not a useful `got`."""
    try:
        text = repr(value)
    except Exception:
        text = "<unprintable>"
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _matches(actual, expected):
    """Is `actual` the array `expected`, to within the match tolerance?"""
    if actual is None or expected is None or actual.shape != expected.shape:
        return False
    try:
        return bool(np.allclose(actual, expected, rtol=MATCH_RTOL, atol=MATCH_ATOL,
                                equal_nan=True))
    except Exception:
        return False


def _unusable_data(source):
    """Is this "data" something a filter can't be judged against?

    Missing, flat, or full of NaN. It matters because a constant run of numbers
    is a fixed point of every filter in this notebook — and NaNs match NaNs under
    `equal_nan` — so a cell that overwrote the measurements with one would sail
    through the comparison below without having filtered anything. The notebook's
    own data is never like this, so the row only ever appears for a cell that
    replaced it (or deleted it).
    """
    if source is None:
        return True
    try:
        return not bool(np.isfinite(source).all()) or float(np.ptp(source)) == 0.0
    except Exception:
        return True


def _data_check(ctx, name, source):
    """Fail one clear row, rather than grade a submission against nothing."""
    return ctx.require(f"`{name}` still holds the measurements",
                       not _unusable_data(source),
                       "That is the noisy sensor data the notebook made for you "
                       "further up — filter it rather than replacing it. If it isn't "
                       "there at all, run the notebook's cells from the top again.",
                       got=_short(source))


def _rms(estimate, truth):
    """Root-mean-square error of `estimate` against `truth`, or None."""
    if estimate is None or truth is None or estimate.shape != truth.shape:
        return None
    try:
        error = float(np.sqrt(np.mean((estimate - truth) ** 2)))
    except Exception:
        return None
    return None if math.isnan(error) or math.isinf(error) else error


# ---------------------------------------------------------------------------
# The three filters, re-implemented exactly as the notebook defines them, so a
# submission is compared against what its own gain *should* have produced.
# ---------------------------------------------------------------------------

def _moving_average(data, window):
    """The notebook's moving average: a trailing window that grows to size."""
    out = []
    for i in range(len(data)):
        out.append(np.mean(data[max(0, i - window + 1):i + 1]))
    return np.array(out)


def _low_pass(data, alpha, start=None):
    """The notebook's low-pass filter.

    This is also the alpha filter: `x = x + alpha * (z - x)` rearranges to
    `x = alpha * z + (1 - alpha) * x`, so the challenge's two "different"
    filters are one recursion, and one implementation grades both.
    """
    value = data[0] if start is None else start
    out = []
    for measurement in data:
        value = alpha * measurement + (1 - alpha) * value
        out.append(value)
    return np.array(out)


def _alpha_beta_gamma(measurements, dt, alpha, beta, gamma):
    """The notebook's alpha-beta-gamma filter: predict, residual, correct."""
    positions, velocities, accelerations = [], [], []
    position, velocity = 0, 0
    acceleration = measurements[0]
    for z in measurements:
        predicted_position = position + velocity * dt + 0.5 * acceleration * dt ** 2
        predicted_velocity = velocity + acceleration * dt
        predicted_acceleration = acceleration
        residual = z - predicted_acceleration
        position = predicted_position + alpha * (0.5 * dt ** 2) * residual
        velocity = predicted_velocity + beta * dt * residual
        acceleration = predicted_acceleration + gamma * residual
        positions.append(position)
        velocities.append(velocity)
        accelerations.append(acceleration)
    return np.array(positions), np.array(velocities), np.array(accelerations)


def _moving_average_forms(data, window):
    """Arrays we accept as "a `window`-sample moving average of `data`".

    The notebook's own trailing version first, then the centred convolution a
    student might reach for instead. Both are honest moving averages and both
    are as long as the data, which is what the plot needs.

    The window is bounded before building the convolution kernel: a student can
    type any number they like, and `np.ones(10 ** 12)` would take the grading
    subprocess out rather than fail a row.
    """
    forms = [_moving_average(data, window)]
    if 1 <= window <= data.size:
        try:
            forms.append(np.convolve(data, np.ones(window) / window, mode="same"))
        except Exception:
            pass
    return forms


def _low_pass_forms(data, alpha):
    """Arrays we accept as "a low-pass / alpha filter of `data` at `alpha`".

    Seeding the estimate with the first measurement (as the notebook does) or
    with zero are both reasonable; the second just costs a startup transient.
    """
    return [_low_pass(data, alpha), _low_pass(data, alpha, start=0.0)]


# ---------------------------------------------------------------------------
# Did the result reach a plot? Figures are captured and closed before a checker
# runs, so `plt.get_fignums()` is always empty here — the AST is the only
# evidence there is.
# ---------------------------------------------------------------------------

def _plot_calls(ctx):
    calls = []
    for node in ast.walk(ctx.tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name in PLOT_FUNCTIONS:
            calls.append(node)
    return calls


def _plots_in_a_loop(ctx):
    """Is the plotting done inside a `for`? Then it draws a series per pass.

    `for ax, signal in zip(axes, (ma, lp, af)): ax.plot(time, signal)` is three
    curves that count as one call in the AST, and there is no way to know
    statically how many passes it makes — so a loop counts as enough of them.
    """
    drawn = {id(call) for call in _plot_calls(ctx)}
    return any(isinstance(node, ast.For)
               and any(id(inner) in drawn for inner in ast.walk(node))
               for node in ast.walk(ctx.tree))


def _plot_inputs(ctx):
    """Every expression that ends up on a plot.

    Each plotting call's arguments, plus the iterable of any `for` loop that
    does its plotting inside — a loop is handed the signals up front, and the
    call itself only ever names the loop variable.
    """
    calls = _plot_calls(ctx)
    drawn = {id(call) for call in calls}
    inputs = []
    for call in calls:
        inputs.extend(call.args)
        inputs.extend(keyword.value for keyword in call.keywords)
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.For) and any(id(inner) in drawn
                                             for inner in ast.walk(node)):
            inputs.append(node.iter)
    return inputs


def _plotted_names(ctx, wanted):
    """Which of `wanted` are handed to a plot.

    We walk the whole argument, so `plt.plot(time, filtered)`,
    `ax1.plot(time, filtered[5:])` and `plt.scatter(t, np.asarray(filtered))`
    all count — as does `plt.plot(t, a, t, b, t, c)`, which draws three curves
    from one call.
    """
    found = set()
    for expression in _plot_inputs(ctx):
        for node in ast.walk(expression):
            if isinstance(node, ast.Name) and node.id in wanted:
                found.add(node.id)
    return found


def _plot_uses(ctx, wanted):
    """Is one of `wanted` handed to a plot?"""
    return bool(_plotted_names(ctx, wanted))


def _plotted(ctx, *names):
    """Is one of `names` drawn — or, failing that, something else this cell made?

    The fallback keeps a student who plots the filter call inline instead of the
    variable (`plt.plot(time, moving_average(z, 5))`) from being marked down for
    a plot that is plainly on the screen.
    """
    wanted = {name for name in names if name}
    return _plot_uses(ctx, wanted) or _plot_uses(ctx, set(FILTER_FUNCTIONS))


# ---------------------------------------------------------------------------
# The two "change the knob and re-run" exercises share all of their structure,
# so they share one body: find the gain, check it moved, re-run the filter from
# it, and confirm the answer got plotted.
# ---------------------------------------------------------------------------

def _signal_candidates(ctx, source):
    """Names that could be holding the filtered signal, best first.

    The names the cell ships with lead, whatever they hold — if `filtered` is a
    string, saying so is far better feedback than quietly grading something else
    instead. Behind them come any other run of numbers this cell made that is as
    long as the measurements, so a student who called theirs `ma` is still read.
    """
    own = _own(ctx)
    named = [name for name in FILTERED_VARS if name in own]
    others = []
    for name in sorted(own):
        if name in named or name in GIVEN_SIGNALS:
            continue
        array = _array(ctx.env.get(name))
        if array is not None and array.shape == source.shape:
            others.append(name)
    return named + others


def _agreeing_pair(ctx, source, signal_names, gains, expected_forms):
    """The (signal, gain) pair the student's own numbers agree on, if any.

    Several names can plausibly be the knob — the notebook's own moving average
    keeps its slice in a variable called `window` — so rather than trusting
    whichever we happen to look at first, we look for the pair that reproduces:
    the gain which, re-run through the filter, gives the signal the cell left
    behind. Nothing here can pass a submission that never ran the filter, since
    a pair only counts when the numbers match.
    """
    for name in signal_names:
        array = _array(ctx.env.get(name))
        if array is None or array.shape != source.shape:
            continue
        for gain_name, gain in gains:
            if any(_matches(array, form) for form in expected_forms(source, gain)):
                return name, gain_name, gain
    return None


def _check_reruns_filter(ctx, sets_label, noun, gain_names, to_gain, gain_message,
                         default, default_message, source_name, expected_forms,
                         match_label, match_message):
    """The body both "change the knob" exercises share.

    Find the knob, check it is a number a filter could use and that it moved off
    the shipped value, then re-run the filter from it and compare — which is the
    only way to catch a submission whose plotted curve came from a different
    number than the one the cell now says.
    """
    source = _array(ctx.env.get(source_name))
    if _unusable_data(source) and not _data_check(ctx, source_name, source):
        return  # nothing below this could mean anything

    signal_options = _signal_candidates(ctx, source)
    gain_options = [name for name in gain_names if name in _own(ctx)]
    usable = [(name, to_gain(ctx.env.get(name))) for name in gain_options]
    usable = [(name, gain) for name, gain in usable if gain is not None]

    agreed = _agreeing_pair(ctx, source, signal_options, usable, expected_forms)
    if agreed is not None:
        filtered_variable, gain_variable, gain = agreed
    else:
        # Nothing agrees. Fall back to the likeliest name of each, so the rows
        # below still talk about what the student actually wrote.
        filtered_variable = signal_options[0] if signal_options else None
        gain_variable, gain = usable[0] if usable else (None, None)
        if gain_variable is None and gain_options:
            gain_variable = gain_options[0]

    if not ctx.require(f"Sets {sets_label}", gain_variable,
                       "Assign it in this cell, keeping the name the cell came with: "
                       f"`{gain_names[0]} = ...`."):
        return
    if not ctx.require(f"{noun} is a usable value", gain is not None,
                       gain_message, got=_short(ctx.env.get(gain_variable))):
        return
    ctx.require(f"{noun} is your own choice", _changed(float(gain), default),
                default_message,
                expected=f"anything but {default:g}", got=f"{gain:g}")

    if not ctx.require("Stores the filtered signal", filtered_variable,
                       "Keep the assignment the cell came with — `filtered = ...` — "
                       "so the plot has something to draw."):
        return
    filtered = _array(ctx.env.get(filtered_variable))
    if not ctx.require("The filtered signal is a run of numbers", filtered is not None,
                       "The filter returns one estimate per measurement; store that "
                       "whole array, not a single number.",
                       got=_short(ctx.env.get(filtered_variable))):
        return

    if not ctx.require("There is one filtered value per measurement",
                       filtered.shape == source.shape,
                       "Filter the whole measurement array, from the first sample "
                       "to the last.",
                       expected=f"{source.size} values",
                       got=f"{filtered.size} values"):
        return
    ctx.require(match_label,
                any(_matches(filtered, form) for form in expected_forms(source, gain)),
                match_message,
                expected=f"the filter re-run at {gain:g}",
                got="a signal those numbers don't produce")

    ctx.require("The filtered signal is plotted", _plotted(ctx, filtered_variable),
                "Keep the plotting lines — the whole point of the exercise is seeing "
                "what your value did to the curve.")


def check_ex1(ctx):
    """"Change the `window_size` below. Try 3, 10, 30."" — the moving average."""
    if not _ran(ctx):
        return
    _check_reruns_filter(
        ctx,
        sets_label="a window size",
        noun="The window size",
        gain_names=WINDOW_VARS,
        to_gain=_whole,
        gain_message="A window is a count of measurements to average, so it has to be "
                     "a whole number of samples, at least 1.",
        default=EX1_DEFAULT_WINDOW,
        default_message=f"That's still the {EX1_DEFAULT_WINDOW} the cell came with. "
                        "It is one of the suggested values, so you may have meant it "
                        "— but the exercise is the comparison: run it again at 10 and "
                        "at 30 and watch the curve get smoother and later.",
        source_name="measured_altitude",
        expected_forms=_moving_average_forms,
        match_label="The signal is the moving average over your window",
        match_message="Your filtered signal isn't the moving average of that window. "
                      "Pass the variable to the filter — "
                      "`moving_average(measured_altitude, number_of_values_avg)` — so "
                      "the picture always matches the number you set.",
    )


def check_ex2(ctx):
    """"Change `alpha` below. Try 0.05, 0.2, 0.8."" — the low-pass filter."""
    if not _ran(ctx):
        return

    def usable_alpha(value):
        number = _number(value)
        return number if number is not None and 0 < number <= 1 else None

    _check_reruns_filter(
        ctx,
        sets_label="an alpha",
        noun="Alpha",
        gain_names=ALPHA_VARS,
        to_gain=usable_alpha,
        gain_message="Alpha is the share of the new measurement you trust, so it "
                     "belongs between 0 and 1: at 1 the filter just copies the "
                     "sensor, near 0 it barely listens to it.",
        default=EX2_DEFAULT_ALPHA,
        default_message=f"That's still the {EX2_DEFAULT_ALPHA:g} the cell came with. "
                        "Confusingly it is also one of the three suggestions, so if "
                        "you picked it on purpose, good — now run it again at 0.05 "
                        "and at 0.8, because the exercise is the difference between "
                        "them.",
        source_name="measured_altitude",
        expected_forms=_low_pass_forms,
        match_label="The signal is the low-pass re-run at your alpha",
        match_message="Your filtered signal isn't the low-pass of that alpha. Pass "
                      "the variable to the filter — "
                      "`low_pass_filter(measured_altitude, alpha)` — so the picture "
                      "always matches the number you set.",
    )


def check_ex3(ctx):
    """"Change `alpha`, `beta`, and `gamma`." — the alpha-beta-gamma filter.

    The prompt says in as many words that you do not need perfect values, so
    nothing here judges how good the estimates are. All that has to be true is
    that the three gains are numbers a filter could use, that at least one of
    them is the student's, and that the three output arrays really did come out
    of the filter run at those gains rather than being left over from the
    worked example above.
    """
    if not _ran(ctx):
        return

    names = tuple(_defined_as(ctx, candidates, _number)
                  for candidates in (ALPHA_VARS, BETA_VARS, GAMMA_VARS))
    if not ctx.require("Sets alpha, beta and gamma", all(names),
                       "Keep all three assignments the cell came with — "
                       "`ALPHA = ...`, `BETA = ...`, `GAMMA = ...` — and change the "
                       "numbers on them.",
                       got=", ".join(n or "missing" for n in names)):
        return

    gains = tuple(_number(ctx.env.get(name)) for name in names)
    low, high = GAIN_RANGE
    corrects = {"Alpha": "position", "Beta": "velocity", "Gamma": "acceleration"}
    for label, name, gain in zip(("Alpha", "Beta", "Gamma"), names, gains):
        ctx.require(f"{label} is a usable gain",
                    gain is not None and low <= gain <= high,
                    f"{label} scales how much of the residual is applied to the "
                    f"{corrects[label]} estimate. It is a plain number, normally "
                    "between 0 and 1 — at 0 the filter ignores the measurement, at 1 "
                    "it swallows it whole.",
                    expected=f"{low:g} to {high:g}", got=_short(ctx.env.get(name)))
    if any(gain is None for gain in gains):
        return

    ctx.require("At least one gain is your own",
                any(_changed(gain, default)
                    for gain, default in zip(gains, EX3_DEFAULT_GAINS)),
                "These are still the three values the cell came with. Try a bigger "
                "alpha, then a bigger beta, then a gamma large enough to make the "
                "acceleration trace go jittery — that comparison is the exercise.",
                expected="a gain you picked",
                got=", ".join(f"{g:g}" for g in gains))

    dt = _number(ctx.env.get("dt"))
    ctx.require("The time step still matches the data",
                dt is not None and abs(dt - DT) <= CHANGED_EPS,
                f"The measurements are {DT:g} s apart, so `dt = {DT:g}`. Changing dt "
                "changes the physics the filter assumes, not how much it trusts the "
                "sensor — the gains are the knobs here.",
                expected=f"{DT:g}", got=_short(ctx.env.get("dt")))

    estimate_names = tuple(_defined_as(ctx, candidates, _array)
                           for candidates in (POSITION_VARS, VELOCITY_VARS, ACCEL_VARS))
    if not ctx.require("Produces position, velocity and acceleration estimates",
                       all(estimate_names),
                       "Keep the three-way unpacking the cell came with: "
                       "`position_estimate, velocity_estimate, acceleration_estimate "
                       "= alpha_beta_gamma_filter(...)`.",
                       got=", ".join(n or "missing" for n in estimate_names)):
        return
    estimates = tuple(_array(ctx.env.get(name)) for name in estimate_names)

    source = _array(ctx.env.get("measured_acceleration"))
    if _unusable_data(source) and not _data_check(ctx, "measured_acceleration", source):
        return  # nothing below this could mean anything
    if not ctx.require("Each estimate has one value per measurement",
                       all(e is not None and e.shape == source.shape
                           for e in estimates),
                       "Run the filter over the whole measurement array, and "
                       "keep all three arrays it hands back.",
                       expected=f"{source.size} values each",
                       got=", ".join("missing" if e is None else str(e.size)
                                     for e in estimates)):
        return
    reference = _alpha_beta_gamma(source, dt if dt is not None else DT, *gains)
    wrong = [label for label, have, want
             in zip(("position", "velocity", "acceleration"), estimates, reference)
             if not _matches(have, want)]
    ctx.require("The estimates come from the filter run at your gains", not wrong,
                "Pass your variables into the call — "
                "`alpha_beta_gamma_filter(measured_acceleration, dt, alpha=ALPHA, "
                "beta=BETA, gamma=GAMMA)` — and re-run the cell, so the plots show "
                "the gains you set rather than the ones above.",
                expected="the filter re-run at " + ", ".join(f"{g:g}" for g in gains),
                got=("the " + " and ".join(wrong) + " estimate"
                     + ("s don't" if len(wrong) > 1 else " doesn't")
                     + " match") if wrong else "")

    # Deliberately "one of them reaches a plot", not all three: a student who
    # deleted two panels to stare at the third has still done the exercise, and
    # the label says only what is actually verified.
    ctx.require("The estimates are plotted", _plotted(ctx, *estimate_names),
                "Keep the panels — seeing position, velocity and acceleration react "
                "together is how you tell which gain did what.")


def check_filter_the_altitude(ctx):
    """The final challenge: three filters over one seeded altitude signal.

    Pinned from the cell with `#% checker: filter_the_altitude`. The three
    signals are found by their *values*, not their names — anything the cell
    itself binds that equals the moving average, or the low-pass, counts —
    because the challenge never tells the student what to call them. A name
    holding the measurements unchanged is set aside as the raw trace: it is what
    the filtered curves are drawn against, not one of them.

    The alpha filter is the low-pass recursion (`x += alpha * (z - x)` is
    `alpha * z + (1 - alpha) * x` rearranged), so reusing `low_pass_filter` for
    it is perfectly correct and accepted. The only thing that has to differ is
    that three separate signals exist, since the task is to plot three curves.
    """
    if not _ran(ctx):
        return

    measured = _array(ctx.env.get("measured_altitude"))
    truth = _array(ctx.env.get("true_altitude"))
    if _unusable_data(measured) and not _data_check(ctx, "measured_altitude", measured):
        return  # nothing below this could mean anything

    signals = {}
    raw_names = {"measured_altitude"}  # the sensor trace, under any second name
    for name in sorted(_own(ctx)):
        if name in GIVEN_SIGNALS:
            continue
        array = _array(ctx.env.get(name))
        if array is None or array.shape != measured.shape:
            continue
        if _matches(array, measured):
            raw_names.add(name)  # a second name for the raw sensor isn't a filter
        else:
            signals[name] = array

    # No early return here: an empty cell should be told each of the three
    # things it is missing, which is the useful feedback on a challenge.
    ctx.require("Builds three filtered signals from the measurements",
                len(signals) >= 3,
                "Store each filter's output in its own variable, one value per "
                "measurement — e.g. "
                f"`ma = moving_average(measured_altitude, {CHALLENGE_WINDOW})` — so "
                "all three curves are there to plot.",
                expected="3 filtered signals",
                got=f"{len(signals)} ({', '.join(signals) or 'none'})")

    average_forms = _moving_average_forms(measured, CHALLENGE_WINDOW)
    low_pass_forms = _low_pass_forms(measured, CHALLENGE_ALPHA)
    averages = [n for n, a in signals.items() if any(_matches(a, f) for f in average_forms)]
    low_passes = [n for n, a in signals.items() if any(_matches(a, f) for f in low_pass_forms)]

    ctx.require(f"A {CHALLENGE_WINDOW}-sample moving average of the measurements",
                averages,
                "None of your signals is the measurements averaged "
                f"{CHALLENGE_WINDOW} samples at a time. The notebook's helper does it: "
                f"`moving_average(measured_altitude, {CHALLENGE_WINDOW})` — and filter "
                "the noisy measurements, not the true altitude, which a real rocket "
                "never gets to see.",
                expected=f"a window of {CHALLENGE_WINDOW}")
    ctx.require(f"A low-pass filter at alpha = {CHALLENGE_ALPHA:g}", low_passes,
                "None of your signals is the low-pass of the measurements at alpha = "
                f"{CHALLENGE_ALPHA:g}: `low_pass_filter(measured_altitude, "
                f"{CHALLENGE_ALPHA:g})`.",
                expected=f"alpha = {CHALLENGE_ALPHA:g}")
    ctx.require("An alpha filter kept as its own signal", len(low_passes) >= 2,
                "The alpha filter is `x = x + alpha * (z - x)`, which is the low-pass "
                "recursion rearranged — so writing it out, or calling "
                "`low_pass_filter` again, are both fine. What's missing is a *second* "
                "variable holding it: the challenge asks for three curves on the plot.",
                expected="2 signals with this shape (low-pass and alpha filter)",
                got=f"{len(low_passes)}")

    # Every filter should sit closer to the truth than the raw sensor does —
    # that reduction is what "filtering" means, and it is why the exercise
    # bothers giving you a true_altitude you'd never have on a real flight.
    found = [(label, signals[names[0]])
             for label, names in (("moving average", averages), ("low-pass", low_passes))
             if names]
    if len(low_passes) >= 2:
        found.append(("alpha filter", signals[low_passes[1]]))
    raw = _rms(measured, truth)
    if raw is not None and found:
        errors = [(label, _rms(signal, truth)) for label, signal in found]
        ctx.require("Every filter is closer to the truth than the raw sensor",
                    all(e is not None and e < raw for _, e in errors),
                    "A filter that doesn't beat the raw measurements isn't buying you "
                    "anything. Check you're filtering `measured_altitude` and that "
                    "each estimate lines up in time with it.",
                    expected=f"error below the sensor's {raw:.1f} m",
                    got="; ".join(f"{label} {'?' if e is None else format(e, '.1f')} m"
                                  for label, e in errors))

    # Three ways to draw three curves, all of them fine: three plotting calls,
    # one call handed all three signals (`plt.plot(t, a, t, b, t, c)`), or a loop
    # whose pass count nothing can know statically.
    drawn = len(_plot_calls(ctx))
    plotted = _plotted_names(ctx, set(signals))
    ctx.require("All three filtered signals are plotted",
                drawn >= MIN_PLOTTED_SERIES or len(plotted) >= MIN_PLOTTED_SERIES
                or _plots_in_a_loop(ctx),
                "Draw the three filtered signals against the measurements — that side "
                "by side is the whole question: which is smoothest, which reacts "
                "fastest.",
                expected=f"{MIN_PLOTTED_SERIES} or more series",
                got=f"{len(plotted)} of your signals, in {drawn} plotting call"
                    + ("" if drawn == 1 else "s"))
    ctx.require("The raw measurements are on the plot too",
                _plot_uses(ctx, raw_names),
                "Plot the noisy measurements behind the filtered curves — without "
                "them there's nothing to judge the smoothing against. A scatter works "
                "well: `plt.scatter(time, measured_altitude, s=10)`.")
