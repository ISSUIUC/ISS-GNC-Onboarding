"""Behaviour checker for the Introduction module's final rocket-flight challenge.

The challenge is deliberately open-ended: the student picks their own climb
rate, tank size, burn rate and print formatting. There is no single right
answer to diff against, so instead of comparing variables we read the flight
log back out of their output and verify the *simulation is self-consistent*:

  * a `while` loop actually produced the log (not a row of hard-coded prints),
  * altitude climbs by exactly their own velocity, every second,
  * fuel falls at a steady rate and lands exactly on empty,
  * "Liftoff!" fires before the first second, "Low Fuel!" fires the moment the
    tank drops under 20% (and not while it's still full), "Engine Cutoff!"
    fires once, at zero.

The only formatting we insist on is that altitude and fuel are printed with
their name beside the number ("Altitude: 50 m", "Fuel: 90%") — exactly like the
worked example in the prompt. Everything else is the student's choice.
"""

import ast
import re

# Variable names we accept for each quantity. `ctx.defined` only looks at names
# the student's own code assigns, so short physics shorthands are safe here —
# they can't collide with leftovers from earlier notebook cells.
# Order matters: the first match wins, so the canonical name leads and the
# single letters trail.
ALTITUDE_VARS = ("altitude", "alt", "height", "elevation",
                 "starting_altitude", "start_altitude", "current_altitude",
                 "starting_height", "start_height", "current_height",
                 "rocket_altitude", "rocket_height", "altitude_m", "alt_m",
                 "distance", "position", "pos",
                 "h", "y")
VELOCITY_VARS = ("velocity", "vertical_velocity", "vel", "speed", "vertical_speed",
                 "climb_rate", "climb_speed", "climb", "climbrate",
                 "ascent_rate", "ascent_speed", "ascent", "rise_rate", "rise",
                 "rocket_speed", "rocket_velocity", "velocity_mps", "start_velocity",
                 "starting_velocity", "initial_velocity",
                 "v")
FUEL_VARS = ("fuel", "fuel_remaining", "fuel_left", "fuel_percent", "fuel_percentage",
             "fuel_pct", "fuel_level", "fuel_amount", "fuel_supply", "remaining_fuel",
             "propellant", "propellant_remaining", "fuel_tank", "tank", "gas", "f")

# Words we accept as *labels in the printed log*. Deliberately stricter than the
# variable names: "h" would happily match "Time: 1 h".
ALTITUDE_LABELS = ("altitude", "alt", "height", "elevation", "distance", "position")
FUEL_LABELS = ("fuel", "propellant")

NUMBER = r"(-?\d+(?:\.\d+)?)"
TOL = 1e-6
LOW_FUEL_FRACTION = 0.20

# Lines announcing an event, not reporting telemetry — skip them when reading
# numbers so "Low Fuel! 10% left" can't masquerade as an extra fuel reading.
EVENTS = {
    "liftoff": r"lift\s*-?\s*off",
    "low fuel": r"low\s+fuel",
    "cutoff": r"cut\s*-?\s*off",
}
EVENT_LINE = re.compile("|".join(EVENTS.values()), re.IGNORECASE)


def _readings(lines, words):
    """Every "<word> ... <number>" reading in the log, as (line index, value).

    Tolerates any separator between the label and the number, so
    "Altitude: 50 m", "altitude = 50" and "Altitude 50m" all read as 50.
    """
    pattern = re.compile(r"\b(?:%s)\w*[^0-9\n-]{0,12}%s" % ("|".join(words), NUMBER), re.I)
    found = []
    for index, line in enumerate(lines):
        if EVENT_LINE.search(line):
            continue
        match = pattern.search(line)
        if match:
            found.append((index, float(match.group(1))))
    return found


def _event_lines(lines, key):
    pattern = re.compile(EVENTS[key], re.IGNORECASE)
    return [i for i, line in enumerate(lines) if pattern.search(line)]


def _steps_around(log, line):
    """Which second(s) an event line could belong to.

    A warning printed between two telemetry lines is genuinely ambiguous — the
    student may print it before or after that second's numbers — so we return
    both neighbouring readings and let the caller accept either. Judging an
    ambiguous case in the student's favour is the right way to be wrong.
    """
    before = [i for i, (index, _) in enumerate(log) if index <= line]
    after = [i for i, (index, _) in enumerate(log) if index >= line]
    steps = set()
    if before:
        steps.add(before[-1])
    if after:
        steps.add(after[0])
    return steps


def _tank_size(ctx, name, fuel_values, burn):
    """How full a full tank is: the first number the student assigned to it."""
    if name:
        for node in ast.walk(ctx.tree):
            if (isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, (int, float))
                    and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)):
                return float(node.value.value)
    return fuel_values[0] + burn  # fall back to reconstructing it from the log


def _prints_inside_loop(ctx):
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.While):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and getattr(inner.func, "id", None) == "print":
                    return True
    return False


def check(ctx):
    if not ctx.require("Runs without an error", ctx.ok,
                       "Fix the error shown in the output above, then check again."):
        return
    if not ctx.require("Prints a flight log", bool(ctx.lines),
                       "Your program didn't print anything."):
        return

    # --- the tools the challenge asks you to use ---------------------------
    ctx.require("Uses a `while` loop to fly the rocket", ctx.uses("while"),
                "The flight should run in `while fuel > 0:` — a fixed sequence of "
                "prints isn't a simulation.")
    ctx.require("Uses `if` statements for the flight events", ctx.count("if") >= 2,
                "You need at least two: one for the low-fuel warning, one for "
                "engine cutoff.")
    ctx.require("The log is printed from inside the loop", _prints_inside_loop(ctx),
                "Every second's telemetry should be printed by the loop itself.")

    altitude_name = ctx.defined(*ALTITUDE_VARS)
    velocity_name = ctx.defined(*VELOCITY_VARS)
    fuel_name = ctx.defined(*FUEL_VARS)
    ctx.require("Stores the starting altitude in a variable", altitude_name,
                "Give it a recognisable name, e.g. `altitude = 0`.")
    ctx.require("Stores the vertical velocity in a variable", velocity_name,
                "Give it a recognisable name, e.g. `velocity = 50`.")
    ctx.require("Stores the fuel remaining in a variable", fuel_name,
                "Give it a recognisable name, e.g. `fuel = 100`.")

    # --- read the flight log back out of the output ------------------------
    altitude_log = _readings(ctx.lines, ALTITUDE_LABELS)
    fuel_log = _readings(ctx.lines, FUEL_LABELS)
    label_hint = ('Label each number so it can be read back, like the example: '
                  'print(f"Altitude: {altitude} m").')
    if not ctx.require("Prints the altitude every second", len(altitude_log) >= 3,
                       f"I found {len(altitude_log)} altitude readings in your output. "
                       + label_hint):
        return
    if not ctx.require("Prints the fuel remaining every second", len(fuel_log) >= 3,
                       f"I found {len(fuel_log)} fuel readings in your output. "
                       + label_hint):
        return
    ctx.require("Reports altitude and fuel together each second",
                len(altitude_log) == len(fuel_log),
                f"{len(altitude_log)} altitude readings but {len(fuel_log)} fuel "
                "readings — every second should report both.")

    altitudes = [v for _, v in altitude_log]
    fuels = [v for _, v in fuel_log]
    climbs = [b - a for a, b in zip(altitudes, altitudes[1:])]
    burns = [a - b for a, b in zip(fuels, fuels[1:])]

    # --- the physics has to hold together ----------------------------------
    ctx.require("The rocket climbs every second", all(c > TOL for c in climbs),
                "Altitude has to increase each second.",
                got=" -> ".join(f"{v:g}" for v in altitudes[:8]))
    ctx.require("It climbs by the vertical velocity each second",
                climbs and max(climbs) - min(climbs) <= TOL,
                "Each second should add the same amount — your velocity — to the "
                "altitude.",
                got=" -> ".join(f"{v:g}" for v in altitudes[:8]))
    velocity = ctx.get(*VELOCITY_VARS)
    if climbs and isinstance(velocity, (int, float)) and not isinstance(velocity, bool):
        ctx.require("Altitude gain matches the velocity variable",
                    abs(climbs[0] - float(velocity)) <= TOL,
                    "The altitude in your log doesn't go up by your own velocity — "
                    "use the variable in the update, e.g. `altitude += velocity`.",
                    expected=f"+{float(velocity):g} m per second",
                    got=f"+{climbs[0]:g} m per second")

    ctx.require("Fuel drops every second", all(b > TOL for b in burns),
                "Fuel has to decrease each second, or the loop will never end.",
                got=" -> ".join(f"{v:g}" for v in fuels[:8]))
    ctx.require("Fuel burns at a steady rate", burns and max(burns) - min(burns) <= TOL,
                "Burn the same amount of fuel every second.",
                got=" -> ".join(f"{v:g}" for v in fuels[:8]))
    ctx.require("Fuel never goes negative", min(fuels) >= -TOL,
                "The engine should stop at empty, not keep burning past it.",
                got=f"lowest reading: {min(fuels):g}")
    ctx.require("The tank ends up empty", abs(fuels[-1]) <= TOL,
                "The flight should run until the fuel reaches exactly 0.",
                expected="0", got=f"{fuels[-1]:g}")

    # --- the events fire at the right moments ------------------------------
    burn = burns[0] if burns else 0.0
    threshold = LOW_FUEL_FRACTION * _tank_size(ctx, fuel_name, fuels, burn)
    first_reading = min(altitude_log[0][0], fuel_log[0][0])

    liftoff = _event_lines(ctx.lines, "liftoff")
    ctx.require('Announces "Liftoff!" before the first second',
                liftoff and liftoff[0] < first_reading,
                'Print "Liftoff!" once, before the loop starts.' if liftoff
                else 'Nothing in your output says "Liftoff!".')

    low_steps = [i for i, value in enumerate(fuels) if value < threshold - TOL]
    warnings = [_steps_around(fuel_log, line) for line in _event_lines(ctx.lines, "low fuel")]
    if ctx.require('Warns "Low Fuel!" when the tank drops below 20%', warnings,
                   'Nothing in your output says "Low Fuel!". With your numbers it '
                   f'should appear once the fuel goes under {threshold:g}.'):
        ctx.require("The warning only appears below 20%",
                    all(any(step in low_steps for step in steps) for steps in warnings),
                    f"You warned while the tank was still above {threshold:g} — guard "
                    "the warning with an `if`.")
        ctx.require("The warning appears as soon as the fuel is low",
                    low_steps and any(low_steps[0] in steps for steps in warnings),
                    f"The fuel first drops under {threshold:g} at second "
                    f"{low_steps[0] + 1} of your log — warn there, not later."
                    if low_steps else
                    f"Your fuel never gets below 20% of the tank ({threshold:g}).")

    cutoffs = _event_lines(ctx.lines, "cutoff")
    if ctx.require('Announces "Engine Cutoff!" when the fuel is gone', cutoffs,
                   'Nothing in your output says "Engine Cutoff!".'):
        ctx.require("Cutoff is announced exactly once", len(cutoffs) == 1,
                    f"It appears {len(cutoffs)} times — the engine only cuts out once.",
                    expected="1", got=str(len(cutoffs)))
        cutoff_steps = sorted(_steps_around(fuel_log, cutoffs[0]))
        ctx.require("Cutoff happens at zero fuel",
                    any(abs(fuels[step]) <= TOL for step in cutoff_steps),
                    "Announce the cutoff on the second where the fuel reaches 0, "
                    "not before.",
                    expected="fuel 0",
                    got=", ".join(f"second {step + 1}: fuel {fuels[step]:g}"
                                  for step in cutoff_steps))
