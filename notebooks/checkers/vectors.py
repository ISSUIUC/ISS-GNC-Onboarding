"""Behaviour checkers for Vectors.ipynb.

One function per exercise, named after it (`check_ex1` grades `vectors-ex1`);
the engine binds them automatically. See `introduction.py` for the same pattern
and `rocket_flight.py` for a full-size checker.
"""

import re

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
TOL = 1e-6

# Both vectors are the student's own choice, so we accept any sensible name.
POSITION_VARS = ("p", "position", "p0", "p_0", "pos", "position0", "position_0",
                 "start_position", "starting_position", "initial_position", "r", "x")
VELOCITY_VARS = ("v", "velocity", "vel", "v0", "v_0", "speed", "velocity0")
NEXT_VARS = ("p1", "p_1", "position1", "position_1", "p_t1", "p_next", "next_position",
             "new_position", "p_new", "position_at_1", "p_final", "final_position",
             "pos1", "pos_1")


def _vector(value):
    """`value` as a plain list of floats, or None if it isn't a vector."""
    if value is None or isinstance(value, (str, bytes, dict)):
        return None
    try:
        items = list(value)
    except TypeError:
        return None
    out = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            try:
                item = float(item)  # numpy scalars
            except (TypeError, ValueError):
                return None
        out.append(float(item))
    return out or None


def check_ex1(ctx):
    """Define a position `p`, a velocity `v`, and `p` at t = 1; print all three.

    The numbers are the student's to pick, so there is nothing to diff against.
    What has to hold is the physics: after one second, position = p + v.
    """
    if not ctx.require("Runs without an error", ctx.ok,
                       "Fix the error shown in the output above, then check again."):
        return

    position = _vector(ctx.get(*POSITION_VARS))
    velocity = _vector(ctx.get(*VELOCITY_VARS))
    later = _vector(ctx.get(*NEXT_VARS))

    if not ctx.require("Defines a position vector `p`", position,
                       "Give it a recognisable name and make it a vector, e.g. "
                       "p = np.array([0, 0, 0])."):
        return
    if not ctx.require("Defines a velocity vector `v`", velocity,
                       "Give it a recognisable name and make it a vector, e.g. "
                       "v = np.array([12, 5, -3])."):
        return
    if not ctx.require("Defines the position at t = 1", later,
                       "Store it in a second variable, e.g. p1 = p + v."):
        return

    if not ctx.require("The three vectors are the same size",
                       len(position) == len(velocity) == len(later),
                       "A position and a velocity in the same space have the same "
                       "number of components.",
                       expected=f"{len(position)} components",
                       got=f"p: {len(position)}, v: {len(velocity)}, p(t=1): {len(later)}"):
        return

    expected = [a + b for a, b in zip(position, velocity)]
    ctx.require("Position at t = 1 is p + v",
                all(abs(a - b) <= TOL for a, b in zip(expected, later)),
                "After one second the rocket has moved by its velocity, so add the "
                "two vectors: p1 = p + v.",
                expected=str(expected), got=str(later))

    printed = len(NUMBER.findall(ctx.stdout))
    needed = len(position) + len(velocity) + len(later)
    ctx.require("Prints all three vectors", printed >= needed,
                "Print each one — the starting position, the velocity, and the "
                "position after one second.",
                expected=f"{needed} numbers in the output", got=f"{printed}")
