"""Behaviour checkers for Introduction.ipynb.

One function per exercise, named after it: `check_ex1` grades `introduction-ex1`,
and so on. The engine binds them automatically — the notebook cells need no
`#% checker:` directive. (Exercise 7, the rocket flight, is big enough to keep
its own file: see `rocket_flight.py`.)

These cover the "practice" exercises that used to pass on *ran without an
error* alone, so the task actually has to be done. They stay deliberately
lenient about spelling and formatting: the point is to catch a student who
didn't do the exercise, not one who named a variable differently.
"""

import re

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

# The text each cell ships with — an unmodified value means the task wasn't done.
EX1_DEFAULT = "Hello GNC Team!"


def _ran(ctx, hint="Fix the error shown in the output above, then check again."):
    """Every exercise starts here: nothing else is meaningful if it crashed."""
    return ctx.require("Runs without an error", ctx.ok, hint)


def _printed(ctx, value) -> bool:
    """Did `value` show up in the output, however it was formatted?

    Numbers are matched numerically (so `5`, `5.0` and `05` all count) and text
    case-insensitively, because we only care that the student printed the thing.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        text = str(value).strip().lower()
        return bool(text) and text in ctx.stdout.lower()
    return any(abs(float(m) - float(value)) < 1e-9 for m in NUMBER.findall(ctx.stdout))


def check_ex1(ctx):
    """"Change the text inside the quotation marks to print your own message." """
    if not _ran(ctx):
        return
    if not ctx.require("Uses `print(...)`", ctx.uses("print"),
                       "Keep the print statement — just change what's inside it."):
        return
    message = ctx.stdout.strip()
    if not ctx.require("Prints a message", bool(message),
                       "Nothing was printed. Put your message inside the quotation "
                       'marks, e.g. print("Hello from Eddie!").'):
        return
    ctx.require("The message is your own", message != EX1_DEFAULT,
                "That's still the message the cell came with — change the text "
                "inside the quotation marks to something of your own.",
                got=message)


def check_ex2(ctx):
    """Create `MIDAS_color` and `MIDAS_num_pins`, then print both."""
    if not _ran(ctx):
        return

    color_name = ctx.defined("MIDAS_color", "MIDAS_colour", "midas_color", "midas_colour")
    pins_name = ctx.defined("MIDAS_num_pins", "MIDAS_num_pin", "MIDAS_pins",
                            "midas_num_pins", "midas_pins")
    if not ctx.require("Creates `MIDAS_color`", color_name,
                       "Assign it in this cell, exactly as spelled in the prompt: "
                       'MIDAS_color = "red".'):
        return
    if not ctx.require("Creates `MIDAS_num_pins`", pins_name,
                       "Assign it in this cell, exactly as spelled in the prompt: "
                       "MIDAS_num_pins = 40."):
        return

    color = ctx.get(color_name)
    pins = ctx.get(pins_name)
    ctx.require("`MIDAS_color` holds text", isinstance(color, str) and color.strip(),
                "A colour is text, so it goes in quotation marks — "
                'MIDAS_color = "red".',
                got=repr(color))
    ctx.require("`MIDAS_num_pins` holds a number",
                isinstance(pins, (int, float)) and not isinstance(pins, bool),
                "A count is a plain number — no quotation marks: "
                "MIDAS_num_pins = 40.",
                got=repr(pins))

    ctx.require("Prints `MIDAS_color`", _printed(ctx, color),
                "Print the variable itself: print(MIDAS_color).",
                expected=str(color), got=ctx.stdout.strip() or "(no output)")
    ctx.require("Prints `MIDAS_num_pins`", _printed(ctx, pins),
                "Print the variable itself: print(MIDAS_num_pins).",
                expected=str(pins), got=ctx.stdout.strip() or "(no output)")


def check_silence_the_ifs(ctx):
    """"Modify the code below so nothing prints." (`#% checker: silence_the_ifs`)

    The exercise is about *conditions*: change the values the ifs test (or the
    branches they run) until none of them reaches a print. So we check both
    halves — the output is empty, and the if-statements are still there.
    """
    if not _ran(ctx, "An error isn't the same as silence — the cell has to run "
                     "cleanly and print nothing. Fix the error shown above."):
        return

    printed = ctx.stdout.strip()
    ctx.require("Nothing is printed", not printed,
                "Your code still prints. Look at which branch each `if` takes: "
                "the printed lines below tell you which conditions are still true.",
                expected="(no output)", got=printed)

    ctx.require("The if-statements are still there", ctx.count("if") >= 3,
                "Deleting the code isn't the exercise — keep the conditionals and "
                "change what they test (or what they do) so no branch prints.",
                expected="the 3 if-statements from the starter code",
                got=f"{ctx.count('if')} left")

    ctx.require("`print` still means print", "print" not in ctx.assigned,
                "Redefining `print` silences the cell without changing a single "
                "condition — nice trick, but work with the `if`s instead.")
