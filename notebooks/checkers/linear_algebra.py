"""Behaviour checkers for Linear_Algebra.ipynb.

One function per exercise. The two "Your Turn" cells bind by position
(`check_ex1` grades `linear_algebra-ex1`), the way `introduction.py` and
`vectors.py` do; the final challenge is pinned in the cell itself with
`#% checker: matrix_boss`, so adding an exercise above it can never quietly
re-point it.

The first two exercises let the student invent their own numbers, so there is
nothing to diff against and no variable names to look for. What we do instead
is collect every array the submission made — the variables it assigned, plus
any `np.array([...])` it wrote inline — decide which of them are the matrix and
the vector the task is about, and then recompute the answer ourselves: the
matrix times the vector for the first, `np.linalg.solve` for the second. Solve
gets at A-inverse-times-b by a different route than inverting does, so it is a
real cross-check on the student's inverse rather than a restatement of it.
Whatever they printed then has to match. Someone who reached for `*` instead of
`@`, or who "inverted" a matrix by taking 1/A element by element, fails;
someone who picked their own numbers, named things oddly and printed them with
a label of their own passes.

The final challenge is the other way round: A, B and b are given, so every
answer is a known matrix, and we look for each one either in the output — read
back out of the brackets, so any spacing or labelling will do — or in a
variable, since the prompt says "compute" rather than "print" for most of the
tasks. What is checked strictly there is the working: the products have to come
from `@`, the transpose from `.T`, and the system from `np.linalg.solve` rather
than from an inverse, because those are the tools the tasks name.

Two habits run through all three. Reading the answers back off the screen is
deliberately forgiving — numpy's bracket layout, a label in front, one number
per line, a rounded-off `print(np.round(x, 2))` — because how a student lays
out their output is not what is being taught here. Deciding whether they did
the work is not forgiving: a task's answer has to be paired with the tool that
task names *and* not be a number typed straight into the cell, or a single
decorative `@` would carry a column of pasted-in matrices to full marks.
"""

import ast
import re

import numpy as np


# --- tunables ---------------------------------------------------------------

# How close a number has to be to count as right. numpy prints eight
# significant digits, so don't tighten this much past 1e-7 or a correct answer
# read back out of the printed output will start failing.
TOL = 1e-6

# The arrays the final challenge hands the student. Every answer it asks for is
# derived from these, so if you edit the numbers in the notebook cell, edit
# them here and nowhere else.
GIVEN_A = [[2, 1], [4, 3]]
GIVEN_B = [[1, 0], [2, 1]]
GIVEN_VECTOR = [5, 11]

# A variable whose name says "inverse" is taken at its word, so that an inverse
# worked out wrongly is caught where it happened. Widen if students start
# calling it something else entirely.
INVERSE_NAME = re.compile(r"inv", re.IGNORECASE)

# Enough of an answer to "are AB and BA the same?" — `print(AB == BA)` prints
# "False" all by itself, which is why the word list alone gets most of them,
# and a student who writes the verdict out longhand says "AB != BA".
DIFFERENT_WORDS = re.compile(r"!=|≠|\b(?:false|no|not|isn't|aren't|differs?|different|"
                             r"differently|unequal|noncommutative)\b", re.IGNORECASE)

# How much of the output is scanned for numbers, and how many arrays are
# considered when working out which one is "the matrix". These exercises print
# a handful of small matrices and make two or three of them; a student who
# prints a megabyte in a loop, or builds forty arrays, mustn't be able to make
# grading time out. Raise MAX_CANDIDATES only alongside a timing check —
# pairing cost grows with the square of it.
MAX_OUTPUT = 20000
MAX_LINES = 400
MAX_CANDIDATES = 8

# How much rounding a *printed* answer is allowed to have. `print(np.round(x, 2))`
# and f"{x:.2f}" are ordinary ways to show an ugly decimal, so a printed number
# counts if it matches the true answer rounded to this many places. Only the
# output is judged this leniently — a variable still has to hold the real value.
# Lower the ceiling if students start passing on numbers that only agree to one
# decimal; raise it if a legitimate `:.1f` is being marked wrong.
DISPLAY_DIGITS = (6, 5, 4, 3, 2, 1)

# Calls whose value is a verdict rather than the numbers handed to them. A
# matrix written out inside one of these is a student checking their own
# working against the prompt, and none of it reaches the output.
COMPARISONS = ("array_equal", "array_equiv", "allclose", "isclose", "equal")

NUMBER = re.compile(r"-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?")
SHAPE_2X2 = re.compile(r"\(\s*2\s*,\s*2\s*\)")

# `print(list(A @ x))` prints numpy scalars as "np.int64(17)", and the 64 is not
# one of the student's numbers. Strip the wrapper before reading numbers out.
DTYPE_CALL = re.compile(r"\b(?:np|numpy)\.\w*?\d+\(")


# --- reading the student's arrays ------------------------------------------

def _ran(ctx):
    """Every exercise starts here: nothing else means anything if it crashed."""
    return ctx.require("Runs without an error", ctx.ok,
                       "Fix the error shown in the output above, then check again.")


def _cached(ctx, key, build):
    """Work something out about the submission once, not once per candidate.

    `_pair` tries every matrix against every vector and each try asks what the
    student printed. Without this the output is re-read hundreds of times, and
    a cell that printed a few thousand lines takes the whole grading run past
    its timeout — which the student sees as their correct answer failing.
    """
    store = ctx.__dict__.setdefault("_cache", {})
    if key not in store:
        store[key] = build()
    return store[key]


def _numeric(value):
    """`value` as a float array, or None if it isn't a rectangle of numbers.

    Everything downstream assumes it can index and compare what it is handed,
    so anything ragged, empty, textual, scalar or higher than two dimensions is
    turned away here rather than blowing up in the middle of a check.
    """
    if value is None or isinstance(value, (str, bytes, bool, dict, set)):
        return None
    try:
        array = np.asarray(value, dtype=float)
    except Exception:
        return None  # a student's own class can raise anything from __array__
    if array.ndim not in (1, 2) or array.size == 0:
        return None
    return array


def _dotted(node):
    """`np.linalg.solve` back out of the AST of that expression."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _assign_order(ctx):
    """Names the submission assigns, in the order it assigns them.

    `A, x = np.array(...), np.array(...)` binds both names in one statement, so
    tuple and list targets are unpacked here too — otherwise a submission that
    happens to use one line instead of two looks as if it made no arrays.
    """
    names = []

    def bind(target):
        if isinstance(target, ast.Name):
            if target.id not in names:
                names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                bind(element)

    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bind(target)
    return names


def _literal(node):
    """The array a bare `[...]` or an `np.array([...])` call evaluates to.

    Only genuine literals: anything with a name or a call inside it comes back
    as None, because the point of asking is "did the student type this number
    in, or work it out?".
    """
    if isinstance(node, ast.Call):
        if _dotted(node.func).rsplit(".", 1)[-1] not in ("array", "asarray", "matrix"):
            return None
        node = node.args[0] if node.args else None
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    try:
        return _numeric(ast.literal_eval(node))
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None


def _literal_arrays(ctx):
    """`np.array([...])` written straight into an expression, never assigned.

    `print(A @ np.array([5, 6]))` is a perfectly good way to do these
    exercises, and there is no variable left behind to find it by.
    """
    found = []
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.Call):
            array = _literal(node)
            if array is not None:
                found.append((None, array))
    return found


def _assigned_arrays(ctx):
    """The arrays sitting in variables the submission assigned.

    `ctx.defined` only reports names the submission itself assigns, so the
    matrices left lying around by the worked examples above can't sneak in.
    """
    found = []
    for name in _assign_order(ctx):
        if ctx.defined(name):
            array = _numeric(ctx.env[name])
            if array is not None:
                found.append((name, array))
    return found


def _arrays(ctx):
    """Every array this submission made, as (name or None, array) pairs.

    Inline literals count here because this is how the *inputs* are found, and
    a matrix written straight into the expression that uses it is still a
    matrix the student made. They deliberately don't count as an *answer* —
    see `_holds`.
    """
    return _assigned_arrays(ctx) + _literal_arrays(ctx)


def _matrices(arrays):
    """The 2-D arrays that are really matrices, not a vector written as a row."""
    return [(name, a) for name, a in arrays if a.ndim == 2 and min(a.shape) > 1]


def _vectors(arrays):
    """Flat arrays, plus row/column vectors written with the extra brackets."""
    return [(name, a) for name, a in arrays if a.ndim == 1 or min(a.shape) == 1]


# --- reading the student's output ------------------------------------------

def _spans(text):
    """The outermost `[...]` groups in `text`, brackets included."""
    spans, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char == "[":
            if depth == 0:
                start = index
            depth += 1
        elif char == "]" and depth:
            depth -= 1
            if depth == 0:
                spans.append(text[start:index + 1])
    return spans


def _parse_span(span, depth=0):
    """One printed `[...]` group as a nested list of floats, or None.

    numpy splits a matrix over several lines and pads it with whatever spacing
    it likes ("[[ 3  1]\\n [ 6  4]]"), so the brackets are the structure and the
    whitespace is noise. A group holding anything that isn't a number — a
    boolean mask, say — reads as None and is ignored.
    """
    if depth > 2:
        return None  # nothing this exercise cares about nests that deep
    inner = span[1:-1]
    rows = _spans(inner)
    if rows:
        parsed = [_parse_span(row, depth + 1) for row in rows]
        return None if any(row is None for row in parsed) else parsed
    numbers = NUMBER.findall(inner)
    return [float(n) for n in numbers] if numbers else None


def _printed_arrays(ctx):
    """Every array in the output, whatever numpy's spacing did to it."""
    def build():
        found = []
        for span in _spans(DTYPE_CALL.sub("(", ctx.stdout[:MAX_OUTPUT])):
            parsed = _parse_span(span)
            if parsed is None:
                continue
            array = _numeric(parsed)
            if array is not None:
                found.append(array)
        return found

    return _cached(ctx, "printed", build)


def _bare_lines(ctx):
    """Numbers printed without brackets — the student's own formatting.

    A line with a bracket in it belongs to an array numpy printed, and that is
    already handled properly; this is for `print(answer[0], answer[1])`.
    """
    def build():
        rows = []
        for line in ctx.lines[:MAX_LINES]:
            if "[" in line or "]" in line:
                continue
            numbers = [float(n) for n in NUMBER.findall(DTYPE_CALL.sub("(", line))]
            if numbers:
                rows.append(numbers)
        return rows

    return _cached(ctx, "bare", build)


def _same(got, want):
    """The same numbers in the same order.

    Shape differences that don't change the numbers — a column vector, a
    matrix printed flat — are the student's business, not ours.
    """
    if got is None or want is None:
        return False
    try:
        if got.shape == want.shape:
            return bool(np.allclose(got, want, rtol=TOL, atol=TOL))
        if got.size == want.size and (got.ndim == 1 or want.ndim == 1 or 1 in got.shape):
            return bool(np.allclose(got.ravel(), want.ravel(), rtol=TOL, atol=TOL))
    except (TypeError, ValueError):
        return False
    return False


def _renderings(want):
    """`want` as the student might have put it on screen: exact, or rounded.

    Nobody reports a filter gain to fifteen places, and `print(np.round(x, 2))`
    is the obvious thing to write when the answer is an ugly decimal. Worked
    out once per question rather than once per candidate — the output can hold
    hundreds of arrays to compare against.
    """
    versions = [want]
    for digits in DISPLAY_DIGITS:
        try:
            versions.append(np.round(want, digits))
        except (TypeError, ValueError):
            break
    return versions


def _displayed(got, versions):
    """Does something read off the screen match any rendering of the answer?

    A stored variable is still held to `_same`; this leniency is for printed
    numbers only.
    """
    return any(_same(got, version) for version in versions)


def _shows(ctx, want):
    """Did they print these numbers, however they formatted them?"""
    want = _numeric(want)
    if want is None:
        return False
    versions = _renderings(want)
    if any(_displayed(array, versions) for array in _printed_arrays(ctx)):
        return True
    flat = want.ravel()
    if flat.size < 2:
        return False
    flats = _renderings(flat)
    rows = _bare_lines(ctx)
    for index, numbers in enumerate(rows):
        # Somewhere along one line: "x = 0.2 -0.1", "Answer: 3 17 (done)".
        for start in range(len(numbers) - flat.size + 1):
            if _displayed(np.asarray(numbers[start:start + flat.size], dtype=float), flats):
                return True
        # Or spread down the page, a row (or a single number) per line, the way
        # `print(x[0])` then `print(x[1])` comes out. Whole lines only, so a
        # stray number at the end of a label can't start a lucky run.
        run = list(numbers)
        for following in rows[index + 1:]:
            if len(run) >= flat.size:
                break
            run += following
        if len(run) == flat.size and _displayed(np.asarray(run, dtype=float), flats):
            return True
    return False


def _holds(ctx, want):
    """Is the answer sitting in one of the variables they assigned?"""
    want = _numeric(want)
    if want is None:
        return False
    return any(_same(array, want) for _, array in _assigned_arrays(ctx))


def _answered(ctx, want):
    """"Compute A + B" is satisfied by printing it *or* by storing it."""
    return _shows(ctx, want) or _holds(ctx, want)


# --- did they work it out, or type it in? ----------------------------------

def _literal_names(ctx):
    """Names that were only ever given a value typed out in full.

    `A = np.array([[1, 2], [3, 4]])` is an input the student wrote down;
    `answer = A @ x` is one they worked out. Telling the two apart is the whole
    basis of "did they do this, or paste it in?".
    """
    typed_only = {}
    for node in ast.walk(ctx.tree):
        if not isinstance(node, ast.Assign):
            continue
        typed = _literal(node.value) is not None
        for target in node.targets:
            if isinstance(target, ast.Name):
                typed_only[target.id] = typed_only.get(target.id, True) and typed
    return {name for name, typed in typed_only.items() if typed}


def _rendered(node):
    """The parts of a printed expression whose numbers reach the screen.

    Everything is descended into — an f-string, `.tolist()`, a round — except a
    comparison, which prints one word however big the matrices inside it are.
    Iterative, so a silly depth of nesting can't blow the stack.
    """
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, ast.Compare):
            continue
        if (isinstance(current, ast.Call)
                and _dotted(current.func).rsplit(".", 1)[-1] in COMPARISONS):
            continue
        yield current
        stack.extend(ast.iter_child_nodes(current))


def _typed_on_screen(ctx):
    """Arrays that reached the output straight from something typed out.

    Anything printed: a literal written into the `print`, or a name that was
    only ever given a literal. Both mean the number on screen was typed rather
    than worked out.
    """
    typed_only = _literal_names(ctx)
    found = []
    for node in ast.walk(ctx.tree):
        if not (isinstance(node, ast.Call)
                and _dotted(node.func).rsplit(".", 1)[-1] == "print"):
            continue
        for argument in node.args:
            for inner in _rendered(argument):
                array = _literal(inner)
                if array is None and isinstance(inner, ast.Name) and inner.id in typed_only:
                    array = _numeric(ctx.env.get(inner.id))
                if array is not None:
                    found.append(array)
    return found


def _typed_arrays(ctx):
    """Arrays the submission *presents as an answer* by typing them out.

    Assigned to a name, or printed. A literal buried in an argument to
    `np.array_equal` is a student checking their own working against a value
    from the prompt, which is a different thing entirely, so it is left out.
    """
    found = _typed_on_screen(ctx)
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.Assign):
            array = _literal(node.value)
            if array is not None:
                found.append(array)
    return found


def _worked_arrays(ctx):
    """Variables holding something the submission actually worked out."""
    typed_only = _literal_names(ctx)
    return [(name, array) for name, array in _assigned_arrays(ctx)
            if name not in typed_only]


def _typed(ctx, want, *inputs):
    """Is this "answer" *only* ever a number the student typed out?

    Three ways out of it, and each one is a student who did the work. The
    exercises' own inputs are typed out — that is how you make a matrix — so
    anything matching one of them is exempt. An answer that reached the screen
    from something other than a typed-out value was printed by a calculation.
    And an answer sitting in a variable they computed is theirs however many
    times they also wrote it down by hand to check against. What is left is a
    submission presenting a result it never worked out, which is the one thing
    these open-ended cells cannot afford to give marks for.
    """
    want = _numeric(want)
    if want is None:
        return False
    if any(_same(_numeric(given), want) for given in inputs if given is not None):
        return False
    if _shows(ctx, want) and not any(_same(array, want)
                                     for array in _typed_on_screen(ctx)):
        return False
    if any(_same(array, want) for _, array in _worked_arrays(ctx)):
        return False
    return any(_same(array, want) for array in _typed_arrays(ctx))


def _computed(ctx, want, worked, *inputs):
    """Did they *work out* this answer, rather than type it in?

    Two things have to hold. The tool the task names has to appear — you cannot
    print A @ B without writing A @ B, so this costs a correct submission
    nothing. And the answer itself must not be one the student typed out:
    without that, a single decorative `@` somewhere in the cell would carry a
    column of pasted-in matrices to full marks.
    """
    return bool(worked) and _answered(ctx, want) and not _typed(ctx, want, *inputs)


def _fmt(array):
    """An array as the student would see it printed."""
    try:
        return np.array2string(np.asarray(array, dtype=float), precision=4,
                               suppress_small=True)
    except Exception:
        return repr(array)


# --- the maths --------------------------------------------------------------

def _is_identity(array):
    try:
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            return False
        return bool(np.allclose(array, np.eye(array.shape[0]), rtol=TOL, atol=TOL))
    except (TypeError, ValueError):
        return False


def _product(matrix, vector):
    """matrix @ vector, or None if those two can't be multiplied."""
    try:
        return np.asarray(matrix @ vector, dtype=float)
    except Exception:
        return None


def _either_way(ctx):
    """A @ x — or x @ A, if that is the one the student went with.

    The prompt says "multiply them" without saying which side the vector goes
    on, and both are honest matrix multiplication, so whichever one their
    output matches is the one to grade. Nothing is lost by being generous
    here: the answer still has to be a real product of their own two arrays.
    """
    def answer(matrix, vector):
        forward = _product(matrix, vector)
        backward = _product(vector, matrix)
        for value in (forward, backward):
            if value is not None and (_shows(ctx, value) or _holds(ctx, value)):
                return value
        return forward if forward is not None else backward
    return answer


def _solution(matrix, vector):
    """The x in `matrix @ x = vector`, or None if there isn't one."""
    try:
        return np.asarray(np.linalg.solve(matrix, vector), dtype=float)
    except Exception:
        return None


def _inverse(matrix):
    try:
        return np.asarray(np.linalg.inv(matrix), dtype=float)
    except Exception:
        return None


def _determinant(matrix):
    try:
        return float(np.linalg.det(matrix))
    except Exception:
        return None


def _calls(ctx, *names):
    """Does the submission call any of these, however numpy was imported?

    Matching on the last part of the dotted name means `np.linalg.solve(...)`,
    `la.solve(...)` and a bare `solve(...)` after `from numpy.linalg import
    solve` all count — the exercise is about reaching for the right tool, not
    about how the import was spelled.
    """
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.Call):
            dotted = _dotted(node.func)
            if dotted and dotted.rsplit(".", 1)[-1] in names:
                return True
    return False


def _matmuls(ctx):
    """How many times the `@` operator appears."""
    return sum(1 for node in ast.walk(ctx.tree)
               if isinstance(node, (ast.BinOp, ast.AugAssign))
               and isinstance(node.op, ast.MatMult))


def _multiplied(ctx):
    """Any matrix multiplication at all, `@` or not.

    The exercises ask for `@` and say so in their own check; this is the wider
    question of whether a product was computed rather than typed in, so
    `np.dot` and `np.matmul` count. A student who used one of those loses the
    `@` row and nothing else.
    """
    return _matmuls(ctx) >= 1 or _calls(ctx, "dot", "matmul")


def _added(ctx):
    """Any addition of two things that could be matrices.

    `np.add(A, B)` counts, and so does any `+` — except one gluing text
    together, since `print("A + B:" + " ")` is punctuation, not arithmetic.
    """
    if _calls(ctx, "add"):
        return True
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            if not any(isinstance(side, ast.Constant) and isinstance(side.value, (str, bytes))
                       for side in (node.left, node.right)):
                return True
    return False


def _pair(ctx, matrices, vectors, answer):
    """Which array is "the matrix" and which is "the vector".

    A submission can leave several arrays behind — the matrix, the vector, the
    result, a stray experiment — and nothing in the code says which is which.
    So try every combination and keep the one whose answer the student actually
    printed (or at least stored), preferring a matrix that isn't the identity.
    A matrix whose name says "inverse" is ranked last however well it scores —
    it is the answer to half the exercise, not the A the exercise started from,
    and mistaking the two makes every message below point at the wrong array.
    With nothing to go on this falls back to the first of each, so the feedback
    still points at something the student recognises. Only the first few of
    each are tried: these cells make two or three arrays, and every extra one
    costs a re-read of the whole output.
    """
    best = best_key = None
    for m_name, matrix in matrices[:MAX_CANDIDATES]:
        for v_name, vector in vectors[:MAX_CANDIDATES]:
            value = answer(matrix, vector)
            found = 0 if value is None else 2 if _shows(ctx, value) else \
                1 if _holds(ctx, value) else 0
            key = (0 if m_name and INVERSE_NAME.search(m_name) else 1,
                   found, 0 if _is_identity(matrix) else 1)
            if best_key is None or key > best_key:
                best_key, best = key, (m_name, matrix, v_name, vector, value)
    return best


def _shape_checks(ctx, matrix, vector):
    """The two "you were asked for 2 by 2" rows, shared by both Your Turns."""
    ctx.require("The matrix is 2 by 2", matrix.shape == (2, 2),
                "The exercise asks for two rows of two numbers each — "
                "`print(A.shape)` should say (2, 2).",
                expected="(2, 2)", got=str(matrix.shape))
    ctx.require("The vector has 2 elements", vector.size == 2,
                "A 2 by 2 matrix can only multiply a vector with one entry per "
                "column, so this one needs exactly two numbers.",
                expected="2 numbers", got=f"{vector.size}")


# --- the exercises ----------------------------------------------------------

def check_ex1(ctx):
    """"Create a 2 by 2 matrix and a 2 element vector, multiply with `@`."

    The numbers are the student's own, so all that can be verified is that the
    answer they printed really is their matrix times their vector — and that
    they got there with `@`, which is the whole point of the cell. Copying the
    worked example's numbers is fine: the prompt never asks for new ones.
    """
    if not _ran(ctx):
        return

    arrays = _arrays(ctx)
    matrices, vectors = _matrices(arrays), _vectors(arrays)

    if not ctx.require("Creates a matrix", matrices,
                       "A matrix is a list of rows, so it needs a bracket around "
                       "each one: A = np.array([[1, 2], [3, 4]])."):
        return
    if not ctx.require("Creates a vector", vectors,
                       "A vector is a flat list of numbers: x = np.array([5, 6])."):
        return

    name, matrix, _, vector, product = _pair(ctx, matrices, vectors, _either_way(ctx))
    _shape_checks(ctx, matrix, vector)

    ctx.require("Multiplies with the `@` operator", _matmuls(ctx) >= 1,
                "`*` multiplies element by element, which is a different sum "
                "entirely. Matrix multiplication is `@`: answer = A @ x.")

    if not ctx.require("The matrix and the vector fit together", product is not None,
                       "A @ x only works when the number of columns in A matches "
                       "the number of entries in x — (2x2) @ (2,) gives a "
                       "2-element vector back.",
                       expected="a 2 by 2 matrix and a 2-element vector",
                       got=f"{'`%s`' % name if name else 'your matrix'} is "
                           f"{matrix.shape}, your vector is {vector.shape}"):
        return

    ctx.require("Works out the matrix times the vector",
                _computed(ctx, product, _multiplied(ctx), matrix, vector),
                "Each row of the matrix gets dotted with the vector — row 1 "
                "times x, then row 2 times x. Let numpy do it: answer = A @ x.",
                expected=_fmt(product), got=ctx.stdout.strip() or "(nothing printed)")
    ctx.require("Prints the answer", _shows(ctx, product),
                "Print the result so you can see it: print(answer).",
                expected=_fmt(product), got=ctx.stdout.strip() or "(nothing printed)")


def check_ex2(ctx):
    """"Find A inverse, multiply A-inverse by b, print the answer."

    Again the numbers are the student's, so the inverse is checked by what it
    does rather than by what it is: multiplied back into A it has to give the
    identity, and the answer has to agree with `np.linalg.solve(A, b)`, which
    reaches the same place without ever forming an inverse.
    """
    if not _ran(ctx):
        return

    arrays = _arrays(ctx)
    matrices, vectors = _matrices(arrays), _vectors(arrays)

    if not ctx.require("Creates a matrix `A`", matrices,
                       "A matrix is a list of rows, so it needs a bracket around "
                       "each one: A = np.array([[4, 7], [2, 6]])."):
        return
    if not ctx.require("Creates a vector `b`", vectors,
                       "A vector is a flat list of numbers: b = np.array([1, 2])."):
        return

    name, matrix, _, vector, answer = _pair(ctx, matrices, vectors, _solution)
    label = f"`{name}`" if name else "your matrix"
    _shape_checks(ctx, matrix, vector)

    ctx.require("The matrix isn't the identity matrix", not _is_identity(matrix),
                "The identity matrix is its own inverse, so this exercise would "
                "hand you b straight back. Put some numbers off the diagonal, "
                "e.g. np.array([[4, 7], [2, 6]]).")

    determinant = _determinant(matrix)
    if not ctx.require(f"{label} can be inverted",
                       determinant is not None and abs(determinant) > TOL,
                       "Only a square matrix has an inverse, and only one whose "
                       "determinant isn't zero — if one row is a multiple of "
                       "another there is nothing to undo. Nudge one of the numbers.",
                       expected="a square matrix with a non-zero determinant",
                       got=f"{matrix.shape} matrix, determinant "
                           + ("undefined" if determinant is None else f"{determinant:g}")):
        return

    inverse = _inverse(matrix)
    claimed = _claimed_inverse(ctx, matrices, matrix, inverse)
    if claimed is None:
        ctx.require("Finds the inverse of A", _calls(ctx, "inv", "pinv"),
                    "numpy will do it for you — store it in a variable so you can "
                    "look at it: A_inv = np.linalg.inv(A).")
    else:
        claimed_name, claimed_array = claimed
        claimed_label = f"`{claimed_name}`" if claimed_name else "the one you made"
        ctx.require("Finds the inverse of A",
                    _same(_product(matrix, claimed_array), np.eye(matrix.shape[0])),
                    f"A matrix times its inverse has to come out as the identity "
                    f"matrix, and {label} @ {claimed_label} doesn't. Watch out for "
                    "1/A, which divides element by element and is not an inverse — "
                    "use np.linalg.inv(A).",
                    expected=_fmt(inverse), got=_fmt(claimed_array))

    if not ctx.require("A and b fit together", answer is not None,
                       "For A_inv @ b to work, b needs one entry per row of A.",
                       expected="a 2 by 2 matrix and a 2-element vector",
                       got=f"{label} is {matrix.shape}, your vector is {vector.shape}"):
        return

    ctx.require("Multiplies the inverse by b",
                _computed(ctx, answer, _multiplied(ctx), matrix, vector),
                "x = A_inv @ b. Remember `@` and not `*` — and mind the order, "
                "the inverse goes on the left.",
                expected=_fmt(answer), got=ctx.stdout.strip() or "(nothing printed)")
    ctx.require("Prints the answer", _shows(ctx, answer),
                "Print the result so you can see it: print(x).",
                expected=_fmt(answer), got=ctx.stdout.strip() or "(nothing printed)")


def _claimed_inverse(ctx, matrices, matrix, inverse):
    """The array the student is treating as A's inverse, if there is one.

    A name with "inv" in it is taken at its word — that is how a wrongly worked
    out inverse gets caught where it happened, rather than three checks later.
    Failing that, any matrix they made that genuinely is the inverse counts,
    whatever they chose to call it.
    """
    for name, array in matrices:
        if name and INVERSE_NAME.search(name):
            return name, array
    if inverse is not None:
        for name, array in matrices:
            if _same(array, inverse):
                return name, array
    return None


def check_matrix_boss(ctx):
    """The Final Matrix Boss (pinned with `#% checker: matrix_boss`).

    A, B and b are given, so every answer is known and each task gets its own
    row: shape, sum, both products, transpose, the solve, and the answer to
    "are AB and BA the same?". A task the prompt words as "compute" is happy
    with a variable holding the right numbers; the one it words as "print" has
    to reach the output. The tools are checked as well as the answers — `@` for
    the products, `.T` for the transpose, `np.linalg.solve` for the system —
    because otherwise a cell of hand-typed answers would score full marks, and
    the point of the challenge is doing it with numpy.
    """
    if not _ran(ctx):
        return
    ctx.require("Prints its results", bool(ctx.lines),
                "Nothing was printed. Print each answer as you work it out — "
                "that's how you see what the matrices did.")

    given_a = np.asarray(GIVEN_A, dtype=float)
    given_b = np.asarray(GIVEN_B, dtype=float)
    given_vector = np.asarray(GIVEN_VECTOR, dtype=float)

    givens = (given_a, given_b, given_vector)
    named = (("A", given_a), ("B", given_b), ("b", given_vector))
    ctx.require("Uses the A, B and b the cell gives you",
                all(_unchanged(ctx, key, value) for key, value in named),
                       "Leave those three lines as they are and build new "
                       "variables out of them (AB = A @ B, and so on) — the "
                       "answers below are worked out from the given numbers.",
                expected=f"A =\n{_fmt(given_a)}\nB =\n{_fmt(given_b)}\n"
                         f"b = {_fmt(given_vector)}",
                got=f"A =\n{_fmt(ctx.env.get('A'))}\n"
                    f"B =\n{_fmt(ctx.env.get('B'))}\n"
                    f"b = {_fmt(ctx.env.get('b'))}")

    total = given_a + given_b
    ab = given_a @ given_b
    ba = given_b @ given_a
    transpose = given_a.T
    solution = np.linalg.solve(given_a, given_vector)

    ctx.require("Prints the shape of A", _shape_printed(ctx),
                "`A.shape` gives the number of rows and columns as a pair — "
                "print(A.shape). Shapes are the first thing to check when a "
                "matrix multiplication refuses to work.",
                expected="(2, 2)", got=ctx.stdout.strip() or "(nothing printed)")

    ctx.require("Computes A + B", _computed(ctx, total, _added(ctx), *givens),
                "Matrix addition is element by element, and numpy does it with a "
                "plain +: print(A + B).",
                expected=_fmt(total))
    ctx.require("Computes AB", _computed(ctx, ab, _multiplied(ctx), *givens),
                "AB is A @ B — each row of A dotted with each column of B.",
                expected=_fmt(ab))
    ctx.require("Computes BA", _computed(ctx, ba, _multiplied(ctx), *givens),
                "BA is B @ A. Swapping the order is a different sum, which is the "
                "whole reason this task exists.",
                expected=_fmt(ba))
    ctx.require("Multiplies the matrices with `@`", _matmuls(ctx) >= 1,
                "The products want the matrix multiplication operator: "
                "AB = A @ B and BA = B @ A. `*` would multiply element by "
                "element, and typing the answers in by hand isn't the exercise.",
                expected="A @ B and B @ A", got=f"{_matmuls(ctx)} use(s) of `@`")

    ctx.require("Computes A transpose",
                _computed(ctx, transpose, _uses_transpose(ctx), *givens),
                "The transpose flips rows into columns: A.T.",
                expected=_fmt(transpose))
    ctx.require("Uses `.T` for the transpose", _uses_transpose(ctx),
                "numpy has it built in — A.T (or np.transpose(A)) — so you never "
                "have to retype a matrix sideways.")

    ctx.require("Solves Ax = b with `np.linalg.solve`", _calls(ctx, "solve"),
                "Use x = np.linalg.solve(A, b). It gets the same answer as "
                "inverting A and multiplying, but faster and with less rounding "
                "error, which is why flight code solves rather than inverts.")
    ctx.require("The solution to Ax = b is right",
                _computed(ctx, solution, _calls(ctx, "solve", "inv", "pinv"), *givens),
                "Check the order of the arguments — np.linalg.solve(A, b) solves "
                "A x = b, so A comes first.",
                expected=_fmt(solution), got=ctx.stdout.strip() or "(nothing printed)")

    ctx.require("Answers whether AB and BA are the same", _compared(ctx),
                "The last task is to compare them: print(AB == BA), or "
                "print(np.array_equal(AB, BA)). They aren't equal — that is what "
                "\"matrix multiplication is not commutative\" means, and it is why "
                "the order of rotations matters on a rocket.",
                expected="a printed comparison of AB and BA")


def _unchanged(ctx, name, value):
    """Is the cell's own `A` (or `B`, or `b`) still the one it was handed?

    Deleting the given lines doesn't hide the problem: the notebook's earlier
    examples define an `A` and a `b` too, and those are replayed as setup, so
    the name would still be there holding the wrong numbers.
    """
    return _same(_numeric(ctx.env.get(name)), value) if name in ctx.env else False


def _shape_printed(ctx):
    """Did (2, 2) reach the output, as a pair or as two separate numbers?"""
    if SHAPE_2X2.search(ctx.stdout):
        return True
    return any(numbers == [2.0, 2.0] for numbers in _bare_lines(ctx))


def _uses_transpose(ctx):
    if _calls(ctx, "transpose"):
        return True
    return any(isinstance(node, ast.Attribute) and node.attr in ("T", "transpose")
               for node in ast.walk(ctx.tree))


def _compared(ctx):
    """Did they actually answer "are AB and BA the same?"

    A comparison in the code counts even if the wording of the printed answer
    is their own, and a printed answer counts even if they worked it out by
    eye — `print(AB == BA)` puts "False" in the output either way.
    """
    if DIFFERENT_WORDS.search(ctx.stdout):
        return True
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.Compare) and any(
                isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
            return True
    return _calls(ctx, "array_equal", "array_equiv", "allclose", "equal")
