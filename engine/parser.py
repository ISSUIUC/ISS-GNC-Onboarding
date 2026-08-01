"""Parse a lead-authored .ipynb into a Module the engine can render + grade.

The authoring convention (documented fully in AUTHORING.md) is designed so the
GNC lead barely changes how they already write notebooks:

* Markdown cells                -> teaching content (rendered with LaTeX).
* Plain code cells              -> worked examples (shown read-only, and used as
                                   shared "setup" context for later exercises).
* Code cells marked as an       -> an interactive editor the student fills in.
  exercise

A code cell becomes an *exercise* when any of these are true:
    - it contains a `#% exercise` directive, or
    - the markdown cell right before it mentions "Your Turn" or "Challenge", or
    - it contains a "write ... here" placeholder comment.

An exercise becomes *graded* (auto-checked) when the lead adds:
    #% check: name1, name2~0.001        # variables to compare against reference
    ### BEGIN SOLUTION
    name1 = ...                         # the reference answer (hidden from student)
    ### END SOLUTION

Without a solution block an exercise is still fully usable: it renders as a
"practice" cell the student can run, just with no pass/fail. This lets existing
notebooks work untouched, and get grading later, one exercise at a time.

Directives (lines beginning with `#%`, stripped from what the student sees):
    #% exercise                         mark this cell as an exercise
    #% id: some-slug                    stable id (default: module-ex<N>)
    #% title: Compute the products      short title (default: nearby heading)
    #% check: a, b~1e-6, c              variables to grade (optional abs tol via ~)
    #% check_output                     also require stdout to match reference
    #% check_output_contains: Liftoff!  require substrings in stdout (comma list)
    #% points: 2                        weight for this exercise (default 1)
    #% reveal: false                    hide expected values in feedback
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


SOLUTION_BEGIN = re.compile(r"^\s*###\s*BEGIN\s+SOLUTION\s*$", re.IGNORECASE)
SOLUTION_END = re.compile(r"^\s*###\s*END\s+SOLUTION\s*$", re.IGNORECASE)
STUB_BEGIN = re.compile(r"^\s*###\s*BEGIN\s+STUB\s*$", re.IGNORECASE)
STUB_END = re.compile(r"^\s*###\s*END\s+STUB\s*$", re.IGNORECASE)
DIRECTIVE = re.compile(r"^\s*#%\s*([a-zA-Z_]+)\s*(?::\s*(.*))?$")
PLACEHOLDER = re.compile(r"write\b.*\bhere", re.IGNORECASE)
HEADING = re.compile(r"^#{1,6}\s*(.*)$")
# strip a leading emoji / symbol run (and following spaces) from a heading
EMOJI_PREFIX = re.compile(r"^[^\w(]+")


@dataclass
class Check:
    """One thing to verify about a student's submission."""

    kind: str  # "vars" | "output" | "output_contains"
    variables: list[tuple[str, float | None]] = field(default_factory=list)
    substrings: list[str] = field(default_factory=list)


@dataclass
class Exercise:
    id: str
    title: str
    stub_code: str  # what the student sees / starts editing
    reference_code: str  # full solution, run to derive expected answers
    checks: list[Check] = field(default_factory=list)
    points: int = 1
    reveal: bool = True
    cell_index: int = 0  # position of this cell in the notebook

    @property
    def graded(self) -> bool:
        return bool(self.checks)


@dataclass
class Block:
    kind: str  # "markdown" | "code" | "exercise"
    # markdown -> raw markdown source; code -> source; exercise -> None
    source: str = ""
    outputs: str = ""  # saved text output for a worked-example code cell
    exercise: Exercise | None = None
    cell_index: int = 0  # position in the notebook, for setup ordering


@dataclass
class Module:
    id: str
    title: str
    path: str
    blocks: list[Block] = field(default_factory=list)

    @property
    def exercises(self) -> list[Exercise]:
        return [b.exercise for b in self.blocks if b.kind == "exercise"]

    def exercise(self, exercise_id: str) -> Exercise | None:
        for ex in self.exercises:
            if ex.id == exercise_id:
                return ex
        return None

    def setup_code(self, before_cell_index: int) -> list[str]:
        """Code to run *before* an exercise, reconstructing notebook state.

        Every code cell that appears earlier in the notebook contributes: plain
        examples contribute their code verbatim; earlier exercises contribute
        their reference solution (so grading each exercise assumes canonical
        state upstream). Executed best-effort by the runner.
        """
        cells: list[str] = []
        for b in self.blocks:
            if b.kind == "code" and b.exercise is None:
                if b.cell_index < before_cell_index:
                    cells.append(b.source)
            elif b.kind == "exercise" and b.exercise is not None:
                if b.exercise.cell_index < before_cell_index:
                    cells.append(b.exercise.reference_code)
        return cells


def _slugify(text: str) -> str:
    slug = re.sub(r"[^\w]+", "-", text.strip().lower()).strip("-")
    return slug or "item"


def _clean_text(text: str) -> str:
    text = EMOJI_PREFIX.sub("", text).strip(" *:#")
    text = re.sub(r"\*+", "", text)
    return text[:80]


def _clean_heading(md: str) -> str:
    """Pull a short human title out of a markdown cell's first heading/line."""
    for line in md.splitlines():
        line = line.strip()
        if not line:
            continue
        m = HEADING.match(line)
        text = _clean_text(m.group(1) if m else line)
        if text:
            return text
    return ""


def _exercise_title(prev_markdown: str, number: int) -> str:
    """Name an exercise from a nearby heading, falling back to `Exercise N`.

    The teaching notebooks head each task with a generic "Your Turn" or a
    descriptive "...Challenge...". We keep the descriptive ones and number the
    rest, so the sidebar reads e.g. "Exercise 3" / "Final Challenge: ...".
    """
    headings = [
        _clean_text(m.group(1))
        for line in prev_markdown.splitlines()
        if (m := HEADING.match(line.strip()))
    ]
    for h in headings:
        if "challenge" in h.lower():
            return h
    return f"Exercise {number}"


def _looks_like_exercise(source: str, prev_markdown: str) -> bool:
    if DIRECTIVE_HAS(source, "exercise"):
        return True
    if PLACEHOLDER.search(source):
        return True
    heading = prev_markdown.lower()
    return "your turn" in heading or "challenge" in heading


def DIRECTIVE_HAS(source: str, key: str) -> bool:
    for line in source.splitlines():
        m = DIRECTIVE.match(line)
        if m and m.group(1).lower() == key.lower():
            return True
    return False


def _parse_directives(source: str) -> dict[str, str]:
    """Collect `#% key: value` directives (last one wins for repeats)."""
    out: dict[str, str] = {}
    for line in source.splitlines():
        m = DIRECTIVE.match(line)
        if m:
            key = m.group(1).lower()
            val = (m.group(2) or "").strip()
            out[key] = val
    return out


def _split_solution(source: str) -> tuple[str, str, bool]:
    """Return (student_stub, reference_code, had_solution).

    Two optional regions carve the cell into a student view and a grading view:

    * `### BEGIN SOLUTION ... ### END SOLUTION` — reference only (the answer;
      hidden from the student).
    * `### BEGIN STUB ... ### END STUB` — student only (buggy/starter code the
      student edits; not run when computing the reference answer).

    Lines outside both regions appear in both. Directive (`#%`) lines are dropped
    from both.
    """
    stub_lines: list[str] = []
    ref_lines: list[str] = []
    in_solution = False
    in_stub = False
    had_solution = False
    for line in source.splitlines():
        if DIRECTIVE.match(line):
            continue  # directives never run and are never shown
        if SOLUTION_BEGIN.match(line):
            in_solution = True
            had_solution = True
            continue
        if SOLUTION_END.match(line):
            in_solution = False
            continue
        if STUB_BEGIN.match(line):
            in_stub = True
            continue
        if STUB_END.match(line):
            in_stub = False
            continue
        if in_solution:
            ref_lines.append(line)  # reference only
        elif in_stub:
            stub_lines.append(line)  # student only
        else:
            ref_lines.append(line)
            stub_lines.append(line)

    stub = "\n".join(stub_lines).rstrip()
    reference = "\n".join(ref_lines).rstrip()
    # If stripping the solution left a truly empty cell, give the student a
    # prompt. (A cell that already has a "# Write ... here" comment is kept.)
    if not stub.strip():
        stub = "# Write your code here"
    return stub, reference, had_solution


def _parse_checks(directives: dict[str, str]) -> list[Check]:
    checks: list[Check] = []
    if "check" in directives:
        variables: list[tuple[str, float | None]] = []
        for token in directives["check"].split(","):
            token = token.strip()
            if not token:
                continue
            if "~" in token:
                name, tol = token.split("~", 1)
                try:
                    variables.append((name.strip(), float(tol)))
                except ValueError:
                    variables.append((name.strip(), None))
            else:
                variables.append((token, None))
        if variables:
            checks.append(Check(kind="vars", variables=variables))
    if "check_output" in directives:
        checks.append(Check(kind="output"))
    if "check_output_contains" in directives:
        subs = [s.strip() for s in directives["check_output_contains"].split(",") if s.strip()]
        if subs:
            checks.append(Check(kind="output_contains", substrings=subs))
    return checks


def _cell_output_text(cell: dict) -> str:
    chunks: list[str] = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            chunks.append("".join(out.get("text", [])))
        elif out.get("output_type") in ("execute_result", "display_data"):
            data = out.get("data", {})
            if "text/plain" in data:
                chunks.append("".join(data["text/plain"]))
        elif out.get("output_type") == "error":
            chunks.append("\n".join(out.get("traceback", [])))
    return "".join(chunks).rstrip()


def parse_notebook(path: str | Path) -> Module:
    path = Path(path)
    nb = json.loads(path.read_text(encoding="utf-8"))
    module_id = _slugify(path.stem)

    blocks: list[Block] = []
    title = path.stem.replace("_", " ")
    title_found = False
    prev_markdown = ""
    ex_counter = 0

    cells = nb.get("cells", [])
    for cell_index, cell in enumerate(cells):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))

        if cell_type == "markdown":
            if not title_found:
                h = _clean_heading(source)
                if h:
                    title = h
                    title_found = True
            block = Block(kind="markdown", source=source)
            block.cell_index = cell_index
            blocks.append(block)
            prev_markdown = source
            continue

        if cell_type != "code":
            continue

        if _looks_like_exercise(source, prev_markdown):
            ex_counter += 1
            directives = _parse_directives(source)
            stub, reference, _ = _split_solution(source)
            checks = _parse_checks(directives)
            ex_id = directives.get("id") or f"{module_id}-ex{ex_counter}"
            ex_title = directives.get("title") or _exercise_title(prev_markdown, ex_counter)
            points = 1
            if directives.get("points"):
                try:
                    points = max(1, int(directives["points"]))
                except ValueError:
                    points = 1
            reveal = directives.get("reveal", "true").strip().lower() not in ("false", "0", "no")
            exercise = Exercise(
                id=ex_id,
                title=ex_title,
                stub_code=stub,
                reference_code=reference,
                checks=checks,
                points=points,
                reveal=reveal,
                cell_index=cell_index,
            )
            block = Block(kind="exercise", exercise=exercise)
            block.cell_index = cell_index
            blocks.append(block)
        else:
            block = Block(
                kind="code",
                source=source.rstrip(),
                outputs=_cell_output_text(cell),
            )
            block.cell_index = cell_index
            blocks.append(block)

        prev_markdown = ""  # consumed

    return Module(id=module_id, title=title, path=str(path), blocks=blocks)


def discover_modules(notebooks_dir: str | Path) -> list[Module]:
    """Parse every notebook in a directory, sorted by filename."""
    notebooks_dir = Path(notebooks_dir)
    modules: list[Module] = []
    for nb_path in sorted(notebooks_dir.glob("*.ipynb")):
        if ".ipynb_checkpoints" in str(nb_path):
            continue
        try:
            modules.append(parse_notebook(nb_path))
        except Exception as exc:  # a broken notebook shouldn't kill the whole app
            print(f"[engine] failed to parse {nb_path}: {exc}")
    return modules
