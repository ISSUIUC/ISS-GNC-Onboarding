"""Turn a Module into render-ready HTML for the browser.

Two jobs that need care:

* **Markdown with LaTeX.** The teaching notebooks are full of `$...$` and bare
  `\\begin{bmatrix}...\\end{bmatrix}`. If we hand that straight to the markdown
  processor it mangles the math (underscores become emphasis, backslashes get
  eaten). So we *stash* every math span behind a placeholder, run markdown, then
  restore the raw math for MathJax to typeset in the browser.

* **Code highlighting.** Worked-example cells are shown read-only with Pygments
  highlighting; exercise cells become editors on the client side.
"""

from __future__ import annotations

import re

import markdown as md_lib
from pygments import highlight as _pyg_highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

from .parser import Module

_PYGMENTS_FORMATTER = HtmlFormatter(nowrap=True)
_LEXER = PythonLexer()

# Math patterns, extracted in this order. Each entry is (regex, wrap?) where
# `wrap` means the match is a *bare* environment that we re-delimit as inline
# math so MathJax always typesets it in math mode.
_MATH_PATTERNS = [
    (re.compile(r"\$\$.*?\$\$", re.DOTALL), False),
    (re.compile(r"\\\[.*?\\\]", re.DOTALL), False),
    (re.compile(r"\\\(.*?\\\)", re.DOTALL), False),
    (re.compile(r"\$[^$\n]+?\$"), False),
    (re.compile(r"\\begin\{[a-zA-Z*]+\}.*?\\end\{[a-zA-Z*]+\}", re.DOTALL), True),
]

_MD = md_lib.Markdown(
    # nl2br because notebook authors write single newlines expecting the hard
    # break Jupyter gives them ("Time: 1 s" / "Altitude: 50 m" on two lines).
    extensions=["fenced_code", "tables", "sane_lists", "nl2br", "codehilite"],
    extension_configs={"codehilite": {"guess_lang": False, "css_class": "highlight"}},
)

_LIST_ITEM = re.compile(r"^ {0,3}(?:[-*+]|\d{1,9}[.)])\s+\S")
_FENCE = re.compile(r"^ {0,3}(?:```|~~~)")


def _space_out_lists(text: str) -> str:
    """Insert the blank line Python-Markdown needs before a list.

    CommonMark — and so the notebook editor the cells were written in — lets a
    bullet list interrupt a paragraph. Python-Markdown does not, and swallows
    the items into the paragraph, so the bullets collapse onto one line.
    """
    out: list[str] = []
    in_fence = in_list = False
    prev_blank = True
    for line in text.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
        if in_fence:
            out.append(line)
            continue
        if not line.strip():
            prev_blank = True
            out.append(line)
            continue
        is_item = bool(_LIST_ITEM.match(line))
        indented = line.startswith("  ")
        # A blank line then unindented prose ends the list; blank lines *inside*
        # a list (loose items, continuation paragraphs) must not.
        if in_list and prev_blank and not is_item and not indented:
            in_list = False
        if is_item and not in_list and not prev_blank:
            out.append("")
        in_list = in_list or is_item
        prev_blank = False
        out.append(line)
    return "\n".join(out)


def render_markdown(source: str) -> str:
    stash: list[str] = []

    def protect(text: str) -> str:
        for pattern, wrap in _MATH_PATTERNS:
            def repl(m: re.Match) -> str:
                raw = m.group(0)
                if wrap:
                    raw = r"\(" + raw + r"\)"
                stash.append(raw)
                # Alphanumeric token markdown will leave untouched.
                return f" mathstash{len(stash) - 1}z "
            text = pattern.sub(repl, text)
        return text

    # After protect(), so a multi-line math block can't be mistaken for prose
    # followed by a list.
    protected = _space_out_lists(protect(source))
    _MD.reset()
    html = _MD.convert(protected)

    def restore(m: re.Match) -> str:
        return stash[int(m.group(1))]

    return re.sub(r"mathstash(\d+)z", restore, html)


def highlight(code: str) -> str:
    return _pyg_highlight(code, _LEXER, _PYGMENTS_FORMATTER).rstrip("\n")


def pygments_css() -> str:
    """Token colours for both themes, scoped to the <html data-theme> attribute."""
    light = HtmlFormatter(style="gruvbox-light").get_style_defs(
        ':root[data-theme="light"] .highlight'
    )
    dark = HtmlFormatter(style="nord").get_style_defs(
        ':root[data-theme="dark"] .highlight'
    )
    return f"{light}\n{dark}"


def render_blocks(module: Module) -> list[dict]:
    """Flatten a module into view blocks the template can loop over."""
    views: list[dict] = []
    for block in module.blocks:
        if block.kind == "markdown":
            views.append({"kind": "markdown", "html": render_markdown(block.source)})
        elif block.kind == "code":
            views.append(
                {
                    "kind": "code",
                    "html": highlight(block.source),
                    "output": block.outputs,
                }
            )
        elif block.kind == "exercise":
            views.append({"kind": "exercise", "exercise": block.exercise})
    return views
