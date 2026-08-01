"""ISS GNC Onboarding engine.

A tiny, self-contained "mini-PrairieLearn": it turns a lead-authored Jupyter
notebook into an interactive, auto-graded onboarding module. Pure Python, no
external service required.

Pipeline:
    notebooks/*.ipynb  --parser-->  Module  --renderer-->  interactive HTML
                                       |
                                       +--grader--> per-exercise feedback + score
"""

from .parser import Module, Block, Exercise, Check, parse_notebook, discover_modules

__all__ = [
    "Module",
    "Block",
    "Exercise",
    "Check",
    "parse_notebook",
    "discover_modules",
]
