#!/usr/bin/env python3
"""Soft character-specific structural priors for ambiguity resolution.

These priors do NOT describe glyph geometry or stroke paths. They record
the structural features that a well-formed instance of each character is
expected to have: number of strokes, closed loops, a dominant vertical
spine, a horizontal bar, descenders, detached marks, approximate vertical
position of any loops.

The ambiguity resolver uses them only to break ties between otherwise-
equivalent decompositions. Any character not in the table falls back to a
permissive DEFAULT_PRIOR (geometry decides).

Guidelines:
  - strokes/loops are inclusive ranges; writers legitimately produce
    different counts (e.g. some people write '4' in one stroke, others
    in two). A range captures that.
  - Boolean features use Optional[bool]: True/False = prior expects,
    None = don't care.
  - loop_positions holds vertical centroid targets in [0, 1] (0 = top,
    1 = bottom). Mismatch is lightly penalized; geometric spread
    between handwritten samples is expected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CharPrior:
    strokes: tuple[int, int] = (1, 6)
    loops: tuple[int, int] = (0, 3)
    spine: Optional[bool] = None             # long dominant vertical stroke
    bar: Optional[bool] = None               # long dominant horizontal stroke
    descender: Optional[bool] = None         # pen dips into bottom ~12% of bbox
    detached_mark: Optional[bool] = None     # small disconnected mark (i/j dot)
    loop_positions: Optional[list[float]] = None  # vertical centers in [0,1]


DEFAULT_PRIOR = CharPrior()


_PRIORS: dict[str, CharPrior] = {
    # ── Digits ───────────────────────────────────────────────────────────
    "0": CharPrior(strokes=(1, 1), loops=(1, 1), spine=False, bar=False, descender=False),
    "1": CharPrior(strokes=(1, 3), loops=(0, 0), spine=True, bar=False, descender=False),
    "2": CharPrior(strokes=(1, 1), loops=(0, 0), spine=False, bar=False, descender=False),
    "3": CharPrior(strokes=(1, 1), loops=(0, 0), spine=False, bar=False, descender=False),
    "4": CharPrior(strokes=(2, 3), loops=(0, 0), spine=True, bar=True, descender=False),
    "5": CharPrior(strokes=(2, 3), loops=(0, 0), spine=False, bar=True, descender=False),
    "6": CharPrior(strokes=(1, 1), loops=(1, 1), spine=False, bar=False, descender=False,
                   loop_positions=[0.70]),
    "7": CharPrior(strokes=(1, 2), loops=(0, 0), spine=False, bar=True, descender=False),
    "8": CharPrior(strokes=(1, 1), loops=(2, 2), spine=False, bar=False, descender=False,
                   loop_positions=[0.30, 0.70]),
    "9": CharPrior(strokes=(1, 2), loops=(1, 1), spine=False, bar=False, descender=True,
                   loop_positions=[0.30]),

    # ── Lowercase ────────────────────────────────────────────────────────
    "a": CharPrior(strokes=(1, 2), loops=(1, 1), spine=True, bar=False, descender=False,
                   loop_positions=[0.65]),
    "b": CharPrior(strokes=(1, 2), loops=(1, 1), spine=True, bar=False, descender=False,
                   loop_positions=[0.70]),
    "c": CharPrior(strokes=(1, 1), loops=(0, 0), spine=False, bar=False, descender=False),
    "d": CharPrior(strokes=(1, 2), loops=(1, 1), spine=True, bar=False, descender=False,
                   loop_positions=[0.65]),
    "e": CharPrior(strokes=(1, 1), loops=(0, 1), spine=False, bar=False, descender=False),
    "f": CharPrior(strokes=(1, 3), loops=(0, 0), spine=True, bar=True, descender=True),
    "g": CharPrior(strokes=(1, 2), loops=(1, 2), spine=False, bar=False, descender=True,
                   loop_positions=[0.40]),
    "h": CharPrior(strokes=(1, 2), loops=(0, 0), spine=True, bar=False, descender=False),
    "i": CharPrior(strokes=(2, 2), loops=(0, 0), spine=True, bar=False, descender=False,
                   detached_mark=True),
    "j": CharPrior(strokes=(2, 2), loops=(0, 0), spine=True, bar=False, descender=True,
                   detached_mark=True),
    "k": CharPrior(strokes=(1, 3), loops=(0, 0), spine=True, bar=False, descender=False),
    "l": CharPrior(strokes=(1, 2), loops=(0, 0), spine=True, bar=False, descender=False),
    "m": CharPrior(strokes=(1, 3), loops=(0, 0), spine=False, bar=False, descender=False),
    "n": CharPrior(strokes=(1, 2), loops=(0, 0), spine=False, bar=False, descender=False),
    "o": CharPrior(strokes=(1, 1), loops=(1, 1), spine=False, bar=False, descender=False),
    "p": CharPrior(strokes=(1, 2), loops=(1, 1), spine=True, bar=False, descender=True,
                   loop_positions=[0.55]),
    "q": CharPrior(strokes=(1, 2), loops=(1, 1), spine=True, bar=False, descender=True,
                   loop_positions=[0.55]),
    "r": CharPrior(strokes=(1, 2), loops=(0, 0), spine=False, bar=False, descender=False),
    "s": CharPrior(strokes=(1, 1), loops=(0, 0), spine=False, bar=False, descender=False),
    "t": CharPrior(strokes=(2, 2), loops=(0, 0), spine=True, bar=True, descender=False),
    "u": CharPrior(strokes=(1, 2), loops=(0, 0), spine=False, bar=False, descender=False),
    "v": CharPrior(strokes=(1, 1), loops=(0, 0), spine=False, bar=False, descender=False),
    "w": CharPrior(strokes=(1, 2), loops=(0, 0), spine=False, bar=False, descender=False),
    "x": CharPrior(strokes=(2, 2), loops=(0, 0), spine=False, bar=False, descender=False),
    "y": CharPrior(strokes=(1, 2), loops=(0, 0), spine=False, bar=False, descender=True),
    "z": CharPrior(strokes=(1, 1), loops=(0, 0), spine=False, bar=False, descender=False),

    # ── Uppercase (common ones; others fall through to default) ──────────
    "A": CharPrior(strokes=(2, 3), loops=(0, 1), spine=False, bar=True, descender=False),
    "B": CharPrior(strokes=(1, 3), loops=(2, 2), spine=True, bar=False, descender=False),
    "C": CharPrior(strokes=(1, 1), loops=(0, 0), spine=False, bar=False, descender=False),
    "D": CharPrior(strokes=(1, 2), loops=(1, 1), spine=True, bar=False, descender=False),
    "E": CharPrior(strokes=(1, 4), loops=(0, 0), spine=True, bar=True, descender=False),
    "F": CharPrior(strokes=(1, 3), loops=(0, 0), spine=True, bar=True, descender=False),
    "G": CharPrior(strokes=(1, 2), loops=(0, 1), spine=False, bar=False, descender=False),
    "H": CharPrior(strokes=(2, 3), loops=(0, 0), spine=True, bar=True, descender=False),
    "I": CharPrior(strokes=(1, 3), loops=(0, 0), spine=True, bar=False, descender=False),
    "J": CharPrior(strokes=(1, 2), loops=(0, 0), spine=True, bar=False, descender=True),
    "K": CharPrior(strokes=(1, 3), loops=(0, 0), spine=True, bar=False, descender=False),
    "L": CharPrior(strokes=(1, 2), loops=(0, 0), spine=True, bar=False, descender=False),
    "M": CharPrior(strokes=(1, 4), loops=(0, 0), spine=True, bar=False, descender=False),
    "N": CharPrior(strokes=(1, 3), loops=(0, 0), spine=True, bar=False, descender=False),
    "O": CharPrior(strokes=(1, 1), loops=(1, 1), spine=False, bar=False, descender=False),
    "P": CharPrior(strokes=(1, 2), loops=(1, 1), spine=True, bar=False, descender=False,
                   loop_positions=[0.30]),
    "Q": CharPrior(strokes=(1, 2), loops=(1, 1), spine=False, bar=False, descender=False),
    "R": CharPrior(strokes=(1, 3), loops=(1, 1), spine=True, bar=False, descender=False,
                   loop_positions=[0.30]),
    "S": CharPrior(strokes=(1, 1), loops=(0, 0), spine=False, bar=False, descender=False),
    "T": CharPrior(strokes=(2, 2), loops=(0, 0), spine=True, bar=True, descender=False),
    "U": CharPrior(strokes=(1, 1), loops=(0, 0), spine=False, bar=False, descender=False),
    "V": CharPrior(strokes=(1, 1), loops=(0, 0), spine=False, bar=False, descender=False),
    "W": CharPrior(strokes=(1, 2), loops=(0, 0), spine=False, bar=False, descender=False),
    "X": CharPrior(strokes=(2, 2), loops=(0, 0), spine=False, bar=False, descender=False),
    "Y": CharPrior(strokes=(1, 3), loops=(0, 0), spine=True, bar=False, descender=False),
    "Z": CharPrior(strokes=(1, 1), loops=(0, 0), spine=False, bar=False, descender=False),

    # ── Symbols ──────────────────────────────────────────────────────────
    "$": CharPrior(strokes=(2, 3), loops=(0, 1), spine=True, bar=False, descender=True),
    ".": CharPrior(strokes=(1, 1), loops=(0, 0), spine=False, bar=False, descender=False),
    ",": CharPrior(strokes=(1, 1), loops=(0, 0), spine=False, bar=False, descender=True),
    "!": CharPrior(strokes=(2, 2), loops=(0, 0), spine=True, bar=False, descender=False),
    "?": CharPrior(strokes=(1, 2), loops=(0, 0), spine=False, bar=False, descender=False),
    "(": CharPrior(strokes=(1, 1), loops=(0, 0), spine=False, bar=False, descender=False),
    ")": CharPrior(strokes=(1, 1), loops=(0, 0), spine=False, bar=False, descender=False),
    "+": CharPrior(strokes=(2, 2), loops=(0, 0), spine=True, bar=True, descender=False),
    "-": CharPrior(strokes=(1, 1), loops=(0, 0), spine=False, bar=True, descender=False),
    "=": CharPrior(strokes=(2, 2), loops=(0, 0), spine=False, bar=True, descender=False),
}


def get_prior(char: Optional[str]) -> CharPrior:
    if not char or len(char) > 1:
        return DEFAULT_PRIOR
    return _PRIORS.get(char, DEFAULT_PRIOR)
