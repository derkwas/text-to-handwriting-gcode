#!/usr/bin/env python3
"""Per-character stroke-shape profiles.

Where `char_priors.py` describes a character's structural topology
(stroke count, loops, spine/bar/descender flags) for ambiguity
resolution, this module describes the **shape** of each expected
stroke — line vs arc vs s-curve — so the fitter can refit a
skeletonised centreline with the primitive that matches how a human
actually writes that letter.

Design notes
------------
- A `StrokeProfile` lists segments in the canonical writing order,
  connected end-to-end. `cusps_between` is a per-join flag: True means
  a sharp corner (as between the diagonals of 'w'), False means a
  smooth tangent-continuous join (as where a straight entry leads
  into a curve in 'e').
- `tolerant=True` lets the fitter accept extra/missing strokes rather
  than bail out to the generic spline fallback. Use it when people
  legitimately write the character with different stroke counts
  (e.g. '9' as one smooth pen motion vs loop + tail).
- Geometry decides; profiles bias. If the profile-guided fit's
  residual against the generic fit is much worse, the caller should
  fall back rather than force a bad shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Kind = Literal["line", "arc", "s_curve"]


@dataclass(frozen=True)
class Segment:
    kind: Kind


@dataclass(frozen=True)
class StrokeProfile:
    segments: tuple[Segment, ...]
    closed: bool = False
    cusps_between: tuple[bool, ...] = ()


@dataclass(frozen=True)
class CharStrokeProfile:
    strokes: tuple[StrokeProfile, ...]
    tolerant: bool = False


def _single(kind: Kind, closed: bool = False) -> StrokeProfile:
    return StrokeProfile(segments=(Segment(kind),), closed=closed)


def _multi(kinds: list[Kind], cusps: list[bool], closed: bool = False) -> StrokeProfile:
    assert len(cusps) == len(kinds) - 1, "cusps_between has len(segments)-1 entries"
    return StrokeProfile(
        segments=tuple(Segment(k) for k in kinds),
        cusps_between=tuple(cusps),
        closed=closed,
    )


# Phase 1: diagnostic glyphs (o, e, 9, 4, $) + obvious wins (C, w).
# Everything else falls back to the generic smoothing spline.
_PROFILES: dict[str, CharStrokeProfile] = {
    # Single smooth arc, open to the right.
    "C": CharStrokeProfile(strokes=(_single("arc"),)),

    # Closed loop.
    "o": CharStrokeProfile(strokes=(_single("arc", closed=True),)),

    # Four straight diagonals meeting at three sharp cusps. Tolerant
    # because the skeletoniser often splits 'w' into 2+ strokes (e.g.
    # a hairline branch at one cusp) — the longest stroke still gets
    # the 4-line profile treatment; leftovers use the generic fit.
    "w": CharStrokeProfile(
        strokes=(_multi(["line", "line", "line", "line"],
                        cusps=[True, True, True]),),
        tolerant=True,
    ),

    # Horizontal entry stroke flowing smoothly into a loop. Tolerant
    # because some hands split the crossbar from the loop, others do
    # it as one pen motion.
    "e": CharStrokeProfile(
        strokes=(_multi(["line", "arc"], cusps=[False]),),
        tolerant=True,
    ),

    # Two strokes: vertical + diagonal meeting at a cusp, then the
    # horizontal crossbar. Tolerant because block '4' and cursive '4'
    # decompose differently.
    "4": CharStrokeProfile(
        strokes=(
            _multi(["line", "line"], cusps=[True]),
            _single("line"),
        ),
        tolerant=True,
    ),

    # Closed loop (head) + tail arc. Tolerant: some writers do it as
    # one continuous pen motion.
    "9": CharStrokeProfile(
        strokes=(
            _single("arc", closed=True),
            _single("arc"),
        ),
        tolerant=True,
    ),

    # S-curve spine crossed by a vertical bar.
    "$": CharStrokeProfile(
        strokes=(
            _single("s_curve"),
            _single("line"),
        ),
        tolerant=True,
    ),
}


def get_profile(char: str) -> CharStrokeProfile | None:
    """Return a character's stroke profile, or None if not defined."""
    return _PROFILES.get(char)


__all__ = [
    "Segment",
    "StrokeProfile",
    "CharStrokeProfile",
    "get_profile",
]
