#!/usr/bin/env python3
"""Character-specific scoring rules.

These scorers run IN ADDITION to the generic CharPrior scoring in
`char_priors.py`. Each takes a StructuralFeatures and returns a soft
penalty (>= 0). Lower = better match to the character's expected
structural signature.

They encode fine-grained distinctions that the generic spine/bar/desc
flags can't capture on their own:

    B  vs 8   — spine + right-attached bowls vs independent stacked loops
    P  vs R   — upper bowl only vs bowl + diagonal leg
    D  vs O   — spine + right bowl vs centered standalone loop
    O  vs 0   — same core shape, 0 slightly more elongated
    Q         — loop + short lower-right tail

The `CONFUSABLES` map is used by the resolver to add a **discrimination**
penalty: if a decomposition scores strictly better under a confusable
character's rules than under its own, that candidate is penalized. This
prevents the winner from "looking more like" a related character.

Keep these rules soft. A candidate that loses on one or two features but
is geometrically dominant still wins — these penalties are a few units
each, against a geometry cost measured in degrees.
"""

from __future__ import annotations

from typing import Callable

from ambiguity import StructuralFeatures


CustomScorer = Callable[[StructuralFeatures], float]


# Score of 0 = ideal match. Every rule mismatch adds a small penalty. The
# magnitudes are chosen to be comparable to generic CharPrior penalties
# (0.4-1.2 per feature).

def _score_B(f: StructuralFeatures) -> float:
    p = 0.0
    if not f.has_vertical_spine:
        p += 0.9                         # B needs the left backbone
    if f.n_right_bulges < 1:
        p += 1.0                         # need at least one attached bowl
    elif f.n_right_bulges < 2:
        p += 0.4                         # prefer two, but one is OK
    # Penalize two standalone loops with no spine — that's an 8 shape.
    if f.n_closed_loops >= 2 and not f.has_vertical_spine:
        p += 1.4
    # Penalize strongly-balanced central lobes with no spine (8-like).
    if f.loop_y_spread >= 0.35 and not f.has_vertical_spine:
        p += 0.6
    return p


def _score_8(f: StructuralFeatures) -> float:
    p = 0.0
    # 8 has no dominant side spine — it's symmetric around a vertical axis
    # but without a true vertical stroke as a backbone.
    if f.has_vertical_spine:
        p += 0.9
    if f.n_closed_loops < 2:
        p += 0.8                         # strongly prefer 2 closed lobes
    # Reward loops that are stacked (vertically separated ~ half the glyph).
    if f.loop_y_spread > 0:
        # ideal spread ~0.40; penalize deviation (loops too close OR too far).
        p += min(0.8, abs(f.loop_y_spread - 0.40) * 1.5)
    elif f.n_closed_loops >= 2:
        p += 0.3
    return p


def _score_P(f: StructuralFeatures) -> float:
    p = 0.0
    if not f.has_vertical_spine:
        p += 0.9
    if f.n_right_bulges < 1:
        p += 0.9                         # P needs the one upper bulge
    # No lower bulge — if there are two right-side bulges, that's a B.
    if f.n_right_bulges >= 2:
        p += 0.9
    # No diagonal leg — if present, it's an R.
    if f.has_diagonal_leg:
        p += 1.0
    # No second closed loop below.
    lower_loops = sum(1 for y in f.loop_vertical_positions if y >= 0.55)
    if lower_loops >= 1:
        p += 0.6
    return p


def _score_R(f: StructuralFeatures) -> float:
    p = 0.0
    if not f.has_vertical_spine:
        p += 0.9
    if f.n_right_bulges < 1:
        p += 0.6                         # want the upper bowl, but soft
    # R's distinguishing feature vs P: the diagonal leg.
    if not f.has_diagonal_leg:
        p += 1.0
    # Penalize a second closed bowl stacked below (that's B).
    lower_loops = sum(1 for y in f.loop_vertical_positions if y >= 0.55)
    if lower_loops >= 1:
        p += 0.7
    return p


def _score_D(f: StructuralFeatures) -> float:
    p = 0.0
    if not f.has_vertical_spine:
        p += 1.0                         # D without a spine reads as O
    if f.n_right_bulges < 1:
        p += 0.7
    # Penalize strongly-centered standalone loops when no spine (O-like).
    if f.n_closed_loops >= 1 and not f.has_vertical_spine:
        for cx in f.loop_horizontal_positions:
            if 0.35 <= cx <= 0.65:
                p += 0.8                 # centered loop + no spine = O
                break
    return p


def _score_O(f: StructuralFeatures) -> float:
    p = 0.0
    if f.has_vertical_spine:
        p += 0.9                         # O should not have a backbone
    if f.n_closed_loops != 1:
        p += 0.6
    if f.has_descender:
        p += 0.4                         # tails are atypical
    if f.has_diagonal_leg:
        p += 0.8                         # legs are atypical
    return p


def _score_0(f: StructuralFeatures) -> float:
    p = 0.0
    # 0 and O share the same rules; 0 is slightly more elongated, but we
    # don't enforce an aspect-ratio penalty here — it's a stylistic choice.
    if f.n_closed_loops != 1:
        p += 0.6
    if f.has_diagonal_leg:
        p += 0.8
    if f.has_vertical_spine:
        p += 0.6                         # 0 rarely has a straight backbone
    return p


def _score_Q(f: StructuralFeatures) -> float:
    p = 0.0
    if f.n_closed_loops != 1:
        p += 0.6
    # Q's distinguishing feature: the short lower-right tail.
    if not f.tail_lower_right:
        p += 0.8
    # Penalize a true long descender (that reads more like g / y).
    if f.has_descender and not f.tail_lower_right:
        p += 0.3
    return p


CUSTOM_SCORERS: dict[str, CustomScorer] = {
    "B": _score_B,
    "8": _score_8,
    "P": _score_P,
    "R": _score_R,
    "D": _score_D,
    "O": _score_O,
    "0": _score_0,
    "Q": _score_Q,
}


# Per-target discrimination: when scoring a candidate for target T, also
# score it under each confusable C. If the candidate looks strictly more
# like C than T, penalize T. This makes the resolver prefer decompositions
# that fit the recognized character better than their confusable neighbors.
CONFUSABLES: dict[str, list[str]] = {
    "B": ["8"],
    "8": ["B"],
    "P": ["R", "B"],
    "R": ["P", "B"],
    "D": ["O", "0"],
    "O": ["D", "0", "Q"],
    "0": ["O", "Q"],
    "Q": ["O", "0"],
}


def score_custom(char: str | None, features: StructuralFeatures) -> float:
    if not char:
        return 0.0
    fn = CUSTOM_SCORERS.get(char)
    return fn(features) if fn else 0.0


def score_discrimination(char: str | None, features: StructuralFeatures) -> float:
    """Add penalty if the decomposition looks more like a confusable than *char*.

    Returns max(0, min_confusable_penalty - target_penalty), so the extra
    penalty only kicks in when a confusable scores STRICTLY better.
    """
    if not char:
        return 0.0
    rivals = CONFUSABLES.get(char)
    if not rivals:
        return 0.0
    target_pen = score_custom(char, features)
    best_rival = min(
        (score_custom(r, features) for r in rivals if r in CUSTOM_SCORERS),
        default=None,
    )
    if best_rival is None:
        return 0.0
    if best_rival < target_pen:
        return target_pen - best_rival     # magnitude of "fits rival better"
    return 0.0
