#!/usr/bin/env python3
"""Character-aware ambiguity resolution for stroke decompositions.

Slots between graph_cleanup and stroke_fit:

    cleanup -> [resolve] -> fit

Why: step 6 (stroke_decompose) picks junction pairings greedily by local
tangent continuity. For many glyphs the greedy choice is obvious, but for
junctions where two pairings are geometrically close (and ONE of them
matches the recognized character's typical structure) the geometry alone
cannot pick. This module enumerates plausible pairings, extracts a small
set of structural features, and chooses the candidate with the best
combined score: `geom_cost + alpha * prior_penalty`.

Soft, not hard:
  - Priors are a *tiebreaker*, not a template.
  - Clearly-better geometry always wins over prior disagreement (alpha
    is small relative to the cost of a bad pairing).
  - Unknown characters fall through to a permissive prior, giving the
    same result as pure geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Optional

import numpy as np

from char_priors import DEFAULT_PRIOR, CharPrior, get_prior
from stroke_decompose import (
    JunctionPairing,
    SideId,
    Stroke,
    StrokeDecomposition,
    _other_side,
    _tangent_at_side,
    _traverse,
)
from stroke_graph import Edge, StrokeGraph


# ── Candidate generation ─────────────────────────────────────────────────────

# Penalty per unpaired side at a junction. Keeps the total cost on the same
# scale as pair turn angles (degrees).
_UNPAIRED_PENALTY: float = 30.0
# Top-K matchings per junction to keep before the cartesian product.
_MATCHINGS_PER_JUNCTION: int = 3


def _enumerate_matchings(
    sides: list[SideId],
    tangents: dict[SideId, Optional[np.ndarray]],
    max_dot: float,
) -> list[tuple[list[tuple[SideId, SideId, float]], list[SideId]]]:
    """All matchings of `sides` respecting the turn threshold.

    Returns list of (pairs, unpaired). Each pair is (side_a, side_b,
    turn_degrees). A matching may be partial: any side that has no valid
    partner or is deliberately skipped lands in `unpaired`.
    """
    # Pre-filter pair costs by threshold.
    valid: dict[tuple[SideId, SideId], float] = {}
    for i, a in enumerate(sides):
        ta = tangents.get(a)
        if ta is None:
            continue
        for b in sides[i + 1:]:
            tb = tangents.get(b)
            if tb is None:
                continue
            d = float(np.dot(ta, tb))
            if d <= max_dot:
                valid[(a, b)] = d

    results: list[tuple[list[tuple[SideId, SideId, float]], list[SideId]]] = []

    def _turn_deg(dot: float) -> float:
        theta = math.acos(max(-1.0, min(1.0, dot)))
        return math.degrees(math.pi - theta)

    def recurse(
        remaining: list[SideId],
        acc_pairs: list[tuple[SideId, SideId, float]],
        acc_unp: list[SideId],
    ) -> None:
        if not remaining:
            results.append((list(acc_pairs), list(acc_unp)))
            return
        if len(remaining) == 1:
            results.append((list(acc_pairs), list(acc_unp) + [remaining[0]]))
            return
        first = remaining[0]
        for i in range(1, len(remaining)):
            second = remaining[i]
            key = (first, second) if (first, second) in valid else (second, first)
            if key not in valid:
                continue
            turn = _turn_deg(valid[key])
            rest = remaining[1:i] + remaining[i + 1:]
            recurse(rest, acc_pairs + [(first, second, turn)], acc_unp)
        # Also: leave `first` unpaired and continue.
        recurse(remaining[1:], acc_pairs, acc_unp + [first])

    recurse(list(sides), [], [])

    # Dedup: different recursion paths can produce the same set of pairs.
    seen: set[frozenset] = set()
    deduped: list[tuple[list[tuple[SideId, SideId, float]], list[SideId]]] = []
    for pairs, unp in results:
        key = frozenset(frozenset((p[0], p[1])) for p in pairs)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((pairs, unp))
    return deduped


def _matching_cost(
    pairs: list[tuple[SideId, SideId, float]],
    unpaired: list[SideId],
) -> float:
    """Geometric cost in degrees: sum of pair turns + unpaired penalty."""
    return sum(p[2] for p in pairs) + _UNPAIRED_PENALTY * len(unpaired)


def _enumerate_candidates(
    graph: StrokeGraph,
    turn_max_degrees: float,
    max_candidates: int,
) -> list[tuple[list[JunctionPairing], float]]:
    max_dot = -math.cos(math.radians(turn_max_degrees))

    edges_by_id = {e.id: e for e in graph.edges}
    sides_by_node: dict[int, list[SideId]] = {}
    for e in graph.edges:
        if e.start_node is not None:
            sides_by_node.setdefault(e.start_node, []).append((e.id, "a"))
        if e.end_node is not None:
            sides_by_node.setdefault(e.end_node, []).append((e.id, "b"))

    per_junction: list[tuple[int, list]] = []
    for junc in graph.junctions:
        sides = sides_by_node.get(junc.id, [])
        tangents: dict[SideId, Optional[np.ndarray]] = {
            sid: _tangent_at_side(edges_by_id[sid[0]], sid[1]) for sid in sides
        }
        matchings = _enumerate_matchings(sides, tangents, max_dot)
        # Sort by (unpaired ascending, cost ascending): prefer decompositions
        # with more pen-continuations, then smoother turns.
        matchings.sort(
            key=lambda m: (len(m[1]), _matching_cost(m[0], m[1]))
        )
        per_junction.append((junc.id, matchings[:_MATCHINGS_PER_JUNCTION]))

    if not per_junction:
        return [([], 0.0)]

    candidates: list[tuple[list[JunctionPairing], float]] = []
    for combo in product(*(m for _, m in per_junction)):
        total = 0.0
        pairings: list[JunctionPairing] = []
        for (jid, _), (pairs, unp) in zip(per_junction, combo):
            pairings.append(JunctionPairing(node_id=jid, pairs=pairs, unpaired=unp))
            total += _matching_cost(pairs, unp)
        candidates.append((pairings, total))

    candidates.sort(key=lambda c: c[1])
    return candidates[:max_candidates]


def _build_decomposition(
    graph: StrokeGraph,
    pairings: list[JunctionPairing],
) -> StrokeDecomposition:
    pair_map: dict[SideId, SideId] = {}
    for jp in pairings:
        for (a, b, _) in jp.pairs:
            pair_map[a] = b
            pair_map[b] = a
    strokes = _traverse(graph, pair_map, pairings)
    return StrokeDecomposition(graph=graph, pairings=pairings, strokes=strokes)


# ── Feature extraction ──────────────────────────────────────────────────────

@dataclass
class StructuralFeatures:
    n_strokes: int
    n_closed_loops: int
    has_vertical_spine: bool
    has_horizontal_bar: bool
    has_descender: bool
    has_detached_mark: bool
    loop_vertical_positions: list[float]   # y-centroids in [0, 1]

    # ── Character-rule features ──────────────────────────────────────────
    loop_horizontal_positions: list[float] = field(default_factory=list)  # x centers in [0, 1]
    spine_x: Optional[float] = None        # normalized x of detected spine
    n_right_bulges: int = 0                # non-spine strokes extending right of spine
    n_left_bulges: int = 0                 # non-spine strokes extending left of spine
    has_diagonal_leg: bool = False         # stroke going diag-down from a junction
    tail_lower_right: bool = False         # short stroke ending in lower-right quadrant
    loop_y_spread: float = 0.0             # max(loopY)-min(loopY); 0 if <2 loops

    def as_row(self) -> str:
        bits = []
        if self.has_vertical_spine:   bits.append("spine")
        if self.has_horizontal_bar:   bits.append("bar")
        if self.has_descender:        bits.append("desc")
        if self.has_detached_mark:    bits.append("mark")
        if self.has_diagonal_leg:     bits.append("leg")
        if self.tail_lower_right:     bits.append("qtail")
        flags = ",".join(bits) if bits else "-"
        loops = (
            ",".join(f"{y:.2f}" for y in self.loop_vertical_positions)
            if self.loop_vertical_positions else "-"
        )
        bulges = f"R{self.n_right_bulges}/L{self.n_left_bulges}"
        return (
            f"strokes={self.n_strokes} loops={self.n_closed_loops} "
            f"flags={flags} loopY=[{loops}] {bulges}"
        )


def _bbox_from_graph(graph: StrokeGraph) -> tuple[int, int, int, int]:
    ys, xs = np.where(graph.skeleton)
    if xs.size == 0:
        H, W = graph.shape
        return (0, 0, W, H)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def extract_features(
    decomp: StrokeDecomposition,
    bbox: tuple[int, int, int, int],
) -> StructuralFeatures:
    x0, y0, x1, y1 = bbox
    W = max(1, x1 - x0)
    H = max(1, y1 - y0)
    strokes = decomp.strokes

    n_strokes = len(strokes)
    n_closed = sum(1 for s in strokes if s.closed)

    # Per-stroke spine/bar detection: a stroke qualifies as a spine only if
    # it is tall, narrow, AND nearly straight (high bbox-diagonal / arc-length
    # ratio). This rules out "long curly path that happens to span the height"
    # (e.g. a '9' traversal which goes tail + loop).
    # Spine/bar detection runs on the cleaned GRAPH edges, not the decomposed
    # strokes. A graph edge is a single skeleton chain between nodes, so it
    # doesn't have the composite-traversal problem that ruins per-stroke
    # detection on loops (where every loop has a vertical + horizontal
    # segment that would falsely trigger). An edge qualifies as a spine/bar
    # iff its own bounding box is tall-narrow / wide-short AND the pixel
    # chain is nearly straight.
    has_spine = False
    has_bar = False
    spine_cx: Optional[float] = None
    stroke_lengths: list[int] = [len(s.pixels) for s in strokes]

    for e in decomp.graph.edges:
        if len(e.pixels) < 4:
            continue
        xs_e = [p[0] for p in e.pixels]
        ys_e = [p[1] for p in e.pixels]
        dx = max(xs_e) - min(xs_e)
        dy = max(ys_e) - min(ys_e)
        arc = max(1, len(e.pixels) - 1)
        straightness = math.hypot(dx, dy) / arc   # 1.0 = perfectly straight
        if (
            dy >= 0.35 * H
            and dy >= 2.5 * max(1, dx)
            and straightness >= 0.65
        ):
            has_spine = True
            if spine_cx is None:
                spine_cx = sum(xs_e) / len(xs_e)
        if (
            dx >= 0.35 * W
            and dx >= 2.0 * max(1, dy)
            and straightness >= 0.65
        ):
            has_bar = True

    # Fallback: pixel-window scan on the *head and tail* of open strokes.
    # Closed strokes are skipped entirely — any vertical run inside a loop
    # belongs to the curve, not to a spine. Open strokes whose start or end
    # has a long near-vertical segment get flagged.
    if not has_spine:
        spine_window = max(10, int(0.50 * H))
        candidate_cxs: list[float] = []
        for s in strokes:
            if s.closed or len(s.pixels) < spine_window + 1:
                continue
            xs_s = [p[0] for p in s.pixels]
            ys_s = [p[1] for p in s.pixels]
            if (max(ys_s) - min(ys_s)) < 1.2 * max(1, (max(xs_s) - min(xs_s))):
                continue
            for a, b in (
                (s.pixels[0], s.pixels[spine_window]),
                (s.pixels[-1 - spine_window], s.pixels[-1]),
            ):
                wdy = abs(b[1] - a[1])
                wdx = abs(b[0] - a[0])
                if wdy >= 0.85 * spine_window and wdy >= 4.0 * max(1, wdx):
                    has_spine = True
                    candidate_cxs.append((a[0] + b[0]) / 2)
        if has_spine and spine_cx is None and candidate_cxs:
            # Prefer the leftmost qualifying x — the spine in most English
            # glyphs (B, D, P, R, h, l, i, j, k) sits on the left edge.
            spine_cx = min(candidate_cxs)
    if not has_bar:
        bar_window = max(10, int(0.45 * W))
        for s in strokes:
            if s.closed or len(s.pixels) < bar_window + 1:
                continue
            xs_s = [p[0] for p in s.pixels]
            ys_s = [p[1] for p in s.pixels]
            if (max(xs_s) - min(xs_s)) < 1.2 * max(1, (max(ys_s) - min(ys_s))):
                continue
            for a, b in (
                (s.pixels[0], s.pixels[bar_window]),
                (s.pixels[-1 - bar_window], s.pixels[-1]),
            ):
                wdx = abs(b[0] - a[0])
                wdy = abs(b[1] - a[1])
                if wdx >= 0.85 * bar_window and wdx >= 4.0 * max(1, wdy):
                    has_bar = True
                    break
            if has_bar:
                break

    # Descender: ink has a downward tail. Gap between 90th-percentile y and
    # max y, relative to glyph height.
    has_descender = False
    ys_all: list[int] = [p[1] for s in strokes for p in s.pixels]
    if ys_all:
        ys_sorted = sorted(ys_all)
        p90 = ys_sorted[min(len(ys_sorted) - 1, int(0.90 * len(ys_sorted)))]
        max_y = ys_sorted[-1]
        if H > 0 and (max_y - p90) >= 0.08 * H:
            has_descender = True

    # Detached mark: small stroke entirely in the top 30%.
    detached = False
    if stroke_lengths:
        max_len = max(stroke_lengths) or 1
        top_zone = y0 + 0.30 * H
        for s in strokes:
            if not s.pixels or len(s.pixels) >= 0.35 * max_len:
                continue
            ys_s = [p[1] for p in s.pixels]
            if max(ys_s) <= top_zone:
                detached = True
                break

    # Loop positions — both vertical and horizontal, normalized to [0, 1].
    loop_ys: list[float] = []
    loop_xs: list[float] = []
    for s in strokes:
        if not s.closed or not s.pixels:
            continue
        xs_s = [p[0] for p in s.pixels]
        ys_s = [p[1] for p in s.pixels]
        cy = sum(ys_s) / len(ys_s)
        cx = sum(xs_s) / len(xs_s)
        loop_ys.append((cy - y0) / H)
        loop_xs.append((cx - x0) / W)

    # Bulges: strokes whose bbox extends substantially to the right/left of
    # the spine x. A pure-spine stroke would have max_x close to spine_cx,
    # so it naturally doesn't count; a composite "spine + bowl" stroke will
    # extend out to the bowl side and count there.
    n_right_bulges = 0
    n_left_bulges = 0
    if spine_cx is not None:
        for s in strokes:
            if not s.pixels:
                continue
            xs_s = [p[0] for p in s.pixels]
            if (max(xs_s) - spine_cx) > 0.20 * W:
                n_right_bulges += 1
            if (spine_cx - min(xs_s)) > 0.20 * W:
                n_left_bulges += 1

    # Diagonal leg: a stroke that leaves a junction DOWN-and-sideways and
    # terminates at an endpoint below the junction. Typical of R's leg and
    # K's legs. Rejects ascending strokes (they're not "legs") and near-
    # vertical strokes (those are spines / tails).
    has_leg = False
    junction_ids = {n.id for n in decomp.graph.junctions}
    endpoint_ids = {n.id for n in decomp.graph.endpoints}
    for s in strokes:
        if not s.pixels:
            continue
        starts_at_j = s.start_node in junction_ids
        ends_at_j = s.end_node in junction_ids
        starts_at_e = s.start_node in endpoint_ids
        ends_at_e = s.end_node in endpoint_ids
        # Need one end at a junction, the other at an endpoint.
        if starts_at_j and ends_at_e:
            p_junc, p_far = s.pixels[0], s.pixels[-1]
        elif ends_at_j and starts_at_e:
            p_junc, p_far = s.pixels[-1], s.pixels[0]
        else:
            continue
        # The endpoint must be BELOW the junction (positive dy).
        if p_far[1] <= p_junc[1]:
            continue
        # The junction should sit in the upper half of the glyph — a leg
        # hangs from an upper bowl (R, K). If the junction is mid-or-lower,
        # this is probably just a tail (e, a, 9), not a leg.
        if (p_junc[1] - y0) / H > 0.45:
            continue
        # The endpoint should be in the lower portion.
        if (p_far[1] - y0) / H < 0.65:
            continue
        dy = p_far[1] - p_junc[1]
        dx = abs(p_far[0] - p_junc[0])
        if dy < 0.30 * H:
            continue
        # Leg must be MORE vertical than horizontal (dy > dx) but not pure
        # vertical (dx >= 0.35*dy). This rules out e/a-style tails where the
        # stroke curves out more horizontally, and also rules out pure
        # descenders on j/9/y.
        if dx >= 0.35 * dy and dx < dy:
            has_leg = True
            break

    # Q-style tail: a short stroke whose last pixel is in the lower-right
    # quadrant (x > 0.55, y > 0.55) AND whose direction is down-right.
    tail_lower_right = False
    if stroke_lengths:
        max_len = max(stroke_lengths) or 1
        for s in strokes:
            if not s.pixels or len(s.pixels) >= 0.45 * max_len:
                continue
            p_start = s.pixels[0]
            p_end = s.pixels[-1]
            nx = (p_end[0] - x0) / W
            ny = (p_end[1] - y0) / H
            if nx < 0.50 or ny < 0.55:
                continue
            if (p_end[0] - p_start[0]) >= 0 and (p_end[1] - p_start[1]) > 0:
                tail_lower_right = True
                break

    # Vertical spread of loops — useful for distinguishing balanced (8) vs
    # single-loop (O, 0) vs stacked-with-spine (B) cases.
    loop_spread = 0.0
    if len(loop_ys) >= 2:
        loop_spread = max(loop_ys) - min(loop_ys)

    return StructuralFeatures(
        n_strokes=n_strokes,
        n_closed_loops=n_closed,
        has_vertical_spine=has_spine,
        has_horizontal_bar=has_bar,
        has_descender=has_descender,
        has_detached_mark=detached,
        loop_vertical_positions=loop_ys,
        loop_horizontal_positions=loop_xs,
        spine_x=(spine_cx - x0) / W if spine_cx is not None else None,
        n_right_bulges=n_right_bulges,
        n_left_bulges=n_left_bulges,
        has_diagonal_leg=has_leg,
        tail_lower_right=tail_lower_right,
        loop_y_spread=loop_spread,
    )


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_prior(features: StructuralFeatures, prior: CharPrior) -> float:
    """Soft penalty (dimensionless). Lower = better match to the prior."""
    penalty = 0.0

    s_min, s_max = prior.strokes
    n = features.n_strokes
    if n < s_min:
        penalty += (s_min - n) * 1.0
    elif n > s_max:
        penalty += (n - s_max) * 0.8

    l_min, l_max = prior.loops
    l = features.n_closed_loops
    if l < l_min:
        penalty += (l_min - l) * 1.2
    elif l > l_max:
        penalty += (l - l_max) * 1.2

    for prior_val, feat_val, weight in (
        (prior.spine, features.has_vertical_spine, 0.8),
        (prior.bar, features.has_horizontal_bar, 0.6),
        (prior.descender, features.has_descender, 0.4),
        (prior.detached_mark, features.has_detached_mark, 0.5),
    ):
        if prior_val is None:
            continue
        if prior_val != feat_val:
            penalty += weight

    # Loop-position hint: gently penalize if the nearest observed loop is
    # far from the expected y. Ignore if counts don't line up.
    if prior.loop_positions and features.loop_vertical_positions:
        for expected in prior.loop_positions:
            closest = min(
                features.loop_vertical_positions,
                key=lambda y: abs(y - expected),
            )
            penalty += 0.4 * abs(closest - expected)

    return penalty


@dataclass
class ScoredCandidate:
    decomposition: StrokeDecomposition
    features: StructuralFeatures
    geom_cost: float
    prior_penalty: float
    custom_penalty: float = 0.0
    discrim_penalty: float = 0.0
    total_score: float = 0.0


# Weight on prior penalty relative to geometry cost (degrees). alpha = 6 means
# a full unit of prior mismatch is worth ~6 degrees of extra turn angle — big
# enough to flip ties, small enough that a clearly-worse pairing never wins.
_PRIOR_ALPHA: float = 6.0


def resolve(
    graph: StrokeGraph,
    char: Optional[str] = None,
    turn_max_degrees: float = 120.0,
    max_candidates: int = 6,
    prior_alpha: float = _PRIOR_ALPHA,
) -> tuple[StrokeDecomposition, list[ScoredCandidate]]:
    """Return (best_decomposition, all_scored_candidates).

    If `char` is None or unknown, priors are permissive and geometry
    decides (equivalent to the greedy step-6 output for most glyphs).
    """
    ranked = _enumerate_candidates(graph, turn_max_degrees, max_candidates)
    if not ranked:
        empty = StrokeDecomposition(
            graph=graph, pairings=[],
            strokes=_traverse(graph, {}, []),
        )
        return empty, []

    bbox = _bbox_from_graph(graph)
    prior = get_prior(char)

    # Char-specific + discrimination scorers live in a separate module so
    # the core resolver stays small. Import lazily to avoid import-cycle risks.
    from char_rules import score_custom, score_discrimination

    scored: list[ScoredCandidate] = []
    for pairings, geom_cost in ranked:
        decomp = _build_decomposition(graph, pairings)
        features = extract_features(decomp, bbox)
        generic_pen = score_prior(features, prior)
        custom_pen = score_custom(char, features)
        discrim_pen = score_discrimination(char, features)
        total = geom_cost + prior_alpha * (generic_pen + custom_pen + discrim_pen)
        scored.append(ScoredCandidate(
            decomposition=decomp,
            features=features,
            geom_cost=geom_cost,
            prior_penalty=generic_pen,
            custom_penalty=custom_pen,
            discrim_penalty=discrim_pen,
            total_score=total,
        ))

    scored.sort(key=lambda c: c.total_score)
    return scored[0].decomposition, scored
