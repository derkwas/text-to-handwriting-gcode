#!/usr/bin/env python3
"""Decompose a cleaned StrokeGraph into plausible pen strokes.

At each junction we decide, locally, which edge the pen "continued into":
two incident edges whose leaving-tangents point most opposite to each other
form a smooth pass-through; they get paired. Junctions of odd degree leave
one edge-end unpaired (a stroke boundary). A pair is rejected if it would
imply a turn sharper than `turn_max_degrees` — those sides are left unpaired
and become stroke starts/ends instead.

A stroke is then built by chaining edges: start at a free side (endpoint,
unpaired junction side, or any side of an otherwise-closed component),
traverse the edge, at the far end consult the pairing map for the
partner side, and repeat until we hit a free side or close on ourselves.

Non-goals at this step: smoothing, resampling, spline fitting, pen
sequencing between strokes. The output is stroke *candidates* — each one
is an ordered list of edges (with direction flags) plus the concatenated
pixel chain, nothing more.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from stroke_graph import Edge, StrokeGraph


Pixel = tuple[int, int]
# An edge-side identifies which end of an edge we are looking at.
# Side "a" is the pixels[0] end (at edge.start_node).
# Side "b" is the pixels[-1] end (at edge.end_node).
SideId = tuple[int, str]


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class JunctionPairing:
    node_id: int
    # ((side_a, side_b, turn_degrees), ...)
    pairs: list[tuple[SideId, SideId, float]] = field(default_factory=list)
    unpaired: list[SideId] = field(default_factory=list)


@dataclass
class Stroke:
    id: int
    # (edge_id, reversed_flag). reversed=True means edge traversed b->a.
    edge_sequence: list[tuple[int, bool]]
    pixels: list[Pixel]
    closed: bool
    start_node: Optional[int]
    end_node: Optional[int]


@dataclass
class StrokeDecomposition:
    graph: StrokeGraph
    pairings: list[JunctionPairing]
    strokes: list[Stroke]


# ── Tangent estimation ──────────────────────────────────────────────────────

def _tangent_at_side(edge: Edge, side: str, k: int = 4) -> Optional[np.ndarray]:
    """Unit vector pointing from the junction-end pixel INTO the edge.

    Uses a short lookahead of `k` pixels to smooth over 1-pixel staircase
    artifacts from the skeleton.
    """
    pixels = edge.pixels
    n = len(pixels)
    if n < 2:
        return None
    k = max(1, min(k, n - 1))
    if side == "a":
        p_near, p_far = pixels[0], pixels[k]
    else:
        p_near, p_far = pixels[-1], pixels[-1 - k]
    dx = p_far[0] - p_near[0]
    dy = p_far[1] - p_near[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return None
    return np.array([dx / norm, dy / norm])


# ── Per-junction pairing ────────────────────────────────────────────────────

def _pair_junctions(
    graph: StrokeGraph,
    turn_max_degrees: float,
) -> list[JunctionPairing]:
    # Accept a pair iff its implied turn angle <= turn_max.
    # turn = pi - theta, theta = angle between leaving tangents,
    # so cos(theta) = dot <= -cos(turn_max).
    max_dot = -math.cos(math.radians(turn_max_degrees))

    edges = {e.id: e for e in graph.edges}
    sides_by_node: dict[int, list[SideId]] = {}
    for e in graph.edges:
        if e.start_node is not None:
            sides_by_node.setdefault(e.start_node, []).append((e.id, "a"))
        if e.end_node is not None:
            sides_by_node.setdefault(e.end_node, []).append((e.id, "b"))

    pairings: list[JunctionPairing] = []
    for junc in graph.junctions:
        sides = sides_by_node.get(junc.id, [])
        jp = JunctionPairing(node_id=junc.id)
        if len(sides) < 2:
            jp.unpaired = list(sides)
            pairings.append(jp)
            continue

        tangents: dict[SideId, Optional[np.ndarray]] = {
            sid: _tangent_at_side(edges[sid[0]], sid[1]) for sid in sides
        }
        pair_cost: dict[tuple[SideId, SideId], float] = {}
        for i, a in enumerate(sides):
            for b in sides[i + 1:]:
                ta, tb = tangents[a], tangents[b]
                if ta is None or tb is None:
                    continue
                pair_cost[(a, b)] = float(np.dot(ta, tb))

        # Greedy min-weight matching. Optimal for k<=3 and typically fine
        # for k==4 (X-crossings) given real handwriting geometry.
        remaining = list(sides)
        while len(remaining) >= 2:
            best: Optional[tuple[SideId, SideId]] = None
            best_cost = float("inf")
            for i, a in enumerate(remaining):
                for b in remaining[i + 1:]:
                    c = pair_cost.get((a, b))
                    if c is None:
                        continue
                    if c < best_cost:
                        best_cost = c
                        best = (a, b)
            if best is None or best_cost > max_dot:
                break
            a, b = best
            theta = math.acos(max(-1.0, min(1.0, best_cost)))
            turn_deg = math.degrees(math.pi - theta)
            jp.pairs.append((a, b, turn_deg))
            remaining.remove(a)
            remaining.remove(b)
        jp.unpaired = remaining
        pairings.append(jp)

    return pairings


# ── Traversal ────────────────────────────────────────────────────────────────

def _other_side(side: SideId) -> SideId:
    eid, s = side
    return (eid, "b" if s == "a" else "a")


def _node_of_side(side: SideId, edges: dict[int, Edge]) -> Optional[int]:
    eid, s = side
    e = edges[eid]
    return e.start_node if s == "a" else e.end_node


def _traverse(
    graph: StrokeGraph,
    pair_map: dict[SideId, SideId],
    pairings: list[JunctionPairing],
) -> list[Stroke]:
    edges = {e.id: e for e in graph.edges}
    consumed: set[int] = set()
    strokes: list[Stroke] = []

    def build(start_side: SideId) -> Stroke:
        seq: list[tuple[int, bool]] = []
        pix: list[Pixel] = []
        start_node_id = _node_of_side(start_side, edges)
        end_node_id: Optional[int] = None
        closed = False
        current = start_side

        while True:
            eid, sd = current
            if eid in consumed:
                # Came back to the very start → closed cycle.
                closed = True
                break
            consumed.add(eid)

            e = edges[eid]
            chain = list(e.pixels) if sd == "a" else list(reversed(e.pixels))
            if pix and pix[-1] == chain[0]:
                pix.extend(chain[1:])
            else:
                pix.extend(chain)
            seq.append((eid, sd == "b"))

            oppo = _other_side(current)
            partner = pair_map.get(oppo)
            if partner is None:
                end_node_id = _node_of_side(oppo, edges)
                break
            if partner == start_side:
                closed = True
                break
            current = partner

        # Geometric closure catches cases the pair_map can't flag: pure cycles
        # (no node, no pair_map entry) and any path whose first/last pixel
        # happen to coincide.
        if not closed and pix and pix[0] == pix[-1]:
            closed = True

        return Stroke(
            id=len(strokes),
            edge_sequence=seq,
            pixels=pix,
            closed=closed,
            start_node=None if closed else start_node_id,
            end_node=None if closed else end_node_id,
        )

    # 1. Strokes rooted at endpoint-nodes.
    for node in graph.endpoints:
        start: Optional[SideId] = None
        for e in graph.edges:
            if e.id in consumed:
                continue
            if e.start_node == node.id:
                start = (e.id, "a")
                break
            if e.end_node == node.id:
                start = (e.id, "b")
                break
        if start is not None:
            strokes.append(build(start))

    # 2. Strokes rooted at unpaired junction sides.
    for jp in pairings:
        for sid in jp.unpaired:
            if sid[0] in consumed:
                continue
            strokes.append(build(sid))

    # 3. Remaining edges belong to closed components (pure cycles or
    #    fully-paired loops). Emit each as a closed stroke.
    for e in graph.edges:
        if e.id in consumed:
            continue
        strokes.append(build((e.id, "a")))

    return strokes


# ── Entry point ──────────────────────────────────────────────────────────────

def decompose(graph: StrokeGraph, turn_max_degrees: float = 120.0) -> StrokeDecomposition:
    pairings = _pair_junctions(graph, turn_max_degrees)
    pair_map: dict[SideId, SideId] = {}
    for jp in pairings:
        for (a, b, _) in jp.pairs:
            pair_map[a] = b
            pair_map[b] = a
    strokes = _traverse(graph, pair_map, pairings)
    return StrokeDecomposition(graph=graph, pairings=pairings, strokes=strokes)
