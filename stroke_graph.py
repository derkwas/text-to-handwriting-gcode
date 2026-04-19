#!/usr/bin/env python3
"""Topology-preserving graph extractor for skeletonized glyphs.

This is the replacement for learn_font.trace_skeleton at the representation
layer. It does NOT patch the greedy tracer — it gives a new, richer output
that later stages (spur pruning, junction pairing, stroke reconstruction)
will consume.

Representation
--------------
Given a boolean skeleton of a glyph:

- Node: a skeleton pixel with degree 1 (endpoint) or degree >= 3 (junction).
- Edge: an ordered pixel chain that starts and ends at a node and passes
  through only degree-2 interior pixels in between.
- Pure cycle: a connected component with no node at all (e.g. hollow 'o').
  Emitted as a closed-loop edge with start_node = end_node = None and
  pixels[0] == pixels[-1].
- Self-loop: an edge that leaves a junction and returns to the same
  junction (e.g. the interior loop of 'e' that rejoins at the tail
  crossing). Emitted with start_node == end_node and pixels[0] == pixels[-1].

An edge is a loop iff pixels[0] == pixels[-1].

Deliberate non-goals at this stage
----------------------------------
- No spur pruning: short dangling edges from thinning are preserved so
  they remain visible in diagnostics.
- No junction resolution: adjacent junction pixels are NOT merged, so
  skeletonize's "fat intersection" clumps show up as tiny edges between
  junction nodes. This is intentional — it is the defect we want the
  next stage (pruning + cluster collapse) to fix.
- No co-linear edge merging.
- No resampling or smoothing.

Think of a StrokeGraph as the topology snapshot that later stages rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


Pixel = tuple[int, int]


@dataclass
class Node:
    id: int
    pos: Pixel
    kind: str      # "endpoint" or "junction"
    degree: int


@dataclass
class Edge:
    id: int
    pixels: list[Pixel]
    start_node: Optional[int]   # Node.id; None for a pure cycle
    end_node: Optional[int]     # Node.id; None for a pure cycle

    @property
    def is_loop(self) -> bool:
        return self.pixels[0] == self.pixels[-1]

    @property
    def is_self_loop(self) -> bool:
        """Loop anchored at a junction (start_node == end_node, both not None)."""
        return (
            self.is_loop
            and self.start_node is not None
            and self.start_node == self.end_node
        )

    @property
    def is_pure_cycle(self) -> bool:
        """Loop with no attached node (a free-floating cycle)."""
        return self.is_loop and self.start_node is None and self.end_node is None

    @property
    def length(self) -> int:
        return len(self.pixels)


@dataclass
class StrokeGraph:
    nodes: list[Node]
    edges: list[Edge]
    skeleton: np.ndarray
    shape: tuple[int, int]

    @property
    def endpoints(self) -> list[Node]:
        return [n for n in self.nodes if n.kind == "endpoint"]

    @property
    def junctions(self) -> list[Node]:
        return [n for n in self.nodes if n.kind == "junction"]

    @property
    def loop_edges(self) -> list[Edge]:
        return [e for e in self.edges if e.is_loop]


# ── Extraction ───────────────────────────────────────────────────────────────

def _neighbors(p: Pixel, pixels: set[Pixel]) -> list[Pixel]:
    x, y = p
    return [
        (x + dx, y + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (dx, dy) != (0, 0) and (x + dx, y + dy) in pixels
    ]


def extract_stroke_graph(skel: np.ndarray) -> StrokeGraph:
    """Decompose a boolean skeleton array into a StrokeGraph."""
    H, W = skel.shape
    ys, xs = np.where(skel)
    if xs.size == 0:
        return StrokeGraph(nodes=[], edges=[], skeleton=skel, shape=(H, W))

    pixels: set[Pixel] = set(zip(xs.tolist(), ys.tolist()))
    degree: dict[Pixel, int] = {p: len(_neighbors(p, pixels)) for p in pixels}

    # ── Nodes ────────────────────────────────────────────────────────────
    nodes: list[Node] = []
    pos_to_node_id: dict[Pixel, int] = {}
    for p, d in degree.items():
        if d == 1:
            kind = "endpoint"
        elif d >= 3:
            kind = "junction"
        else:
            continue
        node = Node(id=len(nodes), pos=p, kind=kind, degree=d)
        nodes.append(node)
        pos_to_node_id[p] = node.id

    # ── Edges ────────────────────────────────────────────────────────────
    edges: list[Edge] = []
    walked_step: set[tuple[Pixel, Pixel]] = set()

    def walk(start: Pixel, first: Pixel) -> list[Pixel]:
        """Walk from `start` through `first` along degree-2 interior pixels
        until the chain lands on another node (or closes back on start)."""
        path = [start, first]
        walked_step.add((start, first))
        prev, cur = start, first
        while degree.get(cur, 0) == 2:
            nxt: Optional[Pixel] = None
            for n in _neighbors(cur, pixels):
                if n != prev:
                    nxt = n
                    break
            if nxt is None:
                break
            path.append(nxt)
            prev, cur = cur, nxt
        # Mark the reverse of the final step so we don't re-walk this edge
        # from the other end when we iterate that node's neighbors.
        if len(path) >= 2:
            walked_step.add((path[-1], path[-2]))
        return path

    # Phase 1: every edge incident to a node
    for node in nodes:
        for nb in _neighbors(node.pos, pixels):
            if (node.pos, nb) in walked_step:
                continue
            chain = walk(node.pos, nb)
            if len(chain) < 2:
                continue
            edges.append(Edge(
                id=len(edges),
                pixels=chain,
                start_node=pos_to_node_id.get(chain[0]),
                end_node=pos_to_node_id.get(chain[-1]),
            ))

    # Phase 2: pure cycles — components with only degree-2 pixels, never
    # touched by Phase 1.
    consumed: set[Pixel] = set()
    for e in edges:
        consumed.update(e.pixels)
    remaining = pixels - consumed
    while remaining:
        start = next(iter(remaining))
        path: list[Pixel] = [start]
        prev: Optional[Pixel] = None
        cur = start
        safety = len(pixels) + 2
        while safety > 0:
            options = [n for n in _neighbors(cur, pixels) if n != prev]
            if not options:
                break
            nxt = options[0]
            if nxt == start:
                path.append(start)   # close the cycle explicitly
                break
            path.append(nxt)
            prev, cur = cur, nxt
            safety -= 1
        for p in path:
            remaining.discard(p)
        if len(path) >= 2:
            edges.append(Edge(
                id=len(edges),
                pixels=path,
                start_node=None,
                end_node=None,
            ))

    return StrokeGraph(nodes=nodes, edges=edges, skeleton=skel, shape=(H, W))


# ── Reporting ────────────────────────────────────────────────────────────────

def edge_kind(edge: Edge) -> str:
    if edge.is_pure_cycle:
        return "pure-cycle"
    if edge.is_self_loop:
        return "self-loop"
    return "open"


def summarize(graph: StrokeGraph) -> dict:
    """Machine-readable summary for tests and report printing."""
    return {
        "nodes": len(graph.nodes),
        "endpoints": len(graph.endpoints),
        "junctions": len(graph.junctions),
        "edges": len(graph.edges),
        "loops": len(graph.loop_edges),
        "pure_cycles": sum(1 for e in graph.edges if e.is_pure_cycle),
        "self_loops": sum(1 for e in graph.edges if e.is_self_loop),
        "edge_rows": [
            {
                "id": e.id,
                "kind": edge_kind(e),
                "start": e.start_node,
                "end": e.end_node,
                "length": e.length,
            }
            for e in graph.edges
        ],
    }
