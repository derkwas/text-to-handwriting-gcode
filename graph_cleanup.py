#!/usr/bin/env python3
"""Topology cleanup for StrokeGraph: junction cluster collapse + spur pruning.

This is step 4 in the pipeline. Input is the raw StrokeGraph produced by
stroke_graph.extract_stroke_graph, which faithfully captures the skeleton's
topology including the two defects we want to remove here:

  - Junction clumps: one physical intersection becomes a cluster of 3-6
    adjacent junction pixels, because skeletonize can't produce a single-
    pixel crossing at angled intersections. Multi-edge "cycles" between
    cluster members are fake loops; real loops (e.g. the interior of 'e')
    show up as a LONG edge also spanning two cluster members.

  - Spurs: short endpoint-attached edges left by thinning brush-like stroke
    ends or raster irregularities.

Pipeline:

  1. clump collapse (threshold = clump_k * stroke_width)
     - Union-find over junctions connected by short edges.
     - Each cluster of >= 2 junctions becomes one canonical junction node
       at the centroid (snapped to the nearest skeleton pixel).
     - Internal short edges between cluster members are dropped.
     - Long edges between cluster members become SELF-LOOPS on the
       canonical node (this is how e_0 / 9_0 get their true interior loop).

  2. spur pruning (threshold = spur_k * stroke_width)
     - Drop edges with exactly one endpoint-node and length below the
       threshold. Never drop edges where both ends are junctions — that
       could disconnect a cycle.

  3. dissolve degree-2 junctions
     - A junction left with only 2 incident edges after pruning is no
       longer a topological intersection. Merge its two edges into one,
       or convert a lone self-loop into a pure cycle.

Non-goals at this step: stroke pairing, co-linear merging, smoothing,
spline fitting. The output is a cleaner topology, nothing more.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from stroke_graph import Edge, Node, StrokeGraph


# ── Audit ────────────────────────────────────────────────────────────────────

@dataclass
class CleanupAudit:
    stroke_width: float
    spur_threshold: int
    clump_threshold: int
    # Each entry: {"members": [old_ids...], "canonical_pos": (x, y)}
    collapsed_clusters: list[dict] = field(default_factory=list)
    # Each entry: {"id", "length", "start", "end"} — IDs are post-collapse
    pruned_edges: list[dict] = field(default_factory=list)
    # Old node IDs that were dissolved (deg-2 removal)
    dissolved_nodes: list[int] = field(default_factory=list)


# ── Stroke-width estimate ────────────────────────────────────────────────────

def estimate_stroke_width(binary_ink: np.ndarray, skeleton: np.ndarray) -> float:
    """Median stroke diameter in pixels.

    The distance transform at a skeleton pixel gives the radius to the
    nearest background pixel. Doubling the median radius gives a robust
    estimate of stroke diameter that ignores thin tails.
    """
    ink_u8 = (binary_ink > 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(ink_u8, cv2.DIST_L2, 3)
    radii = dist[skeleton]
    if radii.size == 0:
        return 2.0
    return 2.0 * float(np.median(radii))


# ── Main entry point ─────────────────────────────────────────────────────────

def clean_graph(
    graph: StrokeGraph,
    binary_ink: np.ndarray,
    spur_k: float = 0.75,
    clump_k: float = 1.0,
) -> tuple[StrokeGraph, CleanupAudit]:
    """Apply clump collapse, spur pruning, and deg-2 dissolution.

    spur_k and clump_k are multipliers on the estimated stroke width. With
    the defaults, a 6-pixel-thick stroke gets a ~4-pixel spur threshold and
    a ~6-pixel clump threshold.
    """
    stroke_width = estimate_stroke_width(binary_ink, graph.skeleton)
    spur_threshold = max(3, int(round(spur_k * stroke_width)))
    clump_threshold = max(2, int(round(clump_k * stroke_width)))

    audit = CleanupAudit(
        stroke_width=stroke_width,
        spur_threshold=spur_threshold,
        clump_threshold=clump_threshold,
    )

    # Mutable shallow copies so we don't clobber the input graph.
    nodes: dict[int, Node] = {
        n.id: Node(n.id, n.pos, n.kind, n.degree) for n in graph.nodes
    }
    edges: dict[int, Edge] = {
        e.id: Edge(e.id, list(e.pixels), e.start_node, e.end_node)
        for e in graph.edges
    }

    _collapse_clumps(nodes, edges, clump_threshold, graph.skeleton, audit)
    _prune_spurs(nodes, edges, spur_threshold, audit)
    _reclassify_after_prune(nodes, edges)
    _dissolve_deg2(nodes, edges, audit)

    cleaned = _rebuild(graph.skeleton, graph.shape, nodes, edges)
    return cleaned, audit


# ── Phase 1: junction cluster collapse ──────────────────────────────────────

def _collapse_clumps(
    nodes: dict[int, Node],
    edges: dict[int, Edge],
    clump_threshold: int,
    skeleton: np.ndarray,
    audit: CleanupAudit,
) -> None:
    junction_ids = {nid for nid, n in nodes.items() if n.kind == "junction"}
    if len(junction_ids) < 2:
        return

    # Union-find: link two junctions iff a short edge connects them.
    parent = {nid: nid for nid in junction_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for e in edges.values():
        if (
            e.start_node in junction_ids
            and e.end_node in junction_ids
            and e.start_node != e.end_node
            and e.length <= clump_threshold
        ):
            union(e.start_node, e.end_node)

    clusters: dict[int, list[int]] = {}
    for nid in junction_ids:
        clusters.setdefault(find(nid), []).append(nid)

    # Precompute skeleton pixel list for nearest-pixel snapping.
    ys, xs = np.where(skeleton)
    skel_pixels = list(zip(xs.tolist(), ys.tolist()))

    for root, members in clusters.items():
        if len(members) < 2:
            continue

        positions = [nodes[m].pos for m in members]
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        canonical_pos = min(
            skel_pixels, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2
        )
        canonical_id = root

        audit.collapsed_clusters.append({
            "members": sorted(members),
            "canonical_pos": canonical_pos,
        })

        # Replace the root with the canonical node; drop other members.
        nodes[canonical_id] = Node(canonical_id, canonical_pos, "junction", 0)
        for m in members:
            if m != canonical_id and m in nodes:
                del nodes[m]

        member_set = set(members)
        to_delete: list[int] = []
        for eid, e in edges.items():
            starts_in = e.start_node in member_set
            ends_in = e.end_node in member_set
            if not (starts_in or ends_in):
                continue
            # Short internal edge (clump-only) — discard.
            if starts_in and ends_in and e.length <= clump_threshold:
                to_delete.append(eid)
                continue
            # Long internal edge → keep; will become a self-loop on canonical.
            if starts_in:
                e.pixels = [canonical_pos] + list(e.pixels[1:])
                e.start_node = canonical_id
            if ends_in:
                e.pixels = list(e.pixels[:-1]) + [canonical_pos]
                e.end_node = canonical_id
        for eid in to_delete:
            del edges[eid]


# ── Phase 2: spur pruning ────────────────────────────────────────────────────

def _prune_spurs(
    nodes: dict[int, Node],
    edges: dict[int, Edge],
    threshold: int,
    audit: CleanupAudit,
) -> None:
    endpoint_ids = {nid for nid, n in nodes.items() if n.kind == "endpoint"}

    to_remove: list[int] = []
    for eid, e in edges.items():
        starts_at_endpt = e.start_node in endpoint_ids
        ends_at_endpt = e.end_node in endpoint_ids
        # Exactly one end is an endpoint (i.e. the other is a junction).
        # Edges between two endpoints are whole isolated strokes — never prune.
        if (starts_at_endpt ^ ends_at_endpt) and e.length <= threshold:
            to_remove.append(eid)

    for eid in to_remove:
        e = edges[eid]
        endpoint_side = (
            e.start_node if e.start_node in endpoint_ids else e.end_node
        )
        audit.pruned_edges.append({
            "id": eid,
            "length": e.length,
            "start": e.start_node,
            "end": e.end_node,
        })
        del edges[eid]
        if endpoint_side in nodes:
            del nodes[endpoint_side]


# ── Phase 2b: reclassify junctions whose degree dropped after pruning ───────

def _reclassify_after_prune(
    nodes: dict[int, Node],
    edges: dict[int, Edge],
) -> None:
    """Spur pruning can leave former junctions with degree 0 or 1.

    Degree 0 means an orphan — remove it.
    Degree 1 means it is topologically an endpoint now (a stroke terminates
    there), so relabel it. Degree 2 is handled by the next phase.
    """
    degree = {nid: 0 for nid in nodes}
    for e in edges.values():
        if e.start_node in degree:
            degree[e.start_node] += 1
        if e.end_node in degree:
            degree[e.end_node] += 1

    for nid, d in list(degree.items()):
        n = nodes.get(nid)
        if n is None or n.kind != "junction":
            continue
        if d == 0:
            del nodes[nid]
        elif d == 1:
            n.kind = "endpoint"
            n.degree = 1


# ── Phase 3: dissolve degree-2 junctions ────────────────────────────────────

def _dissolve_deg2(
    nodes: dict[int, Node],
    edges: dict[int, Edge],
    audit: CleanupAudit,
) -> None:
    # A junction is "degree 2" when it has exactly two edge-endpoints
    # referencing it (self-loops count as two). Such a node is no longer a
    # real topological intersection and should be removed.
    changed = True
    while changed:
        changed = False
        degree = {nid: 0 for nid in nodes}
        for e in edges.values():
            if e.start_node in degree:
                degree[e.start_node] += 1
            if e.end_node in degree:
                degree[e.end_node] += 1

        for nid, d in list(degree.items()):
            if d != 2 or nid not in nodes:
                continue
            # Only dissolve junctions — endpoints are where strokes begin/end.
            if nodes[nid].kind != "junction":
                continue

            incident: list[tuple[int, str]] = []
            for eid, e in edges.items():
                if e.start_node == nid:
                    incident.append((eid, "start"))
                if e.end_node == nid:
                    incident.append((eid, "end"))
            eids = {x[0] for x in incident}

            if len(eids) == 1:
                # Lone self-loop → free-floating pure cycle.
                eid = incident[0][0]
                e = edges[eid]
                e.start_node = None
                e.end_node = None
                del nodes[nid]
                audit.dissolved_nodes.append(nid)
                changed = True
                continue

            if len(eids) == 2:
                (e1_id, e1_side), (e2_id, e2_side) = incident
                e1, e2 = edges[e1_id], edges[e2_id]
                # Orient so nid is the join pixel: e1 ends at nid, e2 starts at nid.
                chain1 = (
                    list(e1.pixels) if e1_side == "end" else list(reversed(e1.pixels))
                )
                chain2 = (
                    list(e2.pixels) if e2_side == "start" else list(reversed(e2.pixels))
                )
                merged = chain1 + chain2[1:]  # drop duplicate join pixel
                new_start = e1.start_node if e1_side == "end" else e1.end_node
                new_end = e2.end_node if e2_side == "start" else e2.start_node
                new_id = min(e1_id, e2_id)
                del edges[max(e1_id, e2_id)]
                del edges[min(e1_id, e2_id)]
                edges[new_id] = Edge(new_id, merged, new_start, new_end)
                del nodes[nid]
                audit.dissolved_nodes.append(nid)
                changed = True
                continue


# ── Rebuild with fresh IDs ───────────────────────────────────────────────────

def _rebuild(
    skeleton: np.ndarray,
    shape: tuple[int, int],
    nodes: dict[int, Node],
    edges: dict[int, Edge],
) -> StrokeGraph:
    id_map: dict[int, int] = {}
    new_nodes: list[Node] = []
    for old_id in sorted(nodes.keys()):
        n = nodes[old_id]
        new_id = len(new_nodes)
        id_map[old_id] = new_id
        new_nodes.append(Node(new_id, n.pos, n.kind, 0))

    new_edges: list[Edge] = []
    for old_eid in sorted(edges.keys()):
        e = edges[old_eid]
        new_id = len(new_edges)
        ns = None if e.start_node is None else id_map[e.start_node]
        ne = None if e.end_node is None else id_map[e.end_node]
        new_edges.append(Edge(new_id, e.pixels, ns, ne))

    degree = {n.id: 0 for n in new_nodes}
    for e in new_edges:
        if e.start_node is not None:
            degree[e.start_node] += 1
        if e.end_node is not None:
            degree[e.end_node] += 1
    for n in new_nodes:
        n.degree = degree[n.id]

    return StrokeGraph(nodes=new_nodes, edges=new_edges, skeleton=skeleton, shape=shape)
