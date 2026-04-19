#!/usr/bin/env python3
"""Beautification / fitting stage for decomposed strokes.

For each stroke produced by stroke_decompose.decompose:

  1. Arc-length resampling
     - Uniform spacing along the chain to remove the per-pixel density
       bias of skeleton traversal. Default 1.5 px.
  2. Gaussian smoothing
     - Removes 1-px staircase artifacts without visibly distorting the
       shape. Default sigma = 1.2 samples.
     - Open strokes: endpoints pinned exactly (no drift).
     - Closed strokes: wrap mode on the interior, closure preserved.
  3. Ramer-Douglas-Peucker simplification
     - Reduces the smoothed polyline to a minimal set of points ("control
       points") that preserve the visible shape within `epsilon` px.
       Default 0.6 px.
  4. Cubic B-spline fit
     - A parametric spline through the simplified points, used to
       evaluate a dense smooth curve for rendering and G-code sampling.

Non-goals: pen-ordering, kerning, baseline snapping, SVG export.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import interpolate, ndimage

from stroke_decompose import Stroke, StrokeDecomposition


# ── Data ─────────────────────────────────────────────────────────────────────

@dataclass
class FittedStroke:
    id: int
    raw_pixels: list[tuple[int, int]]
    resampled: np.ndarray       # (N, 2)
    smoothed: np.ndarray        # (N, 2)
    simplified: np.ndarray      # (M, 2) — control points
    closed: bool
    start_node: Optional[int]
    end_node: Optional[int]
    spline: Optional[tuple]     # (tck, u) from splprep, or None if too short

    @property
    def num_raw(self) -> int:
        return len(self.raw_pixels)

    @property
    def num_resampled(self) -> int:
        return len(self.resampled)

    @property
    def num_smoothed(self) -> int:
        return len(self.smoothed)

    @property
    def num_control(self) -> int:
        return len(self.simplified)

    def sample_curve(self, n: int = 128) -> np.ndarray:
        """Densely evaluate the fitted spline; fall back to the simplified polyline."""
        if self.spline is None or n < 2:
            return self.simplified.copy()
        tck, _u = self.spline
        t = np.linspace(0.0, 1.0, n)
        xs, ys = interpolate.splev(t, tck)
        return np.column_stack([xs, ys])


# ── Resampling ───────────────────────────────────────────────────────────────

def _arc_length_resample(points: np.ndarray, spacing: float, closed: bool) -> np.ndarray:
    if len(points) < 2:
        return points.copy()

    # For closed strokes, make sure the chain is explicitly closed.
    if closed and not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    diffs = np.diff(points, axis=0)
    segments = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(segments)])
    total = float(cumlen[-1])
    if total < max(spacing, 1e-6):
        return points.copy()

    n_samples = max(4, int(round(total / spacing)) + 1)
    t = np.linspace(0.0, total, n_samples)
    xs = np.interp(t, cumlen, points[:, 0])
    ys = np.interp(t, cumlen, points[:, 1])
    out = np.column_stack([xs, ys])
    if closed:
        out[-1] = out[0]
    return out


# ── Smoothing ────────────────────────────────────────────────────────────────

def _gaussian_smooth(points: np.ndarray, sigma: float, closed: bool) -> np.ndarray:
    if len(points) < 3 or sigma <= 0.0:
        return points.copy()

    if closed:
        core = points[:-1] if np.allclose(points[0], points[-1]) else points
        x = ndimage.gaussian_filter1d(core[:, 0], sigma, mode="wrap")
        y = ndimage.gaussian_filter1d(core[:, 1], sigma, mode="wrap")
        smoothed = np.column_stack([x, y])
        smoothed = np.vstack([smoothed, smoothed[0]])
    else:
        x = ndimage.gaussian_filter1d(points[:, 0], sigma, mode="reflect")
        y = ndimage.gaussian_filter1d(points[:, 1], sigma, mode="reflect")
        smoothed = np.column_stack([x, y])
        # Pin endpoints exactly — prevents smoothing from pulling them off the
        # natural pen landing/lift points.
        smoothed[0] = points[0]
        smoothed[-1] = points[-1]
    return smoothed


# ── RDP simplification ──────────────────────────────────────────────────────

def _point_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    norm = float(np.linalg.norm(d))
    if norm < 1e-9:
        return np.linalg.norm(points - a, axis=1)
    cross = d[0] * (points[:, 1] - a[1]) - d[1] * (points[:, 0] - a[0])
    return np.abs(cross) / norm


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    n = len(points)
    if n < 3 or epsilon <= 0.0:
        return points.copy()

    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True

    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 - i0 < 2:
            continue
        middle = points[i0 + 1:i1]
        d = _point_segment_distance(middle, points[i0], points[i1])
        idx = int(np.argmax(d))
        if d[idx] > epsilon:
            kept = i0 + 1 + idx
            keep[kept] = True
            stack.append((i0, kept))
            stack.append((kept, i1))

    return points[keep]


# ── Spline fitting ──────────────────────────────────────────────────────────

def _fit_spline(points: np.ndarray, closed: bool):
    n = len(points)
    if n < 2:
        return None
    try:
        if closed:
            pts = points
            if not np.allclose(pts[0], pts[-1]):
                pts = np.vstack([pts, pts[0]])
            k = max(1, min(3, len(pts) - 1))
            tck, u = interpolate.splprep(
                [pts[:, 0], pts[:, 1]], s=0.0, per=1, k=k,
            )
        else:
            k = max(1, min(3, n - 1))
            tck, u = interpolate.splprep(
                [points[:, 0], points[:, 1]], s=0.0, per=0, k=k,
            )
        return (tck, u)
    except Exception:
        return None


# ── Entry points ─────────────────────────────────────────────────────────────

def fit_stroke(
    stroke: Stroke,
    resample_spacing: float = 1.5,
    smooth_sigma: float = 1.2,
    simplify_epsilon: float = 0.6,
) -> FittedStroke:
    if not stroke.pixels:
        empty = np.empty((0, 2))
        return FittedStroke(
            id=stroke.id, raw_pixels=[], resampled=empty, smoothed=empty,
            simplified=empty, closed=stroke.closed,
            start_node=stroke.start_node, end_node=stroke.end_node, spline=None,
        )

    raw = np.array(stroke.pixels, dtype=float)
    resampled = _arc_length_resample(raw, resample_spacing, stroke.closed)
    smoothed = _gaussian_smooth(resampled, smooth_sigma, stroke.closed)
    simplified = _rdp(smoothed, simplify_epsilon)
    spline = _fit_spline(simplified, stroke.closed)

    return FittedStroke(
        id=stroke.id,
        raw_pixels=list(stroke.pixels),
        resampled=resampled,
        smoothed=smoothed,
        simplified=simplified,
        closed=stroke.closed,
        start_node=stroke.start_node,
        end_node=stroke.end_node,
        spline=spline,
    )


def fit_decomposition(decomp: StrokeDecomposition) -> list[FittedStroke]:
    return [fit_stroke(s) for s in decomp.strokes]


# ── Profile-guided refitting ────────────────────────────────────────────────
#
# These helpers consume the generic FittedStroke output and refit each stroke
# using a character-specific StrokeProfile (from stroke_profiles.py). The
# guiding idea: the generic smoother produces a faithful-but-bland centreline;
# the profile tells us which parts of that centreline should be straight, arc,
# or cubic-Bezier, and where the cusps are. Each segment is refit with a
# primitive matching its kind, then stitched with either sharp corners
# (cusp=True) or averaged endpoints (cusp=False) for tangent continuity.
#
# A quality gate compares the profile-refit against the generic fit via RMS
# deviation; if the profile fight produces something substantially worse, the
# caller falls back to the generic output for that stroke rather than force a
# bad shape. This keeps the pipeline safe to deploy on unseen hands.

def _discrete_curvature(pts: np.ndarray) -> np.ndarray:
    n = len(pts)
    k = np.zeros(n)
    if n < 3:
        return k
    v1 = pts[1:-1] - pts[:-2]
    v2 = pts[2:] - pts[1:-1]
    dot = (v1 * v2).sum(axis=1)
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    # arctan2(|cross|, dot) is numerically stable and gives the turn angle
    # at each interior vertex in [0, pi].
    k[1:-1] = np.arctan2(np.abs(cross), dot)
    return k


def _top_k_cusps(pts: np.ndarray, k: int, min_sep_frac: float = 0.08) -> list[int]:
    if k <= 0 or len(pts) < 3:
        return []
    curv = _discrete_curvature(pts)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1]) if cum[-1] > 0 else 1.0
    min_sep = total * min_sep_frac
    order = np.argsort(-curv)
    picked: list[int] = []
    for idx in order:
        i = int(idx)
        if i <= 0 or i >= len(pts) - 1:
            continue
        if all(abs(cum[i] - cum[p]) > min_sep for p in picked):
            picked.append(i)
            if len(picked) >= k:
                break
    picked.sort()
    return picked


def _fit_line(pts: np.ndarray, n: int = 8) -> np.ndarray:
    a, b = pts[0], pts[-1]
    t = np.linspace(0.0, 1.0, max(2, n))
    return np.outer(1.0 - t, a) + np.outer(t, b)


def _fit_arc(pts: np.ndarray, n: int = 48, closed: bool = False) -> np.ndarray:
    if len(pts) < 3:
        return pts.copy()
    x = pts[:, 0]
    y = pts[:, 1]
    # Algebraic circle fit: x^2 + y^2 + D x + E y + F = 0.
    A = np.column_stack([x, y, np.ones_like(x)])
    rhs = -(x ** 2 + y ** 2)
    try:
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return pts.copy()
    D, E, F = sol
    cx, cy = -D / 2.0, -E / 2.0
    r2 = cx * cx + cy * cy - F
    if r2 <= 0.0:
        return pts.copy()
    r = float(np.sqrt(r2))

    if closed:
        a0 = float(np.arctan2(y[0] - cy, x[0] - cx))
        # Follow the input chain's winding direction so the sampled circle
        # matches the way the pen traced the original loop.
        mid = pts[len(pts) // 2]
        am = float(np.arctan2(mid[1] - cy, mid[0] - cx))
        ccw = ((am - a0) % (2.0 * np.pi)) <= np.pi
        span = 2.0 * np.pi if ccw else -2.0 * np.pi
        angles = a0 + np.linspace(0.0, span, max(8, n))
    else:
        a0 = float(np.arctan2(y[0] - cy, x[0] - cx))
        a1 = float(np.arctan2(y[-1] - cy, x[-1] - cx))
        mid = pts[len(pts) // 2]
        am = float(np.arctan2(mid[1] - cy, mid[0] - cx))
        # Two possible arcs between a0 and a1; pick the one that passes through
        # the midpoint angle. Without this, a C-shape can render as the
        # complementary 270° arc.
        ccw_span = (a1 - a0) % (2.0 * np.pi)
        am_offset = (am - a0) % (2.0 * np.pi)
        ccw = am_offset <= ccw_span + 1e-9
        span = ccw_span if ccw else -((a0 - a1) % (2.0 * np.pi))
        angles = a0 + np.linspace(0.0, span, max(8, n))

    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])


def _fit_s_curve(pts: np.ndarray, n: int = 48) -> np.ndarray:
    if len(pts) < 4:
        return pts.copy()
    p0 = pts[0].astype(float)
    p3 = pts[-1].astype(float)
    # Endpoint tangents from a short look-in so one-pixel staircase doesn't
    # dominate the direction estimate.
    look = max(1, min(4, len(pts) // 4))
    t0 = pts[look] - p0
    t3 = p3 - pts[-1 - look]

    def _norm(v: np.ndarray) -> np.ndarray:
        nrm = float(np.linalg.norm(v))
        return v / nrm if nrm > 1e-9 else v

    t0 = _norm(t0)
    t3 = _norm(t3)
    chord = float(np.linalg.norm(p3 - p0))
    k = chord / 3.0
    p1 = p0 + k * t0
    p2 = p3 - k * t3

    t = np.linspace(0.0, 1.0, max(8, n))[:, None]
    bez = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 \
        + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3
    return bez


def _fit_by_kind(pts: np.ndarray, kind: str, closed: bool = False, n: int = 48) -> np.ndarray:
    if kind == "line":
        return _fit_line(pts, n=max(2, n // 4))
    if kind == "arc":
        # A pure circle is too rigid for closed handwritten loops (an 'o' is
        # a natural ellipse with personal variation, not a geometric circle).
        # For closed-loop single-arc strokes we pass the already-smoothed
        # polyline through unchanged — the profile still marks it as an arc
        # for coloring, we just don't force a primitive.
        if closed:
            return pts.copy()
        return _fit_arc(pts, n=n, closed=False)
    if kind == "s_curve":
        return _fit_s_curve(pts, n=n)
    return pts.copy()


def _rms_deviation(a: np.ndarray, b: np.ndarray) -> float:
    """Arc-length-resampled RMS distance between two polylines."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    m = max(len(a), len(b), 16)

    def _resample(arr: np.ndarray, count: int) -> np.ndarray:
        seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        if total < 1e-9:
            return np.repeat(arr[:1], count, axis=0)
        t = np.linspace(0.0, total, count)
        return np.column_stack([
            np.interp(t, cum, arr[:, 0]),
            np.interp(t, cum, arr[:, 1]),
        ])

    A = _resample(a, m)
    B = _resample(b, m)
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))


def _refit_stroke_with_profile(
    polyline: np.ndarray,
    profile,  # StrokeProfile, imported lazily by caller
) -> tuple[np.ndarray, list[tuple[int, int, str]]]:
    """Refit `polyline` using `profile`. Returns (points, spans) where each
    span is (start_idx, end_idx, kind) into the returned points array."""
    n_segs = len(profile.segments)
    if n_segs == 0 or len(polyline) < 2:
        return polyline.copy(), []

    splits_needed = n_segs - 1
    split_indices: list[int] = []
    if splits_needed > 0:
        split_indices = _top_k_cusps(polyline, splits_needed)
        # If curvature didn't surface enough peaks (e.g. a nearly-smooth input
        # against a cusp-heavy profile), fall back to evenly-spaced splits so
        # the profile structure still applies.
        while len(split_indices) < splits_needed:
            pos = len(split_indices) + 1
            idx = int(round(pos * (len(polyline) - 1) / (splits_needed + 1)))
            idx = max(1, min(len(polyline) - 2, idx))
            if idx in split_indices:
                idx = min(len(polyline) - 2, idx + 1)
                if idx in split_indices:
                    break
            split_indices.append(idx)
        split_indices = sorted(split_indices[:splits_needed])

    bounds = [0] + split_indices + [len(polyline) - 1]

    parts: list[np.ndarray] = []
    for i, seg in enumerate(profile.segments):
        a = bounds[i]
        b = bounds[i + 1]
        if b <= a:
            continue
        sub = polyline[a:b + 1]
        if len(sub) < 2:
            continue
        fitted = _fit_by_kind(sub, seg.kind, closed=(profile.closed and n_segs == 1))
        parts.append(fitted.astype(float))

    if not parts:
        return polyline.copy(), []

    # Stitch: adjacent segments share a boundary point. Cusp=True anchors the
    # shared point to the original polyline's split pixel (preserves corner);
    # cusp=False averages the two endpoints (encourages tangent continuity).
    for i in range(1, len(parts)):
        prev = parts[i - 1]
        cur = parts[i]
        cusp_idx = i - 1
        cusp = (
            profile.cusps_between[cusp_idx]
            if cusp_idx < len(profile.cusps_between)
            else False
        )
        if cusp and cusp_idx + 1 < len(bounds) - 1:
            anchor = polyline[bounds[i]].astype(float)
        else:
            anchor = (prev[-1] + cur[0]) / 2.0
        prev[-1] = anchor
        cur[0] = anchor

    combined_parts = [parts[0]]
    offsets = [0, len(parts[0]) - 1]
    for part in parts[1:]:
        combined_parts.append(part[1:])  # drop duplicated shared endpoint
        offsets.append(offsets[-1] + len(part) - 1)
    combined = np.vstack(combined_parts)

    spans: list[tuple[int, int, str]] = []
    for i in range(len(parts)):
        spans.append((offsets[i], offsets[i + 1], profile.segments[i].kind))
    return combined, spans


def _stroke_points(fs: FittedStroke) -> np.ndarray:
    """Dense polyline for a FittedStroke, used as the refit input."""
    if fs.spline is not None:
        return fs.sample_curve(n=128)
    return fs.simplified.copy()


def _endpoint_outgoing(pts: np.ndarray, at_end: bool, look: int = 4) -> Optional[np.ndarray]:
    """Unit vector pointing AWAY from the stroke at its chosen endpoint.

    at_end=True → tangent at pts[-1] pointing outward (continuation direction
    past the last pixel). at_end=False → tangent at pts[0] pointing outward
    (i.e. the direction the pen was coming FROM when it landed at pts[0]).
    """
    if len(pts) < 2:
        return None
    k = max(1, min(look, len(pts) - 1))
    if at_end:
        v = pts[-1] - pts[-1 - k]
    else:
        v = pts[0] - pts[k]
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def _merge_continuation_score(
    pi: np.ndarray, side_i: str, pj: np.ndarray, side_j: str,
) -> Optional[float]:
    """Angle (radians) between the outgoing tangent of pi at side_i and the
    incoming tangent of pj at side_j. 0 = straight-through continuation,
    pi = hairpin. Returns None when tangents can't be estimated."""
    ti = _endpoint_outgoing(pi, at_end=(side_i == "end"))
    # For a smooth continuation, the OUTGOING direction leaving pi should
    # be the OPPOSITE of the outgoing direction leaving pj (because if both
    # strokes "point away" from the shared junction, the pen's path through
    # is reversed between them). dot(ti, tj) ≈ -1 means smooth pass-through.
    tj = _endpoint_outgoing(pj, at_end=(side_j == "end"))
    if ti is None or tj is None:
        return None
    dot = float(np.clip(np.dot(ti, tj), -1.0, 1.0))
    # Smoothness angle: 0 when dot == -1 (straight-through), pi when dot == +1
    # (hairpin). Lower = better merge.
    return float(np.arccos(-dot))


def _merge_strokes_to_target(
    fitted: list[FittedStroke],
    target: int,
    endpoint_px_threshold: float = 12.0,
    max_merge_angle_rad: float = math.radians(80.0),
) -> list[dict]:
    """Greedy merge of open strokes until count == target.

    A pair is eligible if they share a graph junction node OR have endpoints
    within `endpoint_px_threshold` of each other, AND their continuation
    angle at that join is below `max_merge_angle_rad`. At each step the pair
    with the smallest continuation angle wins; ties break on endpoint
    distance.

    Each entry in the returned list is a surrogate dict with the fields
    _apply_profile needs: 'pts', 'closed', 'length'.
    """
    work: list[dict] = []
    for fs in fitted:
        pts = _stroke_points(fs)
        if len(pts) < 2:
            continue
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        work.append({
            "pts": pts.astype(float),
            "closed": bool(fs.closed),
            "start_node": fs.start_node,
            "end_node": fs.end_node,
            "length": length,
        })

    while len(work) > target:
        best: Optional[tuple[int, int, str, str, float, float]] = None
        for i in range(len(work)):
            if work[i]["closed"]:
                continue
            for j in range(i + 1, len(work)):
                if work[j]["closed"]:
                    continue
                for si in ("start", "end"):
                    for sj in ("start", "end"):
                        node_i = work[i][("start_node" if si == "start" else "end_node")]
                        node_j = work[j][("start_node" if sj == "start" else "end_node")]
                        pi = work[i]["pts"]
                        pj = work[j]["pts"]
                        epi = pi[0] if si == "start" else pi[-1]
                        epj = pj[0] if sj == "start" else pj[-1]
                        dist = float(np.linalg.norm(epi - epj))
                        shares = (
                            node_i is not None and node_j is not None
                            and node_i == node_j
                        )
                        if not shares and dist > endpoint_px_threshold:
                            continue
                        angle = _merge_continuation_score(pi, si, pj, sj)
                        if angle is None or angle > max_merge_angle_rad:
                            continue
                        # Primary key: smaller angle (smoother join). Tie-
                        # break on smaller endpoint distance so junction
                        # neighbours beat farther-away strokes.
                        key = (angle, dist)
                        if best is None or key < (best[4], best[5]):
                            best = (i, j, si, sj, angle, dist)
        if best is None:
            break

        i, j, si, sj, _ang, _dist = best
        pi = work[i]["pts"]
        pj = work[j]["pts"]
        # Orient so pi's active end is LAST and pj's active end is FIRST,
        # then concatenate with the duplicated shared endpoint dropped.
        if si == "start":
            pi = pi[::-1]
        if sj == "end":
            pj = pj[::-1]
        merged_pts = np.vstack([pi, pj[1:]])

        new_start_node = work[i][("end_node" if si == "start" else "start_node")]
        new_end_node = work[j][("end_node" if sj == "start" else "start_node")]
        merged_length = float(
            np.sum(np.linalg.norm(np.diff(merged_pts, axis=0), axis=1))
        )
        keep = min(i, j)
        drop = max(i, j)
        work[keep] = {
            "pts": merged_pts,
            "closed": False,
            "start_node": new_start_node,
            "end_node": new_end_node,
            "length": merged_length,
        }
        del work[drop]

    return work


def _apply_profile(fitted: list[FittedStroke], cprofile) -> Optional[list[dict]]:
    """Attempt a profile-guided refit of `fitted`. Returns a render-ready
    list, or None when the profile structure doesn't match the decomposition
    (caller falls back to the generic output).

    The first step is a continuity-based merge pass: a character's '$'-like
    S-curve that the skeletoniser has split into two halves at the crossing
    with the bar gets re-joined if the two halves share a junction AND
    their tangents align smoothly. Merging to the profile's expected count
    fixes length-based stroke assignment (without it, the straight bar
    beats a half-S on length and misclaims the 's_curve' slot)."""
    if not fitted or not cprofile.strokes:
        return None

    expected = len(cprofile.strokes)

    # 1. Merge continuations until we have at most `expected` strokes.
    work = _merge_strokes_to_target(fitted, expected)
    actual = len(work)
    if actual < expected:
        # Decomp + merge produced fewer strokes than the profile expects;
        # we'd have to synthesise splits which is out of scope. Fall back.
        return None
    if actual > expected and not cprofile.tolerant:
        return None

    # 2. Assign decomp strokes to profile strokes by descending length.
    # After the merge pass, length is a reliable proxy for "this is the
    # primary pen motion" because artifact branches have been absorbed.
    order = sorted(range(actual), key=lambda i: -work[i]["length"])
    primary = order[:expected]

    result: list[dict] = []
    for idx, sp in zip(primary, cprofile.strokes):
        generic_pts = work[idx]["pts"]
        length = work[idx]["length"]
        if len(generic_pts) < 2:
            continue

        refit_pts, spans = _refit_stroke_with_profile(generic_pts, sp)
        if len(refit_pts) < 2:
            refit_pts = generic_pts
            spans = []

        deviation = _rms_deviation(generic_pts, refit_pts)
        # Per-stroke quality gate: refit must stay within ~15% of the
        # stroke length (in RMS px) of the generic smoother, else drop
        # the structure constraint for that stroke.
        tolerance = max(2.0, 0.15 * max(length, 1.0))
        if deviation > tolerance:
            refit_pts = generic_pts
            spans = []

        coords = [(float(p[0]), float(p[1])) for p in refit_pts]
        stroke_dict: dict = {
            "points": coords,
            "closed": bool(work[idx]["closed"]) or sp.closed,
        }
        if spans:
            stroke_dict["segments"] = [
                {"start": int(s), "end": int(e), "kind": str(k)}
                for (s, e, k) in spans
            ]
        result.append(stroke_dict)

    # Leftover strokes (tolerant profiles where merge couldn't consolidate
    # to the expected count) are dropped: the profile is the authoritative
    # statement of how many pen motions this character has.
    return result if result else None


# ── Glue for external consumers (glyph_editor_app) ──────────────────────────

def fit_glyph_from_pil(img, char: "str | None" = None) -> list[dict]:
    """Run the full pipeline on a PIL.Image and return a render-ready path list.

    If `char` is provided, the character-aware ambiguity resolver is used to
    pick the best of several candidate decompositions, and — when a
    `stroke_profiles` entry exists for that character — each stroke is
    refit with its character-specific primitives (line/arc/s-curve) and
    cusps. Otherwise pure geometry decides (equivalent to the step-6
    greedy output).

    Returns a list of {"points": [(x, y), ...], "closed": bool} dicts in the
    source image's pixel coordinate space. When a profile is applied, each
    dict also carries "segments": [{"start", "end", "kind"}, ...] so the
    editor overlay can colour-code line/arc/s-curve runs.

    Returns [] if the glyph is too small or empty. Does not raise —
    callers can use the result directly.
    """
    from PIL import Image as _Image
    from skimage.morphology import skeletonize as _skeletonize

    from learn_font import binarize
    from stroke_graph import extract_stroke_graph
    from graph_cleanup import clean_graph
    from stroke_decompose import decompose

    try:
        if img.mode == "RGBA":
            bg = _Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img_rgb = bg
        else:
            img_rgb = img.convert("RGB")

        ink = binarize(img_rgb)
        skel = _skeletonize(ink > 0)
        raw = extract_stroke_graph(skel)
        cleaned, _audit = clean_graph(raw, ink)
        if char:
            from ambiguity import resolve
            decomp, _scored = resolve(cleaned, char=char)
        else:
            decomp = decompose(cleaned)
        fitted = fit_decomposition(decomp)
    except Exception:
        return []

    if char:
        try:
            from stroke_profiles import get_profile
            cprofile = get_profile(char)
        except Exception:
            cprofile = None
        if cprofile is not None:
            try:
                shaped = _apply_profile(fitted, cprofile)
            except Exception:
                shaped = None
            if shaped is not None:
                return shaped

    result: list[dict] = []
    for fs in fitted:
        pts = fs.sample_curve(n=96) if fs.spline is not None else fs.simplified
        if len(pts) < 2:
            continue
        coords = [(float(p[0]), float(p[1])) for p in pts]
        result.append({"points": coords, "closed": bool(fs.closed)})

    return result
