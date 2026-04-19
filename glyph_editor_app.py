#!/usr/bin/env python3
"""Interactive glyph editor.

Features
--------
- Load one or more glyph libraries (default: glyphs_hw,glyphs)
- Show all glyph variants in a scrollable grid
- Click a glyph to edit
- Drag glyph with mouse to shift it
- Arrow keys to nudge by 1px
- Save shifted glyph back to PNG
- Delete a variant and update index.json
- Grid auto-refreshes/rearranges after edits/deletes

Usage
-----
python glyph_editor_app.py
python glyph_editor_app.py --glyphs glyphs_hw,glyphs
python glyph_editor_app.py --glyphs glyphs_hw --glyphs glyphs
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None

try:
    from skimage.morphology import skeletonize as sk_skeletonize
except Exception:  # noqa: BLE001
    sk_skeletonize = None


@dataclass
class GlyphVariant:
    label: str
    lib_path: Path
    idx_path: Path
    rel_path: str
    abs_path: Path
    orig_h: int | None


@dataclass
class VectorizationResult:
    paths: list[list[tuple[float, float]]]
    svg_path: str
    binary_debug: Image.Image | None
    skeleton_debug: Image.Image | None
    graph_debug: Image.Image | None
    final_debug: Image.Image | None


def safe_label_dir(label: str) -> str:
    """Convert label to a Windows-safe directory name."""
    if label == "":
        return "empty"
    if len(label) == 1 and label.isalpha() and label.islower():
        return label
    if len(label) == 1 and label.isalpha() and label.isupper():
        return f"upper_{label}"

    named = {
        " ": "space",
        "/": "slash",
        "\\": "backslash",
        ":": "colon",
        "*": "asterisk",
        "?": "question",
        '"': "quote",
        "<": "lt",
        ">": "gt",
        "|": "pipe",
    }
    if label in named:
        return named[label]

    out = []
    for ch in label:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append(f"u{ord(ch):04x}")
    s = "".join(out).strip(".")
    return s or "glyph"


def parse_library_paths(raw_values: list[str]) -> list[Path]:
    base_dir = Path(__file__).resolve().parent
    out: list[Path] = []
    seen: set[str] = set()
    for raw in raw_values:
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue

            p = Path(part)
            if not p.is_absolute():
                # Resolve relative paths against this script's directory
                # so launching from another cwd still works.
                p = (base_dir / p).resolve()

            k = str(p)
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
    return out


def load_variants(library_paths: list[Path]) -> list[GlyphVariant]:
    variants: list[GlyphVariant] = []
    for lib in library_paths:
        idx_path = lib / "index.json"
        if not idx_path.exists():
            continue
        with open(idx_path, encoding="utf-8") as f:
            index = json.load(f)
        glyphs = index.get("glyphs", {})
        for label, entries in glyphs.items():
            for entry in entries:
                if isinstance(entry, dict):
                    rel = str(entry.get("path", ""))
                    orig_h = entry.get("orig_h")
                else:
                    rel = str(entry)
                    orig_h = None
                if not rel:
                    continue
                variants.append(
                    GlyphVariant(
                        label=label,
                        lib_path=lib,
                        idx_path=idx_path,
                        rel_path=rel,
                        abs_path=lib / rel,
                        orig_h=orig_h,
                    )
                )
    variants.sort(key=lambda v: (v.label, str(v.lib_path), v.rel_path))
    return variants


def shift_image_rgba(img: Image.Image, dx: int, dy: int) -> Image.Image:
    """Shift RGBA pixels inside same canvas size, clipping overflow."""
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(img, (dx, dy), img)
    return out


def scale_image_rgba(img: Image.Image, scale: float) -> Image.Image:
    """Scale RGBA glyph around its center and keep all pixels visible."""
    s = max(0.1, float(scale))
    if abs(s - 1.0) < 1e-6:
        return img.copy()

    nw = max(1, round(img.width * s))
    nh = max(1, round(img.height * s))
    scaled = img.resize((nw, nh), Image.NEAREST)

    out_w = max(img.width, nw)
    out_h = max(img.height, nh)
    out = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
    px = (out_w - nw) // 2
    py = (out_h - nh) // 2
    out.paste(scaled, (px, py), scaled)
    return out


def is_single_lower_ascii(label: str) -> bool:
    return len(label) == 1 and "a" <= label <= "z"


def _adaptive_block_size(w: int, h: int) -> int:
    """Return an odd adaptive-threshold block size based on glyph size."""
    base = max(15, int(min(w, h) * 0.25))
    if base % 2 == 0:
        base += 1
    return max(15, base)


def _smooth_open_path(points: list[tuple[float, float]], passes: int = 1) -> list[tuple[float, float]]:
    """Light endpoint-preserving smoothing for centerline paths."""
    out = points[:]
    for _ in range(max(0, passes)):
        if len(out) <= 2:
            break
        nxt = [out[0]]
        for i in range(1, len(out) - 1):
            x0, y0 = out[i - 1]
            x1, y1 = out[i]
            x2, y2 = out[i + 1]
            nxt.append((0.2 * x0 + 0.6 * x1 + 0.2 * x2, 0.2 * y0 + 0.6 * y1 + 0.2 * y2))
        nxt.append(out[-1])
        out = nxt
    return out


def _path_length(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(
        math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
        for i in range(1, len(points))
    )


def _resample_open_path(points: list[tuple[float, float]], step: float = 1.5) -> list[tuple[float, float]]:
    """Resample an open polyline at near-uniform spacing."""
    if len(points) <= 2 or step <= 1e-6:
        return points[:]

    out = [points[0]]
    acc = 0.0
    prev = points[0]
    i = 1
    while i < len(points):
        cur = points[i]
        seg_len = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        if seg_len <= 1e-9:
            i += 1
            continue
        if acc + seg_len >= step:
            t = (step - acc) / seg_len
            nx = prev[0] + (cur[0] - prev[0]) * t
            ny = prev[1] + (cur[1] - prev[1]) * t
            new_pt = (nx, ny)
            out.append(new_pt)
            prev = new_pt
            acc = 0.0
        else:
            acc += seg_len
            prev = cur
            i += 1

    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


def _resample_closed_path(points: list[tuple[float, float]], step: float = 2.0) -> list[tuple[float, float]]:
    if len(points) < 3 or step <= 1e-6:
        return points[:]
    loop = points[:]
    if loop[0] != loop[-1]:
        loop = loop + [loop[0]]

    out = [loop[0]]
    acc = 0.0
    prev = loop[0]
    i = 1
    while i < len(loop):
        cur = loop[i]
        seg_len = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        if seg_len <= 1e-9:
            i += 1
            continue
        if acc + seg_len >= step:
            t = (step - acc) / seg_len
            nx = prev[0] + (cur[0] - prev[0]) * t
            ny = prev[1] + (cur[1] - prev[1]) * t
            new_pt = (nx, ny)
            out.append(new_pt)
            prev = new_pt
            acc = 0.0
        else:
            acc += seg_len
            prev = cur
            i += 1

    if len(out) > 2 and out[-1] != out[0]:
        out.append(out[0])
    return out


def _simplify_path_dp(points: list[tuple[float, float]], eps_ratio: float = 0.015) -> list[tuple[float, float]]:
    """Simplify polyline with Douglas-Peucker using adaptive epsilon."""
    if cv2 is None or len(points) < 3:
        return points[:]
    arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    xs = arr[:, 0, 0]
    ys = arr[:, 0, 1]
    diag = math.hypot(float(np.max(xs) - np.min(xs)), float(np.max(ys) - np.min(ys)))
    epsilon = max(0.6, min(3.0, diag * eps_ratio))
    approx = cv2.approxPolyDP(arr, epsilon=epsilon, closed=False)
    out = [(float(p[0][0]), float(p[0][1])) for p in approx]
    if len(out) < 2:
        return points[:]
    return out


def _fit_catmull_rom_cubic(
    points: list[tuple[float, float]],
    samples_per_seg: int = 8,
) -> tuple[list[tuple[float, float]], str]:
    """Fit smooth cubic curve via Catmull-Rom conversion and return sampled points + SVG C commands."""
    if len(points) < 2:
        return points[:], ""

    if len(points) == 2:
        svg = f"M {points[0][0]:.2f} {points[0][1]:.2f} L {points[1][0]:.2f} {points[1][1]:.2f}"
        return points[:], svg

    sampled: list[tuple[float, float]] = [points[0]]
    cmds = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]

    def sample_cubic(
        p0: tuple[float, float],
        c1: tuple[float, float],
        c2: tuple[float, float],
        p3: tuple[float, float],
        steps: int,
    ) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for i in range(1, steps + 1):
            t = i / steps
            u = 1.0 - t
            x = (u * u * u) * p0[0] + 3 * (u * u) * t * c1[0] + 3 * u * (t * t) * c2[0] + (t * t * t) * p3[0]
            y = (u * u * u) * p0[1] + 3 * (u * u) * t * c1[1] + 3 * u * (t * t) * c2[1] + (t * t * t) * p3[1]
            out.append((x, y))
        return out

    n = len(points)
    for i in range(n - 1):
        p0 = points[i - 1] if i - 1 >= 0 else points[i]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[i + 2] if i + 2 < n else points[i + 1]

        c1 = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)

        cmds.append(f"C {c1[0]:.2f} {c1[1]:.2f} {c2[0]:.2f} {c2[1]:.2f} {p2[0]:.2f} {p2[1]:.2f}")
        sampled.extend(sample_cubic(p1, c1, c2, p2, max(4, samples_per_seg)))

    return sampled, " ".join(cmds)


def _paths_to_svg(paths: list[list[tuple[float, float]]]) -> str:
    chunks: list[str] = []
    for p in paths:
        if len(p) < 2:
            continue
        cmd = [f"M {p[0][0]:.2f} {p[0][1]:.2f}"]
        for x, y in p[1:]:
            cmd.append(f"L {x:.2f} {y:.2f}")
        chunks.append(" ".join(cmd))
    return " ".join(chunks)


def _draw_processing_overlay(
    size: tuple[int, int],
    raw_paths: list[list[tuple[float, float]]],
    simplified_paths: list[list[tuple[float, float]]],
    final_paths: list[list[tuple[float, float]]],
) -> Image.Image:
    h, w = size
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    for p in raw_paths:
        if len(p) < 2:
            continue
        arr = np.array(p, dtype=np.float32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [arr.astype(np.int32)], False, (180, 180, 180), 1, lineType=cv2.LINE_8)

    for p in simplified_paths:
        if len(p) < 2:
            continue
        arr = np.array(p, dtype=np.float32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [arr.astype(np.int32)], False, (220, 140, 40), 1, lineType=cv2.LINE_AA)

    for p in final_paths:
        if len(p) < 2:
            continue
        arr = np.array(p, dtype=np.float32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [arr.astype(np.int32)], False, (40, 40, 40), 1, lineType=cv2.LINE_AA)

    return Image.fromarray(canvas, mode="RGB")


def _draw_inferred_graph_debug(size: tuple[int, int], paths: list[list[tuple[float, float]]]) -> Image.Image:
    h, w = size
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    for p in paths:
        if len(p) < 2:
            continue
        arr = np.array(p, dtype=np.float32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [arr.astype(np.int32)], False, (70, 70, 70), 1, lineType=cv2.LINE_AA)
    return Image.fromarray(canvas, mode="RGB")


def _reconstruct_loop_midpath(
    outer: np.ndarray,
    inner: np.ndarray,
    angle_samples: int = 80,
) -> list[tuple[float, float]]:
    """Build a loop-like center path midway between outer and inner contours."""
    op = outer[:, 0, :].astype(np.float32)
    ip = inner[:, 0, :].astype(np.float32)
    if op.shape[0] < 8 or ip.shape[0] < 8:
        return []

    cx = float(np.mean(op[:, 0]))
    cy = float(np.mean(op[:, 1]))

    oa = np.arctan2(op[:, 1] - cy, op[:, 0] - cx)
    orad = np.hypot(op[:, 0] - cx, op[:, 1] - cy)
    ia = np.arctan2(ip[:, 1] - cy, ip[:, 0] - cx)
    irad = np.hypot(ip[:, 0] - cx, ip[:, 1] - cy)

    def angdiff(a: np.ndarray, b: float) -> np.ndarray:
        d = a - b
        return np.abs((d + np.pi) % (2 * np.pi) - np.pi)

    pts: list[tuple[float, float]] = []
    for t in np.linspace(-math.pi, math.pi, num=max(24, angle_samples), endpoint=False):
        oi = int(np.argmin(angdiff(oa, float(t))))
        ii = int(np.argmin(angdiff(ia, float(t))))
        ro = float(orad[oi])
        ri = float(irad[ii])
        if ro <= ri + 0.5:
            continue
        r = 0.5 * (ro + ri)
        pts.append((cx + r * math.cos(float(t)), cy + r * math.sin(float(t))))

    if len(pts) < 8:
        return []
    pts = _resample_closed_path(pts, step=2.0)
    return pts


def _loop_paths_from_counters(cleaned: np.ndarray) -> list[list[tuple[float, float]]]:
    """Detect counters/holes and reconstruct loop-like center paths."""
    contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or len(contours) == 0:
        return []
    h = hierarchy[0]
    loops: list[list[tuple[float, float]]] = []
    for i, c in enumerate(contours):
        # Only process top-level outer contours
        parent = int(h[i][3])
        child = int(h[i][2])
        if parent != -1 or child == -1:
            continue
        outer_area = cv2.contourArea(c)
        if outer_area < 25:
            continue

        # Use largest child hole as the counter for loop reconstruction.
        best_child = -1
        best_area = 0.0
        ci = child
        while ci != -1:
            a = abs(float(cv2.contourArea(contours[ci])))
            if a > best_area:
                best_area = a
                best_child = ci
            ci = int(h[ci][0])

        if best_child == -1:
            continue
        if best_area < max(8.0, outer_area * 0.02):
            continue

        loop = _reconstruct_loop_midpath(c, contours[best_child], angle_samples=96)
        if len(loop) >= 8:
            loops.append(loop)
    return loops


def _center_path_inside_ink(
    points: list[tuple[float, float]],
    dist_map: np.ndarray,
    iterations: int = 2,
) -> list[tuple[float, float]]:
    """Nudge path toward medial center of ink while preserving continuity."""
    if len(points) < 3:
        return points[:]

    h, w = dist_map.shape
    out = points[:]
    for _ in range(max(0, iterations)):
        nxt = [out[0]]
        for i in range(1, len(out) - 1):
            x, y = out[i]
            best = (x, y)
            best_score = -1e9
            sx = int(round(x))
            sy = int(round(y))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    xx = sx + dx
                    yy = sy + dy
                    if xx < 0 or yy < 0 or xx >= w or yy >= h:
                        continue
                    # favor larger stroke-center distance and small displacement
                    score = float(dist_map[yy, xx]) - 0.25 * math.hypot(xx - x, yy - y)
                    if score > best_score:
                        best_score = score
                        best = (float(xx), float(yy))
            nxt.append(best)
        nxt.append(out[-1])
        out = _smooth_open_path(nxt, passes=1)
    return out


def _make_binary_debug_image(binary_ink: np.ndarray) -> Image.Image:
    # Show classic black-ink on white-paper debug image.
    show = np.where(binary_ink > 0, 0, 255).astype(np.uint8)
    return Image.fromarray(show, mode="L").convert("RGB")


def _draw_skeleton_debug(skel: np.ndarray) -> Image.Image:
    show = np.where(skel, 0, 255).astype(np.uint8)
    return Image.fromarray(show, mode="L").convert("RGB")


def _draw_graph_overlay(
    skel: np.ndarray,
    endpoints: set[tuple[int, int]],
    junctions: set[tuple[int, int]],
) -> Image.Image:
    h, w = skel.shape
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    canvas[skel] = (80, 80, 80)
    for x, y in endpoints:
        cv2.circle(canvas, (x, y), 2, (220, 30, 30), -1, lineType=cv2.LINE_AA)
    for x, y in junctions:
        cv2.circle(canvas, (x, y), 2, (30, 70, 220), -1, lineType=cv2.LINE_AA)
    return Image.fromarray(canvas, mode="RGB")


# Colour palette – up to 8 distinct colours for per-stroke debug rendering.
_STROKE_COLORS: list[tuple[int, int, int]] = [
    (220, 30, 30),    # red
    (30, 100, 220),   # blue
    (20, 170, 60),    # green
    (200, 120, 10),   # orange
    (150, 30, 200),   # purple
    (20, 180, 180),   # teal
    (200, 80, 145),   # pink
    (100, 100, 100),  # grey
]


def _draw_final_path_debug(size: tuple[int, int], points: list[tuple[float, float]]) -> Image.Image:
    h, w = size
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    if len(points) >= 2:
        arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [arr.astype(np.int32)], False, (40, 40, 40), 1, lineType=cv2.LINE_AA)
    return Image.fromarray(canvas, mode="RGB")


def _coalesce_junction_clusters(
    paths: list[list[tuple[int, int]]],
    junctions: set[tuple[int, int]],
    max_cluster_path: float = 12.0,
) -> tuple[list[list[tuple[int, int]]], set[tuple[int, int]]]:
    """Collapse nearby-junction clusters caused by thick ink into single nodes.

    When skeletonize() processes thick ink it often produces a *cluster* of
    adjacent junction pixels instead of one clean branching point.  The short
    segments that join those junction pixels prevent the later selective-merge
    step from connecting the arm paths across the cluster.

    This function:
      1. Finds clusters by treating any junction-to-junction path whose arc
         length < max_cluster_path as an intra-cluster edge.
      2. Replaces each cluster with its centroid.
      3. Re-routes all path endpoints that were cluster members to the centroid.
      4. Discards intra-cluster paths (they become zero-length self-loops).
    """
    if not junctions:
        return paths, junctions

    j_list = list(junctions)
    parent: dict[tuple[int, int], tuple[int, int]] = {j: j for j in j_list}

    def find(j: tuple[int, int]) -> tuple[int, int]:
        while parent[j] != j:
            parent[j] = parent[parent[j]]
            j = parent[j]
        return j

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Union junctions connected by short paths AND that are spatially close.
    # The spatial distance constraint prevents junctions spread around a ring
    # (e.g. 'o' skeleton) from all merging via transitive union-find closure.
    # Only junctions that are BOTH close in arc length AND in Euclidean space
    # represent genuine thick-ink pixel clusters that should be collapsed.
    for p in paths:
        if p[0] not in junctions or p[-1] not in junctions:
            continue
        eucl = math.hypot(p[0][0] - p[-1][0], p[0][1] - p[-1][1])
        if eucl >= 4.0:
            continue  # spatially separate — structural junction, keep distinct
        arc = sum(
            math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1])
            for i in range(1, len(p))
        )
        if arc < max_cluster_path:
            union(p[0], p[-1])

    # Build centroid for each cluster.
    clusters: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for j in j_list:
        clusters.setdefault(find(j), []).append(j)

    j_to_center: dict[tuple[int, int], tuple[int, int]] = {}
    for members in clusters.values():
        cx = int(round(sum(m[0] for m in members) / len(members)))
        cy = int(round(sum(m[1] for m in members) / len(members)))
        center = (cx, cy)
        for m in members:
            j_to_center[m] = center

    def reroute(pt: tuple[int, int]) -> tuple[int, int]:
        return j_to_center.get(pt, pt)

    new_paths: list[list[tuple[int, int]]] = []
    for p in paths:
        p0 = reroute(p[0])
        pn = reroute(p[-1])
        if p0 == pn:
            continue  # genuine intra-cluster artefact — discard
        new_paths.append([p0] + list(p[1:-1]) + [pn])

    new_junctions = set(j_to_center.values())
    return new_paths, new_junctions


def _draw_final_paths_debug(size: tuple[int, int], paths: list[list[tuple[float, float]]]) -> Image.Image:
    """Render final stroke segments with one distinct colour per stroke."""
    h, w = size
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    for i, p in enumerate(paths):
        if len(p) < 2:
            continue
        color = _STROKE_COLORS[i % len(_STROKE_COLORS)]
        arr = np.array(p, dtype=np.float32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [arr.astype(np.int32)], False, color, 2, lineType=cv2.LINE_AA)
        # Mark start (square) and end (circle) of each stroke.
        sx, sy = int(round(p[0][0])), int(round(p[0][1]))
        ex, ey = int(round(p[-1][0])), int(round(p[-1][1]))
        cv2.rectangle(canvas, (sx - 2, sy - 2), (sx + 2, sy + 2), color, -1)
        cv2.circle(canvas, (ex, ey), 3, color, -1, lineType=cv2.LINE_AA)
    return Image.fromarray(canvas, mode="RGB")


def _skel_neighbors(p: tuple[int, int], pixels: set[tuple[int, int]]) -> list[tuple[int, int]]:
    x, y = p
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            q = (x + dx, y + dy)
            if q in pixels:
                out.append(q)
    return out


def _prune_spurs(skel: np.ndarray, spur_len: int = 8) -> np.ndarray:
    """Prune short endpoint branches attached to junctions."""
    out = skel.copy().astype(bool)
    changed = True
    while changed:
        changed = False
        ys, xs = np.where(out)
        pixels: set[tuple[int, int]] = set(zip(xs.tolist(), ys.tolist()))
        if not pixels:
            break
        adj: dict[tuple[int, int], list[tuple[int, int]]] = {p: _skel_neighbors(p, pixels) for p in pixels}
        deg = {p: len(adj[p]) for p in pixels}
        endpoints = [p for p, d in deg.items() if d == 1]

        to_remove: set[tuple[int, int]] = set()
        for ep in endpoints:
            path = [ep]
            prev = None
            cur = ep
            steps = 0
            while True:
                nbrs = [n for n in adj[cur] if n != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                path.append(nxt)
                steps += 1
                prev, cur = cur, nxt
                if deg[cur] != 2:
                    break
                if steps > spur_len:
                    break

            if steps <= spur_len and len(path) > 1 and deg.get(path[-1], 0) >= 3:
                # Keep the junction node, remove short spur pixels leading to it.
                to_remove.update(path[:-1])

        if to_remove:
            changed = True
            for x, y in to_remove:
                out[y, x] = False

    return out


def _endpoint_direction(path: list[tuple[int, int]], at_start: bool) -> tuple[float, float] | None:
    if len(path) < 2:
        return None
    if at_start:
        x0, y0 = path[0]
        x1, y1 = path[1]
    else:
        x0, y0 = path[-1]
        x1, y1 = path[-2]
    vx = float(x1 - x0)
    vy = float(y1 - y0)
    n = math.hypot(vx, vy)
    if n <= 1e-6:
        return None
    return (vx / n, vy / n)


def _selective_merge_at_junctions(
    paths: list[list[tuple[int, int]]],
    junctions: set[tuple[int, int]],
    dot_threshold: float = -0.5,
) -> list[list[tuple[int, int]]]:
    """Merge skeleton segments through junctions ONLY when angle-continuity warrants it.

    Two segments at a junction are merged if their directions away from the
    junction are nearly opposite (dot < dot_threshold, i.e. roughly collinear).
    Perpendicular or branching segments are left as separate strokes.

    This preserves:
    - horizontal bars of 't', 'f'
    - crossing diagonals of 'x'
    - branches in '4', 'k', '$'
    while still merging smooth continuations in 's', 'n', 'u', etc.
    """
    work: list[list[tuple[int, int]] | None] = [p[:] for p in paths]

    changed = True
    while changed:
        changed = False
        for j in junctions:
            incidences: list[tuple[int, bool, tuple[float, float]]] = []
            for i, p in enumerate(work):
                if p is None or len(p) < 2:
                    continue
                if p[0] == j:
                    d = _endpoint_direction(p, at_start=True)
                    if d is not None:
                        incidences.append((i, True, d))
                if p[-1] == j:
                    d = _endpoint_direction(p, at_start=False)
                    if d is not None:
                        incidences.append((i, False, d))

            if len(incidences) < 2:
                continue

            best_pair: tuple[int, bool, int, bool] | None = None
            best_dot = float("inf")
            for a in range(len(incidences)):
                for b in range(a + 1, len(incidences)):
                    ia, sa, da = incidences[a]
                    ib, sb, db = incidences[b]
                    if ia == ib:
                        continue
                    dot = da[0] * db[0] + da[1] * db[1]
                    if dot < best_dot:
                        best_dot = dot
                        best_pair = (ia, sa, ib, sb)

            # Only merge when the best pair is nearly collinear.
            if best_pair is None or best_dot > dot_threshold:
                continue

            ia, sa, ib, sb = best_pair
            pa = work[ia]
            pb = work[ib]
            if pa is None or pb is None:
                continue

            a_seg = list(reversed(pa)) if sa else pa[:]
            b_seg = pb[:] if sb else list(reversed(pb))

            merged = a_seg + b_seg[1:]
            work[ia] = merged
            work[ib] = None
            changed = True
            break

    return [p for p in work if p is not None and len(p) >= 2]


def _extract_skeleton_paths(skel: np.ndarray) -> tuple[
    list[list[tuple[int, int]]],
    set[tuple[int, int]],
    set[tuple[int, int]],
]:
    ys, xs = np.where(skel)
    if xs.size == 0:
        return [], set(), set()

    pixels: set[tuple[int, int]] = set(zip(xs.tolist(), ys.tolist()))
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {p: _skel_neighbors(p, pixels) for p in pixels}
    deg = {p: len(adj[p]) for p in pixels}
    endpoints = {p for p, d in deg.items() if d == 1}
    junctions = {p for p, d in deg.items() if d >= 3}
    specials = endpoints | junctions

    def edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        return (a, b) if a <= b else (b, a)

    used_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    paths: list[list[tuple[int, int]]] = []

    for s in specials:
        for n in adj[s]:
            ek = edge_key(s, n)
            if ek in used_edges:
                continue
            used_edges.add(ek)
            path = [s]
            prev = s
            cur = n
            while True:
                path.append(cur)
                if cur in specials and cur != s:
                    break
                nbrs = [q for q in adj[cur] if q != prev]
                if not nbrs:
                    break
                nxt = None
                for q in nbrs:
                    ek2 = edge_key(cur, q)
                    if ek2 not in used_edges:
                        nxt = q
                        break
                if nxt is None:
                    nxt = nbrs[0]
                ek2 = edge_key(cur, nxt)
                if ek2 in used_edges:
                    break
                used_edges.add(ek2)
                prev, cur = cur, nxt
            if len(path) >= 2:
                paths.append(path)

    for p in pixels:
        for n in adj[p]:
            ek = edge_key(p, n)
            if ek in used_edges:
                continue
            used_edges.add(ek)
            path = [p, n]
            prev = p
            cur = n
            while True:
                nbrs = [q for q in adj[cur] if q != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                ek2 = edge_key(cur, nxt)
                if ek2 in used_edges:
                    break
                used_edges.add(ek2)
                path.append(nxt)
                prev, cur = cur, nxt
                if cur == p:
                    break
            if len(path) >= 2:
                paths.append(path)

    return paths, endpoints, junctions


def vectorize_glyph_from_alpha(
    img: Image.Image,
) -> VectorizationResult:
    """Vectorize glyph into centerline skeleton paths suitable for plotting."""
    if cv2 is None or sk_skeletonize is None:
        return VectorizationResult([], "", None, None, None, None)

    rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = (rgba[:, :, 3].astype(np.float32) / 255.0)[..., None]
    # Composite over white so transparent background doesn't collapse to black.
    comp = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    gray = cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY)

    # 1) Binarization (adaptive threshold)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    block_size = _adaptive_block_size(gray.shape[1], gray.shape[0])
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        7,
    )

    # 2) Noise cleaning (open/close + remove small contours)
    k = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_min = max(12.0, float(gray.shape[0] * gray.shape[1]) * 0.001)
    mask = np.zeros_like(cleaned)
    for c in contours:
        if cv2.contourArea(c) >= area_min:
            cv2.drawContours(mask, [c], -1, 255, thickness=-1)
    cleaned = mask

    # Glyph-size proportional thresholds used throughout the rest of the pipeline.
    size_diag = math.hypot(gray.shape[1], gray.shape[0])
    min_stroke_len = max(6.0, size_diag * 0.07)

    # 3) Skeletonization to centerline (weak cue)
    skel = sk_skeletonize(cleaned > 0)
    skel = _prune_spurs(skel, spur_len=8)

    # 4) Graph extraction + SELECTIVE merging (allow multiple strokes).
    raw_paths, endpoints, junctions = _extract_skeleton_paths(skel)
    # Coalesce nearby junction clusters (thick ink produces clusters of adjacent
    # junction pixels rather than one clean branching node).  Collapsing the
    # cluster to its centroid lets the selective merge correctly route arm paths
    # through the crossing zone.
    raw_paths, junctions = _coalesce_junction_clusters(raw_paths, junctions, max_cluster_path=12.0)
    # Only merge through a junction when the two segments are nearly collinear
    # (dot < -0.5 ≈ angle > 120°).  Perpendicular branches (t-bar, crossings,
    # 4-stem, $-vertical) are left as separate stroke segments.
    merged_paths = _selective_merge_at_junctions(raw_paths, junctions, dot_threshold=-0.5)

    # Drop short junction-to-junction segments that survive after coalescing
    # (any remaining inter-cluster artefacts).
    inter_junc_max = max(10.0, size_diag * 0.10)
    merged_paths = [
        p for p in merged_paths
        if not (
            p[0] in junctions and p[-1] in junctions
            and _path_length([(float(px2), float(py2)) for px2, py2 in p]) < inter_junc_max
        )
    ]

    # Classify skeleton results: cyclic paths (loop closed back on itself, e.g.
    # the oval ring in 'o') vs open chains (endpoint-to-endpoint or tail strokes).
    skel_cyclic: list[list[tuple[int, int]]] = [
        p for p in merged_paths if len(p) >= 4 and p[0] == p[-1]
    ]
    skel_open: list[list[tuple[int, int]]] = [
        p for p in merged_paths if not (len(p) >= 4 and p[0] == p[-1])
    ]

    # 5) Loop priors: detect enclosed counters (o, 0, e, 9 oval, a, g, …).
    loop_paths = _loop_paths_from_counters(cleaned)

    if loop_paths:
        # Counter-based midpath is geometrically more accurate than the
        # skeleton ring.  Drop:
        #  • all cyclic skeleton paths (the loop itself, better represented by midpath)
        #  • open skeleton paths whose endpoints are close together relative to
        #    their length (loop arcs / partial rings, NOT genuine tails)
        def _is_loop_arc(fp: list[tuple[float, float]]) -> bool:
            """True if path is a circular/curved arc rather than a linear tail."""
            plen = _path_length(fp)
            if plen < 1.0:
                return True
            endpoint_dist = math.hypot(fp[0][0] - fp[-1][0], fp[0][1] - fp[-1][1])
            # Tails are roughly straight-ish: their endpoint separation is a
            # large fraction of their arc length.  Curved arcs have endpoints
            # much closer than their arc length.
            return endpoint_dist < plen * 0.45

        skeleton_tails: list[list[tuple[float, float]]] = []
        for p in skel_open:
            fp = [(float(x), float(y)) for x, y in p]
            if _path_length(fp) >= min_stroke_len and not _is_loop_arc(fp):
                skeleton_tails.append(fp)
        inferred_paths: list[list[tuple[float, float]]] = skeleton_tails + loop_paths
    else:
        # No counter loops — use all skeleton paths (cycles + opens),
        # filtered by minimum stroke length to discard noise fragments.
        inferred_paths = [
            [(float(x), float(y)) for x, y in p]
            for p in merged_paths
            if _path_length([(float(x), float(y)) for x, y in p]) >= min_stroke_len
        ]

    size2d = (gray.shape[0], gray.shape[1])
    dbg_bin = _make_binary_debug_image(cleaned)
    # Topology panel: skeleton pixels + endpoint (red) and junction (blue) markers.
    dbg_topo = _draw_graph_overlay(skel, endpoints, junctions)

    if not inferred_paths:
        dbg_decomp = _draw_final_paths_debug(size2d, [])
        dbg_final = _draw_final_paths_debug(size2d, [])
        return VectorizationResult([], "", dbg_bin, dbg_topo, dbg_decomp, dbg_final)

    # Decomposition debug: show inferred strokes in per-stroke colours BEFORE fitting.
    dbg_decomp = _draw_final_paths_debug(size2d, inferred_paths)

    # 6) Post-processing per stroke: resample → smooth → center-seek → simplify → curve fit.
    paths: list[list[tuple[float, float]]] = []
    svg_chunks: list[str] = []
    dist_map = cv2.distanceTransform((cleaned > 0).astype(np.uint8), cv2.DIST_L2, 3)

    for p in inferred_paths:
        if len(p) < 2 or _path_length(p) < 4.0:
            continue

        pr = _resample_open_path(p, step=1.4)
        ps = _smooth_open_path(pr, passes=1)
        pc = _center_path_inside_ink(ps, dist_map, iterations=2)
        simp = _simplify_path_dp(pc, eps_ratio=0.015)
        if len(simp) < 2:
            continue

        final_pts, svg_curve = _fit_catmull_rom_cubic(simp, samples_per_seg=8)
        if len(final_pts) >= 2:
            paths.append(final_pts)
            if svg_curve:
                svg_chunks.append(svg_curve)

    # 7) SVG output: one sub-path per stroke (no fill, open lines).
    svg_path = " ".join(svg_chunks) if svg_chunks else _paths_to_svg(paths)

    # Final debug: fitted strokes in per-stroke colours.
    dbg_final = _draw_final_paths_debug(size2d, paths)
    return VectorizationResult(paths, svg_path, dbg_bin, dbg_topo, dbg_decomp, dbg_final)


def _fit_glyph_strokes(img: Image.Image, char: str | None = None) -> list[dict]:
    """Run the external graph->cleanup->decompose->fit pipeline on *img*.

    `char` is the recognized label; when provided, the ambiguity resolver
    uses a soft character-specific prior to break ties between otherwise
    equivalent decompositions.

    Returns [{"points": [(x, y), ...], "closed": bool}, ...] in source-image
    pixel coordinates. Returns [] if the pipeline or its deps are unavailable.
    """
    try:
        from stroke_fit import fit_glyph_from_pil
    except Exception:
        return []
    return fit_glyph_from_pil(img, char=char)


class GlyphEditorApp:
    def __init__(self, root: tk.Tk, library_paths: list[Path]) -> None:
        self.root = root
        self.root.title("Glyph Editor")
        self.root.geometry("1300x860")

        self.library_paths = library_paths
        self.variants: list[GlyphVariant] = []
        self.filtered_indices: list[int] = []

        self.cell_w = 120
        self.cell_h = 130
        self.thumb_box = 90
        self.grid_cols = 8

        self.selected_global_idx: int | None = None
        self.editor_orig_img: Image.Image | None = None
        self.editor_scale = 1.0
        self.editor_offset_x = 0.0
        self.editor_offset_y = 0.0
        self.glyph_user_scale = 1.0
        self.drag_last_x = 0
        self.drag_last_y = 0
        self.pending_auto_fit = False
        self._suppress_scale_callback = False
        self.last_guide_bbox: tuple[int, int, int, int] | None = None
        self.last_baseline_y = 0
        self.editor_vector_paths: list[list[tuple[float, float]]] = []
        self.editor_vector_svg_path = ""
        # Fitted strokes from the graph->cleanup->decompose->fit pipeline.
        # Each entry: {"points": [(x, y), ...], "closed": bool} in source-image pixels.
        self.editor_fitted_strokes: list[dict] = []

        self.ref_orig_h = 30.0
        self.lowercase_floor_ratio = 0.72
        self.guide_preview_height_ratio = 0.56
        self.label_target_orig_h: dict[str, float] = {}

        self.grid_photo_refs: list[ImageTk.PhotoImage] = []
        self.editor_photo_ref: ImageTk.PhotoImage | None = None
        self.vector_debug_photo_refs: list[ImageTk.PhotoImage] = []
        self.vector_debug_labels: list[ttk.Label] = []

        # Font-guide overlay state (reference letter behind glyph)
        families = sorted(set(tkfont.families(self.root)))
        fallback_family = "Times New Roman" if "Times New Roman" in families else (
            "Arial" if "Arial" in families else (families[0] if families else "TkDefaultFont")
        )
        self.guide_show_var = tk.BooleanVar(value=True)
        self.guide_family = fallback_family
        self.auto_fit_var = tk.BooleanVar(value=True)
        self.vector_show_var = tk.BooleanVar(value=(cv2 is not None and sk_skeletonize is not None))

        self.glyph_scale_var = tk.DoubleVar(value=1.0)
        self.glyph_scale_display_var = tk.StringVar(value="1.00x")

        self._build_ui()
        self.reload_variants()

    def _build_ui(self) -> None:
        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=3)
        paned.add(right, weight=2)

        top = ttk.Frame(left)
        top.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(top, text="Libraries:").pack(side=tk.LEFT)
        self.libs_var = tk.StringVar(value=",".join(str(p) for p in self.library_paths))
        self.libs_entry = ttk.Entry(top, textvariable=self.libs_var, width=70)
        self.libs_entry.pack(side=tk.LEFT, padx=(6, 8), fill=tk.X, expand=True)

        ttk.Button(top, text="Reload", command=self.on_reload_clicked).pack(side=tk.LEFT)

        ttk.Label(top, text="Filter:").pack(side=tk.LEFT, padx=(12, 4))
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", lambda *_: self.apply_filter())
        ttk.Entry(top, textvariable=self.filter_var, width=18).pack(side=tk.LEFT)

        self.count_var = tk.StringVar(value="0 glyphs")
        ttk.Label(top, textvariable=self.count_var).pack(side=tk.LEFT, padx=(12, 0))

        grid_frame = ttk.Frame(left)
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.grid_canvas = tk.Canvas(grid_frame, bg="#eeeeee", highlightthickness=0)
        self.grid_scroll = ttk.Scrollbar(grid_frame, orient=tk.VERTICAL, command=self.grid_canvas.yview)
        self.grid_canvas.configure(yscrollcommand=self.grid_scroll.set)
        self.grid_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.grid_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.grid_canvas.bind("<Button-1>", self.on_grid_click)
        self.grid_canvas.bind("<Configure>", self.on_grid_resize)
        self.grid_canvas.bind_all("<MouseWheel>", self.on_mousewheel)

        # Right panel
        info = ttk.LabelFrame(right, text="Selection")
        info.pack(fill=tk.X, padx=8, pady=8)

        self.sel_label_var = tk.StringVar(value="Label: -")
        self.sel_lib_var = tk.StringVar(value="Library: -")
        self.sel_path_var = tk.StringVar(value="Path: -")
        self.sel_size_var = tk.StringVar(value="Size: -")
        self.sel_offset_var = tk.StringVar(value="Offset: (0, 0)")
        self.new_label_var = tk.StringVar(value="")

        ttk.Label(info, textvariable=self.sel_label_var).pack(anchor="w", padx=8, pady=(8, 2))
        ttk.Label(info, textvariable=self.sel_lib_var).pack(anchor="w", padx=8, pady=2)
        ttk.Label(info, textvariable=self.sel_path_var, wraplength=460).pack(anchor="w", padx=8, pady=2)
        ttk.Label(info, textvariable=self.sel_size_var).pack(anchor="w", padx=8, pady=2)
        ttk.Label(info, textvariable=self.sel_offset_var).pack(anchor="w", padx=8, pady=(2, 8))

        relabel_row = ttk.Frame(info)
        relabel_row.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Label(relabel_row, text="New label:").pack(side=tk.LEFT)
        relabel_entry = ttk.Entry(relabel_row, textvariable=self.new_label_var, width=14)
        relabel_entry.pack(side=tk.LEFT, padx=(6, 6))
        relabel_entry.bind("<Return>", lambda _e: self.apply_relabel())
        ttk.Button(relabel_row, text="Apply Label", command=self.apply_relabel).pack(side=tk.LEFT)

        editor_group = ttk.LabelFrame(right, text="Editor (drag to move)")
        editor_group.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.editor_canvas = tk.Canvas(editor_group, width=520, height=520, bg="#d9d9d9", highlightthickness=1)
        self.editor_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.editor_canvas.bind("<Button-1>", self.on_editor_press)
        self.editor_canvas.bind("<B1-Motion>", self.on_editor_drag)
        self.editor_canvas.bind("<Configure>", lambda _e: self.redraw_editor())

        guide_controls = ttk.Frame(editor_group)
        guide_controls.pack(fill=tk.X, padx=8, pady=(0, 6))

        ttk.Checkbutton(
            guide_controls,
            text="Show font guide",
            variable=self.guide_show_var,
            command=self.redraw_editor,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            guide_controls,
            text="Auto-fit glyph to guide",
            variable=self.auto_fit_var,
        ).pack(side=tk.LEFT, padx=(10, 0))

        ttk.Label(guide_controls, text="Guide locked to output metrics").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Checkbutton(
            guide_controls,
            text="Show vector overlay",
            variable=self.vector_show_var,
            command=self.redraw_editor,
        ).pack(side=tk.LEFT, padx=(10, 0))
        if cv2 is None or sk_skeletonize is None:
            ttk.Label(guide_controls, text="(install opencv-python + scikit-image)", foreground="#a33").pack(
                side=tk.LEFT,
                padx=(4, 0),
            )

        scale_controls = ttk.Frame(editor_group)
        scale_controls.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Label(scale_controls, text="Glyph scale:").pack(side=tk.LEFT)

        scale_slider = ttk.Scale(
            scale_controls,
            from_=0.3,
            to=3.0,
            variable=self.glyph_scale_var,
            orient=tk.HORIZONTAL,
            length=220,
            command=self.on_glyph_scale_changed,
        )
        scale_slider.pack(side=tk.LEFT, padx=(8, 8))
        ttk.Label(scale_controls, textvariable=self.glyph_scale_display_var, width=6).pack(side=tk.LEFT)
        self.glyph_scale_var.trace_add("write", lambda *_: self.on_glyph_scale_changed())
        ttk.Button(scale_controls, text="Reset Scale", command=self.reset_glyph_scale).pack(side=tk.LEFT)
        ttk.Button(scale_controls, text="Auto Fit Now", command=self.auto_fit_current_glyph).pack(side=tk.LEFT, padx=(8, 0))

        buttons = ttk.Frame(editor_group)
        buttons.pack(fill=tk.X, padx=8, pady=(0, 8))

        ttk.Button(buttons, text="Save Transform", command=self.save_shift).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Reset Shift", command=self.reset_shift).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(buttons, text="Delete Variant", command=self.delete_variant).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(buttons, text="Next", command=self.select_next).pack(side=tk.LEFT, padx=(8, 0))

        hint = ttk.Label(
            editor_group,
            text=(
                "Tip: drag with mouse to move glyph. Arrow keys nudge by 1px. "
                "Shift+Arrow nudge by 5px. Scale glyph to match the fixed guide."
            ),
        )
        hint.pack(anchor="w", padx=8, pady=(0, 8))

        debug_group = ttk.LabelFrame(editor_group, text="Vector Debug")
        debug_group.pack(fill=tk.X, padx=8, pady=(0, 8))
        debug_titles = ["Original", "Binary", "Topology", "Stroke Decomp.", "Final Strokes"]
        row = ttk.Frame(debug_group)
        row.pack(fill=tk.X, padx=6, pady=6)
        for i, title in enumerate(debug_titles):
            cell = ttk.Frame(row)
            cell.grid(row=0, column=i, sticky="n", padx=4)
            ttk.Label(cell, text=title).pack()
            lbl = ttk.Label(cell)
            lbl.pack()
            self.vector_debug_labels.append(lbl)

        self.root.bind("<Left>", lambda e: self.nudge(-1, 0, e))
        self.root.bind("<Right>", lambda e: self.nudge(1, 0, e))
        self.root.bind("<Up>", lambda e: self.nudge(0, -1, e))
        self.root.bind("<Down>", lambda e: self.nudge(0, 1, e))
        self.root.bind("<Shift-Left>", lambda e: self.nudge(-5, 0, e))
        self.root.bind("<Shift-Right>", lambda e: self.nudge(5, 0, e))
        self.root.bind("<Shift-Up>", lambda e: self.nudge(0, -5, e))
        self.root.bind("<Shift-Down>", lambda e: self.nudge(0, 5, e))

    def on_mousewheel(self, event: tk.Event) -> None:
        if self.grid_canvas.winfo_exists():
            self.grid_canvas.yview_scroll(int(-event.delta / 120), "units")

    def on_grid_resize(self, _event: tk.Event) -> None:
        w = max(1, self.grid_canvas.winfo_width())
        self.grid_cols = max(1, w // self.cell_w)
        self.rebuild_grid()

    def on_reload_clicked(self) -> None:
        self.library_paths = parse_library_paths([self.libs_var.get()])
        self.reload_variants()

    def reload_variants(self) -> None:
        self.variants = load_variants(self.library_paths)
        all_orig_h = [v.orig_h for v in self.variants if v.orig_h is not None]
        self.ref_orig_h = float(statistics.median(all_orig_h)) if all_orig_h else 30.0

        lower_orig_h = [v.orig_h for v in self.variants if v.orig_h is not None and is_single_lower_ascii(v.label)]
        if lower_orig_h:
            lower_q1 = float(np.percentile(lower_orig_h, 25))
            self.lowercase_floor_ratio = max(0.72, lower_q1 / self.ref_orig_h)
        else:
            self.lowercase_floor_ratio = 0.72

        # Per-label stable target orig_h, snapshotted at load time. Save edits
        # mutate v.orig_h on disk but must NOT shift the guide the user is
        # fitting to, so we read this map (not v.orig_h) for guide sizing.
        by_label: dict[str, list[int]] = {}
        for v in self.variants:
            if v.orig_h is not None:
                by_label.setdefault(v.label, []).append(v.orig_h)
        self.label_target_orig_h = {
            label: float(statistics.median(vals)) for label, vals in by_label.items()
        }

        self.selected_global_idx = None
        self.editor_orig_img = None
        self.editor_offset_x = 0.0
        self.editor_offset_y = 0.0
        self.glyph_user_scale = 1.0
        self.glyph_scale_var.set(1.0)
        self.apply_filter()

    def apply_filter(self) -> None:
        q = self.filter_var.get().strip().lower()
        self.filtered_indices = []
        for i, v in enumerate(self.variants):
            if not q or q in v.label.lower() or q in v.rel_path.lower():
                self.filtered_indices.append(i)
        self.count_var.set(f"{len(self.filtered_indices)} glyphs")
        self.rebuild_grid()

    def checker_bg(self, w: int, h: int, s: int = 8) -> Image.Image:
        yy, xx = np.indices((h, w))
        mask = ((xx // s) + (yy // s)) % 2 == 0
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[mask] = (245, 245, 245)
        arr[~mask] = (220, 220, 220)
        return Image.fromarray(arr, mode="RGB")

    def make_thumb(self, variant: GlyphVariant, selected: bool) -> ImageTk.PhotoImage:
        canvas = self.checker_bg(self.thumb_box, self.thumb_box)
        draw = Image.new("RGBA", (self.thumb_box, self.thumb_box), (0, 0, 0, 0))

        if variant.abs_path.exists():
            g = Image.open(variant.abs_path).convert("RGBA")
            scale = min((self.thumb_box - 8) / max(1, g.width), (self.thumb_box - 8) / max(1, g.height))
            nw = max(1, round(g.width * scale))
            nh = max(1, round(g.height * scale))
            g = g.resize((nw, nh), Image.LANCZOS)
            px = (self.thumb_box - nw) // 2
            py = (self.thumb_box - nh) // 2
            draw.paste(g, (px, py), g)
        else:
            # Missing file indicator
            miss = Image.new("RGBA", (self.thumb_box, self.thumb_box), (255, 0, 0, 55))
            draw.alpha_composite(miss)

        out = canvas.convert("RGBA")
        out.alpha_composite(draw)

        border = Image.new("RGBA", out.size, (0, 0, 0, 0))
        bpx = border.load()
        bcol = (30, 110, 240, 255) if selected else (150, 150, 150, 255)
        for x in range(self.thumb_box):
            bpx[x, 0] = bcol
            bpx[x, self.thumb_box - 1] = bcol
        for y in range(self.thumb_box):
            bpx[0, y] = bcol
            bpx[self.thumb_box - 1, y] = bcol
        out.alpha_composite(border)

        return ImageTk.PhotoImage(out.convert("RGB"))

    def rebuild_grid(self) -> None:
        self.grid_canvas.delete("all")
        self.grid_photo_refs.clear()

        cols = max(1, self.grid_cols)
        x_pad = 10
        y_pad = 10

        for n, global_idx in enumerate(self.filtered_indices):
            row = n // cols
            col = n % cols
            x0 = x_pad + col * self.cell_w
            y0 = y_pad + row * self.cell_h

            v = self.variants[global_idx]
            selected = self.selected_global_idx == global_idx
            photo = self.make_thumb(v, selected)
            self.grid_photo_refs.append(photo)

            self.grid_canvas.create_image(x0, y0, image=photo, anchor="nw")
            self.grid_canvas.create_text(
                x0 + 2,
                y0 + self.thumb_box + 12,
                text=f"{v.label} | {v.lib_path.name}",
                anchor="nw",
                fill="#222",
                font=("Segoe UI", 9),
            )

        total_rows = math.ceil(max(1, len(self.filtered_indices)) / cols)
        content_h = y_pad * 2 + total_rows * self.cell_h
        content_w = x_pad * 2 + cols * self.cell_w
        self.grid_canvas.configure(scrollregion=(0, 0, content_w, content_h))

    def on_grid_click(self, event: tk.Event) -> None:
        cols = max(1, self.grid_cols)
        x = self.grid_canvas.canvasx(event.x)
        y = self.grid_canvas.canvasy(event.y)

        x_pad = 10
        y_pad = 10
        col = int((x - x_pad) // self.cell_w)
        row = int((y - y_pad) // self.cell_h)
        if col < 0 or row < 0:
            return
        idx_in_filtered = row * cols + col
        if idx_in_filtered < 0 or idx_in_filtered >= len(self.filtered_indices):
            return

        self.selected_global_idx = self.filtered_indices[idx_in_filtered]
        self.load_selected_into_editor()
        self.rebuild_grid()

    def load_selected_into_editor(self) -> None:
        v = self.current_variant()
        if v is None:
            return
        if not v.abs_path.exists():
            messagebox.showerror("Missing file", f"File does not exist:\n{v.abs_path}")
            return

        self.editor_orig_img = Image.open(v.abs_path).convert("RGBA")
        vec = vectorize_glyph_from_alpha(self.editor_orig_img)
        self.editor_vector_paths = vec.paths
        self.editor_vector_svg_path = vec.svg_path
        self.editor_fitted_strokes = _fit_glyph_strokes(self.editor_orig_img, v.label)
        self._update_vector_debug_panel(self.editor_orig_img, vec)
        self.editor_offset_x = 0.0
        self.editor_offset_y = 0.0
        self._set_glyph_scale(1.0, redraw=False)
        self.pending_auto_fit = bool(self.auto_fit_var.get())

        self.sel_label_var.set(f"Label: {v.label}")
        self.sel_lib_var.set(f"Library: {v.lib_path}")
        self.sel_path_var.set(f"Path: {v.rel_path}")
        self.sel_size_var.set(f"Size: {self.editor_orig_img.width} x {self.editor_orig_img.height}")
        self.sel_offset_var.set("Offset: (0, 0)")
        self.new_label_var.set(v.label)

        self.redraw_editor()

    def current_variant(self) -> GlyphVariant | None:
        if self.selected_global_idx is None:
            return None
        if self.selected_global_idx < 0 or self.selected_global_idx >= len(self.variants):
            return None
        return self.variants[self.selected_global_idx]

    def _debug_thumb(self, img: Image.Image, size: int = 110) -> ImageTk.PhotoImage:
        disp = img.convert("RGB").copy()
        disp.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), (245, 245, 245))
        px = (size - disp.width) // 2
        py = (size - disp.height) // 2
        canvas.paste(disp, (px, py))
        return ImageTk.PhotoImage(canvas)

    def _update_vector_debug_panel(self, original: Image.Image, vec: VectorizationResult) -> None:
        imgs = [
            original.convert("RGB"),
            vec.binary_debug if vec.binary_debug is not None else Image.new("RGB", (64, 64), (255, 255, 255)),
            vec.skeleton_debug if vec.skeleton_debug is not None else Image.new("RGB", (64, 64), (255, 255, 255)),
            vec.graph_debug if vec.graph_debug is not None else Image.new("RGB", (64, 64), (255, 255, 255)),
            vec.final_debug if vec.final_debug is not None else Image.new("RGB", (64, 64), (255, 255, 255)),
        ]
        self.vector_debug_photo_refs = [self._debug_thumb(im) for im in imgs]
        for lbl, ph in zip(self.vector_debug_labels, self.vector_debug_photo_refs):
            lbl.configure(image=ph)

    def redraw_editor(self) -> None:
        self.editor_canvas.delete("all")
        if self.editor_orig_img is None:
            return

        cw = max(10, self.editor_canvas.winfo_width())
        ch = max(10, self.editor_canvas.winfo_height())

        # Checker background
        bg = self.checker_bg(cw, ch, s=14)
        bg_photo = ImageTk.PhotoImage(bg)
        self.editor_photo_ref = bg_photo
        self.editor_canvas.create_image(0, 0, image=bg_photo, anchor="nw")

        center_x = cw // 2
        baseline_y = int(ch * 0.72)
        self.last_baseline_y = baseline_y
        self.last_guide_bbox = None

        src = self.editor_orig_img
        v = self.current_variant()

        max_w = max(40, cw - 80)
        max_h = max(40, ch - 120)
        # Vertical room available above the baseline, with a top margin so
        # the glyph cap is not flush with the canvas edge.
        avail_above_baseline = max(40, baseline_y - 10)

        # Effective ref_h that fits BOTH the guide bbox and the glyph PNG
        # canvas height (incl. user-scale and PNG padding) in the canvas.
        # Without this clamp, glyphs whose orig_h is well above the median
        # (e.g. '$' with descender) overflow the top of the visible canvas.
        ref_h = max(72, int(ch * self.guide_preview_height_ratio))
        glyph_ratio = 1.0
        guide_ratio = 1.0
        padding_factor = 1.0
        alpha_h = 1
        if v is not None and v.orig_h is not None and self.ref_orig_h > 0:
            glyph_ratio = float(v.orig_h) / float(self.ref_orig_h)
            guide_ratio = self._guide_ratio_for_variant(v)
            alpha_h = self._glyph_alpha_height(src)
            padding_factor = float(src.height) / max(1.0, float(alpha_h))
            scale_for_user = max(1.0, float(self.glyph_user_scale))
            needed_per_ref = max(
                padding_factor * glyph_ratio * scale_for_user,
                guide_ratio,
                1.0,
            )
            if ref_h * needed_per_ref > avail_above_baseline:
                ref_h = max(40, int(round(avail_above_baseline / needed_per_ref)))

        # Draw reference font glyph behind the editable glyph.
        if self.guide_show_var.get() and v is not None and v.label:
            guide_text = v.label
            target_h = max(24, int(round(ref_h * guide_ratio)))

            gsize = max(12, target_h)
            gid = self.editor_canvas.create_text(
                center_x,
                baseline_y,
                text=guide_text,
                fill="#9a9a9a",
                font=(self.guide_family, gsize),
                anchor="s",
            )
            bbox = self.editor_canvas.bbox(gid)
            if bbox is not None:
                gh = max(1, bbox[3] - bbox[1])
                refined = max(12, int(round(gsize * target_h / gh)))
                self.editor_canvas.itemconfigure(gid, font=(self.guide_family, refined))
                self.last_guide_bbox = self.editor_canvas.bbox(gid)

        # Glyph display scale anchored to v.orig_h via the SAME ref_h the
        # guide uses, so "glyph height vs guide height" is invariant across
        # save (without this, fit-to-canvas snaps the post-save bigger PNG
        # back to canvas size and the glyph visually shrinks every save).
        if v is not None and v.orig_h is not None and self.ref_orig_h > 0:
            target_display_h = ref_h * glyph_ratio
            self.editor_scale = target_display_h / max(1, alpha_h)
        else:
            self.editor_scale = min(max_w / max(1, src.width), max_h / max(1, src.height), 8.0)
        self.editor_scale = max(0.05, min(self.editor_scale, 8.0))

        if self.pending_auto_fit and self.auto_fit_var.get():
            self.pending_auto_fit = False
            self.auto_fit_current_glyph(redraw=False)

        scaled_src = scale_image_rgba(src, self.glyph_user_scale)

        shifted = shift_image_rgba(scaled_src, int(round(self.editor_offset_x)), int(round(self.editor_offset_y)))
        nw = max(1, round(shifted.width * self.editor_scale))
        nh = max(1, round(shifted.height * self.editor_scale))
        disp = shifted.resize((nw, nh), Image.NEAREST)

        px = center_x - (nw // 2)
        py = baseline_y - nh

        photo = ImageTk.PhotoImage(disp)
        self.editor_photo_ref = photo
        self.editor_canvas.create_image(px, py, image=photo, anchor="nw")

        if self.vector_show_var.get():
            # Prefer the step-7 fitted strokes (graph-cleanup + spline fit) for
            # the vector overlay; fall back to the legacy vectorization when
            # the fit pipeline returns nothing (small/empty glyphs).
            if self.editor_fitted_strokes:
                self._draw_fitted_overlay(
                    strokes=self.editor_fitted_strokes,
                    src_w=src.width,
                    src_h=src.height,
                    canvas_x=px,
                    canvas_y=py,
                    user_scale=self.glyph_user_scale,
                    shift_x=int(round(self.editor_offset_x)),
                    shift_y=int(round(self.editor_offset_y)),
                    display_scale=self.editor_scale,
                )
            elif self.editor_vector_paths:
                self._draw_vector_overlay(
                    source_paths=self.editor_vector_paths,
                    src_w=src.width,
                    src_h=src.height,
                    canvas_x=px,
                    canvas_y=py,
                    user_scale=self.glyph_user_scale,
                    shift_x=int(round(self.editor_offset_x)),
                    shift_y=int(round(self.editor_offset_y)),
                    display_scale=self.editor_scale,
                    num_strokes=len(self.editor_vector_paths),
                )

        # Baseline frame around the glyph PNG, expanded to envelope the
        # guide bbox so a per-class-larger guide (e.g. '$' with descender
        # while this variant's orig_h is below the class median) does not
        # appear to "poke out" of the frame.
        frame_w = max(1, round(scaled_src.width * self.editor_scale))
        frame_h = max(1, round(scaled_src.height * self.editor_scale))
        fx0 = center_x - (frame_w // 2)
        fy0 = baseline_y - frame_h
        fx1 = fx0 + frame_w
        fy1 = fy0 + frame_h
        if self.last_guide_bbox is not None:
            gx0, gy0, gx1, gy1 = self.last_guide_bbox
            fx0 = min(fx0, gx0)
            fy0 = min(fy0, gy0)
            fx1 = max(fx1, gx1)
            fy1 = max(fy1, gy1)
        self.editor_canvas.create_rectangle(fx0, fy0, fx1, fy1, outline="#4b4b4b")
        self.editor_canvas.create_line(20, baseline_y, cw - 20, baseline_y, fill="#7d7d7d", dash=(3, 3))

        self.sel_offset_var.set(
            f"Offset: ({int(round(self.editor_offset_x))}, {int(round(self.editor_offset_y))})"
        )

    def on_editor_press(self, event: tk.Event) -> None:
        self.drag_last_x = event.x
        self.drag_last_y = event.y

    def on_editor_drag(self, event: tk.Event) -> None:
        if self.editor_orig_img is None:
            return
        dx_disp = event.x - self.drag_last_x
        dy_disp = event.y - self.drag_last_y
        self.drag_last_x = event.x
        self.drag_last_y = event.y

        self.editor_offset_x += dx_disp / max(1e-6, self.editor_scale)
        self.editor_offset_y += dy_disp / max(1e-6, self.editor_scale)
        self.redraw_editor()

    def nudge(self, dx: int, dy: int, _event: tk.Event) -> None:
        if self.editor_orig_img is None:
            return
        self.editor_offset_x += dx
        self.editor_offset_y += dy
        self.redraw_editor()

    def on_glyph_scale_changed(self, *_args: object) -> None:
        if self._suppress_scale_callback:
            return
        try:
            s = float(self.glyph_scale_var.get())
        except Exception:
            return
        self._set_glyph_scale(s, redraw=True)

    def _set_glyph_scale(self, scale: float, redraw: bool) -> None:
        clamped = min(3.0, max(0.3, float(scale)))
        self.glyph_user_scale = clamped
        self.glyph_scale_display_var.set(f"{clamped:.2f}x")
        self._suppress_scale_callback = True
        try:
            self.glyph_scale_var.set(clamped)
        finally:
            self._suppress_scale_callback = False
        if redraw:
            self.redraw_editor()

    def _guide_ratio_for_variant(self, v: GlyphVariant) -> float:
        # Use the stable per-label target (snapshotted at reload), not the
        # per-variant orig_h, so saving a rescaled glyph does not move the
        # guide the user was fitting to.
        ratio = 1.0
        if self.ref_orig_h > 0:
            target = self.label_target_orig_h.get(v.label)
            if target is None and v.orig_h is not None:
                target = float(v.orig_h)
            if target is not None:
                ratio = float(target) / float(self.ref_orig_h)
        if is_single_lower_ascii(v.label):
            ratio = max(ratio, self.lowercase_floor_ratio)
        return ratio

    def _target_guide_height_px(self, canvas_h: int, v: GlyphVariant) -> int:
        ratio = self._guide_ratio_for_variant(v)
        ref_h = max(72, int(canvas_h * self.guide_preview_height_ratio))
        return max(24, int(round(ref_h * ratio)))

    def _glyph_alpha_height(self, img: Image.Image) -> int:
        alpha = img.getchannel("A")
        bbox = alpha.getbbox()
        if bbox is None:
            return max(1, img.height)
        return max(1, bbox[3] - bbox[1])

    def _glyph_alpha_bbox(self, img: Image.Image) -> tuple[int, int, int, int]:
        alpha = img.getchannel("A")
        bbox = alpha.getbbox()
        if bbox is None:
            return (0, 0, img.width, img.height)
        return bbox

    def _draw_vector_overlay(
        self,
        source_paths: list[list[tuple[float, float]]],
        src_w: int,
        src_h: int,
        canvas_x: int,
        canvas_y: int,
        user_scale: float,
        shift_x: int,
        shift_y: int,
        display_scale: float,
        num_strokes: int = 1,
    ) -> None:
        # Match the exact centering logic used by scale_image_rgba.
        s = max(0.1, float(user_scale))
        nw = max(1, round(src_w * s))
        nh = max(1, round(src_h * s))
        out_w = max(src_w, nw)
        out_h = max(src_h, nh)
        off_x = (out_w - nw) // 2
        off_y = (out_h - nh) // 2

        # Use the same per-stroke colour palette as the debug panel.
        tk_colors = ["#dc1e1e", "#1e64dc", "#14aa3c", "#c87810",
                     "#961ec8", "#14b4b4", "#c8509a", "#646464"]

        for stroke_idx, path in enumerate(source_paths):
            if len(path) < 2:
                continue
            color = tk_colors[stroke_idx % len(tk_colors)]
            points: list[float] = []
            for x, y in path:
                x2 = off_x + (x * s) + shift_x
                y2 = off_y + (y * s) + shift_y
                xd = canvas_x + (x2 * display_scale)
                yd = canvas_y + (y2 * display_scale)
                points.extend([xd, yd])
            self.editor_canvas.create_line(
                *points,
                fill=color,
                width=2,
                smooth=True,
                splinesteps=72,
            )

    def _draw_fitted_overlay(
        self,
        strokes: list[dict],
        src_w: int,
        src_h: int,
        canvas_x: int,
        canvas_y: int,
        user_scale: float,
        shift_x: int,
        shift_y: int,
        display_scale: float,
    ) -> None:
        """Draw the step-7 fitted-strokes overlay (cubic-spline curves).

        Mirrors the coordinate transform used by _draw_vector_overlay so the
        two overlays align pixel-perfectly when both are visible.
        """
        s = max(0.1, float(user_scale))
        nw = max(1, round(src_w * s))
        nh = max(1, round(src_h * s))
        out_w = max(src_w, nw)
        out_h = max(src_h, nh)
        off_x = (out_w - nw) // 2
        off_y = (out_h - nh) // 2

        # Use the same per-stroke palette as the legacy vector overlay so the
        # visual style is consistent; only the underlying geometry changes.
        fitted_colors = ["#dc1e1e", "#1e64dc", "#14aa3c", "#c87810",
                         "#961ec8", "#14b4b4", "#c8509a", "#646464"]
        # When a stroke carries per-segment kinds (from a character profile),
        # colour by kind so mis-classifications are obvious at a glance.
        kind_colors = {
            "line": "#dc1e1e",     # red
            "arc": "#1e64dc",      # blue
            "s_curve": "#14aa3c",  # green
        }

        def _to_canvas(x: float, y: float) -> tuple[float, float]:
            x2 = off_x + (x * s) + shift_x
            y2 = off_y + (y * s) + shift_y
            return (canvas_x + (x2 * display_scale),
                    canvas_y + (y2 * display_scale))

        for stroke_idx, stroke in enumerate(strokes):
            path = stroke.get("points", [])
            if len(path) < 2:
                continue

            # Explicitly close the loop geometrically for rendering if the
            # source didn't already end at the start pixel.
            if stroke.get("closed") and path[0] != path[-1]:
                path = list(path) + [path[0]]

            segments = stroke.get("segments")
            if segments:
                # Profile-guided fit: colour each segment by its kind.
                for seg in segments:
                    a = int(seg.get("start", 0))
                    b = int(seg.get("end", 0))
                    if b <= a or a >= len(path):
                        continue
                    sub = path[a: b + 1]
                    if len(sub) < 2:
                        continue
                    color = kind_colors.get(str(seg.get("kind", "")),
                                            fitted_colors[stroke_idx % len(fitted_colors)])
                    coords: list[float] = []
                    for x, y in sub:
                        xd, yd = _to_canvas(x, y)
                        coords.extend([xd, yd])
                    self.editor_canvas.create_line(
                        *coords, fill=color, width=2, smooth=False,
                    )
            else:
                color = fitted_colors[stroke_idx % len(fitted_colors)]
                coords = []
                for x, y in path:
                    xd, yd = _to_canvas(x, y)
                    coords.extend([xd, yd])
                # Our paths are already spline-sampled upstream; disable Tk's
                # own smoothing so the rendered curve matches the fit exactly.
                self.editor_canvas.create_line(
                    *coords, fill=color, width=2, smooth=False,
                )

    def auto_fit_current_glyph(self, redraw: bool = True) -> None:
        v = self.current_variant()
        if v is None or self.editor_orig_img is None:
            return

        ch = max(10, self.editor_canvas.winfo_height())
        baseline_y = self.last_baseline_y if self.last_baseline_y > 0 else int(ch * 0.72)

        # Prefer actual rendered guide bounds for fitting.
        if self.last_guide_bbox is not None:
            guide_top = int(self.last_guide_bbox[1])
            guide_bottom = int(self.last_guide_bbox[3])
            target_h = max(1, guide_bottom - guide_top)
        else:
            target_h = self._target_guide_height_px(ch, v)
            guide_top = baseline_y - target_h

        glyph_bbox = self._glyph_alpha_bbox(self.editor_orig_img)
        glyph_h = max(1, glyph_bbox[3] - glyph_bbox[1])

        # Solve scale from top-bottom target span, then refine from rendered bbox
        # so display-space top/bottom both align as closely as possible.
        needed_scale = target_h / max(1e-6, glyph_h * self.editor_scale)
        for _ in range(2):
            self._set_glyph_scale(needed_scale, redraw=False)
            scaled_src = scale_image_rgba(self.editor_orig_img, self.glyph_user_scale)
            scaled_bbox = self._glyph_alpha_bbox(scaled_src)
            display_h = max(1e-6, (scaled_bbox[3] - scaled_bbox[1]) * self.editor_scale)
            needed_scale *= target_h / display_h

        self._set_glyph_scale(needed_scale, redraw=False)
        scaled_src = scale_image_rgba(self.editor_orig_img, self.glyph_user_scale)
        scaled_bbox = self._glyph_alpha_bbox(scaled_src)
        nh = max(1, round(scaled_src.height * self.editor_scale))
        py = baseline_y - nh

        # Enforce top and bottom alignment together; if raster rounding creates a
        # tiny mismatch, center the residual so both edges are equally close.
        dy_top = ((guide_top - py) / max(1e-6, self.editor_scale)) - scaled_bbox[1]
        dy_bottom = ((guide_bottom - py) / max(1e-6, self.editor_scale)) - scaled_bbox[3]
        self.editor_offset_y = (dy_top + dy_bottom) / 2.0

        if redraw:
            self.redraw_editor()

    def reset_glyph_scale(self) -> None:
        self._set_glyph_scale(1.0, redraw=True)

    def reset_shift(self) -> None:
        if self.editor_orig_img is None:
            return
        self.editor_offset_x = 0.0
        self.editor_offset_y = 0.0
        self.redraw_editor()

    def save_shift(self) -> None:
        v = self.current_variant()
        if v is None or self.editor_orig_img is None:
            return

        applied_scale = float(self.glyph_user_scale)
        base = scale_image_rgba(self.editor_orig_img, self.glyph_user_scale)
        dx = int(round(self.editor_offset_x))
        dy = int(round(self.editor_offset_y))
        shifted = shift_image_rgba(base, dx, dy)
        shifted.save(v.abs_path)

        if v.orig_h is not None and abs(applied_scale - 1.0) > 1e-6:
            new_orig_h = max(1, int(round(float(v.orig_h) * applied_scale)))
            if new_orig_h != v.orig_h:
                try:
                    self._update_variant_orig_h(v, new_orig_h)
                    v.orig_h = new_orig_h
                except Exception as ex:  # noqa: BLE001
                    messagebox.showwarning(
                        "Metadata update warning",
                        f"Saved PNG, but could not update orig_h in index.json:\n{ex}",
                    )

        self.editor_orig_img = shifted
        vec = vectorize_glyph_from_alpha(self.editor_orig_img)
        self.editor_vector_paths = vec.paths
        self.editor_vector_svg_path = vec.svg_path
        self.editor_fitted_strokes = _fit_glyph_strokes(self.editor_orig_img, v.label)
        self._update_vector_debug_panel(self.editor_orig_img, vec)
        self.editor_offset_x = 0.0
        self.editor_offset_y = 0.0
        self._set_glyph_scale(1.0, redraw=False)
        self.pending_auto_fit = bool(self.auto_fit_var.get())
        self.sel_size_var.set(f"Size: {shifted.width} x {shifted.height}")
        self.redraw_editor()
        self.rebuild_grid()

    def _update_variant_orig_h(self, v: GlyphVariant, new_orig_h: int) -> None:
        with open(v.idx_path, encoding="utf-8") as f:
            index = json.load(f)

        glyphs = index.get("glyphs", {})
        entries = list(glyphs.get(v.label, []))
        updated = False
        new_entries = []

        for e in entries:
            if isinstance(e, dict):
                rel = str(e.get("path", ""))
                if rel == v.rel_path and not updated:
                    ne = dict(e)
                    ne["orig_h"] = int(new_orig_h)
                    new_entries.append(ne)
                    updated = True
                else:
                    new_entries.append(e)
            else:
                rel = str(e)
                if rel == v.rel_path and not updated:
                    new_entries.append({"path": rel, "orig_h": int(new_orig_h)})
                    updated = True
                else:
                    new_entries.append(e)

        if not updated:
            raise ValueError("Selected variant was not found in index.json.")

        glyphs[v.label] = new_entries
        index["version"] = 2
        index["glyphs"] = glyphs
        with open(v.idx_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def _remove_variant_from_index(self, v: GlyphVariant) -> None:
        with open(v.idx_path, encoding="utf-8") as f:
            index = json.load(f)

        glyphs = index.get("glyphs", {})
        entries = list(glyphs.get(v.label, []))
        kept = []
        for e in entries:
            if isinstance(e, dict):
                rel = str(e.get("path", ""))
            else:
                rel = str(e)
            if rel != v.rel_path:
                kept.append(e)

        if kept:
            glyphs[v.label] = kept
        else:
            glyphs.pop(v.label, None)

        index["glyphs"] = glyphs
        with open(v.idx_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def delete_variant(self) -> None:
        v = self.current_variant()
        if v is None:
            return

        ok = messagebox.askyesno(
            "Delete variant",
            f"Delete this glyph variant?\n\nLabel: {v.label}\nFile: {v.abs_path}",
        )
        if not ok:
            return

        try:
            if v.abs_path.exists():
                v.abs_path.unlink()
            self._remove_variant_from_index(v)
        except Exception as ex:  # noqa: BLE001
            messagebox.showerror("Delete failed", str(ex))
            return

        self.reload_variants()

    def apply_relabel(self) -> None:
        v = self.current_variant()
        if v is None:
            return

        new_label = self.new_label_var.get().strip()
        if not new_label:
            messagebox.showwarning("Invalid label", "Label cannot be empty.")
            return
        if new_label == v.label:
            return

        ok = messagebox.askyesno(
            "Relabel variant",
            (
                f"Change label from {v.label!r} to {new_label!r}?\n\n"
                f"This will move the PNG into the new label folder and update index.json."
            ),
        )
        if not ok:
            return

        try:
            with open(v.idx_path, encoding="utf-8") as f:
                index = json.load(f)

            glyphs = index.get("glyphs", {})
            old_entries = list(glyphs.get(v.label, []))

            # Remove the selected variant from old label bucket.
            kept_old = []
            moved_entry = None
            for e in old_entries:
                rel = str(e.get("path", "")) if isinstance(e, dict) else str(e)
                if rel == v.rel_path and moved_entry is None:
                    moved_entry = e
                else:
                    kept_old.append(e)

            if moved_entry is None:
                messagebox.showerror(
                    "Relabel failed",
                    "Could not find selected variant in index.json.",
                )
                return

            if kept_old:
                glyphs[v.label] = kept_old
            else:
                glyphs.pop(v.label, None)

            # Move the file into a folder for the new label.
            old_abs = v.abs_path
            new_dir = v.lib_path / safe_label_dir(new_label)
            new_dir.mkdir(parents=True, exist_ok=True)

            stem = safe_label_dir(new_label)
            suffix = old_abs.suffix or ".png"
            n = 0
            while True:
                candidate = new_dir / f"{stem}_{n}{suffix}"
                if not candidate.exists():
                    break
                n += 1

            old_abs.rename(candidate)
            new_rel = str(candidate.relative_to(v.lib_path)).replace("\\", "/")

            # Ensure v2 dict entry format so orig_h can be retained.
            if isinstance(moved_entry, dict):
                new_entry = dict(moved_entry)
                new_entry["path"] = new_rel
                if "orig_h" not in new_entry:
                    new_entry["orig_h"] = v.orig_h
            else:
                new_entry = {"path": new_rel, "orig_h": v.orig_h}

            new_bucket = list(glyphs.get(new_label, []))
            new_bucket.append(new_entry)
            glyphs[new_label] = new_bucket

            index["version"] = 2
            index["glyphs"] = glyphs
            with open(v.idx_path, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)

        except Exception as ex:  # noqa: BLE001
            messagebox.showerror("Relabel failed", str(ex))
            return

        self.reload_variants()

        # Try to keep user on the relabeled variant.
        for i, item in enumerate(self.variants):
            if item.label == new_label and item.lib_path == v.lib_path and item.rel_path == new_rel:
                self.selected_global_idx = i
                self.load_selected_into_editor()
                break
        self.rebuild_grid()

    def select_next(self) -> None:
        if not self.filtered_indices:
            return
        if self.selected_global_idx is None:
            self.selected_global_idx = self.filtered_indices[0]
        else:
            try:
                cur = self.filtered_indices.index(self.selected_global_idx)
                nxt = (cur + 1) % len(self.filtered_indices)
                self.selected_global_idx = self.filtered_indices[nxt]
            except ValueError:
                self.selected_global_idx = self.filtered_indices[0]

        self.load_selected_into_editor()
        self.rebuild_grid()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive glyph editor")
    parser.add_argument(
        "--glyphs",
        action="append",
        default=None,
        help="Glyph library folder(s). Repeat and/or comma-separate.",
    )
    args = parser.parse_args()

    raw_glyphs = args.glyphs if args.glyphs is not None else ["glyphs_merged"]
    libs = parse_library_paths(raw_glyphs)
    if not libs:
        raise SystemExit("No glyph libraries specified.")

    root = tk.Tk()
    app = GlyphEditorApp(root, libs)
    app.root.mainloop()


if __name__ == "__main__":
    main()
