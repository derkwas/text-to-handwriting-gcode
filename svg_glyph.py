#!/usr/bin/env python3
"""Minimal SVG glyph model: parse, sample, transform, serialize.

Accepts the subset of SVG path data the extractor emits: M, L, H, V, C, Z
(absolute). Cubic Beziers are the only curve type (PDF source only emits c,
v, y as cubics; we treat them uniformly). Everything lives in the glyph's
local coordinate space (0, 0) at the top-left of its viewBox.

Public API:
    parse_svg_file(path)        -> Glyph
    Glyph.sample_polylines()    -> list[list[(x, y)]]   # for drawing
    Glyph.apply_transform(...)  -> Glyph                 # translate+scale
    Glyph.to_svg()              -> str
    rasterize_glyph(g, w, h, pad=0, bg=(0,0,0,0), fg=(0,0,0,255)) -> PIL.Image

No dependency on cairo/rsvg; rasterization uses PIL.ImageDraw.polygon which
is good enough for grid thumbnails and editor preview.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageDraw


# ── Segment types ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MoveTo:
    x: float
    y: float

@dataclass(frozen=True)
class LineTo:
    x: float
    y: float

@dataclass(frozen=True)
class CubicTo:
    # Cubic Bezier; previous endpoint is the implicit start.
    x1: float
    y1: float
    x2: float
    y2: float
    x: float
    y: float

@dataclass(frozen=True)
class ClosePath:
    pass


Segment = MoveTo | LineTo | CubicTo | ClosePath


# ── Path-d parser ───────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[MLHVCZmlhvcz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _tokenize(d: str) -> list[str]:
    return _TOKEN_RE.findall(d)


def parse_d(d: str) -> list[Segment]:
    """Parse an SVG path 'd' attribute into a flat segment list.

    Supports both absolute and relative variants of M, L, H, V, C, Z. Quads
    and arcs aren't emitted by our extractor, so they're not handled; any
    unknown op raises.
    """
    toks = _tokenize(d)
    segs: list[Segment] = []
    i = 0
    cx, cy = 0.0, 0.0       # current point
    sx, sy = 0.0, 0.0       # subpath start (for Z)
    last_cmd: str | None = None

    def _num() -> float:
        nonlocal i
        v = float(toks[i])
        i += 1
        return v

    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            cmd = t
            i += 1
        else:
            # Implicit repeat of the previous command with new operands.
            # SVG spec: after M, implicit repeats are L (and l after m).
            if last_cmd is None:
                raise ValueError(f"path: number before any command: {t!r}")
            if last_cmd == "M":
                cmd = "L"
            elif last_cmd == "m":
                cmd = "l"
            else:
                cmd = last_cmd

        rel = cmd.islower()

        if cmd in ("M", "m"):
            x = _num(); y = _num()
            if rel:
                x += cx; y += cy
            segs.append(MoveTo(x, y))
            cx, cy = x, y
            sx, sy = x, y
        elif cmd in ("L", "l"):
            x = _num(); y = _num()
            if rel:
                x += cx; y += cy
            segs.append(LineTo(x, y))
            cx, cy = x, y
        elif cmd in ("H", "h"):
            x = _num()
            if rel:
                x += cx
            segs.append(LineTo(x, cy))
            cx = x
        elif cmd in ("V", "v"):
            y = _num()
            if rel:
                y += cy
            segs.append(LineTo(cx, y))
            cy = y
        elif cmd in ("C", "c"):
            x1 = _num(); y1 = _num()
            x2 = _num(); y2 = _num()
            x = _num(); y = _num()
            if rel:
                x1 += cx; y1 += cy
                x2 += cx; y2 += cy
                x += cx; y += cy
            segs.append(CubicTo(x1, y1, x2, y2, x, y))
            cx, cy = x, y
        elif cmd in ("Z", "z"):
            segs.append(ClosePath())
            cx, cy = sx, sy
        else:
            raise ValueError(f"path: unsupported command {cmd!r}")

        last_cmd = cmd

    return segs


def segs_to_d(segs: list[Segment]) -> str:
    """Inverse of parse_d. Emits absolute-coordinate form for simplicity."""
    parts: list[str] = []
    for s in segs:
        if isinstance(s, MoveTo):
            parts.append(f"M {_fmt(s.x)} {_fmt(s.y)}")
        elif isinstance(s, LineTo):
            parts.append(f"L {_fmt(s.x)} {_fmt(s.y)}")
        elif isinstance(s, CubicTo):
            parts.append(
                f"C {_fmt(s.x1)} {_fmt(s.y1)} "
                f"{_fmt(s.x2)} {_fmt(s.y2)} "
                f"{_fmt(s.x)} {_fmt(s.y)}"
            )
        elif isinstance(s, ClosePath):
            parts.append("Z")
    return " ".join(parts)


def _fmt(v: float) -> str:
    return f"{v:.3f}"


# ── Glyph ───────────────────────────────────────────────────────────────────

@dataclass
class Glyph:
    """A parsed SVG glyph — one or more subpaths, each a segment list.

    Subpaths are each a list of segments starting with MoveTo (and ending
    at an optional ClosePath). This is how a plotter will draw them: lift
    pen at each new subpath.
    """
    subpaths: list[list[Segment]] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    fill: str = "black"

    # ── Construction ────────────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: Path | str) -> "Glyph":
        p = Path(path)
        doc = ET.parse(str(p)).getroot()
        # Namespaces: SVG is xmlns="http://www.w3.org/2000/svg"
        ns = doc.tag.split("}", 1)[0].lstrip("{") if "}" in doc.tag else None
        ns_tag = (lambda name: f"{{{ns}}}{name}") if ns else (lambda name: name)

        vb = doc.attrib.get("viewBox")
        w, h = 0.0, 0.0
        if vb:
            vals = [float(x) for x in vb.strip().split()]
            if len(vals) == 4:
                w, h = vals[2], vals[3]
        if not w:
            try:
                w = float(doc.attrib.get("width", "0") or 0)
            except ValueError:
                w = 0.0
        if not h:
            try:
                h = float(doc.attrib.get("height", "0") or 0)
            except ValueError:
                h = 0.0

        # Collect all <path d="..."> elements (also inside <g>).
        paths: list[list[Segment]] = []
        fill = "black"
        for el in doc.iter(ns_tag("path")):
            d = el.attrib.get("d")
            if not d:
                continue
            segs = parse_d(d)
            # Split on MoveTo so each subpath is self-contained.
            current: list[Segment] = []
            for s in segs:
                if isinstance(s, MoveTo) and current:
                    paths.append(current)
                    current = [s]
                else:
                    current.append(s)
            if current:
                paths.append(current)
            f = el.attrib.get("fill")
            if f and f != "none":
                fill = f
        # Check <g fill="...">
        for g in doc.iter(ns_tag("g")):
            f = g.attrib.get("fill")
            if f and f != "none":
                fill = f
                break
        return cls(subpaths=paths, width=w, height=h, fill=fill)

    # ── Geometry ────────────────────────────────────────────────────────

    def bbox(self) -> tuple[float, float, float, float]:
        """Bounding box over sampled points (approximate but tight enough)."""
        xs: list[float] = []
        ys: list[float] = []
        for poly in self.sample_polylines(n_per_bezier=12):
            for x, y in poly:
                xs.append(x); ys.append(y)
        if not xs:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(xs), min(ys), max(xs), max(ys))

    # ── Sampling for rendering ──────────────────────────────────────────

    def sample_polylines(self, n_per_bezier: int = 16) -> list[list[tuple[float, float]]]:
        """One polyline per subpath. Cubic Beziers are sampled uniformly in
        parameter; lines contribute their endpoints. ClosePath appends the
        subpath-start point."""
        out: list[list[tuple[float, float]]] = []
        for sub in self.subpaths:
            pts: list[tuple[float, float]] = []
            start = (0.0, 0.0)
            cx, cy = 0.0, 0.0
            for s in sub:
                if isinstance(s, MoveTo):
                    if pts:
                        out.append(pts)
                        pts = []
                    pts.append((s.x, s.y))
                    start = (s.x, s.y)
                    cx, cy = s.x, s.y
                elif isinstance(s, LineTo):
                    pts.append((s.x, s.y))
                    cx, cy = s.x, s.y
                elif isinstance(s, CubicTo):
                    # Sample n_per_bezier+1 points; skip t=0 (it equals last pt).
                    for k in range(1, n_per_bezier + 1):
                        t = k / n_per_bezier
                        mt = 1.0 - t
                        x = (mt ** 3) * cx + 3 * (mt ** 2) * t * s.x1 \
                            + 3 * mt * (t ** 2) * s.x2 + (t ** 3) * s.x
                        y = (mt ** 3) * cy + 3 * (mt ** 2) * t * s.y1 \
                            + 3 * mt * (t ** 2) * s.y2 + (t ** 3) * s.y
                        pts.append((x, y))
                    cx, cy = s.x, s.y
                elif isinstance(s, ClosePath):
                    if pts and pts[-1] != start:
                        pts.append(start)
                    cx, cy = start
            if pts:
                out.append(pts)
        return out

    # ── Transforms ──────────────────────────────────────────────────────

    def apply_transform(self, tx: float = 0.0, ty: float = 0.0,
                         sx: float = 1.0, sy: float | None = None) -> "Glyph":
        """Return a NEW glyph with coords transformed: first scaled around the
        glyph's viewBox center, then translated by (tx, ty). width/height
        follow the scale; subpath coords are rewritten in absolute form."""
        if sy is None:
            sy = sx
        # Scale around the current center so the glyph stays visually centered.
        cx, cy = self.width / 2.0, self.height / 2.0

        def _t(x: float, y: float) -> tuple[float, float]:
            # scale around (cx, cy), then translate by (tx, ty).
            nx = cx + (x - cx) * sx + tx
            ny = cy + (y - cy) * sy + ty
            return nx, ny

        new_subs: list[list[Segment]] = []
        for sub in self.subpaths:
            out: list[Segment] = []
            for s in sub:
                if isinstance(s, MoveTo):
                    nx, ny = _t(s.x, s.y)
                    out.append(MoveTo(nx, ny))
                elif isinstance(s, LineTo):
                    nx, ny = _t(s.x, s.y)
                    out.append(LineTo(nx, ny))
                elif isinstance(s, CubicTo):
                    x1, y1 = _t(s.x1, s.y1)
                    x2, y2 = _t(s.x2, s.y2)
                    nx, ny = _t(s.x, s.y)
                    out.append(CubicTo(x1, y1, x2, y2, nx, ny))
                elif isinstance(s, ClosePath):
                    out.append(ClosePath())
            new_subs.append(out)

        # Recompute width/height from bbox of the transformed glyph so the
        # viewBox hugs the content (matches how the extractor wrote it).
        tmp = Glyph(subpaths=new_subs, width=self.width, height=self.height, fill=self.fill)
        x0, y0, x1, y1 = tmp.bbox()
        if x1 <= x0 or y1 <= y0:
            return tmp
        # Translate so (x0, y0) lands at (0, 0).
        dx, dy = -x0, -y0
        final_subs: list[list[Segment]] = []
        for sub in new_subs:
            out2: list[Segment] = []
            for s in sub:
                if isinstance(s, MoveTo):
                    out2.append(MoveTo(s.x + dx, s.y + dy))
                elif isinstance(s, LineTo):
                    out2.append(LineTo(s.x + dx, s.y + dy))
                elif isinstance(s, CubicTo):
                    out2.append(CubicTo(
                        s.x1 + dx, s.y1 + dy,
                        s.x2 + dx, s.y2 + dy,
                        s.x + dx, s.y + dy,
                    ))
                elif isinstance(s, ClosePath):
                    out2.append(ClosePath())
            final_subs.append(out2)
        return Glyph(subpaths=final_subs, width=x1 - x0, height=y1 - y0, fill=self.fill)

    def translate_only(self, tx: float, ty: float) -> "Glyph":
        """Translate without rescaling/recropping — viewBox grows if needed."""
        new_subs: list[list[Segment]] = []
        for sub in self.subpaths:
            out: list[Segment] = []
            for s in sub:
                if isinstance(s, MoveTo):
                    out.append(MoveTo(s.x + tx, s.y + ty))
                elif isinstance(s, LineTo):
                    out.append(LineTo(s.x + tx, s.y + ty))
                elif isinstance(s, CubicTo):
                    out.append(CubicTo(
                        s.x1 + tx, s.y1 + ty,
                        s.x2 + tx, s.y2 + ty,
                        s.x + tx, s.y + ty,
                    ))
                elif isinstance(s, ClosePath):
                    out.append(ClosePath())
            new_subs.append(out)
        return Glyph(subpaths=new_subs, width=self.width, height=self.height,
                     fill=self.fill)

    # ── Serialization ───────────────────────────────────────────────────

    def to_svg(self) -> str:
        body_parts: list[str] = []
        for sub in self.subpaths:
            body_parts.append(f'<path d="{segs_to_d(sub)}"/>')
        body = "\n".join(body_parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {_fmt(self.width)} {_fmt(self.height)}" '
            f'width="{_fmt(self.width)}" height="{_fmt(self.height)}">\n'
            f'<g fill="{self.fill}" stroke="none">\n'
            f'{body}\n'
            f'</g>\n'
            f'</svg>\n'
        )

    def save(self, path: Path | str) -> None:
        Path(path).write_text(self.to_svg(), encoding="utf-8")


# ── Centerline extraction ───────────────────────────────────────────────────

def compute_centerlines(
    glyph: Glyph,
    raster_long_side: int = 320,
    pad_px: int = 4,
) -> list[dict]:
    """Return centerline polylines for a filled-outline glyph.

    The glyph's closed subpaths represent ink outlines (the shape of the
    pen stroke, not the centerline). To recover a plotter-friendly centre
    track we rasterise the filled shape, skeletonise it, and trace the
    skeleton with the same graph + decompose + fit pipeline the raster
    editor uses.

    Returns a list of {"points": [(x, y), ...], "closed": bool} in the
    glyph's LOCAL coordinate space (same frame as the SVG source). Empty
    list if the glyph has no ink or the dependencies are missing.
    """
    if glyph.width <= 0 or glyph.height <= 0 or not glyph.subpaths:
        return []
    try:
        import numpy as np
        from skimage.morphology import skeletonize
        from stroke_graph import extract_stroke_graph
        from graph_cleanup import clean_graph
        from stroke_decompose import decompose
        from stroke_fit import fit_decomposition
    except Exception:
        return []

    # Pick raster size so the longer side is raster_long_side.
    aspect = glyph.height / glyph.width if glyph.width > 0 else 1.0
    if glyph.width >= glyph.height:
        w = raster_long_side
        h = max(16, int(round(raster_long_side * aspect)))
    else:
        h = raster_long_side
        w = max(16, int(round(raster_long_side / max(1e-6, aspect))))

    # Black ink on white paper so binarize-equivalent code reads it correctly.
    img = rasterize_glyph(
        glyph, w, h,
        pad=pad_px,
        bg=(255, 255, 255, 255),
        fg=(0, 0, 0, 255),
    )
    arr = np.array(img.convert("L"))
    ink_mask = (arr < 128).astype(np.uint8) * 255
    if int(ink_mask.sum()) == 0:
        return []

    skel = skeletonize(ink_mask > 0)
    raw = extract_stroke_graph(skel)
    cleaned, _audit = clean_graph(raw, ink_mask)
    decomp = decompose(cleaned)
    fitted = fit_decomposition(decomp)

    # Mirror rasterize_glyph's placement math so we can invert it.
    inner_w = max(1, w - 2 * pad_px)
    inner_h = max(1, h - 2 * pad_px)
    scale = min(inner_w / glyph.width, inner_h / glyph.height)
    draw_w = glyph.width * scale
    draw_h = glyph.height * scale
    ox = (w - draw_w) / 2.0
    oy = (h - draw_h) / 2.0

    def _px_to_glyph(px: float, py: float) -> tuple[float, float]:
        return ((px - ox) / scale, (py - oy) / scale)

    out: list[dict] = []
    for fs in fitted:
        if fs.spline is not None:
            pts = fs.sample_curve(n=128)
        else:
            pts = fs.simplified
        if len(pts) < 2:
            continue
        converted = [_px_to_glyph(float(p[0]), float(p[1])) for p in pts]
        out.append({"points": converted, "closed": bool(fs.closed)})
    return out


def transform_centerline_polyline(
    polyline: list[tuple[float, float]],
    base_glyph: Glyph,
    user_scale: float,
    tx: float,
    ty: float,
) -> list[tuple[float, float]]:
    """Mirror `Glyph._glyph_working_view`'s transform chain for a polyline:
    (1) scale around base_glyph centre by user_scale, (2) recrop to the
    scaled bbox's origin when user_scale != 1, (3) translate by (tx, ty)."""
    if abs(user_scale - 1.0) < 1e-9 and abs(tx) < 1e-9 and abs(ty) < 1e-9:
        return list(polyline)

    cx = base_glyph.width / 2.0
    cy = base_glyph.height / 2.0
    s = float(user_scale)

    scaled = [(cx + (x - cx) * s, cy + (y - cy) * s) for x, y in polyline]

    if abs(s - 1.0) >= 1e-9:
        x0, y0, _x1, _y1 = base_glyph.bbox()
        rx0 = cx + (x0 - cx) * s
        ry0 = cy + (y0 - cy) * s
        scaled = [(x - rx0, y - ry0) for x, y in scaled]

    if abs(tx) >= 1e-9 or abs(ty) >= 1e-9:
        scaled = [(x + tx, y + ty) for x, y in scaled]

    return scaled


# ── Rasterization (for tkinter preview & grid thumbnails) ───────────────────

def rasterize_glyph(
    glyph: Glyph,
    out_w: int,
    out_h: int,
    pad: int = 2,
    bg: tuple[int, int, int, int] = (0, 0, 0, 0),
    fg: tuple[int, int, int, int] = (0, 0, 0, 255),
    n_per_bezier: int = 16,
) -> Image.Image:
    """Render the glyph to an out_w x out_h RGBA image with aspect preserved.

    Each subpath is filled as a polygon (handwriting in the PDF is stored as
    closed filled outlines, so this is visually correct). For open subpaths
    (no ClosePath), we fall back to stroking a thin polyline.
    """
    img = Image.new("RGBA", (out_w, out_h), bg)
    if glyph.width <= 0 or glyph.height <= 0:
        return img

    inner_w = max(1, out_w - 2 * pad)
    inner_h = max(1, out_h - 2 * pad)
    scale = min(inner_w / glyph.width, inner_h / glyph.height)
    draw_w = glyph.width * scale
    draw_h = glyph.height * scale
    ox = (out_w - draw_w) / 2.0
    oy = (out_h - draw_h) / 2.0

    def _map(p: tuple[float, float]) -> tuple[float, float]:
        return (ox + p[0] * scale, oy + p[1] * scale)

    draw = ImageDraw.Draw(img)
    for sub in glyph.subpaths:
        # Determine if closed.
        has_close = any(isinstance(s, ClosePath) for s in sub)
        poly: list[tuple[float, float]] = []
        start = (0.0, 0.0)
        cx, cy = 0.0, 0.0
        for s in sub:
            if isinstance(s, MoveTo):
                poly.append(_map((s.x, s.y)))
                start = (s.x, s.y)
                cx, cy = s.x, s.y
            elif isinstance(s, LineTo):
                poly.append(_map((s.x, s.y)))
                cx, cy = s.x, s.y
            elif isinstance(s, CubicTo):
                for k in range(1, n_per_bezier + 1):
                    t = k / n_per_bezier
                    mt = 1.0 - t
                    x = (mt ** 3) * cx + 3 * (mt ** 2) * t * s.x1 \
                        + 3 * mt * (t ** 2) * s.x2 + (t ** 3) * s.x
                    y = (mt ** 3) * cy + 3 * (mt ** 2) * t * s.y1 \
                        + 3 * mt * (t ** 2) * s.y2 + (t ** 3) * s.y
                    poly.append(_map((x, y)))
                cx, cy = s.x, s.y
            elif isinstance(s, ClosePath):
                if poly and poly[-1] != _map(start):
                    poly.append(_map(start))
                cx, cy = start
        if len(poly) >= 3 and has_close:
            draw.polygon(poly, fill=fg)
        elif len(poly) >= 2:
            draw.line(poly, fill=fg, width=max(1, int(round(scale))))
    return img
