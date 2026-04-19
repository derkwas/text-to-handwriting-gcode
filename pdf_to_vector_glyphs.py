#!/usr/bin/env python3
"""Build a vector glyph library from a handwriting PDF.

Pipeline
--------
    PDF  ->  raster render (300 DPI)  ->  Tesseract OCR (char + bbox)
         ->  PDF.get_drawings()       (vector subpaths, PDF-point space)
         ->  subpath-to-char assignment by centroid
         ->  per-glyph SVG + index.json

Each glyph SVG contains exactly the vector strokes Tesseract labelled as
that character, translated so the glyph's top-left is at (0, 0). Coordinates
are PDF points (1/72 inch). A composer scales per glyph.

Replaces the old raster/skeletonise/decompose pipeline for building the
glyph library — vectors come straight from the PDF, no reconstruction.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz
from PIL import Image

# Reuse the existing OCR + loader so Tesseract config (PSM, path) matches the
# rest of the pipeline.
from pipeline.stage1_input import load_image
from pipeline.stage2_ocr import run_ocr


# ── Glyph folder naming (must match glyph_editor_app.safe_label_dir) ────────

def safe_label_dir(label: str) -> str:
    if label == "":
        return "empty"
    if len(label) == 1 and label.isalpha() and label.islower():
        return label
    if len(label) == 1 and label.isalpha() and label.isupper():
        return f"upper_{label}"
    named = {
        " ": "space", "/": "slash", "\\": "backslash", ":": "colon",
        "*": "asterisk", "?": "question", '"': "quote", "<": "lt",
        ">": "gt", "|": "pipe",
    }
    if label in named:
        return named[label]
    out: list[str] = []
    for ch in label:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append(f"u{ord(ch):04x}")
    s = "".join(out).strip(".")
    return s or "glyph"


# ── Subpath extraction from PyMuPDF drawings ────────────────────────────────

@dataclass
class Subpath:
    """One continuous closed path from the PDF, in PDF-point coordinates."""
    items: list  # list of fitz drawing items (('l', p1, p2) / ('c', p1, c1, c2, p2))
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1

    @property
    def centroid(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def _point_eq(p, q, tol: float = 0.01) -> bool:
    return abs(p.x - q.x) < tol and abs(p.y - q.y) < tol


def _items_to_subpaths(items: list) -> list[Subpath]:
    """Split a drawing's item list into subpaths at endpoint discontinuities."""
    subpaths: list[Subpath] = []
    current: list = []
    cur_xs: list[float] = []
    cur_ys: list[float] = []
    last_end = None

    def _flush() -> None:
        if not current:
            return
        x0, x1 = min(cur_xs), max(cur_xs)
        y0, y1 = min(cur_ys), max(cur_ys)
        subpaths.append(Subpath(items=list(current), bbox=(x0, y0, x1, y1)))

    for it in items:
        op = it[0]
        if op == "l":
            p_start, p_end = it[1], it[2]
            pts = [p_start, p_end]
        elif op == "c":
            p_start, c1, c2, p_end = it[1], it[2], it[3], it[4]
            pts = [p_start, c1, c2, p_end]
        elif op == "re":
            r = it[1]
            # Rectangles are self-contained subpaths.
            _flush()
            subpaths.append(Subpath(
                items=[it],
                bbox=(r.x0, r.y0, r.x1, r.y1),
            ))
            current = []
            cur_xs = []
            cur_ys = []
            last_end = None
            continue
        elif op == "qu":
            q = it[1]
            _flush()
            xs = [q.ul.x, q.ur.x, q.lr.x, q.ll.x]
            ys = [q.ul.y, q.ur.y, q.lr.y, q.ll.y]
            subpaths.append(Subpath(
                items=[it],
                bbox=(min(xs), min(ys), max(xs), max(ys)),
            ))
            current = []
            cur_xs = []
            cur_ys = []
            last_end = None
            continue
        else:
            continue

        if last_end is None or not _point_eq(p_start, last_end):
            _flush()
            current = []
            cur_xs = []
            cur_ys = []

        current.append(it)
        for p in pts:
            cur_xs.append(p.x)
            cur_ys.append(p.y)
        last_end = p_end

    _flush()
    return subpaths


def extract_page_subpaths(page: fitz.Page) -> list[Subpath]:
    out: list[Subpath] = []
    for d in page.get_drawings():
        out.extend(_items_to_subpaths(d.get("items", [])))
    return out


# ── Subpath serialisation to SVG path-d ─────────────────────────────────────

def _fmt(v: float) -> str:
    return f"{v:.3f}"


def subpath_to_svg_d(sub: Subpath, dx: float = 0.0, dy: float = 0.0) -> str:
    """Render a Subpath's items as an SVG path 'd' string, translated by
    (dx, dy). Filled paths from the PDF close implicitly; we add a trailing
    Z so the SVG matches."""
    parts: list[str] = []
    last_end = None
    started = False
    for it in sub.items:
        op = it[0]
        if op == "l":
            p1, p2 = it[1], it[2]
            if last_end is None or not _point_eq(p1, last_end):
                parts.append(f"M {_fmt(p1.x + dx)} {_fmt(p1.y + dy)}")
                started = True
            parts.append(f"L {_fmt(p2.x + dx)} {_fmt(p2.y + dy)}")
            last_end = p2
        elif op == "c":
            p1, c1, c2, p2 = it[1], it[2], it[3], it[4]
            if last_end is None or not _point_eq(p1, last_end):
                parts.append(f"M {_fmt(p1.x + dx)} {_fmt(p1.y + dy)}")
                started = True
            parts.append(
                f"C {_fmt(c1.x + dx)} {_fmt(c1.y + dy)} "
                f"{_fmt(c2.x + dx)} {_fmt(c2.y + dy)} "
                f"{_fmt(p2.x + dx)} {_fmt(p2.y + dy)}"
            )
            last_end = p2
        elif op == "re":
            r = it[1]
            parts.append(
                f"M {_fmt(r.x0 + dx)} {_fmt(r.y0 + dy)} "
                f"H {_fmt(r.x1 + dx)} V {_fmt(r.y1 + dy)} "
                f"H {_fmt(r.x0 + dx)} Z"
            )
            last_end = None
            started = False
        elif op == "qu":
            q = it[1]
            pts = [q.ul, q.ur, q.lr, q.ll]
            parts.append(f"M {_fmt(pts[0].x + dx)} {_fmt(pts[0].y + dy)}")
            for p in pts[1:]:
                parts.append(f"L {_fmt(p.x + dx)} {_fmt(p.y + dy)}")
            parts.append("Z")
            last_end = None
            started = False
    if started:
        parts.append("Z")
    return " ".join(parts)


# ── Assignment ──────────────────────────────────────────────────────────────

@dataclass
class CharSlot:
    char: str
    # bbox in PDF-point space (y grows downward, matching fitz.Page.rect).
    x0: float
    y0: float
    x1: float
    y1: float
    subpaths: list[Subpath]

    def contains(self, cx: float, cy: float, pad: float = 0.0) -> bool:
        return (self.x0 - pad) <= cx <= (self.x1 + pad) \
           and (self.y0 - pad) <= cy <= (self.y1 + pad)


def assign_subpaths(
    subpaths: list[Subpath],
    boxes: Iterable,
    pixel_to_pt: float,
) -> list[CharSlot]:
    """Assign each subpath to the character bbox containing its centroid.

    `boxes` are OCR CharBoxes with pixel-space .left/.top/.right/.bottom
    (top-left origin). We rescale by `pixel_to_pt` to match PDF points.
    """
    slots: list[CharSlot] = []
    for b in boxes:
        slots.append(CharSlot(
            char=b.char,
            x0=b.left * pixel_to_pt,
            y0=b.top * pixel_to_pt,
            x1=b.right * pixel_to_pt,
            y1=b.bottom * pixel_to_pt,
            subpaths=[],
        ))

    # First pass: strict containment.
    unassigned: list[Subpath] = []
    for sp in subpaths:
        cx, cy = sp.centroid
        placed = False
        for s in slots:
            if s.contains(cx, cy):
                s.subpaths.append(sp)
                placed = True
                break
        if not placed:
            unassigned.append(sp)

    # Second pass: nearest-slot fallback for strokes that fell between boxes
    # (e.g. an 'i' dot or a descender that extends past the OCR bbox).
    for sp in unassigned:
        cx, cy = sp.centroid
        best = None
        best_d = float("inf")
        for s in slots:
            sx = 0.5 * (s.x0 + s.x1)
            sy = 0.5 * (s.y0 + s.y1)
            d = (sx - cx) ** 2 + (sy - cy) ** 2
            if d < best_d:
                best_d = d
                best = s
        if best is not None:
            # Don't snap strokes that are clearly far from any char (likely
            # page noise). The threshold is 2× the median box size.
            med = _median_box_size(slots)
            if best_d ** 0.5 <= 2.0 * med:
                best.subpaths.append(sp)

    return slots


def _median_box_size(slots: list[CharSlot]) -> float:
    sizes = []
    for s in slots:
        sizes.append(max(s.x1 - s.x0, s.y1 - s.y0))
    if not sizes:
        return 20.0
    sizes.sort()
    return sizes[len(sizes) // 2]


# ── Glyph SVG writing ───────────────────────────────────────────────────────

def write_glyph_svg(
    slot: CharSlot,
    out_path: Path,
) -> tuple[float, float]:
    """Write one glyph SVG. Coordinates are translated so the glyph's
    alpha top-left is at (0, 0). Returns (width, height) in PDF points."""
    # Union bbox of the assigned subpaths (tighter than the OCR bbox).
    xs0, ys0, xs1, ys1 = [], [], [], []
    for sp in slot.subpaths:
        xs0.append(sp.bbox[0]); ys0.append(sp.bbox[1])
        xs1.append(sp.bbox[2]); ys1.append(sp.bbox[3])
    if not xs0:
        return 0.0, 0.0
    x0, y0 = min(xs0), min(ys0)
    x1, y1 = max(xs1), max(ys1)
    w = max(0.1, x1 - x0)
    h = max(0.1, y1 - y0)

    dx, dy = -x0, -y0
    path_d_parts = [subpath_to_svg_d(sp, dx=dx, dy=dy) for sp in slot.subpaths]
    body = "\n".join(f'<path d="{d}"/>' for d in path_d_parts if d)

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {_fmt(w)} {_fmt(h)}" '
        f'width="{_fmt(w)}" height="{_fmt(h)}">\n'
        f'<g fill="black" stroke="none">\n'
        f'{body}\n'
        f'</g>\n'
        f'</svg>\n'
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    return w, h


# ── Index writing ───────────────────────────────────────────────────────────

def build_index(
    slots: list[CharSlot],
    glyphs_root: Path,
    source: Path,
) -> dict:
    index: dict = {
        "version": 3,
        "format": "svg",
        "source": str(source),
        "glyphs": {},
    }
    counts: dict[str, int] = {}
    for slot in slots:
        if not slot.subpaths:
            continue
        label = slot.char
        label_dir = safe_label_dir(label)
        counts[label_dir] = counts.get(label_dir, 0) + 1
        idx = counts[label_dir] - 1
        filename = f"{label_dir}_{idx}.svg"
        out = glyphs_root / label_dir / filename
        w, h = write_glyph_svg(slot, out)
        if w == 0.0 and h == 0.0:
            counts[label_dir] -= 1
            continue
        entry = {
            "path": f"{label_dir}/{filename}",
            "orig_h": h,
            "orig_w": w,
            "bbox_pdf": [slot.x0, slot.y0, slot.x1, slot.y1],
        }
        index["glyphs"].setdefault(label, []).append(entry)
    return index


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path, help="Input handwriting PDF")
    ap.add_argument("--out", type=Path, default=Path("glyphs_vector"),
                    help="Output directory (default: glyphs_vector/)")
    ap.add_argument("--dpi", type=int, default=300,
                    help="Render DPI for OCR (default 300)")
    ap.add_argument("--psm", type=int, default=6,
                    help="Tesseract page-segmentation mode (default 6)")
    ap.add_argument("--page", type=int, default=None,
                    help="Process a single page (default: all pages)")
    args = ap.parse_args()

    doc = fitz.open(str(args.pdf))
    n_pages = len(doc)
    if args.page is not None:
        pages = [args.page]
    else:
        pages = list(range(n_pages))

    out_root: Path = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    total_index: dict = {
        "version": 3,
        "format": "svg",
        "source": str(args.pdf),
        "glyphs": {},
    }
    counts: dict[str, int] = {}

    for page_idx in pages:
        page = doc[page_idx]
        sample = load_image(args.pdf, page=page_idx, dpi=args.dpi)
        boxes = run_ocr(sample.original, psm=args.psm)

        subpaths = extract_page_subpaths(page)
        # Raster pixel  ->  PDF point:  points_per_inch / pixels_per_inch
        pixel_to_pt = 72.0 / float(sample.dpi)

        slots = assign_subpaths(subpaths, boxes, pixel_to_pt)

        # Merge this page's slots into the global index.
        for slot in slots:
            if not slot.subpaths:
                continue
            label = slot.char
            label_dir = safe_label_dir(label)
            counts[label_dir] = counts.get(label_dir, 0) + 1
            idx = counts[label_dir] - 1
            filename = f"{label_dir}_{idx}.svg"
            out_file = out_root / label_dir / filename
            w, h = write_glyph_svg(slot, out_file)
            if w == 0.0:
                counts[label_dir] -= 1
                continue
            total_index["glyphs"].setdefault(label, []).append({
                "path": f"{label_dir}/{filename}",
                "orig_h": round(h, 3),
                "orig_w": round(w, 3),
                "page": page_idx,
                "bbox_pdf": [round(slot.x0, 3), round(slot.y0, 3),
                             round(slot.x1, 3), round(slot.y1, 3)],
            })

        assigned = sum(1 for s in slots if s.subpaths)
        total_sub = sum(len(s.subpaths) for s in slots)
        print(f"page {page_idx}: OCR {len(boxes)} chars, "
              f"{len(subpaths)} subpaths -> {assigned} glyphs "
              f"({total_sub} subpaths assigned)")

    index_path = out_root / "index.json"
    index_path.write_text(
        json.dumps(total_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    total_glyphs = sum(len(v) for v in total_index["glyphs"].values())
    print(f"\nwrote {index_path}")
    print(f"  {len(total_index['glyphs'])} unique labels, "
          f"{total_glyphs} total glyph files")


if __name__ == "__main__":
    main()
