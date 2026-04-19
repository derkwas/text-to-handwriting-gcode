#!/usr/bin/env python3
"""Convert a vector-path PDF of handwriting to SVG.

The input PDF stores handwriting as filled closed paths (outlines of each
stroke). This script extracts those paths directly and emits them as SVG
<path> elements — no rasterisation, no skeletonisation, no reconstruction.
The SVG is pixel-identical to the PDF's vector content.

If you plan to feed this to a pen plotter, note that the paths are stroke
OUTLINES (each path is a closed loop around the ink shape), so the plotter
will trace the edge of every stroke. For single-pass centerline output,
that's a second conversion step (outline -> medial axis) which lives
elsewhere.

Usage:
    python pdf_to_svg.py handwriting_raw/"handwriting (1).pdf"
    python pdf_to_svg.py IN.pdf --out OUT.svg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import fitz


def _fmt(v: float) -> str:
    # 3-decimal precision is enough for plotter-scale output and keeps the
    # SVG compact (4050 items per page turns into a ~1 MB SVG otherwise).
    return f"{v:.3f}"


def _point_eq(a: fitz.Point, b: fitz.Point, tol: float = 0.01) -> bool:
    return abs(a.x - b.x) < tol and abs(a.y - b.y) < tol


def items_to_path_d(items: list) -> str:
    """Convert a PyMuPDF drawing items list to an SVG path 'd' string.

    Items arrive in PDF content-stream order. We infer subpath boundaries
    from endpoint continuity: when an item's start point doesn't coincide
    with the previous item's end, we emit a new 'M' (moveto) and mark the
    previous subpath closed.
    """
    parts: list[str] = []
    last_end: fitz.Point | None = None
    subpath_start: fitz.Point | None = None

    def _close_subpath() -> None:
        # Filled paths in PDF are implicitly closed at a path-painting
        # operator; reflect that in SVG with a trailing Z.
        if subpath_start is not None and last_end is not None \
                and not _point_eq(subpath_start, last_end):
            parts.append("Z")
        elif subpath_start is not None:
            parts.append("Z")

    for item in items:
        op = item[0]
        if op == "l":
            p1, p2 = item[1], item[2]
        elif op == "c":
            # Cubic Bezier: (op, p_start, c1, c2, p_end)
            p1, p2 = item[1], item[4]
        elif op == "re":
            # Rectangle (unusual in handwriting PDFs but handle gracefully).
            r = item[1]
            if last_end is not None:
                _close_subpath()
            parts.append(f"M {_fmt(r.x0)} {_fmt(r.y0)}")
            parts.append(f"H {_fmt(r.x1)}")
            parts.append(f"V {_fmt(r.y1)}")
            parts.append(f"H {_fmt(r.x0)}")
            parts.append("Z")
            last_end = fitz.Point(r.x0, r.y0)
            subpath_start = last_end
            continue
        elif op == "qu":
            # Quadrilateral.
            q = item[1]
            pts = [q.ul, q.ur, q.lr, q.ll]
            if last_end is not None:
                _close_subpath()
            parts.append(f"M {_fmt(pts[0].x)} {_fmt(pts[0].y)}")
            for p in pts[1:]:
                parts.append(f"L {_fmt(p.x)} {_fmt(p.y)}")
            parts.append("Z")
            last_end = pts[0]
            subpath_start = last_end
            continue
        else:
            # Unknown op — skip.
            continue

        if last_end is None or not _point_eq(p1, last_end):
            if last_end is not None:
                _close_subpath()
            parts.append(f"M {_fmt(p1.x)} {_fmt(p1.y)}")
            subpath_start = p1

        if op == "l":
            parts.append(f"L {_fmt(p2.x)} {_fmt(p2.y)}")
        else:  # "c"
            c1, c2 = item[2], item[3]
            parts.append(
                f"C {_fmt(c1.x)} {_fmt(c1.y)} "
                f"{_fmt(c2.x)} {_fmt(c2.y)} "
                f"{_fmt(p2.x)} {_fmt(p2.y)}"
            )
        last_end = p2

    if last_end is not None:
        _close_subpath()

    return " ".join(parts)


def _color(rgb: tuple[float, float, float] | None, default: str) -> str:
    if rgb is None:
        return default
    r, g, b = rgb
    return f"#{int(round(r * 255)):02x}{int(round(g * 255)):02x}{int(round(b * 255)):02x}"


def page_to_svg(page: fitz.Page) -> str:
    w, h = page.rect.width, page.rect.height
    drawings = page.get_drawings()

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {_fmt(w)} {_fmt(h)}" '
        f'width="{_fmt(w)}" height="{_fmt(h)}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    for d in drawings:
        items = d.get("items", [])
        if not items:
            continue
        d_str = items_to_path_d(items)
        if not d_str:
            continue
        dtype = d.get("type", "s")  # 's' stroke, 'f' fill, 'fs' both
        attrs: list[str] = [f'd="{d_str}"']
        if "f" in dtype:
            attrs.append(f'fill="{_color(d.get("fill"), "black")}"')
            fo = d.get("fill_opacity")
            if fo is not None and fo < 1.0:
                attrs.append(f'fill-opacity="{fo:.3f}"')
        else:
            attrs.append('fill="none"')
        if "s" in dtype:
            sw = d.get("width", 1.0)
            attrs.append(f'stroke="{_color(d.get("color"), "black")}"')
            attrs.append(f'stroke-width="{sw:.3f}"')
            attrs.append('stroke-linecap="round"')
            attrs.append('stroke-linejoin="round"')
        else:
            attrs.append('stroke="none"')
        parts.append(f'<path {" ".join(attrs)}/>')

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path, help="Input PDF")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output SVG (default: <pdf stem>.svg next to the PDF). "
                         "Multi-page PDFs get _p0, _p1, ... suffixes.")
    args = ap.parse_args()

    doc = fitz.open(str(args.pdf))
    n_pages = len(doc)
    if n_pages == 0:
        raise SystemExit(f"{args.pdf}: empty PDF")

    out_base = args.out or args.pdf.with_suffix(".svg")
    stem = out_base.with_suffix("")

    for i, page in enumerate(doc):
        svg = page_to_svg(page)
        if n_pages == 1:
            out = out_base
        else:
            out = stem.with_name(f"{stem.name}_p{i}").with_suffix(".svg")
        out.write_text(svg, encoding="utf-8")
        n_drawings = len(page.get_drawings())
        total_items = sum(len(d.get("items", [])) for d in page.get_drawings())
        print(f"wrote {out}  ({n_drawings} drawings, {total_items} segments)")


if __name__ == "__main__":
    main()
