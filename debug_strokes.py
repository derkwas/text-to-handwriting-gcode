#!/usr/bin/env python3
"""Step-6 diagnostic: stroke decomposition by junction pairing.

For each diagnostic glyph samples this script renders three panels:

  LEFT   - cleaned graph (step-4 output).
  MIDDLE - junction pairings: thick colored arcs between paired sides at
           each junction (one color per pair), with the turn angle printed
           at the arc midpoint. Unpaired sides are marked with a black X
           at their departure pixel.
  RIGHT  - final stroke decomposition: each stroke in its own color.
           Closed strokes are drawn dashed. Pen-start dot + s<id> label.

Prints per-junction pairing decisions and per-stroke edge sequences.

This runs on the cleaned graph from graph_cleanup.clean_graph and does NOT
touch learn_font.py. Output -> debug/strokes/<label>.png.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.morphology import skeletonize as sk_skeletonize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import cm

from learn_font import binarize
from stroke_graph import Edge, StrokeGraph, extract_stroke_graph
from graph_cleanup import clean_graph
from stroke_decompose import (
    JunctionPairing,
    SideId,
    Stroke,
    StrokeDecomposition,
    decompose,
)


PROJECT = Path(__file__).resolve().parent
OUT_DIR = PROJECT / "debug" / "strokes"

SAMPLES: list[tuple[str, Path]] = [
    ("o_0", PROJECT / "glyphs_hw" / "o" / "o_0.png"),
    ("o_1", PROJECT / "glyphs_hw" / "o" / "o_1.png"),
    ("e_0", PROJECT / "glyphs_hw" / "e" / "e_0.png"),
    ("e_1", PROJECT / "glyphs_hw" / "e" / "e_1.png"),
    ("9_0", PROJECT / "glyphs_hw" / "9" / "9_0.png"),
    ("4_0", PROJECT / "glyphs_hw" / "4" / "4_0.png"),
    ("dollar_0", PROJECT / "glyphs_merged" / "u0024" / "u0024_0.png"),
]


# ── Glyph loading ────────────────────────────────────────────────────────────

def load_glyph_rgb(path: Path) -> Image.Image:
    im = Image.open(path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert("RGB")


# ── Plot helpers ─────────────────────────────────────────────────────────────

def _prep_axis(ax, img_gray: np.ndarray, title: str) -> None:
    ax.imshow(img_gray, cmap="gray", alpha=0.35)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _label(ax, xy, text, color, dx: int = 4, dy: int = -4):
    ax.annotate(
        text, xy=xy, xytext=(dx, dy), textcoords="offset points",
        color=color, fontsize=8, fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )


def _side_departure(edge: Edge, side: str, k: int = 5) -> tuple[int, int]:
    """Pixel a few steps from the junction-end (used to draw arcs / X marks)."""
    pixels = edge.pixels
    k = max(1, min(k, len(pixels) - 1))
    return pixels[k] if side == "a" else pixels[-1 - k]


# ── Panel 1: cleaned graph ───────────────────────────────────────────────────

def _draw_cleaned_panel(ax, graph: StrokeGraph, img_gray: np.ndarray, title: str) -> None:
    _prep_axis(ax, img_gray, title)
    sy, sx = np.where(graph.skeleton)
    ax.scatter(sx, sy, s=2, c="#00b8d9", alpha=0.30, linewidths=0)

    edge_colors = cm.tab20(np.linspace(0, 1, max(len(graph.edges), 1)))
    for i, e in enumerate(graph.edges):
        xs = [p[0] for p in e.pixels]
        ys = [p[1] for p in e.pixels]
        ls = "--" if e.is_loop else "-"
        ax.plot(xs, ys, color=edge_colors[i % 20], linewidth=1.8,
                linestyle=ls, alpha=0.9)
        mid = e.pixels[len(e.pixels) // 2]
        tag = f"e{e.id}"
        if e.is_pure_cycle:
            tag += " \u21ba"
        elif e.is_self_loop:
            tag += " \u21bb"
        _label(ax, mid, tag, edge_colors[i % 20])

    for n in graph.endpoints:
        ax.scatter([n.pos[0]], [n.pos[1]], s=60, c="#00cc44", marker="o",
                   edgecolor="black", linewidth=1.1, zorder=6)
        _label(ax, n.pos, f"n{n.id}", "#007a2a")
    for n in graph.junctions:
        ax.scatter([n.pos[0]], [n.pos[1]], s=75, c="#e53935", marker="s",
                   edgecolor="black", linewidth=1.1, zorder=7)
        _label(ax, n.pos, f"n{n.id}", "#8b0000")


# ── Panel 2: junction pairings ───────────────────────────────────────────────

def _draw_pairings_panel(
    ax, graph: StrokeGraph, pairings: list[JunctionPairing],
    img_gray: np.ndarray, title: str,
) -> None:
    _prep_axis(ax, img_gray, title)
    edges_by_id = {e.id: e for e in graph.edges}

    sy, sx = np.where(graph.skeleton)
    ax.scatter(sx, sy, s=1, c="#aaa", alpha=0.4, linewidths=0)
    for e in graph.edges:
        xs = [p[0] for p in e.pixels]
        ys = [p[1] for p in e.pixels]
        ax.plot(xs, ys, color="#b5b5b5", linewidth=1.1, alpha=0.7)

    for n in graph.endpoints:
        ax.scatter([n.pos[0]], [n.pos[1]], s=55, c="#00cc44", marker="o",
                   edgecolor="black", linewidth=1.0, zorder=6)
    for n in graph.junctions:
        ax.scatter([n.pos[0]], [n.pos[1]], s=70, c="#e53935", marker="s",
                   edgecolor="black", linewidth=1.0, zorder=7)

    pair_colors = cm.tab10(np.linspace(0, 1, 10))
    arc_idx = 0
    for jp in pairings:
        for (sa, sb, turn) in jp.pairs:
            pa = _side_departure(edges_by_id[sa[0]], sa[1])
            pb = _side_departure(edges_by_id[sb[0]], sb[1])
            color = pair_colors[arc_idx % 10]
            arc_idx += 1
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color,
                    linewidth=2.6, alpha=0.95, zorder=8, solid_capstyle="round")
            mx = (pa[0] + pb[0]) / 2
            my = (pa[1] + pb[1]) / 2
            _label(ax, (mx, my), f"{turn:.0f}\u00b0", color)
        for sid in jp.unpaired:
            p = _side_departure(edges_by_id[sid[0]], sid[1])
            ax.scatter([p[0]], [p[1]], s=90, marker="x",
                       c="black", linewidth=2.2, zorder=9)


# ── Panel 3: stroke decomposition ────────────────────────────────────────────

def _draw_strokes_panel(
    ax, decomp: StrokeDecomposition, img_gray: np.ndarray, title: str,
) -> None:
    _prep_axis(ax, img_gray, title)
    sy, sx = np.where(decomp.graph.skeleton)
    ax.scatter(sx, sy, s=1, c="#ccc", alpha=0.35, linewidths=0)

    n_strokes = max(len(decomp.strokes), 1)
    colors = cm.tab10(np.linspace(0, 1, 10))
    for i, s in enumerate(decomp.strokes):
        if not s.pixels:
            continue
        xs = [p[0] for p in s.pixels]
        ys = [p[1] for p in s.pixels]
        color = colors[i % 10]
        linestyle = "--" if s.closed else "-"
        ax.plot(xs, ys, color=color, linewidth=2.2, alpha=0.95,
                linestyle=linestyle)
        ax.scatter([xs[0]], [ys[0]], s=60, c="white",
                   edgecolor="black", linewidth=1.2, zorder=6)
        tag = f"s{s.id}" + (" (closed)" if s.closed else "")
        _label(ax, (xs[0], ys[0]), tag, color)


# ── Rendering + reporting ────────────────────────────────────────────────────

def render_sample(label: str, image_path: Path, out_path: Path) -> dict:
    im_rgb = load_glyph_rgb(image_path)
    img_gray = np.array(im_rgb.convert("L"))
    ink = binarize(im_rgb)
    skel = sk_skeletonize(ink > 0)

    raw = extract_stroke_graph(skel)
    cleaned, _ = clean_graph(raw, ink)
    decomp = decompose(cleaned)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    _draw_cleaned_panel(
        axes[0], cleaned, img_gray,
        f"{label} - cleaned graph "
        f"({len(cleaned.nodes)} nodes, {len(cleaned.edges)} edges, "
        f"{len(cleaned.loop_edges)} loop(s))",
    )
    _draw_pairings_panel(
        axes[1], cleaned, decomp.pairings, img_gray,
        f"{label} - junction pairings",
    )
    _draw_strokes_panel(
        axes[2], decomp, img_gray,
        f"{label} - strokes: {len(decomp.strokes)} "
        f"({sum(1 for s in decomp.strokes if s.closed)} closed)",
    )

    fig.suptitle(f"{label}   ({image_path.name})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {"label": label, "image": str(image_path), "decomp": decomp}


def _fmt_side(s: SideId) -> str:
    return f"e{s[0]}.{s[1]}"


def _print_sample_report(label: str, decomp: StrokeDecomposition) -> None:
    print(f"\n{'=' * 78}")
    print(f"{label}")
    print("=" * 78)

    graph = decomp.graph
    print(
        f"  graph: {len(graph.nodes)} nodes "
        f"({len(graph.endpoints)}e/{len(graph.junctions)}j), "
        f"{len(graph.edges)} edges"
    )

    if not decomp.pairings:
        print("\n  junction pairings: (no junctions)")
    else:
        print("\n  junction pairings:")
        for jp in decomp.pairings:
            if jp.pairs:
                parts = [
                    f"({_fmt_side(a)},{_fmt_side(b)}) turn={t:.0f}deg"
                    for (a, b, t) in jp.pairs
                ]
                paired = "; ".join(parts)
            else:
                paired = "(none)"
            if jp.unpaired:
                unp = ", ".join(_fmt_side(s) for s in jp.unpaired)
            else:
                unp = "(none)"
            print(f"    n{jp.node_id}: pairs [{paired}]  unpaired [{unp}]")

    print(f"\n  strokes ({len(decomp.strokes)}):")
    for s in decomp.strokes:
        kind = "closed" if s.closed else "open  "
        seq = " -> ".join(
            f"e{eid}{'r' if rev else ''}" for eid, rev in s.edge_sequence
        )
        start = "-" if s.start_node is None else f"n{s.start_node}"
        end = "-" if s.end_node is None else f"n{s.end_node}"
        print(
            f"    s{s.id} {kind}  {start}->{end}  "
            f"[{seq}]  ({len(s.pixels)}px)"
        )


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Writing diagnostics to: {OUT_DIR}")
    results: list[dict] = []

    for label, path in SAMPLES:
        if not path.exists():
            print(f"\n  {label}: MISSING sample at {path}")
            continue
        out_path = OUT_DIR / f"{label}.png"
        result = render_sample(label, path, out_path)
        results.append(result)
        print(f"  wrote {out_path.relative_to(PROJECT)}")

    for r in results:
        _print_sample_report(r["label"], r["decomp"])

    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    header = (
        f"{'sample':<10} "
        f"{'nodes':>5} {'edges':>5}   "
        f"{'pairs':>5} {'unpair':>6}   "
        f"{'strokes':>7} {'closed':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        d: StrokeDecomposition = r["decomp"]
        n_pairs = sum(len(jp.pairs) for jp in d.pairings)
        n_unp = sum(len(jp.unpaired) for jp in d.pairings)
        n_closed = sum(1 for s in d.strokes if s.closed)
        print(
            f"{r['label']:<10} "
            f"{len(d.graph.nodes):>5} {len(d.graph.edges):>5}   "
            f"{n_pairs:>5} {n_unp:>6}   "
            f"{len(d.strokes):>7} {n_closed:>6}"
        )


if __name__ == "__main__":
    main()
