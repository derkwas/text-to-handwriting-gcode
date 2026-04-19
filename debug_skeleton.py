#!/usr/bin/env python3
"""Diagnostic visualizer: raw StrokeGraph vs cleaned StrokeGraph.

For each diagnostic glyph sample this script renders three panels:

  LEFT   - raw graph produced by stroke_graph.extract_stroke_graph.
  MIDDLE - cleaned graph produced by graph_cleanup.clean_graph
           (clump collapse + spur pruning + deg-2 dissolution).
  RIGHT  - learn_font.trace_skeleton output, retained as a baseline.

Loops are drawn dashed; pure-cycle seams get a diamond marker. Endpoints
are green circles, junctions are red squares, both labeled with n<id>.

A per-glyph audit is also printed:

  - node / edge count before and after
  - stroke-width estimate and thresholds used
  - each collapsed junction cluster (old member IDs + canonical position)
  - each pruned spur (old edge id, length, endpoints)
  - each dissolved degree-2 node
  - the cleaned edge table

Outputs: debug/skeleton/<label>.png
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

from learn_font import binarize, trace_skeleton
from stroke_graph import Edge, StrokeGraph, extract_stroke_graph, summarize
from graph_cleanup import CleanupAudit, clean_graph


PROJECT = Path(__file__).resolve().parent
OUT_DIR = PROJECT / "debug" / "skeleton"

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


# ── Rendering ────────────────────────────────────────────────────────────────

def _prep_axis(ax, img_gray: np.ndarray, title: str) -> None:
    ax.imshow(img_gray, cmap="gray", alpha=0.35)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _label(ax, xy, text, color, dx: int = 4, dy: int = -4):
    ax.annotate(
        text,
        xy=xy,
        xytext=(dx, dy),
        textcoords="offset points",
        color=color,
        fontsize=8,
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )


def _draw_edge(ax, edge: Edge, color) -> None:
    xs_ = [p[0] for p in edge.pixels]
    ys_ = [p[1] for p in edge.pixels]
    linestyle = "--" if edge.is_loop else "-"
    linewidth = 2.2 if edge.is_loop else 1.8
    ax.plot(xs_, ys_, color=color, linewidth=linewidth,
            linestyle=linestyle, alpha=0.95)
    mid = edge.pixels[len(edge.pixels) // 2]
    tag = f"e{edge.id}"
    if edge.is_pure_cycle:
        tag += " \u21ba"
    elif edge.is_self_loop:
        tag += " \u21bb"
    _label(ax, mid, tag, color)
    if edge.is_pure_cycle:
        ax.scatter([xs_[0]], [ys_[0]], s=45, facecolor="white",
                   edgecolor=color, linewidth=1.4, zorder=5, marker="D")


def _draw_graph(ax, graph: StrokeGraph, img_gray: np.ndarray, title: str) -> None:
    _prep_axis(ax, img_gray, title)

    sy, sx = np.where(graph.skeleton)
    ax.scatter(sx, sy, s=2, c="#00b8d9", marker="s", alpha=0.30, linewidths=0)

    edge_colors = cm.tab20(np.linspace(0, 1, max(len(graph.edges), 1)))
    for i, edge in enumerate(graph.edges):
        _draw_edge(ax, edge, edge_colors[i % 20])

    if graph.endpoints:
        ex = [n.pos[0] for n in graph.endpoints]
        ey = [n.pos[1] for n in graph.endpoints]
        ax.scatter(
            ex, ey, s=70, c="#00cc44", marker="o",
            edgecolor="black", linewidth=1.2, zorder=6,
            label=f"endpoints ({len(graph.endpoints)})",
        )
        for n in graph.endpoints:
            _label(ax, n.pos, f"n{n.id}", "#007a2a")
    if graph.junctions:
        jx = [n.pos[0] for n in graph.junctions]
        jy = [n.pos[1] for n in graph.junctions]
        ax.scatter(
            jx, jy, s=85, c="#e53935", marker="s",
            edgecolor="black", linewidth=1.2, zorder=7,
            label=f"junctions ({len(graph.junctions)})",
        )
        for n in graph.junctions:
            _label(ax, n.pos, f"n{n.id}", "#8b0000")
    if graph.endpoints or graph.junctions:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)


def render_sample(label: str, image_path: Path, out_path: Path) -> dict:
    im_rgb = load_glyph_rgb(image_path)
    img_gray = np.array(im_rgb.convert("L"))

    ink = binarize(im_rgb)
    skel = sk_skeletonize(ink > 0)

    raw_graph = extract_stroke_graph(skel)
    cleaned_graph, audit = clean_graph(raw_graph, ink)
    legacy_paths = trace_skeleton(skel)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))

    _draw_graph(
        axes[0], raw_graph, img_gray,
        f"{label} — RAW: {len(raw_graph.nodes)} nodes "
        f"({len(raw_graph.endpoints)}e/{len(raw_graph.junctions)}j), "
        f"{len(raw_graph.edges)} edges, {len(raw_graph.loop_edges)} loop(s)",
    )
    _draw_graph(
        axes[1], cleaned_graph, img_gray,
        f"{label} — CLEANED: {len(cleaned_graph.nodes)} nodes "
        f"({len(cleaned_graph.endpoints)}e/{len(cleaned_graph.junctions)}j), "
        f"{len(cleaned_graph.edges)} edges, {len(cleaned_graph.loop_edges)} loop(s) "
        f"  [sw~{audit.stroke_width:.1f}, spur<={audit.spur_threshold}, "
        f"clump<={audit.clump_threshold}]",
    )

    ax2 = axes[2]
    _prep_axis(ax2, img_gray, f"{label} — legacy tracer: {len(legacy_paths)} path(s)")
    tcolors = cm.tab10(np.linspace(0, 1, max(len(legacy_paths), 1)))
    for i, path in enumerate(legacy_paths):
        xs_ = [p[0] for p in path]
        ys_ = [p[1] for p in path]
        color = tcolors[i % 10]
        ax2.plot(xs_, ys_, color=color, linewidth=1.6, alpha=0.95,
                 marker="o", markersize=2.2, markeredgewidth=0)
        ax2.scatter([xs_[0]], [ys_[0]], s=55, c="white",
                    edgecolor="black", linewidth=1.2, zorder=6)
        _label(ax2, (xs_[0], ys_[0]), f"p{i}", color)

    fig.suptitle(f"{label}   ({image_path.name})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "label": label,
        "image": str(image_path),
        "raw": summarize(raw_graph),
        "cleaned": summarize(cleaned_graph),
        "audit": audit,
        "legacy_paths": len(legacy_paths),
    }


# ── Reporting ────────────────────────────────────────────────────────────────

def _print_edge_table(stats: dict) -> None:
    rows = stats["edge_rows"]
    if not rows:
        print("      (no edges)")
        return
    print(f"      {'id':>3}  {'kind':<11}  {'start':>5}  {'end':>5}  {'len':>5}")
    print(f"      {'-'*3}  {'-'*11}  {'-'*5}  {'-'*5}  {'-'*5}")
    for row in rows:
        s = "-" if row["start"] is None else f"n{row['start']}"
        e = "-" if row["end"] is None else f"n{row['end']}"
        print(f"      {row['id']:>3}  {row['kind']:<11}  {s:>5}  {e:>5}  {row['length']:>5}")


def _print_sample_report(result: dict) -> None:
    label = result["label"]
    raw = result["raw"]
    cleaned = result["cleaned"]
    audit: CleanupAudit = result["audit"]

    print(f"\n{'=' * 78}")
    print(f"{label}   ({Path(result['image']).name})")
    print(f"{'=' * 78}")
    print(
        f"  stroke_width~{audit.stroke_width:.2f}px  "
        f"spur<={audit.spur_threshold}px  clump<={audit.clump_threshold}px"
    )
    print(
        f"  before: {raw['nodes']} nodes ({raw['endpoints']}e/{raw['junctions']}j), "
        f"{raw['edges']} edges ({raw['loops']} loop(s))"
    )
    print(
        f"  after : {cleaned['nodes']} nodes ({cleaned['endpoints']}e/{cleaned['junctions']}j), "
        f"{cleaned['edges']} edges ({cleaned['loops']} loop(s))"
    )

    if audit.collapsed_clusters:
        print(f"\n  collapsed junction clusters ({len(audit.collapsed_clusters)}):")
        for c in audit.collapsed_clusters:
            members = ", ".join(f"n{m}" for m in c["members"])
            x, y = c["canonical_pos"]
            print(f"    [{members}] -> canonical at ({x}, {y})")
    else:
        print("\n  collapsed junction clusters: none")

    if audit.pruned_edges:
        print(f"\n  pruned spurs ({len(audit.pruned_edges)}):")
        for p in audit.pruned_edges:
            s = "-" if p["start"] is None else f"n{p['start']}"
            e = "-" if p["end"] is None else f"n{p['end']}"
            print(f"    e{p['id']}  len={p['length']}  {s} -> {e}")
    else:
        print("\n  pruned spurs: none")

    if audit.dissolved_nodes:
        nodes_str = ", ".join(f"n{n}" for n in audit.dissolved_nodes)
        print(f"\n  dissolved deg-2 junctions: {nodes_str}")
    else:
        print("\n  dissolved deg-2 junctions: none")

    print("\n  cleaned edge table:")
    _print_edge_table(cleaned)


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

    for result in results:
        _print_sample_report(result)

    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    header = (
        f"{'sample':<10} "
        f"{'raw n':>5} {'raw e':>5} {'raw L':>5}    "
        f"{'new n':>5} {'new e':>5} {'new L':>5}    "
        f"{'legacy':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['label']:<10} "
            f"{r['raw']['nodes']:>5} {r['raw']['edges']:>5} {r['raw']['loops']:>5}    "
            f"{r['cleaned']['nodes']:>5} {r['cleaned']['edges']:>5} {r['cleaned']['loops']:>5}    "
            f"{r['legacy_paths']:>7}"
        )


if __name__ == "__main__":
    main()
