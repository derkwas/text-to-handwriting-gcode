#!/usr/bin/env python3
"""Step-7 diagnostic: stroke beautification / fitting.

For each diagnostic glyph, renders four panels side-by-side showing the
same per-stroke colouring through each stage of the fit pipeline:

  1. RAW      - the skeleton-pixel chain from stroke_decompose.
  2. RESAMPLED - uniform arc-length samples (default 1.5 px spacing).
  3. SMOOTHED  - Gaussian-smoothed (sigma=1.2), endpoints pinned / closure
                 preserved per stroke semantics.
  4. FITTED    - dense evaluation of the cubic B-spline through the
                 simplified (RDP, eps=0.6 px) control points. The
                 control points are overlaid as hollow circles.

Closed strokes are drawn dashed; open strokes solid. Each stroke keeps the
same colour across all four panels so the eye can follow a single stroke
through the transformation.

Prints per-stroke counts: raw / resampled / control.
Output: debug/fit/<label>.png
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
from stroke_graph import extract_stroke_graph
from graph_cleanup import clean_graph
from stroke_decompose import decompose
from stroke_fit import FittedStroke, fit_decomposition


PROJECT = Path(__file__).resolve().parent
OUT_DIR = PROJECT / "debug" / "fit"

SAMPLES: list[tuple[str, Path]] = [
    ("o_0", PROJECT / "glyphs_hw" / "o" / "o_0.png"),
    ("o_1", PROJECT / "glyphs_hw" / "o" / "o_1.png"),
    ("e_0", PROJECT / "glyphs_hw" / "e" / "e_0.png"),
    ("e_1", PROJECT / "glyphs_hw" / "e" / "e_1.png"),
    ("9_0", PROJECT / "glyphs_hw" / "9" / "9_0.png"),
    ("4_0", PROJECT / "glyphs_hw" / "4" / "4_0.png"),
    ("dollar_0", PROJECT / "glyphs_merged" / "u0024" / "u0024_0.png"),
]


def load_glyph_rgb(path: Path) -> Image.Image:
    im = Image.open(path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert("RGB")


def _prep(ax, img_gray: np.ndarray, title: str) -> None:
    ax.imshow(img_gray, cmap="gray", alpha=0.30)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _label(ax, xy, text, color, dx=4, dy=-4):
    ax.annotate(
        text, xy=xy, xytext=(dx, dy), textcoords="offset points",
        color=color, fontsize=7, fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2.4, foreground="white")],
    )


def _stroke_style(closed: bool) -> tuple[str, float]:
    return ("--" if closed else "-", 1.9)


def _draw_polyline(ax, points: np.ndarray, color, closed: bool,
                   marker: str | None = None, marker_size: float = 10) -> None:
    if len(points) < 1:
        return
    ls, lw = _stroke_style(closed)
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=lw,
            linestyle=ls, alpha=0.95)
    if marker is not None:
        ax.scatter(points[:, 0], points[:, 1], s=marker_size, c=[color],
                   marker=marker, linewidths=0.6, alpha=0.9, zorder=5)


def render_sample(label: str, image_path: Path, out_path: Path) -> dict:
    im_rgb = load_glyph_rgb(image_path)
    img_gray = np.array(im_rgb.convert("L"))
    ink = binarize(im_rgb)
    skel = sk_skeletonize(ink > 0)

    raw = extract_stroke_graph(skel)
    cleaned, _ = clean_graph(raw, ink)
    decomp = decompose(cleaned)
    fitted: list[FittedStroke] = fit_decomposition(decomp)

    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.8))
    _prep(axes[0], img_gray, f"{label} - raw chain")
    _prep(axes[1], img_gray, f"{label} - resampled (1.5px)")
    _prep(axes[2], img_gray, f"{label} - smoothed (sigma=1.2)")
    _prep(axes[3], img_gray, f"{label} - fitted spline + control points")

    n_strokes = max(len(fitted), 1)
    palette = cm.tab10(np.linspace(0, 1, 10))

    for i, fs in enumerate(fitted):
        color = palette[i % 10]

        if fs.raw_pixels:
            raw_pts = np.array(fs.raw_pixels, dtype=float)
            _draw_polyline(axes[0], raw_pts, color, fs.closed,
                           marker=".", marker_size=4)
            _label(axes[0], raw_pts[0], f"s{fs.id}", color)

        if len(fs.resampled) >= 1:
            _draw_polyline(axes[1], fs.resampled, color, fs.closed,
                           marker=".", marker_size=8)

        if len(fs.smoothed) >= 1:
            _draw_polyline(axes[2], fs.smoothed, color, fs.closed)

        curve = fs.sample_curve(n=160)
        if len(curve) >= 2:
            ls, lw = _stroke_style(fs.closed)
            axes[3].plot(curve[:, 0], curve[:, 1], color=color,
                         linewidth=2.2, linestyle=ls, alpha=0.95)
        if len(fs.simplified) >= 1:
            axes[3].scatter(fs.simplified[:, 0], fs.simplified[:, 1],
                            s=28, facecolor="white", edgecolor=color,
                            linewidths=1.3, zorder=6)

    fig.suptitle(f"{label}   ({image_path.name})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "label": label,
        "fitted": fitted,
    }


def _print_sample_report(label: str, fitted: list[FittedStroke]) -> None:
    print(f"\n{'=' * 78}")
    print(f"{label}")
    print("=" * 78)
    if not fitted:
        print("  (no strokes)")
        return
    header = f"  {'id':>3}  {'kind':<6}  {'raw':>5}  {'resamp':>6}  {'smooth':>6}  {'ctrl':>5}  spline"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for fs in fitted:
        kind = "closed" if fs.closed else "open"
        has_spline = "yes" if fs.spline is not None else "no "
        print(
            f"  {fs.id:>3}  {kind:<6}  "
            f"{fs.num_raw:>5}  {fs.num_resampled:>6}  "
            f"{fs.num_smoothed:>6}  {fs.num_control:>5}  {has_spline}"
        )


def main() -> None:
    print(f"Writing diagnostics to: {OUT_DIR}")
    results: list[dict] = []
    for label, path in SAMPLES:
        if not path.exists():
            print(f"  {label}: MISSING sample at {path}")
            continue
        out_path = OUT_DIR / f"{label}.png"
        result = render_sample(label, path, out_path)
        results.append(result)
        print(f"  wrote {out_path.relative_to(PROJECT)}")

    for r in results:
        _print_sample_report(r["label"], r["fitted"])

    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    header = (
        f"{'sample':<10} {'strokes':>7} "
        f"{'raw-tot':>7} {'res-tot':>7} {'ctrl-tot':>8} {'splines':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        fs_list: list[FittedStroke] = r["fitted"]
        raw_tot = sum(fs.num_raw for fs in fs_list)
        res_tot = sum(fs.num_resampled for fs in fs_list)
        ctrl_tot = sum(fs.num_control for fs in fs_list)
        sp = sum(1 for fs in fs_list if fs.spline is not None)
        print(
            f"{r['label']:<10} {len(fs_list):>7} "
            f"{raw_tot:>7} {res_tot:>7} {ctrl_tot:>8} {sp:>7}"
        )


if __name__ == "__main__":
    main()
