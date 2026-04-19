#!/usr/bin/env python3
"""Step-7.5 diagnostic: character-aware ambiguity resolution.

For each diagnostic glyph:
  - runs cleanup + candidate enumeration + prior-based scoring
  - prints all candidates with features and scores (winner marked *)
  - renders the top-3 candidates side-by-side so the effect of the prior
    is visible at a glance

The label each sample represents is hard-coded in SAMPLES so we pass the
right character to the resolver (e.g. dollar_0 -> '$').

Output: debug/ambiguity/<label>.png
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
from ambiguity import ScoredCandidate, resolve, score_prior, extract_features
from char_priors import get_prior
from stroke_decompose import StrokeDecomposition


PROJECT = Path(__file__).resolve().parent
OUT_DIR = PROJECT / "debug" / "ambiguity"

# (label, path, char) — the char the resolver should use as prior.
SAMPLES: list[tuple[str, Path, str]] = [
    ("o_0",   PROJECT / "glyphs_hw" / "o" / "o_0.png",                             "o"),
    ("e_0",   PROJECT / "glyphs_hw" / "e" / "e_0.png",                             "e"),
    ("9_0",   PROJECT / "glyphs_hw" / "9" / "9_0.png",                             "9"),
    ("4_0",   PROJECT / "glyphs_hw" / "4" / "4_0.png",                             "4"),
    ("a_0",   PROJECT / "glyphs_merged" / "a"       / "glyphs_hw1_a_0.png",        "a"),
    # Character-rule targets (B/8/P/R/D/O/0/Q).
    ("B_0",   PROJECT / "glyphs_merged" / "upper_B" / "glyphs_hw1_upper_B_0.png",  "B"),
    ("8_0",   PROJECT / "glyphs_merged" / "8"       / "glyphs_hw1_8_0.png",        "8"),
    ("P_0",   PROJECT / "glyphs_merged" / "upper_P" / "glyphs_hw1_upper_P_0.png",  "P"),
    ("R_0",   PROJECT / "glyphs_merged" / "upper_R" / "glyphs_hw1_upper_R_0.png",  "R"),
    ("D_0",   PROJECT / "glyphs_merged" / "upper_D" / "glyphs_hw1_upper_D_0.png",  "D"),
    ("O_0",   PROJECT / "glyphs_merged" / "upper_O" / "glyphs_hw1_upper_O_0.png",  "O"),
    ("zero0", PROJECT / "glyphs_merged" / "0"       / "glyphs_hw1_0_0.png",        "0"),
    ("Q_0",   PROJECT / "glyphs_merged" / "upper_Q" / "glyphs_hw1_upper_Q_0.png",  "Q"),
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


def _draw_decomp(ax, decomp: StrokeDecomposition, img_gray: np.ndarray,
                 title: str, is_winner: bool) -> None:
    _prep(ax, img_gray, title)
    palette = cm.tab10(np.linspace(0, 1, 10))
    for i, s in enumerate(decomp.strokes):
        if not s.pixels:
            continue
        xs = [p[0] for p in s.pixels]
        ys = [p[1] for p in s.pixels]
        color = palette[i % 10]
        ax.plot(xs, ys, color=color, linewidth=2.1,
                linestyle="--" if s.closed else "-", alpha=0.95)
        ax.scatter([xs[0]], [ys[0]], s=45, c="white",
                   edgecolor="black", linewidth=1.1, zorder=6)
        _label(ax, (xs[0], ys[0]), f"s{s.id}", color)

    if is_winner:
        for spine in ax.spines.values():
            spine.set_edgecolor("#00a040")
            spine.set_linewidth(2.4)


def render_sample(label: str, image_path: Path, char: str, out_path: Path) -> dict:
    im_rgb = load_glyph_rgb(image_path)
    img_gray = np.array(im_rgb.convert("L"))
    ink = binarize(im_rgb)
    skel = sk_skeletonize(ink > 0)
    raw = extract_stroke_graph(skel)
    cleaned, _ = clean_graph(raw, ink)

    winner, scored = resolve(cleaned, char=char)

    # Also record what the geometry-only pick would have been, so the report
    # can flag cases where the prior actually changed the outcome.
    geom_only_idx = min(
        range(len(scored)), key=lambda i: scored[i].geom_cost
    ) if scored else None

    n_panels = min(3, len(scored)) if scored else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 5.2))
    if n_panels == 1:
        axes = [axes]
    winner_id = id(winner)
    for i in range(n_panels):
        cand = scored[i]
        suffix = "  [WINNER]" if id(cand.decomposition) == winner_id else ""
        title = (
            f"#{i} "
            f"geom={cand.geom_cost:.1f}deg  "
            f"prior={cand.prior_penalty:.2f}  "
            f"total={cand.total_score:.1f}{suffix}"
        )
        _draw_decomp(axes[i], cand.decomposition, img_gray, title,
                     is_winner=(id(cand.decomposition) == winner_id))

    fig.suptitle(f"{label}   ({image_path.name})   char={char!r}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "label": label,
        "char": char,
        "winner_idx": 0,   # scored is sorted by total_score
        "geom_only_idx": geom_only_idx,
        "scored": scored,
    }


def _print_report(result: dict) -> None:
    label = result["label"]
    char = result["char"]
    scored: list[ScoredCandidate] = result["scored"]
    geom_idx = result["geom_only_idx"]

    print(f"\n{'=' * 78}")
    print(f"{label}   (char={char!r})")
    print("=" * 78)
    if not scored:
        print("  (no candidates)")
        return

    prior_flipped = (geom_idx is not None and geom_idx != 0)
    print(f"  candidates: {len(scored)}  "
          f"(prior changed pick: {'YES' if prior_flipped else 'no'})")
    print()
    header = (
        f"  {'#':>2}  {'geom':>6}  {'prior':>5}  {'custom':>6}  "
        f"{'disc':>4}  {'total':>6}  {'features':<60}  mark"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, c in enumerate(scored):
        marks = []
        if i == 0:
            marks.append("WIN")
        if geom_idx is not None and i == geom_idx and i != 0:
            marks.append("geom-best")
        mark = ",".join(marks) if marks else ""
        print(
            f"  {i:>2}  {c.geom_cost:>6.1f}  {c.prior_penalty:>5.2f}  "
            f"{c.custom_penalty:>6.2f}  {c.discrim_penalty:>4.2f}  "
            f"{c.total_score:>6.1f}  {c.features.as_row():<60}  {mark}"
        )


def main() -> None:
    print(f"Writing diagnostics to: {OUT_DIR}")
    results: list[dict] = []
    for label, path, char in SAMPLES:
        if not path.exists():
            print(f"  {label}: MISSING {path}")
            continue
        out_path = OUT_DIR / f"{label}.png"
        result = render_sample(label, path, char, out_path)
        results.append(result)
        print(f"  wrote {out_path.relative_to(PROJECT)}")

    for r in results:
        _print_report(r)

    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    header = (
        f"{'sample':<10} {'char':>4}  {'cand':>4}  "
        f"{'win-strokes':>11}  {'flipped':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        scored: list[ScoredCandidate] = r["scored"]
        win = scored[0] if scored else None
        n_strokes = win.features.n_strokes if win else 0
        flipped = (r["geom_only_idx"] is not None and r["geom_only_idx"] != 0)
        print(
            f"{r['label']:<10} {r['char']!r:>4}  {len(scored):>4}  "
            f"{n_strokes:>11}  {('YES' if flipped else 'no'):>7}"
        )


if __name__ == "__main__":
    main()
