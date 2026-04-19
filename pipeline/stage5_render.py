"""Stage 5 – Text Generation & Rendering.

Reads glyphs from the library (Stage 4 output) and composes them into a
handwritten-looking raster image with controlled randomness.

Key behaviours
──────────────
* **Proportional scaling**: all glyphs scale based on their original height
  (orig_h from the scanned handwriting) relative to a reference height
  (median orig_h of common lowercase chars like a, c, e, etc.)
* **Digraph-first matching**: before picking a single-char glyph, check if
  the next 2 characters have a digraph in the library (e.g. ``"th"``).
* **Random variant selection**: when multiple scans of the same character
  exist, one is chosen at random each time.
* **Controlled variation per glyph**:
    - scale jitter  ±SCALE_JITTER  (fraction of target height)
    - rotation      ±ROT_MAX_DEG
    - x advance     ±X_JITTER_PX  (after scaling)
    - y baseline    ±Y_JITTER_PX
* Glyphs are bottom-aligned to a common baseline; all glyphs sit naturally
  on the same baseline (descenders of original handwriting are preserved).

Public API
──────────
    render_text(text, library_path, cfg) -> PIL.Image.Image
"""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ── Default render config ─────────────────────────────────────────────────────

DEFAULT_CFG: dict = {
    # Target glyph height in pixels (glyphs are rescaled to this for reference chars)
    "glyph_height": 80,

    # Canvas settings
    "canvas_width":  1400,
    "margin_x":       60,   # left & right margin in px
    "margin_y":       60,   # top margin in px
    "line_spacing":  1.6,   # multiple of glyph_height between baselines

    # Word / letter spacing (fraction of glyph_height)
    "word_space":    0.55,
    "letter_space":  0.08,

    # Variation parameters
    "scale_jitter":  0.06,   # ± fraction  (e.g. 0.06 → ±6% size variation)
    "rot_max_deg":   3.5,    # ± degrees rotation
    "x_jitter_px":   3,      # ± pixels horizontal wobble
    "y_jitter_px":   4,      # ± pixels vertical wobble (baseline drift)

    # Background and ink colour
    "bg_color":    (255, 255, 255),   # white
    "ink_color":   (15,  15,  60),    # near-black / dark-navy ink

    # Slight blur to smooth compositing edges
    "final_blur_radius": 0.4,

    # Lowercase visibility floor as a fraction of glyph_height.
    # Applied to single-letter lowercase labels (a-z) to keep very small
    # components from becoming visually lost after scaling/jitter/blur.
    "min_lowercase_ratio": 0.72,

    # Maximum number of variants kept per label after merging libraries.
    # Keeps sampling bounded and avoids over-weighting classes with many files.
    "max_variants_per_label": 8,
}


# ── Public API ─────────────────────────────────────────────────────────────────

def render_text(
    text: str,
    library_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    cfg: dict | None = None,
    seed: int | None = None,
) -> Image.Image:
    """Render *text* using glyphs from *library_path* and return a PIL Image.

    Parameters
    ----------
    text:
        The text to render. Newlines create new lines.
    library_path:
        Path (or list of paths) to glyph library folders (must contain
        ``index.json``). When multiple libraries are provided, variants are
        merged and used interchangeably.
    cfg:
        Render configuration dict; missing keys fall back to DEFAULT_CFG.
    seed:
        Optional random seed for reproducible output.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    c = {**DEFAULT_CFG, **(cfg or {})}
    library_paths = _normalize_library_paths(library_path)
    glyph_variants = _build_glyph_variants(
        library_paths,
        max_per_label=int(c["max_variants_per_label"]),
    )

    # Compute reference height from all available glyphs.
    all_orig_h = [
        v["orig_h"]
        for variants in glyph_variants.values()
        for v in variants
        if v["orig_h"] is not None
    ]
    ref_h = statistics.median(all_orig_h) if all_orig_h else 30.0

    glyph_h    = c["glyph_height"]
    line_step  = int(glyph_h * c["line_spacing"])
    word_sp    = int(glyph_h * c["word_space"])
    letter_sp  = int(glyph_h * c["letter_space"])
    margin_x   = c["margin_x"]
    margin_y   = c["margin_y"]
    canvas_w   = c["canvas_width"]

    # Compute lowercase baseline stats (single-letter a-z only). These are
    # used only to enforce a readability floor, not to hardcode specific chars.
    lower_orig_h = [
        v["orig_h"]
        for label, variants in glyph_variants.items()
        if _is_single_lower_ascii(label)
        for v in variants
        if v["orig_h"] is not None
    ]

    min_lower_h = max(8, round(glyph_h * c["min_lowercase_ratio"]))
    if lower_orig_h:
        # If lowercase samples are very small in this library, use the larger of
        # configured ratio and a robust 25th-percentile-derived floor.
        lower_q1 = _percentile(lower_orig_h, 25)
        q1_floor = max(8, round(glyph_h * lower_q1 / ref_h))
        min_lower_h = max(min_lower_h, q1_floor)

    # Multi-char labels sorted longest-first for greedy digraph matching
    labels_by_len = sorted(glyph_variants.keys(), key=len, reverse=True)
    # Alias used by _match_label (just needs truthy values per key)
    glyph_paths = glyph_variants

    # ── Pass 1: measure total canvas height ───────────────────────────────────
    lines     = text.splitlines()
    n_lines   = max(len(lines), 1)
    # Extra headroom for variation (rotations, jitter)
    canvas_h  = margin_y * 2 + line_step * (n_lines - 1) + round(glyph_h * 2.5) + 40

    # ── Pass 2: draw ──────────────────────────────────────────────────────────
    canvas = Image.new("RGB", (canvas_w, canvas_h), c["bg_color"])

    for line_idx, line in enumerate(lines):
        baseline_y = margin_y + glyph_h + line_idx * line_step
        x          = margin_x
        pos        = 0

        while pos < len(line):
            ch = line[pos]

            # Word space
            if ch == " ":
                x  += word_sp + random.randint(-c["x_jitter_px"], c["x_jitter_px"])
                pos += 1
                continue

            # Greedy digraph / single-char lookup
            matched_label, matched_len = _match_label(
                line, pos, labels_by_len, glyph_paths,
            )

            if matched_label is None:
                # Unknown character — advance a bit and skip
                x   += word_sp // 2
                pos += 1
                continue

            # Pick a random variant and scale proportionally based on original height.
            # All glyphs scale by the same factor: glyph_h / ref_h
            # This preserves the natural handwriting proportions.
            variant = random.choice(glyph_variants[matched_label])
            if variant["orig_h"] is not None:
                display_h = max(8, round(glyph_h * variant["orig_h"] / ref_h))
            else:
                display_h = glyph_h

            # Keep single lowercase glyphs readable (e.g. x) across libraries.
            # This is statistical and class-based, not character-specific.
            if _is_single_lower_ascii(matched_label):
                display_h = max(display_h, min_lower_h)
            
            glyph_img = _load_glyph(variant["path"], display_h, c)

            # Variation transforms
            scale  = 1.0 + random.uniform(-c["scale_jitter"], c["scale_jitter"])
            angle  = random.uniform(-c["rot_max_deg"], c["rot_max_deg"])
            dy     = random.randint(-c["y_jitter_px"], c["y_jitter_px"])
            dx_j   = random.randint(-c["x_jitter_px"], c["x_jitter_px"])

            glyph_img = _transform_glyph(glyph_img, scale, angle)

            # All glyphs sit on the baseline. No special descender handling —
            # the original handwriting proportions are preserved from the scans.
            paste_y = baseline_y - display_h + dy
            paste_x = x + dx_j

            # Clamp to canvas bounds
            paste_x = max(0, paste_x)
            paste_y = max(0, paste_y)

            # Composite (alpha channel from glyph acts as mask)
            canvas.paste(glyph_img, (paste_x, paste_y), mask=glyph_img)

            glyph_w = glyph_img.width
            x      += glyph_w + letter_sp
            pos    += matched_len

    # Slight blur to blend compositing edges
    if c["final_blur_radius"] > 0:
        canvas = canvas.filter(
            ImageFilter.GaussianBlur(radius=c["final_blur_radius"])
        )

    return canvas


def missing_glyphs(
    text: str,
    library_path: str | Path | list[str | Path] | tuple[str | Path, ...],
) -> list[str]:
    """Return a list of characters in *text* that have no glyph in the library."""
    library_paths = _normalize_library_paths(library_path)
    glyph_variants = _build_glyph_variants(
        library_paths,
        max_per_label=int(DEFAULT_CFG["max_variants_per_label"]),
    )
    available = set(glyph_variants.keys())
    labels_by_len = sorted(available, key=len, reverse=True)
    # Use a truthy sentinel so _match_label's `glyph_paths.get(label)` check passes
    sentinel: dict[str, list] = {k: [True] for k in available}
    missing = []
    for line in text.splitlines():
        pos = 0
        while pos < len(line):
            if line[pos] == " ":
                pos += 1
                continue
            label, length = _match_label(line, pos, labels_by_len, sentinel)
            if label is None:
                missing.append(line[pos])
                pos += 1
            else:
                pos += length
    return sorted(set(missing))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_index(library_path: Path) -> dict:
    idx_path = library_path / "index.json"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"Glyph library index not found: {idx_path}\n"
            "Run run_pipeline.py first to build the library."
        )
    with open(idx_path, encoding="utf-8") as f:
        return json.load(f)


def _normalize_library_paths(
    library_path: str | Path | list[str | Path] | tuple[str | Path, ...]
) -> list[Path]:
    """Normalize one-or-many library paths into a non-empty list."""
    if isinstance(library_path, (list, tuple)):
        paths = [Path(p) for p in library_path]
    else:
        paths = [Path(library_path)]
    if not paths:
        raise ValueError("At least one glyph library path is required.")
    return paths


def _build_glyph_variants(
    library_paths: list[Path],
    max_per_label: int = 8,
) -> dict[str, list[dict]]:
    """Merge glyph variants from one or more libraries.

    Returns
    -------
    dict[str, list[dict]]
        Mapping label -> list of {"path": Path, "orig_h": int|None}
    """
    merged: dict[str, list[dict]] = {}
    seen: set[tuple[str, str]] = set()

    for lib_path in library_paths:
        index = _load_index(lib_path)
        for char, entries in index.get("glyphs", {}).items():
            out = merged.setdefault(char, [])
            for e in entries:
                if isinstance(e, dict):
                    rel = e["path"]
                    item = {"path": lib_path / rel, "orig_h": e.get("orig_h")}
                else:
                    rel = e
                    item = {"path": lib_path / rel, "orig_h": None}

                # Deduplicate exact same relative asset within same label.
                # We include library root to avoid cross-library collisions.
                key = (char, str(item["path"]))
                if key in seen:
                    continue
                seen.add(key)
                out.append(item)

    # Stable ordering then cap variants per label.
    capped: dict[str, list[dict]] = {}
    for char, variants in merged.items():
        variants = sorted(variants, key=lambda v: str(v["path"]))
        if max_per_label > 0:
            variants = variants[:max_per_label]
        capped[char] = variants

    return capped


def render_glyph_preview(
    library_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    out_path: str | Path,
    max_per_label: int = 8,
    thumb_h: int = 64,
    cols: int = 8,
    padding: int = 10,
    bg_color: tuple[int, int, int] = (245, 245, 245),
    cell_bg: tuple[int, int, int] = (255, 255, 255),
) -> Path:
    """Save a contact-sheet preview of merged glyph variants.

    Each row is one label. Up to *max_per_label* variants are shown per label.
    """
    library_paths = _normalize_library_paths(library_path)
    glyph_variants = _build_glyph_variants(library_paths, max_per_label=max_per_label)
    labels = sorted(glyph_variants.keys())
    if not labels:
        raise ValueError("No glyphs available to preview.")

    # label column + N variant columns
    label_w = 120
    cell_w = thumb_h + 20
    cols = max(1, min(cols, max_per_label))
    canvas_w = padding * 2 + label_w + cols * cell_w
    row_h = thumb_h + 16
    canvas_h = padding * 2 + len(labels) * row_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    for r, label in enumerate(labels):
        y0 = padding + r * row_h

        draw.text((padding + 4, y0 + 4), repr(label), fill=(20, 20, 20))

        variants = glyph_variants[label][:cols]
        for c_idx, variant in enumerate(variants):
            x0 = padding + label_w + c_idx * cell_w
            cell = Image.new("RGB", (cell_w - 6, row_h - 6), cell_bg)

            g = Image.open(variant["path"]).convert("RGBA")
            scale = thumb_h / max(1, g.height)
            nw = max(1, round(g.width * scale))
            nh = max(1, round(g.height * scale))
            g = g.resize((nw, nh), Image.LANCZOS)

            px = max(0, (cell.width - nw) // 2)
            py = max(0, (cell.height - nh) // 2)
            cell.paste(g, (px, py), g)
            canvas.paste(cell, (x0, y0))

    out_path = Path(out_path)
    canvas.save(out_path)
    return out_path


def _is_single_lower_ascii(label: str) -> bool:
    """True for single-character lowercase latin labels (a-z)."""
    return len(label) == 1 and "a" <= label <= "z"


def _percentile(values: list[int], p: float) -> float:
    """Simple percentile with linear interpolation for small lists."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * (p / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _match_label(
    line: str,
    pos: int,
    labels_by_len: list[str],
    glyph_paths: dict[str, list],
) -> tuple[str | None, int]:
    """Return (best_matching_label, length) or (None, 1) if no match."""
    for label in labels_by_len:
        end = pos + len(label)
        if line[pos:end] == label and glyph_paths.get(label):
            return label, len(label)
    return None, 1


def _load_glyph(path: Path, target_h: int, c: dict) -> Image.Image:
    """Load a glyph PNG, recolour ink, and resize to target_h."""
    img = Image.open(path).convert("RGBA")

    # Recolour: replace black ink pixels with configured ink_color
    r, g, b_ch, a = img.split()
    ink = c["ink_color"]
    r = r.point(lambda _: ink[0])
    g = g.point(lambda _: ink[1])
    b_ch = b_ch.point(lambda _: ink[2])
    img = Image.merge("RGBA", (r, g, b_ch, a))

    # Resize proportionally so height = target_h
    orig_w, orig_h = img.size
    if orig_h == 0:
        return img
    new_w = max(1, round(orig_w * target_h / orig_h))
    return img.resize((new_w, target_h), Image.LANCZOS)


def _transform_glyph(img: Image.Image, scale: float, angle_deg: float) -> Image.Image:
    """Apply scale then rotation, keeping alpha, white expansion areas transparent."""
    if scale != 1.0:
        new_w = max(1, round(img.width  * scale))
        new_h = max(1, round(img.height * scale))
        img   = img.resize((new_w, new_h), Image.LANCZOS)

    if abs(angle_deg) > 0.1:
        # Rotate with expand=True so nothing is clipped; fill with transparent
        img = img.rotate(angle_deg, resample=Image.BICUBIC,
                         expand=True, fillcolor=(0, 0, 0, 0))
    return img
