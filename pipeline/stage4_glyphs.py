"""Stage 4 – Glyph Library: extract, normalise, and store character images.

For each :class:`~pipeline.stage3_segment.LabeledComponent` that has a valid
character label:

1. The ink blob is **cropped** from the full binarised image.
2. The crop is **masked** – pixels belonging to neighbouring blobs are zeroed,
   so glyphs don't pollute each other.
3. The crop is **normalised**: resized so its height = ``GLYPH_HEIGHT - 2 × PAD``
   pixels, then padded to give ``GLYPH_HEIGHT`` total height.  Width scales
   proportionally.
4. Saved as an **RGBA PNG** (ink=black, background=transparent) to::

       glyphs/{char_name}/{char_name}_{N}.png

   where N auto-increments so multiple samples accumulate over runs.

5. An ``index.json`` at the library root is updated after every run.

Library index format
────────────────────
.. code-block:: json

    {
      "version": 1,
      "glyphs": {
        "a": ["a/a_0.png", "a/a_1.png"],
        "upper_A": ["upper_A/upper_A_0.png"]
      }
    }
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .stage3_segment import LabeledComponent

# ── Constants ─────────────────────────────────────────────────────────────────

GLYPH_HEIGHT: int = 64   # normalised glyph height in pixels (before padding)
PAD: int          = 4    # transparent padding on each side (pixels)
INDEX_FILE        = "index.json"


# ── Public API ────────────────────────────────────────────────────────────────

def normalize_glyph(
    component: LabeledComponent,
    source_binary: np.ndarray,
    target_h: int = GLYPH_HEIGHT,
    pad: int = PAD,
) -> Image.Image:
    """Crop and normalise one glyph to a consistent height.

    Returns an RGBA PIL Image: ink = black opaque, background = transparent.

    Parameters
    ----------
    component:
        The segmented ink blob (provides bbox + mask).
    source_binary:
        Full-page binarised image (uint8 H×W, 255 = ink).
    target_h:
        Total output image height including padding.
    pad:
        Transparent border added on every side.
    """
    y0, y1 = component.top, component.bottom
    x0, x1 = component.left, component.right

    # Defensive clamp
    H, W  = source_binary.shape
    y0, y1 = max(0, y0), min(H, y1)
    x0, x1 = max(0, x0), min(W, x1)

    crop = source_binary[y0:y1, x0:x1].copy()
    if crop.size == 0:
        return Image.new("RGBA", (target_h, target_h), (0, 0, 0, 0))

    # Mask out neighbouring-blob ink using the component's own mask
    mask_h, mask_w = component.mask.shape
    ch, cw = crop.shape
    mh = min(mask_h, ch)
    mw = min(mask_w, cw)
    crop[:mh, :mw] = np.where(component.mask[:mh, :mw], crop[:mh, :mw], 0)

    # Resize: inner height = target_h - 2*pad
    inner_h = target_h - 2 * pad
    h, w = crop.shape
    if h == 0 or w == 0:
        return Image.new("RGBA", (target_h, target_h), (0, 0, 0, 0))

    scale = inner_h / h
    new_w = max(1, round(w * scale))
    resized = cv2.resize(crop, (new_w, inner_h), interpolation=cv2.INTER_AREA)

    # Build padded canvas
    canvas_w = new_w + 2 * pad
    canvas   = np.zeros((target_h, canvas_w), dtype=np.uint8)
    canvas[pad : pad + inner_h, pad : pad + new_w] = resized

    # Convert to RGBA: alpha channel = ink intensity
    rgba = np.zeros((target_h, canvas_w, 4), dtype=np.uint8)
    rgba[..., 3] = canvas   # alpha
    # RGB stays 0 (black ink); varying alpha gives natural ink density

    return Image.fromarray(rgba, mode="RGBA")


def save_to_library(
    glyph_img: Image.Image,
    char: str,
    library_path: Path,
) -> Path:
    """Save *glyph_img* to ``glyphs/{char_name}/{char_name}_{N}.png``.

    N is auto-incremented so variants accumulate without overwriting.
    Returns the path of the newly saved file.
    """
    char_name = _safe_char_name(char)
    char_dir  = library_path / char_name
    char_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(char_dir.glob(f"{char_name}_*.png"))
    n        = len(existing)
    out_path = char_dir / f"{char_name}_{n}.png"
    glyph_img.save(out_path)
    return out_path


def build_library(
    source_binary: np.ndarray,
    components: list[LabeledComponent],
    library_path: Path | str,
    min_conf: float = 0.0,
    target_h: int = GLYPH_HEIGHT,
) -> dict[str, list[str]]:
    """Extract every labeled component and save to the glyph library.

    Parameters
    ----------
    source_binary:
        Full-page binarised image (uint8 H×W, 255 = ink).
    components:
        Labeled components from Stage 3.
    library_path:
        Root directory for the glyph library (created if absent).
    min_conf:
        Skip labels with confidence below this value.
    target_h:
        Output glyph image height in pixels.

    Returns
    -------
    Dict mapping ``char_name`` → list of relative file paths saved in *this*
    run (not the full library).
    """
    library_path = Path(library_path)
    library_path.mkdir(parents=True, exist_ok=True)

    # saved values: {"path": relative_path, "orig_h": original_pixel_height}
    saved:   dict[str, list[dict]] = {}
    skipped: int = 0

    for comp in components:
        if comp.char is None or comp.conf < min_conf:
            skipped += 1
            continue

        orig_h    = comp.height   # height in original image pixels
        glyph_img = normalize_glyph(comp, source_binary, target_h)
        out_path  = save_to_library(glyph_img, comp.char, library_path)

        rel = str(out_path.relative_to(library_path))
        saved.setdefault(comp.char, []).append({"path": rel, "orig_h": orig_h})

    _update_index(library_path, saved)

    return saved


def print_library_summary(saved: dict[str, list[dict]]) -> None:
    """Print a human-readable glyph-extraction summary."""
    total = sum(len(v) for v in saved.values())
    if not saved:
        print("[GLYPH] No glyphs saved (check OCR quality / confidence).")
        return
    print(f"[GLYPH] Saved {total} glyph image(s) for {len(saved)} unique character(s).")
    for char, variants in sorted(saved.items()):
        print(f"        {char!r:12s} → {len(variants)} variant(s)")


# ── Internal helpers ──────────────────────────────────────────────────────────

# Map special characters that are illegal in filesystem paths to safe names.
_SPECIAL: dict[str, str] = {
    "/":  "slash",    "\\":  "backslash", ":":  "colon",   "*":  "star",
    "?":  "question", '"':   "dquote",    "<":  "lt",       ">":  "gt",
    "|":  "pipe",     " ":   "space",     ".":  "dot",      ",":  "comma",
    "!":  "exclaim",  "@":  "at",         "#":  "hash",     "$":  "dollar",
    "%":  "percent",  "^":  "caret",      "&":  "amp",      "(":  "lparen",
    ")":  "rparen",   "+":  "plus",       "=":  "equals",   "[":  "lbracket",
    "]":  "rbracket", "{":  "lbrace",     "}":  "rbrace",   ";":  "semi",
    "'":  "squote",   "`":  "backtick",   "~":  "tilde",    "-":  "hyphen",
}


def _safe_char_name(char: str) -> str:
    """Convert a character (possibly multi-char digraph) to a safe folder name.

    Examples
    --------
    >>> _safe_char_name('a')   → 'a'
    >>> _safe_char_name('A')   → 'upper_A'
    >>> _safe_char_name('th')  → 'th'
    >>> _safe_char_name('/')   → 'slash'
    """
    if len(char) > 1:
        # digraph: join safe names with underscore
        return "_".join(_safe_char_name(c) for c in char)
    if char in _SPECIAL:
        return _SPECIAL[char]
    if char.isupper():
        return f"upper_{char}"
    return char


def _update_index(library_path: Path, new_entries: dict[str, list[dict]]) -> None:
    """Merge *new_entries* into the library ``index.json``.

    Each entry in new_entries values is a dict: {"path": str, "orig_h": int}.
    Handles upgrading v1 indexes (plain path strings) to v2 (dicts).
    """
    index_path = library_path / INDEX_FILE
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    else:
        index = {"version": 2, "glyphs": {}}

    index["version"] = 2   # upgrade in-place if loading old v1
    glyphs = index.setdefault("glyphs", {})

    for char, variants in new_entries.items():
        existing = glyphs.setdefault(char, [])
        # Normalise existing entries: upgrade plain strings → dicts
        existing = [
            e if isinstance(e, dict) else {"path": e, "orig_h": None}
            for e in existing
        ]
        existing_paths = {e["path"] for e in existing}
        for v in variants:
            if v["path"] not in existing_paths:
                existing.append(v)
        glyphs[char] = existing

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
