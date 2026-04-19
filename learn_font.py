#!/usr/bin/env python3
"""Learn a handwriting font from a scan or photo of natural handwriting.

Usage:
    # Let Tesseract read the text automatically
    python learn_font.py handwriting.png

    # Provide the exact text written in the image for perfect labelling
    python learn_font.py handwriting.png --text "The quick brown fox jumps over the lazy dog"

    # Write to a specific font file
    python learn_font.py handwriting.png --output my_font.json

How it works:
  1. Character bounding boxes are detected via Tesseract.
  2. Each character's ink region is skeletonized (thinned to 1-px strokes).
  3. The skeleton is traced into ordered pen-lift polylines.
  4. Paths are normalized to a 21-unit cap-height coordinate space
     (the same scale as Hershey fonts used by the plotter).
  5. The font JSON is created or updated — run on multiple images to
     accumulate coverage for all characters you need.

The font is used automatically by text_to_gcode.py when
handwriting_font.json exists in the same directory.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image
from skimage.morphology import skeletonize as sk_skeletonize

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

FONT_PATH = Path(__file__).resolve().parent / "handwriting_font.json"
CAP_HEIGHT = 21  # normalized coordinate units, matches Hershey fonts


# ── Font I/O ──────────────────────────────────────────────────────────────────

def load_font(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"version": 1, "cap_height": CAP_HEIGHT, "glyphs": {}}


def save_font(font: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(font, f, indent=2, ensure_ascii=False)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def binarize(image: Image.Image) -> np.ndarray:
    """Return uint8 array: 255 = ink, 0 = paper."""
    arr = np.array(image.convert("L"))
    arr = cv2.GaussianBlur(arr, (3, 3), 0)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


# ── Character detection ───────────────────────────────────────────────────────

def detect_chars(
    image: Image.Image,
    ground_truth: str | None,
) -> list[tuple[str, int, int, int, int]]:
    """Return (char, img_left, img_top, img_right, img_bottom) per character.

    Tesseract image_to_boxes uses y=0 at the *bottom* of the image; this
    function converts to standard top-left origin before returning.
    """
    h = image.height
    raw = pytesseract.image_to_boxes(
        image,
        config="--oem 1 --psm 6 -l eng",
        output_type=pytesseract.Output.DICT,
    )

    # Ground truth: strip spaces/newlines to get a flat character sequence
    gt_chars = (
        list(ground_truth.replace(" ", "").replace("\n", ""))
        if ground_truth
        else None
    )
    gt_idx = 0

    results = []
    for i, char in enumerate(raw.get("char", [])):
        if not char.strip():
            continue

        # Convert from Tesseract's bottom-origin coords to image (top-origin)
        left       = raw["left"][i]
        right      = raw["right"][i]
        img_top    = h - raw["top"][i]
        img_bottom = h - raw["bottom"][i]

        if (right - left) < 3 or (img_bottom - img_top) < 3:
            continue

        if gt_chars and gt_idx < len(gt_chars):
            char = gt_chars[gt_idx]
        gt_idx += 1

        results.append((char, left, img_top, right, img_bottom))

    return results


# ── Skeleton tracing ──────────────────────────────────────────────────────────

def trace_skeleton(skel: np.ndarray) -> list[list[tuple[int, int]]]:
    """Trace a boolean skeleton into ordered polylines.

    Uses greedy forward-chaining that:
    - starts from endpoints (degree-1 pixels) to produce natural stroke order
    - minimises turn angle to keep lines smooth
    - falls back to any unvisited pixel when no endpoints are left
      (handles closed loops such as 'O', '0', 'D')
    """
    ys, xs = np.where(skel)
    if xs.size == 0:
        return []

    pixels: set[tuple[int, int]] = set(zip(xs.tolist(), ys.tolist()))

    def nbrs(x: int, y: int) -> list[tuple[int, int]]:
        return [
            (x + dx, y + dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if (dx, dy) != (0, 0) and (x + dx, y + dy) in pixels
        ]

    degree = {p: len(nbrs(*p)) for p in pixels}
    visited: set[tuple[int, int]] = set()
    paths: list[list[tuple[int, int]]] = []

    def next_start() -> tuple[int, int] | None:
        for p, d in degree.items():
            if p not in visited and d == 1:
                return p
        for p in pixels:
            if p not in visited:
                return p
        return None

    start = next_start()
    while start is not None:
        path: list[tuple[int, int]] = [start]
        visited.add(start)

        while True:
            cur = path[-1]
            prev = path[-2] if len(path) >= 2 else None
            free = [n for n in nbrs(*cur) if n not in visited]
            if not free:
                break
            if prev is None:
                nxt = free[0]
            else:
                # Prefer the neighbor that minimises the turn angle
                dx, dy = cur[0] - prev[0], cur[1] - prev[1]
                nxt = min(
                    free,
                    key=lambda n: abs((n[0] - cur[0]) * dy - (n[1] - cur[1]) * dx),
                )
            visited.add(nxt)
            path.append(nxt)

        if len(path) >= 2:
            paths.append(path)

        start = next_start()

    return paths


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize(
    paths: list[list[tuple[int, int]]],
    crop_h: int,
    crop_w: int,
) -> tuple[list[list[list[float]]], float]:
    """Scale paths to CAP_HEIGHT units with baseline at y=0.

    Image y increases *downward*; we flip so y increases *upward* (same
    convention as Hershey fonts), with y=0 at the character baseline/bottom.

    Returns (normalized_paths, advance_width).
    """
    if not paths:
        return [], round(CAP_HEIGHT * 0.55, 3)

    scale = CAP_HEIGHT / crop_h if crop_h > 0 else 1.0
    all_pts = [pt for p in paths for pt in p]
    min_x = min(pt[0] for pt in all_pts)
    max_x = max(pt[0] for pt in all_pts)

    normed = [
        [
            [round((pt[0] - min_x) * scale, 3),
             round((crop_h - pt[1]) * scale, 3)]   # flip y
            for pt in path
        ]
        for path in paths
    ]

    advance = round((max_x - min_x) * scale * 1.1, 3)  # 10 % right padding
    return normed, advance


# ── Learning ──────────────────────────────────────────────────────────────────

def learn_from_image(
    image_path: Path,
    font_path: Path,
    ground_truth: str | None = None,
) -> dict:
    """Extract glyphs from *image_path* and merge them into *font_path*."""
    image = Image.open(image_path).convert("RGB")
    ink = binarize(image)
    print(f"Image: {image_path.name}  ({image.width}×{image.height}px)")

    chars = detect_chars(image, ground_truth)
    print(f"Detected {len(chars)} character region(s)")

    font = load_font(font_path)
    glyphs = font["glyphs"]
    new_chars: list[str] = []
    skipped = 0

    for char, left, img_top, right, img_bottom in chars:
        crop_h = img_bottom - img_top
        crop_w = right - left
        if crop_h < 4 or crop_w < 4:
            skipped += 1
            continue

        pad = max(2, crop_h // 8)
        H, W = ink.shape
        y0 = max(0, img_top - pad)
        y1 = min(H, img_bottom + pad)
        x0 = max(0, left - pad)
        x1 = min(W, right + pad)
        crop = ink[y0:y1, x0:x1]

        if crop.size == 0:
            skipped += 1
            continue

        skel = sk_skeletonize(crop > 0)
        raw_paths = trace_skeleton(skel)
        if not raw_paths:
            skipped += 1
            continue

        normed, advance = normalize(raw_paths, crop_h, crop_w)
        if char not in glyphs:
            new_chars.append(char)
        glyphs[char] = {"paths": normed, "advance": advance}

    save_font(font, font_path)

    print(f"New glyphs learned : {sorted(set(new_chars)) or '(none — updated existing)'}")
    if skipped:
        print(f"Skipped            : {skipped} region(s) (too small / empty skeleton)")
    print(f"Total glyphs       : {len(glyphs)}")
    print(f"Font written to    : {font_path}")
    return font


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learn a handwriting font from a handwriting image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("image", help="Handwriting image (PNG, JPG, etc.)")
    parser.add_argument(
        "--text", "-t", default=None,
        help="Exact text in the image — overrides OCR for character labelling",
    )
    parser.add_argument(
        "--output", "-o", default=str(FONT_PATH),
        help=f"Font JSON to write/update (default: {FONT_PATH.name})",
    )
    args = parser.parse_args()

    img = Path(args.image)
    if not img.exists():
        print(f"Error: image not found: {img}", file=sys.stderr)
        sys.exit(1)

    learn_from_image(img, Path(args.output), ground_truth=args.text)


if __name__ == "__main__":
    main()
