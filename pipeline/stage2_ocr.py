"""Stage 2 – OCR: extract per-character labels and bounding boxes.

Uses Tesseract ``image_to_boxes`` which returns one row per recognised
character with its bounding box in *Tesseract native coordinates* (y = 0 at
the **bottom** of the image).  This module converts everything to standard
**top-left-origin** coordinates before returning, so no downstream stage
needs to know about Tesseract's quirk.

Typical call
------------
>>> from PIL import Image
>>> from pipeline.stage2_ocr import run_ocr
>>> boxes = run_ocr(Image.open("scan.png"), psm=11)
>>> for b in boxes[:5]:
...     print(b.char, b.left, b.top, b.right, b.bottom)
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytesseract
from PIL import Image

# Adjust if Tesseract is installed elsewhere
_TESS_CMD = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = _TESS_CMD

# Characters that Tesseract sometimes emits that are never valid glyphs
_JUNK_CHARS: frozenset[str] = frozenset({"", " ", "\t", "\n", "\r", "\x0c"})


# ── Public dataclass ──────────────────────────────────────────────────────────

@dataclasses.dataclass
class CharBox:
    """A single character recognised by Tesseract with its image bbox.

    All coordinates use **top-left origin** (y increases downward).
    """
    char:   str
    left:   int
    top:    int
    right:  int
    bottom: int
    conf:   float = 0.0   # reserved for future word-level confidence

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def width(self)  -> int:   return self.right - self.left
    @property
    def height(self) -> int:   return self.bottom - self.top
    @property
    def cx(self)     -> float: return (self.left + self.right) / 2
    @property
    def cy(self)     -> float: return (self.top + self.bottom) / 2
    @property
    def area(self)   -> int:   return self.width * self.height

    def __repr__(self) -> str:
        return (
            f"CharBox({self.char!r} "
            f"x={self.left}-{self.right} y={self.top}-{self.bottom})"
        )


# ── Public API ────────────────────────────────────────────────────────────────

def run_ocr(
    image: Image.Image,
    psm: int = 11,
    lang: str = "eng",
    min_size: int = 3,
) -> list[CharBox]:
    """Run Tesseract on *image* and return a list of :class:`CharBox`.

    Parameters
    ----------
    image:
        PIL image (any mode; converted to RGB internally for Tesseract).
    psm:
        Tesseract page segmentation mode.
        11 = sparse text (good for scattered/chart-style handwriting).
         6 = assume a single uniform block of text.
    lang:
        Tesseract language pack.  ``"eng"`` is safe default.
    min_size:
        Discard bboxes narrower or shorter than this many pixels.
    """
    img_rgb = image.convert("RGB")
    h       = img_rgb.height
    config  = f"--oem 1 --psm {psm} -l {lang}"

    raw = pytesseract.image_to_boxes(
        img_rgb,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    results: list[CharBox] = []
    for i, char in enumerate(raw.get("char", [])):
        if char in _JUNK_CHARS:
            continue

        left  = int(raw["left"][i])
        right = int(raw["right"][i])

        # Tesseract y=0 is at the BOTTOM of the image → flip to top-origin
        tess_top    = int(raw["top"][i])
        tess_bottom = int(raw["bottom"][i])
        top    = h - tess_top
        bottom = h - tess_bottom
        if top > bottom:
            top, bottom = bottom, top   # ensure top < bottom

        if (right - left) < min_size or (bottom - top) < min_size:
            continue

        results.append(CharBox(
            char=char,
            left=left, top=top, right=right, bottom=bottom,
        ))

    return results


def print_ocr_summary(boxes: list[CharBox]) -> None:
    """Print a human-readable summary of OCR results to stdout."""
    if not boxes:
        print("[OCR] No characters detected.")
        return
    text = "".join(b.char for b in boxes)
    unique = sorted(set(text))
    print(f"[OCR] Detected {len(boxes)} character box(es).")
    print(f"[OCR] Unique characters ({len(unique)}): {''.join(unique)}")
    # Show first 120 chars of detected text
    preview = text[:120] + ("…" if len(text) > 120 else "")
    print(f"[OCR] Text preview: {preview!r}")
