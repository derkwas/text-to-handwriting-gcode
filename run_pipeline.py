#!/usr/bin/env python3
"""CLI driver for the handwriting synthesis pipeline (Stages 1–4).

Usage
-----
    python run_pipeline.py <input_file> [options]

Examples
--------
    # Run all 4 stages on the first page of a PDF
    python run_pipeline.py handwriting_raw/Ascii.pdf

    # Use PSM 6 (block text) instead of the default PSM 11 (sparse)
    python run_pipeline.py scan.jpg --psm 6

    # Only process page 2 of a multi-page PDF
    python run_pipeline.py notes.pdf --page 2

    # Save debug images (binary, segmentation overlay, glyph crops)
    python run_pipeline.py scan.jpg --debug

    # Override the glyph library output folder
    python run_pipeline.py scan.jpg --glyphs my_glyphs/

Options
-------
    --page N        PDF page index to render (default: 0)
    --dpi N         Render PDF at this DPI (default: 300)
    --psm N         Tesseract page-segmentation mode (default: 11)
    --glyphs DIR    Glyph library output directory (default: ./glyphs)
    --min-area N    Minimum blob area in pixels to keep (default: 80)
    --min-conf F    Minimum label confidence to save a glyph (default: 0.0)
    --debug         Save diagnostic images to debug/ folder
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Pipeline imports ──────────────────────────────────────────────────────────
from pipeline.stage1_input  import load_image
from pipeline.stage2_ocr    import run_ocr, print_ocr_summary
from pipeline.stage3_segment import find_components, assign_labels, print_segment_summary
from pipeline.stage4_glyphs import build_library, print_library_summary


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_debug_images(
    sample,
    char_boxes,
    labeled_components,
    debug_dir: Path,
) -> None:
    """Write diagnostic PNG files to *debug_dir*."""
    import cv2
    import numpy as np
    from PIL import Image

    debug_dir.mkdir(parents=True, exist_ok=True)

    # 1. Binary image
    bin_path = debug_dir / "1_binary.png"
    Image.fromarray(sample.binary).save(bin_path)
    print(f"[DEBUG] Binary image → {bin_path}")

    # 2. OCR bbox overlay on the original
    ocr_img = np.array(sample.original.copy())
    for cb in char_boxes:
        cv2.rectangle(ocr_img, (cb.left, cb.top), (cb.right, cb.bottom),
                      (0, 120, 255), 1)
        cv2.putText(ocr_img, cb.char, (cb.left, max(0, cb.top - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 255), 1,
                    cv2.LINE_AA)
    ocr_path = debug_dir / "2_ocr_boxes.png"
    Image.fromarray(ocr_img).save(ocr_path)
    print(f"[DEBUG] OCR overlay  → {ocr_path}")

    # 3. Segmentation overlay (green = labeled, red = unlabeled)
    seg_img = np.array(sample.original.copy())
    for lc in labeled_components:
        colour = (0, 200, 60) if lc.char is not None else (220, 40, 40)
        cv2.rectangle(seg_img, (lc.left, lc.top), (lc.right, lc.bottom),
                      colour, 1)
        if lc.char is not None:
            cv2.putText(seg_img, lc.char, (lc.left, max(0, lc.top - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1,
                        cv2.LINE_AA)
    seg_path = debug_dir / "3_segmentation.png"
    Image.fromarray(seg_img).save(seg_path)
    print(f"[DEBUG] Seg overlay  → {seg_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Handwriting synthesis pipeline – Stages 1–4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input",        help="Path to PDF or image file")
    parser.add_argument("--page",       type=int,   default=0,
                        help="PDF page index (default: 0)")
    parser.add_argument("--dpi",        type=int,   default=300,
                        help="Render DPI for PDF (default: 300)")
    parser.add_argument("--psm",        type=int,   default=11,
                        help="Tesseract PSM (default: 11 = sparse text)")
    parser.add_argument("--glyphs",     default="glyphs",
                        help="Glyph library output directory (default: glyphs/)")
    parser.add_argument("--min-area",   type=int,   default=80,
                        help="Min blob area in pixels (default: 80)")
    parser.add_argument("--min-conf",   type=float, default=0.0,
                        help="Min label confidence to save glyph (default: 0.0)")
    parser.add_argument("--debug",      action="store_true",
                        help="Save diagnostic images to debug/")
    args = parser.parse_args(argv)

    input_path   = Path(args.input)
    glyphs_path  = Path(args.glyphs)
    debug_dir    = Path("debug")

    # ── Stage 1: Input ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STAGE 1 – Input: {input_path.name}")
    print(f"{'='*60}")
    sample = load_image(input_path, page=args.page, dpi=args.dpi)
    print(f"[INPUT] Loaded: {sample.width}×{sample.height} px  "
          f"(page={sample.page}, dpi={sample.dpi})")

    # ── Stage 2: OCR ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STAGE 2 – OCR  (psm={args.psm})")
    print(f"{'='*60}")
    char_boxes = run_ocr(sample.original, psm=args.psm)
    print_ocr_summary(char_boxes)

    if not char_boxes:
        print("\n[WARN] OCR returned no characters.")
        print("       Try a different --psm value or a cleaner input image.")
        print("       Segmentation will still run but all labels will be None.")

    # ── Stage 3: Segmentation ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STAGE 3 – Segmentation  (min_area={args.min_area})")
    print(f"{'='*60}")
    components = find_components(sample.binary, min_area=args.min_area)
    print(f"[SEG] {len(components)} connected component(s) found after noise filter.")
    labeled = assign_labels(components, char_boxes)
    print_segment_summary(labeled)

    # ── Stage 4: Glyph Library ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STAGE 4 – Glyph Library  → {glyphs_path}/")
    print(f"{'='*60}")
    saved = build_library(
        source_binary=sample.binary,
        components=labeled,
        library_path=glyphs_path,
        min_conf=args.min_conf,
    )
    print_library_summary(saved)

    # ── Optional debug output ─────────────────────────────────────────────────
    if args.debug:
        print(f"\n{'='*60}")
        print("  DEBUG images")
        print(f"{'='*60}")
        _save_debug_images(sample, char_boxes, labeled, debug_dir)

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
