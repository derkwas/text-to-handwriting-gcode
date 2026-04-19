#!/usr/bin/env python3
"""Render input.txt as a handwritten preview image using the glyph library.

Usage
-----
    python render_preview.py [options]

Options
-------
    --input FILE     Text file to render (default: input.txt)
    --glyph-preview FILE
                    Output glyph-sheet path (default: glyphs_preview.png)
    --glyph-limit N  Max glyph variants per label to use/show (default: 8)
    --glyphs DIR     Glyph library folder. Can be repeated and/or comma-separated.
                    Default: glyphs_hw,glyphs
    --output FILE    Output image path (default: preview.png)
    --height N       Glyph target height in pixels (default: 80)
    --seed N         Random seed for reproducible output
    --show           Open the preview image after rendering

Examples
--------
    python render_preview.py
    python render_preview.py --input input.txt --height 100 --show
    python render_preview.py --seed 42
    python render_preview.py --glyphs glyphs_hw --glyphs glyphs
    python render_preview.py --glyphs glyphs_hw,glyphs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.stage5_render import render_text, missing_glyphs, DEFAULT_CFG
from pipeline.stage5_render import render_glyph_preview


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render input.txt as handwritten preview",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",  default="input.txt")
    parser.add_argument(
        "--glyphs",
        action="append",
        default=None,
        help="Glyph library folder(s). Repeat flag and/or use comma-separated values.",
    )
    parser.add_argument("--output", default="preview.png")
    parser.add_argument("--glyph-preview", default="glyphs_preview.png")
    parser.add_argument("--glyph-limit", type=int, default=8)
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--seed",   type=int, default=None)
    parser.add_argument("--show",   action="store_true",
                        help="Open preview image after saving")
    args = parser.parse_args(argv)

    input_path  = Path(args.input)
    raw_glyphs = args.glyphs if args.glyphs is not None else ["glyphs_merged"]
    glyph_paths = _parse_glyph_paths(raw_glyphs)
    output_path = Path(args.output)
    glyph_preview_path = Path(args.glyph_preview)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    missing_libs = [p for p in glyph_paths if not p.exists()]
    if missing_libs:
        print("Error: one or more glyph libraries were not found:", file=sys.stderr)
        for p in missing_libs:
            print(f"  - {p}", file=sys.stderr)
        print(
            "Run: python run_pipeline.py handwriting_raw/handwriting.pdf --glyphs glyphs_hw",
            file=sys.stderr,
        )
        sys.exit(1)

    text = input_path.read_text(encoding="utf-8").rstrip()
    print(f"Input text ({len(text)} chars):")
    for line in text.splitlines():
        print(f"  {line!r}")

    # Check for missing glyphs
    missing = missing_glyphs(text, glyph_paths)
    if missing:
        print(f"\nWarning: {len(missing)} character(s) have no glyph — will be skipped:")
        print(f"  {missing}")
    else:
        print("\nAll characters covered by the glyph library.")

    glyph_limit = max(1, args.glyph_limit)
    cfg = {
        **DEFAULT_CFG,
        "glyph_height": args.height,
        "max_variants_per_label": glyph_limit,
    }

    print("Using glyph libraries:")
    for p in glyph_paths:
        print(f"  - {p}")

    print(f"\nRendering at glyph height={args.height}px ...")
    img = render_text(text, glyph_paths, cfg=cfg, seed=args.seed)

    img.save(output_path)
    print(f"Saved -> {output_path}  ({img.width}×{img.height} px)")

    if args.show:
        img.show()

    out_sheet = render_glyph_preview(
        glyph_paths,
        glyph_preview_path,
        max_per_label=glyph_limit,
    )
    print(f"Saved -> {out_sheet}  (glyph sheet, max {glyph_limit} per label)")

def _parse_glyph_paths(raw_values: list[str]) -> list[Path]:
    """Parse repeated/comma-separated --glyphs values into unique paths."""
    out: list[Path] = []
    seen: set[str] = set()
    for raw in raw_values:
        for part in raw.split(","):
            p = Path(part.strip())
            if not part.strip():
                continue
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return out


if __name__ == "__main__":
    main()
