# text-to-handwriting-gcode

Create a reusable 'font' and render text on a page with scans of your handwriting!

The app extracts characters from scanned PDF using Tesseract OCR and converts it to an SVG glyph. The glyphs are then formatted on a page to mimic handwriting. This page can be exported as PNG, SVG, or COMING SOON G-Code (Penplotters/3D-printers)

## Installation

Requires Python 3.10+ and:

- `pillow`, `numpy`, `scipy`, `scikit-image`, `opencv-python`
- `pymupdf` (imported as `fitz`) and `pytesseract`
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
  installed on your PATH

```bash
pip install pillow numpy scipy scikit-image opencv-python pymupdf pytesseract
```

## Run

```bash
python svg_glyph_editor_app.py
```

On first launch the app creates `~/Desktop/handwriting_libraries/` and
a `default/` folder inside it. Handwriting libraries live on your
Desktop, not in the repo.


## Shortcuts

| Key | Action |
|---|---|
| Arrow keys | Nudge selected glyph (editor tab) |
| F2 | Open library coverage report |


## Notes

- The `.hwfont` format is just a zip of a library folder; you can open
  one with any archive tool if you ever need to inspect it by hand.
- Libraries are intentionally kept on your Desktop (outside the repo)
  so they can stay private even if the code is public.
