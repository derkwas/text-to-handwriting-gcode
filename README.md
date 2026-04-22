# text-to-handwriting-gcode

Create a reusable 'font' and render text on a page with scans of your handwriting - exportable as PNG, SVG, and G-Code (Penplotters/3D-printers)

## 📖 Overview
This project extracts characters from scanned PDF using Tesseract OCR and converts it into a reusable SVG glyph library. The glyphs are then formatted on a page with naturalness features to mimic handwriting.

## 🎬 Demo

coming soon

- video
- images

## 📦 Installation

Requires Python 3.10+ and:

- `pillow`, `numpy`, `scipy`, `scikit-image`, `opencv-python`
- `pymupdf` (imported as `fitz`) and `pytesseract`
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
  installed on your PATH

```bash
pip install pillow numpy scipy scikit-image opencv-python pymupdf pytesseract
```

## ✏️ Run

```bash
python svg_glyph_editor_app.py
```

> On first launch the app creates `~/Desktop/handwriting_libraries/` and a `default/` folder inside it. Handwriting libraries live on your Desktop, not in the repo.


## ⌨️ Shortcuts

| Key | Action |
|---|---|
| Arrow keys | Nudge selected glyph (editor tab) |
| Ctrl + R | Render the compose page |
| F2 | Open library coverage report |

## ⚠️ Ethical Use Disclaimer ⚠️

This project is intended for **creative, educational, and research purposes**.

It should **not** be used for any malicious intent. This includes impersonation, forgery, and claiming generated handwriting as authentic human work.