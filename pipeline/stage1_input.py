"""Stage 1 – Input loading and preprocessing.

Accepts a PDF (renders the requested page at *dpi*) or any raster image
(PNG, JPG, TIFF …).  Returns a ``SampleImage`` dataclass containing:

* ``original``  – PIL.Image (RGB)
* ``gray``      – numpy uint8 H×W
* ``binary``    – numpy uint8 H×W, 255 = ink / 0 = paper  (Otsu thresholded)
* ``source``    – Path of the input file
* ``page``      – page index (PDF only, else 0)
* ``dpi``       – effective DPI used for rendering
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import pypdfium2 as pdfium
    _HAS_PDFIUM = True
except ImportError:
    _HAS_PDFIUM = False


# ── Public dataclass ──────────────────────────────────────────────────────────

@dataclasses.dataclass
class SampleImage:
    source:   Path
    page:     int
    dpi:      int
    original: Image.Image   # PIL RGB
    gray:     np.ndarray    # uint8 H×W
    binary:   np.ndarray    # uint8 H×W, 255 = ink

    @property
    def width(self)  -> int: return self.original.width
    @property
    def height(self) -> int: return self.original.height


# ── Public API ────────────────────────────────────────────────────────────────

def load_image(
    path: str | Path,
    page: int = 0,
    dpi: int = 300,
) -> SampleImage:
    """Load *path* (PDF or raster) and return a preprocessed ``SampleImage``.

    For PDFs, *page* selects which page to render (0-indexed).
    For raster images, *dpi* is stored as metadata only (not used for scaling).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        original = _render_pdf_page(path, page=page, dpi=dpi)
    else:
        original = Image.open(path).convert("RGB")
        dpi = 96  # assume screen resolution for plain images

    return _make_sample(original, source=path, page=page, dpi=dpi)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _render_pdf_page(path: Path, page: int, dpi: int) -> Image.Image:
    if not _HAS_PDFIUM:
        raise ImportError(
            "pypdfium2 is required to load PDF files.\n"
            "Install it with:  pip install pypdfium2"
        )
    doc = pdfium.PdfDocument(str(path))
    if page >= len(doc):
        raise ValueError(
            f"PDF has {len(doc)} page(s); requested page index {page}"
        )
    pg    = doc[page]
    scale = dpi / 72.0          # PDF internal unit = 1/72 inch
    bmp   = pg.render(scale=scale)
    return bmp.to_pil().convert("RGB")


def _make_sample(
    image: Image.Image,
    source: Path,
    page: int,
    dpi: int,
) -> SampleImage:
    """Derive gray + binary arrays from a PIL image."""
    gray    = np.array(image.convert("L"))
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu binarisation: ink → 255, paper → 0
    _, binary = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return SampleImage(
        source=source, page=page, dpi=dpi,
        original=image, gray=gray, binary=binary,
    )
