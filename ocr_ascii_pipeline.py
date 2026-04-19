"""Generate handwriting_raw/Ascii.ocr.txt from all pages of Ascii.pdf."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.stage1_input import load_image
from pipeline.stage2_ocr import run_ocr
import pypdfium2 as pdfium

pdf_path = Path("handwriting_raw/Ascii.pdf")
n_pages  = len(pdfium.PdfDocument(str(pdf_path)))

all_lines = [
    "=== Ascii.pdf  OCR Output (PSM 6, all pages) ===",
    f"Pages : {n_pages}",
    "",
]

for page in range(n_pages):
    sample = load_image(pdf_path, page=page, dpi=300)
    boxes  = run_ocr(sample.original, psm=6)

    all_lines += [
        f"--- PAGE {page} ({sample.width}x{sample.height} px) ---",
        f"Detected: {len(boxes)} character box(es)",
        "Full sequence:",
        "".join(b.char for b in boxes),
        "",
        f"  {'CHAR':<6} {'LEFT':>5} {'TOP':>5} {'RIGHT':>5} {'BOTTOM':>5}  {'W':>4}  {'H':>4}",
        "  " + "-" * 45,
    ]
    for b in boxes:
        all_lines.append(
            f"  {b.char!r:<6} {b.left:>5} {b.top:>5} {b.right:>5} {b.bottom:>5}"
            f"  {b.width:>4}  {b.height:>4}"
        )
    all_lines.append("")
    print(f"Page {page}: {len(boxes)} chars — {''.join(b.char for b in boxes)[:80]}")

out = Path("handwriting_raw/Ascii.ocr.txt")
out.write_text("\n".join(all_lines), encoding="utf-8")
print(f"\nWritten -> {out}")
