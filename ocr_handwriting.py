"""Generate handwriting_raw/handwriting.ocr.txt from handwriting.pdf."""
import sys
from pathlib import Path

# Run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.stage1_input import load_image
from pipeline.stage2_ocr import run_ocr

sample = load_image("handwriting_raw/handwriting.pdf", page=0, dpi=300)
boxes  = run_ocr(sample.original, psm=6)

lines = [
    "=== handwriting.pdf  OCR Output (PSM 6) ===",
    f"Image size : {sample.width}x{sample.height} px  |  DPI: {sample.dpi}",
    f"Characters : {len(boxes)} boxes detected",
    "",
    "--- Full character sequence ---",
    "".join(b.char for b in boxes),
    "",
    "--- Per-character detail (top-left-origin coords) ---",
    f"  {'CHAR':<6} {'LEFT':>5} {'TOP':>5} {'RIGHT':>5} {'BOTTOM':>5}  {'W':>4}  {'H':>4}",
    "  " + "-" * 45,
]

for b in boxes:
    lines.append(
        f"  {b.char!r:<6} {b.left:>5} {b.top:>5} {b.right:>5} {b.bottom:>5}"
        f"  {b.width:>4}  {b.height:>4}"
    )

out = Path("handwriting_raw/handwriting.ocr.txt")
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Written -> {out}")
print()
print("Detected text:")
print("".join(b.char for b in boxes))
