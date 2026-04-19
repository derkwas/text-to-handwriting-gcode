#!/usr/bin/env python3
"""Single-pass OCR for handwriting_raw/Ascii.pdf.

Behavior:
- Deletes old Ascii*.txt outputs before each run.
- Runs one deterministic TrOCR handwriting pipeline (no tuning loop).
- Writes exactly one output: handwriting_raw/Ascii.ocr.txt
"""

from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import torch
from PIL import Image, ImageFilter, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


MODEL_NAME = "microsoft/trocr-base-handwritten"


def segment_text_lines(cleaned_bw: np.ndarray) -> list[np.ndarray]:
    """Split a page into text-line crops using horizontal projection."""
    ink = 255 - cleaned_bw
    row_activity = ink.sum(axis=1)
    threshold = max(ink.shape[1] * 8, int(row_activity.max() * 0.04))

    lines = []
    start = None
    for y, value in enumerate(row_activity):
        if value > threshold and start is None:
            start = y
        elif value <= threshold and start is not None:
            if y - start >= 10:
                lines.append((start, y))
            start = None
    if start is not None:
        lines.append((start, len(row_activity)))

    crops = []
    for y0, y1 in lines:
        y0 = max(0, y0 - 4)
        y1 = min(cleaned_bw.shape[0], y1 + 4)
        line = cleaned_bw[y0:y1, :]

        # Trim empty side margins so TrOCR focuses on characters.
        col_activity = (255 - line).sum(axis=0)
        cols = np.where(col_activity > 0)[0]
        if cols.size == 0:
            continue
        x0 = max(0, int(cols[0]) - 4)
        x1 = min(line.shape[1], int(cols[-1]) + 5)
        line = line[:, x0:x1]
        crops.append(line)

    return crops


def load_trocr():
    """Load processor/model once for all page decodes."""
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model, device


def decode_line(line_bw: np.ndarray, processor, model, device: str) -> str:
    """Decode one line image into text using TrOCR."""
    pil_line = Image.fromarray(line_bw).convert("RGB")
    pixel_values = processor(images=pil_line, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=48,
            num_beams=1,
            do_sample=False,
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    source_pdf = repo_dir / "handwriting_raw" / "Ascii.pdf"
    output_dir = source_pdf.parent
    output_txt = output_dir / "Ascii.ocr.txt"

    if not source_pdf.exists():
        raise FileNotFoundError(f"Missing input PDF: {source_pdf}")

    # Keep output predictable by removing all previous test outputs each run.
    for path in output_dir.glob("Ascii*.txt"):
        path.unlink(missing_ok=True)

    processor, model, device = load_trocr()

    doc = pdfium.PdfDocument(str(source_pdf))
    pages = []
    for idx in range(len(doc)):
        page = doc[idx]
        image = page.render(scale=5).to_pil().convert("L")
        image = ImageOps.autocontrast(image)
        image = image.filter(ImageFilter.MedianFilter(size=3))
        image = image.point(lambda p: 255 if p > 170 else 0)
        cleaned = np.array(image)

        lines = segment_text_lines(cleaned)
        decoded_lines = []
        for line in lines:
            text = decode_line(line, processor, model, device)
            if text:
                decoded_lines.append(text)

        pages.append(f"--- Page {idx + 1} ---\n" + "\n".join(decoded_lines).strip() + "\n")

    output_txt.write_text("\n".join(pages).strip() + "\n", encoding="utf-8")
    print(f"OCR complete. Pages: {len(doc)}")
    print(f"Output: {output_txt}")


if __name__ == "__main__":
    main()
