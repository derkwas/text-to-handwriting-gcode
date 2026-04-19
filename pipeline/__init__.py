"""Handwriting synthesis pipeline package.

Stages:
    stage1_input    – load & preprocess PDF / image
    stage2_ocr      – OCR character labels + bounding boxes (Tesseract)
    stage3_segment  – connected-component segmentation + label assignment
    stage4_glyphs   – glyph normalisation & library storage
"""
