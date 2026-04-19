"""Generate vector paths from text and simple coordinate input."""

import json
from pathlib import Path

import pyhershey as hershey
import config

_HANDWRITING_FONT_PATH = Path(__file__).resolve().parent / "handwriting_font.json"


def load_handwriting_font(path: Path | None = None) -> dict | None:
    """Return the handwriting font dict if handwriting_font.json exists, else None."""
    p = path or _HANDWRITING_FONT_PATH
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def generate_paths_handwriting(layouts, font_data: dict) -> list:
    """Generate plotter paths using a learned handwriting font.

    Paths are taken directly from the font JSON (normalized to 21-unit cap
    height) and scaled to the font_size configured in config.py.  Unknown
    characters are silently skipped.
    """
    cfg = config.CONFIG
    font_size = cfg['font_size']
    cap_height = font_data.get('cap_height', 21)
    scale = font_size / cap_height
    letter_gap = cfg.get('letter_spacing_factor', 0.2) * font_size
    space_width = cap_height * 0.4 * scale
    glyphs = font_data.get('glyphs', {})

    all_paths = []
    for line_text, base_x, base_y in layouts:
        cursor_x = 0.0
        for char in line_text:
            if char == ' ':
                cursor_x += space_width + letter_gap
                continue
            if char not in glyphs:
                continue
            glyph = glyphs[char]
            for path in glyph['paths']:
                if len(path) < 2:
                    continue
                translated = [
                    (base_x + cursor_x + pt[0] * scale, base_y + pt[1] * scale)
                    for pt in path
                ]
                all_paths.append(translated)
            cursor_x += glyph['advance'] * scale + letter_gap
    return all_paths

def generate_paths(layouts):
    """
    Generate vector paths for the laid out text.
    Returns a list of paths, where each path is a list of (x, y) points.
    """
    cfg = config.CONFIG
    font_name = cfg['font_name']
    font_size = cfg['font_size']
    letter_spacing_factor = cfg['letter_spacing_factor']

    scale = font_size / 21  # Hershey roman_simplex cap height = 21 units

    all_paths = []
    for line_text, base_x, base_y in layouts:
        items = hershey.shape_text(line_text, font_name, 22)
        for item in items:
            glyph = item['glyph']
            pos = item['pos']
            for segment in glyph.segments:
                scaled_path = []
                for point in segment:
                    x, y = point
                    scaled_x = base_x + pos[0] * letter_spacing_factor + x * scale
                    scaled_y = base_y - pos[1] + y * scale
                    scaled_path.append((scaled_x, scaled_y))
                all_paths.append(scaled_path)

    return all_paths


def generate_draw_paths(coord_string):
    """Convert a semicolon-separated list of x,y points to a single path.

    Example coord_string: "0,0;10,0;10,10".
    """
    paths = []
    segment = []
    for part in coord_string.split(';'):
        part = part.strip()
        if not part:
            continue
        try:
            x_str, y_str = part.split(',')
            segment.append((float(x_str), float(y_str)))
        except ValueError:
            # skip invalid entries
            continue
    if segment:
        paths.append(segment)
    return paths


