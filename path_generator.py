"""Generate vector paths from text and simple coordinate input."""

import pyhershey as hershey
import config

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


