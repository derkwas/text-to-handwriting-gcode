# text_layout.py
# Module for laying out text on the page

import pyhershey as hershey
import config


def _measure_width(text, cfg):
    """Return the approximate rendered mm-width of *text*."""
    if not text:
        return 0.0
    scale = cfg['font_size'] / 21.6
    lsf = cfg['letter_spacing_factor']
    max_x = 0.0
    for item in hershey.shape_text(text, cfg['font_name'], 22):
        adv = item['pos'][0] * lsf
        max_x = max(max_x, adv)
        for seg in item['glyph'].segments:
            for px, _ in seg:
                max_x = max(max_x, adv + px * scale)
    return max_x


def layout_text(text):
    """
    Layout the text on the page with word-wrapping.
    Tabs are expanded to 4-space stops. Leading spaces are preserved as
    indentation (pyhershey advances over spaces naturally).
    Returns a list of (line_text, x, y) tuples.
    """
    cfg = config.CONFIG
    margin_left = cfg['margin_left']
    margin_top = cfg['margin_top']
    line_spacing = cfg['font_size'] * cfg['line_spacing']
    available_width = cfg['page_width'] - margin_left - cfg['margin_right']

    wrapped = []
    for paragraph in text.split('\n'):
        # expand tabs to 4-space stops
        paragraph = paragraph.expandtabs(4)

        stripped = paragraph.strip()
        if not stripped:
            wrapped.append('')
            continue

        # preserve leading whitespace as indent prefix
        indent = paragraph[: len(paragraph) - len(paragraph.lstrip(' '))]

        cur_words = []
        for word in stripped.split():
            candidate = indent + ' '.join(cur_words + [word])
            if _measure_width(candidate, cfg) <= available_width:
                cur_words.append(word)
            else:
                if cur_words:
                    wrapped.append(indent + ' '.join(cur_words))
                    # continuation lines keep the same indent
                    cur_words = [word]
                else:
                    # single word wider than page — place it anyway
                    wrapped.append(indent + word)
        if cur_words:
            wrapped.append(indent + ' '.join(cur_words))

    margin_bottom = cfg['margin_bottom']
    layouts = []
    # baseline starts font_size below margin_top so cap tops land exactly at margin_top
    y = margin_top - cfg['font_size']
    for line in wrapped:
        if y < margin_bottom:
            break
        layouts.append((line, margin_left, y))
        y -= line_spacing
    return layouts
