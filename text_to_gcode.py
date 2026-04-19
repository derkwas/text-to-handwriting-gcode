#!/usr/bin/env python3
# text_to_gcode.py
# Main script for text-to-G-code pipeline

import sys
import text_layout
import path_generator
import gcode_generator


def _text_to_paths(layouts):
    """Use handwriting font if available, otherwise fall back to Hershey."""
    font = path_generator.load_handwriting_font()
    if font:
        glyphs = font.get('glyphs', {})
        print(f"Using handwriting font ({len(glyphs)} glyphs)")
        return path_generator.generate_paths_handwriting(layouts, font)
    return path_generator.generate_paths(layouts)


def main():
    if len(sys.argv) < 2:
        print("Usage: python text_to_gcode.py [text|file|draw] <content>")
        sys.exit(1)

    mode = sys.argv[1].lower()
    if mode == 'draw':
        if len(sys.argv) < 3:
            print("Usage for draw: python text_to_gcode.py draw x1,y1;x2,y2;...")
            sys.exit(1)
        coords = sys.argv[2]
        paths = path_generator.generate_draw_paths(coords)
    elif mode == 'file':
        if len(sys.argv) < 3:
            print("Usage for file: python text_to_gcode.py file path/to/file.txt")
            sys.exit(1)
        file_path = sys.argv[2]
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        layouts = text_layout.layout_text(text)
        paths = _text_to_paths(layouts)
    else:
        # everything else treated as text; allow mode keyword 'text' optionally
        text = ' '.join(sys.argv[1:]) if mode != 'text' else (' '.join(sys.argv[2:]) if len(sys.argv) > 2 else '')
        layouts = text_layout.layout_text(text)
        paths = _text_to_paths(layouts)

    # Cap total paths to avoid runaway generation
    max_paths = 10000
    if max_paths > 0 and len(paths) > max_paths:
        print(f"Warning: {len(paths)} paths exceeds limit of {max_paths}, truncating.")
        paths = paths[:max_paths]

    # report path count
    print(f"Generated {len(paths)} path(s)")
    if not paths:
        print("Warning: no drawing paths were produced")
    # Generate G-code
    gcode = gcode_generator.generate_gcode(paths)

    # Output to file
    with open('output.gcode', 'w') as f:
        f.write(gcode)

    print("G-code generated: output.gcode")

if __name__ == '__main__':
    main()