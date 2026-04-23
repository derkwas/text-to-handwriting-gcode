#!/usr/bin/env python3
"""Interactive SVG-glyph editor (parallel to glyph_editor_app.py).

Targets the vector glyph library produced by pdf_to_vector_glyphs.py —
each glyph is an SVG file of cubic-Bezier subpaths in PDF-point space.

Features
--------
- Load a glyph library (default: ~/Desktop/handwriting_libraries/)
- Scrollable grid of thumbnails, click to edit
- Drag to translate, arrow keys nudge, scale slider
- Font-guide overlay (reference letter behind the glyph)
- Auto-fit glyph height to the guide
- Save applies the current transform and updates index.json
- Relabel / Delete / Next

The raster-era bits from glyph_editor_app.py (skeletonize / vectorize /
profile overlays / vector-debug panel) are intentionally absent — the SVG
paths are the authoritative geometry, there's nothing to reconstruct.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from svg_glyph import (
    Glyph,
    centerlines_to_svg,
    compute_centerlines,
    rasterize_glyph,
)


# ── Library model ───────────────────────────────────────────────────────────

@dataclass
class GlyphVariant:
    label: str
    lib_path: Path
    idx_path: Path
    rel_path: str
    abs_path: Path
    orig_h: float | None
    orig_w: float | None
    # Baseline y in glyph-local coordinates (0 at top of glyph bbox). For
    # non-descender chars ≈ orig_h (baseline at bottom of ink); for
    # descenders (g, y, $, …) < orig_h so that ink below baseline renders
    # below the line's baseline. None when unknown (older libraries).
    baseline: float | None = None
    # User-editable display state, persisted verbatim — no scale baking,
    # no offset absorption. Drag / slider modify these directly, Save
    # writes them to index.json, Reload restores them. Numbers you see
    # after save equal what you set; geometry on disk is untouched by
    # user edits (only normalize changes geometry).
    offset_x: float = 0.0
    offset_y: float = 0.0
    user_scale: float = 1.0
    # Advance width (in glyph-local pt) at which the cursor should move
    # to position the NEXT glyph, distinct from the ink bounding box.
    # For characters with trailing tails (q, j, f...) this is smaller
    # than orig_w so the next letter sits against the body, not past
    # the tail. Auto-computed by normalize_library; None for old
    # libraries (layout falls back to orig_w in that case).
    advance_x: float | None = None
    # User override multiplier applied on top of advance_x — tighten or
    # loosen any individual variant that the heuristic got wrong.
    advance_mul: float = 1.0


def safe_label_dir(label: str) -> str:
    if label == "":
        return "empty"
    if len(label) == 1 and label.isalpha() and label.islower():
        return label
    if len(label) == 1 and label.isalpha() and label.isupper():
        return f"upper_{label}"
    named = {
        " ": "space", "/": "slash", "\\": "backslash", ":": "colon",
        "*": "asterisk", "?": "question", '"': "quote", "<": "lt",
        ">": "gt", "|": "pipe",
    }
    if label in named:
        return named[label]
    out: list[str] = []
    for ch in label:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append(f"u{ord(ch):04x}")
    s = "".join(out).strip(".")
    return s or "glyph"


def libraries_root() -> Path:
    """Where user handwriting libraries live — on the Desktop, outside
    the code repo so the libraries can be kept private / excluded from
    version control."""
    return Path.home() / "Desktop" / "handwriting_libraries"


def parse_library_paths(raw_values: list[str]) -> list[Path]:
    # Relative paths resolve against the libraries root (on the
    # Desktop), not the code repo. Absolute paths pass through as-is so
    # you can still point at arbitrary folders if you want.
    root = libraries_root()
    out: list[Path] = []
    seen: set[str] = set()
    for raw in raw_values:
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            p = Path(part)
            if not p.is_absolute():
                p = (root / p).resolve()
            k = str(p)
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
    return out


def load_variants(library_paths: list[Path]) -> list[GlyphVariant]:
    from svg_glyph import ADVANCE_ALGO_VERSION as _CUR_ADVANCE_ALGO
    variants: list[GlyphVariant] = []
    for lib in library_paths:
        idx_path = lib / "index.json"
        if not idx_path.exists():
            continue
        with open(idx_path, encoding="utf-8") as f:
            index = json.load(f)
        if index.get("format") and index.get("format") != "svg":
            # Skip legacy raster indexes. User should use glyph_editor_app.py.
            continue
        # Stored advance_x values are only trustworthy if they came from
        # the current algorithm version. Older or missing version
        # markers mean the numbers might predate a heuristic fix, so
        # we ignore them and let layout lazy-recompute.
        stored_algo = int(index.get("advance_algo_version") or 0)
        advances_fresh = stored_algo >= _CUR_ADVANCE_ALGO
        glyphs = index.get("glyphs", {})
        for label, entries in glyphs.items():
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                rel = str(entry.get("path", ""))
                if not rel:
                    continue
                adv = entry.get("advance_x") if advances_fresh else None
                variants.append(GlyphVariant(
                    label=label,
                    lib_path=lib,
                    idx_path=idx_path,
                    rel_path=rel,
                    abs_path=lib / rel,
                    orig_h=entry.get("orig_h"),
                    orig_w=entry.get("orig_w"),
                    baseline=entry.get("baseline"),
                    offset_x=float(entry.get("offset_x", 0.0)),
                    offset_y=float(entry.get("offset_y", 0.0)),
                    user_scale=float(entry.get("user_scale", 1.0)),
                    advance_x=float(adv) if adv is not None else None,
                    advance_mul=float(entry.get("advance_mul", 1.0)),
                ))
    variants.sort(key=lambda v: (v.label, str(v.lib_path), v.rel_path))
    return variants


def is_single_lower_ascii(label: str) -> bool:
    return len(label) == 1 and "a" <= label <= "z"


# Common ASCII characters that the coverage report surveys. Grouped for
# display; space isn't listed since it's handled as an advance, not a
# glyph.
COMMON_CHARS: tuple[str, ...] = tuple(
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abcdefghijklmnopqrstuvwxyz")
    + list("0123456789")
    + list(".,;:!?'\"`")
    + list("()[]{}")
    + list("-_/\\|")
    + list("@#$%^&*+=~<>")
)


# ── Editor app ──────────────────────────────────────────────────────────────

class SvgGlyphEditorApp:
    def __init__(self, root: tk.Tk, library_paths: list[Path]) -> None:
        self.root = root
        self.root.title("SVG Glyph Editor")
        self.root.geometry("1300x860")

        self.library_paths = library_paths
        self.variants: list[GlyphVariant] = []
        self.filtered_indices: list[int] = []

        self.cell_w = 120
        self.cell_h = 130
        self.thumb_box = 90
        self.grid_cols = 8

        self.selected_global_idx: int | None = None

        # Thumbnail cache: keyed by (abs_path, mtime_ns) so edits auto-
        # invalidate. Without this the grid re-rasterizes every SVG on
        # every click/reload/filter — fine for a few glyphs, brutal for
        # hundreds.
        self._thumb_cache: dict[tuple[str, int], ImageTk.PhotoImage] = {}
        self._thumb_checker_bg: "Image.Image | None" = None
        # Canvas id of the currently-drawn selection highlight rectangle,
        # so selection changes can move/delete it without re-rasterizing
        # the whole grid.
        self._selection_canvas_id: int | None = None

        self.editor_glyph: Glyph | None = None
        self.editor_centerlines: list[dict] = []  # base-glyph coord space
        # Typographic baseline y inside the glyph's local frame. Read-only
        # metadata from normalize, displayed by the editor for reference
        # but not mutated by the user (vertical drag affects offset_y
        # instead, which is purely positional).
        self.editor_baseline: float | None = None
        # User-editable positional state. These are the single source of
        # truth for the glyph's on-screen position and user-visible size:
        # drag modifies them, slider modifies glyph_user_scale. Save just
        # writes them to index.json; Load restores them. Nothing is baked
        # into SVG geometry, nothing is reset on save.
        self.editor_offset_x = 0.0   # horizontal shift (pt)
        self.editor_offset_y = 0.0   # vertical shift (pt)
        self.glyph_user_scale = 1.0  # size multiplier
        self.editor_scale = 1.0      # derived: glyph-pt -> canvas px
        self.drag_last_x = 0
        self.drag_last_y = 0
        self._suppress_scale_callback = False
        self._suppress_advance_callback = False
        self.last_guide_bbox: tuple[int, int, int, int] | None = None
        self.last_baseline_y = 0

        self.ref_orig_h = 8.0
        self.max_orig_h = 10.0
        self.lowercase_floor_ratio = 0.72
        self.guide_preview_height_ratio = 0.56
        self.label_target_orig_h: dict[str, float] = {}

        # Tk-held resources (prevent GC).
        self.grid_photo_refs: list[ImageTk.PhotoImage] = []
        self.editor_photo_ref: ImageTk.PhotoImage | None = None

        families = sorted(set(tkfont.families(self.root)))
        fallback_family = (
            "Times New Roman" if "Times New Roman" in families else
            "Arial" if "Arial" in families else
            (families[0] if families else "TkDefaultFont")
        )
        self.guide_show_var = tk.BooleanVar(value=True)
        self.guide_family = fallback_family
        self.centerline_show_var = tk.BooleanVar(value=True)
        # PIL font is used to measure the TRUE ink bbox of each guide letter
        # (Tk's canvas.bbox returns the font line box, identical for every
        # character). We cache by (family, size) so rendering is cheap.
        self._pil_font_cache: dict[tuple[str, int], "ImageFont.FreeTypeFont"] = {}
        self.glyph_scale_var = tk.DoubleVar(value=1.0)
        self.glyph_scale_display_var = tk.StringVar(value="1.00x")
        # Advance multiplier applied on top of the auto-detected
        # advance_x at compose time. 1.0 = use the detected value.
        # <1 tightens (next letter sits closer), >1 loosens.
        self.glyph_advance_mul = 1.0
        self.glyph_advance_mul_var = tk.DoubleVar(value=1.0)
        self.glyph_advance_mul_display_var = tk.StringVar(value="1.00x")

        self._build_ui()
        self.reload_variants()

    # ── UI scaffolding ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        editor_page = ttk.Frame(self.notebook)
        compose_page = ttk.Frame(self.notebook)
        self.notebook.add(editor_page, text="Edit Glyphs")
        self.notebook.add(compose_page, text="Compose Text")

        self._build_editor_page(editor_page)
        self._build_compose_page(compose_page)

    def _build_editor_page(self, parent: ttk.Frame) -> None:
        paned = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=3)
        paned.add(right, weight=2)

        # Row 1: import + library path + reload
        top = ttk.Frame(left)
        top.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Button(top, text="Import PDF…",
                   command=self.on_import_pdf_clicked).pack(side=tk.LEFT)
        ttk.Button(top, text="New",
                   command=self.on_new_handwriting_clicked
                   ).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(top, text="Save",
                   command=self.on_export_handwriting_clicked
                   ).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(top, text="Load",
                   command=self.on_import_handwriting_clicked
                   ).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(top, text="  Libraries:").pack(side=tk.LEFT)
        self.libs_var = tk.StringVar(value=",".join(str(p) for p in self.library_paths))
        self.libs_entry = ttk.Entry(top, textvariable=self.libs_var, width=70)
        self.libs_entry.pack(side=tk.LEFT, padx=(6, 8), fill=tk.X, expand=True)
        ttk.Button(top, text="Reload", command=self.on_reload_clicked).pack(side=tk.LEFT)

        # Row 2: filter + counts + import status
        top2 = ttk.Frame(left)
        top2.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Label(top2, text="Filter:").pack(side=tk.LEFT)
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", lambda *_: self.apply_filter())
        ttk.Entry(top2, textvariable=self.filter_var, width=18).pack(side=tk.LEFT, padx=(4, 0))
        self.count_var = tk.StringVar(value="0 glyphs")
        ttk.Label(top2, textvariable=self.count_var).pack(side=tk.LEFT, padx=(12, 0))
        self.status_var = tk.StringVar(value="")
        ttk.Label(top2, textvariable=self.status_var,
                   foreground="#555").pack(side=tk.LEFT, padx=(16, 0))

        grid_frame = ttk.Frame(left)
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.grid_canvas = tk.Canvas(grid_frame, bg="#eeeeee", highlightthickness=0)
        self.grid_scroll = ttk.Scrollbar(grid_frame, orient=tk.VERTICAL,
                                          command=self.grid_canvas.yview)
        self.grid_canvas.configure(yscrollcommand=self.grid_scroll.set)
        self.grid_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.grid_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.grid_canvas.bind("<Button-1>", self.on_grid_click)
        self.grid_canvas.bind("<Configure>", self.on_grid_resize)
        self.grid_canvas.bind_all("<MouseWheel>", self.on_mousewheel)

        # ── Right panel ─────────────────────────────────────────────────
        info = ttk.LabelFrame(right, text="Selection")
        info.pack(fill=tk.X, padx=8, pady=8)
        self.sel_label_var = tk.StringVar(value="Label: -")
        self.sel_lib_var = tk.StringVar(value="Library: -")
        self.sel_path_var = tk.StringVar(value="Path: -")
        self.sel_size_var = tk.StringVar(value="Size: -")
        self.sel_offset_var = tk.StringVar(value="Offset: (0, 0)")
        self.new_label_var = tk.StringVar(value="")
        ttk.Label(info, textvariable=self.sel_label_var).pack(anchor="w", padx=8, pady=(8, 2))
        ttk.Label(info, textvariable=self.sel_lib_var).pack(anchor="w", padx=8, pady=2)
        ttk.Label(info, textvariable=self.sel_path_var, wraplength=460).pack(anchor="w", padx=8, pady=2)
        ttk.Label(info, textvariable=self.sel_size_var).pack(anchor="w", padx=8, pady=2)
        ttk.Label(info, textvariable=self.sel_offset_var).pack(anchor="w", padx=8, pady=(2, 8))

        relabel_row = ttk.Frame(info)
        relabel_row.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Label(relabel_row, text="New label:").pack(side=tk.LEFT)
        relabel_entry = ttk.Entry(relabel_row, textvariable=self.new_label_var, width=14)
        relabel_entry.pack(side=tk.LEFT, padx=(6, 6))
        relabel_entry.bind("<Return>", lambda _e: self.apply_relabel())
        ttk.Button(relabel_row, text="Apply Label", command=self.apply_relabel).pack(side=tk.LEFT)

        editor_group = ttk.LabelFrame(right, text="Editor (drag to move)")
        editor_group.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))



        buttons = ttk.Frame(editor_group)
        buttons.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 6))
        ttk.Button(buttons, text="Save Transform",
                   command=self.save_transform).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Reset Shift",
                   command=self.reset_shift).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(buttons, text="Split…",
                   command=self.on_split_glyph_clicked
                   ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(buttons, text="Delete Variant",
                   command=self.delete_variant).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(buttons, text="Next", command=self.select_next).pack(
            side=tk.LEFT, padx=(8, 0))
        self.editor_save_status = tk.StringVar(value="")
        ttk.Label(buttons, textvariable=self.editor_save_status,
                  foreground="#2a7c2a").pack(side=tk.LEFT, padx=(12, 0))

        scale_controls = ttk.Frame(editor_group)
        scale_controls.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 6))
        ttk.Label(scale_controls, text="Glyph scale:").pack(side=tk.LEFT)
        ttk.Scale(scale_controls, from_=0.3, to=3.0,
                  variable=self.glyph_scale_var,
                  orient=tk.HORIZONTAL, length=220,
                  command=self.on_glyph_scale_changed).pack(side=tk.LEFT, padx=(8, 8))
        ttk.Label(scale_controls, textvariable=self.glyph_scale_display_var,
                  width=6).pack(side=tk.LEFT)
        self.glyph_scale_var.trace_add("write",
                                        lambda *_: self.on_glyph_scale_changed())
        # "Reset" clears scale + offsets so the glyph snaps back to the
        # canvas line box with no user adjustment.
        ttk.Button(scale_controls, text="Reset",
                   command=self.auto_fit_current_glyph
                   ).pack(side=tk.LEFT)

        # Advance (×) — how tightly the NEXT letter sits to this one.
        # Independent of size; only affects compose layout, not what
        # you see in the editor canvas.
        adv_controls = ttk.Frame(editor_group)
        adv_controls.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 6))
        ttk.Label(adv_controls, text="Advance (×):").pack(side=tk.LEFT)
        ttk.Scale(adv_controls, from_=0.3, to=1.5,
                   variable=self.glyph_advance_mul_var,
                   orient=tk.HORIZONTAL, length=220,
                   command=self.on_advance_mul_changed
                   ).pack(side=tk.LEFT, padx=(8, 8))
        ttk.Label(adv_controls,
                   textvariable=self.glyph_advance_mul_display_var,
                   width=6).pack(side=tk.LEFT)
        self.glyph_advance_mul_var.trace_add(
            "write", lambda *_: self.on_advance_mul_changed()
        )
        ttk.Button(adv_controls, text="Reset",
                    command=lambda: self._set_advance_mul(1.0)
                    ).pack(side=tk.LEFT)

        guide_controls = ttk.Frame(editor_group)
        guide_controls.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 6))
        ttk.Checkbutton(guide_controls, text="Show font guide",
                        variable=self.guide_show_var,
                        command=self.redraw_editor).pack(side=tk.LEFT)
        ttk.Checkbutton(guide_controls, text="Show centerline",
                        variable=self.centerline_show_var,
                        command=self.redraw_editor).pack(side=tk.LEFT, padx=(10, 0))

        # Canvas last — fills whatever vertical space is left. A moderate
        # reqheight (not 520) so the frame doesn't force an over-tall window.
        self.editor_canvas = tk.Canvas(editor_group, width=480, height=360,
                                        bg="#d9d9d9", highlightthickness=1)
        self.editor_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                                 padx=8, pady=8)
        self.editor_canvas.bind("<Button-1>", self.on_editor_press)
        self.editor_canvas.bind("<B1-Motion>", self.on_editor_drag)
        self.editor_canvas.bind("<Configure>", lambda _e: self.redraw_editor())

        # Key bindings — only fire when the Edit Glyphs tab is active, so
        # typing in the Compose Text input doesn't nudge the hidden glyph.
        def _if_editor(fn):
            def handler(_e):
                if getattr(self, "notebook", None) is None:
                    fn()
                    return
                try:
                    if self.notebook.index("current") == 0:
                        fn()
                except Exception:
                    fn()
            return handler

        self.root.bind("<Left>", _if_editor(lambda: self.nudge(-0.2, 0)))
        self.root.bind("<Right>", _if_editor(lambda: self.nudge(0.2, 0)))
        self.root.bind("<Up>", _if_editor(lambda: self.nudge(0, -0.2)))
        self.root.bind("<Down>", _if_editor(lambda: self.nudge(0, 0.2)))
        # F2 = coverage report. bind_all so it fires regardless of which
        # tab or widget has focus (including the compose Text box).
        self.root.bind_all("<F2>", self.on_show_coverage_report)
        # Ctrl+R = render. Also bind_all so it works while typing in
        # the compose text box — that's the common case (edit, then
        # re-render).
        self.root.bind_all(
            "<Control-r>",
            lambda _e: (self.on_compose_render(), "break")[1],
        )
        self.root.bind_all(
            "<Control-R>",
            lambda _e: (self.on_compose_render(), "break")[1],
        )
        self._coverage_window: tk.Toplevel | None = None

    # ── Compose page ────────────────────────────────────────────────────

    def _build_compose_page(self, parent: ttk.Frame) -> None:
        """Right-hand tab: type text, render it using the loaded glyph
        library, preview the composition, optionally save as PNG/SVG."""
        paned = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=3)

        ttk.Label(left, text="Text to render:").pack(anchor="w",
                                                     padx=8, pady=(8, 2))
        self.compose_text = tk.Text(left, height=10, wrap="word",
                                    font=("Consolas", 11))
        self.compose_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        # Default sample text lives in sample_text.txt next to the app
        # so it's easy to swap out without touching code. Falls back to
        # a short sentence if the file is missing or unreadable.
        sample_path = Path(__file__).resolve().parent / "sample_text.txt"
        try:
            sample_text = sample_path.read_text(encoding="utf-8")
        except Exception:
            sample_text = "The quick brown fox\njumps over the lazy dog"
        self.compose_text.insert("1.0", sample_text)
        ttk.Label(
            left,
            text="Tip: put '---PAGE---' on its own line to start a new page.",
            foreground="#666",
        ).pack(anchor="w", padx=8, pady=(0, 4))

        ctrl = ttk.Frame(left)
        ctrl.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(ctrl, text="Page:").grid(row=0, column=0, sticky="w")
        ttk.Label(ctrl, text="8.5 × 11 in (Letter)",
                  foreground="#666").grid(row=0, column=1, columnspan=3,
                                          sticky="w", padx=(4, 0))
        ttk.Label(ctrl, text="Margin (in):").grid(row=1, column=0, sticky="w",
                                                   pady=(6, 0))
        self.compose_margin_var = tk.DoubleVar(value=0.5)
        ttk.Spinbox(ctrl, from_=0.0, to=3.0, increment=0.1,
                    textvariable=self.compose_margin_var, width=6
                    ).grid(row=1, column=1, padx=(4, 12), sticky="w",
                           pady=(6, 0))
        ttk.Label(ctrl, text="Zoom:").grid(row=1, column=2, sticky="w",
                                            pady=(6, 0))
        # Zoom 1.0 = whole page fits the preview canvas. Above 1 scales
        # up (scrolls), below 1 shrinks. Changing zoom re-displays the
        # already-rendered pages — no re-render. Ctrl+scroll over the
        # preview also works.
        self.compose_scale_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(ctrl, from_=1.0, to=5.0, increment=0.1,
                    textvariable=self.compose_scale_var, width=6
                    ).grid(row=1, column=3, padx=(4, 0), sticky="w",
                           pady=(6, 0))

        ttk.Label(ctrl, text="Pen thickness (pt):").grid(
            row=2, column=0, sticky="w", pady=(6, 0))
        # Default 0.6 pt ≈ 0.21 mm — a typical fineliner / technical pen
        # width. Units are PDF points so values translate directly to
        # plotter output dimensions.
        self.compose_thickness_var = tk.DoubleVar(value=0.6)
        ttk.Spinbox(ctrl, from_=0.05, to=5.0, increment=0.05,
                    textvariable=self.compose_thickness_var, width=6,
                    ).grid(row=2, column=1, padx=(4, 12),
                           sticky="w", pady=(6, 0))

        # Global font-size multiplier. 1.0 = glyph's natural size; smaller
        # shrinks everything, larger blows it up. When ruled-paper is on
        # the baselines still snap to blue rules regardless, so large
        # sizes will encroach on the line above and small ones leave a
        # gap — pick a size that fits the rule spacing (~20 pt cap).
        ttk.Label(ctrl, text="Font size (×):").grid(
            row=2, column=2, sticky="w", pady=(6, 0))
        self.compose_font_size_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(ctrl, from_=0.3, to=2.5, increment=0.05,
                    textvariable=self.compose_font_size_var, width=6,
                    ).grid(row=2, column=3, padx=(4, 0),
                           sticky="w", pady=(6, 0))

        # Baseline offset (pt): shift every line up (-) or down (+)
        # relative to the blue rules. Useful when a handwriting library
        # tends to sit too low / high on the rule by default.
        ttk.Label(ctrl, text="Baseline Δ (pt):").grid(
            row=3, column=0, sticky="w", pady=(6, 0))
        self.compose_baseline_offset_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(ctrl, from_=-15.0, to=15.0, increment=0.5,
                    textvariable=self.compose_baseline_offset_var, width=6,
                    ).grid(row=3, column=1, padx=(4, 12),
                           sticky="w", pady=(6, 0))

        self.compose_random_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Randomize variants",
                        variable=self.compose_random_var
                        ).grid(row=4, column=0, columnspan=4,
                               sticky="w", pady=(6, 0))

        # Overlay the raw centerline polylines on top of the stroked output
        # so you can see exactly what shape the pen will draw.
        self.compose_vectors_show_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Overlay vector centerlines (red)",
                        variable=self.compose_vectors_show_var
                        ).grid(row=5, column=0, columnspan=4,
                               sticky="w", pady=(2, 0))

        self.compose_ruled_paper_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Ruled paper background",
                        variable=self.compose_ruled_paper_var
                        ).grid(row=6, column=0, columnspan=4,
                               sticky="w", pady=(2, 0))

        # Naturalness panel (collapsible — click the header to
        # hide/show the body so it doesn't hog vertical space when
        # you're not tweaking it).
        #   • Geometric "feel" effects (position / rotation / drift)
        #     are sliders — you drag, you see.
        #   • Dimensional effects (size, word / letter spacing) and
        #     absolute minimum-gap buffers are spinboxes — you want
        #     to type a specific number.
        nat_outer = ttk.Frame(left, relief="groove", borderwidth=1)
        nat_outer.pack(fill=tk.X, padx=8, pady=(4, 6))

        self._nat_expanded = True
        self._nat_header_text_var = tk.StringVar(value="▾  Naturalness")
        ttk.Button(nat_outer,
                    textvariable=self._nat_header_text_var,
                    command=self._toggle_naturalness_panel,
                    style="Toolbutton"
                    ).pack(fill=tk.X)

        nat = ttk.Frame(nat_outer)
        nat.pack(fill=tk.X)
        nat.columnconfigure(1, weight=1)
        # Remember the body so the toggle can hide/show it.
        self._nat_body = nat

        self.nat_pos_var = tk.BooleanVar(value=True)
        self.nat_pos_strength_var = tk.DoubleVar(value=1.0)
        self.nat_rot_var = tk.BooleanVar(value=True)
        self.nat_rot_strength_var = tk.DoubleVar(value=1.0)
        self.nat_size_var = tk.BooleanVar(value=True)
        self.nat_size_strength_var = tk.DoubleVar(value=1.0)
        self.nat_drift_var = tk.BooleanVar(value=True)
        self.nat_drift_strength_var = tk.DoubleVar(value=1.0)
        self.nat_spacing_var = tk.BooleanVar(value=True)
        self.nat_spacing_strength_var = tk.DoubleVar(value=1.0)
        self.nat_letterspacing_var = tk.BooleanVar(value=True)
        self.nat_letterspacing_strength_var = tk.DoubleVar(value=1.0)
        self.nat_anticlump_var = tk.BooleanVar(value=True)
        # Random early line-break: occasionally wrap to the next line
        # before reaching the right margin, matching real handwriting
        # where the eye says "this word will fit but I'm close enough".
        self.nat_earlybreak_var = tk.BooleanVar(value=True)
        self.nat_earlybreak_strength_var = tk.DoubleVar(value=1.0)
        # Minimum absolute distances, in PDF points. Floors applied
        # after spacing jitter so letters/words never get tighter than
        # these regardless of strength settings.
        self.nat_word_buffer_var = tk.DoubleVar(value=3.0)
        self.nat_letter_buffer_var = tk.DoubleVar(value=0.15)

        # Two-column body: glyph-transform knobs on the left, spacing /
        # layout knobs on the right. Each sub-frame runs its own grid
        # so slider+readout widths don't interact across the divider.
        nat_left = ttk.Frame(nat)
        nat_left.grid(row=0, column=0, sticky="nsew", padx=(2, 4))
        nat_right = ttk.Frame(nat)
        nat_right.grid(row=0, column=1, sticky="nsew", padx=(4, 2))
        nat.columnconfigure(0, weight=1)
        nat.columnconfigure(1, weight=1)

        # Row counters per column.
        rows = {"L": 0, "R": 0}

        def _slider_row(parent: ttk.Frame, side: str,
                         cb_var: tk.BooleanVar,
                         strength_var: tk.DoubleVar,
                         label: str) -> None:
            r = rows[side]
            disp_var = tk.StringVar(value=f"{strength_var.get():.2f}")
            strength_var.trace_add(
                "write",
                lambda *_, s=strength_var, d=disp_var:
                    d.set(f"{s.get():.2f}"),
            )
            ttk.Checkbutton(parent, text=label, variable=cb_var
                             ).grid(row=r, column=0, sticky="w",
                                    padx=(2, 0), pady=1)
            ttk.Scale(parent, from_=0.0, to=2.0,
                       variable=strength_var,
                       orient="horizontal", length=90
                       ).grid(row=r, column=1, sticky="ew",
                              padx=(3, 3), pady=1)
            ttk.Label(parent, textvariable=disp_var, width=4
                       ).grid(row=r, column=2, sticky="w",
                              padx=(0, 2), pady=1)
            rows[side] = r + 1

        def _spinbox_row(parent: ttk.Frame, side: str,
                          cb_var: tk.BooleanVar | None,
                          value_var: tk.DoubleVar,
                          label: str,
                          from_: float, to: float,
                          increment: float) -> None:
            r = rows[side]
            if cb_var is not None:
                ttk.Checkbutton(parent, text=label, variable=cb_var
                                 ).grid(row=r, column=0, sticky="w",
                                        padx=(2, 0), pady=1)
            else:
                ttk.Label(parent, text=label
                            ).grid(row=r, column=0, sticky="w",
                                   padx=(2, 0), pady=1)
            ttk.Spinbox(parent, from_=from_, to=to, increment=increment,
                         textvariable=value_var, width=5
                         ).grid(row=r, column=1, columnspan=2,
                                sticky="w", padx=(3, 3), pady=1)
            rows[side] = r + 1

        # ── Left: glyph-transform jitters ───────────────────────────────
        _slider_row(nat_left, "L", self.nat_pos_var,
                     self.nat_pos_strength_var,   "Position")
        _slider_row(nat_left, "L", self.nat_rot_var,
                     self.nat_rot_strength_var,   "Rotation")
        _slider_row(nat_left, "L", self.nat_drift_var,
                     self.nat_drift_strength_var, "Baseline drift")
        _spinbox_row(nat_left, "L", self.nat_size_var,
                      self.nat_size_strength_var,
                      "Size jitter", 0.0, 2.0, 0.1)

        # ── Right: spacing / layout ─────────────────────────────────────
        _spinbox_row(nat_right, "R", self.nat_spacing_var,
                      self.nat_spacing_strength_var,
                      "Word spacing",    0.0, 2.0, 0.1)
        _spinbox_row(nat_right, "R", self.nat_letterspacing_var,
                      self.nat_letterspacing_strength_var,
                      "Letter spacing",  0.0, 2.0, 0.1)
        _spinbox_row(nat_right, "R", self.nat_earlybreak_var,
                      self.nat_earlybreak_strength_var,
                      "Early break",     0.0, 2.0, 0.1)
        _spinbox_row(nat_right, "R", None, self.nat_word_buffer_var,
                      "Word buffer pt",   0.0, 20.0, 0.25)
        _spinbox_row(nat_right, "R", None, self.nat_letter_buffer_var,
                      "Letter buffer pt", 0.0, 5.0, 0.05)

        # Anti-clump sits below the two columns as a full-width
        # checkbutton since it's boolean and fits naturally at the end.
        ttk.Checkbutton(nat, text="Anti-clump variant picking",
                         variable=self.nat_anticlump_var
                         ).grid(row=1, column=0, columnspan=2,
                                sticky="w", padx=(4, 0), pady=(2, 2))

        btn_row = ttk.Frame(left)
        btn_row.pack(fill=tk.X, padx=8, pady=(4, 6))
        ttk.Button(btn_row, text="Render (Ctrl+R)",
                   command=self.on_compose_render).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Export…",
                   command=self.on_compose_export_clicked
                   ).pack(side=tk.LEFT, padx=(8, 0))

        self.compose_status = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.compose_status,
                  foreground="#555", wraplength=320, justify="left"
                  ).pack(fill=tk.X, padx=8, pady=(4, 8))

        # Preview: scrollable canvas.
        preview_frame = ttk.Frame(right)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.compose_canvas = tk.Canvas(preview_frame, bg="#ffffff",
                                        highlightthickness=1,
                                        highlightbackground="#aaa")
        vsb = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL,
                            command=self.compose_canvas.yview)
        hsb = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL,
                            command=self.compose_canvas.xview)
        self.compose_canvas.configure(yscrollcommand=vsb.set,
                                      xscrollcommand=hsb.set)
        self.compose_canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        # Live zoom: re-display (no re-render) when the zoom value
        # changes or the preview canvas is resized. Ctrl+wheel zooms
        # too so you can dial in by scrolling over the preview.
        self.compose_scale_var.trace_add(
            "write", lambda *_: self._schedule_compose_display_refresh()
        )
        self.compose_canvas.bind(
            "<Configure>",
            lambda _e: self._schedule_compose_display_refresh(),
        )
        self.compose_canvas.bind(
            "<Control-MouseWheel>", self._on_compose_ctrl_wheel
        )

        # Preview state.
        self.compose_photo_ref: ImageTk.PhotoImage | None = None
        self.compose_last_image: "Image.Image | None" = None
        # Per-page high-DPI PIL images. Kept on self so zoom / window
        # resize can re-display without re-rendering glyphs.
        self.compose_page_images: list["Image.Image"] = []
        self._compose_refresh_pending: str | None = None
        self.compose_last_text: str = ""
        # Cache of composed glyph layouts for SVG export: each element is
        # {"x_pt": float, "y_pt": float, "glyph": Glyph, "label": str}.
        self.compose_layout: list[dict] = []
        # Centerline cache: {variant_abs_path: [{"points": [...], "closed": bool}, ...]}.
        # compute_centerlines is ~10–20 ms per glyph; caching means rendering a
        # 200-char page after the first render is basically instant. Cleared on
        # library reload (normalize changes geometry).
        self.compose_centerline_cache: dict[Path, list[dict]] = {}

    # ── Compose logic ───────────────────────────────────────────────────

    def _variants_by_label(self) -> dict[str, list[GlyphVariant]]:
        by_label: dict[str, list[GlyphVariant]] = {}
        for v in self.variants:
            by_label.setdefault(v.label, []).append(v)
        return by_label

    def _layout_text(
        self,
        text: str,
        variants_by_label: dict[str, list[GlyphVariant]],
        line_width_pt: float,
        line_height_pt: float | None = None,
        first_baseline_pt: float | None = None,
        rng=None,
        naturalness: dict | None = None,
        font_size: float = 1.0,
    ) -> tuple[list[dict], float, float, set[str]]:
        """Compute per-glyph placements for `text`.

        Returns (positions, total_width_pt, total_height_pt, missing_chars)
        where positions is a list of dicts:
            {"x_pt", "y_baseline_pt", "glyph", "label"}
        Missing characters (not in the library) are represented by a blank
        space (advance only), and their labels are collected in
        missing_chars.

        `line_height_pt` / `first_baseline_pt` override the defaults when
        snapping to external rules (e.g. blue college-rule lines).

        `rng` / `naturalness` drive the per-glyph jitters (position, size,
        rotation, darkness) and anti-clump variant picking. When rng is
        None or naturalness flags are off the output is deterministic.
        """
        import random as _random

        if rng is None:
            rng = _random.Random(0)
        nat = naturalness or {}

        all_orig_h = [v.orig_h for vs in variants_by_label.values()
                      for v in vs if v.orig_h is not None]
        ref_h = max(all_orig_h) if all_orig_h else 10.0
        # Font size scales advances, kerning, and the free-form (non-
        # ruled) line height so everything grows/shrinks together. When
        # the caller supplied a line-height override (ruled paper) we
        # leave it alone — baselines still land on blue rules regardless
        # of font size.
        if line_height_pt is None:
            line_height_pt = ref_h * 1.35 * font_size
        descender_reserve = ref_h * 0.25 * font_size
        space_advance = ref_h * 0.32 * font_size
        kerning = ref_h * 0.05 * font_size

        rotate_counters: dict[str, int] = {}
        last_picked_idx: dict[str, int] = {}
        loaded_cache: dict[Path, Glyph] = {}

        def _pick_variant(ch: str) -> GlyphVariant | None:
            vs = variants_by_label.get(ch)
            if not vs:
                return None
            if self.compose_random_var.get():
                # Random mode. With anti-clump on, avoid repeating the
                # variant we just used for this char (when >1 exist).
                if nat.get("anti_clump") and len(vs) > 1 and ch in last_picked_idx:
                    candidates = [
                        (j, cand) for j, cand in enumerate(vs)
                        if j != last_picked_idx[ch]
                    ]
                    j, pick = rng.choice(candidates)
                else:
                    j = rng.randrange(len(vs))
                    pick = vs[j]
                last_picked_idx[ch] = j
                return pick
            idx = rotate_counters.get(ch, 0)
            rotate_counters[ch] = idx + 1
            return vs[idx % len(vs)]

        def _load(v: GlyphVariant) -> Glyph | None:
            g = loaded_cache.get(v.abs_path)
            if g is not None:
                return g
            try:
                g = Glyph.from_file(v.abs_path)
            except Exception:
                return None
            loaded_cache[v.abs_path] = g
            return g

        positions: list[dict] = []
        missing: set[str] = set()
        cur_x = 0.0
        cur_baseline = first_baseline_pt if first_baseline_pt is not None else ref_h
        max_x = 0.0

        # Pulled up here so both the space path and the glyph path see
        # the same values. Each amplitude feature has its own strength
        # so e.g. you can have max rotation with zero position jitter.
        pos_on = bool(nat.get("pos"))
        pos_strength = float(nat.get("pos_strength", 1.0))
        rot_on = bool(nat.get("rot"))
        rot_strength = float(nat.get("rot_strength", 1.0))
        size_on = bool(nat.get("size"))
        size_strength = float(nat.get("size_strength", 1.0))
        spacing_on = bool(nat.get("spacing"))
        spacing_strength = float(nat.get("spacing_strength", 1.0))
        letterspacing_on = bool(nat.get("letterspacing"))
        letterspacing_strength = float(
            nat.get("letterspacing_strength", 1.0)
        )
        word_buffer_pt = float(nat.get("word_buffer", 0.0))
        letter_buffer_pt = float(nat.get("letter_buffer", 0.0))
        earlybreak_on = bool(nat.get("earlybreak"))
        earlybreak_strength = float(nat.get("earlybreak_strength", 1.0))

        def _space_adv() -> float:
            """Width of ONE inter-word space. Jittered per occurrence
            when word-spacing jitter is on so spacing isn't rhythmic.
            ±25% × strength, clamped to [-40%, +40%] so very high
            strengths can't collapse words. Additionally floored at
            max(60% of base, word_buffer_pt) so there's always a clear
            visible gap between words."""
            if not spacing_on:
                return max(space_advance, word_buffer_pt)
            var = rng.uniform(-0.25, 0.25) * spacing_strength
            var = max(-0.4, min(0.4, var))
            candidate = space_advance * (1.0 + var)
            return max(space_advance * 0.6, word_buffer_pt, candidate)

        # Word-level wrapping: we buffer a whole word's glyphs in
        # `word_buf`, and only commit them to `positions` when we hit
        # a word boundary (space, newline, end of input, missing char).
        # At that point we can check if the full word fits on the
        # current line — if not, we wrap ALL of the word's letters
        # together instead of breaking mid-word, which is what a human
        # actually does.
        word_buf: list[dict] = []

        def _commit_letter_to_positions(item: dict, x_pt: float,
                                          y_baseline: float) -> None:
            positions.append({
                "x_pt": x_pt + item["off_x"],
                "y_baseline_pt": y_baseline,
                "glyph": item["g"],
                "label": item["ch"],
                "baseline_local": item["baseline_local_scaled"] - item["off_y"],
                "scale": item["scale"],
                "width": item["g_w"],
                "height": item["g_h"],
                "abs_path": item["abs_path"],
                "x_jitter_pt": item["x_jitter"],
                "y_jitter_pt": item["y_jitter"],
                "rot_deg": item["rot_deg"],
                "size_mul": item["size_mul"],
            })

        def _flush_word() -> None:
            """Place everything in word_buf on the current line (wrapping
            to a new line first if the word doesn't fit)."""
            nonlocal cur_x, cur_baseline, max_x, word_buf
            if not word_buf:
                return
            # Total word advance = first letter + sum of (kerning + adv) for the rest.
            total = word_buf[0]["adv_w"]
            for it in word_buf[1:]:
                total += it["kerning_before"] + it["adv_w"]

            overflow_current = cur_x + total > line_width_pt and cur_x > 0

            # Occasional "I'd rather start the next line" break — fires
            # only near the right margin and only when the word would
            # still fit. Real handwriting does this when the eye senses
            # the line is basically done.
            do_early = False
            if (not overflow_current
                    and earlybreak_on and cur_x > 0
                    and earlybreak_strength > 0):
                ratio = cur_x / max(1.0, line_width_pt)
                if ratio > 0.7:
                    prob = min(
                        0.75,
                        ((ratio - 0.7) / 0.3) * 0.5 * earlybreak_strength,
                    )
                    if rng.random() < prob:
                        do_early = True

            if overflow_current or do_early:
                cur_x = 0.0
                cur_baseline += line_height_pt
                # If even a fresh line can't hold this word, fall back
                # to letter-wise wrapping for this single word so it
                # doesn't run off the page. This is the one case where
                # a word MAY get split — unavoidable if it's too long.
                if total > line_width_pt:
                    x_cursor = 0.0
                    for idx, it in enumerate(word_buf):
                        gap = it["kerning_before"] if idx > 0 else 0.0
                        if x_cursor + gap + it["adv_w"] > line_width_pt \
                                and x_cursor > 0:
                            cur_baseline += line_height_pt
                            x_cursor = 0.0
                            gap = 0.0
                        x_cursor += gap
                        _commit_letter_to_positions(it, x_cursor, cur_baseline)
                        x_cursor += it["adv_w"]
                    cur_x = x_cursor
                    max_x = max(max_x, cur_x)
                    word_buf = []
                    return

            x_cursor = cur_x
            for idx, it in enumerate(word_buf):
                if idx > 0:
                    x_cursor += it["kerning_before"]
                _commit_letter_to_positions(it, x_cursor, cur_baseline)
                x_cursor += it["adv_w"]
            cur_x = x_cursor
            max_x = max(max_x, cur_x)
            word_buf = []

        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "\n":
                _flush_word()
                cur_x = 0.0
                cur_baseline += line_height_pt
                i += 1
                continue
            if ch == "\t":
                _flush_word()
                # Tabs jitter too — treated as four spaces, each with
                # its own variation.
                cur_x += sum(_space_adv() for _ in range(4))
                if cur_x > line_width_pt:
                    cur_x = 0.0
                    cur_baseline += line_height_pt
                i += 1
                continue
            if ch.isspace():
                _flush_word()
                adv = _space_adv()
                if cur_x + adv > line_width_pt and cur_x > 0:
                    cur_x = 0.0
                    cur_baseline += line_height_pt
                else:
                    cur_x += adv
                    max_x = max(max_x, cur_x)
                i += 1
                continue

            v = _pick_variant(ch)
            if v is None:
                # Missing char: commit the pending word, then treat
                # this as a space-width gap. Breaks the surrounding
                # word in half — unavoidable since we have no glyph to
                # place.
                _flush_word()
                missing.add(ch)
                if cur_x + space_advance > line_width_pt and cur_x > 0:
                    cur_x = 0.0
                    cur_baseline += line_height_pt
                cur_x += space_advance
                max_x = max(max_x, cur_x)
                i += 1
                continue

            g = _load(v)
            if g is None:
                _flush_word()
                missing.add(ch)
                cur_x += space_advance
                i += 1
                continue

            # Effective size for this placement = glyph.width × v.user_scale.
            # Both offset_x and offset_y scale with user_scale so the drag
            # reads the same visual distance on the composed page as it did
            # in the editor.
            scale = float(v.user_scale or 1.0) * font_size
            g_w = g.width * scale
            g_h = g.height * scale
            baseline_local_scaled = float(
                v.baseline if v.baseline is not None else g.height
            ) * scale
            off_x = float(v.offset_x or 0.0) * scale
            off_y = float(v.offset_y or 0.0) * scale

            # Advance width: how far the cursor moves to place the next
            # glyph. Distinct from g_w so that e.g. a 'q' tail can
            # overhang into the space of the next letter. Lazy-compute
            # for libraries that predate the normalize pass that writes
            # it, then cache on the variant for the rest of this render.
            if v.advance_x is None:
                try:
                    from svg_glyph import compute_advance_x
                    v.advance_x = float(compute_advance_x(g))
                except Exception:
                    v.advance_x = float(g.width)
            base_adv = float(v.advance_x)
            adv_w = base_adv * float(v.advance_mul or 1.0) * scale

            # Per-glyph naturalness draws. Each amplitude has its own
            # strength knob — flipping a checkbox off cleanly zeros
            # that feature regardless of the others.
            x_jitter = (rng.uniform(-0.6, 0.6) * pos_strength
                        if pos_on else 0.0)
            y_jitter = (rng.uniform(-0.6, 0.6) * pos_strength
                        if pos_on else 0.0)
            rot_deg = (rng.uniform(-1.8, 1.8) * rot_strength
                       if rot_on else 0.0)
            size_mul = (1.0 + rng.uniform(-0.04, 0.04) * size_strength
                        if size_on else 1.0)

            # Kerning applies BETWEEN letters within a word, so the
            # first letter of a word uses 0 here.
            if len(word_buf) == 0:
                eff_kerning = 0.0
            else:
                eff_kerning = kerning
                if letterspacing_on:
                    prop = max(-kerning * 0.6, min(
                        kerning * 0.6,
                        rng.uniform(-kerning * 0.6, kerning * 0.6)
                        * letterspacing_strength,
                    ))
                    abs_jit = max(-0.35, min(0.35,
                        rng.uniform(-0.25, 0.25) * letterspacing_strength,
                    ))
                    eff_kerning += prop + abs_jit
                eff_kerning = max(letter_buffer_pt, eff_kerning)

            word_buf.append({
                "ch": ch,
                "g": g,
                "scale": scale,
                "g_w": g_w,
                "g_h": g_h,
                "baseline_local_scaled": baseline_local_scaled,
                "off_x": off_x,
                "off_y": off_y,
                "abs_path": v.abs_path,
                "adv_w": adv_w,
                "kerning_before": eff_kerning,
                "x_jitter": x_jitter,
                "y_jitter": y_jitter,
                "rot_deg": rot_deg,
                "size_mul": size_mul,
            })
            i += 1

        # End of input — don't forget the last word.
        _flush_word()

        total_w = max(max_x, 1.0)
        total_h = cur_baseline + descender_reserve
        return positions, total_w, total_h, missing

    def _get_centerlines_cached(self, abs_path, glyph: Glyph) -> list[dict]:
        """Return centerline polylines for a glyph, cached by variant path.
        First access for a variant runs compute_centerlines (~10–20 ms);
        subsequent access is a dict lookup. Cache is cleared on library
        reload (normalize invalidates glyph geometry)."""
        if abs_path is not None and abs_path in self.compose_centerline_cache:
            return self.compose_centerline_cache[abs_path]
        try:
            cl = compute_centerlines(glyph)
        except Exception:
            cl = []
        if abs_path is not None:
            self.compose_centerline_cache[abs_path] = cl
        return cl

    # ── Naturalness helpers ─────────────────────────────────────────────

    @staticmethod
    def _rotate_points(
        points: list[tuple[float, float]],
        cx: float, cy: float, rot_deg: float,
    ) -> list[tuple[float, float]]:
        """Rotate `points` about (cx, cy) by `rot_deg` degrees. Pivot is
        chosen by the caller; typically the glyph's baseline centre so
        the rotated baseline still sits on the rule."""
        if abs(rot_deg) < 1e-6:
            return points
        rad = math.radians(rot_deg)
        c, s = math.cos(rad), math.sin(rad)
        return [
            (c * (x - cx) - s * (y - cy) + cx,
             s * (x - cx) + c * (y - cy) + cy)
            for x, y in points
        ]

    # Page-break markers. A line whose stripped content matches any of
    # these (case-insensitive) starts a new page. `\f` (form feed) is also
    # honoured for pasted content from word processors.
    PAGE_BREAK_TOKENS = ("---PAGE---", "[PAGE]", "\\PAGE", "<PAGEBREAK>")

    @classmethod
    def _split_pages(cls, text: str) -> list[str]:
        """Split `text` into one string per page on page-break markers."""
        tokens = {t.upper() for t in cls.PAGE_BREAK_TOKENS}
        pages: list[list[str]] = [[]]
        for line in text.split("\n"):
            if "\f" in line:
                # Form feed splits the line itself — respect it.
                parts = line.split("\f")
                pages[-1].append(parts[0])
                for part in parts[1:]:
                    pages.append([part])
                continue
            if line.strip().upper() in tokens:
                pages.append([])
                continue
            pages[-1].append(line)
        return ["\n".join(p) for p in pages]

    def _render_single_page(
        self,
        text: str,
        variants_by_label: dict[str, list[GlyphVariant]],
        *,
        page_w_pt: float,
        page_h_pt: float,
        margin_pt: float,
        px_per_pt: float,
        rng=None,
        naturalness: dict | None = None,
        font_size: float = 1.0,
    ) -> tuple["Image.Image", list[dict], float, set[str],
               float, float, float]:
        """Render one page and return (img, positions, overflow_pt,
        missing_runtime, eff_margin_x_pt, eff_margin_y_pt, content_h_pt)."""
        import random as _random
        if rng is None:
            rng = _random.Random(0)
        nat = naturalness or {}

        RULE_LEFT_MARGIN_PT = 90.0
        RULE_LINE_SPACING_PT = 9.0 / 32.0 * 72.0
        RULE_FIRST_LINE_PT = 72.0

        # User-configurable vertical offset applied to every baseline
        # so text can sit higher (−) or lower (+) relative to the blue
        # rules. Kept out of the None-branch for unruled paper because
        # there are no rules to offset from.
        try:
            baseline_offset_pt = float(
                self.compose_baseline_offset_var.get()
            )
        except (ValueError, tk.TclError):
            baseline_offset_pt = 0.0

        if self.compose_ruled_paper_var.get():
            RULE_LEFT_GAP_PT = 4.0
            eff_margin_x_pt = RULE_LEFT_MARGIN_PT + RULE_LEFT_GAP_PT
            eff_margin_y_pt = 0.0
            line_height_override: float | None = RULE_LINE_SPACING_PT
            # First baseline sits on the TOP blue rule — the "title"
            # line. Ascenders lean into the 1" top margin above, which
            # matches how people write on notebook paper. The offset
            # shifts every baseline uniformly; rules stay put.
            first_baseline_override: float | None = (
                RULE_FIRST_LINE_PT + baseline_offset_pt
            )
            line_width_pt = max(10.0, page_w_pt - eff_margin_x_pt - margin_pt)
        else:
            eff_margin_x_pt = margin_pt
            eff_margin_y_pt = margin_pt
            line_height_override = None
            first_baseline_override = None
            line_width_pt = max(10.0, page_w_pt - 2.0 * margin_pt)

        positions, _total_w_pt, content_h_pt, missing_runtime = self._layout_text(
            text, variants_by_label, line_width_pt,
            line_height_pt=line_height_override,
            first_baseline_pt=first_baseline_override,
            rng=rng, naturalness=nat,
            font_size=font_size,
        )

        W = max(1, int(round(page_w_pt * px_per_pt)))
        H = max(1, int(round(page_h_pt * px_per_pt)))
        overflow_pt = max(
            0.0,
            (content_h_pt + eff_margin_y_pt) - (page_h_pt - margin_pt),
        )

        img = Image.new("RGBA", (W, H), (255, 255, 255, 255))
        page_draw = ImageDraw.Draw(img)
        page_draw.rectangle((0, 0, W - 1, H - 1),
                             outline=(150, 150, 150, 255), width=1)

        if self.compose_ruled_paper_var.get():
            RULE_LINE_WIDTH_PX = max(1, int(round(0.5 * px_per_pt)))
            blue = (170, 200, 230, 220)
            red = (220, 110, 110, 220)

            y_pt = RULE_FIRST_LINE_PT
            while y_pt < page_h_pt - 18.0:
                y_px = int(round(y_pt * px_per_pt))
                page_draw.line(
                    [(0, y_px), (W - 1, y_px)],
                    fill=blue, width=RULE_LINE_WIDTH_PX,
                )
                y_pt += RULE_LINE_SPACING_PT

            margin_x_px = int(round(RULE_LEFT_MARGIN_PT * px_per_pt))
            page_draw.line(
                [(margin_x_px, 0), (margin_x_px, H - 1)],
                fill=red, width=RULE_LINE_WIDTH_PX,
            )

        thickness_pt = max(0.05, float(self.compose_thickness_var.get()))
        thickness_px = max(1, int(round(thickness_pt * px_per_pt)))
        ink_rgba = (20, 20, 20, 255)
        stroke_draw = ImageDraw.Draw(img)

        # Baseline drift — coherent across a page rather than per-line
        # random. One wavelength + slope + second-harmonic is picked
        # once per page; each successive line advances the phase by a
        # fixed step and nudges the amplitude slightly. Result: the
        # whole page reads as "this person's hand", not a fresh roll
        # of dice on every line.
        drift_on = bool(nat.get("drift"))
        drift_strength = float(nat.get("drift_strength", 1.0))
        if drift_on:
            page_amp = rng.uniform(0.5, 1.0) * drift_strength
            page_wavelength = rng.uniform(150.0, 230.0)
            page_phase0 = rng.uniform(0.0, 2.0 * math.pi)
            # Phase step per line — large enough that consecutive lines
            # aren't identical, small enough that they feel related.
            page_phase_step = rng.uniform(math.pi / 3.5, math.pi / 2.0)
            # Subtle whole-page slope: sheet feels slightly tilted.
            # 0.002 * strength × ~500 pt line = ~1 pt max shift.
            page_slope = rng.uniform(-1.0, 1.0) * 0.002 * drift_strength
            # Second harmonic adds organic non-sinusoidal shape without
            # being chaotic.
            page_h2_amp = page_amp * rng.uniform(0.2, 0.4)
        else:
            page_amp = 0.0
            page_wavelength = 200.0
            page_phase0 = 0.0
            page_phase_step = 0.0
            page_slope = 0.0
            page_h2_amp = 0.0

        # Per-line params derived from page params. Cached by un-jittered
        # baseline y so all glyphs on the same line share phase/amp.
        line_drift_params: dict[float, tuple[float, float]] = {}
        _line_counter: list[int] = [0]

        def _line_drift(line_y: float, x: float) -> float:
            if not drift_on:
                return 0.0
            params = line_drift_params.get(line_y)
            if params is None:
                idx = _line_counter[0]
                _line_counter[0] += 1
                phase = page_phase0 + idx * page_phase_step
                # Line-to-line amplitude wobble (±15%) so adjacent lines
                # aren't carbon copies.
                amp_line = page_amp * rng.uniform(0.85, 1.15)
                params = (amp_line, phase)
                line_drift_params[line_y] = params
            amp_line, phase = params
            w = 2.0 * math.pi / page_wavelength
            return (
                amp_line * math.sin(w * x + phase)
                + page_h2_amp * math.sin(2.0 * w * x + 2.0 * phase)
                + page_slope * x
            )

        rot_on = bool(nat.get("rot"))
        size_on = bool(nat.get("size"))
        pos_on = bool(nat.get("pos"))

        for item in positions:
            g: Glyph = item["glyph"]
            base_scale = float(item.get("scale", 1.0))
            size_mul = float(item.get("size_mul", 1.0)) if size_on else 1.0
            scale = base_scale * size_mul
            placed_h = float(item.get("height", g.height))
            abs_path = item.get("abs_path")
            baseline_local = item.get("baseline_local") or placed_h
            rot_deg = float(item.get("rot_deg", 0.0)) if rot_on else 0.0
            x_jitter = float(item.get("x_jitter_pt", 0.0)) if pos_on else 0.0
            y_jitter = float(item.get("y_jitter_pt", 0.0)) if pos_on else 0.0

            centerlines = self._get_centerlines_cached(abs_path, g)
            if not centerlines:
                continue

            # Rotation pivot = (glyph-local) centre of the baseline. This
            # keeps the baseline rotation-invariant so rotated glyphs
            # still sit on the blue rule.
            pivot_cx = g.width / 2.0
            pivot_cy = baseline_local

            line_y_key = float(item["y_baseline_pt"])
            glyph_top_pt = line_y_key - float(baseline_local)
            origin_x_pt = eff_margin_x_pt + float(item["x_pt"]) + x_jitter
            origin_y_pt = eff_margin_y_pt + glyph_top_pt + y_jitter

            # Build per-stroke final polylines in PAGE POINTS (post
            # rotation+drift+wobble). These are cached on the item so
            # SVG export can emit identical geometry without re-running
            # the stochastic pipeline.
            page_strokes_for_item: list[dict] = []
            for stroke in centerlines:
                pts = stroke.get("points", [])
                if len(pts) < 2:
                    continue

                # Apply per-glyph rotation in glyph-local space first.
                local_pts = (
                    self._rotate_points(pts, pivot_cx, pivot_cy, rot_deg)
                    if rot_deg else pts
                )

                # Transform to page coords (pt), applying drift per point
                # (drift depends on x so long strokes curve slightly with
                # the line).
                page_pts: list[tuple[float, float]] = []
                for gx, gy in local_pts:
                    px_pt = origin_x_pt + gx * scale
                    py_pt = origin_y_pt + gy * scale
                    py_pt += _line_drift(line_y_key, px_pt)
                    page_pts.append((px_pt, py_pt))

                closed = bool(stroke.get("closed"))
                if closed and page_pts[0] != page_pts[-1]:
                    page_pts.append(page_pts[0])

                page_strokes_for_item.append({
                    "pts_pt": page_pts,
                    "closed": closed,
                })

                # Raster render: to pixels, then draw.
                coords = [(x * px_per_pt, y * px_per_pt) for x, y in page_pts]
                stroke_draw.line(
                    coords, fill=ink_rgba,
                    width=thickness_px, joint="curve",
                )

                if self.compose_vectors_show_var.get():
                    stroke_draw.line(
                        coords, fill=(220, 30, 30, 255),
                        width=1, joint="curve",
                    )

            item["page_strokes"] = page_strokes_for_item

        return (img, positions, overflow_pt, missing_runtime,
                eff_margin_x_pt, eff_margin_y_pt, content_h_pt)

    def on_compose_render(self) -> None:
        text = self.compose_text.get("1.0", "end-1c")
        if not text.strip():
            self.compose_status.set("Text is empty.")
            return
        if not self.variants:
            messagebox.showinfo(
                "No glyphs loaded",
                "Import a PDF or load a glyph library first.",
            )
            return

        variants_by_label = self._variants_by_label()

        # Missing-character check before doing any work. Skip page-break
        # marker lines when scanning so their '-' or '[' chars don't get
        # flagged as missing when those characters aren't in the library.
        page_texts_for_check = self._split_pages(text)
        check_text = "\n".join(page_texts_for_check)
        missing_chars = {
            ch for ch in check_text
            if (not ch.isspace()) and (ch not in variants_by_label)
        }
        if missing_chars:
            listed = ", ".join(repr(c) for c in sorted(missing_chars))
            if not messagebox.askyesno(
                "Missing characters",
                f"The library has no glyphs for: {listed}\n\n"
                "Render anyway, leaving blank space for each missing character?",
            ):
                return

        # Letter portrait: 8.5 × 11 in in PDF points (72 pt/in).
        PAGE_W_PT = 612.0
        PAGE_H_PT = 792.0
        # Internal render resolution — decoupled from the user's zoom so
        # strokes stay crisp when the preview is shrunk to fit the window
        # and when the user zooms in. 3 px/pt → 1836×2376 px page; with
        # LANCZOS resample on display, edges stay smooth at any zoom ≤ 3.
        # Higher (4+) looks marginally better but makes the first render
        # (no centerline cache) take 10s+.
        RENDER_PX_PER_PT = 3.0
        margin_pt = max(0.0, float(self.compose_margin_var.get())) * 72.0
        px_per_pt = RENDER_PX_PER_PT
        zoom = max(1.0, float(self.compose_scale_var.get()))

        # Collect naturalness toggles once so _layout_text and
        # _render_single_page both see the same settings. Flags map to
        # short keys used throughout the render path.
        def _nat_strength(var: tk.DoubleVar) -> float:
            try:
                return max(0.0, float(var.get()))
            except (ValueError, tk.TclError):
                return 0.0

        naturalness = {
            "pos":                  bool(self.nat_pos_var.get()),
            "pos_strength":         _nat_strength(self.nat_pos_strength_var),
            "rot":                  bool(self.nat_rot_var.get()),
            "rot_strength":         _nat_strength(self.nat_rot_strength_var),
            "size":                 bool(self.nat_size_var.get()),
            "size_strength":        _nat_strength(self.nat_size_strength_var),
            "drift":                bool(self.nat_drift_var.get()),
            "drift_strength":       _nat_strength(self.nat_drift_strength_var),
            "spacing":              bool(self.nat_spacing_var.get()),
            "spacing_strength":     _nat_strength(
                self.nat_spacing_strength_var
            ),
            "letterspacing":        bool(self.nat_letterspacing_var.get()),
            "letterspacing_strength": _nat_strength(
                self.nat_letterspacing_strength_var
            ),
            "anti_clump":           bool(self.nat_anticlump_var.get()),
            "earlybreak":           bool(self.nat_earlybreak_var.get()),
            "earlybreak_strength":  _nat_strength(
                self.nat_earlybreak_strength_var
            ),
            "word_buffer":          max(0.0, _nat_strength(
                self.nat_word_buffer_var
            )),
            "letter_buffer":        max(0.0, _nat_strength(
                self.nat_letter_buffer_var
            )),
        }

        # Split text on page-break markers. Empty pages (e.g. trailing
        # break) are dropped so we don't render blank pages nobody asked
        # for.
        page_texts = [p for p in self._split_pages(text) if p.strip()]
        if not page_texts:
            self.compose_status.set("Text is empty.")
            return

        page_images: list["Image.Image"] = []
        page_layouts: list[list[dict]] = []
        total_glyphs = 0
        total_overflow_pt = 0.0
        all_missing: set[str] = set()
        eff_margin_x_pt = margin_pt
        eff_margin_y_pt = margin_pt

        font_size = max(0.05, float(self.compose_font_size_var.get()))

        import random as _random
        # Fresh seed every time Render is clicked so jitters shuffle
        # automatically — no manual reseed button needed. SVG/PNG export
        # read from the cached post-jitter geometry of this render, so
        # the saved file still matches exactly what's on screen.
        render_seed = _random.randint(1, 1_000_000_000)
        for page_idx, page_text in enumerate(page_texts):
            page_rng = _random.Random(
                (render_seed * 1_000_003 + page_idx) & 0xFFFFFFFF
            )
            (img, positions, overflow_pt, missing_runtime,
             eff_margin_x_pt, eff_margin_y_pt, _content_h_pt) = (
                self._render_single_page(
                    page_text, variants_by_label,
                    page_w_pt=PAGE_W_PT, page_h_pt=PAGE_H_PT,
                    margin_pt=margin_pt, px_per_pt=px_per_pt,
                    rng=page_rng, naturalness=naturalness,
                    font_size=font_size,
                )
            )
            for p in positions:
                p["page"] = page_idx
            page_images.append(img)
            page_layouts.append(positions)
            total_glyphs += len(positions)
            total_overflow_pt = max(total_overflow_pt, overflow_pt)
            all_missing |= missing_runtime

        # Combine all pages into one tall image (for PNG export). Pages
        # are stacked vertically with a thin gap filled with page-gutter
        # grey so they read as distinct sheets.
        W = page_images[0].width
        H_single = page_images[0].height
        gap_px = max(1, int(round(12.0 * px_per_pt)))   # ~12 pt gap
        total_h = H_single * len(page_images) + gap_px * max(0, len(page_images) - 1)
        combined = Image.new("RGBA", (W, total_h), (220, 220, 220, 255))
        y = 0
        for img in page_images:
            combined.paste(img, (0, y))
            y += H_single + gap_px

        self.compose_last_image = combined
        self.compose_last_text = text
        self.compose_layout = [p for layout in page_layouts for p in layout]
        self.compose_page_layouts = page_layouts
        self.compose_page_pt = (PAGE_W_PT, PAGE_H_PT)
        self.compose_margin_pt = margin_pt
        self.compose_margin_x_pt = eff_margin_x_pt
        self.compose_margin_y_pt = eff_margin_y_pt
        # Stash the per-page images so zoom / resize can re-display
        # from them without re-rendering every glyph.
        self.compose_page_images = page_images

        self._refresh_compose_display()

        notes = []
        if all_missing:
            notes.append(
                "Missing (blanked): "
                + ", ".join(repr(c) for c in sorted(all_missing))
            )
        if total_overflow_pt > 0:
            notes.append(
                f"Text overflows page by {total_overflow_pt / 72.0:.2f} in"
            )
        note_str = "  |  ".join(notes)
        page_count = len(page_images)
        page_desc = f"{page_count} page" + ("s" if page_count != 1 else "")
        self.compose_status.set(
            f"8.5×11 in — {page_desc} @ zoom {zoom:.2f}× "
            f"(render {px_per_pt:g} px/pt) — "
            f"{total_glyphs} glyphs, margin {margin_pt / 72.0:.2f} in."
            + (f"  {note_str}" if note_str else "")
        )

    # ── Preview display (zoom / resize, no re-render) ──────────────────

    def _schedule_compose_display_refresh(self) -> None:
        """Coalesce rapid zoom/resize events into a single redraw on
        the next idle tick. Prevents LANCZOS-resize thrash when the
        user scrolls the zoom wheel quickly or drags the window edge."""
        if self._compose_refresh_pending is not None:
            try:
                self.root.after_cancel(self._compose_refresh_pending)
            except Exception:
                pass
        self._compose_refresh_pending = self.root.after(
            30, self._refresh_compose_display
        )

    def _refresh_compose_display(self) -> None:
        """Resize the cached per-page images to the current zoom and
        canvas size, then paint them on the compose canvas. Cheap
        enough to run live on the zoom slider; no glyph rendering
        happens here."""
        self._compose_refresh_pending = None
        if not self.compose_page_images:
            return
        page_w_pt, page_h_pt = getattr(
            self, "compose_page_pt", (612.0, 792.0)
        )
        try:
            zoom = max(1.0, float(self.compose_scale_var.get()))
        except (ValueError, tk.TclError):
            return

        try:
            self.compose_canvas.update_idletasks()
        except Exception:
            pass
        canvas_w = max(200, self.compose_canvas.winfo_width())
        canvas_h = max(200, self.compose_canvas.winfo_height())
        inner_w = max(50, canvas_w - 16)
        inner_h = max(50, canvas_h - 16)
        fit_scale = min(inner_w / page_w_pt, inner_h / page_h_pt)
        display_px_per_pt = max(0.1, fit_scale * zoom)
        disp_w = max(1, int(round(page_w_pt * display_px_per_pt)))
        disp_h_page = max(1, int(round(page_h_pt * display_px_per_pt)))
        disp_gap = max(6, int(round(12.0 * display_px_per_pt)))

        self.compose_canvas.delete("all")
        self.compose_photo_refs = []
        y_display = 0
        for img in self.compose_page_images:
            display_img = img.resize((disp_w, disp_h_page), Image.LANCZOS)
            photo = ImageTk.PhotoImage(display_img.convert("RGB"))
            self.compose_photo_refs.append(photo)
            self.compose_canvas.create_image(
                0, y_display, image=photo, anchor="nw",
            )
            y_display += disp_h_page + disp_gap
        self.compose_photo_ref = (
            self.compose_photo_refs[0] if self.compose_photo_refs else None
        )
        total_disp_h = max(
            disp_h_page,
            y_display - disp_gap if self.compose_page_images else disp_h_page,
        )
        self.compose_canvas.configure(
            scrollregion=(0, 0, disp_w, total_disp_h),
        )

    def _on_compose_ctrl_wheel(self, event: tk.Event) -> str:
        """Ctrl+scroll over the preview zooms in/out. Returns "break"
        so the canvas itself doesn't also scroll on the same event."""
        try:
            cur = float(self.compose_scale_var.get())
        except (ValueError, tk.TclError):
            cur = 1.0
        step = 0.1 if event.delta > 0 else -0.1
        new = max(1.0, min(5.0, round(cur + step, 2)))
        if new != cur:
            self.compose_scale_var.set(new)
        return "break"

    def _write_png_file(self, path: Path) -> None:
        """Write the stacked preview image to `path` (raises on error)."""
        if self.compose_last_image is None:
            raise RuntimeError("Nothing to export — render the text first.")
        self.compose_last_image.convert("RGB").save(str(path))

    def _write_svg_files(self, base_path: Path) -> list[Path]:
        """Emit per-page SVGs rooted at `base_path`. Single-page docs
        write `base_path` itself; multi-page docs write `<stem>_p1.svg`,
        `<stem>_p2.svg`, … next to it. Returns the list of paths
        written. Raises on error."""
        page_layouts: list[list[dict]] = getattr(
            self, "compose_page_layouts", [self.compose_layout]
        )
        if not page_layouts or not any(page_layouts):
            raise RuntimeError("Nothing to export — render the text first.")

        page_w, page_h = getattr(self, "compose_page_pt", (612.0, 792.0))
        margin_pt = getattr(self, "compose_margin_pt", 36.0)
        margin_x_pt = getattr(self, "compose_margin_x_pt", margin_pt)
        margin_y_pt = getattr(self, "compose_margin_y_pt", margin_pt)
        thickness_pt = max(0.05, float(self.compose_thickness_var.get()))

        def _path_d(pts: list[tuple[float, float]], closed: bool) -> str:
            if len(pts) < 2:
                return ""
            head = f"M {pts[0][0]:.3f} {pts[0][1]:.3f}"
            rest = " ".join(f"L {x:.3f} {y:.3f}" for x, y in pts[1:])
            return (head + " " + rest + (" Z" if closed else "")).strip()

        def _render_svg(page_positions: list[dict]) -> str:
            parts: list[str] = [
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'viewBox="0 0 {page_w:.3f} {page_h:.3f}" '
                f'width="{page_w / 72.0:.3f}in" '
                f'height="{page_h / 72.0:.3f}in">',
                f'<g fill="none" stroke="black" '
                f'stroke-width="{thickness_pt:.3f}" '
                f'stroke-linecap="round" stroke-linejoin="round">',
            ]
            # Prefer the cached post-jitter polylines so plotter output
            # matches what the user sees in the preview.
            for item in page_positions:
                page_strokes = item.get("page_strokes")
                if page_strokes:
                    for stroke in page_strokes:
                        d = _path_d(stroke["pts_pt"], bool(stroke.get("closed")))
                        if d:
                            parts.append(f'<path d="{d}"/>')
                    continue
                g: Glyph = item["glyph"]
                sc = float(item.get("scale", 1.0))
                placed_h = float(item.get("height", g.height))
                baseline_local = item.get("baseline_local") or placed_h
                abs_path = item.get("abs_path")
                centerlines = self._get_centerlines_cached(abs_path, g)
                if not centerlines:
                    continue
                origin_x = margin_x_pt + float(item["x_pt"])
                origin_y = (
                    margin_y_pt + float(item["y_baseline_pt"])
                    - float(baseline_local)
                )
                for stroke in centerlines:
                    pts = stroke.get("points", [])
                    if len(pts) < 2:
                        continue
                    page_pts = [
                        (origin_x + gx * sc, origin_y + gy * sc)
                        for gx, gy in pts
                    ]
                    closed = bool(stroke.get("closed"))
                    d = _path_d(page_pts, closed)
                    if d:
                        parts.append(f'<path d="{d}"/>')
            parts.append("</g>")
            parts.append("</svg>")
            return "\n".join(parts)

        written: list[Path] = []
        if len(page_layouts) == 1:
            base_path.write_text(
                _render_svg(page_layouts[0]), encoding="utf-8"
            )
            written.append(base_path)
        else:
            stem = base_path.stem
            for idx, page_positions in enumerate(page_layouts, start=1):
                out = base_path.with_name(f"{stem}_p{idx}{base_path.suffix}")
                out.write_text(_render_svg(page_positions), encoding="utf-8")
                written.append(out)
        return written

    def _toggle_naturalness_panel(self) -> None:
        """Show / hide the Naturalness body. The arrow in the header
        (▾ / ▸) reflects the current state."""
        self._nat_expanded = not self._nat_expanded
        if self._nat_expanded:
            self._nat_header_text_var.set("▾  Naturalness")
            self._nat_body.pack(fill=tk.X)
        else:
            self._nat_header_text_var.set("▸  Naturalness")
            self._nat_body.pack_forget()

    # ── Unified export ──────────────────────────────────────────────────

    def on_compose_export_clicked(self) -> None:
        """One-stop export: pick format (PNG / SVG / Both) then filename.
        Replaces the old separate Save-PNG / Save-SVG buttons."""
        if self.compose_last_image is None or not getattr(
            self, "compose_page_layouts", None
        ):
            messagebox.showinfo("Nothing to export",
                                 "Render the text first.")
            return

        fmt_var = tk.StringVar(value="png")

        win = tk.Toplevel(self.root)
        win.title("Export")
        win.transient(self.root)
        win.resizable(False, False)
        win.grab_set()

        ttk.Label(win, text="Choose a format:"
                   ).pack(padx=14, pady=(14, 6), anchor="w")
        for label, value in (
            ("PNG — stacked pages, single image", "png"),
            ("SVG — one file per page, plotter-ready", "svg"),
            ("Both — PNG + SVG with the same basename", "both"),
        ):
            ttk.Radiobutton(win, text=label, variable=fmt_var, value=value
                             ).pack(padx=22, anchor="w")

        btn_row = ttk.Frame(win)
        btn_row.pack(fill=tk.X, padx=12, pady=(12, 12))

        def _go() -> None:
            choice = fmt_var.get()
            win.destroy()
            self._run_export(choice)

        ttk.Button(btn_row, text="Export…", command=_go
                    ).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btn_row, text="Cancel", command=win.destroy
                    ).pack(side=tk.RIGHT)
        win.bind("<Return>", lambda _e: _go())
        win.bind("<Escape>", lambda _e: win.destroy())

    def _run_export(self, fmt: str) -> None:
        """Pop the save dialog appropriate for `fmt` and write the
        file(s). For 'both' the user supplies ONE basename and both a
        .png and matching .svg(s) are written next to it."""
        if fmt == "png":
            path_str = filedialog.asksaveasfilename(
                parent=self.root, title="Export PNG",
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("All files", "*.*")],
            )
            if not path_str:
                return
            try:
                self._write_png_file(Path(path_str))
            except Exception as ex:  # noqa: BLE001
                messagebox.showerror("Export failed", str(ex))
                return
            self.compose_status.set(f"Wrote {path_str}")

        elif fmt == "svg":
            path_str = filedialog.asksaveasfilename(
                parent=self.root, title="Export SVG",
                defaultextension=".svg",
                filetypes=[("SVG", "*.svg"), ("All files", "*.*")],
            )
            if not path_str:
                return
            try:
                written = self._write_svg_files(Path(path_str))
            except Exception as ex:  # noqa: BLE001
                messagebox.showerror("Export failed", str(ex))
                return
            if len(written) == 1:
                self.compose_status.set(f"Wrote {written[0]}")
            else:
                self.compose_status.set(
                    f"Wrote {len(written)} SVG files "
                    f"({written[0].name} … {written[-1].name})"
                )

        elif fmt == "both":
            path_str = filedialog.asksaveasfilename(
                parent=self.root, title="Export (basename)",
                defaultextension="",
                filetypes=[("All files", "*.*")],
            )
            if not path_str:
                return
            base = Path(path_str)
            # Strip any extension the user may have typed so we can
            # attach both .png and .svg ourselves.
            stem_path = base.with_suffix("")
            png_path = stem_path.with_suffix(".png")
            svg_path = stem_path.with_suffix(".svg")
            try:
                self._write_png_file(png_path)
                written = self._write_svg_files(svg_path)
            except Exception as ex:  # noqa: BLE001
                messagebox.showerror("Export failed", str(ex))
                return
            svg_count = len(written)
            self.compose_status.set(
                f"Wrote {png_path.name} + {svg_count} SVG file"
                + ("s" if svg_count != 1 else "")
            )

    # ── Library / grid ──────────────────────────────────────────────────

    def on_mousewheel(self, event: tk.Event) -> None:
        if self.grid_canvas.winfo_exists():
            self.grid_canvas.yview_scroll(int(-event.delta / 120), "units")

    def on_grid_resize(self, _event: tk.Event) -> None:
        w = max(1, self.grid_canvas.winfo_width())
        self.grid_cols = max(1, w // self.cell_w)
        self.rebuild_grid()

    def on_reload_clicked(self) -> None:
        self.library_paths = parse_library_paths([self.libs_var.get()])
        self.reload_variants()

    # ── Handwriting save / load ─────────────────────────────────────────
    #
    # A .hwfont file is just a zip of the primary library folder: the
    # per-character variant subfolders + index.json (which carries all
    # the user_scale / offset / baseline edits). Using the zip format
    # means the file survives fine if anything in this app breaks — you
    # can always unzip it manually.

    # ── Coverage report ─────────────────────────────────────────────────

    def on_show_coverage_report(self, event=None) -> None:  # noqa: ARG002
        """Open (or focus) a window listing every common ASCII
        character and how many variants the current library has for it.
        Bound to F2."""
        existing = getattr(self, "_coverage_window", None)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.lift()
                    existing.focus_set()
                    return
            except tk.TclError:
                pass

        win = tk.Toplevel(self.root)
        win.title("Library coverage")
        win.geometry("480x600")
        self._coverage_window = win

        def _on_close() -> None:
            self._coverage_window = None
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _on_close)
        # Escape closes the window — common expectation for a popup.
        win.bind("<Escape>", lambda _e: _on_close())

        header = ttk.Label(win, text="", anchor="w", wraplength=460)
        header.pack(padx=10, pady=(10, 4), fill=tk.X)

        tree_frame = ttk.Frame(win)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 6))

        columns = ("character", "count", "category")
        tree = ttk.Treeview(tree_frame, columns=columns,
                             show="headings", height=22)
        tree.column("character", width=90, anchor="center")
        tree.column("count", width=90, anchor="center")
        tree.column("category", width=220, anchor="w")

        vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                             command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        tree.tag_configure("missing", background="#ffdcdc")
        tree.tag_configure("sparse",  background="#fff2cc")

        def _category(c: str) -> str:
            if c.isupper():   return "Uppercase letter"
            if c.islower():   return "Lowercase letter"
            if c.isdigit():   return "Digit"
            if c in ".,;:!?'\"`":
                return "Punctuation"
            if c in "()[]{}":
                return "Bracket"
            if c in "-_/\\|":
                return "Dash / slash"
            return "Symbol"

        def _sort_by(col: str, reverse: bool) -> None:
            items = [(tree.set(k, col), k) for k in tree.get_children("")]
            if col == "count":
                items.sort(key=lambda t: int(t[0]), reverse=reverse)
            else:
                items.sort(key=lambda t: t[0], reverse=reverse)
            for index, (_, k) in enumerate(items):
                tree.move(k, "", index)
            tree.heading(col, command=lambda c=col, r=reverse:
                          _sort_by(c, not r))

        tree.heading("character", text="Character",
                      command=lambda: _sort_by("character", False))
        tree.heading("count", text="Variants",
                      command=lambda: _sort_by("count", False))
        tree.heading("category", text="Category",
                      command=lambda: _sort_by("category", False))

        def _populate() -> None:
            by_label = self._variants_by_label()
            tree.delete(*tree.get_children())
            total = len(COMMON_CHARS)
            have = sum(1 for c in COMMON_CHARS if c in by_label)
            sparse = sum(
                1 for c in COMMON_CHARS if len(by_label.get(c, [])) == 1
            )
            missing = total - have
            header.config(text=(
                f"{have} of {total} characters have variants  ·  "
                f"{missing} missing  ·  {sparse} sparse (1 variant)"
            ))
            for c in COMMON_CHARS:
                count = len(by_label.get(c, []))
                if count == 0:
                    tag = "missing"
                elif count == 1:
                    tag = "sparse"
                else:
                    tag = ""
                tree.insert(
                    "", tk.END,
                    values=(c, count, _category(c)),
                    tags=(tag,) if tag else (),
                )
            # Default sort: by count ascending so missing/sparse rows
            # surface at the top.
            _sort_by("count", False)

        btn_row = ttk.Frame(win)
        btn_row.pack(fill=tk.X, padx=10, pady=(0, 4))
        ttk.Button(btn_row, text="Refresh",
                    command=_populate).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Close",
                    command=_on_close).pack(side=tk.RIGHT)
        ttk.Label(win, text="F2 toggles this window · Esc closes it",
                   foreground="#888"
                   ).pack(padx=10, pady=(0, 10), anchor="w")

        _populate()

    def on_new_handwriting_clicked(self) -> None:
        """Start a fresh, empty handwriting library in its own folder.
        The current library stays on disk — use the libs textbox or
        Load handwriting to come back to it later."""
        from tkinter.simpledialog import askstring

        folder_name = askstring(
            "New handwriting",
            "Name for the new (empty) handwriting library:",
            parent=self.root, initialvalue="my_handwriting",
        )
        if folder_name is None:
            return
        folder_name = folder_name.strip()
        for bad in ("/", "\\", ":", "*", "?", "\"", "<", ">", "|"):
            folder_name = folder_name.replace(bad, "_")
        if not folder_name:
            messagebox.showerror("Invalid name",
                                  "Folder name can't be empty.")
            return

        root = libraries_root()
        dest = root / folder_name
        if dest.exists():
            messagebox.showerror(
                "Folder exists",
                f"A folder named '{folder_name}' already exists in\n"
                f"{root}\n"
                "Pick a different name or delete it first.",
            )
            return

        try:
            dest.mkdir(parents=True)
            # Seed an empty index so the loader treats it as a valid —
            # just empty — library rather than failing on missing file.
            (dest / "index.json").write_text(
                '{"version": 3, "format": "svg", "source": null, "glyphs": {}}',
                encoding="utf-8",
            )
        except Exception as ex:  # noqa: BLE001
            messagebox.showerror("Create failed", str(ex))
            return

        self.libs_var.set(str(dest))
        self.library_paths = parse_library_paths([str(dest)])
        self.reload_variants()
        self.compose_centerline_cache.clear()
        self.status_var.set(f"Started new handwriting in {dest}")

    def on_export_handwriting_clicked(self) -> None:
        """Zip the primary library folder into a single .hwfont file."""
        import zipfile

        if not self.library_paths:
            messagebox.showinfo("Nothing to save",
                                 "No handwriting library is loaded.")
            return
        src = Path(self.library_paths[0])
        if not src.is_dir():
            messagebox.showerror(
                "Save failed",
                f"Library folder not found:\n{src}",
            )
            return

        path_str = filedialog.asksaveasfilename(
            parent=self.root, title="Save handwriting",
            defaultextension=".hwfont",
            initialfile=f"{src.name}.hwfont",
            filetypes=[("Handwriting", "*.hwfont"), ("All files", "*.*")],
        )
        if not path_str:
            return

        try:
            with zipfile.ZipFile(path_str, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in src.rglob("*"):
                    if not file.is_file():
                        continue
                    rel = file.relative_to(src)
                    # Only include index.json and anything inside the
                    # per-character subfolders. Loose files at root
                    # (debug PNGs, scratch files) are dev noise and
                    # shouldn't travel with the handwriting.
                    if len(rel.parts) == 1 and rel.name != "index.json":
                        continue
                    zf.write(file, arcname=str(rel).replace("\\", "/"))
        except Exception as ex:  # noqa: BLE001
            messagebox.showerror("Save failed", str(ex))
            return
        self.status_var.set(f"Saved handwriting to {path_str}")

    def on_import_handwriting_clicked(self) -> None:
        """Pick a .hwfont, ask what to call the new library folder,
        extract into it, and reload."""
        import zipfile
        from tkinter.simpledialog import askstring

        path_str = filedialog.askopenfilename(
            parent=self.root, title="Load handwriting",
            filetypes=[("Handwriting", "*.hwfont"),
                       ("Zip archive", "*.zip"),
                       ("All files", "*.*")],
        )
        if not path_str:
            return

        src_file = Path(path_str)
        default_name = src_file.stem
        folder_name = askstring(
            "Name the library",
            "Folder name for this handwriting:",
            parent=self.root, initialvalue=default_name,
        )
        if folder_name is None:
            return
        folder_name = folder_name.strip()
        # Sanitize against path escapes. This is a folder NAME, not a
        # path — flatten slashes and reject empties.
        for bad in ("/", "\\", ":", "*", "?", "\"", "<", ">", "|"):
            folder_name = folder_name.replace(bad, "_")
        if not folder_name:
            messagebox.showerror("Invalid name",
                                  "Folder name can't be empty.")
            return

        import shutil

        root = libraries_root()
        dest = root / folder_name

        # Allow loading "over" an existing folder — common case is
        # re-loading a handwriting you saved earlier into the same
        # slot. Confirm since it's destructive, then wipe the folder
        # before extracting so stale variants from the old library
        # don't mix in with the loaded ones.
        if dest.exists():
            if not messagebox.askyesno(
                "Replace existing handwriting?",
                f"'{folder_name}' already exists in\n{root}\n\n"
                "Replace its contents with the handwriting you're loading?\n"
                "The current contents will be permanently deleted.",
                icon="warning",
                default="no",
            ):
                return
            try:
                shutil.rmtree(dest)
            except Exception as ex:  # noqa: BLE001
                messagebox.showerror(
                    "Couldn't replace folder",
                    f"Failed to remove the existing folder:\n{ex}",
                )
                return

        try:
            dest.mkdir(parents=True)
            with zipfile.ZipFile(src_file, "r") as zf:
                zf.extractall(dest)
        except Exception as ex:  # noqa: BLE001
            # Clean up the partial extraction so the name is free for
            # another try.
            shutil.rmtree(dest, ignore_errors=True)
            messagebox.showerror("Load failed", str(ex))
            return

        self.libs_var.set(str(dest))
        self.library_paths = parse_library_paths([str(dest)])
        self.reload_variants()
        # Invalidate compose-side caches so the new library's glyphs
        # aren't rendered with old centerline cache entries keyed off
        # previous variant paths.
        self.compose_centerline_cache.clear()
        self.status_var.set(f"Loaded handwriting from {src_file.name}")

    def on_import_pdf_clicked(self) -> None:
        """Pick one or more PDFs and APPEND them to the current library.

        Multi-select is enabled so the user can grab a whole batch at
        once — all of them go into the same folder, new variants
        numbered continuing from the existing `_N` suffixes so nothing
        on disk gets overwritten. Normalize runs once at the end of the
        batch (not per-PDF) since it's expensive and the result is the
        same either way."""
        pdf_paths_tuple = filedialog.askopenfilenames(
            parent=self.root,
            title="Select handwriting PDF(s) — multi-select supported",
            filetypes=[("PDF", "*.pdf"), ("All files", "*.*")],
            initialdir=str(Path.home() / "Desktop"),
        )
        if not pdf_paths_tuple:
            return
        pdf_paths = [Path(p) for p in pdf_paths_tuple]

        # Target = current library folder. If none, fall back to a
        # `default` library under the Desktop libraries root so nothing
        # lands inside the code repo.
        if self.library_paths:
            out_dir = Path(self.library_paths[0])
        else:
            out_dir = libraries_root() / "default"
        out_dir.mkdir(parents=True, exist_ok=True)

        total_added = 0
        total_labels_final = 0
        self.root.config(cursor="wait")
        try:
            from pdf_to_vector_glyphs import import_pdf_to_library

            def _progress(msg: str) -> None:
                self._set_status(msg)

            for i, pdf_path in enumerate(pdf_paths, start=1):
                self._set_status(
                    f"Importing ({i}/{len(pdf_paths)}) {pdf_path.name}…"
                )
                try:
                    stats = import_pdf_to_library(
                        pdf_path=pdf_path,
                        out_dir=out_dir,
                        progress=_progress,
                        merge=True,
                        normalize=False,
                    )
                except Exception as ex:  # noqa: BLE001
                    messagebox.showerror(
                        "Import failed",
                        f"{pdf_path.name}\n\n{type(ex).__name__}: {ex}",
                    )
                    continue
                total_added += stats["total_written"]
                total_labels_final = stats["n_unique_labels"]

            # Normalize once after the batch so all newly-added glyphs
            # share the same proportions as the guide font. Skipping the
            # per-PDF pass saved real time on multi-file imports.
            if total_added > 0:
                self._set_status("Normalizing glyphs to guide font…")
                try:
                    from svg_glyph import normalize_library
                    normalize_library(out_dir, progress=_progress)
                except Exception as ex:  # noqa: BLE001
                    self._set_status(f"Normalize skipped: {ex}")
        finally:
            self.root.config(cursor="")

        self._set_status(
            f"Added {total_added} glyphs from {len(pdf_paths)} PDF(s); "
            f"library now has {total_labels_final} unique labels."
        )

        # Point at the target dir (in case this was the first import) and
        # reload the grid so the new variants show up immediately.
        self.libs_var.set(str(out_dir))
        self.library_paths = parse_library_paths([str(out_dir)])
        self.reload_variants()

    def _set_status(self, msg: str) -> None:
        self.status_var.set(msg)
        # update_idletasks forces the Tk event loop to repaint the label
        # without re-entering user event handlers — safe to call from the
        # middle of a synchronous operation.
        try:
            self.root.update_idletasks()
        except Exception:
            pass

    def reload_variants(self) -> None:
        self.variants = load_variants(self.library_paths)
        # Invalidate compose centerline cache — library may have been
        # normalized or PDF-imported, which changes glyph geometry.
        if hasattr(self, "compose_centerline_cache"):
            self.compose_centerline_cache.clear()
        # Drop thumbnail cache: after a reload the variant set usually
        # changes wholesale, so old entries are mostly dead weight.
        # (Individual edits don't need this — the mtime-keyed cache
        # self-invalidates per file.)
        self._thumb_cache.clear()

        all_orig_h = [v.orig_h for v in self.variants
                      if v.orig_h is not None]
        self.ref_orig_h = float(statistics.median(all_orig_h)) if all_orig_h else 8.0
        # Library-wide max (actually 95th percentile to ignore OCR-junk
        # outliers). Used as the display-scale DENOMINATOR so the library's
        # tallest glyph fills the canvas line box and all other glyphs are
        # proportionally smaller — preserving 'H' taller than 'o' etc.
        if all_orig_h:
            sorted_h = sorted(all_orig_h)
            idx = max(0, int(len(sorted_h) * 0.95) - 1)
            self.max_orig_h = float(sorted_h[idx])
        else:
            self.max_orig_h = self.ref_orig_h
        lower_orig_h = [v.orig_h for v in self.variants
                        if v.orig_h is not None and is_single_lower_ascii(v.label)]
        if lower_orig_h:
            lower_q1 = float(np.percentile(lower_orig_h, 25))
            self.lowercase_floor_ratio = max(0.72, lower_q1 / self.ref_orig_h)
        else:
            self.lowercase_floor_ratio = 0.72

        by_label: dict[str, list[float]] = {}
        for v in self.variants:
            if v.orig_h is not None:
                by_label.setdefault(v.label, []).append(v.orig_h)
        self.label_target_orig_h = {
            k: float(statistics.median(vs)) for k, vs in by_label.items()
        }

        self.selected_global_idx = None
        self.editor_glyph = None
        self.editor_offset_x = 0.0
        self.editor_offset_y = 0.0
        self.glyph_user_scale = 1.0
        self.glyph_scale_var.set(1.0)
        self.apply_filter()

    def apply_filter(self) -> None:
        q = self.filter_var.get().strip().lower()
        self.filtered_indices = []
        for i, v in enumerate(self.variants):
            if not q or q in v.label.lower() or q in v.rel_path.lower():
                self.filtered_indices.append(i)
        self.count_var.set(f"{len(self.filtered_indices)} glyphs")
        self.rebuild_grid()

    def checker_bg(self, w: int, h: int, s: int = 8) -> Image.Image:
        yy, xx = np.indices((h, w))
        mask = ((xx // s) + (yy // s)) % 2 == 0
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[mask] = (245, 245, 245)
        arr[~mask] = (220, 220, 220)
        return Image.fromarray(arr, mode="RGB")

    def _checker_bg_cached(self) -> "Image.Image":
        """Checker background for thumbnails is pure decoration and
        identical for every tile — compute it once, hand back copies."""
        if (self._thumb_checker_bg is None
                or self._thumb_checker_bg.size != (self.thumb_box, self.thumb_box)):
            self._thumb_checker_bg = self.checker_bg(
                self.thumb_box, self.thumb_box
            )
        return self._thumb_checker_bg

    def make_thumb(self, variant: GlyphVariant) -> ImageTk.PhotoImage:
        """Rasterize a variant into a PhotoImage tile. Selection
        highlighting is handled separately on the canvas, so this
        output is stable per (path, mtime) and cacheable."""
        try:
            mtime_ns = variant.abs_path.stat().st_mtime_ns
        except OSError:
            mtime_ns = 0
        key = (str(variant.abs_path), mtime_ns)
        cached = self._thumb_cache.get(key)
        if cached is not None:
            return cached

        canvas = self._checker_bg_cached().copy()
        if variant.abs_path.exists():
            try:
                g = Glyph.from_file(variant.abs_path)
                # Coarse bezier sampling — thumbnails are 90 px, fine
                # curve detail is invisible and halves the non-zero
                # scanline cost.
                img = rasterize_glyph(
                    g, self.thumb_box, self.thumb_box,
                    pad=6, n_per_bezier=4,
                )
                canvas.paste(img.convert("RGB"), mask=img.split()[3])
            except Exception:
                miss = Image.new(
                    "RGBA", (self.thumb_box, self.thumb_box),
                    (255, 0, 0, 55),
                )
                canvas = Image.alpha_composite(
                    canvas.convert("RGBA"), miss
                ).convert("RGB")
        else:
            miss = Image.new(
                "RGBA", (self.thumb_box, self.thumb_box),
                (255, 0, 0, 55),
            )
            canvas = Image.alpha_composite(
                canvas.convert("RGBA"), miss
            ).convert("RGB")

        out = canvas.convert("RGBA")
        # Default neutral border. Selection is drawn on the canvas as a
        # separate rectangle so selection changes don't invalidate the
        # cache.
        ImageDraw.Draw(out).rectangle(
            (0, 0, self.thumb_box - 1, self.thumb_box - 1),
            outline=(150, 150, 150, 255), width=1,
        )
        photo = ImageTk.PhotoImage(out.convert("RGB"))
        self._thumb_cache[key] = photo
        return photo

    def _update_selection_highlight(self) -> None:
        """Draw/move the selection rectangle on the grid canvas. Called
        whenever selected_global_idx changes; O(1) regardless of grid
        size."""
        if self._selection_canvas_id is not None:
            try:
                self.grid_canvas.delete(self._selection_canvas_id)
            except tk.TclError:
                pass
            self._selection_canvas_id = None
        if self.selected_global_idx is None:
            return
        try:
            n = self.filtered_indices.index(self.selected_global_idx)
        except ValueError:
            return
        cols = max(1, self.grid_cols)
        row, col = divmod(n, cols)
        x_pad = y_pad = 10
        x0 = x_pad + col * self.cell_w
        y0 = y_pad + row * self.cell_h
        self._selection_canvas_id = self.grid_canvas.create_rectangle(
            x0, y0,
            x0 + self.thumb_box - 1, y0 + self.thumb_box - 1,
            outline="#1e6ef0", width=3,
        )

    def rebuild_grid(self) -> None:
        self.grid_canvas.delete("all")
        self.grid_photo_refs.clear()
        # Previous selection rect belonged to the just-deleted canvas
        # items; reset so _update_selection_highlight draws a fresh one.
        self._selection_canvas_id = None

        cols = max(1, self.grid_cols)
        x_pad = 10
        y_pad = 10
        for n, global_idx in enumerate(self.filtered_indices):
            row, col = divmod(n, cols)
            x0 = x_pad + col * self.cell_w
            y0 = y_pad + row * self.cell_h
            v = self.variants[global_idx]
            photo = self.make_thumb(v)
            self.grid_photo_refs.append(photo)
            self.grid_canvas.create_image(x0, y0, image=photo, anchor="nw")
            self.grid_canvas.create_text(
                x0 + 2, y0 + self.thumb_box + 12,
                text=f"{v.label} | {v.lib_path.name}",
                anchor="nw", fill="#222", font=("Segoe UI", 9),
            )
        total_rows = math.ceil(max(1, len(self.filtered_indices)) / cols)
        content_h = y_pad * 2 + total_rows * self.cell_h
        content_w = x_pad * 2 + cols * self.cell_w
        self.grid_canvas.configure(scrollregion=(0, 0, content_w, content_h))
        self._update_selection_highlight()

    def on_grid_click(self, event: tk.Event) -> None:
        cols = max(1, self.grid_cols)
        x = self.grid_canvas.canvasx(event.x)
        y = self.grid_canvas.canvasy(event.y)
        x_pad = 10
        y_pad = 10
        col = int((x - x_pad) // self.cell_w)
        row = int((y - y_pad) // self.cell_h)
        if col < 0 or row < 0:
            return
        idx_in_filtered = row * cols + col
        if idx_in_filtered < 0 or idx_in_filtered >= len(self.filtered_indices):
            return
        self.selected_global_idx = self.filtered_indices[idx_in_filtered]
        self.load_selected_into_editor()
        # Just move the selection rect — don't re-rasterize the whole
        # grid for a selection change.
        self._update_selection_highlight()

    # ── Selection / load ────────────────────────────────────────────────

    def current_variant(self) -> GlyphVariant | None:
        if self.selected_global_idx is None:
            return None
        if not (0 <= self.selected_global_idx < len(self.variants)):
            return None
        return self.variants[self.selected_global_idx]

    def load_selected_into_editor(self) -> None:
        v = self.current_variant()
        if v is None:
            return
        if not v.abs_path.exists():
            messagebox.showerror("Missing file", f"File does not exist:\n{v.abs_path}")
            return
        try:
            self.editor_glyph = Glyph.from_file(v.abs_path)
        except Exception as ex:  # noqa: BLE001
            messagebox.showerror("Parse failed", f"Could not parse SVG:\n{ex}")
            return

        try:
            self.editor_centerlines = compute_centerlines(self.editor_glyph)
        except Exception:
            self.editor_centerlines = []

        # Baseline defaults to bottom of the glyph when unknown (matches the
        # old "bottom-align" behaviour for non-descender chars).
        self.editor_baseline = (
            float(v.baseline) if v.baseline is not None
            else float(self.editor_glyph.height)
        )
        # Restore user-editable state from index. Drag and slider mutate
        # these; Save persists them verbatim.
        self.editor_offset_x = float(v.offset_x or 0.0)
        self.editor_offset_y = float(v.offset_y or 0.0)
        self._set_glyph_scale(float(v.user_scale or 1.0), redraw=False)
        self._set_advance_mul(float(v.advance_mul or 1.0))

        self.sel_label_var.set(f"Label: {v.label}")
        self.sel_lib_var.set(f"Library: {v.lib_path}")
        self.sel_path_var.set(f"Path: {v.rel_path}")
        self.sel_size_var.set(
            f"Size: {self.editor_glyph.width:.2f} x {self.editor_glyph.height:.2f} pt")
        self.sel_offset_var.set("Offset: (0.00, 0.00)")
        self.new_label_var.set(v.label)

        self.redraw_editor()

    # ── Editor rendering ────────────────────────────────────────────────

    # Windows TTF filenames for the families we use as guide fonts.
    _FAMILY_TO_TTF = {
        "Times New Roman": "times.ttf",
        "Arial": "arial.ttf",
        "Courier New": "cour.ttf",
        "Consolas": "consola.ttf",
    }

    def _get_pil_font(self, size_px: int) -> "ImageFont.FreeTypeFont | None":
        """Look up or build a PIL FreeType font for the current guide
        family at `size_px`. Used to measure real ink bboxes per character
        (Tk's canvas bbox returns the line box, which is identical for
        every letter and therefore useless for per-glyph sizing)."""
        key = (self.guide_family, int(size_px))
        font = self._pil_font_cache.get(key)
        if font is not None:
            return font
        ttf = self._FAMILY_TO_TTF.get(self.guide_family, "arial.ttf")
        for candidate in (ttf, "arial.ttf"):
            try:
                font = ImageFont.truetype(candidate, int(size_px))
                break
            except Exception:
                font = None
        if font is not None:
            self._pil_font_cache[key] = font
        return font

    def _guide_ink_bbox_px(
        self, label: str, gsize: int, baseline_y: int, center_x: int,
    ) -> tuple[int, int, int, int] | None:
        """Return the expected CANVAS-pixel ink bbox of the Tk-rendered
        guide letter for this label — computed via PIL so it reflects the
        actual ink extent (not the font line box). Coordinates are aligned
        assuming the guide was drawn with its typographic baseline at
        `baseline_y` and centred on `center_x`."""
        pil_font = self._get_pil_font(gsize)
        if pil_font is None:
            return None
        try:
            bb = pil_font.getbbox(label)
            ascent_px = pil_font.getmetrics()[0]
        except Exception:
            return None
        if bb is None:
            return None
        # bb is (left, top, right, bottom) in the font's local box, where
        # the baseline sits at y = ascent_px. Translate to canvas coords
        # with baseline pinned at baseline_y and the character centred.
        ink_w = bb[2] - bb[0]
        x_left = int(round(center_x - ink_w / 2))
        x_right = x_left + int(round(ink_w))
        y_top = int(round(baseline_y - (ascent_px - bb[1])))
        y_bot = int(round(baseline_y + (bb[3] - ascent_px)))
        return (x_left, y_top, x_right, y_bot)

    def _guide_ratio_for_variant(self, v: GlyphVariant) -> float:
        ratio = 1.0
        if self.ref_orig_h > 0:
            target = self.label_target_orig_h.get(v.label)
            if target is None and v.orig_h is not None:
                target = float(v.orig_h)
            if target is not None:
                ratio = float(target) / float(self.ref_orig_h)
        if is_single_lower_ascii(v.label):
            ratio = max(ratio, self.lowercase_floor_ratio)
        return ratio

    def redraw_editor(self) -> None:
        """Fixed-geometry render. The canvas has a fixed baseline line and a
        "line box" that defines the target glyph height; glyphs are scaled
        to fill that box and positioned by their baseline metadata. User
        scale and drag layer on top as simple multiplicative/additive
        adjustments — no apply_transform during render, so geometry isn't
        re-cropped between frames and saves behave predictably."""
        self.editor_canvas.delete("all")
        if self.editor_glyph is None:
            return
        cw = max(10, self.editor_canvas.winfo_width())
        ch = max(10, self.editor_canvas.winfo_height())

        # Background.
        bg = self.checker_bg(cw, ch, s=14)
        bg_photo = ImageTk.PhotoImage(bg)
        self.editor_bg_ref = bg_photo
        self.editor_canvas.create_image(0, 0, image=bg_photo, anchor="nw")

        # Fixed canvas geometry: baseline near lower third, cap line above,
        # descender reserve below. These set the "line box" glyphs fit into.
        center_x = cw // 2
        baseline_y = int(ch * 0.70)
        cap_above_px = int(ch * 0.42)   # cap-to-baseline target
        descender_px = int(ch * 0.20)   # baseline-to-descender reserve
        self.last_baseline_y = baseline_y
        self.last_guide_bbox = None

        v = self.current_variant()

        # Draw the reference font letter with TYPOGRAPHIC baseline aligned
        # to baseline_y (so 'g', 'y', '$', etc. naturally descend below).
        # last_guide_bbox is set from PIL's per-char ink bbox, NOT Tk's
        # canvas.bbox (which returns the font line-box — identical for all
        # characters and useless for per-glyph alignment).
        if self.guide_show_var.get() and v is not None and v.label:
            gsize = max(12, cap_above_px)
            font_obj = tkfont.Font(family=self.guide_family, size=-gsize)
            ascent = font_obj.metrics("ascent")
            if ascent > 0 and ascent != cap_above_px:
                gsize = max(12, int(round(gsize * cap_above_px / max(1, ascent))))
                font_obj = tkfont.Font(family=self.guide_family, size=-gsize)
                ascent = font_obj.metrics("ascent")
            self.editor_canvas.create_text(
                center_x, baseline_y - ascent, text=v.label,
                fill="#bcbcbc",
                font=font_obj, anchor="n",
            )
            # Measure the TRUE ink bbox via PIL; Tk canvas.bbox returns the
            # font line-box height for every letter.
            self.last_guide_bbox = self._guide_ink_bbox_px(
                v.label, gsize, baseline_y, center_x,
            )

        # Auto-fit-by-default: scale this glyph so its HEIGHT matches the
        # guide's rendered bbox. The guide is a typed font letter, so Tk
        # already handles character-class sizing ('o' small, 'H' big, 'g'
        # with descender). Matching the guide bbox makes every loaded glyph
        # land at the typographically-appropriate size without per-glyph
        # setup — the editor becomes "cleanup only".
        #
        # Fallback (no guide drawn): use the library-wide max_orig_h so all
        # glyphs at least share a consistent reference scale.
        target_line_h_px = cap_above_px + descender_px
        if self.last_guide_bbox is not None:
            gy0 = self.last_guide_bbox[1]
            gy1 = self.last_guide_bbox[3]
            guide_h_px = max(1, gy1 - gy0)
            base_scale = guide_h_px / max(1.0, float(self.editor_glyph.height))
        else:
            ref_h = float(self.max_orig_h) if self.max_orig_h > 0 else float(self.editor_glyph.height)
            base_scale = target_line_h_px / max(1.0, ref_h)
        display_scale = base_scale * max(0.05, float(self.glyph_user_scale))
        self.editor_scale = display_scale

        # Glyph position. When a guide is drawn, align the glyph's INK BBOX
        # top to the guide's ink bbox top — this matches the character
        # visually regardless of baseline-measurement noise in the data.
        # Without a guide, fall back to baseline-aligned positioning.
        off_x_px = self.editor_offset_x * display_scale
        off_y_px = self.editor_offset_y * display_scale
        glyph_w_px = max(1, int(round(self.editor_glyph.width * display_scale)))
        glyph_h_px = max(1, int(round(self.editor_glyph.height * display_scale)))
        glyph_left_px = center_x - glyph_w_px / 2 + off_x_px
        if self.last_guide_bbox is not None:
            glyph_top_px = float(self.last_guide_bbox[1]) + off_y_px
        else:
            baseline_local = (
                float(self.editor_baseline)
                if self.editor_baseline is not None
                else float(self.editor_glyph.height)
            )
            glyph_top_px = (
                baseline_y - baseline_local * display_scale + off_y_px
            )

        # Rasterize the glyph AT ITS NATURAL GEOMETRY (no apply_transform).
        raster = rasterize_glyph(
            self.editor_glyph, glyph_w_px, glyph_h_px,
            pad=0, fg=(20, 20, 20, 235),
        )
        photo = ImageTk.PhotoImage(raster)
        self.editor_photo_ref = photo
        self.editor_canvas.create_image(
            int(round(glyph_left_px)), int(round(glyph_top_px)),
            image=photo, anchor="nw",
        )

        # Centerline overlay — base glyph points mapped through the same
        # display transform the rasteriser uses.
        if (self.centerline_show_var.get()
                and self.editor_centerlines
                and self.editor_glyph is not None):
            for stroke in self.editor_centerlines:
                pts = stroke.get("points", [])
                if len(pts) < 2:
                    continue
                coords: list[float] = []
                for gx, gy in pts:
                    coords.append(glyph_left_px + gx * display_scale)
                    coords.append(glyph_top_px + gy * display_scale)
                if stroke.get("closed") and len(coords) >= 4:
                    coords.extend(coords[:2])
                if len(coords) >= 4:
                    self.editor_canvas.create_line(
                        *coords, fill="#dc1e1e", width=2, smooth=False,
                    )

        # Baseline line (dashed) + frame around the glyph.
        self.editor_canvas.create_line(
            20, baseline_y, cw - 20, baseline_y,
            fill="#7d7d7d", dash=(3, 3),
        )
        fx0 = int(round(glyph_left_px))
        fy0 = int(round(glyph_top_px))
        fx1 = fx0 + glyph_w_px
        fy1 = fy0 + glyph_h_px
        self.editor_canvas.create_rectangle(
            fx0, fy0, fx1, fy1, outline="#4b4b4b",
        )

        effective_baseline = (
            (float(self.editor_baseline)
             if self.editor_baseline is not None
             else float(self.editor_glyph.height))
            * float(self.glyph_user_scale) - float(self.editor_offset_y)
        )
        self.sel_offset_var.set(
            f"Offset: ({self.editor_offset_x:+.2f}, {self.editor_offset_y:+.2f}) pt  "
            f"baseline: {effective_baseline:.2f} pt"
        )

    # ── Mouse / keyboard ────────────────────────────────────────────────

    def on_editor_press(self, event: tk.Event) -> None:
        self.drag_last_x = event.x
        self.drag_last_y = event.y

    def on_editor_drag(self, event: tk.Event) -> None:
        if self.editor_glyph is None:
            return
        dx_disp = event.x - self.drag_last_x
        dy_disp = event.y - self.drag_last_y
        self.drag_last_x = event.x
        self.drag_last_y = event.y
        # Convert pixels to glyph units.
        self.editor_offset_x += dx_disp / max(1e-6, self.editor_scale)
        self.editor_offset_y += dy_disp / max(1e-6, self.editor_scale)
        self.redraw_editor()

    def nudge(self, dx: float, dy: float) -> None:
        """Nudge in glyph-local units (points). ~0.2 = ~1 px at 300 DPI render."""
        if self.editor_glyph is None:
            return
        self.editor_offset_x += dx
        self.editor_offset_y += dy
        self.redraw_editor()

    def on_glyph_scale_changed(self, *_args: object) -> None:
        if self._suppress_scale_callback:
            return
        try:
            s = float(self.glyph_scale_var.get())
        except Exception:
            return
        self._set_glyph_scale(s, redraw=True)

    def _set_glyph_scale(self, scale: float, redraw: bool) -> None:
        clamped = min(3.0, max(0.3, float(scale)))
        self.glyph_user_scale = clamped
        self.glyph_scale_display_var.set(f"{clamped:.2f}x")
        self._suppress_scale_callback = True
        try:
            self.glyph_scale_var.set(clamped)
        finally:
            self._suppress_scale_callback = False
        if redraw:
            self.redraw_editor()

    def reset_glyph_scale(self) -> None:
        self._set_glyph_scale(1.0, redraw=True)

    # ── Advance multiplier handlers ─────────────────────────────────────

    def on_advance_mul_changed(self, *_args: object) -> None:
        if self._suppress_advance_callback:
            return
        try:
            v = float(self.glyph_advance_mul_var.get())
        except Exception:
            return
        self._set_advance_mul(v)

    def _set_advance_mul(self, mul: float) -> None:
        clamped = min(1.5, max(0.3, float(mul)))
        self.glyph_advance_mul = clamped
        self.glyph_advance_mul_display_var.set(f"{clamped:.2f}x")
        self._suppress_advance_callback = True
        try:
            self.glyph_advance_mul_var.set(clamped)
        finally:
            self._suppress_advance_callback = False

    def reset_shift(self) -> None:
        if self.editor_glyph is None:
            return
        self.editor_offset_x = 0.0
        self.editor_offset_y = 0.0
        self.redraw_editor()

    # ── Auto-fit ────────────────────────────────────────────────────────

    def auto_fit_current_glyph(self, redraw: bool = True) -> None:
        """Snap the glyph to fill the canvas line box — i.e. reset the
        user scale to 1.0. In the new render model the line box IS the
        fitting target (display_scale already fits editor_glyph.height
        to target_line_h_px when user scale == 1), so "auto-fit" is just
        "clear the user's manual scaling"."""
        if self.editor_glyph is None:
            return
        self.editor_offset_x = 0.0
        self.editor_offset_y = 0.0
        self._set_glyph_scale(1.0, redraw=redraw)

    # ── Save / relabel / delete ─────────────────────────────────────────

    def save_transform(self) -> None:
        """Bake scale + horizontal offset into the SVG geometry; fold the
        vertical offset into the baseline metadata (don't bake it into
        geometry — apply_transform re-crops to content bbox, which would
        silently cancel any pure translation)."""
        v = self.current_variant()
        if v is None or self.editor_glyph is None:
            return

        # Pure metadata write — no geometry mutation, no state reset. The
        # user's drag + slider values are persisted verbatim to index.json;
        # reload reads them back. Guarantees:
        #   • offset numbers stay at what the user set
        #   • glyph doesn't shift visually
        #   • successive saves are idempotent
        # Compose / plotter export read v.user_scale and v.offset_x/y to
        # produce final geometry, so the SVG on disk stays the normalized
        # reference version.
        offset_x_pt = float(self.editor_offset_x)
        offset_y_pt = float(self.editor_offset_y)
        user_scale = float(self.glyph_user_scale)
        advance_mul = float(self.glyph_advance_mul)

        try:
            self._update_variant_metrics(
                v,
                new_orig_h=v.orig_h,
                new_orig_w=v.orig_w,
                offset_x=offset_x_pt,
                offset_y=offset_y_pt,
                user_scale=user_scale,
                advance_mul=advance_mul,
            )
            v.offset_x = offset_x_pt
            v.offset_y = offset_y_pt
            v.user_scale = user_scale
            v.advance_mul = advance_mul
        except Exception as ex:  # noqa: BLE001
            messagebox.showwarning(
                "Metadata update warning",
                f"Could not update index.json:\n{ex}",
            )

        # Keep the plotter sibling SVG in sync with any recent normalization.
        # (Geometry didn't change on this save, but recomputing costs <20ms
        # and guarantees the plotter file exists alongside the outline.)
        try:
            if not self.editor_centerlines:
                self.editor_centerlines = compute_centerlines(self.editor_glyph)
            plotter_path = v.abs_path.with_suffix(".plotter.svg")
            plotter_svg = centerlines_to_svg(
                self.editor_centerlines,
                width=self.editor_glyph.width,
                height=self.editor_glyph.height,
                stroke_px=max(0.05, self.editor_glyph.height * 0.02),
            )
            plotter_path.write_text(plotter_svg, encoding="utf-8")
        except Exception:
            pass

        self.rebuild_grid()

        self.editor_save_status.set(
            f"Saved ✓  (scale {user_scale:.2f}×, offset {offset_x_pt:+.2f}, {offset_y_pt:+.2f} pt)"
        )
        self.root.after(2500, lambda: self.editor_save_status.set(""))

    def _update_variant_metrics(
        self,
        v: GlyphVariant,
        new_orig_h: float | None = None,
        new_orig_w: float | None = None,
        baseline: float | None = None,
        offset_x: float | None = None,
        offset_y: float | None = None,
        user_scale: float | None = None,
        advance_mul: float | None = None,
    ) -> None:
        with open(v.idx_path, encoding="utf-8") as f:
            index = json.load(f)
        glyphs = index.get("glyphs", {})
        entries = list(glyphs.get(v.label, []))
        new_entries = []
        updated = False
        for e in entries:
            if isinstance(e, dict) and str(e.get("path", "")) == v.rel_path and not updated:
                ne = dict(e)
                if new_orig_h is not None:
                    ne["orig_h"] = float(new_orig_h)
                if new_orig_w is not None:
                    ne["orig_w"] = float(new_orig_w)
                if baseline is not None:
                    ne["baseline"] = float(baseline)
                if offset_x is not None:
                    ne["offset_x"] = float(offset_x)
                if offset_y is not None:
                    ne["offset_y"] = float(offset_y)
                if user_scale is not None:
                    ne["user_scale"] = float(user_scale)
                if advance_mul is not None:
                    ne["advance_mul"] = float(advance_mul)
                new_entries.append(ne)
                updated = True
            else:
                new_entries.append(e)
        if not updated:
            raise ValueError("Selected variant not found in index.json")
        glyphs[v.label] = new_entries
        index["glyphs"] = glyphs
        index["version"] = 3
        index["format"] = "svg"
        with open(v.idx_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def _remove_variant_from_index(self, v: GlyphVariant) -> None:
        with open(v.idx_path, encoding="utf-8") as f:
            index = json.load(f)
        glyphs = index.get("glyphs", {})
        entries = list(glyphs.get(v.label, []))
        kept = [e for e in entries
                if not (isinstance(e, dict) and str(e.get("path", "")) == v.rel_path)]
        if kept:
            glyphs[v.label] = kept
        else:
            glyphs.pop(v.label, None)
        index["glyphs"] = glyphs
        with open(v.idx_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def delete_variant(self) -> None:
        v = self.current_variant()
        if v is None:
            return
        if not messagebox.askyesno(
            "Delete variant",
            f"Delete this glyph variant?\n\nLabel: {v.label}\nFile: {v.abs_path}",
        ):
            return
        try:
            if v.abs_path.exists():
                v.abs_path.unlink()
            self._remove_variant_from_index(v)
        except Exception as ex:  # noqa: BLE001
            messagebox.showerror("Delete failed", str(ex))
            return
        self.reload_variants()

    # ── Split glyph ─────────────────────────────────────────────────────

    def _write_new_variant(
        self, lib_path: Path, label: str, new_glyph: "Glyph",
        baseline_ratio: float,
    ) -> None:
        """Create a new variant entry for `label` in `lib_path`. Writes
        the SVG into the appropriate per-label subfolder (auto-numbered
        _N suffix) and appends an index.json entry with orig_h /
        orig_w / baseline / advance_x populated."""
        from svg_glyph import compute_advance_x

        idx_path = lib_path / "index.json"
        with open(idx_path, encoding="utf-8") as f:
            index = json.load(f)
        glyphs = index.setdefault("glyphs", {})
        entries: list[dict] = glyphs.setdefault(label, [])

        label_dir = safe_label_dir(label)
        label_folder = lib_path / label_dir
        label_folder.mkdir(parents=True, exist_ok=True)

        # Next available _N — scan existing entries AND any files left
        # in the folder (in case index and disk got out of sync).
        next_n = 0
        for entry in entries:
            stem = Path(entry.get("path", "")).stem
            if stem.startswith(f"{label_dir}_"):
                try:
                    next_n = max(next_n, int(stem[len(label_dir) + 1:]) + 1)
                except ValueError:
                    pass
        for existing in label_folder.glob(f"{label_dir}_*.svg"):
            stem = existing.stem
            try:
                next_n = max(next_n, int(stem[len(label_dir) + 1:]) + 1)
            except ValueError:
                pass

        filename = f"{label_dir}_{next_n}.svg"
        out_file = label_folder / filename
        new_glyph.save(out_file)

        new_entry: dict = {
            "path": f"{label_dir}/{filename}",
            "orig_h": round(new_glyph.height, 3),
            "orig_w": round(new_glyph.width, 3),
            "baseline": round(new_glyph.height * baseline_ratio, 3),
        }
        try:
            new_entry["advance_x"] = round(compute_advance_x(new_glyph), 3)
        except Exception:
            pass
        entries.append(new_entry)

        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def on_split_glyph_clicked(self) -> None:
        """Open a dialog that bisects the current glyph at a user-chosen
        x-coordinate. Used to fix OCR mis-merges where two letters got
        wrapped into a single variant (e.g. 'il', 'ti', '11')."""
        v = self.current_variant()
        if v is None or self.editor_glyph is None:
            messagebox.showinfo("No glyph",
                                 "Select a glyph to split first.")
            return

        from svg_glyph import split_glyph_at_x, suggest_split_x

        glyph = self.editor_glyph
        orig_baseline = (
            float(v.baseline) if v.baseline is not None
            else glyph.height * 0.9
        )
        baseline_ratio = (
            orig_baseline / glyph.height if glyph.height > 0 else 0.9
        )

        win = tk.Toplevel(self.root)
        win.title(f"Split glyph: {v.label!r}")
        win.transient(self.root)
        win.resizable(False, False)
        win.grab_set()

        # Preview canvas — rasterize the glyph and overlay a draggable
        # cut line. Canvas coords are independent of glyph-local; we
        # keep a scale factor to convert between them.
        canvas_w, canvas_h = 480, 220
        canvas = tk.Canvas(win, width=canvas_w, height=canvas_h,
                            bg="white", highlightthickness=1,
                            highlightbackground="#888")
        canvas.pack(padx=12, pady=(12, 8))

        px_per_pt = min(
            (canvas_w - 20) / max(1.0, glyph.width),
            (canvas_h - 20) / max(1.0, glyph.height),
        )
        draw_w = max(1, int(glyph.width * px_per_pt))
        draw_h = max(1, int(glyph.height * px_per_pt))
        img = rasterize_glyph(glyph, draw_w, draw_h,
                               pad=0, fg=(0, 0, 0, 255))
        photo = ImageTk.PhotoImage(img.convert("RGB"))
        img_x = (canvas_w - draw_w) // 2
        img_y = (canvas_h - draw_h) // 2
        canvas.create_image(img_x, img_y, image=photo, anchor="nw")
        canvas._photo_ref = photo  # prevent GC  # type: ignore[attr-defined]

        cut_var = tk.DoubleVar(value=suggest_split_x(glyph))
        line_id = [canvas.create_line(0, 0, 0, 0,
                                        fill="#e33", width=2,
                                        dash=(4, 3))]

        def _redraw_line(*_args: object) -> None:
            try:
                cut_x_glyph = float(cut_var.get())
            except (ValueError, tk.TclError):
                return
            cx = img_x + cut_x_glyph * px_per_pt
            canvas.coords(line_id[0], cx, 0, cx, canvas_h)
        _redraw_line()
        cut_var.trace_add("write", _redraw_line)

        # Drag the line directly on the canvas.
        def _on_drag(event: tk.Event) -> None:
            x_glyph = (event.x - img_x) / max(1e-6, px_per_pt)
            x_glyph = max(0.0, min(float(glyph.width), x_glyph))
            cut_var.set(x_glyph)
        canvas.bind("<B1-Motion>", _on_drag)
        canvas.bind("<Button-1>", _on_drag)

        # Controls row: slider + Auto.
        ctrl_row = ttk.Frame(win)
        ctrl_row.pack(fill=tk.X, padx=12, pady=(0, 8))
        ttk.Label(ctrl_row, text="Cut at (pt):").pack(side=tk.LEFT)
        ttk.Scale(ctrl_row, from_=0.0, to=float(glyph.width),
                   variable=cut_var,
                   orient="horizontal", length=220,
                   ).pack(side=tk.LEFT, padx=(6, 6), fill=tk.X, expand=True)
        ttk.Button(
            ctrl_row, text="Auto-detect",
            command=lambda: cut_var.set(suggest_split_x(glyph)),
        ).pack(side=tk.LEFT)

        # Labels row.
        lbl_row = ttk.Frame(win)
        lbl_row.pack(fill=tk.X, padx=12, pady=(0, 8))
        ttk.Label(lbl_row, text="Left label:").grid(row=0, column=0,
                                                      sticky="w")
        first_char = v.label[0] if v.label else ""
        last_char = v.label[-1] if len(v.label) > 1 else ""
        left_label_var = tk.StringVar(value=first_char)
        ttk.Entry(lbl_row, textvariable=left_label_var, width=6
                   ).grid(row=0, column=1, padx=(6, 14))
        ttk.Label(lbl_row, text="Right label:").grid(row=0, column=2,
                                                       sticky="w")
        right_label_var = tk.StringVar(value=last_char)
        ttk.Entry(lbl_row, textvariable=right_label_var, width=6
                   ).grid(row=0, column=3, padx=(6, 0))

        # Confirm / Cancel.
        btn_row = ttk.Frame(win)
        btn_row.pack(fill=tk.X, padx=12, pady=(4, 12))

        def _do_split() -> None:
            left_label = left_label_var.get().strip()
            right_label = right_label_var.get().strip()
            if not left_label or not right_label:
                messagebox.showerror(
                    "Labels required",
                    "Enter a label for both halves."
                )
                return
            try:
                cut_x = float(cut_var.get())
            except (ValueError, tk.TclError):
                return
            try:
                left_g, right_g = split_glyph_at_x(glyph, cut_x)
            except Exception as ex:  # noqa: BLE001
                messagebox.showerror("Split failed",
                                      f"{type(ex).__name__}: {ex}")
                return
            if not left_g.subpaths or not right_g.subpaths:
                messagebox.showerror(
                    "Split failed",
                    "One side ended up empty. Move the cut line so "
                    "both sides contain ink.",
                )
                return

            try:
                self._write_new_variant(
                    v.lib_path, left_label, left_g, baseline_ratio,
                )
                self._write_new_variant(
                    v.lib_path, right_label, right_g, baseline_ratio,
                )
                # Remove the original variant's entry + file so it
                # doesn't haunt the library as a duplicate.
                self._remove_variant_from_index(v)
                try:
                    if v.abs_path.exists():
                        v.abs_path.unlink()
                    plotter_twin = v.abs_path.with_suffix(".plotter.svg")
                    if plotter_twin.exists():
                        plotter_twin.unlink()
                except Exception:
                    pass
            except Exception as ex:  # noqa: BLE001
                messagebox.showerror("Split failed",
                                      f"{type(ex).__name__}: {ex}")
                return

            win.destroy()
            self.reload_variants()
            if hasattr(self, "_thumb_cache"):
                self._thumb_cache.clear()
            self.compose_centerline_cache.clear()
            self.rebuild_grid()
            self.editor_save_status.set(
                f"Split into '{left_label}' + '{right_label}'"
            )
            self.root.after(
                3000, lambda: self.editor_save_status.set("")
            )

        ttk.Button(btn_row, text="Cancel",
                    command=win.destroy).pack(side=tk.RIGHT)
        ttk.Button(btn_row, text="Split",
                    command=_do_split
                    ).pack(side=tk.RIGHT, padx=(0, 6))
        win.bind("<Escape>", lambda _e: win.destroy())
        win.bind("<Return>", lambda _e: _do_split())

    def apply_relabel(self) -> None:
        v = self.current_variant()
        if v is None:
            return
        new_label = self.new_label_var.get().strip()
        if not new_label:
            messagebox.showwarning("Invalid label", "Label cannot be empty.")
            return
        if new_label == v.label:
            return
        if not messagebox.askyesno(
            "Relabel variant",
            f"Change label from {v.label!r} to {new_label!r}?\n\n"
            f"This moves the SVG into the new label folder and updates index.json.",
        ):
            return

        try:
            with open(v.idx_path, encoding="utf-8") as f:
                index = json.load(f)
            glyphs = index.get("glyphs", {})
            entries = list(glyphs.get(v.label, []))
            moved_entry = None
            kept_old = []
            for e in entries:
                if (isinstance(e, dict) and str(e.get("path", "")) == v.rel_path
                        and moved_entry is None):
                    moved_entry = e
                else:
                    kept_old.append(e)
            if moved_entry is None:
                messagebox.showerror(
                    "Relabel failed",
                    "Could not find selected variant in index.json.",
                )
                return
            if kept_old:
                glyphs[v.label] = kept_old
            else:
                glyphs.pop(v.label, None)

            old_abs = v.abs_path
            new_dir = v.lib_path / safe_label_dir(new_label)
            new_dir.mkdir(parents=True, exist_ok=True)
            stem = safe_label_dir(new_label)
            suffix = old_abs.suffix or ".svg"
            n = 0
            while True:
                candidate = new_dir / f"{stem}_{n}{suffix}"
                if not candidate.exists():
                    break
                n += 1
            old_abs.rename(candidate)
            new_rel = str(candidate.relative_to(v.lib_path)).replace("\\", "/")

            new_entry = dict(moved_entry)
            new_entry["path"] = new_rel
            glyphs.setdefault(new_label, []).append(new_entry)
            index["glyphs"] = glyphs
            index["version"] = 3
            index["format"] = "svg"
            with open(v.idx_path, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
        except Exception as ex:  # noqa: BLE001
            messagebox.showerror("Relabel failed", str(ex))
            return

        self.reload_variants()
        # Try to keep user on the relabeled variant.
        for i, item in enumerate(self.variants):
            if (item.label == new_label and item.lib_path == v.lib_path
                    and item.rel_path == new_rel):
                self.selected_global_idx = i
                self.load_selected_into_editor()
                break
        self.rebuild_grid()

    def select_next(self) -> None:
        if not self.filtered_indices:
            return
        if self.selected_global_idx is None:
            self.selected_global_idx = self.filtered_indices[0]
        else:
            try:
                cur = self.filtered_indices.index(self.selected_global_idx)
                nxt = (cur + 1) % len(self.filtered_indices)
                self.selected_global_idx = self.filtered_indices[nxt]
            except ValueError:
                self.selected_global_idx = self.filtered_indices[0]
        self.load_selected_into_editor()
        # Selection change only — no need to rebuild the grid.
        self._update_selection_highlight()


# ── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive SVG glyph editor")
    ap.add_argument("--glyphs", action="append", default=None,
                    help="Glyph library folder(s). Repeat and/or comma-separate.")
    args = ap.parse_args()

    if args.glyphs is not None:
        libs = parse_library_paths(args.glyphs)
    else:
        # No explicit --glyphs: use whatever already lives under the
        # Desktop libraries root. If nothing's there yet, start with an
        # empty `default/` so the app has something to point at.
        root = libraries_root()
        root.mkdir(parents=True, exist_ok=True)
        existing = sorted(
            p for p in root.iterdir()
            if p.is_dir() and (p / "index.json").exists()
        )
        if existing:
            libs = [existing[0]]
        else:
            default = root / "default"
            default.mkdir(parents=True, exist_ok=True)
            (default / "index.json").write_text(
                '{"version": 3, "format": "svg", "source": null, "glyphs": {}}',
                encoding="utf-8",
            )
            libs = [default]
    if not libs:
        raise SystemExit("No glyph libraries specified.")

    # Warm the centerline deps (skimage + stroke_graph chain) before the Tk
    # event loop so the first glyph click doesn't stall for ~12s on cold
    # import. ~0.5s at startup is invisible; 12s mid-click is not.
    try:
        import skimage.morphology  # noqa: F401
        import stroke_graph  # noqa: F401
        import graph_cleanup  # noqa: F401
        import stroke_decompose  # noqa: F401
        import stroke_fit  # noqa: F401
    except Exception:
        pass

    root = tk.Tk()
    SvgGlyphEditorApp(root, libs)
    root.mainloop()


if __name__ == "__main__":
    main()
