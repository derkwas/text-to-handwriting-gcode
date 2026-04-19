#!/usr/bin/env python3
"""Interactive SVG-glyph editor (parallel to glyph_editor_app.py).

Targets the vector glyph library produced by pdf_to_vector_glyphs.py —
each glyph is an SVG file of cubic-Bezier subpaths in PDF-point space.

Features
--------
- Load a glyph library (default: glyphs_vector)
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
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from svg_glyph import (
    Glyph,
    compute_centerlines,
    rasterize_glyph,
    transform_centerline_polyline,
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


def parse_library_paths(raw_values: list[str]) -> list[Path]:
    base_dir = Path(__file__).resolve().parent
    out: list[Path] = []
    seen: set[str] = set()
    for raw in raw_values:
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            p = Path(part)
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            k = str(p)
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
    return out


def load_variants(library_paths: list[Path]) -> list[GlyphVariant]:
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
        glyphs = index.get("glyphs", {})
        for label, entries in glyphs.items():
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                rel = str(entry.get("path", ""))
                if not rel:
                    continue
                variants.append(GlyphVariant(
                    label=label,
                    lib_path=lib,
                    idx_path=idx_path,
                    rel_path=rel,
                    abs_path=lib / rel,
                    orig_h=entry.get("orig_h"),
                    orig_w=entry.get("orig_w"),
                ))
    variants.sort(key=lambda v: (v.label, str(v.lib_path), v.rel_path))
    return variants


def is_single_lower_ascii(label: str) -> bool:
    return len(label) == 1 and "a" <= label <= "z"


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
        self.editor_glyph: Glyph | None = None
        self.editor_centerlines: list[dict] = []  # base-glyph coord space
        self.editor_offset_x = 0.0   # translation in glyph-local units
        self.editor_offset_y = 0.0
        self.glyph_user_scale = 1.0
        self.editor_scale = 1.0      # display scale (glyph-units -> canvas px)
        self.drag_last_x = 0
        self.drag_last_y = 0
        self.pending_auto_fit = False
        self._suppress_scale_callback = False
        self.last_guide_bbox: tuple[int, int, int, int] | None = None
        self.last_baseline_y = 0

        self.ref_orig_h = 8.0
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
        self.auto_fit_var = tk.BooleanVar(value=True)
        self.centerline_show_var = tk.BooleanVar(value=True)
        self.glyph_scale_var = tk.DoubleVar(value=1.0)
        self.glyph_scale_display_var = tk.StringVar(value="1.00x")

        self._build_ui()
        self.reload_variants()

    # ── UI scaffolding ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=3)
        paned.add(right, weight=2)

        top = ttk.Frame(left)
        top.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(top, text="Libraries:").pack(side=tk.LEFT)
        self.libs_var = tk.StringVar(value=",".join(str(p) for p in self.library_paths))
        self.libs_entry = ttk.Entry(top, textvariable=self.libs_var, width=70)
        self.libs_entry.pack(side=tk.LEFT, padx=(6, 8), fill=tk.X, expand=True)
        ttk.Button(top, text="Reload", command=self.on_reload_clicked).pack(side=tk.LEFT)
        ttk.Label(top, text="Filter:").pack(side=tk.LEFT, padx=(12, 4))
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", lambda *_: self.apply_filter())
        ttk.Entry(top, textvariable=self.filter_var, width=18).pack(side=tk.LEFT)
        self.count_var = tk.StringVar(value="0 glyphs")
        ttk.Label(top, textvariable=self.count_var).pack(side=tk.LEFT, padx=(12, 0))

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
        self.editor_canvas = tk.Canvas(editor_group, width=520, height=520,
                                        bg="#d9d9d9", highlightthickness=1)
        self.editor_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.editor_canvas.bind("<Button-1>", self.on_editor_press)
        self.editor_canvas.bind("<B1-Motion>", self.on_editor_drag)
        self.editor_canvas.bind("<Configure>", lambda _e: self.redraw_editor())

        guide_controls = ttk.Frame(editor_group)
        guide_controls.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Checkbutton(guide_controls, text="Show font guide",
                        variable=self.guide_show_var,
                        command=self.redraw_editor).pack(side=tk.LEFT)
        ttk.Checkbutton(guide_controls, text="Auto-fit glyph to guide",
                        variable=self.auto_fit_var).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Checkbutton(guide_controls, text="Show centerline",
                        variable=self.centerline_show_var,
                        command=self.redraw_editor).pack(side=tk.LEFT, padx=(10, 0))

        scale_controls = ttk.Frame(editor_group)
        scale_controls.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Label(scale_controls, text="Glyph scale:").pack(side=tk.LEFT)
        ttk.Scale(scale_controls, from_=0.3, to=3.0,
                  variable=self.glyph_scale_var,
                  orient=tk.HORIZONTAL, length=220,
                  command=self.on_glyph_scale_changed).pack(side=tk.LEFT, padx=(8, 8))
        ttk.Label(scale_controls, textvariable=self.glyph_scale_display_var,
                  width=6).pack(side=tk.LEFT)
        self.glyph_scale_var.trace_add("write",
                                        lambda *_: self.on_glyph_scale_changed())
        ttk.Button(scale_controls, text="Reset Scale",
                   command=self.reset_glyph_scale).pack(side=tk.LEFT)
        ttk.Button(scale_controls, text="Auto Fit Now",
                   command=self.auto_fit_current_glyph).pack(side=tk.LEFT, padx=(8, 0))

        buttons = ttk.Frame(editor_group)
        buttons.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(buttons, text="Save Transform",
                   command=self.save_transform).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Reset Shift",
                   command=self.reset_shift).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(buttons, text="Delete Variant",
                   command=self.delete_variant).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(buttons, text="Next", command=self.select_next).pack(
            side=tk.LEFT, padx=(8, 0))

        ttk.Label(editor_group, text=(
            "Tip: drag to move glyph. Arrow keys nudge; Shift+Arrow nudges by 5x. "
            "Scale slider resizes the glyph around its center."
        )).pack(anchor="w", padx=8, pady=(0, 8))

        # Key bindings.
        self.root.bind("<Left>", lambda e: self.nudge(-0.2, 0))
        self.root.bind("<Right>", lambda e: self.nudge(0.2, 0))
        self.root.bind("<Up>", lambda e: self.nudge(0, -0.2))
        self.root.bind("<Down>", lambda e: self.nudge(0, 0.2))
        self.root.bind("<Shift-Left>", lambda e: self.nudge(-1.0, 0))
        self.root.bind("<Shift-Right>", lambda e: self.nudge(1.0, 0))
        self.root.bind("<Shift-Up>", lambda e: self.nudge(0, -1.0))
        self.root.bind("<Shift-Down>", lambda e: self.nudge(0, 1.0))

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

    def reload_variants(self) -> None:
        self.variants = load_variants(self.library_paths)

        all_orig_h = [v.orig_h for v in self.variants
                      if v.orig_h is not None]
        self.ref_orig_h = float(statistics.median(all_orig_h)) if all_orig_h else 8.0
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

    def make_thumb(self, variant: GlyphVariant, selected: bool) -> ImageTk.PhotoImage:
        canvas = self.checker_bg(self.thumb_box, self.thumb_box)
        if variant.abs_path.exists():
            try:
                g = Glyph.from_file(variant.abs_path)
                img = rasterize_glyph(g, self.thumb_box, self.thumb_box, pad=6)
                canvas.paste(img.convert("RGB"), mask=img.split()[3])
            except Exception:
                miss = Image.new("RGBA", (self.thumb_box, self.thumb_box), (255, 0, 0, 55))
                canvas = Image.alpha_composite(canvas.convert("RGBA"), miss).convert("RGB")
        else:
            miss = Image.new("RGBA", (self.thumb_box, self.thumb_box), (255, 0, 0, 55))
            canvas = Image.alpha_composite(canvas.convert("RGBA"), miss).convert("RGB")

        # Border
        out = canvas.convert("RGBA")
        bcol = (30, 110, 240, 255) if selected else (150, 150, 150, 255)
        border = Image.new("RGBA", out.size, (0, 0, 0, 0))
        bpx = border.load()
        for x in range(self.thumb_box):
            bpx[x, 0] = bcol
            bpx[x, self.thumb_box - 1] = bcol
        for y in range(self.thumb_box):
            bpx[0, y] = bcol
            bpx[self.thumb_box - 1, y] = bcol
        out.alpha_composite(border)
        return ImageTk.PhotoImage(out.convert("RGB"))

    def rebuild_grid(self) -> None:
        self.grid_canvas.delete("all")
        self.grid_photo_refs.clear()

        cols = max(1, self.grid_cols)
        x_pad = 10
        y_pad = 10
        for n, global_idx in enumerate(self.filtered_indices):
            row, col = divmod(n, cols)
            x0 = x_pad + col * self.cell_w
            y0 = y_pad + row * self.cell_h
            v = self.variants[global_idx]
            selected = self.selected_global_idx == global_idx
            photo = self.make_thumb(v, selected)
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
        self.rebuild_grid()

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

        self.editor_offset_x = 0.0
        self.editor_offset_y = 0.0
        self._set_glyph_scale(1.0, redraw=False)
        self.pending_auto_fit = bool(self.auto_fit_var.get())

        self.sel_label_var.set(f"Label: {v.label}")
        self.sel_lib_var.set(f"Library: {v.lib_path}")
        self.sel_path_var.set(f"Path: {v.rel_path}")
        self.sel_size_var.set(
            f"Size: {self.editor_glyph.width:.2f} x {self.editor_glyph.height:.2f} pt")
        self.sel_offset_var.set("Offset: (0.00, 0.00)")
        self.new_label_var.set(v.label)

        self.redraw_editor()

    # ── Editor rendering ────────────────────────────────────────────────

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

    def _glyph_working_view(self) -> Glyph | None:
        """Return the current editor_glyph with user_scale and offset applied,
        ready for rendering. None if nothing loaded."""
        if self.editor_glyph is None:
            return None
        g = self.editor_glyph
        if abs(self.glyph_user_scale - 1.0) > 1e-6:
            g = g.apply_transform(sx=self.glyph_user_scale)
        if abs(self.editor_offset_x) > 1e-9 or abs(self.editor_offset_y) > 1e-9:
            g = g.translate_only(self.editor_offset_x, self.editor_offset_y)
        return g

    def redraw_editor(self) -> None:
        self.editor_canvas.delete("all")
        if self.editor_glyph is None:
            return
        cw = max(10, self.editor_canvas.winfo_width())
        ch = max(10, self.editor_canvas.winfo_height())

        # Checker background.
        bg = self.checker_bg(cw, ch, s=14)
        bg_photo = ImageTk.PhotoImage(bg)
        self.editor_photo_ref = bg_photo
        self.editor_canvas.create_image(0, 0, image=bg_photo, anchor="nw")

        center_x = cw // 2
        baseline_y = int(ch * 0.72)
        self.last_baseline_y = baseline_y
        self.last_guide_bbox = None

        v = self.current_variant()
        max_h = max(40, ch - 120)
        avail_above_baseline = max(40, baseline_y - 10)

        # Choose an effective ref_h that fits BOTH the glyph and the guide.
        ref_h = max(72, int(ch * self.guide_preview_height_ratio))
        glyph_ratio = 1.0
        guide_ratio = 1.0
        if v is not None and v.orig_h is not None and self.ref_orig_h > 0:
            glyph_ratio = float(v.orig_h) / float(self.ref_orig_h)
            guide_ratio = self._guide_ratio_for_variant(v)
            scale_for_user = max(1.0, float(self.glyph_user_scale))
            needed = max(glyph_ratio * scale_for_user, guide_ratio, 1.0)
            if ref_h * needed > avail_above_baseline:
                ref_h = max(40, int(round(avail_above_baseline / needed)))

        # Draw the reference font letter.
        if self.guide_show_var.get() and v is not None and v.label:
            target_h = max(24, int(round(ref_h * guide_ratio)))
            gsize = max(12, target_h)
            gid = self.editor_canvas.create_text(
                center_x, baseline_y, text=v.label, fill="#9a9a9a",
                font=(self.guide_family, gsize), anchor="s",
            )
            bbox = self.editor_canvas.bbox(gid)
            if bbox is not None:
                gh = max(1, bbox[3] - bbox[1])
                refined = max(12, int(round(gsize * target_h / gh)))
                self.editor_canvas.itemconfigure(
                    gid, font=(self.guide_family, refined))
                self.last_guide_bbox = self.editor_canvas.bbox(gid)

        # Display-scale the glyph so its height lands at ref_h * glyph_ratio.
        # (Matches the raster editor's invariant: on save the glyph stays the
        # same visible size relative to the guide.)
        g_view = self._glyph_working_view()
        if g_view is None:
            return

        if v is not None and v.orig_h is not None and self.ref_orig_h > 0:
            target_display_h = ref_h * glyph_ratio
            self.editor_scale = target_display_h / max(1e-6, self.editor_glyph.height)
        else:
            self.editor_scale = (
                (ch - 120) / max(1.0, self.editor_glyph.height or 1.0)
            )
        self.editor_scale = max(0.05, min(self.editor_scale, 400.0))

        # Pending auto-fit executes once per load, using the guide bbox from
        # the draw we just did.
        if self.pending_auto_fit and self.auto_fit_var.get():
            self.pending_auto_fit = False
            self.auto_fit_current_glyph(redraw=False)
            g_view = self._glyph_working_view()

        # Rasterize the current glyph view at the display size and paste.
        disp_w = max(1, int(round(g_view.width * self.editor_scale)))
        disp_h = max(1, int(round(g_view.height * self.editor_scale)))
        raster = rasterize_glyph(g_view, disp_w, disp_h, pad=0,
                                 fg=(20, 20, 20, 235))
        px = center_x - (disp_w // 2)
        py = baseline_y - disp_h
        photo = ImageTk.PhotoImage(raster)
        self.editor_photo_ref = photo  # keep ref to both bg & glyph
        self.editor_canvas.create_image(px, py, image=photo, anchor="nw")

        # Centerline overlay (medial axis of the filled glyph).
        if (self.centerline_show_var.get()
                and self.editor_centerlines
                and self.editor_glyph is not None):
            for stroke in self.editor_centerlines:
                pts = stroke.get("points", [])
                if len(pts) < 2:
                    continue
                # Base glyph -> g_view space (scale + recrop + translate).
                gpts = transform_centerline_polyline(
                    pts, self.editor_glyph,
                    self.glyph_user_scale,
                    self.editor_offset_x, self.editor_offset_y,
                )
                if stroke.get("closed") and gpts and gpts[0] != gpts[-1]:
                    gpts = list(gpts) + [gpts[0]]
                coords: list[float] = []
                for gx, gy in gpts:
                    coords.append(px + gx * self.editor_scale)
                    coords.append(py + gy * self.editor_scale)
                if len(coords) >= 4:
                    self.editor_canvas.create_line(
                        *coords, fill="#dc1e1e", width=2, smooth=False,
                    )

        # Frame + baseline.
        fx0 = px
        fy0 = py
        fx1 = px + disp_w
        fy1 = baseline_y
        if self.last_guide_bbox is not None:
            gx0, gy0, gx1, gy1 = self.last_guide_bbox
            fx0 = min(fx0, gx0)
            fy0 = min(fy0, gy0)
            fx1 = max(fx1, gx1)
            fy1 = max(fy1, gy1)
        self.editor_canvas.create_rectangle(fx0, fy0, fx1, fy1, outline="#4b4b4b")
        self.editor_canvas.create_line(20, baseline_y, cw - 20, baseline_y,
                                        fill="#7d7d7d", dash=(3, 3))

        self.sel_offset_var.set(
            f"Offset: ({self.editor_offset_x:.2f}, {self.editor_offset_y:.2f})"
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

    def reset_shift(self) -> None:
        if self.editor_glyph is None:
            return
        self.editor_offset_x = 0.0
        self.editor_offset_y = 0.0
        self.redraw_editor()

    # ── Auto-fit ────────────────────────────────────────────────────────

    def auto_fit_current_glyph(self, redraw: bool = True) -> None:
        if self.editor_glyph is None or self.last_guide_bbox is None:
            return
        g_top = self.last_guide_bbox[1]
        g_bot = self.last_guide_bbox[3]
        target_h_px = max(1, g_bot - g_top)
        # Display glyph height = editor_glyph.height * user_scale * editor_scale.
        # Solve for user_scale that brings display height == target_h_px.
        cur_display_h = self.editor_glyph.height * self.editor_scale
        if cur_display_h <= 1e-6:
            return
        needed = target_h_px / cur_display_h
        self._set_glyph_scale(needed, redraw=False)
        # Center the glyph's bbox around the guide bbox (y).
        # After setting user scale, the raster is centered on baseline_y.
        # Additional fine alignment isn't as useful for SVG since the paths
        # already live in tight bbox coordinates; leave offset at (0,0).
        if redraw:
            self.redraw_editor()

    # ── Save / relabel / delete ─────────────────────────────────────────

    def save_transform(self) -> None:
        v = self.current_variant()
        if v is None or self.editor_glyph is None:
            return

        applied_scale = float(self.glyph_user_scale)
        transformed = self.editor_glyph.apply_transform(
            tx=self.editor_offset_x,
            ty=self.editor_offset_y,
            sx=applied_scale,
        )
        try:
            transformed.save(v.abs_path)
        except Exception as ex:  # noqa: BLE001
            messagebox.showerror("Save failed", str(ex))
            return

        new_orig_h = round(transformed.height, 3)
        new_orig_w = round(transformed.width, 3)
        try:
            self._update_variant_metrics(v, new_orig_h, new_orig_w)
            v.orig_h = new_orig_h
            v.orig_w = new_orig_w
        except Exception as ex:  # noqa: BLE001
            messagebox.showwarning(
                "Metadata update warning",
                f"Saved SVG but could not update index.json:\n{ex}",
            )

        # Reset in-memory transform state to the new baseline.
        self.editor_glyph = transformed
        try:
            self.editor_centerlines = compute_centerlines(self.editor_glyph)
        except Exception:
            self.editor_centerlines = []
        self.editor_offset_x = 0.0
        self.editor_offset_y = 0.0
        self._set_glyph_scale(1.0, redraw=False)
        self.pending_auto_fit = bool(self.auto_fit_var.get())
        self.sel_size_var.set(f"Size: {transformed.width:.2f} x {transformed.height:.2f} pt")
        self.redraw_editor()
        self.rebuild_grid()

    def _update_variant_metrics(
        self, v: GlyphVariant, new_orig_h: float, new_orig_w: float,
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
                ne["orig_h"] = float(new_orig_h)
                ne["orig_w"] = float(new_orig_w)
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
        self.rebuild_grid()


# ── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive SVG glyph editor")
    ap.add_argument("--glyphs", action="append", default=None,
                    help="Glyph library folder(s). Repeat and/or comma-separate.")
    args = ap.parse_args()

    raw = args.glyphs if args.glyphs is not None else ["glyphs_vector"]
    libs = parse_library_paths(raw)
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
