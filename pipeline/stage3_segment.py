"""Stage 3 – Segmentation: find ink blobs and assign OCR labels.

Two-phase approach
──────────────────
1. **Find components** – ``cv2.connectedComponentsWithStats`` over the
   binarised image.  Tiny blobs (noise) and oversized blobs (stray marks or
   whole lines) are filtered by area.

2. **Assign labels** – each component is matched to the best-fitting
   :class:`~pipeline.stage2_ocr.CharBox` via IoU.  If IoU is below a
   threshold, a nearest-centroid fallback is used.  Components that cannot
   be matched are labelled ``None`` (noise / unlabeled).

Special cases
─────────────
- **Digraphs**: if two OCR CharBoxes both have high IoU with the same
  component, it is stored as a two-character label (e.g. ``"th"``, ``"ll"``).
- **Noise**: components smaller than *min_area* are dropped before
  label assignment.
"""
from __future__ import annotations

import dataclasses
import math

import cv2
import numpy as np

from .stage2_ocr import CharBox


# ── Public dataclasses ────────────────────────────────────────────────────────

@dataclasses.dataclass
class Component:
    """A single connected-ink blob, before label assignment."""
    comp_id: int
    left:    int
    top:     int
    right:   int
    bottom:  int
    area:    int
    mask:    np.ndarray   # boolean crop of the blob (H×W), aligned to bbox

    @property
    def width(self)  -> int:   return self.right - self.left
    @property
    def height(self) -> int:   return self.bottom - self.top
    @property
    def cx(self)     -> float: return (self.left + self.right) / 2
    @property
    def cy(self)     -> float: return (self.top + self.bottom) / 2


@dataclasses.dataclass
class LabeledComponent:
    """A connected-ink blob that has been assigned a character label."""
    char:   str | None    # None ⇒ unlabeled / noise
    conf:   float         # matching confidence [0, 1]
    left:   int
    top:    int
    right:  int
    bottom: int
    area:   int
    mask:   np.ndarray    # boolean crop aligned to bbox

    @property
    def width(self)  -> int:   return self.right - self.left
    @property
    def height(self) -> int:   return self.bottom - self.top
    @property
    def cx(self)     -> float: return (self.left + self.right) / 2
    @property
    def cy(self)     -> float: return (self.top + self.bottom) / 2

    def __repr__(self) -> str:
        return (
            f"LabeledComponent({self.char!r} conf={self.conf:.2f} "
            f"x={self.left}-{self.right} y={self.top}-{self.bottom} "
            f"area={self.area})"
        )


# ── Public API ────────────────────────────────────────────────────────────────

def find_components(
    binary: np.ndarray,
    min_area: int = 80,
    max_area: int = 0,
) -> list[Component]:
    """Locate every connected ink blob in *binary* (ink = 255).

    Parameters
    ----------
    binary:
        uint8 H×W array where 255 = ink and 0 = paper.
    min_area:
        Blobs with fewer pixels are discarded as noise.
    max_area:
        Blobs with more pixels are discarded (e.g. bleed-through, borders).
        0 = no upper limit.

    Returns
    -------
    Components sorted in reading order (top-to-bottom, left-to-right).
    """
    n_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    comps: list[Component] = []
    for i in range(1, n_labels):   # label 0 = background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue

        x  = int(stats[i, cv2.CC_STAT_LEFT])
        y  = int(stats[i, cv2.CC_STAT_TOP])
        w  = int(stats[i, cv2.CC_STAT_WIDTH])
        ht = int(stats[i, cv2.CC_STAT_HEIGHT])

        mask = (label_map[y : y + ht, x : x + w] == i)
        comps.append(Component(
            comp_id=i,
            left=x, top=y, right=x + w, bottom=y + ht,
            area=area, mask=mask,
        ))

    # Reading order: row-band (top // row_band_px) then left-to-right
    row_band = _estimate_row_band(comps)
    comps.sort(key=lambda c: (c.top // row_band, c.left))
    return comps


def assign_labels(
    components: list[Component],
    char_boxes: list[CharBox],
    iou_thresh: float = 0.10,
    centroid_factor: float = 2.0,
) -> list[LabeledComponent]:
    """Match each connected component to its best OCR :class:`CharBox`.

    Matching strategy
    ------------------
    1. Compute IoU between every (component, char_box) pair.
    2. If the best IoU ≥ *iou_thresh*, that box wins.
       - If a *second* CharBox also has IoU ≥ iou_thresh the component is
         stored as a **digraph** (e.g. ``"th"``).
    3. Otherwise fall back to the nearest unmatched CharBox centroid,
       accepting it only if the distance is within
       ``max(comp.width, comp.height) × centroid_factor``.
    4. Components with no suitable match get ``char = None``.

    Parameters
    ----------
    iou_thresh:
        Minimum IoU to accept a direct overlap match.
    centroid_factor:
        How many "character diameters" away a centroid match is still
        acceptable.
    """
    if not char_boxes:
        return [
            LabeledComponent(
                char=None, conf=0.0,
                left=c.left, top=c.top, right=c.right, bottom=c.bottom,
                area=c.area, mask=c.mask,
            )
            for c in components
        ]

    used_boxes: set[int] = set()
    comp_char: dict[int, tuple[str, float]] = {}

    # ── 1) Global IoU-first one-to-one matching (avoids local greedy swaps) ──
    all_pairs: list[tuple[float, int, int]] = []
    for i, comp in enumerate(components):
        for j, cb in enumerate(char_boxes):
            iou = _iou(comp, cb)
            if iou >= iou_thresh:
                all_pairs.append((iou, i, j))
    all_pairs.sort(reverse=True, key=lambda t: t[0])

    used_comps: set[int] = set()
    for iou, i, j in all_pairs:
        if i in used_comps or j in used_boxes:
            continue
        comp_char[i] = (char_boxes[j].char, iou)
        used_comps.add(i)
        used_boxes.add(j)

    # ── 2) Digraph recovery on still-unmatched components ───────────────────
    for i, comp in enumerate(components):
        if i in comp_char:
            continue
        candidates = [
            (j, _iou(comp, cb))
            for j, cb in enumerate(char_boxes)
            if j not in used_boxes and _iou(comp, cb) >= iou_thresh
        ]
        if len(candidates) < 2:
            continue
        candidates.sort(key=lambda t: t[1], reverse=True)
        j0, iou0 = candidates[0]
        j1, iou1 = candidates[1]
        cb0, cb1 = char_boxes[j0], char_boxes[j1]
        if cb0.cx > cb1.cx:
            j0, j1 = j1, j0
            cb0, cb1 = cb1, cb0
            iou0, iou1 = iou1, iou0
        comp_char[i] = (cb0.char + cb1.char, (iou0 + iou1) / 2)
        used_boxes.add(j0)
        used_boxes.add(j1)

    # ── 3) Centroid fallback for remaining unmatched components ─────────────
    for i, comp in enumerate(components):
        if i in comp_char:
            continue
        max_dist = max(comp.width, comp.height) * centroid_factor
        best_dist_idx, best_dist = -1, math.inf
        for j, cb in enumerate(char_boxes):
            if j in used_boxes:
                continue
            dist = math.hypot(comp.cx - cb.cx, comp.cy - cb.cy)
            if dist < best_dist:
                best_dist, best_dist_idx = dist, j

        if best_dist_idx >= 0 and best_dist <= max_dist:
            conf = max(0.0, 1.0 - best_dist / max_dist)
            comp_char[i] = (char_boxes[best_dist_idx].char, conf)
            used_boxes.add(best_dist_idx)

    labeled: list[LabeledComponent] = []
    for i, comp in enumerate(components):
        if i in comp_char:
            char, conf = comp_char[i]
        else:
            char, conf = None, 0.0
        labeled.append(LabeledComponent(
            char=char, conf=conf,
            left=comp.left, top=comp.top,
            right=comp.right, bottom=comp.bottom,
            area=comp.area, mask=comp.mask,
        ))

    return labeled


def print_segment_summary(components: list[LabeledComponent]) -> None:
    """Print a human-readable summary of segmentation results."""
    total    = len(components)
    labeled  = [c for c in components if c.char is not None]
    unlabeled = total - len(labeled)
    chars    = sorted({c.char for c in labeled})
    print(f"[SEG] {total} component(s) found.")
    print(f"[SEG] {len(labeled)} labeled, {unlabeled} unlabeled/noise.")
    if chars:
        print(f"[SEG] Unique labels ({len(chars)}): {''.join(chars)}")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _iou(comp: Component, cb: CharBox) -> float:
    """Intersection-over-union between a component bbox and a CharBox."""
    ix1 = max(comp.left, cb.left)
    iy1 = max(comp.top,  cb.top)
    ix2 = min(comp.right, cb.right)
    iy2 = min(comp.bottom, cb.bottom)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter  = (ix2 - ix1) * (iy2 - iy1)
    area_a = comp.width * comp.height
    area_b = cb.width   * cb.height
    return inter / (area_a + area_b - inter)


def _estimate_row_band(comps: list[Component]) -> int:
    """Estimate a row-band height for reading-order sort.

    Uses the median component height as a proxy for line height, floored
    at 10 px to avoid degenerate cases.
    """
    if not comps:
        return 20
    heights = sorted(c.height for c in comps)
    median_h = heights[len(heights) // 2]
    return max(10, int(median_h * 0.75))
