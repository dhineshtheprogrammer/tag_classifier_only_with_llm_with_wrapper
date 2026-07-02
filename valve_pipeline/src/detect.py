from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    score: float
    source: str          # "template" | "cc"
    template_label: str = field(default="")


# Loads each reference valve image from disk, binarizes it, and returns a dict mapping label names to their binary template arrays.
def load_templates(
    refs_dir: str | Path,
    reference_map: dict[str, str],
) -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    for filename, label in reference_map.items():
        path = Path(refs_dir) / filename
        if not path.exists():
            raise FileNotFoundError(f"Reference image not found: {path}")
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Cannot read reference image: {path}")
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        templates[label] = binary
    return templates


# Finds all candidate valve regions in the binary image by combining template matching and connected-component analysis, then removes overlapping boxes via NMS.
def detect_candidates(
    binary_img: np.ndarray,
    templates: dict[str, np.ndarray],
    config: dict,
    debug: bool = False,
    debug_dir: str | Path | None = None,
    stem: str = "img",
) -> list[Box]:
    det_cfg = config["detection"]
    scales = det_cfg["scales"]
    angles = det_cfg["angles"]
    threshold = det_cfg["match_threshold"]
    nms_iou = det_cfg["nms_iou"]

    assert len(scales) > 0, "detection.scales must not be empty"
    assert len(angles) > 0, "detection.angles must not be empty"

    template_boxes = _template_match_boxes(binary_img, templates, scales, angles, threshold)
    cc_boxes = _cc_boxes(
        binary_img,
        det_cfg["cc_min_area"],
        det_cfg["cc_max_area"],
        det_cfg["cc_aspect_range"],
    )

    all_boxes = template_boxes + cc_boxes

    if debug and debug_dir is not None:
        _save_debug_boxes(
            binary_img,
            all_boxes,
            Path(debug_dir) / f"{stem}_candidates_pre_nms.png",
        )

    merged = non_max_suppression(all_boxes, nms_iou)

    # Fallback border filter: reject candidates whose centre falls within the
    # configured margin fraction of the image edge (catches any border element
    # that the ROI crop in pipeline.py didn't already exclude).
    margin_frac = det_cfg.get("border_margin", 0.0)
    if margin_frac > 0:
        ih, iw = binary_img.shape[:2]
        merged = _filter_border_candidates(merged, ih, iw, margin_frac)

    if debug and debug_dir is not None:
        _save_debug_boxes(
            binary_img,
            merged,
            Path(debug_dir) / f"{stem}_candidates_post_nms.png",
        )

    return merged


# Slices out each detected box region from the original color image with a small padding, returning only crops that are large enough to classify.
def crop_candidates(
    original_img: np.ndarray,
    boxes: list[Box],
    pad: int = 4,
) -> list[tuple[Box, np.ndarray]]:
    H, W = original_img.shape[:2]
    results = []
    for box in boxes:
        x1 = max(0, box.x - pad)
        y1 = max(0, box.y - pad)
        x2 = min(W, box.x + box.w + pad)
        y2 = min(H, box.y + box.h + pad)
        crop = original_img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            continue
        results.append((box, crop))
    return results


# Removes duplicate overlapping boxes by keeping the highest-scoring box and suppressing any other box whose IoU with it exceeds the given threshold.
def non_max_suppression(boxes: list[Box], iou_threshold: float) -> list[Box]:
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
    kept: list[Box] = []
    suppressed = [False] * len(sorted_boxes)
    for i, box in enumerate(sorted_boxes):
        if suppressed[i]:
            continue
        kept.append(box)
        for j in range(i + 1, len(sorted_boxes)):
            if not suppressed[j] and _iou(box, sorted_boxes[j]) > iou_threshold:
                suppressed[j] = True
    return kept


# Calculates the Intersection over Union (IoU) ratio between two boxes to measure how much they overlap.
def _iou(a: Box, b: Box) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = int(a.w) * int(a.h) + int(b.w) * int(b.h) - inter
    return inter / union if union > 0 else 0.0


# Removes candidates whose centre point falls within margin_frac of any image edge, filtering out border-strip elements (grid numbers, title block cells, etc.) that survive ROI cropping.
def _filter_border_candidates(
    boxes: list[Box], img_h: int, img_w: int, margin_frac: float
) -> list[Box]:
    mx = int(img_w * margin_frac)
    my = int(img_h * margin_frac)
    kept = []
    for b in boxes:
        cx = b.x + b.w // 2
        cy = b.y + b.h // 2
        if mx < cx < img_w - mx and my < cy < img_h - my:
            kept.append(b)
    return kept


# Rotates a template image by the given angle while expanding the canvas to prevent any part of the symbol from being clipped.
def _rotate_template(img: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0:
        return img
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += new_w / 2 - cx
    M[1, 2] += new_h / 2 - cy
    return cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# Slides each reference template across the binary image at every combination of scale and angle, creating a Box wherever the match score meets the threshold.
def _template_match_boxes(
    binary_img: np.ndarray,
    templates: dict[str, np.ndarray],
    scales: list[float],
    angles: list[float],
    threshold: float,
) -> list[Box]:
    ih, iw = binary_img.shape[:2]
    boxes: list[Box] = []
    for label, tmpl in templates.items():
        for scale in scales:
            scaled = cv2.resize(
                tmpl, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
            for angle in angles:
                rotated = _rotate_template(scaled, angle)
                th, tw = rotated.shape[:2]
                if th > ih or tw > iw:
                    continue
                result = cv2.matchTemplate(binary_img, rotated, cv2.TM_CCOEFF_NORMED)
                locs = np.where(result >= threshold)
                for y, x in zip(*locs):
                    boxes.append(
                        Box(
                            x=int(x), y=int(y), w=int(tw), h=int(th),
                            score=float(result[y, x]),
                            source="template",
                            template_label=label,
                        )
                    )
    return boxes


# Finds candidate boxes by extracting connected blobs from the binary image and keeping only those whose area and aspect ratio fall within the configured valve size range.
def _cc_boxes(
    binary_img: np.ndarray,
    cc_min_area: int,
    cc_max_area: int,
    cc_aspect_range: list[float],
) -> list[Box]:
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    boxes: list[Box] = []
    for i in range(1, n_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (cc_min_area <= area <= cc_max_area):
            continue
        aspect = w / h if h > 0 else 999.0
        if not (cc_aspect_range[0] <= aspect <= cc_aspect_range[1]):
            continue
        boxes.append(Box(x=x, y=y, w=w, h=h, score=1.0, source="cc"))
    return boxes


# Draws all candidate boxes onto a copy of the binary image (red for template matches, orange for CC blobs) and saves it as a debug PNG.
def _save_debug_boxes(img: np.ndarray, boxes: list[Box], path: Path) -> None:
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for box in boxes:
        color = (0, 0, 255) if box.source == "template" else (255, 128, 0)
        cv2.rectangle(vis, (box.x, box.y), (box.x + box.w, box.y + box.h), color, 1)
    cv2.imwrite(str(path), vis)
