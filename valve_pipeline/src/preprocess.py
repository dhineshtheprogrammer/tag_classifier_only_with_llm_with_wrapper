from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# Detects the inner drawing content area by finding the longest horizontal and vertical Hough lines that form an inward-facing rectangle, excluding the border strips that contain grid numbers, title blocks, and revision markers.
def detect_drawing_roi(gray_img: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Returns (x, y, w, h) of the inner drawing area, or None if detection fails.
    The returned rect excludes the outer border strips (grid refs, title block, etc.).
    """
    h, w = gray_img.shape[:2]

    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    edges = cv2.Canny(binary, 50, 150)

    min_dim = min(h, w)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=int(min_dim * 0.25),
        minLineLength=int(min_dim * 0.5),
        maxLineGap=30,
    )
    if lines is None:
        return None

    h_positions: list[int] = []
    v_positions: list[int] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if abs(y2 - y1) < abs(x2 - x1) * 0.1:   # mostly horizontal
            h_positions.append((y1 + y2) // 2)
        elif abs(x2 - x1) < abs(y2 - y1) * 0.1:  # mostly vertical
            v_positions.append((x1 + x2) // 2)

    if not h_positions or not v_positions:
        return None

    def _cluster(positions: list[int], gap: int = 15) -> list[int]:
        positions = sorted(set(positions))
        clusters: list[list[int]] = [[positions[0]]]
        for p in positions[1:]:
            if p - clusters[-1][-1] <= gap:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        return [int(np.median(c)) for c in clusters]

    h_lines = _cluster(h_positions)
    v_lines = _cluster(v_positions)

    # Discard lines sitting on the very image border (within 1%)
    edge_h, edge_w = int(h * 0.01), int(w * 0.01)
    h_inner = [y for y in h_lines if edge_h < y < h - edge_h]
    v_inner = [x for x in v_lines if edge_w < x < w - edge_w]
    if not h_inner or not v_inner:
        return None

    # Pick the innermost line from each edge quadrant
    top_cands = sorted([y for y in h_inner if y < h * 0.25])
    bot_cands = sorted([y for y in h_inner if y > h * 0.75], reverse=True)
    left_cands = sorted([x for x in v_inner if x < w * 0.25])
    right_cands = sorted([x for x in v_inner if x > w * 0.75], reverse=True)

    top   = top_cands[-1]   if top_cands   else edge_h
    bot   = bot_cands[-1]   if bot_cands   else h - edge_h
    left  = left_cands[-1]  if left_cands  else edge_w
    right = right_cands[-1] if right_cands else w - edge_w

    if right <= left or bot <= top:
        return None

    # Sanity check: ROI must cover at least 50% of image area
    if (right - left) * (bot - top) < 0.5 * h * w:
        return None

    # Step 1px inside the border line itself
    return (left + 1, top + 1, right - left - 2, bot - top - 2)


# Converts a raw schematic image to a clean binary image (white symbols on black) by applying Otsu thresholding, median blur denoising, and optional deskew correction.
def preprocess(
    image_path: str | Path,
    config: dict,
    debug: bool = False,
    debug_dir: str | Path | None = None,
) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    if img.shape[0] < 100 or img.shape[1] < 100:
        print(f"[preprocess] Warning: very small image {img.shape} — template matching may find nothing")

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Auto-polarity: valve lines should be white (255) on black (0)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    ksize = config.get("preprocess", {}).get("median_ksize", 3)
    denoised = cv2.medianBlur(binary, ksize)

    if config.get("preprocess", {}).get("deskew", True):
        denoised = _deskew(denoised)

    if debug and debug_dir is not None:
        stem = Path(image_path).stem
        out = Path(debug_dir) / f"{stem}_preprocessed.png"
        cv2.imwrite(str(out), denoised)

    return denoised


# Estimates the document skew angle in degrees by detecting horizontal lines via Hough transform and taking the median of their angles.
def _estimate_skew(binary: np.ndarray) -> float:
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10
    )
    if lines is None:
        return 0.0
    angles = [
        np.degrees(np.arctan2(y2 - y1, x2 - x1))
        for x1, y1, x2, y2 in lines[:, 0]
    ]
    horiz = [a for a in angles if abs(a) < 45]
    return float(np.median(horiz)) if horiz else 0.0


# Rotates the binary image to correct its skew angle; returns the image unchanged if the skew is less than 0.5 degrees.
def _deskew(binary: np.ndarray) -> np.ndarray:
    skew = _estimate_skew(binary)
    if abs(skew) < 0.5:
        return binary
    h, w = binary.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), skew, 1.0)
    return cv2.warpAffine(
        binary, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0
    )
