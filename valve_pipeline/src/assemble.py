from __future__ import annotations

import cv2
import numpy as np

from .detect import Box

_LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "ball":      (0, 255, 0),
    "butterfly": (0, 128, 255),
    "threeway":  (255, 128, 0),
    "pinch":     (128, 0, 255),
    "gate":      (0, 215, 255),
    "oilpump":   (100,100,100),
    "coriolismeter": (100, 0, 100),
}


# Draws colored bounding boxes and label text onto the original image for every detection that passes the confidence threshold, and returns the annotated image alongside the structured result records.
def assemble(
    original_img: np.ndarray,
    crops: list[tuple[Box, np.ndarray]],
    results: list[dict],
    config: dict,
) -> tuple[np.ndarray, list[dict]]:
    if len(crops) != len(results):
        raise ValueError(
            f"crops ({len(crops)}) and results ({len(results)}) must have the same length"
        )

    confidence_floor = config["classify"]["confidence_floor"]
    annotated = original_img.copy()
    records: list[dict] = []

    for (box, _crop), result in zip(crops, results):
        label = result["label"]
        conf = result["confidence"]

        if label == "unknown" or conf < confidence_floor:
            continue

        color = _LABEL_COLORS.get(label, (200, 200, 200))
        if label not in _LABEL_COLORS:
            print(f"[assemble] Warning: unknown label color for '{label}', using grey")

        cv2.rectangle(
            annotated,
            (box.x, box.y),
            (box.x + box.w, box.y + box.h),
            color, 2,
        )
        text = f"{label} {conf:.2f}"
        cv2.putText(
            annotated, text,
            (box.x, max(0, box.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

        records.append({
            "bbox": [box.x, box.y, box.w, box.h],
            "label": label,
            "confidence": conf,
            "detection_source": box.source,
            "match_score": box.score,
        })

    return annotated, records
