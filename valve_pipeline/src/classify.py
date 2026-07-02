from __future__ import annotations

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import openai

from .detect import Box

VALID_LABELS = {"ball", "butterfly", "threeway", "pinch", "gate", "oilpump", "coriolismeter", "unknown"}


# Reads all reference valve images from disk, encodes them as base64, and builds the list of message content blocks that will be sent to the LLM as visual examples.
def build_reference_payload(
    refs_dir: str | Path,
    reference_map: dict[str, str],
) -> list:
    items = []
    for filename, label in reference_map.items():
        path = Path(refs_dir) / filename
        if not path.exists():
            raise FileNotFoundError(f"Reference image not found: {path}")
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = path.suffix.lstrip(".")
        items.append({"type": "text", "text": f"Reference — {label}:"})
        items.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{ext};base64,{b64}",
                "detail": "low",
            },
        })
    return items


# Sends a single cropped image to the OpenAI vision model alongside the reference valve images and returns the predicted label and confidence as a dict.
def classify_crop(
    crop_bgr: "np.ndarray",
    reference_payload: list,
    config: dict,
    client: openai.OpenAI,
) -> dict:
    try:
        ok, buf = cv2.imencode(".png", crop_bgr)
        if not ok:
            return {"label": "unknown", "confidence": 0.0}
        crop_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception:
        return {"label": "unknown", "confidence": 0.0}

    system_msg = {
        "role": "system",
        "content": (
            "You are a strict P&ID valve symbol classifier. "
            "Your job is to determine if a cropped image from a P&ID schematic "
            "shows one of the known valve types, or something else entirely. "
            "P&ID schematics contain many non-valve elements: straight pipe lines, "
            "pipe elbows, tee junctions, reducers, instruments, pumps, tanks, "
            "text labels, and dimension lines. These must all be classified as 'unknown'. "
            "Additionally, classify as 'unknown' any drawing border elements, including: "
            "grid reference numbers or letters (e.g. row numbers 1–8, column letters A–M), "
            "revision clouds or revision marker boxes, title block cells (project name, "
            "scale, date, drawing number), engineer or approval stamps, scale bars, "
            "legend boxes, and any rectangular frame or box that is part of the "
            "drawing's administrative border rather than the process schematic. "
            "Only classify as a valve type if the crop clearly and unambiguously matches "
            "one of the reference valve symbols. When in doubt, use 'unknown'. "
            "Reply with ONLY a JSON object."
        ),
    }

    valid_labels = "|".join(sorted(VALID_LABELS - {"unknown"}))

    user_content = [
        {"type": "text", "text": "Here are the reference valve types:"},
        *reference_payload,
        {"type": "text", "text": "Now classify this unknown crop from a P&ID schematic:"},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{crop_b64}",
                "detail": "low",
            },
        },
        {
            "type": "text",
            "text": (
                f'\nReply with ONLY valid JSON in this exact format:\n'
                f'{{"label": "<{valid_labels}|unknown>", "confidence": <0.0 to 1.0>}}\n'
                f'Use "unknown" for: pipe lines, elbows, tees, text, instruments, '
                f'pumps, tanks, or anything that is not clearly one of the reference valve symbols.'
            ),
        },
    ]

    response = client.chat.completions.create(
        model=config["model"],
        messages=[system_msg, {"role": "user", "content": user_content}],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    if not raw:
        return {"label": "unknown", "confidence": 0.0}
    result = json.loads(raw)

    label = str(result.get("label", "unknown")).strip().lower()
    if label not in VALID_LABELS:
        label = "unknown"
    confidence = float(result.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    return {"label": label, "confidence": confidence}


# Wraps classify_crop with exponential-backoff retry logic for rate limit errors, returning an "unknown" result if all attempts fail.
def _call_with_retry(
    crop_bgr: "np.ndarray",
    reference_payload: list,
    config: dict,
    client: openai.OpenAI,
    max_retries: int = 4,
) -> dict:
    for attempt in range(max_retries):
        try:
            return classify_crop(crop_bgr, reference_payload, config, client)
        except openai.RateLimitError:
            wait = 2 ** attempt
            print(f"[classify] Rate limit hit — retrying in {wait}s (attempt {attempt + 1})")
            time.sleep(wait)
        except openai.APIError as e:
            print(f"[classify] API error (non-retryable): {e}")
            break
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[classify] Parse error: {e}")
            break
    return {"label": "unknown", "confidence": 0.0}


# Classifies all cropped regions in parallel using a thread pool, preserving their original order, and returns one result dict per crop.
def classify_all(
    crops: list[tuple[Box, "np.ndarray"]],
    reference_payload: list,
    config: dict,
    client: openai.OpenAI,
) -> list[dict]:
    if not crops:
        return []

    max_workers = config["classify"]["max_workers"]
    results: list[dict | None] = [None] * len(crops)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_call_with_retry, crop, reference_payload, config, client): idx
            for idx, (_box, crop) in enumerate(crops)
        }
        done = 0
        total = len(crops)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"[classify] Unhandled error at index {idx}: {e}")
                results[idx] = {"label": "unknown", "confidence": 0.0}
            done += 1
            print(f"[classify] {done}/{total} crops classified", end="\r")

    print()
    return [r if r is not None else {"label": "unknown", "confidence": 0.0} for r in results]
