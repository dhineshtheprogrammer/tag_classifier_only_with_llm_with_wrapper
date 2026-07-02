# P&ID Valve Detection & Classification Pipeline

## Objective

Given a P&ID schematic image, detect valve symbols using classical computer vision (no ML training), classify each one via the OpenAI vision API using reference symbol images as few-shot exemplars, and output an annotated image + structured JSON.

---

## Hard Constraints

- No YOLO, no trained detection models. Detection is OpenCV template matching + connected-component analysis only.
- No fine-tuning. Classification is few-shot prompting against the OpenAI API.
- Runs fully local except OpenAI API calls. API key from `OPENAI_API_KEY` env var.
- Detection is high-recall (over-propose). False positives are filtered by classification's `"unknown"` label. Missed valves cannot be recovered.

---

## Tech Stack

| Package | Version |
|---------|---------|
| Python | 3.11+ |
| opencv-python | 4.13.0.92 |
| numpy | 2.4.6 |
| openai | 2.43.0 |
| pillow | 12.2.0 |
| python-dotenv | 1.2.2 |
| pyyaml | 6.0.3 |

---

## Project Structure

```
valve_pipeline/
├── refs/                  # reference symbol PNGs (one per valve type)
├── input/                 # schematics to process
├── output/                # annotated images + JSON (auto-created at runtime)
│   └── debug/             # debug images per stage (auto-created when --debug is passed)
├── config.yaml            # all tunables
├── .env                   # OPENAI_API_KEY (fill in your key)
├── requirements.txt       # pinned dependencies
└── src/
    ├── __init__.py
    ├── preprocess.py      # Stage 1 — image preprocessing
    ├── detect.py          # Stage 2 — candidate detection + cropping
    ├── classify.py        # Stage 3 — OpenAI vision classification
    ├── assemble.py        # Stage 4 — filter, annotate, build records
    └── pipeline.py        # Stage 5 — orchestrator + CLI entry point
```

---

## Adding a New Valve Type

1. Place the reference PNG in `refs/` (e.g., `gate.png`)
2. Add it to `config.yaml` under `reference_map`:
   ```yaml
   reference_map:
     gate.png: gate
   ```
3. Add the label to `VALID_LABELS` in `src/classify.py`:
   ```python
   VALID_LABELS = {"ball", "butterfly", "threeway", "pinch", "gate", "unknown"}
   ```
4. Add a bbox color in `src/assemble.py`:
   ```python
   "gate": (0, 215, 255),
   ```

No other changes required — the pipeline picks up new types automatically.

---

## Configuration (`config.yaml`)

```yaml
model: gpt-5-mini-2025-08-07

detection:
  scales: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
  angles: [0, 45, 90, 135, 180, 225, 270, 315]
  match_threshold: 0.45      # lower to increase recall; raise to reduce false proposals
  nms_iou: 0.3               # non-max suppression overlap threshold
  cc_min_area: 80            # minimum connected-component area (px)
  cc_max_area: 8000          # maximum connected-component area (px)
  cc_aspect_range: [0.3, 4.0]  # width/height ratio filter

classify:
  confidence_floor: 0.7      # drop results below this confidence
  max_workers: 4             # parallel API calls

paths:
  refs_dir: refs
  input_dir: input
  output_dir: output
  debug_dir: output/debug

reference_map:
  ball.png: ball
  butterfly.png: butterfly
  three-way.png: threeway
  pinch.png: pinch
  gate.png: gate
```

---

## Pipeline Stages

### Stage 1 — `preprocess.py`

**Function:** `preprocess(image_path, config, debug=False, debug_dir=None) -> np.ndarray`

1. Loads the schematic as grayscale; raises `FileNotFoundError` on failure.
2. Applies Otsu binary threshold (`THRESH_BINARY_INV + THRESH_OTSU`).
3. **Auto-polarity check** — flips the binary image if `mean > 127` so valve lines always appear as white (255) on black (0), regardless of schematic background color.
4. Median blur denoising (`ksize=3`, configurable via `preprocess.median_ksize` in config).
5. Optional deskew — estimates dominant line angle from `HoughLinesP` on Canny edges; only rotates if skew > 0.5°.
6. Saves `output/debug/<stem>_preprocessed.png` when `debug=True`.

**Returns:** 2D `uint8` numpy array (binary image).

---

### Stage 2 — `detect.py`

**Dataclass:**

```python
@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    score: float
    source: str          # "template" | "cc"
    template_label: str  # which template matched (empty for cc)
```

**Functions:**

#### `load_templates(refs_dir, reference_map) -> dict[str, np.ndarray]`
Loads each reference image as grayscale and binarizes it the same way as the schematic (Otsu + auto-polarity). Raises `FileNotFoundError` if any reference is missing.

#### `detect_candidates(binary_img, templates, config, debug, debug_dir, stem) -> list[Box]`
Runs both detectors, merges results, and applies NMS.

**Detector A — Template Matching:**
- Nested loop: `label → scale → angle`
- Each template is resized then rotated using `_rotate_template()`, which uses an **expanded canvas** to prevent corner clipping at non-orthogonal angles.
- `cv2.matchTemplate(TM_CCOEFF_NORMED)` — all locations scoring ≥ `match_threshold` become candidates.
- Templates larger than the schematic at a given scale/angle are skipped.

**Detector B — Connected Components:**
- `cv2.connectedComponentsWithStats(binary_img, connectivity=8)`
- Filters by `cc_min_area ≤ area ≤ cc_max_area` and `cc_aspect_range` (width/height ratio).
- Each surviving component becomes a candidate with `score=1.0`.

**Merge + NMS:**
- All template and CC boxes are concatenated.
- `non_max_suppression(boxes, iou_threshold)` — greedy score-descending suppression using IoU.

**Debug saves:**
- `<stem>_candidates_pre_nms.png` — all raw boxes (red = template, blue = CC).
- `<stem>_candidates_post_nms.png` — surviving boxes after NMS.

#### `crop_candidates(original_img, boxes, pad=4) -> list[tuple[Box, np.ndarray]]`
Crops from the **original BGR image** (not binarized) with small padding. Crops smaller than 4×4 px are skipped.

---

### Stage 3 — `classify.py`

**Functions:**

#### `build_reference_payload(refs_dir, reference_map) -> list`
Base64-encodes all reference images **once at startup**. Reused across every crop. Uses `"detail": "low"` to reduce token cost.

#### `classify_crop(crop_bgr, reference_payload, config, client) -> dict`
Builds the few-shot vision prompt and calls the OpenAI API.

**Prompt structure** (single user message, mixed content list):

1. System message — instructs the model to be strict: explicitly reject pipe lines, elbows, tees, text, instruments, pumps, tanks, and any non-valve element as `"unknown"`
2. Text: `"Here are the reference valve types:"`
3. Reference images interleaved with label text
4. Text: `"Now classify this unknown crop from a P&ID schematic:"`
5. The crop image (base64 PNG, `detail: low`)
6. Instruction to reply **only** as JSON: `{"label": "<ball|butterfly|threeway|pinch|gate|unknown>", "confidence": <0.0–1.0>}`

**API parameters:** `response_format={"type": "json_object"}`

> **Note:** `temperature` and `max_completion_tokens` are intentionally omitted for compatibility with reasoning models (GPT-5 series). Reasoning models reject `temperature=0` and need hundreds of internal reasoning tokens before producing output — a tight `max_completion_tokens` silently returns empty responses.

**Validation:** label is lowercased and stripped; rejected if not in `VALID_LABELS`; confidence is clamped to `[0, 1]`. Empty response content returns `{"label": "unknown", "confidence": 0.0}`.

#### `classify_all(crops, reference_payload, config, client) -> list[dict]`
Parallel classification using `ThreadPoolExecutor(max_workers)`.

Each call is wrapped in `_call_with_retry()`:
- Retries on `openai.RateLimitError` with exponential backoff: 1s, 2s, 4s, 8s (4 attempts max).
- Non-retryable errors (`APIError`, `JSONDecodeError`) return `{"label": "unknown", "confidence": 0.0}` immediately.
- Results are aligned 1:1 with the input `crops` list.

---

### Stage 4 — `assemble.py`

**Function:** `assemble(original_img, crops, results, config) -> tuple[np.ndarray, list[dict]]`

1. Validates `len(crops) == len(results)`; raises `ValueError` otherwise.
2. Filters out results where `label == "unknown"` or `confidence < confidence_floor`.
3. Draws a colored bounding box and `"label conf"` text on a copy of the original image.
4. Builds a `records` list for JSON output.

**Label colors:**

| Label | Color |
|-------|-------|
| ball | Green |
| butterfly | Blue-orange |
| threeway | Orange |
| pinch | Purple |
| gate | Yellow |

**Record schema:**
```json
{
  "bbox": [x, y, w, h],
  "label": "gate",
  "confidence": 0.92,
  "detection_source": "template",
  "match_score": 0.74
}
```

---

### Stage 5 — `pipeline.py`

**Function:** `run(schematic_path, config_path="config.yaml", debug=False) -> list[dict]`

Chains all stages in order:
1. `load_dotenv()` — validates `OPENAI_API_KEY` is set; raises `EnvironmentError` with a clear message if not.
2. Loads `config.yaml`.
3. Creates `output/` and `output/debug/` directories.
4. Stage 1: `preprocess()`
5. Stage 2: `load_templates()` → `detect_candidates()` → `crop_candidates()`
6. Stage 3: `build_reference_payload()` (once) → `classify_all()`
7. Stage 4: `assemble()`
8. Writes `output/<stem>_annotated.png` and `output/<stem>_results.json`.

**Output JSON format:**
```json
{
  "schematic": "input/diagram.png",
  "detections": [
    {
      "bbox": [x, y, w, h],
      "label": "gate",
      "confidence": 0.91,
      "detection_source": "template",
      "match_score": 0.68
    }
  ]
}
```

---

## CLI Usage

Run from inside the `valve_pipeline/` directory:

```bash
# Single schematic
python -m src.pipeline input/diagram.png

# Single schematic with debug images
python -m src.pipeline input/diagram.png --debug

# All images in input/ directory
python -m src.pipeline --batch

# All images with debug + custom config
python -m src.pipeline --batch --debug --config custom_config.yaml
```

---

## Before First Run

1. Fill in your API key in `.env`:
   ```
   OPENAI_API_KEY=sk-...
   ```
2. Place reference symbol PNGs in `refs/` and register them in `config.yaml` under `reference_map`.
3. Place at least one schematic image in `input/`.
4. Install dependencies (if not already done):
   ```bash
   pip install -r requirements.txt
   ```

---

## Known Model Compatibility (GPT-5 series)

GPT-5 reasoning models (`gpt-5-mini-*`, `gpt-5-*`) have different API constraints from GPT-4o:

| Parameter | GPT-4o | GPT-5 series |
|-----------|--------|--------------|
| `max_tokens` | supported | **not supported** — use `max_completion_tokens` |
| `temperature=0` | supported | **not supported** — only default (1) allowed |
| `max_completion_tokens` (small) | fine | **dangerous** — model uses hundreds of reasoning tokens internally before writing output; a small cap returns empty content silently |

The current `classify.py` omits both `temperature` and `max_completion_tokens` to work correctly across all model families.

---

## Verification Checklist

| Stage | How to verify |
|-------|--------------|
| Dependencies | `python -c "import cv2, numpy, openai, PIL, dotenv, yaml; print('OK')"` |
| preprocess | Open `output/debug/<stem>_preprocessed.png` — valve linework should appear as white blobs on black |
| detect | Open `output/debug/<stem>_candidates_post_nms.png` — every valve on the schematic should have at least one box |
| classify | Run on a known crop; confirm valid JSON label and confidence > 0.7 |
| assemble | Check annotated PNG for tight, correctly labeled boxes; validate JSON with `python -m json.tool output/<stem>_results.json` |
| End-to-end | `python -m src.pipeline input/diagram.png --debug`; count records vs. visible valves |

---

## Tuning Guide

### Too many false positives (pipes/text classified as valves)

1. Raise `classify.confidence_floor` (e.g., `0.7` → `0.75` or `0.8`).
2. Raise `detection.match_threshold` to reduce low-quality template proposals.
3. Tighten `detection.cc_aspect_range` to exclude elongated pipe blobs.
4. Raise `detection.cc_min_area` to exclude tiny noise fragments.

### Too many missed valves (under-detection)

1. Lower `detection.match_threshold` (e.g., `0.45` → `0.35`).
2. Add smaller scales to `detection.scales` (e.g., add `0.2`, `0.25`).
3. Widen `detection.cc_aspect_range`.
4. Lower `classify.confidence_floor` (e.g., `0.7` → `0.6`).

### Small input images (thumbnails / screenshots)

Detection thresholds must be relaxed when the input is much smaller than the reference symbols:
- Lower `match_threshold` to `0.35` or below
- Lower `cc_min_area` to `30`
- Widen `cc_aspect_range` to `[0.2, 6.0]`
- Add small scales `0.3`, `0.4` to cover cases where templates exceed the image height

---

## Upgrade Path

If template matching recall remains insufficient after threshold tuning, the planned upgrade is to add **ORB/SIFT feature matching** as a third candidate source inside `detect.py`. No other stage changes are required — the `Box` dataclass and `non_max_suppression` already support a third `source` value.
