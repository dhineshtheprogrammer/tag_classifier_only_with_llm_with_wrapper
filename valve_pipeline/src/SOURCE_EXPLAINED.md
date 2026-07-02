# Valve Pipeline — Source Code Explained

## File Overview

### `__init__.py`
Empty — just marks the directory as a Python package.

---

### `pipeline.py` — Orchestrator
The entry point. Calls the other 4 stages in sequence:
1. `preprocess` → binarized image
2. `detect_candidates` → list of `Box` objects
3. `classify_all` → label + confidence for each box
4. `assemble` → annotated image + JSON records

Also handles CLI args (`--batch`, `--debug`), loads config from `config.yaml`, loads the OpenAI API key, and writes the final `.png` and `.json` output files.

---

### `preprocess.py` — Stage 1: Image Cleanup
Takes the raw schematic and returns a clean binary image:
1. **Grayscale** — reads the image in grayscale
2. **Otsu thresholding** (`THRESH_BINARY_INV + THRESH_OTSU`) — auto-picks the best threshold, inverts so symbols are white on black
3. **Auto-polarity fix** — if >50% of pixels end up white, flips it (handles light/dark schematics)
4. **Median blur** — removes salt-and-pepper noise (kernel size from config)
5. **Deskew** — uses Hough line detection to estimate document skew angle, then rotates to correct it if the skew > 0.5°

---

### `detect.py` — Stage 2: Finding Candidates (Core of box creation)

This is where all boxes come from. Two independent methods run and their results are merged:

#### Method 1 — Template Matching (`_template_match_boxes`)
```
For each reference valve symbol (template):
  For each scale (e.g. 0.8x, 1.0x, 1.2x):
    Resize the template
    For each angle (e.g. 0°, 45°, 90°, -45°):
      Rotate the resized template
      Slide it across the binary image using cv2.matchTemplate (TM_CCOEFF_NORMED)
      Every pixel where score >= threshold → create a Box(x, y, w, h, score, source="template")
```
The `w` and `h` of the box come directly from the template's dimensions at that scale/angle.

#### Method 2 — Connected Components (`_cc_boxes`)
```
cv2.connectedComponentsWithStats on the binary image
For each blob (connected region of white pixels):
  Read its bounding box (x, y, w, h) and area from OpenCV stats
  If area is within [cc_min_area, cc_max_area]
  AND aspect ratio (w/h) is within cc_aspect_range:
    Create a Box(x, y, w, h, score=1.0, source="cc")
```
This catches valve symbols the template missed by finding any compact blob.

#### NMS — Non-Max Suppression
Both box lists are combined, then overlapping boxes are removed:
- Sort by score descending
- Keep the highest-scoring box, suppress any other box with IoU > threshold
- IoU (Intersection over Union) is calculated in `_iou()`

#### `crop_candidates`
For each surviving box, slices the **original color image** with 4px padding on each side → returns `(Box, crop_image)` pairs.

---

### `classify.py` — Stage 3: Vision LLM Classification
For each cropped image:
1. Encodes the crop as base64 PNG
2. Sends it to **OpenAI GPT** (model from config) with:
   - All reference valve images embedded in the prompt (base64, `detail: low`)
   - System prompt describing it's a strict P&ID classifier
   - Instruction to reply **only** with `{"label": "...", "confidence": 0.0}`
3. Parses the JSON response, validates label is in `VALID_LABELS`, clamps confidence to [0, 1]
4. Retries up to 4 times on rate limit errors (exponential backoff)
5. All crops run **in parallel** via `ThreadPoolExecutor`

Valid labels: `ball`, `butterfly`, `threeway`, `pinch`, `gate`, `oilpump`, `coriolismeter`, `unknown`

---

### `assemble.py` — Stage 4: Drawing Boxes on the Image
For each `(box, crop)` + classification result pair:
- **Skips** if label is `"unknown"` or confidence < `confidence_floor` (from config)
- Draws a colored `cv2.rectangle` on the original image using the box's `(x, y, w, h)`
- Draws the label text (e.g. `"ball 0.92"`) just above the box with `cv2.putText`
- Appends a record dict with `bbox`, `label`, `confidence`, `detection_source`, `match_score` to the output list

Label colors:

| Label          | Color (BGR)       |
|----------------|-------------------|
| ball           | Green             |
| butterfly      | Orange            |
| threeway       | Blue-orange       |
| pinch          | Purple            |
| gate           | Gold              |
| oilpump        | Dark grey         |
| coriolismeter  | Dark purple       |

---

## Box Lifecycle Summary

```
preprocess.py    → binary image (white symbols on black)
      ↓
detect.py        → template match at N scales × M angles  ──┐
                 → connected component blobs               ──┤ merge
                 → NMS removes duplicates                    ↓
                 → Box(x, y, w, h, score, source)
      ↓
detect.py        → crop_candidates: slice original color image per box (+4px pad)
      ↓
classify.py      → GPT vision: is this crop a valve? which type?
      ↓
assemble.py      → cv2.rectangle + cv2.putText on original image using box coords
                 → write annotated PNG + JSON
```
