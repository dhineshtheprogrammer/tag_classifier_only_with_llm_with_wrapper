# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Detects and classifies valve symbols in P&ID (Piping & Instrumentation Diagram) schematics. Detection uses classical OpenCV (no ML training); classification uses the OpenAI vision API with few-shot reference images. Outputs an annotated PNG and a structured JSON file.

**Core design constraint:** Detection is intentionally high-recall (over-proposes candidates). False positives are filtered by the LLM returning `"unknown"`. A valve missed in Stage 2 cannot be recovered in Stage 3.

## Running the Pipeline

All commands must be run from inside `valve_pipeline/`:

```bash
# Single image
python -m src.pipeline input/diagram.png

# Single image with debug intermediate images saved to output/debug/
python -m src.pipeline input/diagram.png --debug

# Batch process all images in input/
python -m src.pipeline --batch

# Custom config
python -m src.pipeline input/diagram.png --config custom_config.yaml --debug
```

## Running the Streamlit UI

```bash
cd valve_pipeline
streamlit run app.py
# Open http://localhost:8501
```

## Setup

```bash
cd valve_pipeline
pip install -r requirements.txt
# Add OPENAI_API_KEY=sk-... to valve_pipeline/.env
```

Verify dependencies:
```bash
python -c "import cv2, numpy, openai, PIL, dotenv, yaml; print('All dependencies OK')"
```

## Architecture: 4-Stage Pipeline

```
Input image
  │
  ▼  preprocess.py   Grayscale → Otsu threshold → auto-polarity → denoise → deskew
  │
  ▼  detect.py       Template matching (scale × angle grid) + Connected components → NMS → crop
  │
  ▼  classify.py     Few-shot GPT vision: reference symbols + crop → {label, confidence}
  │
  ▼  assemble.py     Filter unknowns & low-confidence → draw boxes → build JSON records
  │
  ▼  Output: annotated PNG + results JSON
```

`pipeline.py` is the orchestrator and CLI entry point — it chains all four stages, loads config, loads the API key, and writes outputs.

### Stage 1 — `preprocess.py`
Returns a 2D `uint8` binary numpy array. Key step: auto-polarity check flips the image if `mean > 127`, so valve linework is always white on black regardless of schematic background color. Optional deskew via `HoughLinesP` (only applied if skew > 0.5°).

### Stage 2 — `detect.py`
All bounding boxes originate here via two independent methods that are merged and NMS-filtered:
- **Template matching:** For each reference symbol × scale × angle, slides the template over the binary image; every location scoring ≥ `match_threshold` becomes a `Box`.
- **Connected components:** `cv2.connectedComponentsWithStats` filtered by area (`cc_min_area`/`cc_max_area`) and aspect ratio (`cc_aspect_range`). Each surviving blob becomes a `Box` with `score=1.0`.
- NMS (greedy, score-descending IoU suppression) removes duplicates from the merged list.
- `crop_candidates()` slices the **original BGR image** (not binarized) per box with 4px padding.

The `Box` dataclass: `(x, y, w, h, score, source, template_label)` where `source` is `"template"` or `"cc"`.

### Stage 3 — `classify.py`
Builds a single few-shot vision prompt: all reference images (base64, `detail: low`) + the crop → expects JSON `{"label": "...", "confidence": 0.0}`. Runs all crops in parallel via `ThreadPoolExecutor`. Retries on `RateLimitError` with exponential backoff (1s, 2s, 4s, 8s).

**Important:** `temperature` and `max_completion_tokens` are intentionally omitted from API calls. GPT-5 reasoning models reject `temperature=0` and silently return empty content when `max_completion_tokens` is too small (due to internal reasoning tokens).

### Stage 4 — `assemble.py`
Filters out `"unknown"` labels and results below `confidence_floor`, then draws colored rectangles + label text on the original image and builds the output record list.

## Adding a New Valve Type

1. Place a reference PNG in `refs/` (e.g., `check.png`)
2. Register in `config.yaml` under `reference_map`: `check.png: check`
3. Add the label to `VALID_LABELS` in `src/classify.py`
4. Add a BGR color in `_LABEL_COLORS` in `src/assemble.py`

No other changes needed.

## Key Configuration (`config.yaml`)

| Key | Default | Effect |
|---|---|---|
| `model` | `gpt-5-mini-2025-08-07` | OpenAI model for classification |
| `detection.match_threshold` | `0.45` | Lower → more recall, more false positives |
| `detection.nms_iou` | `0.3` | NMS overlap threshold |
| `detection.cc_min_area` | `80` | Min blob area in pixels |
| `classify.confidence_floor` | `0.7` | Drop detections below this |
| `classify.max_workers` | `4` | Parallel OpenAI API calls |

### Tuning: Too many false positives
Raise `match_threshold` (→ 0.55), raise `confidence_floor` (→ 0.8), tighten `cc_aspect_range`, raise `cc_min_area`.

### Tuning: Missed valves
Lower `match_threshold` (→ 0.35), add smaller scales, widen `cc_aspect_range`, lower `confidence_floor`.

### Tuning: Small input images (screenshots/thumbnails)
Lower `match_threshold` to 0.35, lower `cc_min_area` to 30, widen `cc_aspect_range` to `[0.2, 6.0]`.

## Output Format

- **Annotated PNG:** `output/<stem>_annotated.png`
- **JSON:** `output/<stem>_results.json` — `{"schematic": "...", "detections": [{"bbox": [x, y, w, h], "label": "gate", "confidence": 0.91, "detection_source": "template", "match_score": 0.68}]}`
- **Debug images** (with `--debug`): `output/debug/<stem>_preprocessed.png`, `_candidates_pre_nms.png`, `_candidates_post_nms.png`

Validate JSON: `python -m json.tool output/<stem>_results.json`

## Valid Valve Labels

`ball`, `butterfly`, `threeway`, `pinch`, `gate`, `oilpump`, `coriolismeter`, `unknown`
