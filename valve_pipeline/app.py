from __future__ import annotations

import io
import json
import os
import sys
import contextlib
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_DIR))

st.set_page_config(
    page_title="P&ID Valve Classifier",
    layout="wide",
)

st.title("P&ID Valve Classifier")
st.caption("Upload a P&ID schematic to detect and classify valve symbols using computer vision + GPT vision.")

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-… (leave blank to use .env)",
        help="Overrides the key in valve_pipeline/.env for this session.",
    )

    st.divider()
    st.subheader("Detection")
    match_threshold = st.slider("Match threshold", 0.1, 1.0, 0.45, 0.05,
                                help="Minimum template-match score to keep a candidate.")
    nms_iou = st.slider("NMS IoU", 0.1, 1.0, 0.3, 0.05,
                        help="Overlap threshold for non-maximum suppression.")

    st.subheader("Classification")
    confidence_floor = st.slider("Confidence floor", 0.0, 1.0, 0.7, 0.05,
                                 help="Discard detections below this confidence.")

    st.divider()
    debug_mode = st.checkbox("Debug mode", value=False,
                             help="Show intermediate pipeline images (preprocessing, pre/post NMS).")

# ── Main layout ────────────────────────────────────────────────────────────────

upload_col, result_col = st.columns(2, gap="large")

with upload_col:
    st.subheader("Input")
    uploaded = st.file_uploader(
        "Upload schematic",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        label_visibility="collapsed",
    )
    if uploaded:
        st.image(uploaded, caption=uploaded.name, use_container_width=True)

run_btn = st.button("Run Pipeline", type="primary", disabled=uploaded is None)

# ── Pipeline execution ─────────────────────────────────────────────────────────

if run_btn and uploaded:
    # Persist uploaded file to input/
    input_dir = PIPELINE_DIR / "input"
    input_dir.mkdir(exist_ok=True)
    input_path = input_dir / uploaded.name
    input_path.write_bytes(uploaded.getbuffer())

    # Optionally inject API key
    if api_key_input.strip():
        os.environ["OPENAI_API_KEY"] = api_key_input.strip()

    # Patch config with sidebar values (write a temporary config)
    import yaml

    config_path = PIPELINE_DIR / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["detection"]["match_threshold"] = match_threshold
    config["detection"]["nms_iou"] = nms_iou
    config["classify"]["confidence_floor"] = confidence_floor

    tmp_config_path = PIPELINE_DIR / "_ui_config.yaml"
    with open(tmp_config_path, "w") as f:
        yaml.dump(config, f)

    # Run pipeline inside valve_pipeline/ so relative paths resolve correctly
    log_capture = io.StringIO()
    records: list[dict] = []
    error_msg: str | None = None

    prev_cwd = os.getcwd()

    status_placeholder = st.empty()
    with status_placeholder.status("Running pipeline…", expanded=True) as status_box:
        try:
            from src.pipeline import run  # import here so PIPELINE_DIR is on sys.path

            os.chdir(PIPELINE_DIR)

            with contextlib.redirect_stdout(log_capture):
                records = run(
                    schematic_path=str(input_path),
                    config_path=str(tmp_config_path),
                    debug=debug_mode,
                )
            status_box.update(label=f"Done — {len(records)} valve(s) detected", state="complete")
        except Exception as exc:
            error_msg = str(exc)
            status_box.update(label="Pipeline failed", state="error")
        finally:
            os.chdir(prev_cwd)
            if tmp_config_path.exists():
                tmp_config_path.unlink()

    # Show pipeline log
    log_text = log_capture.getvalue()
    if log_text:
        with st.expander("Pipeline log", expanded=False):
            st.code(log_text, language=None)

    if error_msg:
        st.error(f"Error: {error_msg}")
        st.stop()

    # ── Results ────────────────────────────────────────────────────────────────

    stem = input_path.stem
    output_dir = PIPELINE_DIR / "output"
    annotated_path = output_dir / f"{stem}_annotated.png"
    json_path = output_dir / f"{stem}_results.json"

    with result_col:
        st.subheader("Output")
        if annotated_path.exists():
            st.image(str(annotated_path), caption="Annotated schematic", use_container_width=True)
        else:
            st.info("No annotated image generated.")

    st.divider()

    # Metrics row
    label_counts: dict[str, int] = {}
    for r in records:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    metric_cols = st.columns(max(1, len(label_counts) + 1))
    metric_cols[0].metric("Total valves", len(records))
    for i, (lbl, cnt) in enumerate(sorted(label_counts.items()), start=1):
        if i < len(metric_cols):
            metric_cols[i].metric(lbl.capitalize(), cnt)

    # Detections table
    if records:
        st.subheader("Detections")
        import pandas as pd

        df = pd.DataFrame(records)
        df["x"] = df["bbox"].apply(lambda b: b[0])
        df["y"] = df["bbox"].apply(lambda b: b[1])
        df["w"] = df["bbox"].apply(lambda b: b[2])
        df["h"] = df["bbox"].apply(lambda b: b[3])
        df = df.drop(columns=["bbox"])
        df = df[["label", "confidence", "detection_source", "match_score", "x", "y", "w", "h"]]
        df["confidence"] = df["confidence"].round(3)
        df["match_score"] = df["match_score"].round(3)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No valves detected. Try lowering the match threshold or confidence floor.")

    # Raw JSON
    with st.expander("Raw JSON"):
        if json_path.exists():
            st.json(json.loads(json_path.read_text()))
        else:
            st.json({"schematic": str(input_path), "detections": records})

    # Download buttons
    dl_col1, dl_col2 = st.columns(2)
    if json_path.exists():
        dl_col1.download_button(
            "Download JSON",
            data=json_path.read_bytes(),
            file_name=json_path.name,
            mime="application/json",
        )
    if annotated_path.exists():
        dl_col2.download_button(
            "Download annotated image",
            data=annotated_path.read_bytes(),
            file_name=annotated_path.name,
            mime="image/png",
        )

    # Debug images
    if debug_mode:
        debug_dir = output_dir / "debug"
        debug_imgs = sorted(debug_dir.glob(f"{stem}_*.png")) if debug_dir.exists() else []
        if debug_imgs:
            st.divider()
            st.subheader("Debug images")
            dcols = st.columns(min(3, len(debug_imgs)))
            for i, p in enumerate(debug_imgs):
                dcols[i % len(dcols)].image(str(p), caption=p.stem, use_container_width=True)
