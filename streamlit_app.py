"""
Streamlit interface for StarSight AI.

This file is intentionally separate from the PySide6 desktop app so both
interfaces can coexist while sharing the same detection/matching engine.
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import streamlit as st

from constellation_data import CONSTELLATION_INFO, STAR_CATALOG, get_edges, get_star_positions_pixels
from star_engine import connect_stars, detect_stars, label_stars, match_constellation_details


def cv_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def build_preview(image_bgr: np.ndarray, zoom: float) -> np.ndarray:
    if zoom == 1.0:
        return image_bgr
    height, width = image_bgr.shape[:2]
    resized = cv2.resize(
        image_bgr,
        (max(1, int(width * zoom)), max(1, int(height * zoom))),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized


def render_sky_map(constellation_name: str, size: int = 720, padding: int = 70) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:] = (8, 16, 28)

    for gx in range(24, size, 42):
        for gy in range(24, size, 42):
            cv2.circle(image, (gx, gy), 1, (28, 50, 74), -1)

    pts = get_star_positions_pixels(constellation_name, size, padding)
    edges = get_edges(constellation_name)
    stars = STAR_CATALOG[constellation_name]["stars"]

    for i, j in edges:
        if i < len(pts) and j < len(pts):
            cv2.line(image, pts[i], pts[j], (80, 180, 255), 2)

    for idx, (x, y) in enumerate(pts):
        mag = stars[idx].get("mag", 3.0)
        radius = max(4, int(9 - mag))
        cv2.circle(image, (x, y), radius + 2, (32, 70, 118), 1)
        cv2.circle(image, (x, y), radius, (255, 255, 255), -1)
        cv2.putText(
            image,
            stars[idx]["name"],
            (x + radius + 8, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (161, 216, 255),
            1,
            cv2.LINE_AA,
        )

    return image


def encode_png(image_bgr: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        return b""
    return buffer.tobytes()


def process_upload(uploaded_file) -> dict | None:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_file.write(uploaded_file.getbuffer())
    temp_file.close()

    try:
        stars, image = detect_stars(temp_file.name)
        if not stars or image is None:
            return None

        constellation, confidence, matched = match_constellation_details(stars)
        draw_stars = matched if matched else stars
        annotated = connect_stars(image.copy(), draw_stars, constellation)

        return {
            "stars": stars,
            "matched_stars": draw_stars,
            "constellation": constellation,
            "confidence": confidence,
            "annotated": annotated,
        }
    finally:
        try:
            os.unlink(temp_file.name)
        except OSError:
            pass


def render_info_panel(constellation_name: str, mythology_mode: str) -> None:
    if constellation_name not in CONSTELLATION_INFO:
        st.info("Upload a constellation image to see lore, seasonality, and reference details.")
        return

    info = CONSTELLATION_INFO[constellation_name]
    myth_text = info["myth"] if mythology_mode == "greek" else info["indian_myth"]

    st.subheader(constellation_name)
    c1, c2 = st.columns(2)
    c1.metric("Main stars", info.get("stars", "N/A"))
    c2.metric("Season", info.get("season", "N/A"))

    st.caption("Mythology")
    st.markdown(
        f"""
        <div style="padding:16px;border:1px solid #29476b;border-radius:16px;background:#0f1d31;color:#edf4ff;">
            {myth_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Fun fact")
    st.markdown(
        f"""
        <div style="padding:16px;border:1px solid #29476b;border-radius:16px;background:#12233a;color:#cfe4ff;">
            {info.get("fact", "N/A")}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="StarSight AI", page_icon="✨", layout="wide")

    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #08111f 0%, #0b1628 100%);
            color: #edf4ff;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        [data-testid="stMetricValue"] {
            color: #edf4ff;
        }
        [data-testid="stMetricLabel"] {
            color: #9db7d7;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("StarSight AI")
    st.caption("Browser-based constellation detection using the same engine as the desktop app.")

    with st.sidebar:
        st.header("Controls")
        show_labels = st.toggle("Show labels", value=False)
        mythology_mode = st.radio("Lore mode", ["greek", "indian"], horizontal=True)
        zoom = st.slider("Preview zoom", min_value=0.6, max_value=2.4, value=1.0, step=0.1)
        st.markdown("---")
        st.write("Supported constellations:")
        st.write(", ".join(STAR_CATALOG.keys()))

    uploaded_file = st.file_uploader(
        "Upload a night-sky or constellation image",
        type=["png", "jpg", "jpeg", "bmp"],
    )

    if uploaded_file is None:
        st.info("Choose an image to analyze.")
        return

    result = process_upload(uploaded_file)
    if result is None:
        st.error("Could not detect enough stars in this image.")
        return

    preview_image = result["annotated"].copy()
    if show_labels and result["matched_stars"] and result["constellation"] in STAR_CATALOG:
        preview_image = label_stars(preview_image, result["matched_stars"], result["constellation"])
    preview_image = build_preview(preview_image, zoom)

    left, right = st.columns([1.7, 1.1], gap="large")

    with left:
        st.subheader("Annotated sky preview")
        m1, m2, m3 = st.columns(3)
        m1.metric("Constellation", result["constellation"])
        m2.metric("Detected stars", len(result["stars"]))
        m3.metric("Confidence", f"{result['confidence']:.1f}%")
        st.image(cv_to_rgb(preview_image), use_container_width=True)

        download_bytes = encode_png(preview_image)
        st.download_button(
            "Download annotated image",
            data=download_bytes,
            file_name="starsight_result.png",
            mime="image/png",
            disabled=not bool(download_bytes),
        )

    with right:
        st.subheader("Constellation briefing")
        render_info_panel(result["constellation"], mythology_mode)
        if result["constellation"] in STAR_CATALOG:
            st.subheader("Reference sky map")
            st.image(
                cv_to_rgb(render_sky_map(result["constellation"])),
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
