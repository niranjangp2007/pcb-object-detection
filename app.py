"""AI Object Detection Suite (100% Offline, CPU-Only PyTorch + YOLO26 + Specialized PCB Module)."""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from pcb_inspector import (
    COMPONENT_ROLES,
    DEFAULT_MARKER_SIZE_MM,
    DetectionItem,
    GeneralDetectionResult,
    InspectionResult,
    generate_printable_aruco_png,
    load_general_model,
    load_models,
    run_general_detection,
    run_inspection,
)

MODE_GENERAL = "General Object Detection"
MODE_PCB = "PCB Inspection (Specialized)"

st.set_page_config(
    page_title="AI Object Detection Suite",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .status-banner {
        padding: 0.9rem 1.2rem;
        border-radius: 0.6rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border: 1px solid transparent;
    }
    .status-pass {
        background-color: #ecfdf5;
        color: #065f46;
        border-color: #a7f3d0;
    }
    .status-fail {
        background-color: #fef2f2;
        color: #991b1b;
        border-color: #fecaca;
    }
    .calib-badge-ok {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        background-color: #eff6ff;
        color: #1d4ed8;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .calib-badge-warn {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        background-color: #fffbeb;
        color: #b45309;
        font-size: 0.85rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading YOLO26n (CPU)...")
def get_cached_general_model():
    """Load and cache the offline YOLO26n general object detection model."""
    return load_general_model()


@st.cache_resource(show_spinner="Loading 3 specialized PCB YOLO models onto CPU...")
def get_cached_pcb_models():
    """Load and cache the three specialized PCB inspection YOLO models."""
    return load_models()


def _release_live_camera() -> None:
    """Release any open cv2.VideoCapture stored in Streamlit session state."""
    cap = st.session_state.pop("live_cap", None)
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass


def _has_server_camera_device() -> bool:
    """Check whether a server-side camera device is likely available (Linux checks /dev/video*)."""
    if sys.platform.startswith("linux"):
        return any(Path("/dev").glob("video*"))
    return True


def _open_local_camera(cam_index: int) -> cv2.VideoCapture | None:
    """Open a local webcam safely, preferring CAP_DSHOW on Windows and handling missing devices."""
    if not _has_server_camera_device():
        return None
    try:
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(int(cam_index), cv2.CAP_DSHOW)
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
        cap = cv2.VideoCapture(int(cam_index))
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
    except Exception:
        return None
    return None


def _acquire_input_image(input_mode: str, key_prefix: str, help_text: str) -> np.ndarray | None:
    """Acquire a single RGB image from Photo Upload or snapshot/single-frame Webcam."""
    if input_mode == "High-Res Photo Upload":
        uploaded = st.file_uploader(
            "Upload Image (PNG, JPG, JPEG, BMP, TIFF, WEBP)",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
            key=f"{key_prefix}_uploader",
        )
        if uploaded is not None:
            pil_img = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
            return np.array(pil_img)
        st.info(help_text)
        return None

    st.subheader("Webcam Capture")
    cam_backend = st.radio(
        "Webcam Source",
        ["Browser Snapshot (st.camera_input)", "Local OpenCV Camera (cv2.VideoCapture)"],
        horizontal=True,
        key=f"{key_prefix}_cam_backend",
    )
    if cam_backend == "Browser Snapshot (st.camera_input)":
        cam_photo = st.camera_input("Capture a frame from your webcam", key=f"{key_prefix}_cam_input")
        if cam_photo is not None:
            pil_img = ImageOps.exif_transpose(Image.open(cam_photo)).convert("RGB")
            return np.array(pil_img)
        return None

    cam_index = st.number_input(
        "Camera Device Index",
        min_value=0,
        max_value=5,
        value=0,
        step=1,
        key=f"{key_prefix}_cam_idx",
    )
    if st.button("Capture & Inspect Local Camera Frame", type="primary", key=f"{key_prefix}_cam_btn"):
        if not _has_server_camera_device():
            st.warning(
                "No server-side camera device (/dev/video*) was detected on this host. "
                "On cloud deployments, switch Webcam Source to **Browser Snapshot (st.camera_input)**."
            )
            return None
        cap = _open_local_camera(int(cam_index))
        try:
            if cap is None or not cap.isOpened():
                st.error(
                    f"Could not open local camera index {int(cam_index)}. "
                    "Use **Browser Snapshot (st.camera_input)** if running remotely."
                )
                return None
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                st.error("Failed to read a frame from the local camera.")
                return None
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        finally:
            if cap is not None:
                cap.release()
    return None


def _run_general_live_stream(
    general_model,
    conf_threshold: float,
    imgsz: int,
    cam_index: int,
) -> None:
    """Stream continuous frames from cv2.VideoCapture and render live YOLO26n detections on CPU."""
    if not _has_server_camera_device():
        _release_live_camera()
        st.info(
            "No server-side camera device (`/dev/video*`) is attached to this Linux host "
            "(typical on Streamlit Community Cloud). Select **Browser Snapshot (st.camera_input)** above "
            "to use your device's browser camera."
        )
        return

    live_running = st.toggle(
        "Start / Stop Live Detection",
        value=False,
        key="general_live_active",
        help="Toggle ON to stream continuous frames from your local webcam with real-time CPU YOLO26n detection.",
    )

    if not live_running:
        _release_live_camera()
        st.info("Toggle **Start / Stop Live Detection** ON to begin live CPU object detection.")
        return

    m1, m2, m3, m4 = st.columns(4)
    fps_slot = m1.empty()
    inf_slot = m2.empty()
    obj_slot = m3.empty()
    cls_slot = m4.empty()
    frame_placeholder = st.empty()
    counts_placeholder = st.empty()

    _release_live_camera()
    cap = _open_local_camera(int(cam_index))
    st.session_state["live_cap"] = cap

    try:
        if cap is None or not cap.isOpened():
            st.error(
                f"Could not open local camera index {int(cam_index)}. Ensure a webcam is connected and not in use, "
                "or switch to **Browser Snapshot (st.camera_input)**."
            )
            return

        while st.session_state.get("general_live_active", False):
            loop_start = time.perf_counter()
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                st.warning("Camera frame read failed; stopping live stream.")
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = run_general_detection(
                image_rgb=frame_rgb,
                model=general_model,
                conf_threshold=float(conf_threshold),
                imgsz=int(imgsz),
            )
            loop_s = max(time.perf_counter() - loop_start, 1e-6)
            live_fps = 1.0 / loop_s

            fps_slot.metric("Live FPS", f"{live_fps:.1f}")
            inf_slot.metric("Inference Time (CPU)", f"{result.inference_ms:.1f} ms")
            obj_slot.metric("Objects in Frame", len(result.detections))
            cls_slot.metric("Unique Classes", len(result.class_counts))

            frame_placeholder.image(
                result.annotated_rgb,
                caption=f"Live YOLO26n CPU Feed ({imgsz}px, conf={conf_threshold:.2f})",
                use_container_width=True,
            )

            if result.class_counts:
                counts_df = pd.DataFrame(
                    [{"Class": name, "Count": cnt} for name, cnt in result.class_counts.items()]
                )
                counts_placeholder.dataframe(counts_df, use_container_width=True, hide_index=True)
            else:
                counts_placeholder.info("No objects above confidence threshold in the current frame.")
    finally:
        _release_live_camera()


def _render_general_results(image_rgb: np.ndarray, result: GeneralDetectionResult) -> None:
    """Render General Object Detection KPIs, annotated output, and class count summary."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Objects Detected", len(result.detections))
    col2.metric("Unique Classes", len(result.class_counts))
    col3.metric("Inference Time (CPU)", f"{result.inference_ms:.1f} ms")
    fps_equiv = (1000.0 / result.inference_ms) if result.inference_ms > 0 else 0.0
    col4.metric("Throughput", f"{fps_equiv:.1f} FPS")

    view_mode = st.radio(
        "Image View",
        ["Annotated Detections", "Side-by-Side (Original vs Annotated)"],
        horizontal=True,
        key="general_view_mode",
    )
    if view_mode == "Annotated Detections":
        st.image(result.annotated_rgb, caption="YOLO26n Object Detection (CPU)", use_container_width=True)
    else:
        left_img, right_img = st.columns(2)
        left_img.image(image_rgb, caption="Original Input", use_container_width=True)
        right_img.image(result.annotated_rgb, caption="Annotated Detections", use_container_width=True)

    st.subheader("Detected Class Count Summary")
    if not result.class_counts:
        st.info("No objects detected at the current confidence threshold. Try lowering the confidence slider.")
        return

    count_cols = st.columns(min(len(result.class_counts), 5))
    for i, (cls_name, count) in enumerate(result.class_counts.items()):
        count_cols[i % len(count_cols)].metric(cls_name, count)

    summary_df = pd.DataFrame(
        [{"Class": cls_name, "Count": cnt} for cls_name, cnt in result.class_counts.items()]
    )
    det_df = pd.DataFrame(
        [
            {
                "#": idx,
                "Class": item.class_name,
                "Confidence": round(item.confidence, 3),
                "Bounding Box (px)": (
                    f"({item.bbox_xyxy[0]:.0f}, {item.bbox_xyxy[1]:.0f}) → "
                    f"({item.bbox_xyxy[2]:.0f}, {item.bbox_xyxy[3]:.0f})"
                ),
                "Size (px)": f"{item.width_px:.0f} × {item.height_px:.0f}",
            }
            for idx, item in enumerate(result.detections, start=1)
        ]
    )

    tab_counts, tab_boxes = st.tabs(["Class Count Summary", "All Detections"])
    with tab_counts:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    with tab_boxes:
        st.dataframe(det_df, use_container_width=True, hide_index=True)
        st.download_button(
            label="Download Detections CSV",
            data=det_df.to_csv(index=False).encode("utf-8"),
            file_name="general_object_detections.csv",
            mime="text/csv",
        )


def _items_to_component_df(items: list[DetectionItem], calibrated: bool) -> pd.DataFrame:
    rows = []
    for idx, item in enumerate(items, start=1):
        dim_mm = (
            f"{item.width_mm:.2f} × {item.height_mm:.2f}"
            if calibrated and item.width_mm is not None and item.height_mm is not None
            else "Uncalibrated"
        )
        area_mm2 = (
            round(item.area_mm2, 2)
            if calibrated and item.area_mm2 is not None
            else None
        )
        rows.append(
            {
                "#": idx,
                "Component Class": item.class_name,
                "Category": item.category,
                "Confidence": round(item.confidence, 3),
                "Dimensions (mm)": dim_mm,
                "Area (mm²)": area_mm2,
                "Size (px)": f"{item.width_px:.0f} × {item.height_px:.0f}",
                "Circuit Role": item.details,
            }
        )
    return pd.DataFrame(rows)


def _items_to_defect_df(
    bare_defects: list[DetectionItem],
    solder_defects: list[DetectionItem],
    calibrated: bool,
) -> pd.DataFrame:
    rows = []
    for idx, item in enumerate([*bare_defects, *solder_defects], start=1):
        dim_mm = (
            f"{item.width_mm:.2f} × {item.height_mm:.2f}"
            if calibrated and item.width_mm is not None and item.height_mm is not None
            else "Uncalibrated"
        )
        rows.append(
            {
                "#": idx,
                "Inspection Stage": item.model_group,
                "Defect Type": item.class_name,
                "Confidence": round(item.confidence, 3),
                "Dimensions (mm)": dim_mm,
                "Size (px)": f"{item.width_px:.0f} × {item.height_px:.0f}",
                "Engineering Impact": item.details,
            }
        )
    return pd.DataFrame(rows)


def _render_pcb_results(image_rgb: np.ndarray, result: InspectionResult) -> None:
    calib = result.calibration
    timings = result.timings_ms

    if result.passed:
        st.markdown(
            f"""
            <div class="status-banner status-pass">
                PASS — 0 defects detected across Bare-Board &amp; Solder/Assembly inspections
                (Total CPU Scan: {timings['total_ms']:.0f} ms)
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="status-banner status-fail">
                FAIL — {result.total_defects} defect(s) detected
                ({len(result.bare_defects)} bare-board, {len(result.solder_defects)} solder/assembly)
                (Total CPU Scan: {timings['total_ms']:.0f} ms)
            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Inspection Verdict", "PASS" if result.passed else "FAIL")
    col2.metric("Total Scan Time", f"{timings['total_ms']:.0f} ms")
    col3.metric("Components Found", len(result.components))
    col4.metric(
        "Defects Found",
        result.total_defects,
        delta=f"{len(result.bare_defects)} bare / {len(result.solder_defects)} solder",
        delta_color="inverse" if result.total_defects > 0 else "off",
    )
    if calib.detected and calib.board_width_mm and calib.board_height_mm:
        col5.metric(
            "Board Size (mm)",
            f"{calib.board_width_mm:.1f} × {calib.board_height_mm:.1f}",
            delta=f"{calib.px_per_mm:.2f} px/mm",
        )
    elif calib.detected and calib.px_per_mm:
        col5.metric("ArUco Scale", f"{calib.px_per_mm:.2f} px/mm", delta="Board contour n/a")
    else:
        col5.metric("Calibration", "Uncalibrated", delta="No 20mm ArUco")

    t_cols = st.columns(4)
    t_cols[0].caption(f"Components (`fine_best.pt @ 1024`): **{timings['components_ms']:.0f} ms**")
    t_cols[1].caption(f"Bare Defects (`bare defects.pt @ 1024`): **{timings['bare_defects_ms']:.0f} ms**")
    t_cols[2].caption(f"Solder Defects (`best solder defect.pt @ 800`): **{timings['solder_defects_ms']:.0f} ms**")
    t_cols[3].caption(f"ArUco Calibration: **{timings['aruco_ms']:.1f} ms**")

    if calib.detected:
        board_note = (
            f" | Estimated Board: **{calib.board_width_mm:.1f} × {calib.board_height_mm:.1f} mm** "
            f"({calib.board_area_mm2:.0f} mm², via {calib.board_measurement_source})"
            if calib.board_width_mm and calib.board_height_mm and calib.board_area_mm2
            else ""
        )
        st.markdown(
            f'<span class="calib-badge-ok">ArUco Calibrated (IDs: {calib.marker_ids}, '
            f'{calib.marker_size_mm:.0f} mm marker = {calib.px_per_mm:.2f} px/mm)</span>{board_note}',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="calib-badge-warn">No DICT_4X4_50 ArUco marker detected — '
            "place the printed 20×20 mm marker next to the PCB for millimeter measurements.</span>",
            unsafe_allow_html=True,
        )

    st.write("")

    view_mode = st.radio(
        "Image View",
        ["Annotated Inspection", "Side-by-Side (Original vs Annotated)"],
        horizontal=True,
        key="pcb_view_mode",
    )
    if view_mode == "Annotated Inspection":
        st.image(result.annotated_rgb, caption="Annotated PCB Inspection (CPU)", use_container_width=True)
    else:
        left_img, right_img = st.columns(2)
        left_img.image(image_rgb, caption="Original Input", use_container_width=True)
        right_img.image(result.annotated_rgb, caption="Annotated Inspection", use_container_width=True)

    st.subheader("Defect Summary")
    if result.total_defects == 0:
        st.success("No bare-board copper defects or solder/assembly defects were detected at this confidence threshold.")
    else:
        defect_df = _items_to_defect_df(result.bare_defects, result.solder_defects, calib.detected)
        st.dataframe(defect_df, use_container_width=True, hide_index=True)

        defect_counts = Counter(d.class_name for d in [*result.bare_defects, *result.solder_defects])
        count_cols = st.columns(min(len(defect_counts), 4))
        for i, (d_name, d_cnt) in enumerate(defect_counts.most_common()):
            count_cols[i % len(count_cols)].metric(f"Defect: {d_name}", d_cnt)

    st.subheader("Component Inspection & Circuit Role Table")
    if not result.components:
        st.info("No PCB components detected at the current confidence threshold. Try lowering the confidence slider.")
    else:
        comp_df = _items_to_component_df(result.components, calib.detected)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        counts_by_class = Counter(c.class_name for c in result.components)
        summary_rows = [
            {
                "Component Class": cls_name,
                "Count": cnt,
                "Category": COMPONENT_ROLES.get(cls_name, {}).get("category", "Unmapped"),
                "Circuit Role": COMPONENT_ROLES.get(cls_name, {}).get("circuit_role", ""),
            }
            for cls_name, cnt in counts_by_class.most_common()
        ]
        with st.expander("Component Class Breakdown & CSV Export", expanded=False):
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
            st.download_button(
                label="Download Component Inspection CSV",
                data=comp_df.to_csv(index=False).encode("utf-8"),
                file_name="pcb_component_inspection.csv",
                mime="text/csv",
            )


def main() -> None:
    st.title("AI Object Detection Suite")
    st.caption(
        "100% Offline CPU-Only Vision Platform — General Object Detection (`yolo26n.pt`) "
        "with a Pluggable Industrial PCB Inspection Module (`fine_best.pt`, `bare defects.pt`, `best solder defect.pt`)"
    )

    with st.sidebar:
        st.header("Platform Module")
        app_mode = st.radio(
            "Detection Mode",
            [MODE_GENERAL, MODE_PCB],
            index=0,
            help="Switch between general-purpose COCO object detection and specialized PCB defect/component inspection.",
        )

        st.divider()
        st.header("Input & Threshold")
        input_mode = st.radio(
            "Input Source",
            ["High-Res Photo Upload", "Webcam Mode"],
            index=0,
        )
        is_general_webcam = app_mode == MODE_GENERAL and input_mode == "Webcam Mode"
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.35 if is_general_webcam else 0.25,
            step=0.05,
            key=f"conf_{app_mode}_{input_mode}",
            help="Minimum detection confidence applied on CPU.",
        )

        if app_mode == MODE_GENERAL:
            general_imgsz = st.select_slider(
                "Inference Resolution (imgsz)",
                options=[320, 480, 640, 800, 1024],
                value=480 if is_general_webcam else 640,
                key=f"imgsz_general_{input_mode}",
            )
            st.info("Running offline `models/yolo26n.pt` (80 COCO classes) on `device='cpu'`.")
        else:
            _release_live_camera()
            st.session_state["general_live_active"] = False
            st.subheader("PCB Overlay Layers")
            show_components = st.checkbox("PCB Components (fine_best.pt @ 1024)", value=True)
            show_bare_defects = st.checkbox("Bare-Board Defects (bare defects.pt @ 1024)", value=True)
            show_solder_defects = st.checkbox("Solder Defects (best solder defect.pt @ 800)", value=True)
            show_calibration = st.checkbox("ArUco & Board Outline", value=True)

            st.divider()
            st.subheader("ArUco Scale Calibration")
            marker_size_mm = st.number_input(
                "Physical Marker Size (mm)",
                min_value=5.0,
                max_value=100.0,
                value=DEFAULT_MARKER_SIZE_MM,
                step=1.0,
                help="Square black-border width of your printed DICT_4X4_50 marker (default 20 mm).",
            )
            marker_id = st.number_input(
                "Printable Marker ID (0–49)",
                min_value=0,
                max_value=49,
                value=0,
                step=1,
            )
            marker_png = generate_printable_aruco_png(
                marker_id=int(marker_id),
                marker_size_mm=float(marker_size_mm),
                dpi=300,
            )
            st.image(
                marker_png,
                caption=f"DICT_4X4_50 ID {int(marker_id)} ({marker_size_mm:.0f}×{marker_size_mm:.0f} mm @ 300 DPI)",
                width=180,
            )
            st.download_button(
                label="Download Printable 20mm Marker (PNG)",
                data=marker_png,
                file_name=f"aruco_4x4_50_id{int(marker_id)}_{int(marker_size_mm)}mm.png",
                mime="image/png",
                use_container_width=True,
            )
            st.caption("Print at 100% scale (disable 'Fit to Page') and place flat beside the PCB.")

    if app_mode == MODE_GENERAL:
        st.subheader("Module: General Object Detection (`yolo26n.pt` · CPU)")
        general_model = get_cached_general_model()
        if input_mode == "High-Res Photo Upload":
            _release_live_camera()
            st.session_state["general_live_active"] = False
            image_rgb = _acquire_input_image(
                input_mode=input_mode,
                key_prefix="general",
                help_text="Upload a photo or switch to Webcam Mode to detect objects offline with YOLO26n on CPU.",
            )
            if image_rgb is not None:
                with st.spinner("Running YOLO26n object detection on CPU..."):
                    general_result = run_general_detection(
                        image_rgb=image_rgb,
                        model=general_model,
                        conf_threshold=float(conf_threshold),
                        imgsz=int(general_imgsz),
                    )
                _render_general_results(image_rgb, general_result)
        else:
            st.subheader("Live Webcam Object Detection")
            cam_backend = st.radio(
                "Webcam Source",
                [
                    "Local OpenCV Live Stream (cv2.VideoCapture)",
                    "Browser Snapshot (st.camera_input)",
                ],
                index=0,
                horizontal=True,
                key="general_webcam_backend",
            )
            if cam_backend == "Local OpenCV Live Stream (cv2.VideoCapture)":
                cam_index = st.number_input(
                    "Camera Device Index",
                    min_value=0,
                    max_value=5,
                    value=0,
                    step=1,
                    key="general_live_cam_idx",
                )
                _run_general_live_stream(
                    general_model=general_model,
                    conf_threshold=float(conf_threshold),
                    imgsz=int(general_imgsz),
                    cam_index=int(cam_index),
                )
            else:
                _release_live_camera()
                st.session_state["general_live_active"] = False
                cam_photo = st.camera_input("Capture a frame from your webcam", key="general_browser_cam")
                if cam_photo is not None:
                    pil_img = ImageOps.exif_transpose(Image.open(cam_photo)).convert("RGB")
                    image_rgb = np.array(pil_img)
                    with st.spinner("Running YOLO26n object detection on CPU..."):
                        general_result = run_general_detection(
                            image_rgb=image_rgb,
                            model=general_model,
                            conf_threshold=float(conf_threshold),
                            imgsz=int(general_imgsz),
                        )
                    _render_general_results(image_rgb, general_result)
    else:
        st.subheader("Module: PCB Inspection (Specialized 3-Model + ArUco Pipeline · CPU)")
        st.caption(
            "`fine_best.pt @ 1024px` (Components) + `bare defects.pt @ 1024px` (Copper Defects) + "
            "`best solder defect.pt @ 800px` (Solder/Assembly) + OpenCV ArUco 20mm Calibration"
        )
        pcb_models = get_cached_pcb_models()
        image_rgb = _acquire_input_image(
            input_mode=input_mode,
            key_prefix="pcb",
            help_text=(
                "Upload a high-resolution PCB photo to run CPU inspection across all three specialized models. "
                "Optionally place the printed 20×20 mm ArUco marker (DICT_4X4_50) beside the board for mm calibration."
            ),
        )
        if image_rgb is not None:
            with st.spinner(
                "Running CPU PCB inspection (Components @ 1024 + Bare Defects @ 1024 + Solder Defects @ 800 + ArUco)..."
            ):
                pcb_result = run_inspection(
                    image_rgb=image_rgb,
                    models=pcb_models,
                    conf_threshold=float(conf_threshold),
                    marker_size_mm=float(marker_size_mm),
                    show_components=show_components,
                    show_bare_defects=show_bare_defects,
                    show_solder_defects=show_solder_defects,
                    show_calibration=show_calibration,
                )
            _render_pcb_results(image_rgb, pcb_result)

        with st.expander("Supported Component Circuit Role Dictionary (fine_best.pt)", expanded=False):
            ref_df = pd.DataFrame(
                [
                    {
                        "Component Class": cls_name,
                        "Category": meta["category"],
                        "Circuit Role": meta["circuit_role"],
                    }
                    for cls_name, meta in COMPONENT_ROLES.items()
                ]
            )
            st.dataframe(ref_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
