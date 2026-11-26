"""Offline PCB inspection engine using CPU-only PyTorch, YOLO models, and OpenCV ArUco calibration."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Keep Ultralytics config local and offline-safe before importing ultralytics.
BASE_DIR = Path(__file__).resolve().parent
_default_yolo_cfg = BASE_DIR / ".yolo_config"
if not os.access(BASE_DIR, os.W_OK):
    _default_yolo_cfg = Path(tempfile.gettempdir()) / ".yolo_config"
os.environ.setdefault("YOLO_CONFIG_DIR", str(_default_yolo_cfg))
os.environ.setdefault("YOLO_VERBOSE", "False")

import cv2
import numpy as np
from PIL import Image, ImageDraw

MODELS_DIR = BASE_DIR / "models"
GENERAL_MODEL_PATH = MODELS_DIR / "yolo26n.pt"
COMPONENT_MODEL_PATH = MODELS_DIR / "fine_best.pt"
BARE_DEFECT_MODEL_PATH = MODELS_DIR / "bare defects.pt"
SOLDER_DEFECT_MODEL_PATH = MODELS_DIR / "best solder defect.pt"

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
DEFAULT_MARKER_SIZE_MM = 20.0

# Complete local mapping for every class in ./models/fine_best.pt
COMPONENT_ROLES: dict[str, dict[str, str]] = {
    "Resistor": {
        "category": "Passive",
        "circuit_role": "Limits electrical current, divides voltage, and sets biasing or pull-up/pull-down logic levels.",
    },
    "Capacitor": {
        "category": "Passive",
        "circuit_role": "Stores electrostatic energy for power-rail decoupling, high-frequency noise filtering, and AC coupling.",
    },
    "Electrolytic Capacitor": {
        "category": "Passive (Polarized)",
        "circuit_role": "Provides high bulk capacitance for DC bus smoothing, ripple suppression, and low-frequency power filtering.",
    },
    "IC": {
        "category": "Active / Logic",
        "circuit_role": "Integrated Circuit executing core computation, signal conditioning, power regulation, or interface control.",
    },
    "Inductor": {
        "category": "Passive / Magnetic",
        "circuit_role": "Stores energy in a magnetic field for DC-DC switching converters, EMI choking, and LC filtering.",
    },
    "Diode": {
        "category": "Semiconductor",
        "circuit_role": "Enforces unidirectional current flow for rectification, flyback clamping, and reverse-polarity protection.",
    },
    "LED": {
        "category": "Optoelectronic",
        "circuit_role": "Light-Emitting Diode providing visual power, diagnostic status, fault, or activity indication.",
    },
    "Transistor": {
        "category": "Semiconductor",
        "circuit_role": "Active switch or amplifier controlling current/voltage drive across load and signal stages.",
    },
    "Oscillator": {
        "category": "Timing",
        "circuit_role": "Generates a precise periodic clock/frequency reference for microcontrollers and synchronous digital logic.",
    },
    "Connector": {
        "category": "Interconnect",
        "circuit_role": "Provides mechanical and electrical mating for external cables, power feeds, headers, or peripheral I/O.",
    },
    "Switch/Button": {
        "category": "Electromechanical",
        "circuit_role": "User-actuated contact for power switching, hardware reset, boot mode selection, or manual control.",
    },
    "Other": {
        "category": "Auxiliary",
        "circuit_role": "Specialized or auxiliary PCB element (e.g., fuse, jumper, test point, heatsink, or mounting hardware).",
    },
    "Marker": {
        "category": "Fiducial / Calibration",
        "circuit_role": "Optical fiducial or calibration reference target used for board alignment and scale registration.",
    },
}

DEFECT_DESCRIPTIONS: dict[str, dict[str, str]] = {
    # Bare-board / copper defects (./models/bare defects.pt)
    "missing_hole": {
        "domain": "Bare Board (Copper)",
        "description": "Expected drill/via hole is absent, blocking through-hole mounting or layer interconnect.",
    },
    "mouse_bite": {
        "domain": "Bare Board (Copper)",
        "description": "Irregular edge notch or copper nibbling along a trace that reduces current-carrying width.",
    },
    "open_circuit": {
        "domain": "Bare Board (Copper)",
        "description": "Broken or disconnected copper trace interrupting electrical continuity.",
    },
    "short": {
        "domain": "Bare Board (Copper)",
        "description": "Unintended copper bridge connecting two isolated nets or traces.",
    },
    "spur": {
        "domain": "Bare Board (Copper)",
        "description": "Protruding copper spike extending from a trace or pad, risking clearance failure.",
    },
    "spurious_copper": {
        "domain": "Bare Board (Copper)",
        "description": "Unwanted residual copper island left on the substrate after etching.",
    },
    "pin_hole": {
        "domain": "Bare Board (Copper)",
        "description": "Small void or pit inside a copper pad or trace that weakens conductivity.",
    },
    # Solder & assembly defects (./models/best solder defect.pt)
    "insufficient_solder": {
        "domain": "Solder / Assembly",
        "description": "Inadequate solder fillet wetting, risking weak mechanical joint or intermittent open.",
    },
    "excess_solder": {
        "domain": "Solder / Assembly",
        "description": "Over-deposited solder mound obscuring lead geometry and risking adjacent bridging.",
    },
    "solder_bridge": {
        "domain": "Solder / Assembly",
        "description": "Unintended solder short spanning across adjacent pins, pads, or leads.",
    },
    "solder_void": {
        "domain": "Solder / Assembly",
        "description": "Gas pocket or cavity within a solder joint reducing thermal and electrical contact.",
    },
    "missing_component": {
        "domain": "Solder / Assembly",
        "description": "Populated pad footprint is missing its designated electronic component.",
    },
    "shifted_component": {
        "domain": "Solder / Assembly",
        "description": "Component is misaligned, tombstoned, or offset from its target landing pads.",
    },
    "burn": {
        "domain": "Solder / Assembly",
        "description": "Thermal charring or scorched flux/substrate damage from excessive heat or arc.",
    },
    "physical_damage": {
        "domain": "Solder / Assembly",
        "description": "Cracked package, chipped substrate, lifted pad, or bent/broken terminal.",
    },
    "oxidation": {
        "domain": "Solder / Assembly",
        "description": "Surface corrosion or tarnished pad/joint degrading solderability and conductivity.",
    },
}


class _LegacyAlbumentationsStub:
    """Placeholder for legacy albumentations transform objects embedded in checkpoint metadata."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: None

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)


class _LegacyModuleStub(types.ModuleType):
    """Module stub returning _LegacyAlbumentationsStub for any legacy transform attribute."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return _LegacyAlbumentationsStub


def _ensure_legacy_albumentations_stubs() -> None:
    """Register lightweight stubs only when legacy albumentations submodules are absent."""
    legacy_modules = (
        "albumentations",
        "albumentations.augmentations",
        "albumentations.augmentations.blur",
        "albumentations.augmentations.blur.transforms",
        "albumentations.augmentations.pixel",
        "albumentations.augmentations.pixel.transforms",
    )
    for mod_name in legacy_modules:
        if mod_name not in sys.modules:
            try:
                __import__(mod_name)
            except ImportError:
                sys.modules[mod_name] = _LegacyModuleStub(mod_name)


@dataclass
class ArUcoCalibration:
    """Calibration metrics derived from an ArUco DICT_4X4_50 marker in the frame."""

    detected: bool
    marker_ids: list[int] = field(default_factory=list)
    marker_size_mm: float = DEFAULT_MARKER_SIZE_MM
    px_per_mm: float | None = None
    mm_per_px: float | None = None
    marker_corners: list[np.ndarray] = field(default_factory=list)
    board_bbox_px: tuple[int, int, int, int] | None = None
    board_width_mm: float | None = None
    board_height_mm: float | None = None
    board_area_mm2: float | None = None
    board_measurement_source: str = "Not detected"


@dataclass
class DetectionItem:
    """Single component or defect detection with optional calibrated mm dimensions."""

    model_group: str
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    width_px: float
    height_px: float
    width_mm: float | None
    height_mm: float | None
    area_mm2: float | None
    category: str = ""
    details: str = ""


@dataclass
class InspectionResult:
    """Complete inspection result returned by the three-model CPU pipeline."""

    annotated_rgb: np.ndarray
    calibration: ArUcoCalibration
    components: list[DetectionItem]
    bare_defects: list[DetectionItem]
    solder_defects: list[DetectionItem]
    timings_ms: dict[str, float]
    passed: bool

    @property
    def total_defects(self) -> int:
        return len(self.bare_defects) + len(self.solder_defects)


@dataclass
class GeneralDetectionResult:
    """Result returned by the general-purpose CPU object detection pipeline."""

    annotated_rgb: np.ndarray
    detections: list[DetectionItem]
    class_counts: dict[str, int]
    inference_ms: float


def _disable_ultralytics_sync() -> None:
    """Disable Ultralytics online telemetry sync for clean offline/cloud operation."""
    try:
        from ultralytics import settings

        if settings.get("sync", True):
            settings.update({"sync": False})
    except Exception:
        pass


def load_general_model() -> Any:
    """Load the offline YOLO26n general object detection model for CPU inference."""
    from ultralytics import YOLO

    _disable_ultralytics_sync()
    if not GENERAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model weights: {GENERAL_MODEL_PATH}")
    return YOLO(str(GENERAL_MODEL_PATH))


def load_models() -> dict[str, Any]:
    """Load all three specialized PCB YOLO models strictly for CPU inference."""
    from ultralytics import YOLO

    _disable_ultralytics_sync()
    _ensure_legacy_albumentations_stubs()
    for path in (COMPONENT_MODEL_PATH, BARE_DEFECT_MODEL_PATH, SOLDER_DEFECT_MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing model weights: {path}")
    models = {
        "components": YOLO(str(COMPONENT_MODEL_PATH)),
        "bare_defects": YOLO(str(BARE_DEFECT_MODEL_PATH)),
        "solder_defects": YOLO(str(SOLDER_DEFECT_MODEL_PATH)),
    }
    return models


def generate_printable_aruco_png(
    marker_id: int = 0,
    marker_size_mm: float = DEFAULT_MARKER_SIZE_MM,
    dpi: int = 300,
) -> bytes:
    """Generate a 300-DPI PNG sheet containing a 20x20 mm DICT_4X4_50 ArUco marker."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    marker_px = max(64, int(round((marker_size_mm / 25.4) * dpi)))
    marker_gray = cv2.aruco.generateImageMarker(dictionary, int(marker_id), marker_px)

    quiet_px = int(round((6.0 / 25.4) * dpi))  # 6 mm quiet zone around marker
    footer_px = int(round((10.0 / 25.4) * dpi))
    canvas_w = marker_px + 2 * quiet_px
    canvas_h = marker_px + 2 * quiet_px + footer_px

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    marker_pil = Image.fromarray(marker_gray).convert("RGB")
    canvas.paste(marker_pil, (quiet_px, quiet_px))

    draw = ImageDraw.Draw(canvas)
    # Subtle cut/alignment tick marks at marker corners
    x0, y0 = quiet_px, quiet_px
    x1, y1 = quiet_px + marker_px - 1, quiet_px + marker_px - 1
    draw.rectangle([x0 - 1, y0 - 1, x1 + 1, y1 + 1], outline=(160, 160, 160), width=1)

    label_y = quiet_px + marker_px + int(round((2.0 / 25.4) * dpi))
    draw.text(
        (quiet_px, label_y),
        f"ArUco DICT_4X4_50 | ID {marker_id}\n{marker_size_mm:.0f} x {marker_size_mm:.0f} mm @ {dpi} DPI (Print 100%)",
        fill=(20, 20, 20),
    )

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", dpi=(dpi, dpi))
    return buf.getvalue()


def _estimate_board_bbox(
    image_bgr: np.ndarray,
    marker_corners: list[np.ndarray],
    component_boxes: list[tuple[float, float, float, float]],
) -> tuple[tuple[int, int, int, int] | None, str]:
    """Estimate the PCB outline in pixels using contour segmentation or component envelope."""
    h, w = image_bgr.shape[:2]
    image_area = float(h * w)

    # Mask out detected ArUco markers so the paper marker card isn't mistaken for the PCB.
    masked = image_bgr.copy()
    for corner in marker_corners:
        pts = corner.reshape(-1, 2).astype(np.int32)
        x, y, bw, bh = cv2.boundingRect(pts)
        pad = max(12, int(0.3 * max(bw, bh)))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        # Fill with median border color to suppress marker edges
        border_color = np.median(image_bgr.reshape(-1, 3), axis=0).astype(np.uint8)
        masked[y0:y1, x0:x1] = border_color

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 40, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_bbox: tuple[int, int, int, int] | None = None
    best_area = 0.0
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = float(bw * bh)
        # Ignore tiny noise and full-frame border artifacts
        if 0.05 * image_area <= area <= 0.94 * image_area and area > best_area:
            best_area = area
            best_bbox = (x, y, x + bw, y + bh)

    if best_bbox is not None:
        return best_bbox, "OpenCV PCB contour"

    if component_boxes:
        xs0 = [b[0] for b in component_boxes]
        ys0 = [b[1] for b in component_boxes]
        xs1 = [b[2] for b in component_boxes]
        ys1 = [b[3] for b in component_boxes]
        return (int(min(xs0)), int(min(ys0)), int(max(xs1)), int(max(ys1))), "Component span (fallback)"

    return None, "Not detected"


def detect_aruco_calibration(
    image_bgr: np.ndarray,
    marker_size_mm: float = DEFAULT_MARKER_SIZE_MM,
) -> ArUcoCalibration:
    """Detect DICT_4X4_50 ArUco markers and compute pixels-per-millimeter scale."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0 or not corners:
        return ArUcoCalibration(detected=False, marker_size_mm=marker_size_mm)

    flat_ids = [int(v) for v in np.asarray(ids).reshape(-1)]
    side_lengths: list[float] = []
    normalized_corners: list[np.ndarray] = []

    for corner in corners:
        pts = np.asarray(corner, dtype=np.float32).reshape(4, 2)
        normalized_corners.append(pts)
        for idx in range(4):
            p1 = pts[idx]
            p2 = pts[(idx + 1) % 4]
            side_lengths.append(float(np.linalg.norm(p2 - p1)))

    avg_side_px = float(np.mean(side_lengths))
    if avg_side_px <= 1e-6:
        return ArUcoCalibration(detected=False, marker_size_mm=marker_size_mm)

    px_per_mm = avg_side_px / float(marker_size_mm)
    mm_per_px = 1.0 / px_per_mm
    return ArUcoCalibration(
        detected=True,
        marker_ids=flat_ids,
        marker_size_mm=float(marker_size_mm),
        px_per_mm=px_per_mm,
        mm_per_px=mm_per_px,
        marker_corners=normalized_corners,
    )


def _extract_detections(
    result: Any,
    model_group: str,
    px_per_mm: float | None,
) -> list[DetectionItem]:
    """Convert an Ultralytics prediction Result into structured DetectionItems."""
    items: list[DetectionItem] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return items

    names = result.names
    xyxy_arr = boxes.xyxy.cpu().numpy()
    conf_arr = boxes.conf.cpu().numpy()
    cls_arr = boxes.cls.cpu().numpy()

    for xyxy, conf, cls_idx in zip(xyxy_arr, conf_arr, cls_arr):
        x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
        w_px = max(0.0, x2 - x1)
        h_px = max(0.0, y2 - y1)
        class_name = str(names.get(int(cls_idx), f"class_{int(cls_idx)}"))

        if px_per_mm is not None and px_per_mm > 0:
            w_mm = w_px / px_per_mm
            h_mm = h_px / px_per_mm
            area_mm2 = w_mm * h_mm
        else:
            w_mm = None
            h_mm = None
            area_mm2 = None

        if model_group == "Component":
            role_info = COMPONENT_ROLES.get(
                class_name,
                {
                    "category": "Unmapped",
                    "circuit_role": "Detected PCB element; verify schematic reference designator.",
                },
            )
            category = role_info["category"]
            details = role_info["circuit_role"]
        elif model_group == "General Object":
            category = "COCO Object"
            details = f"General object class '{class_name}' detected via YOLO26n."
        else:
            defect_info = DEFECT_DESCRIPTIONS.get(
                class_name,
                {
                    "domain": model_group,
                    "description": "Detected PCB manufacturing or assembly anomaly.",
                },
            )
            category = defect_info["domain"]
            details = defect_info["description"]

        items.append(
            DetectionItem(
                model_group=model_group,
                class_name=class_name,
                confidence=float(conf),
                bbox_xyxy=(x1, y1, x2, y2),
                width_px=w_px,
                height_px=h_px,
                width_mm=w_mm,
                height_mm=h_mm,
                area_mm2=area_mm2,
                category=category,
                details=details,
            )
        )
    return items


def _draw_label(
    image_bgr: np.ndarray,
    text: str,
    x: int,
    y: int,
    bg_color: tuple[int, int, int],
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Draw a legible filled label box anchored above (or inside) a bounding box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - th - baseline - 6)
    cv2.rectangle(image_bgr, (x, top), (x + tw + 8, top + th + baseline + 6), bg_color, -1)
    cv2.putText(
        image_bgr,
        text,
        (x + 4, top + th + 3),
        font,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def _render_annotations(
    image_bgr: np.ndarray,
    calibration: ArUcoCalibration,
    components: list[DetectionItem],
    bare_defects: list[DetectionItem],
    solder_defects: list[DetectionItem],
    show_components: bool = True,
    show_bare_defects: bool = True,
    show_solder_defects: bool = True,
    show_calibration: bool = True,
) -> np.ndarray:
    """Render bounding boxes, ArUco markers, and board contour on a copy of the image."""
    canvas = image_bgr.copy()
    line_thickness = max(2, int(round(min(canvas.shape[:2]) / 450.0)))

    if show_calibration and calibration.detected:
        for idx, pts in enumerate(calibration.marker_corners):
            poly = pts.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [poly], True, (255, 191, 0), line_thickness + 1)
            mx, my = int(pts[0][0]), int(pts[0][1])
            mid = calibration.marker_ids[idx] if idx < len(calibration.marker_ids) else idx
            _draw_label(
                canvas,
                f"ArUco #{mid} ({calibration.marker_size_mm:.0f}mm)",
                mx,
                my,
                (180, 120, 0),
            )

        if calibration.board_bbox_px is not None:
            bx1, by1, bx2, by2 = calibration.board_bbox_px
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (255, 220, 60), line_thickness)
            if calibration.board_width_mm and calibration.board_height_mm:
                _draw_label(
                    canvas,
                    f"Board ~ {calibration.board_width_mm:.1f} x {calibration.board_height_mm:.1f} mm",
                    bx1,
                    max(24, by1),
                    (140, 100, 0),
                )

    layers: list[tuple[bool, list[DetectionItem], tuple[int, int, int], str]] = [
        (show_components, components, (60, 200, 110), ""),
        (show_bare_defects, bare_defects, (0, 140, 255), "[BARE] "),
        (show_solder_defects, solder_defects, (45, 55, 235), "[SOLDER] "),
    ]

    for enabled, items, color, prefix in layers:
        if not enabled:
            continue
        for item in items:
            x1, y1, x2, y2 = [int(round(v)) for v in item.bbox_xyxy]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_thickness)
            dim_suffix = (
                f" {item.width_mm:.1f}x{item.height_mm:.1f}mm"
                if item.width_mm is not None and item.height_mm is not None
                else ""
            )
            label = f"{prefix}{item.class_name} {item.confidence:.2f}{dim_suffix}"
            _draw_label(canvas, label, x1, y1, color)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _general_class_color(class_name: str) -> tuple[int, int, int]:
    """Return a deterministic, high-contrast BGR color for a general object class."""
    palette = [
        (56, 189, 248),
        (74, 222, 128),
        (251, 146, 60),
        (244, 114, 182),
        (167, 139, 250),
        (250, 204, 21),
        (45, 212, 191),
        (248, 113, 113),
    ]
    idx = sum(ord(ch) for ch in class_name) % len(palette)
    rgb = palette[idx]
    return (rgb[2], rgb[1], rgb[0])


def run_general_detection(
    image_rgb: np.ndarray,
    model: Any,
    conf_threshold: float = 0.25,
    imgsz: int = 640,
) -> GeneralDetectionResult:
    """Run offline YOLO26n general object detection on CPU."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    t0 = time.perf_counter()
    res = model.predict(
        image_bgr,
        imgsz=imgsz,
        conf=conf_threshold,
        device="cpu",
        verbose=False,
    )[0]
    inference_ms = (time.perf_counter() - t0) * 1000.0

    detections = _extract_detections(res, "General Object", px_per_mm=None)
    canvas = image_bgr.copy()
    line_thickness = max(2, int(round(min(canvas.shape[:2]) / 450.0)))
    class_counts: dict[str, int] = {}

    for item in detections:
        class_counts[item.class_name] = class_counts.get(item.class_name, 0) + 1
        color = _general_class_color(item.class_name)
        x1, y1, x2, y2 = [int(round(v)) for v in item.bbox_xyxy]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_thickness)
        _draw_label(canvas, f"{item.class_name} {item.confidence:.2f}", x1, y1, color, text_color=(15, 15, 15))

    return GeneralDetectionResult(
        annotated_rgb=cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB),
        detections=detections,
        class_counts=dict(sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        inference_ms=inference_ms,
    )


def run_inspection(
    image_rgb: np.ndarray,
    models: dict[str, Any],
    conf_threshold: float = 0.25,
    marker_size_mm: float = DEFAULT_MARKER_SIZE_MM,
    show_components: bool = True,
    show_bare_defects: bool = True,
    show_solder_defects: bool = True,
    show_calibration: bool = True,
) -> InspectionResult:
    """Run ArUco calibration and all three CPU YOLO models on an RGB image."""
    t_total_start = time.perf_counter()
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # 1. ArUco calibration
    t0 = time.perf_counter()
    calibration = detect_aruco_calibration(image_bgr, marker_size_mm=marker_size_mm)
    aruco_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Component detection (imgsz=1024 on CPU)
    t1 = time.perf_counter()
    comp_res = models["components"].predict(
        image_bgr,
        imgsz=1024,
        conf=conf_threshold,
        device="cpu",
        verbose=False,
    )[0]
    comp_ms = (time.perf_counter() - t1) * 1000.0

    # 3. Bare-board defect detection (imgsz=1024 on CPU)
    t2 = time.perf_counter()
    bare_res = models["bare_defects"].predict(
        image_bgr,
        imgsz=1024,
        conf=conf_threshold,
        device="cpu",
        verbose=False,
    )[0]
    bare_ms = (time.perf_counter() - t2) * 1000.0

    # 4. Solder & assembly defect detection (imgsz=800 on CPU)
    t3 = time.perf_counter()
    solder_res = models["solder_defects"].predict(
        image_bgr,
        imgsz=800,
        conf=conf_threshold,
        device="cpu",
        verbose=False,
    )[0]
    solder_ms = (time.perf_counter() - t3) * 1000.0

    components = _extract_detections(comp_res, "Component", calibration.px_per_mm)
    bare_defects = _extract_detections(bare_res, "Bare-Board Defect", calibration.px_per_mm)
    solder_defects = _extract_detections(solder_res, "Solder Defect", calibration.px_per_mm)

    if calibration.detected and calibration.px_per_mm:
        board_bbox, board_source = _estimate_board_bbox(
            image_bgr,
            calibration.marker_corners,
            [c.bbox_xyxy for c in components],
        )
        if board_bbox is not None:
            bx1, by1, bx2, by2 = board_bbox
            bw_mm = (bx2 - bx1) / calibration.px_per_mm
            bh_mm = (by2 - by1) / calibration.px_per_mm
            calibration.board_bbox_px = board_bbox
            calibration.board_width_mm = bw_mm
            calibration.board_height_mm = bh_mm
            calibration.board_area_mm2 = bw_mm * bh_mm
            calibration.board_measurement_source = board_source

    annotated_rgb = _render_annotations(
        image_bgr=image_bgr,
        calibration=calibration,
        components=components,
        bare_defects=bare_defects,
        solder_defects=solder_defects,
        show_components=show_components,
        show_bare_defects=show_bare_defects,
        show_solder_defects=show_solder_defects,
        show_calibration=show_calibration,
    )

    total_ms = (time.perf_counter() - t_total_start) * 1000.0
    timings_ms = {
        "total_ms": total_ms,
        "aruco_ms": aruco_ms,
        "components_ms": comp_ms,
        "bare_defects_ms": bare_ms,
        "solder_defects_ms": solder_ms,
    }
    passed = (len(bare_defects) + len(solder_defects)) == 0

    return InspectionResult(
        annotated_rgb=annotated_rgb,
        calibration=calibration,
        components=components,
        bare_defects=bare_defects,
        solder_defects=solder_defects,
        timings_ms=timings_ms,
        passed=passed,
    )
