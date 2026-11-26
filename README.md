# Objectified — AI Object Detection Suite

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://objectified.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU--Only-EE4C2C?logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/YOLO-Ultralytics-00FFFF)
![OpenCV](https://img.shields.io/badge/OpenCV-Contrib%20%2B%20ArUco-5C3EE8?logo=opencv&logoColor=white)

**Objectified** is a modular, CPU-only computer vision web platform built with Python, Streamlit, Ultralytics YOLO, and OpenCV. It pairs real-time everyday object recognition with a pluggable industrial **Automated Optical Inspection (AOI)** module for Printed Circuit Boards—running 100% offline on standard laptops without requiring a dedicated CUDA GPU, as well as online via Streamlit Community Cloud.

**Live Demo:** [https://objectified.streamlit.app](https://objectified.streamlit.app)

---

## Core Modules

### 1. General Object Detection (Primary Mode)
* Powered by `yolo26n.pt` across **80 COCO object classes** running on `device="cpu"`.
* Supports **High-Res Photo Upload**, **Local OpenCV Live Stream** (`cv2.VideoCapture` for local runs), **Browser Live Stream** (`streamlit-webrtc` for cloud deployments), and **Browser Snapshot** (`st.camera_input`).
* Real-time **Confidence Threshold** and **Inference Resolution (`imgsz`)** sliders, live latency telemetry (`ms`), and a dynamic **Detected Class Count Summary**.

### 2. PCB Inspection (Specialized Industrial Module)
* Synchronized **three-model YOLO ensemble** for automated optical inspection:
  * **Component Detection (`fine_best.pt`)** — Localizes and classifies 13 electronic component categories (`Resistor`, `Capacitor`, `Electrolytic Capacitor`, `IC`, `Inductor`, `Diode`, `LED`, `Transistor`, `Oscillator`, `Connector`, `Switch/Button`, `Other`, `Marker`).
  * **Bare-Board Defects (`bare defects.pt`)** — Identifies 7 bare-board copper anomalies.
  * **Solder & Assembly Defects (`best solder defect.pt`)** — Detects 9 solder and assembly defects.
* Automated **PASS / FAIL** quality verdict banner with per-model CPU scan times (`ms`).
* Toggleable visual overlay layers and **Side-by-Side (Original vs. Annotated)** inspection view.
* **Component Inspection & Circuit Role Table** mapping every detected part to its circuit function via an embedded local dictionary, complete with **CSV export**.

### 3. OpenCV ArUco Optical Metrology
* Built-in `DICT_4X4_50` fiducial marker detection (default `20.00 mm × 20.00 mm`, configurable IDs `0–49`).
* Computes a live pixel-to-millimeter ratio to measure overall PCB board dimensions and individual component footprints in millimeters without specialized optical rigs.
* Includes a sidebar **Download Printable 20mm Marker (PNG)** generator at 300 DPI.

---

## Model & Pipeline Overview

| Module | Weight File | Task | Classes | Default `imgsz` | Runtime |
| :--- | :--- | :--- | :---: | :---: | :--- |
| General Detection | `models/yolo26n.pt` | Everyday object detection | 80 | 480 / 640 | CPU |
| PCB Components | `models/fine_best.pt` | Electronic component classification | 13 | 1024 | CPU |
| Bare-Board QC | `models/bare defects.pt` | Copper & bare-board defect detection | 7 | 1024 | CPU |
| Solder QC | `models/best solder defect.pt` | Solder & assembly defect detection | 9 | 800 | CPU |

---

## Project Structure

```text
pcb-object-detection/
├── app.py                              # Main Streamlit UI (dual-mode router & webcam handlers)
├── pcb_inspector.py                    # 3-model PCB inference, ArUco metrology & circuit role mapping
├── requirements.txt                    # CPU-only PyTorch, Ultralytics, headless OpenCV & WebRTC deps
├── packages.txt                        # Linux system libraries (libgl1, libglib2.0-0) for cloud builds
├── .streamlit/
│   └── config.toml                     # Offline Streamlit configuration (telemetry disabled)
├── .yolo_config/
│   └── Ultralytics/
│       └── settings.json               # Offline Ultralytics configuration (sync disabled)
└── models/
    ├── yolo26n.pt                      # General COCO object detection weights
    ├── fine_best.pt                    # PCB component detection weights
    ├── bare defects.pt                 # Bare-board copper defect weights
    └── best solder defect.pt           # Solder & assembly defect weights
```

---

## Local Setup (CPU-Only, 100% Offline)

### 1. Clone the Repository
```bash
git clone [https://github.com/niranjangp2007/pcb-object-detection.git](https://github.com/niranjangp2007/pcb-object-detection.git)
cd pcb-object-detection
```

### 2. Create a Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Lightweight CPU Dependencies
```bash
pip install --no-cache-dir torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install --no-cache-dir ultralytics opencv-contrib-python streamlit streamlit-webrtc av
```

### 4. Launch the Application
```bash
streamlit run app.py
```
Open `http://127.0.0.1:8501` in your browser.

---

## Usage Tips

* **Best PCB Inspection Results:** Capture a well-lit, flat **90° top-down photo** of the populated side of the PCB and start with the Confidence Threshold between **`0.25` and `0.35`**.
* **Accurate Millimeter Calibration:** Download the 20 mm ArUco marker from the sidebar, print it at **100% actual scale** (disable "Fit to Page"), and place it flat beside the circuit board in the same focal plane.
* **Local vs. Cloud Webcam:**
  * Running locally supports zero-latency **Local OpenCV Live Stream (`cv2.VideoCapture`)**.
  * Cloud deployments (`objectified.streamlit.app`) automatically default to **Browser Live Stream (WebRTC)** and **Browser Snapshot (`st.camera_input`)** since remote Linux containers have no attached `/dev/video*` hardware.

---

## Cloud Deployment (Streamlit Community Cloud)

The repository is pre-configured for Streamlit Community Cloud:
* `requirements.txt` pins `--extra-index-url https://download.pytorch.org/whl/cpu` and `opencv-contrib-python-headless` to avoid heavy CUDA downloads and GUI library errors.
* `packages.txt` installs `libgl1` and `libglib2.0-0` for headless OpenCV execution.
* Models are cached with `@st.cache_resource` and loaded lazily per selected module to stay within cloud memory limits.

---

## Author

**Niranjan G P**  
GitHub: [@niranjangp2007](https://github.com/niranjangp2007)

**Sheshashree Arunkumar**  
GitHub: [@sh-ak](https://github.com/sh-ak)


