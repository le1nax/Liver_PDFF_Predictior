from __future__ import annotations

import io
import secrets
from pathlib import Path
from typing import Dict, List

import numpy as np
import SimpleITK as sitk
import yaml
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
TEST_DATA_DIR = APP_DIR / "test_data"
DATA_DIR = TEST_DATA_DIR / "data"
RESULTS_DIR = APP_DIR / "results"

HOST = "0.0.0.0"
PORT = 1337
PASSWORD = "pdffclassify!"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")

# In-memory tokens for this simple viewer session
TOKENS: Dict[str, bool] = {}

# ---------------------------------------------------------------------------
# YAML test set loading & volume cache
# ---------------------------------------------------------------------------
STUDY_CASES: List[dict] = []
PATIENT_INDEX: Dict[str, dict] = {}

_VOLUME_CACHE: Dict[str, np.ndarray] = {}
_VOLUME_CACHE_ORDER: list[str] = []
_VOLUME_CACHE_MAX = 12


def _cache_get(key: str) -> np.ndarray | None:
    return _VOLUME_CACHE.get(key)


def _cache_set(key: str, value: np.ndarray) -> None:
    if key in _VOLUME_CACHE:
        return
    _VOLUME_CACHE[key] = value
    _VOLUME_CACHE_ORDER.append(key)
    if len(_VOLUME_CACHE_ORDER) > _VOLUME_CACHE_MAX:
        old = _VOLUME_CACHE_ORDER.pop(0)
        _VOLUME_CACHE.pop(old, None)


def _classify_nifti(filename: str) -> str | None:
    name = filename.lower()
    if "segmentation" in name:
        return "mask"
    if "t2" in name:
        return "t2"
    if "ff" in name:
        return "ff"
    return None


def _load_study() -> None:
    """Parse study_test_set.yaml and discover NIfTI files per patient."""
    yaml_path = TEST_DATA_DIR / "study_test_set.yaml"
    if not yaml_path.exists():
        print(f"WARNING: {yaml_path} not found")
        return
    with open(yaml_path) as f:
        manifest = yaml.safe_load(f)
    for entry in manifest.get("cases", []):
        pid = str(entry["patient_id"])
        patient_dir = DATA_DIR / pid
        files: Dict[str, str] = {}
        num_slices = 0
        if patient_dir.is_dir():
            for nii in sorted(patient_dir.glob("*.nii.gz")):
                kind = _classify_nifti(nii.name)
                if kind:
                    files[kind] = nii.name
                    if num_slices == 0:
                        try:
                            reader = sitk.ImageFileReader()
                            reader.SetFileName(str(nii))
                            reader.ReadImageInformation()
                            num_slices = int(reader.GetSize()[2])
                        except Exception:
                            pass
        case = {
            "patient_id": pid,
            "median_ff": entry.get("median_ff"),
            "bin": entry.get("bin"),
            "files": files,
            "num_slices": num_slices,
        }
        STUDY_CASES.append(case)
        PATIENT_INDEX[pid] = case
    with_data = sum(1 for c in STUDY_CASES if c["files"])
    print(f"Loaded {len(STUDY_CASES)} cases from {yaml_path} ({with_data} with image data)")


def _load_volume(patient_id: str, kind: str) -> np.ndarray:
    cache_key = f"{patient_id}:{kind}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    case = PATIENT_INDEX.get(patient_id)
    if not case:
        raise FileNotFoundError(f"Unknown patient: {patient_id}")
    filename = case["files"].get(kind)
    if not filename:
        raise FileNotFoundError(f"No {kind} file for {patient_id}")
    path = DATA_DIR / patient_id / filename
    image = sitk.ReadImage(str(path))
    volume = sitk.GetArrayFromImage(image)
    _cache_set(cache_key, volume)
    return volume


def _slice_to_png(slice_2d: np.ndarray, kind: str) -> bytes:
    if kind == "mask":
        arr = ((slice_2d > 0).astype(np.uint8) * 255)
    elif kind == "t2":
        p1, p99 = np.percentile(slice_2d, [1, 99])
        if p99 <= p1:
            arr = np.zeros_like(slice_2d, dtype=np.uint8)
        else:
            scaled = (slice_2d - p1) / (p99 - p1)
            arr = (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)
    elif kind == "ff":
        ff = slice_2d / 100.0 if slice_2d.max() > 1.5 else slice_2d
        arr = (np.clip(ff, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        arr = np.clip(slice_2d, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


_load_study()

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _is_authorized(req) -> bool:
    token = req.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not token:
        token = req.args.get("token", "").strip()
    return bool(token) and TOKENS.get(token, False)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/app.js")
def app_js():
    return send_from_directory(STATIC_DIR, "app.js")


@app.route("/styles.css")
def styles_css():
    return send_from_directory(STATIC_DIR, "styles.css")


@app.route("/api/health")
def health():
    return jsonify({"ok": True})


@app.route("/api/auth", methods=["POST"])
def auth():
    data = request.get_json(silent=True) or {}
    if data.get("password") != PASSWORD:
        return jsonify({"ok": False}), 401
    token = secrets.token_urlsafe(24)
    TOKENS[token] = True
    return jsonify({"ok": True, "token": token})


@app.route("/api/study")
def study():
    if not _is_authorized(request):
        return jsonify({"error": "unauthorized"}), 401
    cases = []
    for c in STUDY_CASES:
        cases.append({
            "patient_id": c["patient_id"],
            "num_slices": c["num_slices"],
            "kinds": list(c["files"].keys()),
            "median_ff": c.get("median_ff"),  # DEBUG
        })
    return jsonify({
        "ok": True,
        "total_cases": len(STUDY_CASES),
        "cases": cases,
    })


@app.route("/api/info/<patient_id>")
def patient_info(patient_id: str):
    if not _is_authorized(request):
        return jsonify({"error": "unauthorized"}), 401
    case = PATIENT_INDEX.get(patient_id)
    if not case:
        return jsonify({"error": "not found"}), 404
    result = {"patient_id": patient_id, "kinds": {}}
    for kind in case["files"]:
        try:
            vol = _load_volume(patient_id, kind)
            result["kinds"][kind] = {
                "shape": list(vol.shape),
                "slices": int(vol.shape[0]),
            }
        except Exception as e:
            result["kinds"][kind] = {"error": str(e)}
    return jsonify(result)


@app.route("/api/slice/<patient_id>/<kind>/<int:slice_idx>")
def get_slice(patient_id: str, kind: str, slice_idx: int):
    if not _is_authorized(request):
        return jsonify({"error": "unauthorized"}), 401
    try:
        volume = _load_volume(patient_id, kind)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    slice_idx = max(0, min(slice_idx, volume.shape[0] - 1))
    png_bytes = _slice_to_png(volume[slice_idx], kind)
    resp = app.response_class(png_bytes, mimetype="image/png")
    # Slices are immutable for a given patient/kind/slice_idx; enable browser caching.
    resp.headers["Cache-Control"] = "private, max-age=86400, immutable"
    return resp


GRADE_LABELS = {
    "healthy": "Healthy (0-5%)",
    "mild": "Mild Steatosis (5-15%)",
    "moderate": "Moderate Steatosis (15-25%)",
    "severe": "Severe Steatosis (>25%)",
}

BIN_TO_GRADE = {
    "0-0.05": "healthy",
    "0.05-0.15": "mild",
    "0.15-0.25": "moderate",
    "0.25-1.0": "severe",
}


@app.route("/api/submit", methods=["POST"])
def submit():
    if not _is_authorized(request):
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    classifications = data.get("classifications", {})
    if not classifications:
        return jsonify({"error": "no classifications provided"}), 400

    results = []
    for pid, grade in classifications.items():
        case = PATIENT_INDEX.get(pid)
        gt_bin = case["bin"] if case else None
        gt_grade = BIN_TO_GRADE.get(gt_bin) if gt_bin else None
        results.append({
            "patient_id": pid,
            "classification": grade,
            "classification_label": GRADE_LABELS.get(grade, grade),
            "ground_truth": gt_grade,
            "ground_truth_label": GRADE_LABELS.get(gt_grade, str(gt_grade)),
            "correct": grade == gt_grade if gt_grade else None,
        })

    total = len(results)
    correct = sum(1 for r in results if r["correct"] is True)
    incorrect = sum(1 for r in results if r["correct"] is False)

    payload = {
        "submitted_at": datetime.now().isoformat(),
        "total_classified": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": round(correct / total, 4) if total > 0 else 0,
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    out_path = RESULTS_DIR / filename
    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)

    print(f"Results saved to {out_path} ({correct}/{total} correct)")
    return jsonify({
        "ok": True,
        "filename": filename,
        "total": total,
        "correct": correct,
        "accuracy": payload["accuracy"],
    })


# ---------------------------------------------------------------------------
# SSL + main
# ---------------------------------------------------------------------------


def _get_ssl_context():
    """Create a self-signed SSL context for HTTPS support."""
    import ssl
    import tempfile
    import subprocess

    cert_dir = Path(tempfile.gettempdir()) / "study_viewer_ssl"
    cert_dir.mkdir(exist_ok=True)
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"

    if not cert_file.exists() or not key_file.exists():
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", str(key_file), "-out", str(cert_file),
                "-days", "365", "-nodes",
                "-subj", "/CN=localhost",
            ],
            check=True,
            capture_output=True,
        )

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(cert_file), str(key_file))
    return ctx


if __name__ == "__main__":
    print("=" * 80)
    print("Study Viewer Backend")
    print("=" * 80)
    print(f"Starting server on https://localhost:{PORT}")
    print("Use the shared password to access the study view.")
    print("=" * 80)
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
