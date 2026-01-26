from __future__ import annotations

import base64
import json
import mimetypes
import os
import struct
import sys
from array import array
from pathlib import Path
from typing import Dict, Tuple
from urllib.parse import parse_qs, urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


DATA_ROOT = Path(__file__).resolve().parent
WEB_ROOT = DATA_ROOT / "web"


DTYPE_MAP = {
    2: ("B", 1),   # uint8
    4: ("h", 2),   # int16
    8: ("i", 4),   # int32
    16: ("f", 4),  # float32
    64: ("d", 8),  # float64
    256: ("b", 1),  # int8
    512: ("H", 2),  # uint16
    768: ("I", 4),  # uint32
}


class NiftiVolume:
    def __init__(self, path: Path) -> None:
        self.path = path
        (
            self.width,
            self.height,
            self.depth,
            self._dtype_code,
            self._vox_offset,
            self._endian,
        ) = self._read_header()
        self._data = self._read_data()
        self._min, self._max = self._compute_min_max()

    def _read_header(self) -> Tuple[int, int, int, int, float, str]:
        with self.path.open("rb") as f:
            header = f.read(348)
        sizeof_hdr_le = struct.unpack("<I", header[0:4])[0]
        if sizeof_hdr_le == 348:
            endian = "<"
        else:
            sizeof_hdr_be = struct.unpack(">I", header[0:4])[0]
            if sizeof_hdr_be != 348:
                raise ValueError(f"Unrecognized NIfTI header in {self.path}")
            endian = ">"

        dim = struct.unpack(f"{endian}8h", header[40:56])
        dim0 = dim[0]
        if dim0 < 3:
            raise ValueError(f"Unexpected NIfTI dims in {self.path}")
        width, height, depth = dim[1], dim[2], dim[3]
        datatype = struct.unpack(f"{endian}h", header[70:72])[0]
        vox_offset = struct.unpack(f"{endian}f", header[108:112])[0]
        return width, height, depth, datatype, vox_offset, endian

    def _read_data(self) -> array:
        type_info = DTYPE_MAP.get(self._dtype_code)
        if not type_info:
            raise ValueError(f"Unsupported NIfTI datatype {self._dtype_code}")
        typecode, _ = type_info
        count = self.width * self.height * self.depth
        data = array(typecode)
        with self.path.open("rb") as f:
            f.seek(int(self._vox_offset))
            data.fromfile(f, count)
        if (self._endian == ">" and sys.byteorder == "little") or (
            self._endian == "<" and sys.byteorder == "big"
        ):
            data.byteswap()
        return data

    def _compute_min_max(self) -> Tuple[float, float]:
        vmin = float("inf")
        vmax = float("-inf")
        for value in self._data:
            if value != value:
                continue
            if value < vmin:
                vmin = value
            if value > vmax:
                vmax = value
        if vmin == float("inf"):
            vmin = 0.0
            vmax = 1.0
        return vmin, vmax

    def get_slice_bytes(self, z_index: int) -> bytes:
        if z_index < 0 or z_index >= self.depth:
            raise IndexError("z_index out of range")
        size = self.width * self.height
        start = z_index * size
        out = bytearray(size)
        vmin, vmax = self._min, self._max
        denom = vmax - vmin
        if denom == 0:
            return bytes(out)
        data = self._data
        for i in range(size):
            value = data[start + i]
            if value != value:
                value = vmin
            scaled = int((value - vmin) * 255 / denom)
            if scaled < 0:
                scaled = 0
            elif scaled > 255:
                scaled = 255
            out[i] = scaled
        return bytes(out)


def scan_patients() -> Dict[str, Dict[str, Path]]:
    patients: Dict[str, Dict[str, Path]] = {}
    for entry in DATA_ROOT.iterdir():
        if not entry.is_dir():
            continue
        t2_files = list(entry.glob("*_t2_aligned.nii"))
        ff_files = list(entry.glob("*_ff_normalized.nii"))
        seg_files = list(entry.glob("*_segmentation.nii"))
        if not t2_files or not ff_files or not seg_files:
            continue
        patients[entry.name] = {
            "t2": t2_files[0],
            "ff": ff_files[0],
            "seg": seg_files[0],
        }
    return patients


PATIENTS = scan_patients()
VOLUME_CACHE: Dict[Path, NiftiVolume] = {}


def get_volume(path: Path) -> NiftiVolume:
    volume = VOLUME_CACHE.get(path)
    if volume is None:
        volume = NiftiVolume(path)
        VOLUME_CACHE[path] = volume
    return volume


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/patients":
            self._handle_patients()
            return
        if parsed.path == "/api/meta":
            self._handle_meta(parse_qs(parsed.query))
            return
        if parsed.path == "/api/slices":
            self._handle_slices(parse_qs(parsed.query))
            return
        self._handle_static(parsed.path)

    def _handle_patients(self) -> None:
        payload = {"patients": sorted(PATIENTS.keys())}
        self._send_json(payload)

    def _handle_meta(self, query: Dict[str, list]) -> None:
        patient = _get_param(query, "patient")
        if not patient or patient not in PATIENTS:
            self._send_error(400, "Unknown patient")
            return
        t2_path = PATIENTS[patient]["t2"]
        volume = get_volume(t2_path)
        payload = {
            "width": volume.width,
            "height": volume.height,
            "depth": volume.depth,
        }
        self._send_json(payload)

    def _handle_slices(self, query: Dict[str, list]) -> None:
        patient = _get_param(query, "patient")
        if not patient or patient not in PATIENTS:
            self._send_error(400, "Unknown patient")
            return
        try:
            start = int(_get_param(query, "start") or 0)
            count = int(_get_param(query, "count") or 25)
        except ValueError:
            self._send_error(400, "Invalid start/count")
            return
        if count < 1:
            self._send_error(400, "count must be >= 1")
            return

        t2_path = PATIENTS[patient]["t2"]
        ff_path = PATIENTS[patient]["ff"]
        seg_path = PATIENTS[patient]["seg"]
        t2_volume = get_volume(t2_path)
        ff_volume = get_volume(ff_path)
        seg_volume = get_volume(seg_path)

        depth = min(t2_volume.depth, ff_volume.depth, seg_volume.depth)
        if start < 0 or start >= depth:
            self._send_error(400, "start out of range")
            return
        count = min(count, depth - start)

        slices = []
        for z in range(start, start + count):
            t2_bytes = t2_volume.get_slice_bytes(z)
            ff_bytes = ff_volume.get_slice_bytes(z)
            seg_bytes = seg_volume.get_slice_bytes(z)
            slices.append(
                {
                    "z": z,
                    "t2": base64.b64encode(t2_bytes).decode("ascii"),
                    "ff": base64.b64encode(ff_bytes).decode("ascii"),
                    "seg": base64.b64encode(seg_bytes).decode("ascii"),
                }
            )
        payload = {
            "width": t2_volume.width,
            "height": t2_volume.height,
            "depth": depth,
            "start": start,
            "count": count,
            "slices": slices,
        }
        self._send_json(payload)

    def _handle_static(self, path: str) -> None:
        if path == "/":
            path = "/index.html"
        safe_path = Path(path.lstrip("/"))
        full_path = (WEB_ROOT / safe_path).resolve()
        if not str(full_path).startswith(str(WEB_ROOT.resolve())):
            self._send_error(403, "Forbidden")
            return
        if not full_path.exists() or not full_path.is_file():
            self._send_error(404, "Not found")
            return
        content_type, _ = mimetypes.guess_type(full_path.name)
        if not content_type:
            content_type = "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(full_path.stat().st_size))
        self.end_headers()
        with full_path.open("rb") as f:
            self.wfile.write(f.read())

    def _send_json(self, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, code: int, message: str) -> None:
        payload = {"error": message}
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _get_param(query: Dict[str, list], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    return values[0]


def main() -> None:
    port = int(os.environ.get("PORT", "5090"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    print(f"Serving on http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
