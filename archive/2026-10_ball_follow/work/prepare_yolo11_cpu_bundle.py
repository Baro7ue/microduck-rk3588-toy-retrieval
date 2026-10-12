"""Prepare fixed offline Python CPU wheels and reused verification images.

Downloads from PyPI, verifies published SHA-256, records sources.
No board connection, package installation, robot process, or motion.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import urllib.request

root = Path(__file__).resolve().parents[1]
bundle = root / "work/vision-offline/yolo11n-cpu"
wheels = bundle / "wheelhouse"
wheels.mkdir(parents=True, exist_ok=True)
sources = []
pins = [
    ("onnxruntime", "1.28.0", "cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"),
    ("flatbuffers", "25.12.19", "py2.py3-none-any.whl"),
    ("packaging", "25.0", "py3-none-any.whl"),
    ("protobuf", "6.33.6", "cp39-abi3-manylinux2014_aarch64.whl"),
]
requirements = []
for name, version, suffix in pins:
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    metadata = json.load(urllib.request.urlopen(url))
    matches = [f for f in metadata["urls"] if f["filename"].endswith(suffix)]
    assert len(matches) == 1, matches
    item = matches[0]
    target = wheels / item["filename"]
    with urllib.request.urlopen(item["url"]) as response:
        target.write_bytes(response.read())
    sha = hashlib.sha256(target.read_bytes()).hexdigest()
    assert sha == item["digests"]["sha256"], target
    requirements.append(f"{name}=={version} --hash=sha256:{sha}")
    sources.append({"file": str(target.relative_to(bundle)), "url": item["url"],
                    "metadata_url": url, "sha256": sha})
    print("verified", target.name, flush=True)

old = root / "work/npu-offline/lite2-2.3.2-cp311"
old_sources = json.loads((old / "sources.json").read_text())
for name, version in [("numpy", "1.26.4"), ("opencv-python-headless", "4.11.0.86")]:
    candidates = list((old / "wheelhouse").glob(name.replace("-", "_") + "-" + version + "-*.whl"))
    assert len(candidates) == 1, candidates
    target = wheels / candidates[0].name
    shutil.copy2(candidates[0], target)
    sha = hashlib.sha256(target.read_bytes()).hexdigest()
    requirements.append(f"{name}=={version} --hash=sha256:{sha}")
    sources.append({"file": str(target.relative_to(bundle)), "sha256": sha,
                    "reused_from": str(candidates[0]), "original_manifest": str(old / "sources.json")})

bus = root / "work/npu-offline/yolov5-rk3588-smoke/rknn_model_zoo/examples/yolov5/model/bus.jpg"
assert hashlib.sha256(bus.read_bytes()).hexdigest() == "fb4914d123d97c440cd127ef0e98d4bdc68cd88e2683657528928b4a34014e16"
shutil.copy2(bus, bundle / "images/bus.jpg")
camera = root / "task/evidence/G4/camera_transport_20261008_025210_277295Z/camera_first_rotated.png"
shutil.copy2(camera, bundle / "images/camera_floor.png")
sources += [
    {"file": "images/bus.jpg", "url": "https://raw.githubusercontent.com/airockchip/rknn-toolkit2/42aa1d426c0a9e0869b6374edba009f7208a1926/rknpu2/examples/rknn_yolov5_demo/model/bus.jpg",
     "sha256": hashlib.sha256(bus.read_bytes()).hexdigest()},
    {"file": "images/camera_floor.png", "reused_from": str(camera),
     "sha256": hashlib.sha256(camera.read_bytes()).hexdigest()},
]
(bundle / "requirements-lock.txt").write_text("\n".join(requirements) + "\n")
(bundle / "sources.json").write_text(json.dumps({"prepared_at": datetime.now(timezone.utc).isoformat(), "files": sources}, indent=2) + "\n")
