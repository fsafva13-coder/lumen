"""Download the YuNet face detector (OpenCV Zoo) into models/ and verify it."""
import hashlib
import urllib.request
from pathlib import Path

URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
       "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
DEST = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"

if __name__ == "__main__":
    DEST.parent.mkdir(exist_ok=True)
    if not DEST.exists():
        print("Downloading", URL)
        urllib.request.urlretrieve(URL, DEST)
    digest = hashlib.sha256(DEST.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"Checksum mismatch for {DEST.name}; delete it and re-run.")
    print("OK:", DEST)
