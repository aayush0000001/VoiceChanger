"""Utility script to download standard RVC support models (HuBERT ContentVec & RMVPE)
for hardware-accelerated neural voice conversion on RTX / DirectML GPUs.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"

# Known SHA-256 of a file, checked after download and on a file that is already there (None: not published).
SHA256 = {
    "vec-768-layer-12.onnx": "b3886e7dff1495cda514f94f4680a7b1261e05d6929f5c764cdb17934b413c2a",
    "rmvpe.onnx": None,
}

MODELS = {
    "vec-768-layer-12.onnx": [
        "https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/resolve/main/vec-768-layer-12.onnx",
        "https://huggingface.co/DogManTC/test-rvc-onnx/resolve/main/vec-768-layer-12.onnx",
        "https://huggingface.co/ozada/onnx_rvc/resolve/main/vec-768-layer-12.onnx",
    ],
    "rmvpe.onnx": [
        "https://huggingface.co/DogManTC/test-rvc-onnx/resolve/main/rmvpe.onnx",
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx",
    ],
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_file(url_list: list[str], dest_path: Path, sha256: str | None = None) -> bool:
    if dest_path.exists() and dest_path.stat().st_size > 10_000_000:
        if sha256 is None or file_sha256(dest_path) == sha256.lower():
            print(f"[{dest_path.name}] Already downloaded ({dest_path.stat().st_size / (1024*1024):.1f} MB). Skipping.")
            return True
        print(f"[{dest_path.name}] The file there is damaged (checksum mismatch). Downloading it again.")

    temp_path = dest_path.with_suffix(".tmp")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    for url in url_list:
        print(f"Connecting to: {url} ...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(temp_path, "wb") as out_file:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 1024 * 512
                start_time = time.monotonic()
                last_print = 0.0

                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_print >= 0.5 or (total and downloaded >= total):
                        last_print = now
                        pct = (downloaded / total * 100) if total else 0.0
                        mb = downloaded / (1024 * 1024)
                        speed = mb / max(0.01, now - start_time)
                        sys.stdout.write(f"\r  -> Downloading {dest_path.name}: {mb:.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s ")
                        sys.stdout.flush()

            if total and downloaded != total:
                raise OSError(f"connection closed after {downloaded} of {total} bytes")
            if sha256 is not None and file_sha256(temp_path) != sha256.lower():
                raise ValueError("checksum mismatch (the file is damaged or not the expected model)")
            os.replace(temp_path, dest_path)
            print(f"\n  [SUCCESS] {dest_path.name} downloaded successfully.")
            return True
        except Exception as exc:
            print(f"\n  [WARNING] Failed downloading from {url}: {exc}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            continue

    print(f"  [ERROR] Could not download {dest_path.name} from any mirror.")
    return False


def main() -> int:
    print("=" * 65)
    print("V-COMMS // RVC Neural Voice Model Asset Setup (RTX / DirectML)")
    print("=" * 65)
    print(f"Target directory: {MODELS_DIR}\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    success = True
    for model_name, urls in MODELS.items():
        dest = MODELS_DIR / model_name
        ok = download_file(urls, dest, SHA256.get(model_name))
        if not ok:
            success = False

    print("\n" + "=" * 65)
    if success:
        print("ALL RVC ASSETS READY FOR REAL-TIME GPU INFERENCE.")
    else:
        print("SOME ASSETS FAILED TO DOWNLOAD. CHECK NETWORK CONNECTION.")
    print("=" * 65)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
