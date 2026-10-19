"""Utility script to download standard RVC support models (HuBERT ContentVec & RMVPE)
for hardware-accelerated neural voice conversion on RTX / DirectML GPUs.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"

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


def download_file(url_list: list[str], dest_path: Path) -> bool:
    if dest_path.exists() and dest_path.stat().st_size > 10_000_000:
        print(f"[{dest_path.name}] Already downloaded ({dest_path.stat().st_size / (1024*1024):.1f} MB). Skipping.")
        return True

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

                print(f"\n  [SUCCESS] {dest_path.name} downloaded successfully.")
                if temp_path.exists():
                    if dest_path.exists():
                        dest_path.unlink()
                    temp_path.rename(dest_path)
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
        ok = download_file(urls, dest)
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
