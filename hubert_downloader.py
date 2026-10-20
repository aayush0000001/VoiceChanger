"""Background worker to download the HuBERT ContentVec base model on demand without freezing the UI."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal

import appdata

HUBERT_FILENAME = "vec-768-layer-12.onnx"
HUBERT_SHA256 = "b3886e7dff1495cda514f94f4680a7b1261e05d6929f5c764cdb17934b413c2a"
HUBERT_MIRRORS = [
    "https://huggingface.co/DogManTC/test-rvc-onnx/resolve/main/vec-768-layer-12.onnx",
    "https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/resolve/main/vec-768-layer-12.onnx",
    "https://huggingface.co/ozada/onnx_rvc/resolve/main/vec-768-layer-12.onnx",
]


def default_hubert_dest() -> Path:
    dest_dir = appdata.roaming_dir() / "models"
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / HUBERT_FILENAME


class HuBERTDownloadWorker(QThread):
    progress = pyqtSignal(int, float, float)  # percent, downloaded_mb, total_mb
    finished = pyqtSignal(bool, str)          # success, message

    def __init__(self, dest_path: Optional[Path] = None, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.dest_path = dest_path or default_hubert_dest()
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        self.dest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.dest_path.with_suffix(".tmp")
        for url in HUBERT_MIRRORS:
            if self._is_cancelled:
                return
            try:
                hasher = hashlib.sha256()
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                )
                with urllib.request.urlopen(req, timeout=25) as resp, open(temp_path, "wb") as f:
                    total = int(resp.headers.get("Content-Length", 0))
                    total_mb = total / (1024 * 1024) if total else 360.2
                    downloaded = 0
                    chunk_size = 1024 * 256
                    while not self._is_cancelled:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        hasher.update(chunk)
                        downloaded += len(chunk)
                        pct = int((downloaded / total) * 100) if total else 0
                        self.progress.emit(pct, downloaded / (1024 * 1024), total_mb)

                if self._is_cancelled:
                    if temp_path.exists():
                        try:
                            temp_path.unlink()
                        except Exception:
                            pass
                    return

                # Verify SHA256 integrity check against canonical HuBERT weights
                computed_hash = hasher.hexdigest().lower()
                if computed_hash != HUBERT_SHA256.lower():
                    raise ValueError(f"Checksum mismatch for {url}: expected {HUBERT_SHA256}, got {computed_hash}")

                if temp_path.exists() and temp_path.stat().st_size > 100_000_000:
                    if self.dest_path.exists():
                        try:
                            self.dest_path.unlink()
                        except Exception:
                            pass
                    temp_path.rename(self.dest_path)
                    self.finished.emit(True, "HuBERT ContentVec base model verified and installed successfully.")
                    return
            except Exception:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                continue

        if not self._is_cancelled:
            self.finished.emit(
                False, "Could not download the base model from any mirror. Check your internet connection."
            )
