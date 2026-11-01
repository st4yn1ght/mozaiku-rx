"""Daftar model, unduhan otomatis, dan verifikasi SHA-256.

Bobot model berasal dari proyek open-source Lada (lisensi AGPL-3.0),
di-hosting di HuggingFace: https://huggingface.co/ladaapp/lada
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HF_BASE = "https://huggingface.co/ladaapp/lada/resolve/main/"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    kind: str          # "detection" | "restoration"
    filename: str
    sha256: str
    size_bytes: int
    description: str

    @property
    def url(self) -> str:
        return f"{HF_BASE}{self.filename}?download=true"


MODELS: dict[str, ModelSpec] = {
    "v4-accurate": ModelSpec(
        "v4-accurate", "detection", "lada_mosaic_detection_model_v4_accurate.pt",
        "c244d7e49d8f88e264b8dc15f91fb21f5908ad8fb6f300b7bc88462d0801bc1f", 45136630,
        "Deteksi YOLO11m-seg - paling akurat (default)"),
    "v4-fast": ModelSpec(
        "v4-fast", "detection", "lada_mosaic_detection_model_v4_fast.pt",
        "9a6b660d1d3e3797d39515e08b0e72fcc59815f38279faa7a4ab374ab2c1e3b4", 5981796,
        "Deteksi YOLO11n-seg - cepat"),
    "basicvsrpp-v1.2": ModelSpec(
        "basicvsrpp-v1.2", "restoration", "lada_mosaic_restoration_model_generic_v1.2.pth",
        "d404152576ce64fb5b2f315c03062709dac4f5f8548934866cd01c823c8104ee", 78441770,
        "Restorasi BasicVSR++ generik v1.2 (default)"),
}

DEFAULT_DETECTION = "v4-accurate"
DEFAULT_RESTORATION = "basicvsrpp-v1.2"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_model_dir() -> Path:
    env = os.environ.get("MOSAIC_RESTORE_MODELS")
    if env:
        return Path(env)
    local = project_root() / "models"
    try:
        local.mkdir(parents=True, exist_ok=True)
        probe = local / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return local
    except OSError:
        pass
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "mosaic-restore" / "models"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _marker(path: Path) -> Path:
    return path.with_name(path.name + ".verified")


def _is_verified(path: Path, spec: ModelSpec) -> bool:
    marker = _marker(path)
    if not path.exists() or not marker.exists():
        return False
    st = path.stat()
    expected = f"{spec.sha256} {st.st_size} {int(st.st_mtime)}"
    try:
        return marker.read_text().strip() == expected and st.st_size == spec.size_bytes
    except OSError:
        return False


def verify(path: Path, spec: ModelSpec) -> bool:
    if _is_verified(path, spec):
        return True
    if not path.exists() or path.stat().st_size != spec.size_bytes:
        return False
    if sha256_file(path) != spec.sha256:
        return False
    st = path.stat()
    try:
        _marker(path).write_text(f"{spec.sha256} {st.st_size} {int(st.st_mtime)}")
    except OSError:
        pass
    return True


def download(spec: ModelSpec, dest_dir: Path, retries: int = 3, log=print) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / spec.filename
    if verify(dest, spec):
        log(f"  OK (sudah ada, checksum cocok): {spec.filename}")
        return dest
    tmp = dest.with_name(dest.name + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            log(f"Mengunduh {spec.filename} ({spec.size_bytes / 2**20:.0f} MB) ... percobaan {attempt}")
            req = urllib.request.Request(spec.url, headers={"User-Agent": "mosaic-restore-amd/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
                total = int(resp.headers.get("Content-Length") or spec.size_bytes)
                done = 0
                last_print = 0.0
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    now = time.monotonic()
                    if now - last_print > 0.5:
                        last_print = now
                        sys.stderr.write(f"\r  {done / 2**20:7.1f} / {total / 2**20:.1f} MB")
                        sys.stderr.flush()
            sys.stderr.write("\n")
            if sha256_file(tmp) != spec.sha256:
                raise RuntimeError("checksum SHA-256 tidak cocok (file rusak/termodifikasi)")
            shutil.move(str(tmp), str(dest))
            verify(dest, spec)
            log(f"  OK -> {dest}")
            return dest
        except Exception as exc:  # noqa: BLE001 - laporkan & coba lagi
            last_err = exc
            log(f"  Gagal: {exc}")
            time.sleep(2 * attempt)
    if tmp.exists():
        tmp.unlink()
    raise RuntimeError(f"Tidak dapat mengunduh {spec.filename}: {last_err}\n"
                       f"Unduh manual dari {spec.url} lalu taruh di {dest_dir}")


def resolve_model(name_or_path: str, kind: str, model_dir: str | os.PathLike | None = None,
                  auto_download: bool = True, log=print) -> Path:
    """Kembalikan path file model; unduh bila perlu."""
    candidate = Path(name_or_path).expanduser()
    if candidate.suffix in (".pt", ".pth") and candidate.exists():
        return candidate
    spec = MODELS.get(name_or_path)
    if spec is None:
        known = ", ".join(k for k, s in MODELS.items() if s.kind == kind)
        raise ValueError(f"Model '{name_or_path}' tidak dikenal. Pilihan: {known} atau path file.")
    if spec.kind != kind:
        raise ValueError(f"Model '{name_or_path}' adalah model {spec.kind}, bukan {kind}.")
    directory = Path(model_dir) if model_dir else default_model_dir()
    path = directory / spec.filename
    if verify(path, spec):
        return path
    if path.exists():
        log(f"Peringatan: {path.name} tidak lolos verifikasi SHA-256, mengunduh ulang.")
    if not auto_download:
        raise FileNotFoundError(f"Model {spec.filename} belum ada di {directory}. Jalankan dengan --download-models")
    return download(spec, directory, log=log)


def download_all(model_dir: str | os.PathLike | None = None, keys: list[str] | None = None, log=print) -> None:
    directory = Path(model_dir) if model_dir else default_model_dir()
    for key in keys or [DEFAULT_DETECTION, "v4-fast", DEFAULT_RESTORATION]:
        download(MODELS[key], directory, log=log)
