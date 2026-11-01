"""Deteksi & pemilihan backend GPU, dengan fokus pada GPU AMD.

Urutan prioritas pada mode ``auto``:
  1. ROCm/HIP   (PyTorch build ROCm; Linux & Windows, RDNA2/RDNA3/RDNA4, Instinct)
  2. DirectML   (torch-directml; Windows, semua GPU DX12 termasuk RX 5000/6000 lama)
  3. CUDA       (NVIDIA, tetap didukung bila kebetulan tersedia)
  4. CPU        (sangat lambat)

Catatan: PyTorch build ROCm memakai API ``torch.cuda`` (HIP dipetakan ke "cuda").
"""

from __future__ import annotations

import glob
import os
import platform
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field


@dataclass
class DeviceInfo:
    backend: str                 # "rocm" | "directml" | "cuda" | "cpu"
    device: object               # torch.device atau objek device DirectML
    name: str
    vram_bytes: int | None = None
    arch: str | None = None      # mis. "gfx1100" (RX 7900), "gfx1201" (RX 9070)
    notes: list[str] = field(default_factory=list)

    @property
    def is_gpu(self) -> bool:
        return self.backend != "cpu"

    @property
    def supports_amp(self) -> bool:
        """Autocast FP16 hanya andal pada backend ROCm/CUDA."""
        return self.backend in ("rocm", "cuda")

    @property
    def torch_type(self) -> str:
        if self.backend in ("rocm", "cuda"):
            return "cuda"
        if self.backend == "directml":
            return "privateuseone"
        return "cpu"

    def describe(self) -> str:
        parts = [f"{self.backend.upper()}: {self.name}"]
        if self.arch:
            parts.append(self.arch)
        if self.vram_bytes:
            parts.append(f"{self.vram_bytes / 2**30:.1f} GB VRAM")
        return " | ".join(parts)


def prepare_environment(hsa_override_gfx: str | None = None) -> None:
    """Set variabel lingkungan SEBELUM torch di-import."""
    if hsa_override_gfx:
        # Untuk GPU AMD konsumer yang tidak terdaftar resmi di ROCm Linux,
        # mis. RX 6700 XT (gfx1031) -> "10.3.0", RX 7600 (gfx1102) -> "11.0.0".
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = hsa_override_gfx
    # Kurangi spam log MIOpen (peringatan database kernel yang tidak berbahaya).
    os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
    # Ultralytics: jangan cek internet / jangan cetak banner.
    os.environ.setdefault("YOLO_OFFLINE", "1")
    os.environ.setdefault("YOLO_VERBOSE", "False")


def _amd_gpu_present_linux() -> bool:
    for vendor_file in glob.glob("/sys/class/drm/card*/device/vendor"):
        try:
            with open(vendor_file) as fh:
                if fh.read().strip().lower() == "0x1002":
                    return True
        except OSError:
            pass
    return False


def gpu_setup_hint() -> str:
    if sys.platform.startswith("linux"):
        if _amd_gpu_present_linux():
            return ("GPU AMD terdeteksi tetapi PyTorch tidak bisa memakainya.\n"
                    "  - Pastikan PyTorch versi ROCm yang terpasang (jalankan ulang install_linux.sh).\n"
                    "  - Pastikan user ada di grup 'render' dan 'video': sudo usermod -aG render,video $USER\n"
                    "  - Untuk RX 6000 non-resmi (gfx1031/1032/1034/1035/1036) coba: --hsa-override-gfx 10.3.0\n"
                    "  - Untuk RX 7600/780M (gfx1102/gfx1103) bila perlu coba: --hsa-override-gfx 11.0.0")
        return "Tidak ada GPU AMD terdeteksi di /sys/class/drm."
    if sys.platform == "win32":
        return ("PyTorch tidak menemukan GPU. Jalankan ulang install_windows.bat.\n"
                "  - RX 7000/9000 & Ryzen AI: memakai PyTorch ROCm untuk Windows (driver Adrenalin terbaru).\n"
                "  - RX 5000/6000 atau lebih lama: memakai DirectML (opsi cadangan di installer).")
    return "Backend GPU tidak tersedia pada sistem ini."


def _select_hip_or_cuda(index: int, want: str) -> DeviceInfo | None:
    import torch

    if not torch.cuda.is_available():
        return None
    is_hip = getattr(torch.version, "hip", None) is not None
    if want == "rocm" and not is_hip:
        return None
    if want == "cuda" and is_hip:
        return None
    count = torch.cuda.device_count()
    if index >= count:
        raise RuntimeError(f"GPU index {index} tidak ada (hanya {count} GPU terdeteksi).")
    dev = torch.device(f"cuda:{index}")
    torch.cuda.set_device(dev)
    props = torch.cuda.get_device_properties(index)
    arch = getattr(props, "gcnArchName", None)
    if arch:
        arch = arch.split(":")[0]
    # MIOpen/cuDNN akan memilih kernel konvolusi tercepat untuk ukuran input tetap
    # (crop 256x256 & deteksi 640) -> sangat menguntungkan di sini.
    torch.backends.cudnn.benchmark = True
    info = DeviceInfo(
        backend="rocm" if is_hip else "cuda",
        device=dev,
        name=props.name,
        vram_bytes=int(props.total_memory),
        arch=arch,
    )
    if is_hip:
        info.notes.append(f"HIP {torch.version.hip}")
        if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
            info.notes.append(f"HSA_OVERRIDE_GFX_VERSION={os.environ['HSA_OVERRIDE_GFX_VERSION']}")
    return info


def _select_directml(index: int) -> DeviceInfo | None:
    try:
        import torch_directml  # type: ignore
    except Exception:
        return None
    try:
        if not torch_directml.is_available() or torch_directml.device_count() <= index:
            return None
        dev = torch_directml.device(index)
        name = torch_directml.device_name(index)
    except Exception:
        return None
    return DeviceInfo(backend="directml", device=dev, name=str(name).strip(),
                      notes=["DirectML (lebih lambat dari ROCm, FP32)"])


def select_device(preference: str = "auto", index: int = 0) -> DeviceInfo:
    import torch

    pref = (preference or "auto").lower()
    if pref in ("hip", "amd"):
        pref = "rocm"
    if pref == "dml":
        pref = "directml"

    if pref in ("auto", "rocm"):
        info = _select_hip_or_cuda(index, "rocm")
        if info:
            return info
        if pref == "rocm":
            raise RuntimeError("Backend ROCm tidak tersedia.\n" + gpu_setup_hint())

    if pref in ("auto", "directml"):
        info = _select_directml(index)
        if info:
            return info
        if pref == "directml":
            raise RuntimeError("torch-directml tidak terpasang / tidak menemukan GPU.")

    if pref in ("auto", "cuda"):
        info = _select_hip_or_cuda(index, "cuda")
        if info:
            return info
        if pref == "cuda":
            raise RuntimeError("Backend CUDA (NVIDIA) tidak tersedia.")

    if pref in ("auto", "cpu"):
        threads = os.cpu_count() or 4
        torch.set_num_threads(threads)
        info = DeviceInfo(backend="cpu", device=torch.device("cpu"),
                          name=platform.processor() or platform.machine() or "CPU")
        info.notes.append(f"{threads} thread")
        return info

    raise ValueError(f"Pilihan device tidak dikenal: {preference}")


def list_devices() -> list[str]:
    import torch

    lines = [f"PyTorch {torch.__version__}"]
    hip = getattr(torch.version, "hip", None)
    if torch.cuda.is_available():
        kind = f"ROCm/HIP {hip}" if hip else f"CUDA {torch.version.cuda}"
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            arch = getattr(p, "gcnArchName", "") or ""
            lines.append(f"  [{kind}] index {i}: {p.name} {arch} {p.total_memory / 2**30:.1f} GB")
    else:
        lines.append("  ROCm/CUDA: tidak tersedia" + (f" (build HIP {hip})" if hip else ""))
    try:
        import torch_directml  # type: ignore

        for i in range(torch_directml.device_count()):
            lines.append(f"  [DirectML] index {i}: {torch_directml.device_name(i)}")
    except Exception:
        lines.append("  DirectML: tidak terpasang")
    lines.append("  [CPU] selalu tersedia (lambat)")
    return lines


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if any(s in msg for s in ("out of memory", "hiperroroutofmemory", "e_outofmemory", "887a0005")):
        return True
    # DirectML: "Could not allocate tensor ... not enough GPU video memory available"
    return "memory" in msg and any(s in msg for s in ("not enough", "allocate", "insufficient"))


def empty_cache(info: DeviceInfo) -> None:
    if info.backend in ("rocm", "cuda"):
        import torch

        torch.cuda.empty_cache()


def autocast(info: DeviceInfo, enabled: bool):
    """Konteks autocast FP16 (hanya ROCm/CUDA)."""
    if enabled and info.supports_amp:
        import torch

        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()
