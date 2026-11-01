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


class GpuNotUsableError(RuntimeError):
    """GPU terlihat, tetapi build PyTorch yang terpasang tidak bisa menjalankan kernel di sana."""


_KERNEL_ERROR_MARKERS = ("hiperrorinvalidimage", "kernel image is invalid", "no kernel image",
                         "invalid device function", "hiperrornobinaryforgpu", "no binary for gpu")


def gpu_smoke_test(device) -> str | None:
    """Jalankan beban kecil yang mewakili operasi model (conv, pooling, bicubic,
    pixel-shuffle, grid-sample, matmul). Kembalikan None bila OK, atau pesan error.

    ``torch.cuda.is_available()`` saja tidak cukup: GPU bisa "terlihat" walau build
    PyTorch tidak berisi kernel untuk arsitekturnya (hipErrorInvalidImage).
    """
    import torch
    import torch.nn.functional as F

    try:
        x = torch.rand(1, 32, 64, 64, device=device)
        w = torch.rand(32, 32, 3, 3, device=device) * 0.05
        y = F.leaky_relu(F.conv2d(x, w, padding=1), 0.1)
        y = F.silu(F.max_pool2d(y, 3, 1, 1)) + F.avg_pool2d(y, 3, 1, 1)
        y = F.interpolate(y, scale_factor=0.5, mode="bicubic", align_corners=False)
        y = F.pixel_shuffle(y, 2)
        grid = torch.rand(1, 32, 32, 2, device=device) * 2 - 1
        y = F.grid_sample(y, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        z = torch.sigmoid(y).flatten(1) @ torch.tanh(y).flatten(1).t()
        value = float(z.sum().cpu())
        if value != value:
            return "hasil perhitungan NaN"
        return None
    except Exception as exc:  # noqa: BLE001 - laporkan alasan apa adanya
        text = str(exc).strip().splitlines()
        return text[0] if text else type(exc).__name__


def _arch_of(props) -> str | None:
    arch = getattr(props, "gcnArchName", None)
    return arch.split(":")[0] if arch else None


def gpu_fix_hint(archs=()) -> str:
    lines = ["Artinya build PyTorch ROCm yang terpasang tidak berisi kernel untuk arsitektur GPU tersebut."]
    if sys.platform == "win32":
        lines += [
            "Cara memperbaiki:",
            "  1) Jalankan ulang install_windows.bat. Installer kini menguji GPU dengan perhitungan",
            "     sungguhan dan otomatis memilih build yang cocok, atau beralih ke DirectML.",
            "  2) Atau langsung pakai DirectML (mendukung semua GPU AMD, lebih lambat):",
            "     install_windows.bat -DirectML",
        ]
    else:
        lines.append("Cara memperbaiki:")
        if any(a and a.startswith("gfx103") and a != "gfx1030" for a in archs):
            lines.append("  - RX 6000 non-resmi: jalankan dengan --hsa-override-gfx 10.3.0")
        if any(a in ("gfx1101", "gfx1102", "gfx1103") for a in archs):
            lines.append("  - RX 7600/7700/780M: coba --hsa-override-gfx 11.0.0")
        lines.append("  - Atau pasang ulang PyTorch yang sesuai GPU Anda: TORCH_INDEX_URL=<url> ./install_linux.sh")
    lines.append("Sementara waktu (sangat lambat): --device cpu")
    return "\n".join(lines)


def explain_gpu_error(exc: BaseException) -> str | None:
    """Petunjuk perbaikan untuk error kernel HIP/CUDA yang dikenal (None bila bukan)."""
    msg = str(exc).lower()
    return gpu_fix_hint() if any(m in msg for m in _KERNEL_ERROR_MARKERS) else None


def _select_hip_or_cuda(index: int | None, want: str) -> DeviceInfo | None:
    import torch

    if not torch.cuda.is_available():
        return None
    is_hip = getattr(torch.version, "hip", None) is not None
    if (want == "rocm" and not is_hip) or (want == "cuda" and is_hip):
        return None
    count = torch.cuda.device_count()
    devices = [(i, torch.cuda.get_device_properties(i)) for i in range(count)]
    if index is not None:
        if index >= count:
            raise RuntimeError(f"GPU index {index} tidak ada (hanya {count} GPU terdeteksi).")
        order = [devices[index]]
    else:
        # VRAM terbesar dulu: kartu grafis diutamakan daripada iGPU Ryzen ("AMD Radeon(TM) Graphics")
        order = sorted(devices, key=lambda d: int(d[1].total_memory), reverse=True)
    failures = []
    for i, props in order:
        dev = torch.device(f"cuda:{i}")
        err = gpu_smoke_test(dev)
        if err is not None:
            failures.append((i, props.name, _arch_of(props), err))
            continue
        torch.cuda.set_device(dev)
        # MIOpen/cuDNN memilih kernel konvolusi tercepat untuk ukuran input tetap
        # (crop 256x256 & deteksi 640) -> sangat menguntungkan di sini.
        torch.backends.cudnn.benchmark = True
        info = DeviceInfo(backend="rocm" if is_hip else "cuda", device=dev, name=props.name,
                          vram_bytes=int(props.total_memory), arch=_arch_of(props))
        if is_hip:
            info.notes.append(f"HIP {torch.version.hip}")
            if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
                info.notes.append(f"HSA_OVERRIDE_GFX_VERSION={os.environ['HSA_OVERRIDE_GFX_VERSION']}")
        for fi, fname, farch, _ in failures:
            info.notes.append(f"GPU {fi} ({fname}{' ' + farch if farch else ''}) dilewati: tidak didukung build ini")
        return info
    lines = ["GPU terdeteksi, tetapi PyTorch yang terpasang tidak bisa menjalankan perhitungan di GPU ini:"]
    lines += [f"  - GPU {i}: {name}{' (' + arch + ')' if arch else ''}: {err}" for i, name, arch, err in failures]
    raise GpuNotUsableError("\n".join(lines) + "\n" + gpu_fix_hint([a for _, _, a, _ in failures]))


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


def select_device(preference: str = "auto", index: int | None = None) -> DeviceInfo:
    import torch

    pref = (preference or "auto").lower()
    if pref in ("hip", "amd"):
        pref = "rocm"
    if pref == "dml":
        pref = "directml"

    rocm_problem: GpuNotUsableError | None = None
    if pref in ("auto", "rocm"):
        try:
            info = _select_hip_or_cuda(index, "rocm")
        except GpuNotUsableError as exc:
            if pref == "rocm":
                raise
            info, rocm_problem = None, exc
        if info:
            return info
        if pref == "rocm":
            raise RuntimeError("Backend ROCm tidak tersedia.\n" + gpu_setup_hint())

    if pref in ("auto", "directml"):
        info = _select_directml(index or 0)
        if info:
            if rocm_problem is not None:
                info.notes.append("ROCm tidak bisa dipakai di GPU ini -> memakai DirectML")
            return info
        if pref == "directml":
            raise RuntimeError("torch-directml tidak terpasang / tidak menemukan GPU.")

    if rocm_problem is not None:
        # Jangan diam-diam pindah ke CPU (puluhan kali lebih lambat): jelaskan cara memperbaiki.
        raise rocm_problem

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


def list_devices(test: bool = True) -> list[str]:
    """Daftar perangkat; dengan ``test=True`` setiap GPU diuji dengan perhitungan sungguhan."""
    import torch

    lines = [f"PyTorch {torch.__version__}"]
    hip = getattr(torch.version, "hip", None)
    if torch.cuda.is_available():
        kind = f"ROCm/HIP {hip}" if hip else f"CUDA {torch.version.cuda}"
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            status = ""
            if test:
                err = gpu_smoke_test(torch.device(f"cuda:{i}"))
                status = " -> OK" if err is None else f" -> TIDAK BISA DIPAKAI ({err})"
            lines.append(f"  [{kind}] index {i}: {p.name} {_arch_of(p) or ''} "
                         f"{p.total_memory / 2**30:.1f} GB{status}")
    else:
        lines.append("  ROCm/CUDA: tidak tersedia" + (f" (build HIP {hip})" if hip else ""))
    try:
        import torch_directml  # type: ignore

        for i in range(torch_directml.device_count()):
            status = ""
            if test:
                err = gpu_smoke_test(torch_directml.device(i))
                status = " -> OK" if err is None else f" -> GAGAL ({err})"
            lines.append(f"  [DirectML] index {i}: {torch_directml.device_name(i)}{status}")
    except ImportError:
        lines.append("  DirectML: tidak terpasang")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"  DirectML: error ({exc})")
    lines.append("  [CPU] selalu tersedia (lambat)")
    return lines


def hip_selftest(allow_no_gpu: bool = False) -> int:
    """Untuk installer: 0 bila build PyTorch HIP bisa menghitung di minimal satu GPU."""
    import torch

    hip = getattr(torch.version, "hip", None)
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"  torch {torch.__version__} | HIP {hip} | GPU terlihat: {visible}")
    if not hip:
        return 1
    if visible == 0:
        if allow_no_gpu:
            print("  (GPU belum terlihat: cek driver / izin grup render,video; build HIP tetap dipakai)")
            return 0
        return 1
    ok = False
    for i in range(visible):
        p = torch.cuda.get_device_properties(i)
        err = gpu_smoke_test(torch.device(f"cuda:{i}"))
        print(f"  GPU {i}: {p.name} {_arch_of(p) or ''} -> {'OK' if err is None else 'GAGAL: ' + err}")
        ok = ok or err is None
    return 0 if ok else 1


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
