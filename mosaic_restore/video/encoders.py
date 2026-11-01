"""Pemilihan encoder video, mengutamakan encoder hardware AMD.

- Windows : AMD AMF  (hevc_amf / h264_amf / av1_amf [RDNA3+])
- Linux   : VA-API   (hevc_vaapi / h264_vaapi / av1_vaapi [RDNA3+]) via Mesa
- Cadangan: libx265 / libx264 / libsvtav1 (CPU)

Setiap kandidat diuji dengan encode singkat sebelum dipakai.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field

from .probe import ffmpeg_bin

HW_AMF = ("hevc_amf", "h264_amf", "av1_amf")
HW_VAAPI = ("hevc_vaapi", "h264_vaapi", "av1_vaapi")
SOFTWARE = ("libx265", "libx264", "libsvtav1")
ALL_ENCODERS = HW_AMF + HW_VAAPI + SOFTWARE


@dataclass
class EncoderConfig:
    name: str
    pre_input: list[str] = field(default_factory=list)   # opsi global sebelum -i
    video_args: list[str] = field(default_factory=list)
    hardware: bool = False
    keeps_full_range: bool = False   # output 8-bit tetap full range (yuvj420p)

    @property
    def codec_family(self) -> str:
        if self.name.startswith(("hevc", "libx265")):
            return "hevc"
        if self.name.startswith(("av1", "libsvtav1")):
            return "av1"
        return "h264"


def build_encoder(name: str, cq: int, bit_depth: int, preset: str | None = None,
                  vaapi_device: str = "/dev/dri/renderD128", full_range: bool = False) -> EncoderConfig:
    ten_bit = bit_depth > 8
    full8 = full_range and not ten_bit  # sumber yuvj420p
    av1_q = max(1, min(255, round(cq * 4.4)))  # skala QP H.264/HEVC -> qindex AV1
    if name in HW_AMF:
        q = av1_q if name == "av1_amf" else cq
        args = ["-c:v", name, "-quality", preset or "quality", "-rc", "cqp",
                "-qp_i", str(q), "-qp_p", str(min(q + 2, 255 if name == "av1_amf" else 51))]
        if name == "h264_amf":
            args += ["-qp_b", str(min(q + 4, 51))]
        if ten_bit and name != "h264_amf":
            args += ["-pix_fmt", "p010le"]
            if name == "hevc_amf":
                args += ["-profile:v", "main10"]
        else:
            args += ["-pix_fmt", "nv12"]
        return EncoderConfig(name, [], args, hardware=True)
    if name in HW_VAAPI:
        fmt = "p010" if ten_bit and name != "h264_vaapi" else "nv12"
        q = av1_q if name == "av1_vaapi" else cq
        args = ["-vf", f"format={fmt},hwupload", "-c:v", name, "-rc_mode", "CQP", "-qp", str(q)]
        if ten_bit and name == "hevc_vaapi":
            args += ["-profile:v", "main10"]
        return EncoderConfig(name, ["-vaapi_device", vaapi_device], args, hardware=True)
    if name == "libx265":
        pix = "yuv420p10le" if ten_bit else ("yuvj420p" if full8 else "yuv420p")
        args = ["-c:v", "libx265", "-preset", preset or "medium", "-crf", str(cq),
                "-pix_fmt", pix, "-x265-params", "log-level=error"]
        return EncoderConfig(name, [], args, keeps_full_range=full8)
    if name == "libx264":
        args = ["-c:v", "libx264", "-preset", preset or "slow", "-crf", str(max(0, cq - 1)),
                "-pix_fmt", "yuvj420p" if full8 else "yuv420p"]
        return EncoderConfig(name, [], args, keeps_full_range=full8)
    if name == "libsvtav1":
        pix = "yuv420p10le" if ten_bit else "yuv420p"
        args = ["-c:v", "libsvtav1", "-preset", preset or "6", "-crf", str(min(63, round(cq * 1.4))),
                "-pix_fmt", pix]
        return EncoderConfig(name, [], args)
    raise ValueError(f"Encoder tidak dikenal: {name}. Pilihan: {', '.join(ALL_ENCODERS)}")


_TEST_CACHE: dict[tuple, bool] = {}


def test_encoder(cfg: EncoderConfig, bit_depth: int, full_range: bool = False) -> bool:
    key = (cfg.name, tuple(cfg.pre_input), tuple(cfg.video_args), bit_depth, full_range)
    if key in _TEST_CACHE:
        return _TEST_CACHE[key]
    src_fmt = "yuv420p10le" if bit_depth > 8 else ("yuvj420p" if full_range else "yuv420p")
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin", *cfg.pre_input,
           "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=25:duration=0.4,format={src_fmt}",
           *cfg.video_args, "-f", "null", "-"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        ok = res.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    _TEST_CACHE[key] = ok
    return ok


def candidate_order(prefer_amd_hw: bool = True) -> list[str]:
    hw: list[str] = []
    if prefer_amd_hw:
        if sys.platform == "win32":
            hw = ["hevc_amf", "h264_amf"]
        elif sys.platform.startswith("linux"):
            hw = ["hevc_vaapi", "h264_vaapi"]
    return hw + ["libx265", "libx264"]


def select_encoder(requested: str, cq: int, bit_depth: int, preset: str | None = None,
                   vaapi_device: str = "/dev/dri/renderD128", full_range: bool = False,
                   log=print) -> EncoderConfig:
    names = candidate_order() if requested == "auto" else [requested]
    for name in names:
        cfg = build_encoder(name, cq, bit_depth, preset, vaapi_device, full_range)
        if test_encoder(cfg, bit_depth, full_range):
            return cfg
        if requested != "auto":
            raise RuntimeError(f"Encoder '{name}' tidak bisa dipakai di sistem ini (uji encode gagal). "
                               "Pastikan driver GPU & build FFmpeg mendukungnya, atau pakai --encoder auto.")
        log(f"  encoder {name} tidak tersedia, mencoba berikutnya...")
    raise RuntimeError("Tidak ada encoder video yang berfungsi (libx264/libx265 juga gagal). Periksa instalasi FFmpeg.")
