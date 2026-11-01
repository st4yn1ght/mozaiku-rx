"""Membaca metadata video dengan ffprobe."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

_FFMPEG = {"ffmpeg": "ffmpeg", "ffprobe": "ffprobe"}


def set_ffmpeg_paths(ffmpeg: str | None = None, ffprobe: str | None = None) -> None:
    if ffmpeg:
        _FFMPEG["ffmpeg"] = ffmpeg
    if ffprobe:
        _FFMPEG["ffprobe"] = ffprobe
    elif ffmpeg:
        p = Path(ffmpeg)
        cand = p.with_name(p.name.replace("ffmpeg", "ffprobe"))
        if cand.exists():
            _FFMPEG["ffprobe"] = str(cand)


def ffmpeg_bin() -> str:
    return _FFMPEG["ffmpeg"]


def ffprobe_bin() -> str:
    return _FFMPEG["ffprobe"]


def check_ffmpeg() -> str:
    """Pastikan ffmpeg & ffprobe tersedia; kembalikan string versi."""
    for tool in (ffmpeg_bin(), ffprobe_bin()):
        if shutil.which(tool) is None and not Path(tool).exists():
            raise RuntimeError(
                f"'{tool}' tidak ditemukan. Pasang FFmpeg:\n"
                "  Windows: winget install Gyan.FFmpeg   (build ini sudah mendukung AMD AMF)\n"
                "  Linux  : sudo apt install ffmpeg      (atau paket distro Anda)")
    out = subprocess.run([ffmpeg_bin(), "-hide_banner", "-version"], capture_output=True, text=True)
    return out.stdout.splitlines()[0] if out.stdout else "ffmpeg"


def ffmpeg_version() -> tuple[int, int]:
    try:
        out = subprocess.run([ffmpeg_bin(), "-hide_banner", "-version"], capture_output=True, text=True).stdout
        m = re.search(r"version\s+n?(\d+)\.(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    except OSError:
        pass
    return (7, 0)  # build git/nightly: anggap modern


def _fraction(text: str | None) -> Fraction | None:
    if not text or text in ("0/0", "N/A"):
        return None
    try:
        f = Fraction(text)
        return f if f > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def _float(text) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: Fraction
    is_vfr: bool
    nb_frames: int | None
    duration: float | None
    pix_fmt: str
    bit_depth: int
    codec: str
    color_space: str | None
    color_primaries: str | None
    color_transfer: str | None
    color_range: str | None
    start_time: float
    format_start_time: float
    rotation: int
    format_name: str
    audio_codecs: list[str] = field(default_factory=list)
    subtitle_codecs: list[str] = field(default_factory=list)

    @property
    def matrix(self) -> str:
        cs = (self.color_space or "").lower()
        if cs in ("bt709",):
            return "bt709"
        if cs in ("bt2020nc", "bt2020c", "bt2020"):
            return "bt2020"
        if cs in ("smpte170m", "bt470bg", "bt601", "fcc", "smpte240m"):
            return "bt601"
        return "bt709" if self.height >= 720 or self.width >= 1280 else "bt601"

    @property
    def full_range(self) -> bool:
        return (self.color_range or "").lower() in ("pc", "jpeg", "full") or self.pix_fmt.startswith("yuvj")

    @property
    def raw_pix_fmt(self) -> str:
        """Format mentah yang dipakai di pipa ffmpeg (selalu planar 4:2:0)."""
        if self.bit_depth > 8:
            return "yuv420p10le"
        return "yuvj420p" if self.pix_fmt == "yuvj420p" else "yuv420p"

    @property
    def work_bit_depth(self) -> int:
        return 10 if self.bit_depth > 8 else 8

    def estimated_frames(self, start: float | None = None, duration: float | None = None) -> int | None:
        total = self.nb_frames
        if total is None and self.duration:
            total = int(round(self.duration * float(self.fps)))
        if total is None:
            return None
        begin = int(round((start or 0.0) * float(self.fps)))
        remain = max(0, total - begin)
        if duration is not None:
            remain = min(remain, int(round(duration * float(self.fps))))
        return remain


def probe(path: str | Path) -> VideoInfo:
    cmd = [ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe gagal membaca '{path}': {res.stderr.strip()}")
    data = json.loads(res.stdout or "{}")
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    video = None
    for s in streams:
        if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic"):
            video = s
            break
    if video is None:
        raise RuntimeError(f"Tidak ada stream video di '{path}'")

    width, height = int(video["width"]), int(video["height"])
    rotation = 0
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(float(sd["rotation"]))
    if not rotation and "rotate" in video.get("tags", {}):
        rotation = int(float(video["tags"]["rotate"]))
    if abs(rotation) % 180 == 90:  # ffmpeg memutar otomatis saat decode
        width, height = height, width

    avg = _fraction(video.get("avg_frame_rate"))
    real = _fraction(video.get("r_frame_rate"))
    fps = avg or real or Fraction(30000, 1001)
    is_vfr = bool(avg and real and abs(float(avg) - float(real)) > 0.01 * float(real))
    if fps > 240:  # metadata aneh (mis. timebase); jatuh ke r_frame_rate
        fps = real if real and real <= 240 else Fraction(30000, 1001)

    nb = video.get("nb_frames")
    nb_frames = int(nb) if nb and str(nb).isdigit() and int(nb) > 0 else None
    duration = _float(video.get("duration")) or _float(fmt.get("duration"))

    pix_fmt = video.get("pix_fmt") or "yuv420p"
    bits = video.get("bits_per_raw_sample")
    if bits and str(bits).isdigit():
        bit_depth = int(bits)
    else:
        m = re.search(r"(9|10|12|14|16)(le|be)", pix_fmt)
        bit_depth = int(m.group(1)) if m else (10 if "p010" in pix_fmt else 8)

    return VideoInfo(
        path=str(path), width=width, height=height, fps=fps, is_vfr=is_vfr,
        nb_frames=nb_frames, duration=duration, pix_fmt=pix_fmt, bit_depth=bit_depth,
        codec=video.get("codec_name", "?"),
        color_space=video.get("color_space"), color_primaries=video.get("color_primaries"),
        color_transfer=video.get("color_transfer"), color_range=video.get("color_range"),
        start_time=_float(video.get("start_time")) or 0.0,
        format_start_time=_float(fmt.get("start_time")) or 0.0,
        rotation=rotation, format_name=fmt.get("format_name", ""),
        audio_codecs=[s.get("codec_name", "?") for s in streams if s.get("codec_type") == "audio"],
        subtitle_codecs=[s.get("codec_name", "?") for s in streams if s.get("codec_type") == "subtitle"],
    )
