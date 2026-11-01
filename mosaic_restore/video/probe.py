"""Menemukan FFmpeg dan membaca metadata video dengan ffprobe."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

# ---------------------------------------------------------------- lokasi FFmpeg
# Lokasi dicari lalu DIVERIFIKASI dengan menjalankan "-version". Hanya path
# absolut ke executable yang benar-benar bisa dijalankan yang dipakai, sehingga
# entri PATH yang rusak, symlink winget, file .bat/.cmd bernama "ffmpeg", atau
# folder bernama "ffmpeg" tidak lagi menyebabkan crash.
_OVERRIDE: dict[str, str | None] = {"ffmpeg": None, "ffprobe": None}
_RESOLVED: dict[str, str] = {}


def _exe(tool: str) -> str:
    return tool + ".exe" if sys.platform == "win32" else tool


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def set_ffmpeg_paths(ffmpeg: str | None = None, ffprobe: str | None = None) -> None:
    """Paksa lokasi FFmpeg: path ke ffmpeg(.exe) ATAU folder yang berisinya."""
    _OVERRIDE["ffmpeg"] = ffmpeg
    _OVERRIDE["ffprobe"] = ffprobe
    _RESOLVED.clear()


def _expand(path_like: str, tool: str) -> list[Path]:
    p = Path(str(path_like).strip().strip('"')).expanduser()
    if p.is_dir():
        return [p / _exe(tool), p / "bin" / _exe(tool)]
    if tool != "ffmpeg":  # --ffmpeg C:\x\ffmpeg.exe -> ffprobe di folder yang sama
        return [p.with_name(_exe(tool))]
    return [p]


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _candidates(tool: str) -> list[Path]:
    exe = _exe(tool)
    cands: list[Path] = []
    env = os.environ.get("MOSAIC_RESTORE_FFMPEG")
    if env:
        cands += _expand(env, tool)
    root = _project_root()  # FFmpeg portabel di dalam folder program
    cands += [root / "ffmpeg" / "bin" / exe, root / "ffmpeg" / exe, root / "bin" / exe, root / exe]
    cands += sorted(root.glob(f"ffmpeg*/bin/{exe}"), key=_mtime, reverse=True)
    found = shutil.which(tool)
    if found:
        cands.append(Path(found))
    if sys.platform == "win32":
        local = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        winget = local / "Microsoft" / "WinGet"
        cands.append(winget / "Links" / exe)
        packages = winget / "Packages"
        if packages.is_dir():  # langsung ditemukan walau PATH terminal belum diperbarui
            cands += sorted(packages.glob(f"*FFmpeg*/*/bin/{exe}"), key=_mtime, reverse=True)
        for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"), "C:\\"):
            if base:
                cands.append(Path(base) / "ffmpeg" / "bin" / exe)
        cands.append(Path(os.environ.get("ProgramData") or "C:\\ProgramData") / "chocolatey" / "bin" / exe)
        cands.append(Path.home() / "scoop" / "apps" / "ffmpeg" / "current" / "bin" / exe)
        cands.append(Path.home() / "scoop" / "shims" / exe)
    else:
        cands += [Path(d) / tool for d in ("/usr/bin", "/usr/local/bin", "/opt/homebrew/bin", "/snap/bin")]
    return cands


def _try_run(path: Path) -> tuple[str | None, str]:
    """(path_asli, baris_versi) bila bisa dijalankan; (None, alasan) bila tidak."""
    try:
        real = Path(os.path.realpath(path))
    except (OSError, ValueError):
        real = Path(path)
    if not real.exists():
        return None, "tidak ada"
    if not real.is_file():
        return None, "bukan file (folder?)"
    if sys.platform == "win32" and real.suffix.lower() != ".exe":
        return None, "bukan file .exe"
    try:
        res = subprocess.run([str(real), "-hide_banner", "-version"], stdin=subprocess.DEVNULL,
                             capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"gagal dijalankan ({exc})"
    out = res.stdout.decode("utf-8", "replace")
    if res.returncode != 0 or "version" not in out:
        return None, f"tidak merespons dengan benar (kode keluar {res.returncode})"
    return str(real), out.splitlines()[0].strip()


def _not_found_message(tool: str, problems: list[str], override: str | None) -> str:
    lines = [f"{tool} tidak ditemukan atau tidak bisa dijalankan."]
    if override:
        lines.append(f"  Lokasi dari --ffmpeg: {override}")
    if problems:
        lines.append("  Ditemukan tetapi bermasalah:")
        lines += [f"    - {p}" for p in problems[:8]]
    lines += [
        "  Cara memperbaiki (pilih salah satu):",
        "    Windows: winget install Gyan.FFmpeg   (lalu jalankan ulang program ini)",
        "             ATAU ekstrak FFmpeg 'full build' dari gyan.dev ke folder program ini dan beri",
        "             nama folder 'ffmpeg' (sehingga ada ffmpeg\\bin\\ffmpeg.exe)",
        "             ATAU tunjukkan lokasinya: --ffmpeg \"C:\\lokasi\\ffmpeg\\bin\"",
        "    Linux  : sudo apt install ffmpeg",
    ]
    return "\n".join(lines)


def _locate(tool: str, first: list[Path]) -> tuple[str, str]:
    override = _OVERRIDE.get(tool) or (_OVERRIDE.get("ffmpeg") if tool == "ffprobe" else None)
    candidates = _expand(override, tool) if override else first + _candidates(tool)
    problems: list[str] = []
    seen: set[str] = set()
    for cand in candidates:
        key = os.path.normcase(str(cand))
        if key in seen:
            continue
        seen.add(key)
        path, info = _try_run(cand)
        if path:
            return path, info
        if info != "tidak ada":
            problems.append(f"{cand}: {info}")
    raise RuntimeError(_not_found_message(tool, problems, override))


def resolve_ffmpeg() -> tuple[str, str, str]:
    """Kembalikan (path ffmpeg, path ffprobe, baris versi). RuntimeError bila tidak ada."""
    if "ffprobe" not in _RESOLVED:
        ffmpeg, version = _locate("ffmpeg", [])
        ffprobe, _ = _locate("ffprobe", [Path(ffmpeg).with_name(_exe("ffprobe"))])
        _RESOLVED.update(ffmpeg=ffmpeg, ffprobe=ffprobe, version=version)
    return _RESOLVED["ffmpeg"], _RESOLVED["ffprobe"], _RESOLVED["version"]


def ffmpeg_bin() -> str:
    return resolve_ffmpeg()[0]


def ffprobe_bin() -> str:
    return resolve_ffmpeg()[1]


def check_ffmpeg() -> str:
    """Pastikan ffmpeg & ffprobe bisa dijalankan; kembalikan ringkasan untuk log."""
    ffmpeg, _, version = resolve_ffmpeg()
    return f"FFmpeg: {ffmpeg}\n        {version}"


def ffmpeg_version() -> tuple[int, int]:
    m = re.search(r"version\s+n?(\d+)\.(\d+)", resolve_ffmpeg()[2])
    if m:
        return int(m.group(1)), int(m.group(2))
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
    try:
        res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True)
    except OSError as exc:
        raise RuntimeError(f"ffprobe tidak bisa dijalankan ({cmd[0]}): {exc}") from exc
    # ffprobe menulis JSON dalam UTF-8; jangan pakai code page Windows (nama file Jepang!)
    stdout = res.stdout.decode("utf-8", "replace")
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe gagal membaca '{path}': {res.stderr.decode('utf-8', 'replace').strip()}")
    data = json.loads(stdout or "{}")
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
