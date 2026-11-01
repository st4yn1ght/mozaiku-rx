"""Uji integrasi end-to-end (butuh model; jalan di CPU, ~beberapa menit).

Jalankan dengan:  MOSAIC_RESTORE_MODELS=/path/ke/models pytest -q tests/test_pipeline.py
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from synthetic import make_video, mosaic_box  # noqa: E402

MODELS = os.environ.get("MOSAIC_RESTORE_MODELS")
pytestmark = pytest.mark.skipif(
    not MODELS or not (Path(MODELS) / "lada_mosaic_detection_model_v4_accurate.pt").exists(),
    reason="set MOSAIC_RESTORE_MODELS ke folder berisi model untuk uji integrasi")


def run_cli(*args):
    from mosaic_restore.cli import main

    return main(["--device", "cpu", "--model-dir", MODELS, "--overwrite", "-q", *map(str, args)])


def ffprobe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def decode(path, w, h, pix="bgr24"):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", pix, "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float32)


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def test_end_to_end_restores_and_preserves(tmp_path):
    src, clean = tmp_path / "in.mp4", tmp_path / "clean.mp4"
    make_video(src, frames=36, cut_at=24, clean_path=clean)
    out = tmp_path / "out.mp4"
    assert run_cli(src, "-o", out, "--clip-length", "16", "--clip-overlap", "4", "--encoder", "libx264") == 0
    meta = ffprobe(out)
    v = [s for s in meta["streams"] if s["codec_type"] == "video"][0]
    assert int(v["nb_read_frames"]) == 36
    assert any(s["codec_type"] == "audio" for s in meta["streams"])
    a, m, o = decode(clean, 640, 360), decode(src, 640, 360), decode(out, 640, 360)
    inside_before, inside_after, outside = [], [], []
    for t in range(36):
        x0, y0, x1, y1 = mosaic_box(t, 640, 360, 120)
        reg = (slice(y0, y1), slice(x0, x1))
        far = np.ones((360, 640), bool)
        far[max(0, y0 - 40):y1 + 40, max(0, x0 - 40):x1 + 40] = False
        inside_before.append(psnr(m[t][reg], a[t][reg]))
        inside_after.append(psnr(o[t][reg], a[t][reg]))
        outside.append(psnr(o[t][far], m[t][far]))
    assert np.mean(inside_after) > np.mean(inside_before) + 4.0
    assert np.mean(outside) > 38.0


def test_ten_bit_hevc_input(tmp_path):
    src8 = tmp_path / "in8.mp4"
    make_video(src8, frames=16, audio=False)
    src10 = tmp_path / "in10.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src8), "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
                    "-x265-params", "log-level=error:lossless=1", str(src10)], check=True)
    out = tmp_path / "out10.mkv"
    assert run_cli(src10, "-o", out, "--encoder", "libx265") == 0
    v = [s for s in ffprobe(out)["streams"] if s["codec_type"] == "video"][0]
    assert v["pix_fmt"] == "yuv420p10le" and int(v["nb_read_frames"]) == 16


def test_full_range_source_keeps_levels(tmp_path):
    src8 = tmp_path / "in8.mp4"
    make_video(src8, frames=12, audio=False)
    srcj = tmp_path / "inj.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src8), "-vf", "scale=out_range=full",
                    "-c:v", "libx264", "-crf", "6", "-pix_fmt", "yuvj420p", "-color_range", "pc", str(srcj)], check=True)
    for enc in ("libx264", "libsvtav1"):  # yang mempertahankan full range & yang mengonversi
        out = tmp_path / f"out_{enc}.mkv"
        assert run_cli(srcj, "-o", out, "--encoder", enc) == 0
        a, o = decode(srcj, 640, 360), decode(out, 640, 360)
        corner = (slice(None), slice(0, 60), slice(0, 100))
        assert psnr(o[corner], a[corner]) > 36.0, enc


def test_segment_and_mkv(tmp_path):
    src = tmp_path / "in.mp4"
    make_video(src, frames=45)
    assert run_cli(src, "--container", "mkv", "--start", "0.5", "--duration", "0.6", "--encoder", "libx264") == 0
    out = tmp_path / "in.restored.mkv"
    meta = ffprobe(out)
    v = [s for s in meta["streams"] if s["codec_type"] == "video"][0]
    assert abs(int(v["nb_read_frames"]) - 18) <= 1
    assert any(s["codec_type"] == "audio" for s in meta["streams"])


def test_video_without_mosaic_is_skipped(tmp_path):
    # Konten alami tanpa mosaic. (Catatan: pola uji 'testsrc2' tersusun dari blok warna datar
    # sehingga memang terlihat seperti mosaic bagi detektor.)
    src = tmp_path / "plain.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "mandelbrot=size=640x360:rate=25,trim=duration=0.6",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    assert run_cli(src, "--encoder", "libx264") == 0
    assert not (tmp_path / "plain.restored.mp4").exists()


def test_debug_overlay(tmp_path):
    src = tmp_path / "in.mp4"
    make_video(src, frames=12, audio=False)
    out = tmp_path / "dbg.mp4"
    assert run_cli(src, "-o", out, "--debug-overlay", "--encoder", "libx264") == 0
    assert out.exists()
