"""Pembuat video uji sintetis (tanpa konten nyata): adegan bertekstur + area mosaic bergerak."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np


def _scene(width: int, height: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    big_h, big_w = height + 200, width + 300
    yy, xx = np.mgrid[0:big_h, 0:big_w].astype(np.float32)
    img = np.zeros((big_h, big_w, 3), np.float32)
    img[..., 0] = 0.5 + 0.3 * np.sin(xx / 23.0) * np.cos(yy / 31.0)
    img[..., 1] = 0.45 + 0.25 * np.sin((xx + yy) / 40.0)
    img[..., 2] = 0.55 + 0.2 * np.cos(xx / 17.0 - yy / 29.0)
    for _ in range(80):
        c = (int(rng.integers(0, big_w)), int(rng.integers(0, big_h)))
        cv2.circle(img, c, int(rng.integers(6, 40)), tuple(float(v) for v in rng.random(3)), -1)
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    return np.clip(cv2.GaussianBlur(img, (0, 0), 1.2), 0, 1)


def mosaic_box(t: int, width: int, height: int, size: int) -> tuple[int, int, int, int]:
    cx = int(width * 0.5 + width * 0.18 * np.sin(t / 15.0))
    cy = int(height * 0.5 + height * 0.12 * np.cos(t / 21.0))
    return cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2


def apply_mosaic(frame: np.ndarray, box, block: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    reg = frame[y0:y1, x0:x1]
    h, w = reg.shape[:2]
    small = cv2.resize(reg, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    frame[y0:y1, x0:x1] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return frame


def make_video(path: str | Path, width: int = 640, height: int = 360, frames: int = 60, fps: int = 30,
               mosaic_size: int = 120, block: int = 12, cut_at: int | None = None, audio: bool = True,
               clean_path: str | Path | None = None):
    """Tulis video uji. Kembalikan daftar kotak mosaic per frame."""
    scenes = [_scene(width, height, 0), _scene(width, height, 7)]
    boxes = []
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={frames / fps}",
                "-c:a", "aac", "-b:a", "128k"]
    cmd += ["-c:v", "libx264", "-crf", "12", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    clean_proc = None
    if clean_path:
        clean_cmd = cmd[:]
        clean_cmd[-1] = str(clean_path)
        clean_proc = subprocess.Popen(clean_cmd, stdin=subprocess.PIPE)
    for t in range(frames):
        scene = scenes[1] if (cut_at is not None and t >= cut_at) else scenes[0]
        ox, oy = 40 + 2 * t, 30 + t
        frame = scene[oy:oy + height, ox:ox + width].copy()
        clean = (frame * 255).astype(np.uint8)
        box = mosaic_box(t, width, height, mosaic_size)
        boxes.append(box)
        frame = apply_mosaic(frame, box, block)
        proc.stdin.write((frame * 255).astype(np.uint8).tobytes())
        if clean_proc:
            clean_proc.stdin.write(clean.tobytes())
    proc.stdin.close()
    proc.wait()
    if clean_proc:
        clean_proc.stdin.close()
        clean_proc.wait()
    return boxes
