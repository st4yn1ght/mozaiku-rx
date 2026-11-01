"""Decode video ke plane YUV 4:2:0 mentah lewat pipa ffmpeg (thread terpisah)."""

from __future__ import annotations

import collections
import queue
import subprocess
import threading

import numpy as np

from .probe import VideoInfo, ffmpeg_bin, ffmpeg_version


class Frame:
    """Satu frame YUV 4:2:0 planar. Array y/u/v adalah view (bisa ditulis) atas ``buf``."""

    __slots__ = ("index", "buf", "y", "u", "v", "bit_depth")

    def __init__(self, index: int, buf: bytearray, width: int, height: int, bit_depth: int):
        self.index = index
        self.buf = buf
        self.bit_depth = bit_depth
        dtype = np.uint8 if bit_depth == 8 else np.dtype("<u2")
        cw, ch = (width + 1) // 2, (height + 1) // 2
        arr = np.frombuffer(buf, dtype=dtype)
        ys, cs = width * height, cw * ch
        self.y = arr[:ys].reshape(height, width)
        self.u = arr[ys:ys + cs].reshape(ch, cw)
        self.v = arr[ys + cs:ys + 2 * cs].reshape(ch, cw)


def frame_nbytes(width: int, height: int, bit_depth: int) -> int:
    bps = 1 if bit_depth == 8 else 2
    cw, ch = (width + 1) // 2, (height + 1) // 2
    return (width * height + 2 * cw * ch) * bps


def fps_mode_args(info: VideoInfo) -> list[str]:
    mode = "cfr" if info.is_vfr else "passthrough"
    major, minor = ffmpeg_version()
    flag = "-fps_mode" if (major, minor) >= (5, 1) else "-vsync"
    args = [flag, mode]
    if mode == "cfr":
        args += ["-r", f"{info.fps.numerator}/{info.fps.denominator}"]
    return args


class FrameReader:
    def __init__(self, info: VideoInfo, start: float | None = None, duration: float | None = None,
                 hwdec: bool = False, queue_size: int = 12):
        self.info = info
        self.bit_depth = info.work_bit_depth
        self.width, self.height = info.width, info.height
        self.frame_size = frame_nbytes(self.width, self.height, self.bit_depth)
        cmd = [ffmpeg_bin(), "-nostdin", "-hide_banner", "-loglevel", "error"]
        if hwdec:
            cmd += ["-hwaccel", "auto"]
        if start:
            cmd += ["-ss", f"{start:.3f}"]
        cmd += ["-i", info.path]
        if duration:
            cmd += ["-t", f"{duration:.3f}"]
        cmd += ["-map", "0:v:0", "-an", "-sn", "-dn"]
        cmd += fps_mode_args(info)
        cmd += ["-pix_fmt", info.raw_pix_fmt, "-f", "rawvideo", "pipe:1"]
        self.cmd = cmd
        self._stderr: collections.deque[str] = collections.deque(maxlen=40)
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, bufsize=0)
        self._t_err = threading.Thread(target=self._read_stderr, daemon=True)
        self._t_out = threading.Thread(target=self._read_frames, daemon=True)
        self._t_err.start()
        self._t_out.start()
        self.frames_read = 0

    def _read_stderr(self):
        assert self.proc.stderr is not None
        for line in iter(self.proc.stderr.readline, b""):
            self._stderr.append(line.decode("utf-8", "replace").rstrip())

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _read_frames(self):
        stream = self.proc.stdout
        assert stream is not None
        index = 0
        try:
            while not self._stop.is_set():
                buf = bytearray(self.frame_size)
                view = memoryview(buf)
                filled = 0
                while filled < self.frame_size:
                    n = stream.readinto(view[filled:])
                    if not n:
                        break
                    filled += n
                if filled < self.frame_size:
                    break  # EOF (frame parsial dibuang)
                if not self._put(Frame(index, buf, self.width, self.height, self.bit_depth)):
                    return
                index += 1
        except Exception as exc:  # noqa: BLE001 - diteruskan ke konsumen
            self._put(exc)
            return
        self._put(None)

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            self.frames_read += 1
            yield item
        self.proc.wait()
        self._t_err.join(timeout=2)
        if self.proc.returncode not in (0, None) and self.frames_read == 0:
            raise RuntimeError("ffmpeg gagal decode:\n" + "\n".join(self._stderr))

    def close(self):
        self._stop.set()
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
