"""Encode frame YUV mentah ke file video lewat pipa ffmpeg (thread terpisah).

Audio (dan subtitle untuk MKV) disalin langsung dari file asli tanpa re-encode
bila kontainer mendukung codec-nya.
"""

from __future__ import annotations

import collections
import queue
import subprocess
import threading
from pathlib import Path

from .encoders import EncoderConfig
from .probe import VideoInfo, ffmpeg_bin

MP4_AUDIO_OK = {"aac", "mp3", "ac3", "eac3", "alac", "opus", "flac", "mp2"}
_DEFAULT_TAGS = {
    "bt709": ("bt709", "bt709", "bt709"),
    "bt601": ("smpte170m", "smpte170m", "smpte170m"),
    "bt2020": ("bt2020nc", "bt2020", "bt709"),
}


def color_args(info: VideoInfo, enc: EncoderConfig) -> list[str]:
    space, prim, trc = _DEFAULT_TAGS[info.matrix]
    # yuvj420p -> nv12/yuv420p dikonversi ffmpeg ke limited range; 10-bit tidak dikonversi.
    keeps_full = info.full_range and (info.work_bit_depth > 8 or enc.keeps_full_range)
    return ["-colorspace", info.color_space or space,
            "-color_primaries", info.color_primaries or prim,
            "-color_trc", info.color_transfer or trc,
            "-color_range", "pc" if keeps_full else "tv"]


class FrameWriter:
    def __init__(self, out_path: str | Path, info: VideoInfo, enc: EncoderConfig,
                 start: float | None = None, duration: float | None = None, queue_size: int = 12):
        self.out_path = str(out_path)
        ext = Path(out_path).suffix.lower()
        is_mkv = ext in (".mkv", ".webm")
        cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", *enc.pre_input]
        offset = 0.0 if start else max(0.0, info.start_time - info.format_start_time)
        if offset > 0.001:
            cmd += ["-itsoffset", f"{offset:.6f}"]
        cmd += ["-f", "rawvideo", "-pix_fmt", info.raw_pix_fmt, "-s", f"{info.width}x{info.height}",
                "-framerate", f"{info.fps.numerator}/{info.fps.denominator}", "-i", "pipe:0"]
        has_audio = bool(info.audio_codecs)
        has_subs = bool(info.subtitle_codecs) and is_mkv
        if has_audio or has_subs:
            if start:
                cmd += ["-ss", f"{start:.3f}"]
            if duration:
                cmd += ["-t", f"{duration:.3f}"]
            cmd += ["-i", info.path]
        cmd += ["-map", "0:v:0"]
        if has_audio:
            cmd += ["-map", "1:a?"]
            for i, codec in enumerate(info.audio_codecs):
                if is_mkv or codec in MP4_AUDIO_OK:
                    cmd += [f"-c:a:{i}", "copy"]
                else:
                    cmd += [f"-c:a:{i}", "aac", f"-b:a:{i}", "256k"]
        if has_subs:
            cmd += ["-map", "1:s?", "-c:s", "copy"]
        if has_audio or has_subs:
            cmd += ["-map_metadata", "1"]
            if not start and not duration:
                cmd += ["-map_chapters", "1"]
        cmd += enc.video_args
        cmd += color_args(info, enc)
        if ext in (".mp4", ".mov", ".m4v"):
            cmd += ["-movflags", "+faststart"]
            if enc.codec_family == "hevc":
                cmd += ["-tag:v", "hvc1"]
        cmd += [self.out_path]
        self.cmd = cmd
        self._stderr: collections.deque[str] = collections.deque(maxlen=60)
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._error: BaseException | None = None
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, bufsize=0)
        self._t_err = threading.Thread(target=self._read_stderr, daemon=True)
        self._t_in = threading.Thread(target=self._write_frames, daemon=True)
        self._t_err.start()
        self._t_in.start()
        self.frames_written = 0

    def _read_stderr(self):
        assert self.proc.stderr is not None
        for line in iter(self.proc.stderr.readline, b""):
            self._stderr.append(line.decode("utf-8", "replace").rstrip())

    def _write_frames(self):
        stdin = self.proc.stdin
        assert stdin is not None
        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    break
                # pipa tanpa buffer: write() boleh menulis sebagian -> ulangi sampai habis
                view = memoryview(frame.buf)
                while view:
                    written = stdin.write(view)
                    if not written:
                        raise OSError("penulisan ke pipa ffmpeg gagal")
                    view = view[written:]
        except BaseException as exc:  # noqa: BLE001 - dilaporkan saat close()
            self._error = exc
            while True:  # kosongkan antrean agar produsen tidak macet
                try:
                    if self._queue.get_nowait() is None:
                        break
                except queue.Empty:
                    break
        finally:
            try:
                stdin.close()
            except OSError:
                pass

    def write(self, frame) -> None:
        if self._error is not None:
            raise RuntimeError("ffmpeg encoder berhenti:\n" + "\n".join(self._stderr)) from self._error
        self._queue.put(frame)
        self.frames_written += 1

    def close(self) -> None:
        self._queue.put(None)
        self._t_in.join()
        self.proc.wait()
        self._t_err.join(timeout=2)
        if self.proc.returncode != 0 or self._error is not None:
            raise RuntimeError("ffmpeg encode gagal (kode %s):\n%s" % (self.proc.returncode, "\n".join(self._stderr)))

    def abort(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
