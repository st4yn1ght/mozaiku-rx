"""Orkestrasi dua tahap.

Tahap 1 (analisis): decode -> deteksi mosaic per frame (batch, GPU) -> deteksi
pergantian adegan -> pelacakan global (isi celah, filter, penghalusan).

Tahap 2 (restorasi): decode ulang -> crop jendela stabil -> restorasi klip
temporal BasicVSR++ (GPU) -> cross-fade antar klip -> blending YUV -> encode
(AMF/VA-API) + salin audio. Memori dibatasi: frame hanya ditahan sampai semua
klip yang mencakupnya selesai.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..config import Settings
from ..device import DeviceInfo, empty_cache, is_oom_error
from ..models.detector import ColorParams, MosaicDetector
from ..video.encoders import EncoderConfig
from ..video.probe import VideoInfo, probe
from ..video.reader import FrameReader
from ..video.writer import FrameWriter
from .blender import composite, overlay_debug
from .progress import Progress, fmt_time
from .restorer import ClipRestorer
from .tracker import (SceneCutDetector, Track, blend_margins, build_tracks, plan_chunks,
                      plan_windows)


@dataclass
class Analysis:
    frame_count: int
    tracks: list[Track]
    cuts: int
    frames_with_detection: int
    seconds: float


@dataclass
class Result:
    input: str
    output: str | None
    skipped: bool = False
    message: str = ""
    stats: dict = field(default_factory=dict)


class Processor:
    def __init__(self, settings: Settings, dev: DeviceInfo, detector: MosaicDetector,
                 restorer: ClipRestorer, log=print, show_progress: bool = True, debug: bool = False):
        self.s = settings
        self.dev = dev
        self.detector = detector
        self.restorer = restorer
        self.log = log
        self.show_progress = show_progress
        self.debug = debug

    # ================================================================ tahap 1
    def _detect(self, frames, color):
        try:
            return self.detector.detect(frames, color)
        except RuntimeError as exc:
            if not is_oom_error(exc) or len(frames) == 1:
                raise
            empty_cache(self.dev)
            half = len(frames) // 2
            self.s.det_batch = max(1, half)
            return self._detect(frames[:half], color) + self._detect(frames[half:], color)

    def analyze(self, info: VideoInfo) -> Analysis:
        s = self.s
        color = ColorParams(info.matrix, info.full_range, info.work_bit_depth)
        scd = SceneCutDetector() if s.scene_cut else None
        dets_all: list[list] = []
        cuts: list[bool] = []
        batch = []
        prog = Progress("Tahap 1/2 analisis", info.estimated_frames(s.start, s.duration),
                        enabled=self.show_progress)
        with FrameReader(info, s.start, s.duration, s.hwdec) as reader:
            for frame in reader:
                cuts.append(scd.update(frame.y, frame.bit_depth) if scd else False)
                batch.append(frame)
                if len(batch) >= s.det_batch:
                    dets_all.extend(self._detect(batch, color))
                    prog.update(len(batch))
                    batch = []
            if batch:
                dets_all.extend(self._detect(batch, color))
                prog.update(len(batch))
        seconds = prog.close()
        tracks = build_tracks(dets_all, cuts, max_gap=s.max_gap, min_len=s.min_track_len)
        return Analysis(frame_count=len(dets_all), tracks=tracks, cuts=sum(cuts),
                        frames_with_detection=sum(1 for d in dets_all if d), seconds=seconds)

    # ================================================================ tahap 2
    def restore(self, info: VideoInfo, analysis: Analysis, out_path: str | Path,
                encoder: EncoderConfig) -> dict:
        s = self.s
        color = ColorParams(info.matrix, info.full_range, info.work_bit_depth)
        fw, fh = info.width, info.height
        tracks = {t.id: t for t in analysis.tracks}
        windows = {t.id: plan_windows(t, fw, fh, s.context_ratio, s.smooth_sigma, s.dilate_ratio,
                                      s.feather_ratio) for t in analysis.tracks}
        chunks = []
        for t in analysis.tracks:
            chunks += plan_chunks(t, windows[t.id], s.clip_length, s.clip_overlap, s.restore_size, len(chunks))
        chunks_at: dict[int, list] = defaultdict(list)
        cover: dict[tuple[int, int], int] = defaultdict(int)
        for c in chunks:
            for f in range(c.start, c.end + 1):
                chunks_at[f].append(c)
                cover[(c.track_id, f)] += 1
        pending: dict[int, int] = defaultdict(int)
        for (tid, f) in cover:
            pending[f] += 1

        buffer: dict[int, object] = {}
        inputs: dict[int, list] = defaultdict(list)
        metas: dict[int, list] = defaultdict(list)
        acc: dict[tuple[int, int], list] = {}
        next_out = 0
        restored_frames = 0
        writer = FrameWriter(out_path, info, encoder, s.start, s.duration)
        prog = Progress("Tahap 2/2 restorasi", analysis.frame_count, enabled=self.show_progress)
        t_model = 0.0

        def finish_crop(tid: int, f: int, img: torch.Tensor) -> None:
            nonlocal restored_frames
            frame = buffer.get(f)
            if frame is not None:
                tf = tracks[tid].at(f)
                win = windows[tid][f - tracks[tid].start]
                size = max(tf.box[2] - tf.box[0], tf.box[3] - tf.box[1])
                dilate, feather = blend_margins(size, s.dilate_ratio, s.feather_ratio)
                if self.debug:
                    overlay_debug(frame, win, tf.mask, frame.bit_depth)
                else:
                    composite(frame, win, img, tf.mask, dilate, feather, color, fw, fh)
                restored_frames += 1
            pending[f] -= 1

        def run_chunk(c) -> None:
            nonlocal t_model
            crops = inputs.pop(c.id)
            cmeta = metas.pop(c.id)
            if not crops:
                return
            t0 = time.monotonic()
            if self.debug:
                out = torch.stack(crops)
            else:
                out = self.restorer.restore(crops)
            t_model += time.monotonic() - t0
            tid = c.track_id
            track = tracks[tid]
            for j in range(len(crops)):
                f = c.start + j
                win = windows[tid][f - track.start]
                img = ClipRestorer.to_window(out[j], cmeta[j], win) * c.weight(f)
                key = (tid, f)
                entry = acc.get(key)
                if entry is None:
                    entry = acc[key] = [img, cover[key]]
                else:
                    entry[0] = entry[0] + img
                entry[1] -= 1
                if entry[1] == 0:
                    finish_crop(tid, f, entry[0])
                    del acc[key]
            del out

        def flush(force: bool = False) -> None:
            nonlocal next_out
            while next_out in buffer and (force or pending.get(next_out, 0) <= 0):
                writer.write(buffer.pop(next_out))
                prog.update(1)
                next_out += 1

        try:
            with FrameReader(info, s.start, s.duration, s.hwdec) as reader:
                for frame in reader:
                    f = frame.index
                    buffer[f] = frame
                    cache: dict[tuple[int, int], tuple] = {}
                    for c in chunks_at.get(f, ()):
                        key = (c.track_id, c.size)
                        if key not in cache:
                            win = windows[c.track_id][f - tracks[c.track_id].start]
                            cache[key] = self.restorer.prepare_crop(frame, win, c.size, color)
                        crop, meta = cache[key]
                        inputs[c.id].append(crop)
                        metas[c.id].append(meta)
                        if f == c.end:
                            run_chunk(c)
                    flush()
            # video ternyata lebih pendek dari hasil analisis: selesaikan klip yang tersisa
            for c in chunks:
                if c.id in inputs:
                    run_chunk(c)
            for key in list(acc):
                finish_crop(key[0], key[1], acc.pop(key)[0])
            flush(force=True)
            writer.close()
        except BaseException:
            writer.abort()
            raise
        seconds = prog.close()
        return {"frames": writer.frames_written, "restored_crops": restored_frames,
                "chunks": len(chunks), "seconds": seconds, "model_seconds": t_model}

    # ================================================================ utuh
    def process(self, in_path: str | Path, out_path: str | Path, encoder_factory) -> Result:
        info = probe(in_path)
        fps = float(info.fps)
        self.log(f"Video : {Path(in_path).name}")
        self.log(f"        {info.width}x{info.height} {fps:.3f} fps {info.codec} {info.pix_fmt} "
                 f"({info.bit_depth}-bit, matriks {info.matrix}{', full range' if info.full_range else ''})"
                 + (f", durasi {fmt_time(info.duration)}" if info.duration else ""))
        if info.is_vfr:
            self.log("        Peringatan: frame rate variabel -> dikonversi ke CFR.")
        analysis = self.analyze(info)
        total_tf = sum(len(t) for t in analysis.tracks)
        self.log(f"Analisis: {analysis.frame_count} frame, {analysis.frames_with_detection} frame berisi mosaic, "
                 f"{len(analysis.tracks)} jejak, {analysis.cuts} pergantian adegan "
                 f"({fmt_time(analysis.seconds)})")
        if not analysis.tracks:
            return Result(str(in_path), None, skipped=True, message="Tidak ada area mosaic yang terdeteksi.")
        encoder: EncoderConfig = encoder_factory(info)
        self.log(f"Encoder: {encoder.name}{' (hardware)' if encoder.hardware else ''}")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.stem + ".part" + out_path.suffix)
        try:
            stats = self.restore(info, analysis, tmp, encoder)
            os.replace(tmp, out_path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        stats.update(tracks=len(analysis.tracks), track_frames=total_tf,
                     analysis_seconds=analysis.seconds)
        return Result(str(in_path), str(out_path), stats=stats)
