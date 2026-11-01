"""Pelacakan area mosaic antar-frame, pengisian celah, dan perencanaan klip.

Seluruh hasil deteksi tersedia sebelum restorasi (analisis 2-tahap), sehingga
pelacakan bisa memakai informasi masa depan:
  * celah deteksi pendek (frame terlewat) diisi -> tidak ada "kedipan" mosaic,
  * jejak diputus di pergantian adegan -> propagasi temporal tidak bocor antar adegan,
  * jendela crop dihaluskan secara temporal -> input model stabil & konsisten,
  * jejak panjang dipecah menjadi klip yang saling tumpang-tindih + cross-fade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .masks import PackedMask, box_iou, box_union, boxes_touch, union


# ------------------------------------------------------------------ scene cuts
class SceneCutDetector:
    """Deteksi potongan adegan dari selisih thumbnail luma."""

    def __init__(self, threshold: float = 12.0, ratio: float = 4.0):
        self.threshold = threshold
        self.ratio = ratio
        self.prev: np.ndarray | None = None
        self.history: list[float] = []

    def update(self, y_plane: np.ndarray, bit_depth: int = 8) -> bool:
        y8 = y_plane if bit_depth == 8 else (y_plane >> (bit_depth - 8)).astype(np.uint8)
        thumb = cv2.resize(y8, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.prev is None:
            self.prev = thumb
            return False
        diff = float(np.mean(np.abs(thumb - self.prev)))
        self.prev = thumb
        recent = self.history[-15:]
        baseline = float(np.median(recent)) if recent else 0.0
        self.history.append(diff)
        if len(self.history) > 64:
            del self.history[:-32]
        return diff > self.threshold and diff > self.ratio * max(baseline, 2.0)


# ------------------------------------------------------------------ tracks
@dataclass
class TrackFrame:
    box: tuple[int, int, int, int]
    mask: PackedMask
    score: float
    filled: bool = False


@dataclass
class Track:
    id: int
    start: int
    frames: list[TrackFrame] = field(default_factory=list)

    @property
    def end(self) -> int:
        return self.start + len(self.frames) - 1

    def __len__(self) -> int:
        return len(self.frames)

    def at(self, frame_index: int) -> TrackFrame:
        return self.frames[frame_index - self.start]


@dataclass
class _Builder:
    id: int
    entries: dict = field(default_factory=dict)   # frame -> (box, [masks], score)
    last_frame: int = -1
    last_box: tuple = (0, 0, 0, 0)


def _group_detections(dets) -> list[tuple]:
    """Gabungkan deteksi yang saling bersinggungan dalam satu frame."""
    groups: list[list] = []
    for d in sorted(dets, key=lambda d: -d.score):
        merged = None
        for g in groups:
            if boxes_touch(g[0], d.box):
                g[0] = box_union(g[0], d.box)
                g[1].append(d.mask)
                g[2] = max(g[2], d.score)
                merged = g
                break
        if merged is None:
            groups.append([d.box, [d.mask], d.score])
    # gabung ulang bila penggabungan menghasilkan kotak yang bersinggungan
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if boxes_touch(groups[i][0], groups[j][0]):
                    groups[i][0] = box_union(groups[i][0], groups[j][0])
                    groups[i][1] += groups[j][1]
                    groups[i][2] = max(groups[i][2], groups[j][2])
                    del groups[j]
                    changed = True
                    break
            if changed:
                break
    return [tuple(g) for g in groups]


def build_tracks(dets_per_frame: list[list], cuts: list[bool], max_gap: int = 4,
                 min_len: int = 2, strong_score: float = 0.5, weak_score: float = 0.35,
                 weak_max_len: int = 8) -> list[Track]:
    builders: list[_Builder] = []
    finished: list[_Builder] = []
    next_id = 0
    for f, dets in enumerate(dets_per_frame):
        if f < len(cuts) and cuts[f]:
            finished.extend(builders)
            builders = []
        alive = []
        for b in builders:
            (alive if f - b.last_frame <= max_gap + 1 else finished).append(b)
        builders = alive
        if not dets:
            continue
        groups = _group_detections(dets)
        pairs = []
        for gi, (gbox, _, _) in enumerate(groups):
            for bi, b in enumerate(builders):
                pad = 0.1 * max(b.last_box[2] - b.last_box[0], b.last_box[3] - b.last_box[1]) + 8
                if boxes_touch(b.last_box, gbox, margin=pad):
                    pairs.append((box_iou(b.last_box, gbox), gi, bi))
        pairs.sort(reverse=True)
        used_g, used_b = set(), set()
        for _, gi, bi in pairs:
            if gi in used_g or bi in used_b:
                continue
            used_g.add(gi)
            used_b.add(bi)
            gbox, masks, score = groups[gi]
            b = builders[bi]
            b.entries[f] = (gbox, masks, score)
            b.last_frame, b.last_box = f, gbox
        for gi, (gbox, masks, score) in enumerate(groups):
            if gi not in used_g:
                b = _Builder(next_id, {f: (gbox, masks, score)}, f, gbox)
                next_id += 1
                builders.append(b)
    finished.extend(builders)

    tracks: list[Track] = []
    for b in sorted(finished, key=lambda b: min(b.entries)):
        frames_idx = sorted(b.entries)
        best = max(e[2] for e in b.entries.values())
        if len(frames_idx) < min_len and best < strong_score:
            continue  # deteksi tunggal yang lemah
        if len(frames_idx) < weak_max_len and best < weak_score:
            continue  # kedipan singkat yang tidak pernah yakin -> kemungkinan salah deteksi
        track = Track(id=len(tracks), start=frames_idx[0])
        for a, nxt in zip(frames_idx, frames_idx[1:] + [None]):
            box, masks, score = b.entries[a]
            track.frames.append(TrackFrame(box, union(masks), score))
            if nxt is not None and nxt - a > 1:
                # isi celah: gabungan masker frame sebelum & sesudah (konservatif)
                nbox, nmasks, nscore = b.entries[nxt]
                gap_mask = union([union(masks), union(nmasks)])
                gap_box = box_union(box, nbox)
                for _ in range(a + 1, nxt):
                    track.frames.append(TrackFrame(gap_box, gap_mask, min(score, nscore), filled=True))
        tracks.append(track)
    return tracks


# ------------------------------------------------------------------ planning
@dataclass
class Window:
    x0: int
    y0: int
    w: int
    h: int

    @property
    def x1(self) -> int:
        return self.x0 + self.w

    @property
    def y1(self) -> int:
        return self.y0 + self.h


@dataclass
class Chunk:
    id: int
    track_id: int
    start: int
    end: int                       # inklusif
    size: int                      # ukuran input model (persegi)
    weights: dict = field(default_factory=dict)   # frame -> bobot cross-fade (default 1)

    def weight(self, f: int) -> float:
        return self.weights.get(f, 1.0)

    def __len__(self) -> int:
        return self.end - self.start + 1


def _gauss_kernel(sigma: float, radius: int) -> np.ndarray:
    if sigma <= 0 or radius <= 0:
        return np.ones(1)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _smooth(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    r = len(kernel) // 2
    if r == 0 or len(values) == 1:
        return values.astype(np.float64)
    padded = np.pad(values.astype(np.float64), r, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _running_max(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values.astype(np.float64)
    padded = np.pad(values.astype(np.float64), radius, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1)
    return windows.max(axis=1)


def blend_margins(size: float, dilate_ratio: float, feather_ratio: float) -> tuple[int, int]:
    dilate = int(min(24, max(2, round(size * dilate_ratio))))
    feather = int(min(40, max(3, round(size * feather_ratio))))
    return dilate, feather


def plan_windows(track: Track, frame_w: int, frame_h: int, context_ratio: float = 0.15,
                 smooth_sigma: float = 3.0, dilate_ratio: float = 0.03,
                 feather_ratio: float = 0.045) -> list[Window]:
    boxes = np.array([tf.box for tf in track.frames], dtype=np.float64)
    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    size = np.maximum(bw, bh)
    margins = np.array([sum(blend_margins(s, dilate_ratio, feather_ratio)) + 4 for s in size])
    side_req = size * (1.0 + 2.0 * context_ratio) + 2.0 * margins
    radius = max(2, int(math.ceil(2.5 * smooth_sigma)))
    kernel = _gauss_kernel(smooth_sigma, radius)
    side = np.maximum(_smooth(_running_max(side_req, radius), kernel), side_req)
    cx = _smooth((boxes[:, 0] + boxes[:, 2]) / 2.0, kernel)
    cy = _smooth((boxes[:, 1] + boxes[:, 3]) / 2.0, kernel)

    windows = []
    for i in range(len(track.frames)):
        bx0, by0, bx1, by1 = boxes[i]
        s = side[i]
        m = margins[i]
        wx0 = cx[i] - s / 2.0
        wy0 = cy[i] - s / 2.0
        # pastikan area (beserta margin blending) masuk ke jendela
        wx0 = min(wx0, bx0 - m)
        wx0 = max(wx0, bx1 + m - s)
        wy0 = min(wy0, by0 - m)
        wy0 = max(wy0, by1 + m - s)
        ww = min(int(math.ceil(s)), frame_w)
        wh = min(int(math.ceil(s)), frame_h)
        x0 = int(round(min(max(wx0, 0.0), frame_w - ww)))
        y0 = int(round(min(max(wy0, 0.0), frame_h - wh)))
        # sejajarkan ke koordinat genap (subsampling chroma 4:2:0)
        x1, y1 = x0 + ww, y0 + wh
        x0 -= x0 % 2
        y0 -= y0 % 2
        x1 = min(frame_w, x1 + (x1 % 2))
        y1 = min(frame_h, y1 + (y1 % 2))
        windows.append(Window(x0, y0, x1 - x0, y1 - y0))
    return windows


def choose_restore_size(windows: list[Window], base: int = 256, requested: int = 256) -> int:
    if requested and requested > 0:
        return max(256, int(math.ceil(requested / 64.0)) * 64)
    largest = max(max(w.w, w.h) for w in windows)
    size = int(math.ceil(min(largest, 512) / 64.0)) * 64
    return max(base, size)


def plan_chunks(track: Track, windows: list[Window], max_len: int, overlap: int,
                restore_size: int, next_id: int) -> list[Chunk]:
    n = len(track)
    max_len = max(2, max_len)
    overlap = max(0, min(overlap, max_len // 2))
    if n <= max_len:
        spans = [(0, n - 1)]
    else:
        m = math.ceil((n - overlap) / (max_len - overlap))
        length = math.ceil((n + (m - 1) * overlap) / m)
        spans = []
        a = 0
        for k in range(m):
            b = min(n - 1, a + length - 1)
            if k == m - 1:
                b = n - 1
                a = max(0, min(a, b - max_len + 1))
            spans.append((a, b))
            a = b - overlap + 1
    chunks: list[Chunk] = []
    for k, (a, b) in enumerate(spans):
        size = choose_restore_size(windows[a:b + 1], requested=restore_size)
        chunks.append(Chunk(next_id + k, track.id, track.start + a, track.start + b, size))
    for k in range(len(chunks) - 1):
        cur, nxt = chunks[k], chunks[k + 1]
        ov_start, ov_end = nxt.start, cur.end
        count = ov_end - ov_start + 1
        for j in range(count):
            f = ov_start + j
            w_next = (j + 1) / (count + 1)
            nxt.weights[f] = w_next
            cur.weights[f] = 1.0 - w_next
    return chunks
