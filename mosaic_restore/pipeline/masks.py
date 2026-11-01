"""Penyimpanan masker biner yang ringkas (bit-packed + zlib) dalam koordinat frame."""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PackedMask:
    x0: int
    y0: int
    w: int
    h: int
    data: bytes

    @classmethod
    def pack(cls, x0: int, y0: int, mask: np.ndarray) -> "PackedMask":
        mask = np.asarray(mask, dtype=bool)
        h, w = mask.shape
        return cls(int(x0), int(y0), int(w), int(h), zlib.compress(np.packbits(mask.reshape(-1)).tobytes(), 1))

    @classmethod
    def from_box(cls, x0: int, y0: int, x1: int, y1: int) -> "PackedMask":
        return cls.pack(x0, y0, np.ones((max(1, y1 - y0), max(1, x1 - x0)), bool))

    def unpack(self) -> np.ndarray:
        bits = np.frombuffer(zlib.decompress(self.data), dtype=np.uint8)
        return np.unpackbits(bits, count=self.w * self.h).reshape(self.h, self.w).astype(bool)

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x0 + self.w, self.y0 + self.h

    def crop(self, wx0: int, wy0: int, ww: int, wh: int) -> np.ndarray:
        """Masker di dalam jendela (wx0, wy0, ww, wh) sebagai array bool [wh, ww]."""
        out = np.zeros((wh, ww), dtype=bool)
        ax0, ay0 = max(self.x0, wx0), max(self.y0, wy0)
        ax1, ay1 = min(self.x0 + self.w, wx0 + ww), min(self.y0 + self.h, wy0 + wh)
        if ax1 <= ax0 or ay1 <= ay0:
            return out
        m = self.unpack()
        out[ay0 - wy0:ay1 - wy0, ax0 - wx0:ax1 - wx0] = m[ay0 - self.y0:ay1 - self.y0, ax0 - self.x0:ax1 - self.x0]
        return out


def union(masks: list[PackedMask]) -> PackedMask:
    masks = [m for m in masks if m is not None]
    if not masks:
        raise ValueError("daftar masker kosong")
    if len(masks) == 1:
        return masks[0]
    x0 = min(m.x0 for m in masks)
    y0 = min(m.y0 for m in masks)
    x1 = max(m.x0 + m.w for m in masks)
    y1 = max(m.y0 + m.h for m in masks)
    canvas = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for m in masks:
        canvas[m.y0 - y0:m.y0 - y0 + m.h, m.x0 - x0:m.x0 - x0 + m.w] |= m.unpack()
    return PackedMask.pack(x0, y0, canvas)


def box_iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def boxes_touch(a, b, margin: float = 0.0) -> bool:
    return not (a[2] + margin <= b[0] or b[2] + margin <= a[0] or
                a[3] + margin <= b[1] or b[3] + margin <= a[1])


def box_union(a, b):
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])
