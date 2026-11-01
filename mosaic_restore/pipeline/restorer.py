"""Menyiapkan crop klip dan menjalankan model restorasi BasicVSR++ (dengan penanganan VRAM habis)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..device import DeviceInfo, autocast, empty_cache, is_oom_error
from ..models.basicvsrpp import load_restoration_model
from ..models.detector import ColorParams
from ..video.color import yuv_to_bgr
from .tracker import Window


@dataclass
class CropMeta:
    sw: int   # lebar konten (sebelum padding) di kanvas model
    sh: int   # tinggi konten


def _plane_tensor(arr: np.ndarray, device) -> torch.Tensor:
    if arr.dtype != np.uint8:
        arr = arr.astype(np.int16)
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device)


def window_planes(frame, win: Window):
    cy0, cy1 = win.y0 // 2, (win.y1 + 1) // 2
    cx0, cx1 = win.x0 // 2, (win.x1 + 1) // 2
    return (frame.y[win.y0:win.y1, win.x0:win.x1],
            frame.u[cy0:cy1, cx0:cx1],
            frame.v[cy0:cy1, cx0:cx1])


def _resize(x: torch.Tensor, size: tuple[int, int], upscale_mode: str) -> torch.Tensor:
    if tuple(x.shape[-2:]) == size:
        return x
    if size[0] < x.shape[-2]:
        return F.interpolate(x, size=size, mode="area")
    try:
        if upscale_mode == "bicubic":
            return F.interpolate(x, size=size, mode="bicubic", align_corners=False)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    except (RuntimeError, NotImplementedError):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ClipRestorer:
    def __init__(self, model_path, dev: DeviceInfo, fp16: bool = False, recon_batch: int = 8):
        self.dev = dev
        self.model = load_restoration_model(model_path, dev.device)
        self.amp = bool(fp16 and dev.supports_amp)
        self.recon_batch = recon_batch

    # ------------------------------------------------------------------ input
    def prepare_crop(self, frame, win: Window, size: int, color: ColorParams):
        yp, up, vp = window_planes(frame, win)
        dev = self.dev.device
        bgr = yuv_to_bgr(_plane_tensor(yp, dev), _plane_tensor(up, dev), _plane_tensor(vp, dev),
                         color.matrix, color.full_range, color.bit_depth)  # [1,3,h,w]
        h, w = bgr.shape[-2:]
        scale = size / float(max(h, w))
        sw = min(size, max(1, int(round(w * scale))))
        sh = min(size, max(1, int(round(h * scale))))
        x = _resize(bgr, (sh, sw), "bilinear")
        pad_w, pad_h = size - sw, size - sh
        if pad_w or pad_h:
            mode = "reflect" if pad_w < sw and pad_h < sh else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        return x[0].contiguous(), CropMeta(sw, sh)

    # ------------------------------------------------------------------ model
    def _run(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(batch, recon_batch=self.recon_batch, amp=autocast(self.dev, self.amp))

    def restore(self, crops: list[torch.Tensor]) -> torch.Tensor:
        """crops: list [3,S,S] -> tensor [T,3,S,S] (BGR 0..1)."""
        try:
            return self._run(torch.stack(crops, 0))
        except RuntimeError as exc:
            if not is_oom_error(exc) or len(crops) < 16:
                raise
        # VRAM tidak cukup: pecah klip menjadi dua bagian yang tumpang-tindih
        empty_cache(self.dev)
        t = len(crops)
        half = t // 2
        ov = min(8, half // 2)
        first = self.restore(crops[: half + ov])
        second = self.restore(crops[half - ov:])
        out = torch.empty((t,) + tuple(first.shape[1:]), device=first.device, dtype=first.dtype)
        out[: half - ov] = first[: half - ov]
        out[half + ov:] = second[2 * ov:]
        n = 2 * ov
        for j in range(n):
            wgt = (j + 1) / (n + 1)
            out[half - ov + j] = first[half - ov + j] * (1 - wgt) + second[j] * wgt
        return out

    @staticmethod
    def to_window(restored: torch.Tensor, meta: CropMeta, win: Window) -> torch.Tensor:
        """Potong padding & kembalikan crop hasil restorasi [3,S,S] ke ukuran jendela [3,h,w]."""
        x = restored[:, : meta.sh, : meta.sw][None]
        x = _resize(x, (win.h, win.w), "bicubic")
        return x[0].clamp_(0.0, 1.0)
