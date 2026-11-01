"""Menyatukan hasil restorasi ke frame asli dengan tepi bergradasi (feather), di domain YUV.

Piksel dengan alpha 0 tetap identik dengan aslinya (bit-exact).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..models.detector import ColorParams
from ..video.color import bgr_to_yuv, max_code
from .masks import PackedMask
from .restorer import _plane_tensor, window_planes
from .tracker import Window


def make_alpha(mask: torch.Tensor, dilate: int, feather: int,
               taper: tuple[bool, bool, bool, bool] = (False, False, False, False)) -> torch.Tensor:
    """mask [h,w] (0/1) -> alpha [h,w] dalam 0..1.

    Masker diperlebar ``dilate`` piksel (menutup tepi blok mosaic), lalu diberi
    gradasi linear selebar ``feather`` piksel ke arah luar. ``taper`` = (kiri,
    atas, kanan, bawah): paksa alpha menuju 0 di sisi jendela yang tidak
    berbatasan dengan tepi frame.
    """
    m = mask.float()[None, None]
    if dilate > 0:
        m = F.max_pool2d(m, 2 * dilate + 1, stride=1, padding=dilate)
    acc = m.clone()
    cur = m
    for _ in range(max(0, feather)):
        cur = F.max_pool2d(cur, 3, stride=1, padding=1)
        acc = acc + cur
    alpha = acc / float(max(0, feather) + 1)
    alpha = F.avg_pool2d(alpha, 3, stride=1, padding=1, count_include_pad=False)[0, 0]
    if any(taper):
        h, w = alpha.shape
        ramp = float(feather + 2)
        dev, dt = alpha.device, alpha.dtype
        tx = torch.ones(w, device=dev, dtype=dt)
        ty = torch.ones(h, device=dev, dtype=dt)
        xs = torch.arange(w, device=dev, dtype=dt)
        ys = torch.arange(h, device=dev, dtype=dt)
        left, top, right, bottom = taper
        if left:
            tx = torch.minimum(tx, (xs / ramp).clamp(0, 1))
        if right:
            tx = torch.minimum(tx, ((w - 1 - xs) / ramp).clamp(0, 1))
        if top:
            ty = torch.minimum(ty, (ys / ramp).clamp(0, 1))
        if bottom:
            ty = torch.minimum(ty, ((h - 1 - ys) / ramp).clamp(0, 1))
        alpha = alpha * ty[:, None] * tx[None, :]
    return alpha.clamp_(0.0, 1.0)


def composite(frame, win: Window, restored_bgr: torch.Tensor, mask: PackedMask, dilate: int, feather: int,
              color: ColorParams, frame_w: int, frame_h: int) -> None:
    """Campur ``restored_bgr`` [3,h,w] (device) ke plane YUV ``frame`` di dalam ``win`` (in-place)."""
    dev = restored_bgr.device
    region = torch.from_numpy(mask.crop(win.x0, win.y0, win.w, win.h)).to(dev)
    taper = (win.x0 > 0, win.y0 > 0, win.x1 < frame_w, win.y1 < frame_h)
    alpha = make_alpha(region, dilate, feather, taper)

    yp, up, vp = window_planes(frame, win)
    y = _plane_tensor(yp, dev).float()
    u = _plane_tensor(up, dev).float()
    v = _plane_tensor(vp, dev).float()
    yr, ur, vr = bgr_to_yuv(restored_bgr[None], color.matrix, color.full_range, color.bit_depth)
    a_c = F.avg_pool2d(alpha[None, None], 2, 2, ceil_mode=True)[0, 0]
    a_c = a_c[: u.shape[0], : u.shape[1]]
    top = max_code(color.bit_depth)
    y_out = (y + alpha * (yr[0] - y)).round_().clamp_(0, top)
    u_out = (u + a_c * (ur[0][: u.shape[0], : u.shape[1]] - u)).round_().clamp_(0, top)
    v_out = (v + a_c * (vr[0][: v.shape[0], : v.shape[1]] - v)).round_().clamp_(0, top)
    dtype = yp.dtype
    yp[...] = y_out.cpu().numpy().astype(dtype)
    up[...] = u_out.cpu().numpy().astype(dtype)
    vp[...] = v_out.cpu().numpy().astype(dtype)


def overlay_debug(frame, win: Window, mask: PackedMask, bit_depth: int) -> None:
    """Mode debug: tandai area terdeteksi (luma diterangkan) dan bingkai jendela crop."""
    region = mask.crop(win.x0, win.y0, win.w, win.h)
    yp = frame.y[win.y0:win.y1, win.x0:win.x1]
    top = (1 << bit_depth) - 1
    boosted = np.minimum(top, yp.astype(np.int32) + (top // 4))
    yp[region] = boosted[region].astype(yp.dtype)
    yp[0, :] = top
    yp[-1, :] = top
    yp[:, 0] = top
    yp[:, -1] = top
