"""Konversi warna YUV <-> BGR (torch, berjalan di GPU).

Pipeline bekerja langsung pada plane YUV 4:2:0 hasil decode sehingga piksel di
luar area mosaic tetap identik bit-per-bit (tidak ada round-trip RGB) sebelum
di-encode ulang. Hanya area yang direstorasi yang dikonversi.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_KR_KB = {"bt601": (0.299, 0.114), "bt709": (0.2126, 0.0722), "bt2020": (0.2627, 0.0593)}


def _params(bit_depth: int, full_range: bool):
    maxv = float((1 << bit_depth) - 1)
    c_off = float(1 << (bit_depth - 1))
    if full_range:
        return 0.0, maxv, c_off, maxv, maxv
    scale = float(1 << (bit_depth - 8))
    return 16.0 * scale, 219.0 * scale, c_off, 224.0 * scale, maxv


def yuv_to_bgr(y: torch.Tensor, u: torch.Tensor, v: torch.Tensor, matrix: str = "bt709",
               full_range: bool = False, bit_depth: int = 8, half_res: bool = False) -> torch.Tensor:
    """y: [B,H,W], u/v: [B,ceil(H/2),ceil(W/2)] (nilai kode integer) -> BGR [B,3,H,W] float 0..1.

    ``half_res=True`` menghasilkan gambar pada resolusi chroma (luma di-rata-rata 2x2),
    berguna untuk input deteksi yang memang akan diperkecil.
    """
    kr, kb = _KR_KB[matrix]
    y_off, y_rng, c_off, c_rng, _ = _params(bit_depth, full_range)
    yf = y.float()
    if yf.dim() == 2:
        yf, u, v = yf[None], u[None], v[None]
    h, w = yf.shape[-2:]
    uv = torch.stack((u.float(), v.float()), dim=1)  # [B,2,hc,wc]
    if half_res:
        yf = F.avg_pool2d(yf[:, None], 2, 2, ceil_mode=True)[:, 0]
        h, w = yf.shape[-2:]
        uv = uv[..., :h, :w]
    else:
        hc, wc = uv.shape[-2:]
        uv = F.interpolate(uv, size=(hc * 2, wc * 2), mode="bilinear", align_corners=False)[..., :h, :w]
    yn = (yf - y_off) / y_rng
    cb = (uv[:, 0] - c_off) / c_rng
    cr = (uv[:, 1] - c_off) / c_rng
    r = yn + 2.0 * (1.0 - kr) * cr
    b = yn + 2.0 * (1.0 - kb) * cb
    g = (yn - kr * r - kb * b) / (1.0 - kr - kb)
    return torch.stack((b, g, r), dim=1).clamp_(0.0, 1.0)


def bgr_to_yuv(bgr: torch.Tensor, matrix: str = "bt709", full_range: bool = False,
               bit_depth: int = 8):
    """BGR [B,3,H,W] float 0..1 -> (Y [B,H,W], U [B,hc,wc], V [B,hc,wc]) nilai kode float (belum dibulatkan)."""
    kr, kb = _KR_KB[matrix]
    kg = 1.0 - kr - kb
    y_off, y_rng, c_off, c_rng, _ = _params(bit_depth, full_range)
    b, g, r = bgr[:, 0].float(), bgr[:, 1].float(), bgr[:, 2].float()
    yn = kr * r + kg * g + kb * b
    cb = (b - yn) / (2.0 * (1.0 - kb))
    cr = (r - yn) / (2.0 * (1.0 - kr))
    y = yn * y_rng + y_off
    uv = torch.stack((cb, cr), dim=1) * c_rng + c_off
    uv = F.avg_pool2d(uv, 2, 2, ceil_mode=True)
    return y, uv[:, 0], uv[:, 1]


def max_code(bit_depth: int) -> float:
    return float((1 << bit_depth) - 1)
