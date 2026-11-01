"""Modulated deformable convolution (DCNv2) yang portabel.

- Jalur cepat: ``torchvision.ops.deform_conv2d`` (CPU, ROCm/HIP, CUDA).
- Jalur portabel: implementasi murni PyTorch berbasis ``grid_sample`` + matmul,
  berjalan di backend apa pun (termasuk DirectML) tanpa ekstensi C++/HIP.

Konvensi offset sama dengan mmcv/torchvision:
  offset: [B, 2 * G * kh * kw, Ho, Wo] tersusun (G, K, [dy, dx])
  mask:   [B, G * kh * kw, Ho, Wo]
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

_TORCHVISION_OK: dict[str, bool] = {}


def deform_conv2d_portable(x: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor | None,
                           weight: torch.Tensor, bias: torch.Tensor | None = None,
                           stride: int = 1, padding: int = 1, dilation: int = 1) -> torch.Tensor:
    b, c, h, w = x.shape
    c_out, c_in, kh, kw = weight.shape
    if c_in != c:
        raise ValueError("deform conv portabel hanya mendukung groups=1")
    k = kh * kw
    groups = offset.shape[1] // (2 * k)
    ho, wo = offset.shape[-2:]
    dtype, device = offset.dtype, offset.device

    base_y = torch.arange(ho, device=device, dtype=dtype) * stride - padding
    base_x = torch.arange(wo, device=device, dtype=dtype) * stride - padding
    ky = (torch.arange(kh, device=device, dtype=dtype) * dilation).repeat_interleave(kw)
    kx = (torch.arange(kw, device=device, dtype=dtype) * dilation).repeat(kh)

    off = offset.view(b, groups, k, 2, ho, wo)
    pos_y = base_y.view(1, 1, 1, ho, 1) + ky.view(1, 1, k, 1, 1) + off[:, :, :, 0]
    pos_x = base_x.view(1, 1, 1, 1, wo) + kx.view(1, 1, k, 1, 1) + off[:, :, :, 1]
    # koordinat piksel -> [-1, 1] (align_corners=True: -1 = pusat piksel 0)
    grid_x = pos_x * (2.0 / max(w - 1, 1)) - 1.0
    grid_y = pos_y * (2.0 / max(h - 1, 1)) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).view(b * groups, k * ho, wo, 2)

    xg = x.reshape(b * groups, c // groups, h, w)
    sampled = F.grid_sample(xg, grid.to(xg.dtype), mode="bilinear", padding_mode="zeros",
                            align_corners=True)
    sampled = sampled.view(b, groups, c // groups, k, ho, wo)
    if mask is not None:
        sampled = sampled * mask.view(b, groups, 1, k, ho, wo).to(sampled.dtype)
    cols = sampled.reshape(b, c * k, ho * wo)
    out = torch.matmul(weight.reshape(c_out, c * k).to(cols.dtype), cols)
    out = out.view(b, c_out, ho, wo)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1).to(out.dtype)
    return out


def deform_conv2d(x: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor | None,
                  weight: torch.Tensor, bias: torch.Tensor | None = None,
                  padding: int = 1, impl: str = "auto") -> torch.Tensor:
    """Pilih implementasi terbaik untuk device tensor ``x``."""
    dev_type = x.device.type
    if impl == "portable" or dev_type not in ("cpu", "cuda"):
        return deform_conv2d_portable(x, offset, mask, weight, bias, padding=padding)
    if _TORCHVISION_OK.get(dev_type, True):
        try:
            from torchvision.ops import deform_conv2d as tv_deform_conv2d

            dtype = x.dtype
            out = tv_deform_conv2d(x, offset.to(dtype), weight.to(dtype),
                                   None if bias is None else bias.to(dtype),
                                   stride=(1, 1), padding=(padding, padding), dilation=(1, 1),
                                   mask=None if mask is None else mask.to(dtype))
            _TORCHVISION_OK[dev_type] = True
            return out
        except (RuntimeError, NotImplementedError, ImportError) as exc:
            if impl == "torchvision":
                raise
            _TORCHVISION_OK[dev_type] = False
            warnings.warn(f"torchvision deform_conv2d tidak tersedia di {dev_type} ({exc}); "
                          "memakai implementasi portabel.")
    return deform_conv2d_portable(x, offset, mask, weight, bias, padding=padding)
