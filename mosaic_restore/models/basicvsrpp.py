"""Implementasi BasicVSR++ untuk inferensi.

Arsitektur mengikuti makalah "BasicVSR++: Improving Video Super-Resolution with
Enhanced Propagation and Alignment" (Chan et al., CVPR 2022): propagasi
dua arah orde-dua dengan alignment deformable berpandu optical flow (SPyNet).
Nama parameter dibuat kompatibel dengan checkpoint restorasi mosaic Lada
(prefix ``generator_ema.``), varian input resolusi penuh (output = input + residu).

Input/output: tensor [T, 3, H, W] urutan kanal BGR, rentang [0, 1].
H dan W harus kelipatan 4 dan >= 256 (flow dihitung di 1/4 resolusi, min 64).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deform_conv import deform_conv2d

BRANCHES = ("backward_1", "forward_1", "backward_2", "forward_2")

_GRID_CACHE: dict[tuple, torch.Tensor] = {}


def _base_grid(h: int, w: int, device, dtype) -> torch.Tensor:
    key = (h, w, str(device), dtype)
    grid = _GRID_CACHE.get(key)
    if grid is None:
        ys = torch.arange(h, device=device, dtype=dtype)
        xs = torch.arange(w, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((gx, gy), dim=-1)  # [H, W, 2] (x, y)
        if len(_GRID_CACHE) > 32:
            _GRID_CACHE.clear()
        _GRID_CACHE[key] = grid
    return grid


def flow_warp(x: torch.Tensor, flow: torch.Tensor, padding_mode: str = "zeros") -> torch.Tensor:
    """Warp ``x`` [B,C,H,W] dengan flow [B,H,W,2] (dx, dy) dalam satuan piksel."""
    _, _, h, w = x.shape
    pos = _base_grid(h, w, flow.device, flow.dtype).unsqueeze(0) + flow
    gx = pos[..., 0] * (2.0 / max(w - 1, 1)) - 1.0
    gy = pos[..., 1] * (2.0 / max(h - 1, 1)) - 1.0
    grid = torch.stack((gx, gy), dim=-1).to(x.dtype)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)


class ResidualBlockNoBN(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64, num_blocks: int = 15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Sequential(*[ResidualBlockNoBN(out_channels) for _ in range(num_blocks)]),
        )

    def forward(self, x):
        return self.main(x)


class _ConvAct(nn.Module):
    def __init__(self, cin: int, cout: int, act: bool):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 7, 1, 3)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x) if self.act else x


class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            _ConvAct(8, 32, True), _ConvAct(32, 64, True), _ConvAct(64, 32, True),
            _ConvAct(32, 16, True), _ConvAct(16, 2, False))

    def forward(self, x):
        return self.basic_module(x)


class SPyNet(nn.Module):
    """Optical flow piramida (Ranjan & Black, 2017)."""

    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _compute(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        n, _, h, w = ref.shape
        refs = [(ref - self.mean) / self.std]
        supps = [(supp - self.mean) / self.std]
        for _ in range(5):
            refs.append(F.avg_pool2d(refs[-1], kernel_size=2, stride=2, count_include_pad=False))
            supps.append(F.avg_pool2d(supps[-1], kernel_size=2, stride=2, count_include_pad=False))
        refs.reverse()
        supps.reverse()
        flow = ref.new_zeros(n, 2, h // 32, w // 32)
        for level, module in enumerate(self.basic_module):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
            warped = flow_warp(supps[level], flow_up.permute(0, 2, 3, 1), padding_mode="border")
            flow = flow_up + module(torch.cat([refs[level], warped, flow_up], dim=1))
        return flow

    def forward(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        """Flow sehingga ``flow_warp(supp, flow)`` sejajar dengan ``ref``. Output [N,2,H,W]."""
        h, w = ref.shape[-2:]
        h_up = h if h % 32 == 0 else 32 * (h // 32 + 1)
        w_up = w if w % 32 == 0 else 32 * (w // 32 + 1)
        ref_r = F.interpolate(ref, size=(h_up, w_up), mode="bilinear", align_corners=False)
        supp_r = F.interpolate(supp, size=(h_up, w_up), mode="bilinear", align_corners=False)
        flow = F.interpolate(self._compute(ref_r, supp_r), size=(h, w), mode="bilinear", align_corners=False)
        scale = torch.tensor([w / w_up, h / h_up], device=flow.device, dtype=flow.dtype).view(1, 2, 1, 1)
        return flow * scale


class SecondOrderDeformableAlignment(nn.Module):
    """Alignment deformable orde-dua berpandu flow (offset = flow + residu terbatas)."""

    def __init__(self, in_channels: int, out_channels: int, deform_groups: int = 16,
                 max_residue_magnitude: float = 10.0):
        super().__init__()
        self.deform_groups = deform_groups
        self.max_residue_magnitude = max_residue_magnitude
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3, 3))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * out_channels + 4, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, 27 * deform_groups, 3, 1, 1),
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, extra_feat, flow_1, flow_2):
        out = self.conv_offset(torch.cat([extra_feat, flow_1, flow_2], dim=1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        # flow (dx, dy) -> pasangan offset (dy, dx) untuk setiap titik kernel
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)
        return deform_conv2d(x, offset, torch.sigmoid(mask), self.weight, self.bias, padding=1)


class PixelShufflePack(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: int = 2, kernel: int = 3):
        super().__init__()
        self.scale = scale
        self.upsample_conv = nn.Conv2d(in_channels, out_channels * scale * scale, kernel, padding=kernel // 2)

    def forward(self, x):
        return F.pixel_shuffle(self.upsample_conv(x), self.scale)


class BasicVSRPlusPlus(nn.Module):
    def __init__(self, mid_channels: int = 64, num_blocks: int = 15, deform_groups: int = 16,
                 max_residue_magnitude: float = 10.0):
        super().__init__()
        self.mid_channels = mid_channels
        self.spynet = SPyNet()
        self.feat_extract = nn.Sequential(
            nn.Conv2d(3, mid_channels, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlocksWithInputConv(mid_channels, mid_channels, 5),
        )
        self.deform_align = nn.ModuleDict({
            name: SecondOrderDeformableAlignment(2 * mid_channels, mid_channels, deform_groups,
                                                 max_residue_magnitude)
            for name in BRANCHES})
        self.backbone = nn.ModuleDict({
            name: ResidualBlocksWithInputConv((2 + i) * mid_channels, mid_channels, num_blocks)
            for i, name in enumerate(BRANCHES)})
        self.reconstruction = ResidualBlocksWithInputConv(5 * mid_channels, mid_channels, 5)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, 3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, 3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

    # ------------------------------------------------------------------ helpers
    def _flows(self, lqs_ds: torch.Tensor, batch: int = 32):
        """flows_backward[i] = flow(ref=i, supp=i+1); flows_forward[i] = flow(ref=i+1, supp=i)."""
        a, b = lqs_ds[:-1], lqs_ds[1:]
        bwd, fwd = [], []
        for i in range(0, a.shape[0], batch):
            bwd.append(self.spynet(a[i:i + batch], b[i:i + batch]))
            fwd.append(self.spynet(b[i:i + batch], a[i:i + batch]))
        return torch.cat(bwd, 0), torch.cat(fwd, 0)

    def _propagate(self, feats: dict, flows: torch.Tensor, name: str) -> list:
        spatial = feats["spatial"]
        t = len(spatial)
        backward = name.startswith("backward")
        order = list(range(t))[::-1] if backward else list(range(t))
        others = [k for k in BRANCHES if k in feats and k != name]
        align = self.deform_align[name]
        body = self.backbone[name]

        outputs: list[torch.Tensor] = []
        feat_prop = torch.zeros_like(spatial[0])
        for step, idx in enumerate(order):
            current = spatial[idx]
            if step > 0:
                flow_n1 = flows[idx:idx + 1] if backward else flows[idx - 1:idx]
                flow_n1 = flow_n1.to(current.dtype)
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)
                if step > 1:
                    feat_n2 = outputs[-2]
                    flow_n2 = flows[idx + 1:idx + 2] if backward else flows[idx - 2:idx - 1]
                    flow_n2 = flow_n2.to(current.dtype)
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
                cond = torch.cat([cond_n1, current, cond_n2], dim=1)
                feat_prop = align(torch.cat([feat_prop, feat_n2], dim=1), cond, flow_n1, flow_n2)
            feat = torch.cat([current] + [feats[k][idx] for k in others] + [feat_prop], dim=1)
            feat_prop = feat_prop + body(feat)
            outputs.append(feat_prop)
        if backward:
            outputs.reverse()
        return outputs

    def _reconstruct(self, lqs: torch.Tensor, feats: dict, batch: int) -> torch.Tensor:
        t = lqs.shape[0]
        out = torch.empty_like(lqs)
        keys = ("spatial",) + BRANCHES
        for i in range(0, t, batch):
            j = min(t, i + batch)
            hr = torch.cat([torch.cat([feats[k][m] for k in keys], dim=1) for m in range(i, j)], dim=0)
            hr = self.reconstruction(hr)
            hr = F.leaky_relu(self.upsample1(hr), 0.1)
            hr = F.leaky_relu(self.upsample2(hr), 0.1)
            hr = F.leaky_relu(self.conv_hr(hr), 0.1)
            hr = self.conv_last(hr)
            out[i:j] = (hr.to(lqs.dtype) + lqs[i:j]).clamp_(0.0, 1.0)
        return out

    # ------------------------------------------------------------------ forward
    def forward(self, lqs: torch.Tensor, recon_batch: int = 8, amp=None) -> torch.Tensor:
        """lqs: [T, 3, H, W] (BGR, 0..1). ``amp``: konteks autocast opsional (FP16)."""
        if lqs.dim() == 5:
            if lqs.shape[0] != 1:
                raise ValueError("hanya batch 1 klip")
            return self.forward(lqs[0], recon_batch, amp).unsqueeze(0)
        t, c, h, w = lqs.shape
        if h % 4 or w % 4 or h < 256 or w < 256:
            raise ValueError(f"ukuran input harus kelipatan 4 dan >= 256, didapat {h}x{w}")
        from contextlib import nullcontext

        amp_ctx = amp if amp is not None else nullcontext()
        lqs = lqs.float()

        # 1) Optical flow selalu FP32 (akurasi alignment)
        lqs_ds = F.interpolate(lqs, scale_factor=0.25, mode="bicubic", align_corners=False)
        if t > 1:
            flows_bwd, flows_fwd = self._flows(lqs_ds)
        else:
            flows_bwd = flows_fwd = lqs_ds.new_zeros(0, 2, h // 4, w // 4)

        with amp_ctx:
            # 2) Fitur spasial per frame
            spatial = []
            for i in range(0, t, 16):
                f = self.feat_extract(lqs[i:i + 16])
                spatial.extend(f[k:k + 1] for k in range(f.shape[0]))
            feats: dict = {"spatial": spatial}
            # 3) Propagasi dua arah orde-dua (2 iterasi)
            for name in BRANCHES:
                feats[name] = self._propagate(feats, flows_bwd if name.startswith("backward") else flows_fwd, name)
            # 4) Rekonstruksi + residu
            return self._reconstruct(lqs, feats, recon_batch)


# ---------------------------------------------------------------------- loading
def _extract_generator_state(state: dict) -> dict:
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    for prefix in ("generator_ema.", "generator.", "module.", ""):
        sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)} if prefix else dict(state)
        if any(k.startswith("spynet.") for k in sub) and any(k.startswith("backbone.") for k in sub):
            return sub
    raise ValueError("checkpoint tidak berisi generator BasicVSR++ yang dikenali")


def build_from_state_dict(state: dict) -> BasicVSRPlusPlus:
    sd = _extract_generator_state(state)
    if any(k.startswith("feat_extract.main.") for k in sd):
        raise ValueError("varian BasicVSR++ input resolusi rendah (super-resolusi) tidak didukung")
    mid = sd["feat_extract.0.weight"].shape[0]
    block_ids = {int(m.group(1)) for k in sd
                 if (m := re.match(r"backbone\.backward_1\.main\.2\.(\d+)\.", k))}
    num_blocks = max(block_ids) + 1
    groups = sd["deform_align.backward_1.conv_offset.6.weight"].shape[0] // 27
    model = BasicVSRPlusPlus(mid_channels=mid, num_blocks=num_blocks, deform_groups=groups)
    model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    return model


def load_restoration_model(path: str | Path, device) -> BasicVSRPlusPlus:
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    model = build_from_state_dict(state)
    model.eval().requires_grad_(False)
    return model.to(device)
