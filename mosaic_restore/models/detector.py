"""Deteksi area mosaic per frame (YOLO11-seg) dengan pra/pasca-proses sendiri.

Pra-proses (YUV -> RGB letterbox) dan dekode masker dikerjakan di GPU dengan
operasi PyTorch standar sehingga berjalan di ROCm, DirectML maupun CPU.
Hanya kelas "mosaic_nsfw" yang dipakai; mosaic wajah/kepala (kelas SFW) diabaikan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..device import DeviceInfo
from ..pipeline.masks import PackedMask
from ..video.color import yuv_to_bgr
from .safe_load import load_ultralytics_checkpoint


@dataclass
class Detection:
    box: tuple[int, int, int, int]   # x0, y0, x1, y1 (x1/y1 eksklusif), koordinat frame
    score: float
    mask: PackedMask


@dataclass
class ColorParams:
    matrix: str = "bt709"
    full_range: bool = False
    bit_depth: int = 8


def _parse_output(out):
    """Dukung beberapa format output head Segment antar versi Ultralytics."""
    if isinstance(out, (list, tuple)) and len(out) == 2:
        first, second = out
        if isinstance(first, (list, tuple)) and len(first) == 2 and all(torch.is_tensor(t) for t in first):
            return first[0], first[1]
        if torch.is_tensor(first):
            if torch.is_tensor(second) and second.dim() == 4:
                return first, second
            if isinstance(second, (list, tuple)) and torch.is_tensor(second[-1]) and second[-1].dim() == 4:
                return first, second[-1]
            if isinstance(second, dict) and torch.is_tensor(second.get("proto")):
                return first, second["proto"]
    raise RuntimeError("Format output model deteksi tidak dikenali (versi ultralytics tidak cocok?)")


def nms_python(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    """NMS murni PyTorch (cadangan bila operator C++ torchvision tidak bisa dimuat)."""
    order = scores.argsort(descending=True)
    areas = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    keep: list[int] = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if rest.numel() == 0:
            break
        w = (torch.minimum(boxes[i, 2], boxes[rest, 2]) - torch.maximum(boxes[i, 0], boxes[rest, 0])).clamp(min=0)
        h = (torch.minimum(boxes[i, 3], boxes[rest, 3]) - torch.maximum(boxes[i, 1], boxes[rest, 1])).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter).clamp(min=1e-9)
        order = rest[iou <= iou_thr]
    return torch.tensor(keep, dtype=torch.long)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    try:
        from torchvision.ops import nms as tv_nms

        return tv_nms(boxes, scores, iou_thr)
    except (ImportError, RuntimeError, NotImplementedError):
        return nms_python(boxes, scores, iou_thr)


def _to_tensor(planes: list[np.ndarray]) -> torch.Tensor:
    arr = np.stack(planes)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.int16)  # nilai 10-bit muat di int16
    return torch.from_numpy(arr)


class MosaicDetector:
    def __init__(self, path, dev: DeviceInfo, imgsz: int = 640, conf: float = 0.15, iou: float = 0.7,
                 fp16: bool = False, max_det: int = 16, min_size: int = 6):
        ckpt = load_ultralytics_checkpoint(path)
        model = ckpt.get("ema")
        if model is None:
            model = ckpt.get("model")
        if model is None:
            raise RuntimeError("Checkpoint deteksi tidak berisi model")
        names = getattr(model, "names", None) or {0: "mosaic_nsfw"}
        if isinstance(names, (list, tuple)):
            names = dict(enumerate(names))
        self.names = {int(k): str(v) for k, v in names.items()}
        self.target_class = next((k for k, v in self.names.items() if v == "mosaic_nsfw"), 0)
        self.nc = len(self.names)
        model = model.float()
        try:
            model = model.fuse(verbose=False)
        except Exception:  # noqa: BLE001 - fusi opsional
            pass
        model.eval().requires_grad_(False)
        self.dev = dev
        self.device = dev.device
        self.half = bool(fp16 and dev.supports_amp)
        self.model = model.to(self.device)
        if self.half:
            self.model.half()
        strides = getattr(model, "stride", None)
        self.stride = int(max(int(s) for s in strides)) if strides is not None else 32
        self.imgsz = int(math.ceil(imgsz / self.stride) * self.stride)
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.min_size = min_size
        self._antialias = dev.backend in ("rocm", "cuda", "cpu")

    # ------------------------------------------------------------------
    def letterbox(self, h: int, w: int):
        r = min(self.imgsz / h, self.imgsz / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        dw = (self.imgsz - nw) % self.stride
        dh = (self.imgsz - nh) % self.stride
        left = int(round(dw / 2 - 0.1))
        top = int(round(dh / 2 - 0.1))
        return r, nw, nh, nw + dw, nh + dh, left, top

    def _preprocess(self, frames, color: ColorParams, lb) -> torch.Tensor:
        r, nw, nh, wl, hl, left, top = lb
        y = _to_tensor([f.y for f in frames]).to(self.device)
        u = _to_tensor([f.u for f in frames]).to(self.device)
        v = _to_tensor([f.v for f in frames]).to(self.device)
        bgr = yuv_to_bgr(y, u, v, color.matrix, color.full_range, color.bit_depth, half_res=r <= 0.5)
        rgb = bgr.flip(1)
        downscale = nh < rgb.shape[-2]
        try:
            rgb = F.interpolate(rgb, size=(nh, nw), mode="bilinear", align_corners=False,
                                antialias=bool(downscale and self._antialias))
        except (RuntimeError, NotImplementedError):
            self._antialias = False
            rgb = F.interpolate(rgb, size=(nh, nw), mode="bilinear", align_corners=False)
        canvas = torch.full((len(frames), 3, hl, wl), 114.0 / 255.0, device=rgb.device, dtype=rgb.dtype)
        canvas[:, :, top:top + nh, left:left + nw] = rgb
        return canvas.half() if self.half else canvas

    @torch.no_grad()
    def detect(self, frames, color: ColorParams) -> list[list[Detection]]:
        if not frames:
            return []
        h, w = frames[0].y.shape
        lb = self.letterbox(h, w)
        x = self._preprocess(frames, color, lb)
        preds, proto = _parse_output(self.model(x))
        preds = preds.float()
        proto = proto.float()
        return [self._postprocess(preds[i], proto[i], lb, h, w) for i in range(len(frames))]

    def _postprocess(self, pred: torch.Tensor, proto: torch.Tensor, lb, h: int, w: int) -> list[Detection]:
        nc = self.nc
        score, cls = pred[4:4 + nc].max(0)
        keep = (score > self.conf) & (cls == self.target_class)
        if not bool(keep.any()):
            return []
        cand = pred[:, keep].t().contiguous()           # [n, 4+nc+nm]
        cand_score = score[keep]
        xywh = cand[:, :4]
        boxes = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], dim=1)
        boxes_cpu = boxes.cpu()
        scores_cpu = cand_score.cpu()
        order = nms(boxes_cpu, scores_cpu, self.iou)[: self.max_det]
        if order.numel() == 0:
            return []
        coeffs = cand[order.to(cand.device), 4 + nc:]
        nm, ph, pw = proto.shape
        logits = (coeffs @ proto.reshape(nm, -1)).reshape(-1, ph, pw)
        r, nw, nh, wl, hl, left, top = lb

        dets: list[Detection] = []
        for j, idx in enumerate(order.tolist()):
            bx = boxes_cpu[idx].tolist()
            fx0, fy0 = (bx[0] - left) / r, (bx[1] - top) / r
            fx1, fy1 = (bx[2] - left) / r, (bx[3] - top) / r
            x0, y0 = max(0, int(math.floor(fx0))), max(0, int(math.floor(fy0)))
            x1, y1 = min(w, int(math.ceil(fx1))), min(h, int(math.ceil(fy1)))
            if x1 - x0 < self.min_size or y1 - y0 < self.min_size:
                continue
            xs = (torch.arange(x0, x1, device=logits.device, dtype=torch.float32) + 0.5) * r + left
            ys = (torch.arange(y0, y1, device=logits.device, dtype=torch.float32) + 0.5) * r + top
            gx = xs * (2.0 / wl) - 1.0
            gy = ys * (2.0 / hl) - 1.0
            grid = torch.stack(torch.meshgrid(gy, gx, indexing="ij")[::-1], dim=-1)[None]
            m = F.grid_sample(logits[j][None, None], grid, mode="bilinear", padding_mode="border",
                              align_corners=False)[0, 0] > 0
            mask = m.cpu().numpy()
            if not mask.any():
                continue
            ys_idx = np.flatnonzero(mask.any(1))
            xs_idx = np.flatnonzero(mask.any(0))
            ty0, ty1 = int(ys_idx[0]), int(ys_idx[-1]) + 1
            tx0, tx1 = int(xs_idx[0]), int(xs_idx[-1]) + 1
            mask = mask[ty0:ty1, tx0:tx1]
            dets.append(Detection((x0 + tx0, y0 + ty0, x0 + tx1, y0 + ty1), float(scores_cpu[idx]),
                                  PackedMask.pack(x0 + tx0, y0 + ty0, mask)))
        return dets
