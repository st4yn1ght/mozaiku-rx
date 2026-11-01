"""Uji unit cepat (CPU, tanpa unduhan model)."""

import io
import os
from pathlib import Path
import pickle
import zipfile

import numpy as np
import pytest
import torch

from mosaic_restore.models.deform_conv import deform_conv2d_portable
from mosaic_restore.models.safe_load import UnsafeCheckpointError, scan_pickle_globals
from mosaic_restore.pipeline.blender import composite, make_alpha
from mosaic_restore.pipeline.masks import PackedMask, union
from mosaic_restore.pipeline.tracker import (SceneCutDetector, Track, TrackFrame, Window, build_tracks,
                                             plan_chunks, plan_windows)
from mosaic_restore.models.detector import ColorParams, Detection
from mosaic_restore.video.color import bgr_to_yuv, yuv_to_bgr


# ---------------------------------------------------------------- deform conv
@pytest.mark.parametrize("b,c,h,w,g", [(1, 128, 32, 32, 16), (2, 16, 9, 13, 4)])
def test_portable_deform_conv_matches_torchvision(b, c, h, w, g):
    tv = pytest.importorskip("torchvision.ops")
    torch.manual_seed(0)
    x = torch.randn(b, c, h, w)
    off = torch.randn(b, 2 * g * 9, h, w) * 3
    m = torch.rand(b, g * 9, h, w)
    wt = torch.randn(32, c, 3, 3) * 0.05
    bias = torch.randn(32)
    ref = tv.deform_conv2d(x, off, wt, bias, padding=(1, 1), mask=m)
    out = deform_conv2d_portable(x, off, m, wt, bias, padding=1)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------- color
def _smooth_bgr(h=48, w=64):
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32),
                            indexing="ij")
    b = 0.5 + 0.3 * torch.sin(xx / 9.0)
    g = 0.5 + 0.3 * torch.cos(yy / 11.0)
    r = 0.5 + 0.3 * torch.sin((xx + yy) / 13.0)
    return torch.stack((b, g, r))[None]


@pytest.mark.parametrize("matrix", ["bt601", "bt709", "bt2020"])
@pytest.mark.parametrize("full_range,bits", [(False, 8), (True, 8), (False, 10)])
def test_color_roundtrip(matrix, full_range, bits):
    bgr = _smooth_bgr()
    y, u, v = bgr_to_yuv(bgr, matrix, full_range, bits)
    back = yuv_to_bgr(y.round(), u.round(), v.round(), matrix, full_range, bits)
    inner = (back - bgr)[..., 2:-2, 2:-2].abs()
    assert inner.mean() < 1.0 / 255 and inner.max() < 6.0 / 255


def test_bt601_matrix_matches_opencv():
    """Chroma seragam -> metode interpolasi tidak berpengaruh; hanya matriks yang diuji."""
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    for _ in range(5):
        i420 = np.empty((24, 16), np.uint8)
        i420[:16] = rng.integers(16, 236, (16, 16))
        i420[16:20] = rng.integers(16, 241)
        i420[20:24] = rng.integers(16, 241)
        y = torch.from_numpy(i420[:16]).float()[None]
        u = torch.from_numpy(i420[16:20].reshape(8, 8)).float()[None]
        v = torch.from_numpy(i420[20:24].reshape(8, 8)).float()[None]
        ours = (yuv_to_bgr(y, u, v, "bt601", False, 8)[0].permute(1, 2, 0).numpy() * 255)
        ref = cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420).astype(np.float32)
        ok = (ref > 0) & (ref < 255)  # abaikan nilai yang terpotong
        assert np.abs(ours - ref)[ok].max() <= 1.5


# ---------------------------------------------------------------- masks
def test_packed_mask_roundtrip_and_union():
    m = np.zeros((7, 11), bool)
    m[2:5, 3:9] = True
    pm = PackedMask.pack(10, 20, m)
    assert np.array_equal(pm.unpack(), m)
    other = PackedMask.from_box(15, 22, 30, 25)
    u = union([pm, other])
    assert u.box == (10, 20, 30, 27)
    crop = u.crop(10, 20, 20, 7)
    assert crop[2, 3] and crop[3, 19] and not crop[0, 0]


# ---------------------------------------------------------------- tracking
def _det(x0, y0, s=50, score=0.9):
    return Detection((x0, y0, x0 + s, y0 + s), score, PackedMask.from_box(x0, y0, x0 + s, y0 + s))


def test_tracker_fills_gaps_and_splits_on_cut():
    dets = []
    for f in range(30):
        if f in (5, 6, 7):          # celah 3 frame -> harus diisi
            dets.append([])
        else:
            dets.append([_det(100 + f, 80)])
    cuts = [False] * 30
    cuts[20] = True                 # pergantian adegan -> jejak dipecah
    tracks = build_tracks(dets, cuts, max_gap=4, min_len=2)
    assert [(t.start, t.end) for t in tracks] == [(0, 19), (20, 29)]
    filled = [tf.filled for tf in tracks[0].frames]
    assert filled[5:8] == [True, True, True] and not any(filled[:5])


def test_tracker_drops_isolated_false_positive():
    dets = [[] for _ in range(20)]
    dets[10] = [_det(10, 10, score=0.2)]
    assert build_tracks(dets, [False] * 20) == []


def test_tracker_drops_short_weak_flicker_but_keeps_long_weak_track():
    weak = [[_det(50, 50, score=0.2)] if f < 4 else [] for f in range(40)]
    assert build_tracks(weak, [False] * 40) == []
    long_weak = [[_det(50, 50, score=0.2)] for _ in range(40)]
    assert len(build_tracks(long_weak, [False] * 40)) == 1


def test_tracker_does_not_bridge_long_gaps():
    dets = [[_det(100, 100)] if (f < 5 or f > 15) else [] for f in range(20)]
    tracks = build_tracks(dets, [False] * 20, max_gap=4)
    assert len(tracks) == 2


# ---------------------------------------------------------------- planning
def test_windows_cover_box_and_are_even():
    track = Track(0, 0, [TrackFrame((100 + 3 * i, 50, 180 + 3 * i, 140), PackedMask.from_box(0, 0, 1, 1), 0.9)
                         for i in range(40)])
    wins = plan_windows(track, 640, 360)
    for tf, w in zip(track.frames, wins):
        assert w.x0 % 2 == 0 and w.y0 % 2 == 0
        assert w.x0 <= tf.box[0] and w.y0 <= tf.box[1] and w.x1 >= tf.box[2] and w.y1 >= tf.box[3]
        assert 0 <= w.x0 and w.x1 <= 640 and 0 <= w.y0 and w.y1 <= 360
    sizes = [w.w for w in wins]
    assert max(sizes) - min(sizes) <= 4  # ukuran stabil


def test_chunks_cover_track_with_crossfade():
    n = 400
    track = Track(0, 10, [TrackFrame((0, 0, 10, 10), PackedMask.from_box(0, 0, 10, 10), 1.0) for _ in range(n)])
    wins = [Window(0, 0, 64, 64)] * n
    chunks = plan_chunks(track, wins, max_len=180, overlap=24, restore_size=256, next_id=0)
    assert chunks[0].start == 10 and chunks[-1].end == 10 + n - 1
    assert all(len(c) <= 180 for c in chunks)
    for f in range(10, 10 + n):
        total = sum(c.weight(f) for c in chunks if c.start <= f <= c.end)
        assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------- blending
class _Frame:
    def __init__(self, h, w):
        rng = np.random.default_rng(3)
        self.y = rng.integers(16, 236, (h, w), dtype=np.uint8)
        self.u = rng.integers(16, 241, ((h + 1) // 2, (w + 1) // 2), dtype=np.uint8)
        self.v = rng.integers(16, 241, ((h + 1) // 2, (w + 1) // 2), dtype=np.uint8)
        self.bit_depth = 8


def test_alpha_profile():
    m = torch.zeros(64, 64)
    m[24:40, 24:40] = 1
    a = make_alpha(m, dilate=3, feather=6)
    assert a[32, 32] == 1.0 and a[0, 0] == 0.0
    assert 0.0 < a[32, 16] < 1.0  # gradasi di luar area yang diperlebar


def test_composite_keeps_untouched_pixels_bit_exact():
    fr = _Frame(120, 160)
    orig_y, orig_u, orig_v = fr.y.copy(), fr.u.copy(), fr.v.copy()
    win = Window(40, 20, 80, 80)
    mask = PackedMask.from_box(70, 50, 90, 70)
    restored = torch.rand(3, 80, 80)
    composite(fr, win, restored, mask, dilate=2, feather=4, color=ColorParams("bt709", False, 8),
              frame_w=160, frame_h=120)
    changed = fr.y != orig_y
    assert changed[55:65, 75:85].any()
    outside = np.ones_like(changed)
    outside[50 - 10:70 + 10, 70 - 10:90 + 10] = False
    assert not changed[outside].any()
    assert np.array_equal(fr.u[:10], orig_u[:10]) and np.array_equal(fr.v[:, :15], orig_v[:, :15])


# ---------------------------------------------------------------- safety
def _zip_with_pickle(payload: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", payload)
    return buf.getvalue()


def test_scan_rejects_dangerous_globals(tmp_path):
    # pickle protokol 2 yang mereferensikan os.system (TIDAK pernah dieksekusi; hanya dipindai)
    evil = b"\x80\x02cos\nsystem\nq\x00X\x04\x00\x00\x00echoq\x01\x85q\x02Rq\x03."
    p = tmp_path / "evil.pt"
    p.write_bytes(_zip_with_pickle(evil))
    with pytest.raises(UnsafeCheckpointError):
        scan_pickle_globals(p)


def test_scan_accepts_plain_state_dict(tmp_path):
    p = tmp_path / "ok.pt"
    torch.save({"w": torch.zeros(2)}, p)
    found = scan_pickle_globals(p)
    assert all(m.startswith(("torch", "collections")) for m, _ in found)


def test_restricted_unpickler_fallback_loads_local_yolo(tmp_path, monkeypatch):
    """Jalur cadangan untuk PyTorch lama (DirectML): unpickler terbatas memuat checkpoint YOLO buatan lokal."""
    tasks = pytest.importorskip("ultralytics.nn.tasks")
    from mosaic_restore.models import safe_load

    model = tasks.SegmentationModel("yolo11n-seg.yaml", nc=2, verbose=False)
    path = tmp_path / "local_yolo.pt"
    torch.save({"model": model.half(), "ema": None, "version": "test"}, path)
    monkeypatch.setattr(torch.serialization, "add_safe_globals", None, raising=False)
    ckpt = safe_load.load_ultralytics_checkpoint(path)
    assert type(ckpt["model"]).__name__ == "SegmentationModel"
    out = ckpt["model"].float().eval()(torch.zeros(1, 3, 64, 64))
    assert out is not None


def test_scan_tracks_memoized_module_names(tmp_path):
    # Protokol 4: nama modul diambil dari memo (BINGET) saat kelas kedua dari modul yang sama dipakai.
    import collections
    data = pickle.dumps([collections.OrderedDict(), collections.Counter()], protocol=4)
    p = tmp_path / "memo.pt"
    p.write_bytes(_zip_with_pickle(data))
    with pytest.raises(UnsafeCheckpointError) as err:
        scan_pickle_globals(p)
    assert "collections.Counter" in str(err.value)


# ---------------------------------------------------------------- lokasi FFmpeg
import shutil as _shutil  # noqa: E402

from mosaic_restore.video import probe as _probe  # noqa: E402

REAL_FFMPEG = _shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(REAL_FFMPEG is None, reason="ffmpeg tidak terpasang")


@pytest.fixture
def fresh_resolver(monkeypatch):
    monkeypatch.setattr(_probe, "_RESOLVED", {})
    monkeypatch.setattr(_probe, "_OVERRIDE", {"ffmpeg": None, "ffprobe": None})
    yield


@needs_ffmpeg
def test_resolver_skips_broken_candidates(tmp_path, monkeypatch, fresh_resolver):
    folder = tmp_path / "ffmpeg"            # folder bernama "ffmpeg" (bukan executable)
    folder.mkdir()
    bad = tmp_path / "bad" / "ffmpeg"       # file bernama ffmpeg yang gagal dijalankan
    bad.parent.mkdir()
    bad.write_text("#!/bin/sh\nexit 3\n")
    bad.chmod(0o755)
    missing = tmp_path / "nope" / "ffmpeg"  # path tidak ada
    real_dir = Path(REAL_FFMPEG).parent
    monkeypatch.setattr(_probe, "_candidates",
                        lambda tool: [folder, bad, missing, real_dir / tool])
    ffmpeg, ffprobe, version = _probe.resolve_ffmpeg()
    assert Path(ffmpeg).name == "ffmpeg" and Path(ffprobe).name == "ffprobe"
    assert "version" in version


def test_resolver_raises_friendly_error_not_oserror(tmp_path, monkeypatch, fresh_resolver):
    (tmp_path / "ffmpeg").mkdir()
    monkeypatch.setattr(_probe, "_candidates", lambda tool: [tmp_path / "ffmpeg", tmp_path / "x" / tool])
    with pytest.raises(RuntimeError) as err:
        _probe.resolve_ffmpeg()
    assert "winget install Gyan.FFmpeg" in str(err.value)
    assert "bukan file" in str(err.value)


@needs_ffmpeg
def test_override_accepts_folder_with_symlinks(tmp_path, fresh_resolver):
    link_dir = tmp_path / "links"           # meniru folder "WinGet\Links" berisi symlink
    link_dir.mkdir()
    for tool in ("ffmpeg", "ffprobe"):
        (link_dir / tool).symlink_to(_shutil.which(tool))
    _probe.set_ffmpeg_paths(str(link_dir))
    ffmpeg, ffprobe, _ = _probe.resolve_ffmpeg()
    assert Path(ffmpeg) == Path(os.path.realpath(_shutil.which("ffmpeg")))
    assert Path(ffprobe) == Path(os.path.realpath(_shutil.which("ffprobe")))


def test_broken_override_does_not_fall_back_silently(tmp_path, fresh_resolver):
    _probe.set_ffmpeg_paths(str(tmp_path / "salah" / "ffmpeg.exe"))
    with pytest.raises(RuntimeError) as err:
        _probe.resolve_ffmpeg()
    assert "--ffmpeg" in str(err.value)


# ---------------------------------------------------------------- NMS cadangan
def test_python_nms_matches_torchvision():
    tv = pytest.importorskip("torchvision.ops")
    from mosaic_restore.models.detector import nms_python

    torch.manual_seed(5)
    xy = torch.rand(60, 2) * 300
    wh = torch.rand(60, 2) * 120 + 5
    boxes = torch.cat([xy, xy + wh], 1)
    scores = torch.rand(60)
    for thr in (0.3, 0.5, 0.7):
        assert nms_python(boxes, scores, thr).tolist() == tv.nms(boxes, scores, thr).tolist()
