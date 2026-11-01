"""Pengaturan program dan preset kualitas."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace


@dataclass
class Settings:
    # --- perangkat ---
    device: str = "auto"            # auto | rocm | directml | cuda | cpu
    gpu_index: int | None = None    # None = otomatis: GPU berfungsi dengan VRAM terbesar
    fp16: bool = False              # presisi setengah (lebih cepat, sedikit kurang akurat)

    # --- model ---
    model_dir: str | None = None
    det_model: str = "v4-accurate"
    restore_model: str = "basicvsrpp-v1.2"

    # --- deteksi ---
    det_size: int = 640             # resolusi input YOLO (model dilatih di 640)
    det_conf: float = 0.15          # ambang kepercayaan deteksi
    det_iou: float = 0.7            # ambang IoU untuk NMS
    det_batch: int = 8

    # --- pelacakan (tracking) ---
    max_gap: int = 4                # isi celah deteksi yang hilang s/d N frame
    min_track_len: int = 2          # buang jejak (track) yang lebih pendek (kecuali skor tinggi)
    scene_cut: bool = True          # putus jejak saat pergantian adegan

    # --- restorasi ---
    clip_length: int = 180          # panjang klip temporal maksimum (frame)
    clip_overlap: int = 24          # tumpang-tindih antar klip (cross-fade)
    restore_size: int = 256         # 256 = sesuai pelatihan; 0 = otomatis (eksperimental)
    recon_batch: int = 8            # jumlah frame per batch rekonstruksi
    context_ratio: float = 0.15     # konteks di sekitar area mosaic (rasio ukuran kotak)
    smooth_sigma: float = 3.0       # penghalusan temporal posisi jendela crop (frame)

    # --- pencampuran (blending) ---
    dilate_ratio: float = 0.03      # perluasan masker relatif terhadap ukuran area
    feather_ratio: float = 0.045    # lebar gradasi tepi relatif terhadap ukuran area

    # --- encoding ---
    encoder: str = "auto"
    cq: int = 18                    # kualitas (makin kecil makin bagus/besar)
    encoder_preset: str | None = None
    hwdec: bool = False
    vaapi_device: str = "/dev/dri/renderD128"

    # --- segmen (untuk pratinjau) ---
    start: float | None = None
    duration: float | None = None

    def updated(self, **kwargs) -> "Settings":
        valid = {f.name for f in fields(self)}
        clean = {k: v for k, v in kwargs.items() if k in valid and v is not None}
        return replace(self, **clean)


PRESETS: dict[str, dict] = {
    # Kualitas & akurasi maksimum (default).
    "best": dict(det_model="v4-accurate", fp16=False, clip_length=180, clip_overlap=24,
                 max_gap=4, det_conf=0.15, recon_batch=8),
    # Hampir sama bagusnya, lebih cepat di GPU dengan FP16 cepat (RDNA2/3/4).
    "balanced": dict(det_model="v4-accurate", fp16=True, clip_length=150, clip_overlap=16,
                     max_gap=4, det_conf=0.15, recon_batch=8),
    # Untuk GPU kecil / pratinjau cepat.
    "fast": dict(det_model="v4-fast", fp16=True, clip_length=90, clip_overlap=8,
                 max_gap=3, det_conf=0.2, recon_batch=4),
}


def settings_for_preset(name: str) -> Settings:
    if name not in PRESETS:
        raise ValueError(f"Preset tidak dikenal: {name} (pilihan: {', '.join(PRESETS)})")
    return Settings().updated(**PRESETS[name])
