"""Antarmuka baris perintah: mosaic-restore."""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

from . import __version__
from .config import PRESETS, settings_for_preset

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".m4v", ".avi", ".wmv", ".ts", ".m2ts", ".mts", ".flv", ".webm", ".mpg", ".mpeg"}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def parse_time(text: str | None) -> float | None:
    if text is None:
        return None
    parts = str(text).strip().split(":")
    try:
        value = 0.0
        for p in parts:
            value = value * 60 + float(p)
        return value
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"format waktu tidak valid: {text} (contoh: 90, 1:30, 00:01:30)") from exc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mosaic-restore",
        description="Restorasi area mosaic pada video dengan GPU AMD (ROCm / DirectML).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Contoh:\n"
                "  mosaic-restore video.mp4\n"
                "  mosaic-restore D:\\Video\\ -o D:\\Hasil\\ --preset balanced\n"
                "  mosaic-restore video.mp4 --start 10:00 --duration 30   (pratinjau 30 detik)\n"))
    p.add_argument("inputs", nargs="*", help="file video dan/atau folder")
    p.add_argument("-o", "--output", help="file output (untuk 1 input) atau folder output")
    p.add_argument("--preset", choices=list(PRESETS), default="best",
                   help="best = kualitas maksimum (default), balanced = FP16 lebih cepat, fast = paling cepat")
    p.add_argument("--overwrite", action="store_true", help="timpa file output yang sudah ada")
    p.add_argument("--suffix", default=".restored", help="akhiran nama file output (default: .restored)")
    p.add_argument("--container", choices=["mp4", "mkv", "mov"], help="format kontainer output")
    p.add_argument("--keep-going", action="store_true", help="lanjutkan ke video berikutnya bila terjadi error")

    g = p.add_argument_group("perangkat")
    g.add_argument("--device", default="auto", help="auto | rocm | directml | cuda | cpu (default: auto)")
    g.add_argument("--gpu-index", type=int, default=0, help="indeks GPU bila ada lebih dari satu")
    g.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None,
                   help="paksa presisi FP16 aktif/nonaktif (default mengikuti preset)")
    g.add_argument("--hsa-override-gfx", metavar="X.Y.Z",
                   help="Linux/ROCm: set HSA_OVERRIDE_GFX_VERSION (mis. 10.3.0 untuk RX 6700/6600)")
    g.add_argument("--list-devices", action="store_true", help="tampilkan GPU yang terdeteksi lalu keluar")

    g = p.add_argument_group("model")
    g.add_argument("--model-dir", help="folder penyimpanan bobot model")
    g.add_argument("--det-model", help="v4-accurate | v4-fast | path .pt")
    g.add_argument("--restore-model", help="basicvsrpp-v1.2 | path .pth")
    g.add_argument("--download-models", action="store_true", help="unduh semua model lalu keluar (jika tanpa input)")

    g = p.add_argument_group("deteksi & pelacakan")
    g.add_argument("--det-conf", type=float, help="ambang kepercayaan deteksi (default 0.15)")
    g.add_argument("--det-size", type=int, help="resolusi deteksi (default 640)")
    g.add_argument("--det-batch", type=int, help="jumlah frame per batch deteksi (default 8)")
    g.add_argument("--max-gap", type=int, help="isi celah deteksi hingga N frame (default 4)")
    g.add_argument("--no-scene-cut", action="store_true", help="jangan putus jejak di pergantian adegan")

    g = p.add_argument_group("restorasi")
    g.add_argument("--clip-length", type=int, help="panjang klip temporal maks (default 180)")
    g.add_argument("--clip-overlap", type=int, help="tumpang-tindih antar klip (default 24)")
    g.add_argument("--restore-size", type=int,
                   help="ukuran input model: 256 (default, sesuai pelatihan) atau 0 = otomatis hingga 512 (eksperimental)")
    g.add_argument("--debug-overlay", action="store_true",
                   help="jangan restorasi; tandai area terdeteksi (untuk memeriksa deteksi)")

    g = p.add_argument_group("encoding")
    g.add_argument("--encoder", default="auto",
                   help="auto | hevc_amf | h264_amf | av1_amf | hevc_vaapi | h264_vaapi | av1_vaapi | "
                        "libx265 | libx264 | libsvtav1")
    g.add_argument("--cq", type=int, help="kualitas konstan, makin kecil makin bagus (default 18)")
    g.add_argument("--encoder-preset", help="preset encoder (AMF: quality/balanced/speed; x265: slow/medium/...)")
    g.add_argument("--hwdec", action="store_true", help="decode video dengan GPU (hemat CPU untuk 4K/HEVC)")
    g.add_argument("--vaapi-device", help="Linux: node render VA-API (default /dev/dri/renderD128)")
    g.add_argument("--ffmpeg", help="path ke ffmpeg (ffprobe dicari di folder yang sama)")

    g = p.add_argument_group("segmen (pratinjau)")
    g.add_argument("--start", type=parse_time, help="mulai dari waktu ini (detik atau HH:MM:SS)")
    g.add_argument("--duration", type=parse_time, help="hanya proses selama durasi ini")

    p.add_argument("-q", "--quiet", action="store_true", help="tanpa indikator kemajuan")
    p.add_argument("--version", action="version", version=f"mosaic-restore {__version__}")
    return p


def collect_inputs(items: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in items:
        path = Path(item.strip('"')).expanduser()
        if path.is_dir():
            files += sorted(f for f in path.iterdir() if f.suffix.lower() in VIDEO_EXTS and ".restored" not in f.name
                            and ".part." not in f.name)
        elif path.exists():
            files.append(path)
        else:
            log(f"Peringatan: tidak ditemukan: {item}")
    return files


def output_for(src: Path, args, multiple: bool) -> Path:
    ext = f".{args.container}" if args.container else (src.suffix.lower() if src.suffix.lower() in (".mp4", ".mkv", ".mov") else ".mp4")
    if src.suffix.lower() == ".webm" and not args.container:
        ext = ".mkv"
    name = f"{src.stem}{args.suffix}{ext}"
    if args.output:
        out = Path(args.output).expanduser()
        if not multiple and out.suffix.lower() in (".mp4", ".mkv", ".mov") and not out.is_dir():
            return out
        return out / name
    return src.with_name(name)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from .device import prepare_environment

    prepare_environment(args.hsa_override_gfx)

    from .video import probe as probe_mod

    if args.ffmpeg:
        probe_mod.set_ffmpeg_paths(args.ffmpeg)

    from .device import gpu_setup_hint, list_devices, select_device

    if args.list_devices:
        for line in list_devices():
            log(line)
        return 0

    from .models import registry

    if args.download_models:
        registry.download_all(args.model_dir, log=log)
        if not args.inputs:
            return 0

    if not args.inputs:
        build_parser().print_help()
        return 2

    try:
        log(probe_mod.check_ffmpeg())
    except RuntimeError as exc:
        log(f"ERROR: {exc}")
        return 1

    files = collect_inputs(args.inputs)
    if not files:
        log("ERROR: tidak ada file video yang bisa diproses.")
        return 1

    settings = settings_for_preset(args.preset).updated(
        device=args.device, gpu_index=args.gpu_index, fp16=args.fp16, model_dir=args.model_dir,
        det_model=args.det_model, restore_model=args.restore_model, det_conf=args.det_conf,
        det_size=args.det_size, det_batch=args.det_batch, max_gap=args.max_gap,
        clip_length=args.clip_length, clip_overlap=args.clip_overlap, restore_size=args.restore_size,
        encoder=args.encoder, cq=args.cq, encoder_preset=args.encoder_preset, hwdec=args.hwdec or None,
        vaapi_device=args.vaapi_device, start=args.start, duration=args.duration)
    if args.no_scene_cut:
        settings.scene_cut = False

    try:
        dev = select_device(settings.device, settings.gpu_index)
    except Exception as exc:  # noqa: BLE001
        log(f"ERROR: {exc}")
        return 1
    log(f"Perangkat: {dev.describe()}" + (f"  [{', '.join(dev.notes)}]" if dev.notes else ""))
    if not dev.is_gpu:
        log("PERINGATAN: berjalan di CPU - akan SANGAT lambat.\n" + gpu_setup_hint())
    if dev.backend == "directml" and settings.fp16:
        settings.fp16 = False
    if dev.vram_bytes and dev.vram_bytes < 5 * 2**30 and settings.clip_length > 120:
        settings.clip_length = 120
        log("Info: VRAM < 5 GB -> panjang klip dibatasi 120 frame.")

    try:
        det_path = registry.resolve_model(settings.det_model, "detection", settings.model_dir, log=log)
        res_path = registry.resolve_model(settings.restore_model, "restoration", settings.model_dir, log=log)
    except Exception as exc:  # noqa: BLE001
        log(f"ERROR model: {exc}")
        return 1

    import torch

    from .models.detector import MosaicDetector
    from .pipeline.engine import Processor
    from .pipeline.restorer import ClipRestorer
    from .video.encoders import select_encoder

    log(f"Model deteksi  : {det_path.name}")
    log(f"Model restorasi: {res_path.name}")
    log(f"Preset {args.preset}: FP16={'ya' if settings.fp16 and dev.supports_amp else 'tidak'}, "
        f"klip {settings.clip_length} (overlap {settings.clip_overlap}), celah maks {settings.max_gap}, "
        f"ukuran restorasi {settings.restore_size or 'otomatis'}")
    detector = MosaicDetector(det_path, dev, imgsz=settings.det_size, conf=settings.det_conf,
                              iou=settings.det_iou, fp16=settings.fp16)
    restorer = ClipRestorer(res_path, dev, fp16=settings.fp16, recon_batch=settings.recon_batch)
    processor = Processor(settings, dev, detector, restorer, log=log, show_progress=not args.quiet,
                          debug=args.debug_overlay)

    def encoder_factory(info):
        return select_encoder(settings.encoder, settings.cq, info.work_bit_depth, settings.encoder_preset,
                              settings.vaapi_device, full_range=info.full_range, log=log)

    multiple = len(files) > 1
    failures = 0
    for n, src in enumerate(files, 1):
        dst = output_for(src, args, multiple)
        log("")
        log(f"[{n}/{len(files)}] {src}")
        if dst.exists() and not args.overwrite:
            log(f"  Lewati: output sudah ada ({dst}). Pakai --overwrite untuk menimpa.")
            continue
        if dst.resolve() == src.resolve():
            log("  Lewati: path output sama dengan input.")
            continue
        t0 = time.monotonic()
        try:
            result = processor.process(src, dst, encoder_factory)
        except KeyboardInterrupt:
            log("\nDibatalkan oleh pengguna.")
            return 130
        except Exception as exc:  # noqa: BLE001
            failures += 1
            log(f"  GAGAL: {exc}")
            if os.environ.get("MOSAIC_RESTORE_DEBUG"):
                traceback.print_exc()
            if not args.keep_going:
                return 1
            continue
        finally:
            if dev.backend in ("rocm", "cuda"):
                torch.cuda.empty_cache()
        if result.skipped:
            log(f"  {result.message}")
            continue
        st = result.stats
        total = time.monotonic() - t0
        log(f"  Selesai -> {result.output}")
        log(f"  {st['frames']} frame, {st['tracks']} jejak mosaic, {st['chunks']} klip, "
            f"total {total:.1f} dtk ({st['frames'] / max(total, 1e-6):.1f} fps keseluruhan)")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
