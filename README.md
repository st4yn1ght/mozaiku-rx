# Mozaiku RX: Mosaic Restore untuk GPU AMD

Program untuk merestorasi area mosaic (pixelasi) pada video JAV menggunakan AI,
dioptimalkan untuk **GPU AMD** (ROCm di Linux/Windows, DirectML sebagai cadangan
di Windows) dengan encoder video hardware AMD (AMF / VA-API).

> **Penting: apa yang sebenarnya dilakukan program ini.** Mosaic menghapus detail
> asli secara permanen. Tidak ada program yang bisa "membuka" mosaic. Yang dilakukan
> program ini adalah **merekonstruksi** detail yang masuk akal dengan model AI,
> berdasarkan warna blok mosaic dan informasi dari frame-frame di sekitarnya.
> Mosaic tipis atau berblok kecil biasanya menghasilkan gambar yang cukup realistis.
> Mosaic tebal atau berblok besar bisa menghasilkan gambar buram atau artefak.

## Fitur

- **Deteksi akurat**: YOLO11-seg (model `v4-accurate`) mendeteksi area mosaic per frame
  dengan masker per piksel. Hanya mosaic area NSFW yang diproses. Mosaic wajah/kepala
  sengaja **tidak** disentuh.
- **Restorasi temporal**: BasicVSR++ memakai propagasi dua arah orde-dua dengan
  alignment optical flow. Klip hingga 180 frame dengan cross-fade antar klip, sehingga
  hasil stabil dan minim kedipan (flicker).
- **Analisis dua tahap**:
  - deteksi yang terlewat 1–4 frame diisi otomatis (mosaic tidak "muncul sekilas"),
  - jejak diputus di pergantian adegan,
  - jendela crop dihaluskan secara temporal.
- **Piksel lain tidak tersentuh**: pemrosesan langsung di plane YUV. Piksel di luar
  area mosaic identik bit-per-bit dengan aslinya sebelum di-encode ulang.
- **Dukungan AMD penuh**:
  - ROCm/HIP dengan *tuning* kernel MIOpen,
  - DirectML untuk GPU lama di Windows,
  - deformable convolution portabel (tanpa ekstensi C++),
  - encoder hardware `hevc_amf`/`av1_amf` (Windows) dan `hevc_vaapi` (Linux), dengan
    fallback otomatis ke `libx265`.
- **Lainnya**: audio dan subtitle disalin tanpa re-encode, metadata warna
  (BT.601/709/2020, limited/full range) dipertahankan, mendukung video 10-bit,
  pemrosesan banyak file (folder), dan mode pratinjau segmen.
- **Aman**: semua model diverifikasi SHA-256 dan dimuat dengan *unpickler* terbatas
  (tidak bisa menjalankan kode tersembunyi).

## Dukungan GPU AMD

| GPU | Windows | Linux |
|---|---|---|
| Radeon RX 9000 (RDNA4) | ROCm (driver Adrenalin terbaru) | ROCm |
| Radeon RX 7000 (RDNA3), Ryzen AI 300 / AI Max | ROCm | ROCm |
| Radeon RX 6000 (RDNA2) | DirectML (ROCm dicoba otomatis) | ROCm (RX 6700/6600 perlu `--hsa-override-gfx 10.3.0`) |
| RX 5000 / Vega / Polaris / iGPU lama | DirectML | tidak didukung ROCm modern (CPU) |

ROCm **jauh lebih cepat** daripada DirectML. Mode CPU tetap bisa jalan, tetapi sangat
lambat (sekitar 1 frame/detik untuk 360p). Mode CPU hanya cocok untuk uji coba.

Kebutuhan yang disarankan: VRAM 6 GB atau lebih (4 GB tetap bisa dengan klip lebih
pendek), RAM 16 GB, dan ruang disk kosong untuk file output.

## Instalasi

### Unduh kode

```bash
git clone https://github.com/st4yn1ght/mozaiku-rx.git
cd mozaiku-rx
```

Atau unduh ZIP lewat tombol **Code → Download ZIP** di halaman repositori, lalu ekstrak.

### Windows 10/11

1. Perbarui driver **AMD Software: Adrenalin Edition** ke versi terbaru.
2. Pasang **Python 3.12**: `winget install Python.Python.3.12`. Versi 3.12 cocok untuk
   ROCm maupun DirectML.
3. Klik dua kali **`install_windows.bat`**. Installer akan:
   - mendeteksi GPU (mis. RX 7900 XTX → `gfx1100`),
   - mencoba memasang PyTorch ROCm untuk Windows. Jika gagal, installer otomatis beralih
     ke DirectML,
   - menawarkan pemasangan FFmpeg (`winget install Gyan.FFmpeg`, build ini sudah
     mendukung encoder AMD AMF),
   - mengunduh model (~130 MB) dan menampilkan GPU yang terdeteksi.

Opsi installer:

```bat
install_windows.bat -DirectML
```
Paksa DirectML, misalnya untuk RX 5000/6000.

```bat
install_windows.bat -TorchIndexUrl https://repo.amd.com/rocm/whl/gfx120X-all/
```
Gunakan index PyTorch dari panduan resmi AMD bila deteksi otomatis gagal.

```bat
install_windows.bat -SkipTorch
```
Pertahankan PyTorch yang sudah Anda pasang manual di `.venv`.

### Linux (Ubuntu/Fedora/Arch, dll.)

1. Pastikan driver kernel `amdgpu` aktif, lalu masukkan user ke grup GPU. Setelah itu
   login ulang.
   ```bash
   sudo usermod -aG render,video $USER
   ```
2. Pasang Python dan FFmpeg. Paket `mesa-va-drivers` diperlukan untuk encoder VA-API.
   ```bash
   sudo apt install python3 python3-venv ffmpeg mesa-va-drivers
   ```
3. Jalankan installer:
   ```bash
   ./install_linux.sh
   ```
   Installer mencoba wheel PyTorch ROCm dari pytorch.org (rocm7.2 → 6.4), lalu index
   nightly AMD untuk arsitektur GPU Anda.

Variabel opsional untuk installer Linux:
- `PYTHON=python3.12`
- `TORCH_INDEX_URL=<url>`
- `ROCM_VERSIONS="7.2 7.1"`
- `SKIP_TORCH=1`
- `CPU_ONLY=1`

> Alamat index PyTorch untuk ROCm cukup sering berubah. Jika instalasi otomatis gagal,
> ikuti halaman resmi AMD *"Install PyTorch for ROCm"*. Setelah itu jalankan ulang
> installer dengan `-SkipTorch` (Windows) atau `SKIP_TORCH=1` (Linux).

## Pemakaian

**Windows**: seret (drag & drop) satu atau beberapa file video, atau satu folder, ke
**`restore.bat`**. Hasil disimpan di samping file asli dengan nama `nama.restored.mp4`.

**Baris perintah** (`./mosaic-restore` di Linux, `restore.bat` di Windows):

Restorasi satu video dengan kualitas maksimum (default):
```bash
mosaic-restore video.mp4
```

Pratinjau cepat 30 detik mulai menit ke-10. Sangat disarankan dicoba dulu:
```bash
mosaic-restore video.mp4 --start 10:00 --duration 30
```

Proses satu folder ke folder lain dengan preset lebih cepat (FP16):
```bash
mosaic-restore "D:\Video" -o "D:\Hasil" --preset balanced
```

Periksa deteksi saja, tanpa restorasi (area terdeteksi ditandai):
```bash
mosaic-restore video.mp4 --debug-overlay --start 5:00 --duration 20
```

Tampilkan GPU yang terdeteksi:
```bash
mosaic-restore --list-devices
```

### Preset

| Preset | Deteksi | Presisi | Klip | Kegunaan |
|---|---|---|---|---|
| `best` (default) | v4-accurate | FP32 | 180 frame | kualitas & akurasi maksimum |
| `balanced` | v4-accurate | FP16 | 150 frame | lebih cepat di GPU dengan FP16 cepat (RDNA2/3/4); beda kualitas sangat kecil |
| `fast` | v4-fast | FP16 | 90 frame | GPU kecil / pratinjau |

### Opsi penting

| Opsi | Keterangan |
|---|---|
| `--device auto\|rocm\|directml\|cpu` | pilih backend (default otomatis: ROCm → DirectML → CPU) |
| `--encoder auto\|hevc_amf\|av1_amf\|hevc_vaapi\|libx265\|...` | encoder output (default: hardware AMD bila tersedia) |
| `--cq 18` | kualitas output; makin kecil makin bagus/besar (16 = sangat tinggi) |
| `--det-conf 0.15` | ambang deteksi; turunkan (0.1) bila ada mosaic terlewat, naikkan (0.25) bila salah deteksi |
| `--max-gap 4` | isi celah deteksi hingga N frame |
| `--clip-length 180` | panjang klip temporal; lebih panjang lebih stabil, butuh VRAM lebih besar |
| `--restore-size 0` | ukuran input model otomatis hingga 512 untuk area besar (eksperimental; default 256 sesuai pelatihan) |
| `--hwdec` | decode video di GPU (berguna untuk sumber 4K/HEVC) |
| `--container mkv` | format output |
| `--hsa-override-gfx 10.3.0` | Linux: paksa ROCm untuk RX 6700/6600/6500 (gfx1031/1032/1034) |

Daftar lengkap opsi: `mosaic-restore --help`.

## Tips hasil terbaik

- Mulailah dengan pratinjau (`--start`/`--duration`) dan `--debug-overlay` untuk memastikan
  semua area mosaic terdeteksi.
- Preset `best` sudah memakai pengaturan paling akurat. Untuk kualitas file output yang
  lebih tinggi, pakai `--cq 16`.
- Jika mosaic sesekali "berkedip" muncul kembali, coba `--det-conf 0.1 --max-gap 6`.
- Jika grafik berpola kotak-kotak (logo, teks) ikut terdeteksi, naikkan `--det-conf 0.25`.

## Tips kecepatan

- Jalankan pertama di ROCm memang lebih lambat. MIOpen sedang menyetel (*tuning*) kernel
  konvolusi. Hasil tuning disimpan dan dipakai ulang pada jalankan berikutnya.
- Untuk video panjang, gunakan `--preset balanced`.
- Aktifkan `--hwdec` untuk sumber 4K/HEVC agar CPU tidak menjadi bottleneck.

## Pemecahan masalah

Langkah pertama untuk masalah apa pun: jalankan pemeriksaan sistem. Perintah ini
menampilkan GPU/PyTorch, lokasi FFmpeg yang dipakai, encoder AMD yang berfungsi, dan
status model.
```bash
restore.bat --check
```
(Linux: `./mosaic-restore --check`)

| Masalah | Solusi |
|---|---|
| FFmpeg tidak ditemukan, atau error `WinError 2/3` saat mulai | Pilih salah satu: (1) `winget install Gyan.FFmpeg`; (2) ekstrak FFmpeg *full build* dari gyan.dev ke folder program ini dengan nama `ffmpeg` (sehingga ada `ffmpeg\bin\ffmpeg.exe`); (3) tunjukkan lokasinya dengan `--ffmpeg "C:\lokasi\ffmpeg\bin"`. |
| "berjalan di CPU" | Jalankan `mosaic-restore --list-devices`. Perbarui driver, lalu jalankan ulang installer. Di Linux, cek grup `render`/`video`. |
| RX 6700/6600 di Linux tidak terpakai | Tambahkan `--hsa-override-gfx 10.3.0` |
| VRAM habis (out of memory) | Program otomatis memecah klip. Bisa juga pakai `--clip-length 90` atau `--preset fast`. |
| Encoder AMF/VA-API tidak tersedia | Perbarui driver/FFmpeg (Windows: build Gyan). Program otomatis memakai `libx265`. |
| DirectML gagal di Python 3.13 | `torch-directml` butuh Python 3.12. Pasang 3.12, lalu jalankan ulang installer. |
| Detail error | Set variabel `MOSAIC_RESTORE_DEBUG=1` untuk melihat traceback lengkap. |

## Cara kerja

```
Tahap 1 (analisis)
  decode (ffmpeg, YUV 4:2:0)
    -> deteksi YOLO11-seg per batch di GPU
    -> deteksi pergantian adegan
    -> pelacakan: gabung deteksi, isi celah, buang deteksi lemah/sesaat

Tahap 2 (restorasi)
  decode ulang
    -> crop jendela persegi yang dihaluskan secara temporal (+konteks)
    -> 256x256
    -> BasicVSR++ per klip (<=180 frame, tumpang-tindih 24, cross-fade)
    -> kembalikan ke ukuran asli
    -> blending bergradasi di domain YUV (alpha 0 = piksel asli)
    -> encode (AMF / VA-API / x265) + salin audio
```

Struktur proyek:

```
mosaic_restore/
  device.py            pemilihan backend AMD (ROCm/DirectML/CPU)
  config.py            pengaturan & preset
  cli.py               antarmuka baris perintah
  models/
    basicvsrpp.py      implementasi BasicVSR++ (inferensi)
    deform_conv.py     deformable conv portabel + jalur cepat torchvision
    detector.py        deteksi YOLO11-seg dengan pra/pasca-proses di GPU
    safe_load.py       pemuat checkpoint aman
    registry.py        unduhan model + SHA-256
  pipeline/
    tracker.py         pelacakan, isi celah, scene cut, jendela & klip
    restorer.py        crop & inferensi klip (penanganan VRAM habis)
    blender.py         alpha bergradasi & komposit YUV
    engine.py          orkestrasi dua tahap
  video/
    probe.py           metadata ffprobe
    reader.py          decode ffmpeg
    writer.py          encode ffmpeg
    encoders.py        AMF/VA-API/x265
    color.py           konversi warna
tests/                 uji unit & integrasi (video sintetis)
```

## Pengujian

Uji unit (cepat, tanpa model):
```bash
.venv/bin/python -m pytest -q tests/test_units.py
```

Uji integrasi end-to-end pada video sintetis (butuh model, ~2 menit di CPU):
```bash
MOSAIC_RESTORE_MODELS=./models .venv/bin/python -m pytest -q tests/
```

Uji integrasi mencakup:
- PSNR area mosaic terhadap sumber bersih naik >4 dB,
- area lain tetap >38 dB,
- jumlah frame dan audio terjaga,
- video 10-bit, full range, MKV, pratinjau segmen, dan video tanpa mosaic.

## Lisensi, kredit, dan penggunaan yang bertanggung jawab

- Kode program ini: **AGPL-3.0** (lihat `LICENSE`).
- Bobot model deteksi & restorasi: proyek open-source
  [Lada](https://codeberg.org/ladaapp/lada) (AGPL-3.0), diunduh dari
  [HuggingFace ladaapp/lada](https://huggingface.co/ladaapp/lada).
- Arsitektur: *BasicVSR++* (Chan dkk., CVPR 2022), *SPyNet* (Ranjan & Black, 2017),
  [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
- Perangkat lunak pendukung: FFmpeg, PyTorch, AMD ROCm.

Gunakan hanya untuk tontonan pribadi atas video yang Anda peroleh secara sah. Jangan
mendistribusikan hasilnya. Penyebaran versi tanpa sensor melanggar hak cipta dan
hukum di banyak negara, termasuk Jepang dan Indonesia. Mosaic pada wajah sengaja
tidak direstorasi untuk menjaga privasi.
