#!/usr/bin/env bash
# Installer Mosaic Restore AMD untuk Linux + GPU AMD (ROCm).
#
# Variabel opsional:
#   PYTHON=python3.12          interpreter yang dipakai
#   TORCH_INDEX_URL=<url>      paksa index PyTorch tertentu (mis. dari halaman resmi AMD/pytorch.org)
#   ROCM_VERSIONS="7.2 7.1"    urutan index pytorch.org yang dicoba
#   SKIP_TORCH=1               jangan sentuh PyTorch yang sudah terpasang di .venv
#   CPU_ONLY=1                 pasang PyTorch CPU (tanpa GPU)
set -uo pipefail
cd "$(dirname "$0")"

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*"; exit 1; }

PY=${PYTHON:-python3}
command -v "$PY" >/dev/null 2>&1 || die "Python 3 tidak ditemukan (pasang python3 >= 3.10, disarankan 3.12)."
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die "Butuh Python >= 3.10."
"$PY" -c 'import venv' 2>/dev/null || die "Modul venv tidak ada. Ubuntu/Debian: sudo apt install python3-venv"

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  warn "FFmpeg belum terpasang. Ubuntu/Debian: sudo apt install ffmpeg   Fedora: sudo dnf install ffmpeg   Arch: sudo pacman -S ffmpeg"
fi

# ---------------------------------------------------------------- deteksi GPU
GFX=""
if command -v rocminfo >/dev/null 2>&1; then
  GFX=$(rocminfo 2>/dev/null | grep -o -m1 'gfx[0-9a-f]\+' || true)
fi
if [ -z "$GFX" ]; then
  for f in /sys/class/kfd/kfd/topology/nodes/*/properties; do
    v=$(awk '/gfx_target_version/ {print $2}' "$f" 2>/dev/null)
    if [ -n "${v:-}" ] && [ "$v" != "0" ]; then
      GFX=$(printf 'gfx%d%d%x' $((v / 10000)) $(((v / 100) % 100)) $((v % 100)))
      break
    fi
  done
fi
FAMILY=""
HSA_HINT=""
case "$GFX" in
  gfx1200|gfx1201) FAMILY="gfx120X-all" ;;
  gfx1100|gfx1101|gfx1102) FAMILY="gfx110X-all" ;;
  gfx1103) FAMILY="gfx110X-all"; HSA_HINT="11.0.0" ;;
  gfx1150|gfx1151|gfx1152) FAMILY="$GFX" ;;
  gfx1030) FAMILY="gfx103X-dgpu" ;;
  gfx1031|gfx1032|gfx1034|gfx1035|gfx1036) FAMILY="gfx103X-dgpu"; HSA_HINT="10.3.0" ;;
  gfx90a) FAMILY="gfx90a" ;;
  gfx942) FAMILY="gfx94X-dcgpu" ;;
  gfx950) FAMILY="gfx950-dcgpu" ;;
esac
if [ -n "$GFX" ]; then say "GPU AMD terdeteksi: $GFX (keluarga $FAMILY)"; else warn "Arsitektur GPU AMD tidak terdeteksi (driver amdgpu/KFD belum aktif?)."; fi
if ! groups | grep -q -E '\b(render|video)\b'; then
  warn "User belum di grup render/video. Jalankan: sudo usermod -aG render,video \$USER  lalu login ulang."
fi

# ---------------------------------------------------------------- venv
say "Membuat virtualenv .venv"
"$PY" -m venv .venv || die "Gagal membuat venv"
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip wheel >/dev/null || die "Gagal upgrade pip"

torch_hip_ok() {
  python - <<'EOF'
import sys
try:
    import torch, torchvision  # noqa: F401
except Exception as e:
    print("  import torch gagal:", e); sys.exit(1)
hip = getattr(torch.version, "hip", None)
print(f"  torch {torch.__version__} | HIP {hip} | GPU tersedia: {torch.cuda.is_available()}")
sys.exit(0 if hip else 1)
EOF
}

try_pip() {
  say "Mencoba: pip install $*"
  python -m pip install --upgrade --force-reinstall "$@" && torch_hip_ok
}

# ---------------------------------------------------------------- PyTorch
if [ "${SKIP_TORCH:-0}" = "1" ]; then
  say "SKIP_TORCH=1 -> memakai PyTorch yang sudah ada"
elif [ "${CPU_ONLY:-0}" = "1" ]; then
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || die "Gagal memasang PyTorch CPU"
else
  ok=0
  if [ -n "${TORCH_INDEX_URL:-}" ]; then
    try_pip --pre torch torchvision --index-url "$TORCH_INDEX_URL" && ok=1
  fi
  if [ $ok = 0 ]; then
    for v in ${ROCM_VERSIONS:-7.2 7.1 7.0 6.4}; do
      try_pip torch torchvision --index-url "https://download.pytorch.org/whl/rocm$v" && { ok=1; break; }
    done
  fi
  if [ $ok = 0 ] && [ -n "$FAMILY" ]; then
    for fam in "$FAMILY" "${FAMILY/-all/-dgpu}"; do
      try_pip --pre torch torchvision --index-url "https://rocm.nightlies.amd.com/v2/$fam/" && { ok=1; break; }
    done
  fi
  if [ $ok = 0 ]; then
    warn "PyTorch ROCm otomatis gagal dipasang."
    warn "Ikuti panduan resmi AMD (Install PyTorch for ROCm) lalu jalankan ulang: SKIP_TORCH=1 ./install_linux.sh"
    warn "atau set TORCH_INDEX_URL=<url index dari panduan resmi> ./install_linux.sh"
    exit 1
  fi
fi

# ---------------------------------------------------------------- dependensi lain
say "Memasang dependensi (ultralytics, numpy, opencv)"
python - > .constraints.txt <<'EOF'
import torch, torchvision
print(f"torch=={torch.__version__}\ntorchvision=={torchvision.__version__}")
EOF
python -m pip install -r requirements.txt -c .constraints.txt || die "Gagal memasang dependensi"
python -m pip install -e . --no-deps >/dev/null || die "Gagal memasang paket mosaic_restore"

# ---------------------------------------------------------------- model & cek
say "Mengunduh model (sekali saja, ~130 MB)"
python -m mosaic_restore --download-models || warn "Unduh model gagal; akan dicoba lagi saat pemakaian pertama."

cat > mosaic-restore <<'EOF'
#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/.venv/bin/python" -m mosaic_restore "$@"
EOF
chmod +x mosaic-restore

say "Perangkat yang terdeteksi:"
python -m mosaic_restore --list-devices
if [ -n "$HSA_HINT" ]; then
  warn "GPU $GFX tidak resmi didukung ROCm. Jika GPU tidak terpakai, jalankan dengan: --hsa-override-gfx $HSA_HINT"
fi
say "Selesai. Contoh pemakaian:  ./mosaic-restore /path/video.mp4"
