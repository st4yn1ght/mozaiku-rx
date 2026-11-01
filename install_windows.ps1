# Installer Mosaic Restore AMD untuk Windows 10/11 + GPU AMD.
#
#   install_windows.bat                 -> otomatis (ROCm bila didukung, jika gagal DirectML)
#   install_windows.bat -DirectML       -> paksa DirectML (RX 5000/6000/Vega/iGPU lama)
#   install_windows.bat -SkipTorch      -> pertahankan PyTorch yang sudah ada di .venv
#   install_windows.bat -TorchIndexUrl <url>  -> pakai index PyTorch dari panduan resmi AMD
param(
    [switch]$DirectML,
    [switch]$SkipTorch,
    [switch]$Cpu,
    [string]$TorchIndexUrl = $env:TORCH_INDEX_URL
)
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

function Say($m)  { Write-Host "==> $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "[!] $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "[x] $m" -ForegroundColor Red; exit 1 }

Say "Mosaic Restore AMD - installer Windows"

# ------------------------------------------------------------- Python
$pyCmd = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($v in @("3.12", "3.11", "3.13", "3.10")) {
        & py "-$v" -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { $pyCmd = @("py", "-$v"); break }
    }
}
if (-not $pyCmd -and (Get-Command python -ErrorAction SilentlyContinue)) {
    & python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
    if ($LASTEXITCODE -eq 0) { $pyCmd = @("python") }
}
if (-not $pyCmd) {
    Die "Python 3.12 tidak ditemukan. Pasang dengan:  winget install Python.Python.3.12   lalu jalankan ulang."
}
$pyExe = $pyCmd[0]
$pyArgs = @()
if ($pyCmd.Length -gt 1) { $pyArgs = @($pyCmd[1..($pyCmd.Length - 1)]) }
$pyVer = & $pyExe @pyArgs -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
Say "Python $pyVer"

# ------------------------------------------------------------- GPU
$gpuNames = @(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "AMD|Radeon" } | ForEach-Object { $_.Name })
$gfx = $null
foreach ($n in $gpuNames) {
    switch -Regex ($n) {
        "RX 90[67]0"                         { $gfx = if ($n -match "9060") { "gfx1200" } else { "gfx1201" }; break }
        "RX 79\d\d|W7900|W7800"              { $gfx = "gfx1100"; break }
        "RX 7[78]\d\d|W7700"                 { $gfx = "gfx1101"; break }
        "RX 76\d\d|RX 7650|W7600|W7500"      { $gfx = "gfx1102"; break }
        "8060S|8050S|8040S|AI Max"           { $gfx = "gfx1151"; break }
        "890M|880M|860M"                     { $gfx = "gfx1150"; break }
        "840M|820M"                          { $gfx = "gfx1152"; break }
        "780M|760M|740M"                     { $gfx = "gfx1103"; break }
        "RX 69\d\d|RX 68\d\d|W6800"          { $gfx = "gfx1030"; break }
        "RX 67\d\d|RX 6750"                  { $gfx = "gfx1031"; break }
        "RX 66\d\d|RX 6650|W6600"            { $gfx = "gfx1032"; break }
    }
    if ($gfx) { break }
}
if ($gpuNames.Count -gt 0) { Say ("GPU: " + ($gpuNames -join ", ")) } else { Warn "Tidak ada GPU AMD terdeteksi." }
$family = $null; $familyStable = $null
if ($gfx) {
    Say "Arsitektur: $gfx"
    switch -Regex ($gfx) {
        "^gfx120"           { $family = "gfx120X-all"; $familyStable = "gfx120X-all" }
        "^gfx110[0-2]"      { $family = "gfx110X-all"; $familyStable = "gfx110X-dgpu" }
        "^gfx1103"          { $family = "gfx110X-all"; $familyStable = "gfx110X-all" }
        "^gfx115"           { $family = $gfx;          $familyStable = $gfx }
        "^gfx103"           { $family = "gfx103X-dgpu"; $familyStable = "gfx103X-dgpu" }
    }
}

# ------------------------------------------------------------- venv
Say "Membuat virtualenv .venv"
& $pyExe @pyArgs -m venv .venv
if ($LASTEXITCODE -ne 0) { Die "Gagal membuat venv" }
$vpy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip wheel | Out-Null

# Catatan: di dalam fungsi PowerShell, output perintah native ikut menjadi nilai
# kembalian. Semua output diarahkan ke Out-Host agar fungsi hanya mengembalikan bool.
function Test-Hip {
    & $vpy -c "import sys, torch, torchvision; hip = getattr(torch.version, 'hip', None); ok = bool(hip) and torch.cuda.is_available(); print(f'  torch {torch.__version__} | HIP {hip} | GPU: {torch.cuda.is_available()}'); sys.exit(0 if ok else 1)" | Out-Host
    return ($LASTEXITCODE -eq 0)
}
function Try-Pip([string[]]$pipArgs) {
    Say ("Mencoba: pip install " + ($pipArgs -join " "))
    & $vpy -m pip install --upgrade --force-reinstall @pipArgs | Out-Host
    if ($LASTEXITCODE -ne 0) { return $false }
    return (Test-Hip)
}
function Install-DirectML {
    Say "Memasang PyTorch DirectML (torch-directml)"
    if ([version]$pyVer -gt [version]"3.12") {
        Warn "torch-directml hanya tersedia untuk Python <= 3.12. Pasang Python 3.12: winget install Python.Python.3.12"
        return $false
    }
    & $vpy -m pip install --upgrade torch-directml | Out-Host
    if ($LASTEXITCODE -ne 0) { return $false }
    & $vpy -c "import sys, torch_directml as d; ok = d.is_available() and d.device_count() > 0; print('  DirectML:', d.device_name(0) if ok else 'tidak tersedia'); sys.exit(0 if ok else 1)" | Out-Host
    return ($LASTEXITCODE -eq 0)
}

# ------------------------------------------------------------- PyTorch
$torchOk = $false
if ($SkipTorch) {
    Say "SkipTorch: memakai PyTorch yang sudah ada"
    $torchOk = $true
} elseif ($Cpu) {
    & $vpy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    $torchOk = ($LASTEXITCODE -eq 0)
} else {
    if (-not $DirectML) {
        $attempts = @()
        if ($TorchIndexUrl) { $attempts += ,@("--pre", "torch", "torchvision", "--index-url", $TorchIndexUrl) }
        if ($familyStable)  { $attempts += ,@("torch", "torchvision", "--index-url", "https://repo.amd.com/rocm/whl/$familyStable/") }
        if ($gfx)           { $attempts += ,@("torch[device-$gfx]", "torchvision", "--index-url", "https://repo.amd.com/rocm/whl-multi-arch/") }
        if ($family)        { $attempts += ,@("--pre", "torch", "torchvision", "--index-url", "https://rocm.nightlies.amd.com/v2/$family/") }
        foreach ($a in $attempts) {
            if (Try-Pip $a) { $torchOk = $true; break }
        }
        if (-not $torchOk -and $attempts.Count -gt 0) {
            Warn "PyTorch ROCm untuk Windows gagal dipasang/berjalan (perlu driver AMD Adrenalin terbaru)."
        }
    }
    if (-not $torchOk) {
        & $vpy -m pip uninstall -y torch torchvision 2>$null | Out-Null
        $torchOk = Install-DirectML
    }
}
if (-not $torchOk) {
    Warn "GPU tidak bisa dipakai. Memasang PyTorch CPU (lambat) agar program tetap bisa jalan."
    & $vpy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}

# ------------------------------------------------------------- dependensi
Say "Memasang dependensi (ultralytics, numpy, opencv)"
& $vpy -c "import torch, torchvision; open('.constraints.txt','w').write(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\n')"
& $vpy -m pip install -r requirements.txt -c .constraints.txt
if ($LASTEXITCODE -ne 0) { Die "Gagal memasang dependensi" }
& $vpy -m pip install -e . --no-deps | Out-Null

# ------------------------------------------------------------- FFmpeg
# Memakai logika pencarian yang sama dengan program (PATH, folder winget, folder portabel).
Say "Memeriksa FFmpeg"
& $vpy -m mosaic_restore --check-ffmpeg
if ($LASTEXITCODE -ne 0) {
    Warn "FFmpeg dibutuhkan untuk membaca/menulis video (termasuk encoder AMD AMF)."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        $ans = Read-Host "Pasang FFmpeg sekarang via winget (Gyan.FFmpeg)? [Y/n]"
        if ($ans -eq "" -or $ans -match "^[Yy]") {
            winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
            & $vpy -m mosaic_restore --check-ffmpeg
        }
    } else {
        Warn "Unduh FFmpeg 'full build' dari gyan.dev, ekstrak ke folder program ini, beri nama folder 'ffmpeg'."
    }
}

# ------------------------------------------------------------- model & cek
Say "Mengunduh model (sekali saja, ~130 MB)"
& $vpy -m mosaic_restore --download-models
Say "Pemeriksaan sistem:"
& $vpy -m mosaic_restore --check
Say "Selesai! Seret (drag & drop) file video ke restore.bat, atau jalankan:"
Write-Host "    restore.bat ""D:\Video\film.mp4"""
