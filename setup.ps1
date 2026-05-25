# VoiceClone Setup Script for Windows (Y9000P RTX 5060)
# Run this in PowerShell from the VoiceClone directory:
#   powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Stop"
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  VoiceClone Setup - Windows (CUDA 12.4)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

$PythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "3\.(10|11|12)") {
            $PythonCmd = $cmd
            Write-Host "[OK] Found: $ver" -ForegroundColor Green
            break
        }
    } catch {}
}

if (-not $PythonCmd) {
    Write-Host "[ERROR] Python 3.10-3.12 required. Install from https://python.org" -ForegroundColor Red
    exit 1
}

# Step 1: Create virtual environment
Write-Host "`n[1/5] Creating virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    & $PythonCmd -m venv venv
}
$Activate = "venv\Scripts\Activate.ps1"
. $Activate
Write-Host "[OK] Virtual environment ready." -ForegroundColor Green

# Step 2: Install PyTorch with CUDA 12.4
Write-Host "`n[2/5] Installing PyTorch (CUDA 12.4)..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
Write-Host "[OK] PyTorch installed." -ForegroundColor Green

# Step 3: Clone and install CosyVoice
$CosyVoiceDir = "..\CosyVoice"
if (-not (Test-Path $CosyVoiceDir)) {
    Write-Host "`n[3/5] Cloning CosyVoice..." -ForegroundColor Yellow
    git clone https://github.com/FunAudioLLM/CosyVoice.git $CosyVoiceDir
    Write-Host "[OK] CosyVoice cloned." -ForegroundColor Green
    Write-Host "[3/5] Installing CosyVoice dependencies..." -ForegroundColor Yellow
    Push-Location $CosyVoiceDir
    pip install -r requirements.txt
    Pop-Location
    Write-Host "[OK] CosyVoice dependencies installed." -ForegroundColor Green
} else {
    Write-Host "`n[3/5] CosyVoice already exists, skipping clone." -ForegroundColor Yellow
}

# Step 4: Install VoiceClone dependencies
Write-Host "`n[4/5] Installing VoiceClone dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
Write-Host "[OK] VoiceClone dependencies installed." -ForegroundColor Green

# Step 5: Verify CUDA
Write-Host "`n[5/5] Verifying CUDA..." -ForegroundColor Yellow
$cudaCheck = python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')" 2>&1
Write-Host $cudaCheck
if ($cudaCheck -match "True") {
    Write-Host "[OK] CUDA is ready!" -ForegroundColor Green
} else {
    Write-Host "[WARN] CUDA not detected. Check your GPU drivers." -ForegroundColor Magenta
}

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Edit config.yaml and set your DeepSeek API key"
Write-Host "  2. Add voice samples to the voices/ directory"
Write-Host "  3. Run: python app.py"
Write-Host ""
Write-Host "Activate the environment later with:"
Write-Host "  venv\Scripts\Activate.ps1"
Write-Host ""
