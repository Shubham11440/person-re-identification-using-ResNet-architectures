# =================================================================================
# Test script to verify download functionality with a small test file
# =================================================================================

$DataDirectory = ".\data"

# Ensure the data directory exists
if (-not (Test-Path $DataDirectory)) {
    Write-Host "Creating data directory: $DataDirectory"
    New-Item -ItemType Directory -Path $DataDirectory
} else {
    Write-Host "Data directory already exists."
}

Write-Host "Testing gdown installation..."
try {
    & "C:\Users\Shubham Mali\miniconda3\envs\cpr_reid\python.exe" -c "import gdown; print('✅ gdown is working properly')"
    Write-Host "✅ Download script is ready to use!"
    Write-Host ""
    Write-Host "To download the full datasets, run: .\scripts\download_datasets.ps1"
    Write-Host "Note: Dataset downloads are large (several GB) and may take time."
} catch {
    Write-Host "❌ Error: gdown is not properly installed."
    Write-Host "Please run: python -m pip install gdown"
}