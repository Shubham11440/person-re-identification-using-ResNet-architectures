# =================================================================================
# Script to download and extract Person Re-ID benchmark datasets.
# Must be run from the project's root directory.
# Prerequisites: gdown (pip install gdown)
# =================================================================================

# --- Configuration ---
$DataDirectory = ".\data"

# Updated URLs and file IDs for better reliability
$Market1501_GdriveID = "1CaWEZR5X_YKhIV3hDm6LS6h9u0fAZDKC"  # Alternative source
$Market1501_ZipFile = "$DataDirectory\Market-1501-v15.09.15.zip"

$DukeMTMC_GdriveID = "1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O"  # Updated ID
$DukeMTMC_ZipFile = "$DataDirectory\DukeMTMC-reID.zip"

$CUHK03_GdriveID = "1Z8pKlHqCwcd7R0FpiJpPKRFpAZlTmhf7"  # Updated ID
$CUHK03_ZipFile = "$DataDirectory\cuhk03_release.zip"

# --- Functions ---
function Get-DatasetWithGdown {
    param($FileID, $OutputFile, $DatasetName)
    
    Write-Host "Downloading $DatasetName (this may take a while)..."
    try {
        & "C:\Users\Shubham Mali\miniconda3\envs\cpr_reid\python.exe" -m gdown $FileID -O $OutputFile --quiet
        if (Test-Path $OutputFile) {
            Write-Host "$DatasetName downloaded successfully."
            return $true
        } else {
            Write-Host "Failed to download $DatasetName."
            return $false
        }
    } catch {
        Write-Host "Error downloading $DatasetName`: $($_.Exception.Message)"
        return $false
    }
}

function Expand-DatasetArchive {
    param($ZipFile, $DataDirectory, $DatasetName)
    
    Write-Host "Extracting $DatasetName..."
    try {
        Expand-Archive -Path $ZipFile -DestinationPath $DataDirectory -Force
        Remove-Item $ZipFile
        Write-Host "$DatasetName extracted successfully."
        return $true
    } catch {
        Write-Host "Error extracting $DatasetName`: $($_.Exception.Message)"
        return $false
    }
}

# --- Script Execution ---

Write-Host "=== Person Re-ID Dataset Downloader ===" -ForegroundColor Green
Write-Host "This script will download Market-1501, DukeMTMC-reID, and CUHK03 datasets."
Write-Host ""

# Ensure the data directory exists
if (-not (Test-Path $DataDirectory)) {
    Write-Host "Creating data directory: $DataDirectory"
    New-Item -ItemType Directory -Path $DataDirectory
} else {
    Write-Host "Data directory already exists."
}

# --- Download and Extract Market-1501 ---
if (-not (Test-Path "$DataDirectory\Market-1501-v15.09.15")) {
    if (Get-DatasetWithGdown $Market1501_GdriveID $Market1501_ZipFile "Market-1501") {
        Expand-DatasetArchive $Market1501_ZipFile $DataDirectory "Market-1501"
    }
} else {
    Write-Host "Market-1501 already exists. Skipping."
}

# --- Download and Extract DukeMTMC-reID ---
if (-not (Test-Path "$DataDirectory\DukeMTMC-reID")) {
    if (Get-DatasetWithGdown $DukeMTMC_GdriveID $DukeMTMC_ZipFile "DukeMTMC-reID") {
        Expand-DatasetArchive $DukeMTMC_ZipFile $DataDirectory "DukeMTMC-reID"
    }
} else {
    Write-Host "DukeMTMC-reID already exists. Skipping."
}

# --- Download and Extract CUHK03 ---
if (-not (Test-Path "$DataDirectory\cuhk03")) {
    if (Get-DatasetWithGdown $CUHK03_GdriveID $CUHK03_ZipFile "CUHK03") {
        Expand-DatasetArchive $CUHK03_ZipFile $DataDirectory "CUHK03"
    }
} else {
    Write-Host "CUHK03 already exists. Skipping."
}

Write-Host ""
Write-Host "Dataset acquisition complete! 🎉" -ForegroundColor Green

Write-Host "Dataset acquisition complete! 🎉"