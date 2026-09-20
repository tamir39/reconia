# PowerShell Script for Automated Benchmark Pipeline
# Usage:
#   .\scripts\run_benchmark.ps1
#   .\scripts\run_benchmark.ps1 -Models "roberta" -Datasets "sroie", "cord" -Epochs 10 -Patience 5
#   .\scripts\run_benchmark.ps1 -Models "all" -Datasets "sroie", "cord", "docile"

param (
    [string[]]$Models = @("gnn"),
    [string[]]$Datasets = @("sroie", "cord", "docile"),
    [int]$Epochs = 10,
    [int]$Patience = 5,
    [int]$DocileTrainLimit = 500,
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Continue"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " 🚀 Khởi Động Quy Trình Thực Nghiệm Tự Động (Benchmark Pipeline)" -ForegroundColor Cyan
Write-Host " Models:   $($Models -join ', ')" -ForegroundColor Yellow
Write-Host " Datasets: $($Datasets -join ', ')" -ForegroundColor Yellow
Write-Host " Epochs:   $Epochs (Early stopping patience: $Patience)" -ForegroundColor Yellow
Write-Host " Device:   $Device" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan

$PythonExe = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$ArgsList = @(
    "-m", "app.training.run_benchmark",
    "--models"
) + $Models + @(
    "--datasets"
) + $Datasets + @(
    "--epochs", "$Epochs",
    "--patience", "$Patience",
    "--docile-limit-train", "$DocileTrainLimit",
    "--device", "$Device"
)

& $PythonExe $ArgsList

Write-Host "`n✅ Hoàn thành toàn bộ thực nghiệm! Kiểm tra kết quả trong thư mục reports/." -ForegroundColor Green
