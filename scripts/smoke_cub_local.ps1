param(
    [ValidateSet(2,4)][int]$BatchSize = 4,
    [string]$Device = 'cpu',
    [int]$MaxTrainBatches = 1,
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path $PSScriptRoot -Parent
if (-not $Python) { $Python = Join-Path $Repo '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python)) { throw 'Pass -Python with the selex environment interpreter.' }
Push-Location $Repo
try {
    & $Python -u scripts/smoke_cub.py --device $Device --batch-size $BatchSize --max-train-batches $MaxTrainBatches
    if ($LASTEXITCODE -ne 0) { throw "Smoke test failed: $LASTEXITCODE" }
} finally { Pop-Location }
exit 0
