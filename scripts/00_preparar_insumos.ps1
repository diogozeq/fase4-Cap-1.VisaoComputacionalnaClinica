$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "CardioIA Fase 4 - preparo de insumos"
Write-Host "Root: $Root"

$dirs = @(
  "dados/raw/nih_metadata",
  "dados/raw/sample",
  "dados/processed",
  "notebooks",
  "modelos",
  "resultados",
  "docs/prints",
  "app/backend/static",
  "app/backend/templates",
  "app/mobile",
  "scripts"
)

foreach ($d in $dirs) {
  New-Item -ItemType Directory -Force -Path $d | Out-Null
}

$kaggleJson = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
if (-not (Test-Path $kaggleJson)) {
  Write-Host ""
  Write-Host "BLOQUEADO: kaggle.json nao encontrado em $kaggleJson" -ForegroundColor Yellow
  Write-Host "Faca: Kaggle -> Settings -> API -> Create New Token, salve neste caminho."
  Write-Host "Tambem aceite termos: https://www.kaggle.com/datasets/nih-chest-xrays/data"
}

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
  $python = Get-Command py -ErrorAction SilentlyContinue
}

if (-not $python) {
  Write-Host ""
  Write-Host "BLOQUEADO: Python nao encontrado no PATH." -ForegroundColor Yellow
  Write-Host "Opcao sem instalar local: rode notebooks/00_kaggle_bootstrap.py no Kaggle Notebook."
  exit 2
}

$pyCmd = $python.Source
function Invoke-ProjectPython {
  param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $PythonArgs
  )

  if ((Split-Path -Leaf $pyCmd) -ieq "py.exe") {
    & $pyCmd -3 @PythonArgs
  } else {
    & $pyCmd @PythonArgs
  }
}

Write-Host "Python: $pyCmd"

Invoke-ProjectPython -m pip install --upgrade pip kaggle kagglehub pandas scikit-learn tabulate
if ($LASTEXITCODE -ne 0) {
  Write-Host "Falha ao instalar pacotes Python. Tente no Kaggle Notebook." -ForegroundColor Yellow
  exit $LASTEXITCODE
}

if (Test-Path $kaggleJson) {
  Invoke-ProjectPython scripts/download_nih_metadata.py
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Download de metadados falhou. Verifique token/termos Kaggle." -ForegroundColor Yellow
    exit $LASTEXITCODE
  }

  Invoke-ProjectPython scripts/01_eda_nih_subset.py
  if ($LASTEXITCODE -ne 0) {
    Write-Host "EDA falhou. Verifique se Data_Entry_2017.csv foi baixado." -ForegroundColor Yellow
    exit $LASTEXITCODE
  }
}

Write-Host ""
Write-Host "Preparo concluido." -ForegroundColor Green
