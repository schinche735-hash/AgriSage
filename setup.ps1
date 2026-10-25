$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$python = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($version in @('-3.12', '-3.11')) {
        & py $version -c "import sys; print(sys.version)" 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { $python = @('py', $version); break }
    }
}
if (-not $python -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $version = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($version -in @('3.11', '3.12')) { $python = @('python') }
}
if (-not $python) {
    Write-Host 'Python 3.11 or 3.12 is required. Install Python from https://www.python.org/downloads/ and enable the PATH option, then run this script again.' -ForegroundColor Red
    exit 1
}

if (-not (Test-Path '.venv\Scripts\python.exe')) {
    Write-Host 'Creating the project Python environment...'
    if ($python.Count -eq 2) { & $python[0] $python[1] -m venv .venv }
    else { & $python[0] -m venv .venv }
}
if (-not (Test-Path '.venv\.agrisage-dependencies-installed')) {
    Write-Host 'Installing required libraries (first run only; internet required)...'
    & .\.venv\Scripts\python.exe -m pip install --disable-pip-version-check --upgrade pip
    & .\.venv\Scripts\python.exe -m pip install --disable-pip-version-check -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check internet access and Python version, then run setup.ps1 again.' }
    New-Item '.venv\.agrisage-dependencies-installed' -ItemType File -Force | Out-Null
}
if (-not (Test-Path '.env')) {
    Copy-Item '.env.example' '.env'
    Write-Host 'Created .env. Gemini is optional; add your own key there if available.'
}
$env:TF_CPP_MIN_LOG_LEVEL = '2'
$env:OMP_NUM_THREADS = '1'
$env:TF_NUM_INTEROP_THREADS = '1'
$env:TF_NUM_INTRAOP_THREADS = '1'
Get-Content '.env' | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$' -and -not $matches[1].StartsWith('#')) {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
Write-Host 'Starting AgriSage. Open http://127.0.0.1:5000 in your browser. Press Ctrl+C to stop.' -ForegroundColor Green
& .\.venv\Scripts\python.exe app.py
