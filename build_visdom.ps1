# build_visdom.ps1
$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'
$PSNativeCommandUseErrorActionPreference = $false

$root  = $PSScriptRoot
$dist  = Join-Path $root 'dist\Visdom'
$build = Join-Path $root 'build'

Write-Host ">> Stopping any running Visdom.exe"
Get-Process -Name Visdom -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process python,pythonw -ErrorAction SilentlyContinue |
  Where-Object { $_.Path -and $_.Path -like "*\dist\Visdom\*" } |
  Stop-Process -Force -ErrorAction SilentlyContinue

function Remove-Tree-Retry([string]$path) {
  if (-not (Test-Path -LiteralPath $path)) { return }
  for ($i=1; $i -le 10; $i++) {
    try { Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop; return }
    catch { Start-Sleep -Milliseconds (200 * $i); if ($i -eq 10) { throw "Failed to remove '$path' (locked)" } }
  }
}

Write-Host ">> Cleaning build and dist\Visdom"
Remove-Tree-Retry $build
Remove-Tree-Retry $dist

Set-Location -LiteralPath $root

# Choose python from venv if present
$py = $null
if ($env:VIRTUAL_ENV) {
  $cand = Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe'
  if (Test-Path $cand) { $py = $cand }
}
if (-not $py) { $py = (Get-Command python).Source }

Write-Host ">> Versions"
& $py -c "import sys,PyInstaller,platform;print('Python:',sys.version.replace(`n,' '));print('PyInstaller:',getattr(PyInstaller,'__version__','?'));print('OS:',platform.platform())"

Write-Host ">> Building with PyInstaller"
& $py -m PyInstaller .\app.spec --noconfirm --clean

if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }

# Optional: ensure certifi bundled if you added it to spec
# Optional: verify critical outputs exist
$mustExist = @('Visdom.exe','frontend\templates','frontend\static','data','.env')
foreach ($rel in $mustExist) {
  $p = Join-Path $dist $rel
  if (-not (Test-Path $p)) { throw "Missing expected output: $rel" }
}

Write-Host ">> Freezing dependencies"
pip freeze | Out-File -Encoding utf8 (Join-Path $dist 'requirements.freeze.txt')

Write-Host ">> Launching app"
& "$dist\Visdom.exe"
