<#
.SYNOPSIS
    One-command local setup for AI SOAR (Windows PowerShell 5.1+).

.DESCRIPTION
    Creates/repairs the virtual environment, installs pinned dependencies,
    makes sure .env exists, verifies that the package imports and that the
    trained models + demo artifacts are present, then prints the exact
    commands for the demo.

    It never relies on "activating" the venv (activation only edits PATH for
    that shell). Every step calls venv\Scripts\python.exe by full path, so the
    script works from any terminal, including a fresh one.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
    Full setup: venv + dependencies + checks.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -SkipInstall
    Fast dry run: only the verification checks (no downloads).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -Dev -Dashboard -RunApi
    Dev extras (pytest/ruff/mypy), rebuild the dashboard, then start the API.
#>

[CmdletBinding()]
param(
    [switch]$Dev,          # install pytest / ruff / mypy / jupyterlab too
    [switch]$SkipInstall,  # verify only, do not touch the venv
    [switch]$Fresh,        # delete and rebuild the venv from scratch
    [switch]$Dashboard,    # rebuild docs\demo\dashboard.html
    [switch]$Test,         # run pytest after setup
    [switch]$RunApi        # start the inference API in this window
)

$ErrorActionPreference = "Stop"

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

function Step([string]$msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Ok([string]$msg)   { Write-Host "    [ok]  $msg" -ForegroundColor Green }
function Warn([string]$msg) { Write-Host "    [!!]  $msg" -ForegroundColor Yellow }
function Bad([string]$msg)  { Write-Host "    [xx]  $msg" -ForegroundColor Red }

Write-Host ""
Write-Host "AI SOAR bootstrap" -ForegroundColor White
Write-Host "project root: $ProjectRoot" -ForegroundColor DarkGray

# ---------------------------------------------------------------- 1. python --
Step "Locating Python (need 3.10 - 3.12)"
$script:PyExe     = $null
$script:PyExeArgs = @()
$verText          = $null
$probe            = "import sys; print('.'.join(str(x) for x in sys.version_info[:2]))"

if (Get-Command py -ErrorAction SilentlyContinue) {
    $verText = & py -3.12 -c $probe 2>$null
    if ($LASTEXITCODE -eq 0 -and $verText) {
        $script:PyExe = "py"; $script:PyExeArgs = @("-3.12")
    }
}
if (-not $script:PyExe -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $verText = & python -c $probe 2>$null
    if ($LASTEXITCODE -eq 0 -and $verText) { $script:PyExe = "python"; $script:PyExeArgs = @() }
}
if (-not $script:PyExe) {
    Bad "no usable Python found on PATH."
    Write-Host "    Install Python 3.12 from https://www.python.org/downloads/ (tick 'Add to PATH')." -ForegroundColor Yellow
    exit 1
}

$parts = $verText.Split('.')
$major = [int]$parts[0]; $minor = [int]$parts[1]
if ($major -ne 3 -or $minor -lt 10 -or $minor -gt 12) {
    Bad "found Python $verText - this project needs 3.10, 3.11 or 3.12 (3.13+ has no LightGBM wheels)."
    exit 1
}
Ok "Python $verText via '$($script:PyExe) $($script:PyExeArgs -join ' ')'"

# ------------------------------------------------------------------ 2. venv --
$VenvDir = Join-Path $ProjectRoot "venv"
$VenvPy  = Join-Path $VenvDir "Scripts\python.exe"

if ($Fresh -and (Test-Path $VenvDir)) {
    Step "Removing old venv (-Fresh)"
    Remove-Item -Recurse -Force $VenvDir
    Ok "removed"
}

if (-not $SkipInstall) {
    if (-not (Test-Path $VenvPy)) {
        Step "Creating virtual environment"
        & $script:PyExe @script:PyExeArgs -m venv $VenvDir
        if (-not (Test-Path $VenvPy)) { Bad "venv creation failed"; exit 1 }
        Ok "venv\ created"
    } else {
        Step "Virtual environment already present"
        Ok "reusing venv\ (use -Fresh to rebuild)"
    }

    Step "Installing pinned dependencies (requirements.txt)"
    & $VenvPy -m pip install --upgrade pip --quiet
    & $VenvPy -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) { Bad "pip install -r requirements.txt failed"; exit 1 }
    Ok "runtime dependencies installed"

    Step "Installing the ai_soar package (editable)"
    if ($Dev) {
        & $VenvPy -m pip install -e ".[dev]"
    } else {
        & $VenvPy -m pip install -e .
    }
    if ($LASTEXITCODE -ne 0) { Bad "pip install -e . failed"; exit 1 }
    Ok "editable install done (keeps PROJECT_ROOT resolution correct)"
} elseif (-not (Test-Path $VenvPy)) {
    Bad "venv\Scripts\python.exe missing and -SkipInstall was given. Run without -SkipInstall first."
    exit 1
} else {
    Step "Skipping installation (-SkipInstall)"
    Ok "using existing venv\"
}

# ------------------------------------------------------------------- 3. .env --
Step "Checking .env"
$EnvFile = Join-Path $ProjectRoot ".env"
if (Test-Path $EnvFile) {
    Ok ".env present (git-ignored, never committed)"
} else {
    Copy-Item (Join-Path $ProjectRoot ".env.example") $EnvFile
    Warn ".env created from .env.example - edit it if you use an LLM provider (Ollama/OpenAI key)."
}

# --------------------------------------------------------------- 4. verify ---
Step "Verifying the package imports"
$verOut = & $VenvPy -c "import ai_soar; print(ai_soar.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) { Bad "import ai_soar failed: $verOut"; exit 1 }
Ok "import ai_soar -> $verOut"

Step "Verifying configuration"
$cfgOut = & $VenvPy -c "from ai_soar.config import get_settings; s=get_settings(); print(s.paths.models)" 2>&1
if ($LASTEXITCODE -ne 0) { Bad "config load failed: $cfgOut"; exit 1 }
Ok "settings load, models dir = $cfgOut"

Step "Verifying artifacts"
$need = @(
    "artifacts\models\binary_stage1.joblib",
    "artifacts\models\multiclass_stage2.joblib",
    "artifacts\models\model_metadata.json",
    "artifacts\incidents\incidents.jsonl",
    "knowledge_base\family_playbooks_and_policy.md"
)
$missing = 0
foreach ($rel in $need) {
    $p = Join-Path $ProjectRoot $rel
    if (Test-Path $p) {
        $kb = [math]::Round((Get-Item $p).Length / 1KB, 0)
        Ok "$rel  ($kb KB)"
    } else {
        Bad "$rel  MISSING"
        $missing++
    }
}
if ($missing -gt 0) {
    Warn "$missing artifact(s) missing - training/demo outputs were not committed or were deleted."
    Warn "Retrain with:  venv\Scripts\python.exe scripts\train_models.py"
}

# ------------------------------------------------------------ 5. dashboard ---
if ($Dashboard) {
    Step "Rebuilding the demo dashboard"
    & $VenvPy (Join-Path $ProjectRoot "scripts\build_dashboard.py")
    if ($LASTEXITCODE -ne 0) { Bad "build_dashboard.py failed"; exit 1 }
    Ok "docs\demo\dashboard.html"
}

# ----------------------------------------------------------------- 6. tests ---
if ($Test) {
    Step "Running pytest"
    & $VenvPy -m pytest
    if ($LASTEXITCODE -ne 0) { Warn "some tests failed" } else { Ok "tests passed" }
}

# ------------------------------------------------------------- 7. next steps --
Step "Setup complete - demo commands"
Write-Host @"

  1) API (Swagger UI at http://localhost:8000/docs)
     venv\Scripts\python.exe scripts\serve_api.py

  2) Response demo -> writes incidents with ground truth
     venv\Scripts\python.exe scripts\demo_response.py --rows 2000 --fresh --auto-approve

  3) Explanations (LLM on demand; templates for batches)
     venv\Scripts\python.exe scripts\explain_incidents.py --family DDoS --limit 1
     venv\Scripts\python.exe scripts\explain_incidents.py --skip-llm --limit 3

  4) Dashboard (single self-contained HTML file)
     venv\Scripts\python.exe scripts\build_dashboard.py
     start docs\demo\dashboard.html

"@

if ($RunApi) {
    Step "Starting the inference API (Ctrl+C to stop)"
    & $VenvPy (Join-Path $ProjectRoot "scripts\serve_api.py")
}