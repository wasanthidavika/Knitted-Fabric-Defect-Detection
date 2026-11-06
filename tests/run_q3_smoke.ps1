
$ErrorActionPreference = "Stop"

# Run from the main project folder
$ProjectRoot = Get-Location
$env:PYTHONPATH = "$ProjectRoot\src;$ProjectRoot\src\q3"

function Run-Python {
    param([string[]]$Arguments)

    Write-Host "`nRunning: python $($Arguments -join ' ')" -ForegroundColor Cyan
    & python @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Starting Q3 synthetic smoke test..." -ForegroundColor Green

# Use the synthetic test configuration
$env:ISL_CONFIG = "configs/config_smoke.yaml"

# Remove only previous synthetic test outputs
Remove-Item "smoke_out" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "smoke_data" -Recurse -Force -ErrorAction SilentlyContinue

# Step 1: Check shared metrics
Run-Python @("tests/unit_metrics.py")

# Step 2: Generate synthetic dataset
Run-Python @("tests/make_synthetic_q3.py")

# Step 3: Audit synthetic dataset
New-Item -ItemType Directory -Force "smoke_out/audit" | Out-Null
Run-Python @("src/audit_isl_knit.py", "--root", "smoke_data/Full", "--out", "smoke_out/audit")

# Step 4: Prepare data
Run-Python @("src/prepare_data.py", "--check")

# Step 5: Verify split
Run-Python @("src/verify_split.py")

Write-Host "`nQ3 smoke test Phase 1 PASSED." -ForegroundColor Green
