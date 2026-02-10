param(
    [ValidateSet('quick','standard','full')]
    [string]$Profile = 'standard',
    [string]$OutputDir = 'build/reports/gemm-analysis',
    [switch]$SkipPlots
)

Write-Host "=== GEMM Kernel Analysis ===" -ForegroundColor Cyan
Write-Host "Profile: $Profile"
Write-Host "Output:  $OutputDir"

if (-not (Test-Path ".\gradlew.bat")) {
    Write-Error "Run this script from repository root (gradlew.bat not found)."
    exit 1
}

$env:JLC_BENCHMARK_MODE = "true"
$env:JLC_GEMM_ANALYSIS_PROFILE = $Profile
$env:JLC_GEMM_ANALYSIS_OUTPUT_DIR = $OutputDir

Write-Host "Running benchmark harness..." -ForegroundColor Yellow
& .\gradlew.bat --% test --tests net.faulj.benchmark.roofline.GemmKernelAnalysisBenchmarkTest --console=plain --rerun-tasks
if ($LASTEXITCODE -ne 0) {
    Write-Error "Benchmark run failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Remove-Item Env:JLC_BENCHMARK_MODE -ErrorAction SilentlyContinue
Remove-Item Env:JLC_GEMM_ANALYSIS_PROFILE -ErrorAction SilentlyContinue
Remove-Item Env:JLC_GEMM_ANALYSIS_OUTPUT_DIR -ErrorAction SilentlyContinue

$runsCsv = Join-Path $OutputDir "gemm_analysis_runs.csv"
$summaryJson = Join-Path $OutputDir "gemm_analysis_summary.json"

if (-not (Test-Path $runsCsv) -or -not (Test-Path $summaryJson)) {
    Write-Error "Expected artifacts missing: $runsCsv or $summaryJson"
    exit 2
}

Write-Host "Benchmark artifacts generated." -ForegroundColor Green
Write-Host "  - $runsCsv"
Write-Host "  - $summaryJson"

if ($SkipPlots) {
    exit 0
}

$python = Get-Command python -ErrorAction SilentlyContinue
$py = Get-Command py -ErrorAction SilentlyContinue

if (-not $python -and -not $py) {
    Write-Warning "Python not found on PATH. Skipping plot generation."
    Write-Host "Run manually: python tools/gemm_analysis/plot_gemm_analysis.py --csv $runsCsv --summary $summaryJson --out-dir $OutputDir"
    exit 0
}

Write-Host "Generating plots..." -ForegroundColor Yellow
if ($python) {
    & python tools/gemm_analysis/plot_gemm_analysis.py --csv $runsCsv --summary $summaryJson --out-dir $OutputDir
} else {
    & py -3 tools/gemm_analysis/plot_gemm_analysis.py --csv $runsCsv --summary $summaryJson --out-dir $OutputDir
}
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Plot generation failed. Install dependencies: pandas matplotlib numpy"
    exit 0
}

Write-Host "Plots generated in $OutputDir" -ForegroundColor Green
