Param(
    [string]$OutputDir = "build/reports/roofline-sweep",
    [int]$GemmAnchorRuns = 8,
    [int]$GemmAnchorWarmup = 2,
    [switch]$GemmOnly
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
Set-Location $repoRoot

# Prepare output directory (archive existing if present)
if (Test-Path $OutputDir) {
    $timestamp = (Get-Date).ToString('yyyyMMddHHmmss')
    $arch = "${OutputDir}_$timestamp"
    Write-Host "Archiving existing output to $arch"
    Move-Item -Path $OutputDir -Destination $arch -Force
}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

$log = Join-Path $OutputDir "gradle_run.log"
Start-Transcript -Path $log -Force

$args = @(
    "-Djlc.roofline.outputDir=$OutputDir",
    "-Djlc.roofline.gemm_anchor_runs=$GemmAnchorRuns",
    "-Djlc.roofline.gemm_anchor_warmup=$GemmAnchorWarmup",
    "test",
    "--tests",
    "net.faulj.benchmark.roofline.PortableEfficiencyBenchmarkTest",
    "--no-daemon",
    "--console=plain"
)

if ($GemmOnly) {
    $args = @("-Djlc.roofline.gemm_only=true") + $args
}

Write-Host "Running: .\gradlew.bat $($args -join ' ')"
Write-Host "Logging to: $log"

& .\gradlew.bat @args
$exit = $LASTEXITCODE

Stop-Transcript

if ($exit -ne 0) {
    Write-Host "Gradle exited with code $exit" -ForegroundColor Red
    exit $exit
} else {
    Write-Host "Gradle completed successfully" -ForegroundColor Green
}
