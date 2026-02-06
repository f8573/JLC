# Quick Performance Diagnostic Runner

Write-Host "=== Performance Diagnostic & Fix Verification ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "ISSUE FIXED:" -ForegroundColor Green
Write-Host "  Matrix.multiply() now routes to OptimizedBLAS3 instead of BlockedMultiply" -ForegroundColor White
Write-Host "  Expected result: 3-10x speedup in all decomposition algorithms" -ForegroundColor White
Write-Host ""

Write-Host "Running diagnostic to verify fix..." -ForegroundColor Yellow
Write-Host ""

# Run diagnostic
.\gradlew test --tests net.faulj.benchmark.DiagnosticBench --console=plain 2>&1 | Select-String -Pattern "(Testing n=|GEMM|LU|Analysis|✅|❌|⚠️)" | ForEach-Object { $_.Line }

Write-Host ""
Write-Host "==================" -ForegroundColor Cyan
Write-Host ""

$response = Read-Host "Run full comprehensive benchmark? (y/n)"
if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Host ""
    Write-Host "Running comprehensive benchmark (this will take 5-10 minutes)..." -ForegroundColor Yellow
    .\gradlew test --tests net.faulj.benchmark.ComprehensivePerfBenchmark.runBenchmark --console=plain
}

Write-Host ""
Write-Host "✅ Done!" -ForegroundColor Green
