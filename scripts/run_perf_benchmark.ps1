# Comprehensive Performance Benchmark Runner

Write-Host "=== JLC Comprehensive Performance Benchmark ===" -ForegroundColor Cyan
Write-Host ""

# Check if gradlew exists
if (-not (Test-Path ".\gradlew")) {
    Write-Host "ERROR: gradlew not found. Please run from JLC root directory." -ForegroundColor Red
    exit 1
}

# Compile
Write-Host "Compiling..." -ForegroundColor Yellow
.\gradlew compileTestJava --console=plain 2>&1 | Out-Null

if ($LASTEXITCODE -ne 0) {
    Write-Host "Compilation failed! Run '.\gradlew compileTestJava' for details." -ForegroundColor Red
    exit 1
}

Write-Host "Compilation successful!" -ForegroundColor Green
Write-Host ""

# Run benchmark
Write-Host "Running comprehensive performance benchmark..." -ForegroundColor Yellow
Write-Host "This will test:" -ForegroundColor Cyan
Write-Host "  - CPU theoretical peak estimation" -ForegroundColor White
Write-Host "  - Optimized GEMM vs Blocked GEMM" -ForegroundColor White
Write-Host "  - All major decomposition algorithms" -ForegroundColor White
Write-Host "  - Performance vs theoretical hardware limits" -ForegroundColor White
Write-Host ""
Write-Host "Test sizes: 64, 128, 256, 512, 1024" -ForegroundColor White
Write-Host "This may take 5-15 minutes depending on your hardware..." -ForegroundColor Yellow
Write-Host ""

.\gradlew test --tests net.faulj.benchmark.ComprehensivePerfBenchmark --console=plain

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Benchmark completed successfully!" -ForegroundColor Green
    Write-Host ""

    if (Test-Path "comprehensive_performance_results.csv") {
        Write-Host "Results saved to: comprehensive_performance_results.csv" -ForegroundColor Cyan
        Write-Host ""

        # Check if Python is available for visualization
        $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCmd) {
            Write-Host "Python detected! You can generate visualizations with:" -ForegroundColor Yellow
            Write-Host "  python tools\visualize_performance.py comprehensive_performance_results.csv" -ForegroundColor White
            Write-Host ""

            $response = Read-Host "Generate visualizations now? (y/n)"
            if ($response -eq 'y' -or $response -eq 'Y') {
                Write-Host ""
                Write-Host "Installing required packages..." -ForegroundColor Yellow
                python -m pip install pandas matplotlib seaborn -q

                Write-Host "Generating visualizations..." -ForegroundColor Yellow
                python tools\visualize_performance.py comprehensive_performance_results.csv

                if ($LASTEXITCODE -eq 0) {
                    Write-Host ""
                    Write-Host "✅ Visualizations generated!" -ForegroundColor Green
                    Write-Host "Check the current directory for PNG files." -ForegroundColor Cyan
                }
            }
        } else {
            Write-Host "Install Python to generate visualizations:" -ForegroundColor Yellow
            Write-Host "  python tools\visualize_performance.py comprehensive_performance_results.csv" -ForegroundColor White
        }
    }
} else {
    Write-Host ""
    Write-Host "❌ Benchmark failed or was interrupted." -ForegroundColor Red
    Write-Host "Check output above for errors." -ForegroundColor Yellow
}
