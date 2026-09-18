# Dense GEMM sweep: MR=1..8, NR in {4,8}, K-unroll in {2,4,6,8}
# Writes per-config gemm timer CSVs to build/reports
$mrs = 1..8
$nrs = @(4,8)
$kus = @(2,4,6,8)
$env:JAVA_TOOL_OPTIONS = "-Djava.util.concurrent.ForkJoinPool.common.parallelism=8"
foreach ($mr in $mrs) {
    foreach ($nr in $nrs) {
        foreach ($ku in $kus) {
            Write-Host "Running MR=$mr NR=$nr K=$ku"
            $args = @('runGemmTest','--no-daemon','--stacktrace',"-Dla.gemm.mr=$mr","-Dla.gemm.nr=$nr","-Dla.gemm.kunroll=$ku")
            & .\gradlew.bat @args
            Start-Sleep -Milliseconds 200
            $src = "build\reports\caqr_timers_gemm.csv"
            $dst = "build\reports\caqr_timers_gemm_mr${mr}_nr${nr}_k${ku}.csv"
            if (Test-Path $src) {
                Copy-Item -Path $src -Destination $dst -Force
            } else {
                Write-Warning "Missing $src after run"
            }
        }
    }
}
Write-Host "Dense sweep complete. Files under build/reports/"