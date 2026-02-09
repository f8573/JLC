# Fine GEMM sweep around best region
# MR = 1..4, NR = {4,8}, K-unroll = {6,8,10,12}
$mrs = 1..4
$nrs = @(4,8)
$kus = @(6,8,10,12)
$env:JAVA_TOOL_OPTIONS = "-Djava.util.concurrent.ForkJoinPool.common.parallelism=8"
foreach ($mr in $mrs) {
    foreach ($nr in $nrs) {
        foreach ($ku in $kus) {
            Write-Host "Running MR=$mr NR=$nr K=$ku"
            $args = @('runGemmTest','--no-daemon','--stacktrace',"-Dla.gemm.mr=$mr","-Dla.gemm.nr=$nr","-Dla.gemm.kunroll=$ku")
            & .\gradlew.bat @args
            Start-Sleep -Milliseconds 200
            $src = "build\reports\caqr_timers_gemm.csv"
            $dst = "build\reports\caqr_timers_gemm_mr${mr}_nr${nr}_k${ku}_fine.csv"
            if (Test-Path $src) {
                Copy-Item -Path $src -Destination $dst -Force
            } else {
                Write-Warning "Missing $src after run"
            }
        }
    }
}
Write-Host "Fine sweep complete. Files under build/reports (suffix _fine.csv)"