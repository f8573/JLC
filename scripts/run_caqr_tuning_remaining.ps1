$root = Get-Location
$reportDir = Join-Path $root "build\reports"
$logDir = Join-Path $reportDir "caqr_tuning_logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$summaryFile = Join-Path $reportDir "caqr_tuning_summary.txt"

$combos = @(
    @{p=32; fj=8}, @{p=32; fj=16}, @{p=32; fj=32},
    @{p=64; fj=8}, @{p=64; fj=16}, @{p=64; fj=32}
)

foreach ($c in $combos) {
    $p = $c.p
    $fj = $c.fj
    $log = Join-Path $logDir "caqr_p${p}_fj${fj}.log"
    $args = @(
        "test",
        "--tests",
        "net.faulj.benchmark.CAQRMicrobenchTest",
        "-Dla.qr.strategy=CAQR",
        "-Dla.qr.caqr.p=$p",
        "-Djava.util.concurrent.ForkJoinPool.common.parallelism=$fj",
        "--no-daemon",
        "--stacktrace"
    )

    Write-Host "Running gradle for p=$p fj=$fj; log: $log"
    & .\gradlew.bat @args 2>&1 | Tee-Object -FilePath $log
    $rc = $LASTEXITCODE
    if ($rc -ne 0) {
        $errMsg = "FAIL: p=$p fj=$fj exit=$rc"
        $errMsg | Out-File -FilePath $summaryFile -Append
        "---- LOG: $log ----" | Out-File -FilePath $summaryFile -Append
        Get-Content $log | Out-File -FilePath $summaryFile -Append
        Write-Host $errMsg
        exit $rc
    }

    $src = Join-Path $reportDir "caqr_bench.csv"
    if (Test-Path $src) {
        $dest = Join-Path $reportDir ("caqr_bench_CAQR_p${p}_fj${fj}.csv")
        Copy-Item -Path $src -Destination $dest -Force
        $exists = "yes"
    } else {
        $exists = "no"
        $dest = ""
    }

    "$p,$fj,OK,$exists,$dest" | Out-File -FilePath $summaryFile -Append
}

Write-Host "Remaining combos completed (if no errors). Summary at $summaryFile"
