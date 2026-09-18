# PowerShell script to run CAQR tuning sweep and collect logs
$root = Get-Location
$reportDir = Join-Path $root "build\reports"
$logDir = Join-Path $reportDir "caqr_tuning_logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$summaryFile = Join-Path $reportDir "caqr_tuning_summary.txt"

"CAQR tuning summary - $(Get-Date)" | Out-File -FilePath $summaryFile -Encoding utf8
"p,fj,status,caqr_bench_exists,caqr_bench_file" | Out-File -FilePath $summaryFile -Append

$p_values = @(8,16,32,64)
$fj_values = @(8,16,32)

foreach ($p in $p_values) {
    foreach ($fj in $fj_values) {
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
            $destRelative = $dest
        } else {
            $exists = "no"
            $destRelative = ""
        }

        "$p,$fj,OK,$exists,$destRelative" | Out-File -FilePath $summaryFile -Append
    }
}

"ALL DONE" | Out-File -FilePath $summaryFile -Append
Write-Host "Tuning complete. Summary at $summaryFile"
