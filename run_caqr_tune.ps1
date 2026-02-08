$ps = @()
$pValues = @(8,16,32,64)
$fjValues = @(8,16,32)
foreach ($p in $pValues) {
  foreach ($fj in $fjValues) {
    $exe = Join-Path (Get-Location) 'gradlew.bat'
    $args = @(
      'test',
      '--tests', 'net.faulj.benchmark.CAQRMicrobenchTest',
      "-Dla.qr.strategy=CAQR",
      "-Dla.qr.caqr.p=$p",
      "-Djava.util.concurrent.ForkJoinPool.common.parallelism=$fj",
      '--no-daemon',
      '--stacktrace'
    )
    $cmdLine = "$exe $($args -join ' ')"
    $ps += "RUN: $cmdLine"
    try {
      $out = & $exe @args 2>&1
    } catch {
      $out = $_.Exception.Message
    }
    $ps += $out
    if (Test-Path build\reports\caqr_bench.csv) {
      Copy-Item build\reports\caqr_bench.csv build\reports\caqr_bench_CAQR_p${p}_fj${fj}.csv -Force
      $csv = Get-Content build\reports\caqr_bench_CAQR_p${p}_fj${fj}.csv
      $ps += "COPIED: build\\reports\\caqr_bench_CAQR_p${p}_fj${fj}.csv"
      $ps += $csv
    } else {
      $ps += "NO CSV produced for p=$p fj=$fj"
      Set-Content -Encoding utf8 build\reports\caqr_tuning_summary.txt ($ps -join "`n")
      exit 1
    }
  }
}
Set-Content -Encoding utf8 build\reports\caqr_tuning_summary.txt ($ps -join "`n")
Write-Output "DONE"
