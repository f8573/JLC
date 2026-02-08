$OutFile = "content_opt.txt"
if(Test-Path $OutFile){ Remove-Item $OutFile -Force }
"=== assemble ===" | Out-File $OutFile -Encoding utf8
$asmOut = & .\gradlew.bat --% assemble 2>&1
$asmOut | Out-File -Append -Encoding utf8 $OutFile
$asmOut
$asmExit = $LASTEXITCODE
if($asmExit -ne 0){ "`nASSEMBLE FAILED with code $asmExit" | Out-File -Append -Encoding utf8 $OutFile; exit $asmExit }
"`n=== test ===" | Out-File -Append -Encoding utf8 $OutFile
$testOut = & .\gradlew.bat --% test --tests "net.faulj.benchmark.CAQRMicrobenchTest" -Dla.qr.strategy=CAQR -Djava.util.concurrent.ForkJoinPool.common.parallelism=8 --no-daemon --stacktrace 2>&1
$testOut | Out-File -Append -Encoding utf8 $OutFile
$testOut
$testExit = $LASTEXITCODE
if($testExit -ne 0){ "`nTESTS FAILED with code $testExit" | Out-File -Append -Encoding utf8 $OutFile; exit $testExit }
"`n=== CSV copy ===" | Out-File -Append -Encoding utf8 $OutFile
$src = "build\reports\caqr_bench.csv"
$dst = "build\reports\caqr_bench_CAQR_wy.csv"
if(Test-Path $src){ Copy-Item $src $dst -Force; "Copied $src to $dst" | Out-File -Append -Encoding utf8 $OutFile; Get-Content $dst | Out-File -Append -Encoding utf8 $OutFile } else { "Source CSV not found: $src" | Out-File -Append -Encoding utf8 $OutFile; exit 1 }
"Done" | Out-File -Append -Encoding utf8 $OutFile
exit 0
