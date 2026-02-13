$Url='http://localhost:8080/api/benchmark/diagnostic?sizex=512&sizey=512&test=GEMM&iterations=1'
$Out='./build/benchmark_1000_results.csv'
if (-not (Test-Path './build')) { New-Item -ItemType Directory -Path './build' -Force | Out-Null }
if (Test-Path $Out) { Remove-Item $Out }
"idx,timestamp_ms,duration_ms,status" | Out-File -FilePath $Out -Encoding utf8
for ($i=1; $i -le 1000; $i++) {
  $t0 = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
  $sw = [Diagnostics.Stopwatch]::StartNew()
  try {
    $r = Invoke-WebRequest -Uri $Url -Method Get -UseBasicParsing -TimeoutSec 600
    $status = $r.StatusCode
  } catch {
    $status = 'ERR'
  }
  $sw.Stop()
  $dur = $sw.ElapsedMilliseconds
  "$i,$t0,$dur,$status" | Out-File -FilePath $Out -Append -Encoding utf8
  if ($i % 50 -eq 0) { Write-Host "Completed $i requests" }
}
Write-Host "Run complete"