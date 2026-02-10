$csv = Import-Csv "build\reports\caqr_timers.csv"
$csv | Where-Object { $_.tag -eq 'GEMM.total' -or $_.tag -eq 'GEMM.parallel.total' } | Format-Table -AutoSize
