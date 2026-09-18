$files = Get-ChildItem "build\reports\caqr_timers_gemm_mr*_nr*_k*.csv"
$results = @()
foreach ($f in $files) {
    $csv = Import-Csv $f.FullName
    $r = $csv | Where-Object { $_.tag -eq 'GEMM.total' }
    if ($r) {
        $m = [regex]::Match($f.Name,'mr(\d+)').Groups[1].Value
        $n = [regex]::Match($f.Name,'nr(\d+)').Groups[1].Value
        $k = [regex]::Match($f.Name,'k(\d+)').Groups[1].Value
        $results += [PSCustomObject]@{File=$f.Name; MR=[int]$m; NR=[int]$n; K=[int]$k; AvgNanos=[int]$r.avg_nanos}
    }
}
$results | Sort-Object AvgNanos | Select-Object -First 10 | Format-Table -AutoSize
