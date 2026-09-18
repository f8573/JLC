# Refinement sweep around initial GEMM tuning and run CAQR with tuned GEMM params
$rooflineDir = "build/reports/roofline"
$reportsDir = "build/reports"
$logPath = Join-Path $rooflineDir "portable_efficiency_gemm_tuning_log.txt"
$analysisCsv = Join-Path $rooflineDir "portable_efficiency_gemm_tuning_analysis.csv"
$analysisTxt = Join-Path $rooflineDir "portable_efficiency_gemm_tuning_analysis.txt"
$caqrLog = Join-Path $reportsDir "caqr_gemm_tuned_log.txt"
$caqrCsv = Join-Path $reportsDir "caqr_bench_CAQR_gemm_tuned.csv"
$caqrJfr = Join-Path $reportsDir "caqr_gemm_tuned.jfr"
$caqrReport = Join-Path $reportsDir "caqr_gemm_tuning_report.txt"

New-Item -ItemType Directory -Force -Path $rooflineDir | Out-Null
New-Item -ItemType Directory -Force -Path $reportsDir | Out-Null

Start-Transcript -Path $logPath -Force

function FailAndExit($msg, $code=1) {
    Write-Error $msg
    Stop-Transcript
    exit $code
}

Write-Output "[1/..] Reading initial GEMM tuning summary"
$summaryPath = Join-Path $rooflineDir "portable_efficiency_gemm_tuning_summary.csv"
if (-not (Test-Path $summaryPath)) { FailAndExit "Missing initial summary: $summaryPath" 2 }

try { $summary = Import-Csv -Path $summaryPath } catch { FailAndExit "Failed to read $summaryPath: $_" 3 }
if ($summary.Count -eq 0) { FailAndExit "Empty summary $summaryPath" 4 }

# Pick best by measured_gflops (numeric). Tie-breaker: smaller mr then nr.
$best = $summary | ForEach-Object { $_ | Add-Member -NotePropertyName measured_num -NotePropertyValue ([double]($_.measured_gflops)) -PassThru } |
    Sort-Object -Property @{Expression={'[double]$($_.measured_gflops)'};Descending=$true}, @{Expression={'[int]$($_.mr)'};Descending=$false}, @{Expression={'[int]$($_.nr)'};Descending=$false} | Select-Object -First 1

[int]$initBestMr = [int]$best.mr
[int]$initBestNr = [int]$best.nr
Write-Output "Initial best MR=$initBestMr NR=$initBestNr measured_gflops=$($best.measured_gflops)"

# Build refinement grid: offsets -2,0,2 around best (keeps search small)
$deltas = @(-2,0,2)
$mrs = ($deltas | ForEach-Object { $v = $initBestMr + $_; if ($v -lt 2) { $null } else { $v } }) | Where-Object { $_ -ne $null } | Sort-Object -Unique
$nrs = ($deltas | ForEach-Object { $v = $initBestNr + $_; if ($v -lt 2) { $null } else { $v } }) | Where-Object { $_ -ne $null } | Sort-Object -Unique

Write-Output "Refinement MR candidates: $($mrs -join ',')"
Write-Output "Refinement NR candidates: $($nrs -join ',')"

# Initialize analysis CSV
"mr,nr,n,measured_gflops,portable_efficiency_score,roof_gflops,src_csv" | Out-File -FilePath $analysisCsv -Encoding utf8

Write-Output "[2/..] Ensuring project assembled"
& .\gradlew.bat --% assemble --no-daemon --stacktrace 2>&1 | Tee-Object -FilePath (Join-Path $rooflineDir "assemble_refine_output.txt")
if ($LASTEXITCODE -ne 0) { FailAndExit "Gradle assemble failed with exit code $LASTEXITCODE" $LASTEXITCODE }

foreach ($mr in $mrs) {
    foreach ($nr in $nrs) {
        $dst = Join-Path $rooflineDir "portable_efficiency_results_refine_${mr}_${nr}.csv"
        Write-Output "--- Running refine MR=$mr NR=$nr ---"
        $jfr = Join-Path $rooflineDir "gemm_refine_mr${mr}_nr${nr}.jfr"
        $outFile = Join-Path $rooflineDir "gradle_test_refine_mr${mr}_nr${nr}.txt"
        $cmd = ".\gradlew.bat --% test --tests 'net.faulj.benchmark.roofline.PortableEfficiencyBenchmarkTest' -Dla.gemm.mr=$mr -Dla.gemm.nr=$nr -Dorg.gradle.jvmargs='-XX:StartFlightRecording=filename=$jfr,duration=20s,settings=profile' --no-daemon --stacktrace"
        Write-Output $cmd
        $out = & .\gradlew.bat --% test --tests "net.faulj.benchmark.roofline.PortableEfficiencyBenchmarkTest" -Dla.gemm.mr=$mr -Dla.gemm.nr=$nr -Dorg.gradle.jvmargs="-XX:StartFlightRecording=filename=$jfr,duration=20s,settings=profile" --no-daemon --stacktrace 2>&1 | Tee-Object -FilePath $outFile
        if ($LASTEXITCODE -ne 0) { FailAndExit "Gradle test failed for MR=$mr NR=$nr with exit code $LASTEXITCODE" $LASTEXITCODE }

        $src = Join-Path $rooflineDir "portable_efficiency_results.csv"
        if (Test-Path $src) {
            Copy-Item -Path $src -Destination $dst -Force
            Write-Output "Copied $src -> $dst"
        } else {
            Write-Warning "Expected results CSV not found at $src"
            FailAndExit "Missing portable_efficiency_results.csv after run MR=$mr NR=$nr" 2
        }

        # Extract GEMM row for n=2048 or largest n
        try { $csv = Import-Csv -Path $dst } catch { FailAndExit "CSV import failed for $dst: $_" 3 }
        $cols = ($csv | Get-Member -MemberType NoteProperty | Select-Object -ExpandProperty Name)
        $kernelProp = $cols | Where-Object { $_ -match 'kernel|op|name|benchmark' } | Select-Object -First 1
        if (-not $kernelProp) { $kernelProp = $cols[0] }
        $sizeProp = $cols | Where-Object { $_ -match '^n$|^N$|size|problem' } | Select-Object -First 1
        if (-not $sizeProp) { $sizeProp = $cols | Select-Object -Index 1 }
        $measuredProp = $cols | Where-Object { $_ -match 'meas.*gflop|measured|gflop|gflops' } | Select-Object -First 1
        $scoreProp = $cols | Where-Object { $_ -match 'portable.*efficiency|efficiency.*score|efficiency' } | Select-Object -First 1
        $roofProp = $cols | Where-Object { $_ -match 'roof.*gflop|roof_gflop|roof' } | Select-Object -First 1

        $gemmRows = @()
        if ($kernelProp) { $gemmRows = $csv | Where-Object { $_.$kernelProp -match 'GEMM' } }
        if ($gemmRows.Count -eq 0) { $gemmRows = $csv | Where-Object { ($_.PSObject.Properties.Value -join ' ') -match 'GEMM' } }
        if ($gemmRows.Count -eq 0) { Write-Warning "No GEMM rows found in $dst"; continue }

        $sizeVals = @()
        foreach ($r in $gemmRows) {
            $val = $r.PSObject.Properties[$sizeProp].Value
            $intVal = 0
            [int]::TryParse($val, [ref]$intVal) | Out-Null
            $sizeVals += @{ row=$r; n=$intVal }
        }
        $target = $null
        $row2048 = $sizeVals | Where-Object { $_.n -eq 2048 } | Select-Object -First 1
        if ($row2048) { $target = $row2048.row } else { $ordered = $sizeVals | Sort-Object -Property n -Descending; $target = $ordered[0].row }

        $measured = ""; $score = ""; $roof = ""
        if ($measuredProp) { $measured = $target.PSObject.Properties[$measuredProp].Value }
        if ($scoreProp) { $score = $target.PSObject.Properties[$scoreProp].Value }
        if ($roofProp) { $roof = $target.PSObject.Properties[$roofProp].Value }
        $nVal = 0; [int]::TryParse($target.PSObject.Properties[$sizeProp].Value, [ref]$nVal) | Out-Null

        $line = "${mr},${nr},${nVal},${measured},${score},${roof},${dst}"
        $line | Out-File -FilePath $analysisCsv -Append -Encoding utf8
        Write-Output "Refine append: $line"
    }
}

Stop-Transcript

# Pick refined best
if (-not (Test-Path $analysisCsv)) { FailAndExit "Missing analysis CSV $analysisCsv" 5 }
try { $anal = Import-Csv -Path $analysisCsv } catch { FailAndExit "Failed to read $analysisCsv" 6 }
$bestRef = $anal | ForEach-Object { $_ | Add-Member -NotePropertyName measured_num -NotePropertyValue ([double]($_.measured_gflops)) -PassThru } |
    Sort-Object -Property @{Expression={'[double]$($_.measured_gflops)'};Descending=$true}, @{Expression={'[int]$($_.mr)'};Descending=$false}, @{Expression={'[int]$($_.nr)'};Descending=$false} | Select-Object -First 1

[int]$bestMr = [int]$bestRef.mr
[int]$bestNr = [int]$bestRef.nr
Write-Output "Refined best MR=$bestMr NR=$bestNr measured_gflops=$($bestRef.measured_gflops)"

# Run CAQR microbench with tuned GEMM params, p=32, ForkJoin=8, record JFR
Start-Transcript -Path $caqrLog -Force
Write-Output "[3/..] Running CAQR microbench with MR=$bestMr NR=$bestNr p=32 ForkJoin=8"
$caqrCmd = ".\gradlew.bat --% test --tests 'net.faulj.benchmark.CAQRMicrobenchTest' -Dla.qr.strategy=CAQR -Dla.qr.caqr.p=32 -Dla.gemm.mr=$bestMr -Dla.gemm.nr=$bestNr -Djava.util.concurrent.ForkJoinPool.common.parallelism=8 -Dorg.gradle.jvmargs='-XX:StartFlightRecording=filename=$caqrJfr,duration=60s,settings=profile' --no-daemon --stacktrace"
Write-Output $caqrCmd
& .\gradlew.bat --% test --tests "net.faulj.benchmark.CAQRMicrobenchTest" -Dla.qr.strategy=CAQR -Dla.qr.caqr.p=32 -Dla.gemm.mr=$bestMr -Dla.gemm.nr=$bestNr -Djava.util.concurrent.ForkJoinPool.common.parallelism=8 -Dorg.gradle.jvmargs="-XX:StartFlightRecording=filename=$caqrJfr,duration=60s,settings=profile" --no-daemon --stacktrace 2>&1 | Tee-Object -FilePath (Join-Path $reportsDir "gradle_test_caqr_gemm_tuned.txt")
if ($LASTEXITCODE -ne 0) { Stop-Transcript; FailAndExit "CAQR microbench failed with exit code $LASTEXITCODE" $LASTEXITCODE }

# Copy produced caqr_bench.csv
$srcCaqr = Join-Path $reportsDir "caqr_bench.csv"
if (Test-Path $srcCaqr) {
    Copy-Item -Path $srcCaqr -Destination $caqrCsv -Force
    Write-Output "Copied $srcCaqr -> $caqrCsv"
} else {
    Write-Warning "Expected CAQR CSV not found at $srcCaqr"
}

# Write a short tuning report
"CAQR tuning report" | Out-File -FilePath $caqrReport -Encoding utf8
"tuned_mr,$bestMr" | Out-File -FilePath $caqrReport -Append -Encoding utf8
"tuned_nr,$bestNr" | Out-File -FilePath $caqrReport -Append -Encoding utf8
"best_measured_gflops,$($bestRef.measured_gflops)" | Out-File -FilePath $caqrReport -Append -Encoding utf8
"analysis_csv,$analysisCsv" | Out-File -FilePath $caqrReport -Append -Encoding utf8

Stop-Transcript

Write-Output "All done. Analysis: $analysisCsv, CAQR CSV: $caqrCsv, JFR: $caqrJfr"
exit 0
