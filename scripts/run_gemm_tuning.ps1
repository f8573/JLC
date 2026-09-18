# Run GEMM-only tuning sweep using MR/NR overrides
# Creates JFR recordings, copies CSVs, extracts GEMM rows, and writes a summary

$rooflineDir = "build/reports/roofline"
$logPath = Join-Path $rooflineDir "portable_efficiency_gemm_tuning_log.txt"
$summaryPath = Join-Path $rooflineDir "portable_efficiency_gemm_tuning_summary.csv"

New-Item -ItemType Directory -Force -Path $rooflineDir | Out-Null

Start-Transcript -Path $logPath -Force

function FailAndExit($msg, $code=1) {
    Write-Error $msg
    Stop-Transcript
    exit $code
}

Write-Output "[1/..] Building project: running assemble"
$assembleCmd = ".\gradlew.bat --% assemble"
Write-Output $assembleCmd
& .\gradlew.bat --% assemble 2>&1 | Tee-Object -FilePath (Join-Path $rooflineDir "assemble_output.txt")
if ($LASTEXITCODE -ne 0) { FailAndExit "Gradle assemble failed with exit code $LASTEXITCODE" $LASTEXITCODE }

# Initialize summary CSV
"mr,nr,n,measured_gflops,portable_efficiency_score,roof_gflops" | Out-File -FilePath $summaryPath -Encoding utf8

$mrs = @(4,6,8)
$nrs = @(4,6,8)

foreach ($mr in $mrs) {
    foreach ($nr in $nrs) {
        Write-Output "--- Running MR=$mr NR=$nr ---"
        $jfr = Join-Path $rooflineDir "gemm_mr${mr}_nr${nr}.jfr"
        $testCmd = ".\gradlew.bat --% test --tests 'net.faulj.benchmark.roofline.PortableEfficiencyBenchmarkTest' -Dla.gemm.mr=$mr -Dla.gemm.nr=$nr -Dorg.gradle.jvmargs='-XX:StartFlightRecording=filename=$jfr,duration=20s,settings=profile' --no-daemon --stacktrace"
        Write-Output $testCmd

        # Run test and capture output
        $out = & .\gradlew.bat --% test --tests "net.faulj.benchmark.roofline.PortableEfficiencyBenchmarkTest" -Dla.gemm.mr=$mr -Dla.gemm.nr=$nr -Dorg.gradle.jvmargs="-XX:StartFlightRecording=filename=$jfr,duration=20s,settings=profile" --no-daemon --stacktrace 2>&1 | Tee-Object -FilePath (Join-Path $rooflineDir "gradle_test_mr${mr}_nr${nr}.txt")
        if ($LASTEXITCODE -ne 0) { FailAndExit "Gradle test failed for MR=$mr NR=$nr with exit code $LASTEXITCODE" $LASTEXITCODE }

        $src = Join-Path $rooflineDir "portable_efficiency_results.csv"
        $dst = Join-Path $rooflineDir "portable_efficiency_results_mr${mr}_nr${nr}.csv"
        if (Test-Path $src) {
            Copy-Item -Path $src -Destination $dst -Force
            Write-Output "Copied $src -> $dst"
        } else {
            Write-Warning "Expected results CSV not found at $src"
            FailAndExit "Missing portable_efficiency_results.csv after run MR=$mr NR=$nr" 2
        }

        # Parse CSV and extract GEMM row
        try {
            $csv = Import-Csv -Path $dst
        } catch {
            Write-Warning "Failed to import CSV ${dst}: $_"
            FailAndExit "CSV import failed for $dst" 3
        }

        if ($csv.Count -eq 0) {
            Write-Warning "CSV $dst is empty"
            FailAndExit "Empty CSV $dst" 4
        }

        $cols = ($csv | Get-Member -MemberType NoteProperty | Select-Object -ExpandProperty Name)
        $kernelProp = $cols | Where-Object { $_ -match 'kernel|op|name|benchmark' } | Select-Object -First 1
        if (-not $kernelProp) { $kernelProp = $cols[0] }
        $sizeProp = $cols | Where-Object { $_ -match '^n$|^N$|size|problem' } | Select-Object -First 1
        if (-not $sizeProp) { $sizeProp = $cols | Select-Object -Index 1 }
        $measuredProp = $cols | Where-Object { $_ -match 'meas.*gflop|measured|gflop|gflops' } | Select-Object -First 1
        $scoreProp = $cols | Where-Object { $_ -match 'portable.*efficiency|efficiency.*score|efficiency' } | Select-Object -First 1
        $roofProp = $cols | Where-Object { $_ -match 'roof.*gflop|roof_gflop|roof' } | Select-Object -First 1

        if (-not $measuredProp) { $measuredProp = $cols | Where-Object { $_ -match 'meas|value' } | Select-Object -First 1 }
        if (-not $scoreProp) { $scoreProp = $cols | Where-Object { $_ -match 'score|eff' } | Select-Object -First 1 }

        # Find GEMM rows
        $gemmRows = @()
        if ($kernelProp) {
            $gemmRows = $csv | Where-Object { $_.$kernelProp -match 'GEMM' }
        }
        if ($gemmRows.Count -eq 0) {
            # fallback: any field contains GEMM
            $gemmRows = $csv | Where-Object { ($_.PSObject.Properties.Value -join ' ') -match 'GEMM' }
        }
        if ($gemmRows.Count -eq 0) {
            Write-Warning "No GEMM rows found in $dst"
            FailAndExit "No GEMM rows found in $dst" 5
        }

        # Normalize size values and pick row with n=2048 or largest n
        $sizeVals = @()
        foreach ($r in $gemmRows) {
            $val = $r.PSObject.Properties[$sizeProp].Value
            $intVal = 0
            [int]::TryParse($val, [ref]$intVal) | Out-Null
            $sizeVals += @{ row=$r; n=$intVal }
        }
        $targetRow = $null
        $row2048 = $sizeVals | Where-Object { $_.n -eq 2048 } | Select-Object -First 1
        if ($row2048) { $targetRow = $row2048.row }
        else {
            $ordered = $sizeVals | Sort-Object -Property n -Descending
            $targetRow = $ordered[0].row
        }

        # Safely extract numeric fields
        $measured = ""
        $score = ""
        $roof = ""
        if ($measuredProp) { $measured = $targetRow.PSObject.Properties[$measuredProp].Value }
        if ($scoreProp) { $score = $targetRow.PSObject.Properties[$scoreProp].Value }
        if ($roofProp) { $roof = $targetRow.PSObject.Properties[$roofProp].Value }

        $nVal = 0
        [int]::TryParse($targetRow.PSObject.Properties[$sizeProp].Value, [ref]$nVal) | Out-Null

        # Append to summary
        $line = "${mr},${nr},${nVal},${measured},${score},${roof}"
        $line | Out-File -FilePath $summaryPath -Append -Encoding utf8
        Write-Output "Appended to summary: $line"
    }
}

Stop-Transcript
Write-Output "Sweep complete. Summary at $summaryPath, log at $logPath"
exit 0
