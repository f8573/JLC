param(
    [int]$threads = 0
)

# Run CAQR vs Householder microbench and collect CSV outputs.
$gradlew = Join-Path $PSScriptRoot "..\gradlew.bat"
if (!(Test-Path $gradlew)) { $gradlew = Join-Path $PSScriptRoot "..\gradlew" }

if ($threads -gt 0) {
    $fj = "-Djava.util.concurrent.ForkJoinPool.common.parallelism=$threads"
} else {
    $fj = ""
}

$outDir = Join-Path $PSScriptRoot "..\build\reports"
mkdir $outDir -Force | Out-Null

$strategies = @("CAQR","HOUSEHOLDER")
foreach ($s in $strategies) {
    Write-Host "Running microbench with strategy=$s threads=$threads"
    & $gradlew "test" "--tests" "net.faulj.benchmark.CAQRMicrobenchTest" -Dla.qr.strategy=$s $fj
    $src = Join-Path $PSScriptRoot "..\build\reports\caqr_bench.csv"
    if (Test-Path $src) {
        $stamp = Get-Date -Format yyyyMMddHHmmss
        $dest = Join-Path $PSScriptRoot "..\build\reports\caqr_bench_${s}_$stamp.csv"
        Copy-Item $src $dest -Force
        Write-Host "Saved:" $dest
    } else {
        Write-Host "Microbench output not found for strategy $s"
    }
}
