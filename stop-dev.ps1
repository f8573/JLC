$pidFile = "dev-pids.txt"

if (-not (Test-Path $pidFile)) {
    Write-Host "No PID file found."
    exit
}

$pids = Get-Content $pidFile

foreach ($pid in $pids) {
    try {
        Stop-Process -Id $pid -Force
        Write-Host "Stopped PID $pid"
    }
    catch {
        Write-Host "PID $pid not running"
    }
}

# Cleanup
Remove-Item $pidFile
