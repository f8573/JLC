$pidFile = "dev-pids.txt"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontendRoot = Join-Path $repoRoot "frontend"

# Clear old PID file if it exists
if (Test-Path $pidFile) {
    Remove-Item $pidFile
}

# Start backend (Spring Boot)
$backend = Start-Process powershell `
    -ArgumentList "-Command", "cd '$repoRoot'; .\gradlew.bat bootRun" `
    -PassThru `
    -WindowStyle Hidden

# Start frontend (npm)
$frontend = Start-Process powershell `
    -ArgumentList "-Command", "cd '$frontendRoot'; npm run dev" `
    -PassThru `
    -WindowStyle Hidden

# Save PIDs
$backend.Id | Out-File -Append $pidFile
$frontend.Id | Out-File -Append $pidFile

Write-Host "Started processes:"
Write-Host "Backend PID: $($backend.Id)"
Write-Host "Frontend PID: $($frontend.Id)"
Write-Host "Saved to $pidFile"
