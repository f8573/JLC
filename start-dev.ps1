$pidFile = "dev-pids.txt"

# Clear old PID file if it exists
if (Test-Path $pidFile) {
    Remove-Item $pidFile
}

# Start backend (Spring Boot)
$backend = Start-Process powershell `
    -ArgumentList "-Command", "cd 'C:\Users\James\IdeaProjects\JLC'; .\gradlew.bat bootRun" `
    -PassThru `
    -WindowStyle Hidden

# Start frontend (npm)
$frontend = Start-Process powershell `
    -ArgumentList "-Command", "cd 'C:\Users\James\IdeaProjects\JLC\frontend'; npm run dev" `
    -PassThru `
    -WindowStyle Hidden

# Save PIDs
$backend.Id | Out-File -Append $pidFile
$frontend.Id | Out-File -Append $pidFile

Write-Host "Started processes:"
Write-Host "Backend PID: $($backend.Id)"
Write-Host "Frontend PID: $($frontend.Id)"
Write-Host "Saved to $pidFile"
