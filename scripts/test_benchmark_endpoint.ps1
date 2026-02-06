param(
  [int]$Port = 8081,
  [int]$Iterations = 5,
  [int]$StartupWaitSec = 120,
  [int]$RequestTimeoutSec = 900
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$gradlew = Join-Path $root "gradlew.bat"

Write-Host "Starting backend on port $Port..."
$proc = Start-Process -FilePath $gradlew `
  -ArgumentList "bootRun --args='--server.port=$Port' --no-daemon" `
  -WorkingDirectory $root `
  -PassThru

try {
  $started = $false
  for ($i = 0; $i -lt $StartupWaitSec; $i++) {
    Start-Sleep -Seconds 1
    try {
      $ping = Invoke-WebRequest -UseBasicParsing "http://localhost:$Port/api/ping" -TimeoutSec 2
      if ($ping.StatusCode -eq 200) {
        $started = $true
        break
      }
    } catch {}
  }
  if (-not $started) {
    throw "Backend did not start within $StartupWaitSec seconds"
  }

  Write-Host "Backend is up. Querying diagnostic512 iterations=$Iterations..."
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $resp = Invoke-WebRequest -UseBasicParsing "http://localhost:$Port/api/benchmark/diagnostic512?iterations=$Iterations" -TimeoutSec $RequestTimeoutSec
  $sw.Stop()

  $obj = $resp.Content | ConvertFrom-Json
  Write-Host ("STATUS={0}" -f $resp.StatusCode)
  Write-Host ("ELAPSED_MS={0}" -f $sw.ElapsedMilliseconds)
  Write-Host ("ITER_ROWS={0}" -f $obj.iterations.Count)
  Write-Host ("CPU_GFLOPS={0}" -f $obj.cpu.gflops)
  Write-Host "First few rows:"
  $obj.iterations | Select-Object -First 8 | ForEach-Object {
    Write-Host ("  {0} iter={1} ms={2} info={3}" -f $_.operation, $_.iteration, $_.ms, $_.info)
  }
}
finally {
  if ($proc -and -not $proc.HasExited) {
    Write-Host "Stopping backend process $($proc.Id)..."
    Stop-Process -Id $proc.Id -Force
  }
}
