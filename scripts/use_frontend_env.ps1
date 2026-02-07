<#
Loads `frontend/.env`, sets `DISCORD_WEBHOOK_URL` from `VITE_DISCORD_WEBHOOK` (or other keys),
and runs `gradlew.bat bootRun` in the repo root. Intended for local Windows dev only.

Usage:
  powershell -ExecutionPolicy Bypass -File .\scripts\use_frontend_env.ps1

#>
param(
    [string]$EnvFile = "frontend/.env",
    [string]$GradleCmd = ".\gradlew.bat",
    [string]$GradleArgs = "bootRun"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Push-Location $scriptDir
try {
    $repoRoot = Resolve-Path ".\.."
    Set-Location $repoRoot

    if (-not (Test-Path $EnvFile)) {
        Write-Error "Env file not found: $EnvFile"
        exit 1
    }

    $found = $false
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line.Length -eq 0 -or $line.StartsWith('#')) { return }
        if ($line -match '^(?:export\s+)?([^=\s]+)\s*=\s*(.*)$') {
            $key = $matches[1].Trim()
            $val = $matches[2].Trim()
            if ($val.StartsWith('"') -and $val.EndsWith('"')) { $val = $val.Substring(1,$val.Length-2) }
            if ($key -in @('VITE_DISCORD_WEBHOOK','DISCORD_WEBHOOK_URL','DISCORD_WEBHOOK')) {
                $env:DISCORD_WEBHOOK_URL = $val
                Write-Host "Set DISCORD_WEBHOOK_URL from $key"
                $found = $true
            }
        }
    }

    if (-not $found) {
        Write-Warning "No Discord webhook key found in $EnvFile. Set VITE_DISCORD_WEBHOOK or DISCORD_WEBHOOK_URL."
        exit 1
    }

    if (-not $env:DISCORD_WEBHOOK_URL) {
        Write-Error "DISCORD_WEBHOOK_URL is empty after parsing $EnvFile"
        exit 1
    }

    & $GradleCmd $GradleArgs
} finally {
    Pop-Location
}
