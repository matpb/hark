# Installs hark as a per-user Scheduled Task "hark" (no admin) and downloads the default and VAD models unless -NoModels is given.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\install-windows.ps1 [-Binary target\release\hark.exe] [-NoModels]
param([string]$Binary = "", [switch]$NoModels)
$ErrorActionPreference = "Stop"

# Inside a release archive hark.exe sits one level above scripts\; in the repo the build output is used.
if (-not $Binary) {
    $bundled = Join-Path $PSScriptRoot "..\hark.exe"
    if (Test-Path $bundled) { $Binary = $bundled } else { $Binary = "target\release\hark.exe" }
}

if (-not (Test-Path $Binary)) {
    throw "no binary at $Binary; build with: cargo build --release --features vulkan"
}
$binDir = Join-Path $env:LOCALAPPDATA "hark\bin"
New-Item -ItemType Directory -Force -Path $binDir | Out-Null
$exe = Join-Path $binDir "hark.exe"

$existing = Get-ScheduledTask -TaskName "hark" -ErrorAction SilentlyContinue
if ($existing) {
    Stop-ScheduledTask -TaskName "hark" -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}
Copy-Item -Force $Binary $exe
Write-Host "binary: $exe"

$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$log = Join-Path $env:LOCALAPPDATA "hark\hark.log"
$action = New-ScheduledTaskAction -Execute $exe -Argument "serve --log-file `"$log`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "hark" -Action $action -Trigger $trigger -Principal $principal `
    -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName "hark"
Write-Host "scheduled task hark registered and started for $user"
Write-Host "logs: $log"

if (-not $NoModels) {
    $health = "http://127.0.0.1:8797/health"
    Write-Host "waiting for $health"
    $healthy = $false
    $end = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $end) {
        try {
            Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 -Uri $health | Out-Null
            $healthy = $true
            break
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $healthy) { throw "hark did not answer within 60 s; see $log" }
    Write-Host "downloading the default and VAD models (large-v3-turbo is about 1.6 GB)"
    & $exe models pull --defaults
    if ($LASTEXITCODE -ne 0) {
        throw "downloading the models failed; rerun install-windows.ps1, or use -NoModels and put the model files in the models directory yourself"
    }
}
