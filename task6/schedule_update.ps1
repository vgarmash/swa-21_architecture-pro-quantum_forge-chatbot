param(
    [switch]$Register,
    [switch]$RunJob
)

if (-not $Register -and -not $RunJob) {
    $Register = $true
}

$scriptPath = $MyInvocation.MyCommand.Path
$scriptDir = Split-Path -Parent $scriptPath
$projectRoot = Split-Path -Parent $scriptDir
$pythonExe = "python"
$updateScript = Join-Path $scriptDir "update_index.py"
$logFile = Join-Path $scriptDir "update_index.log"

function Write-Log {
    param(
        [string]$Level,
        [string]$Message
    )
    $timestamp = (Get-Date).ToString("s")
    $entry = "$timestamp | $Level | $Message"
    Add-Content -Path $logFile -Value $entry
}

function Invoke-Update {
    param(
        [int]$Attempt
    )

    $start = Get-Date
    & $pythonExe $updateScript
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Log -Level "ERROR" -Message "Scheduled update attempt #$Attempt failed with exit code $exitCode"
    } else {
        Write-Log -Level "INFO" -Message "Scheduled update attempt #$Attempt completed successfully"
    }

    return $exitCode
}

if ($RunJob) {
    $firstAttempt = Invoke-Update -Attempt 1
    if ($firstAttempt -ne 0) {
        Start-Sleep -Seconds 30
        $secondAttempt = Invoke-Update -Attempt 2
        if ($secondAttempt -ne 0) {
            exit $secondAttempt
        }
    }
    exit 0
}

if ($Register) {
    $taskName = "Task6_UpdateChromaIndex"
    $taskDescription = "Daily 06:00 update of Chroma index (task6). Retries once on failure."

    $arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" -RunJob"
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arguments -WorkingDirectory $scriptDir
    $trigger = New-ScheduledTaskTrigger -Daily -At 6:00am
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

    Try {
        Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description $taskDescription -Force | Out-Null
        Write-Host "Scheduled task '$taskName' registered."
        Write-Host "It will run daily at 06:00, execute update_index.py, log to update_index.log, and retry once on error."
        Write-Host "To test immediately: powershell -ExecutionPolicy Bypass -File `"$scriptPath`" -RunJob"
    } Catch {
        Write-Error "Failed to register scheduled task: $_"
        exit 1
    }

    exit 0
}
