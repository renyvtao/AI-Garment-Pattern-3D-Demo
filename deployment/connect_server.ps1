[CmdletBinding()]
param(
    [string]$ConfigPath = (Join-Path $PSScriptRoot 'server_connection.local.json'),
    [switch]$NoBrowser,
    [switch]$NoNewTunnel
)

$ErrorActionPreference = 'Stop'

function Get-CheckedConfig {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Connection config not found: ${Path}`nRun deployment\update_server.ps1 first."
    }

    $value = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([string]::IsNullOrWhiteSpace([string]$value.sshHost) -or
        ([string]$value.sshHost -notmatch '^[A-Za-z0-9.-]+$')) {
        throw 'Invalid sshHost in the connection config.'
    }
    if ([string]::IsNullOrWhiteSpace([string]$value.sshUser) -or
        ([string]$value.sshUser -notmatch '^[A-Za-z_][A-Za-z0-9_.-]*$')) {
        throw 'Invalid sshUser in the connection config.'
    }
    foreach ($name in @('sshPort', 'localPort', 'remotePort')) {
        $number = [int]$value.$name
        if ($number -lt 1 -or $number -gt 65535) {
            throw "${name} in the connection config must be between 1 and 65535."
        }
    }
    if (-not $value.healthTimeoutSeconds) {
        $value | Add-Member -NotePropertyName healthTimeoutSeconds -NotePropertyValue 180
    }
    if (-not $value.remoteProjectRoot) {
        $value | Add-Member -NotePropertyName remoteProjectRoot -NotePropertyValue '/opt/ai-garment-pattern-3d-demo'
    }
    if ([string]$value.remoteProjectRoot -notmatch '^/[A-Za-z0-9._/-]+$') {
        throw 'Invalid remoteProjectRoot in the connection config.'
    }
    return $value
}

function Test-HttpEndpoint {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [int]$TimeoutSeconds = 2
    )
    try {
        $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec $TimeoutSeconds
        return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400)
    } catch {
        return $false
    }
}

function Test-LocalPort {
    param([Parameter(Mandatory = $true)][int]$Port)
    try {
        return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    } catch {
        return (Test-NetConnection -ComputerName '127.0.0.1' -Port $Port -InformationLevel Quiet -WarningAction SilentlyContinue)
    }
}

try {
    $config = Get-CheckedConfig -Path $ConfigPath
    $siteUrl = "http://127.0.0.1:$([int]$config.localPort)/"
    $bodyHealthUrl = "http://127.0.0.1:$([int]$config.localPort)/api/body/health"
    $jobsHealthUrl = "http://127.0.0.1:$([int]$config.localPort)/api/jobs/health"
    $profile = if ($config.profileName) { [string]$config.profileName } else { 'AI Garment' }

    Write-Host "Profile: ${profile}" -ForegroundColor Cyan
    Write-Host "Remote: $($config.sshUser)@$($config.sshHost):$($config.sshPort)"
    Write-Host "Local web app: ${siteUrl}"

    $siteHealthy = Test-HttpEndpoint -Uri $siteUrl
    if (-not $siteHealthy) {
        if (Test-LocalPort -Port ([int]$config.localPort)) {
            throw "Local port $($config.localPort) is occupied by another service. Close the old SSH tunnel or change localPort with update_server.ps1."
        }
        if ($NoNewTunnel) {
            throw 'No healthy tunnel is available and -NoNewTunnel was specified.'
        }

        $tunnelScript = Join-Path $PSScriptRoot 'run_ssh_tunnel.ps1'
        if (-not (Test-Path -LiteralPath $tunnelScript -PathType Leaf)) {
            throw "Tunnel script not found: ${tunnelScript}"
        }

        $escapedTunnelScript = $tunnelScript.Replace("'", "''")
        $tunnelCommand = "& '${escapedTunnelScript}' -HostName '$($config.sshHost)' -Port $($config.sshPort) -UserName '$($config.sshUser)' -LocalPort $($config.localPort) -RemotePort $($config.remotePort) -RemoteProjectRoot '$($config.remoteProjectRoot)'"
        $encodedTunnelCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($tunnelCommand))
        $tunnelArgs = "-NoLogo -NoProfile -ExecutionPolicy Bypass -EncodedCommand ${encodedTunnelCommand}"

        Write-Host 'A separate SSH window is open. Enter the server password there once.' -ForegroundColor Yellow
        Write-Host 'It will start all remote services, keep the tunnel open, and requires no manual server command.'
        $tunnelProcess = Start-Process -FilePath 'powershell.exe' -ArgumentList $tunnelArgs -PassThru

        $timeout = [Math]::Max(15, [int]$config.healthTimeoutSeconds)
        $deadline = (Get-Date).AddSeconds($timeout)
        $lastProgressAt = Get-Date
        Start-Sleep -Seconds 8
        while ((Get-Date) -lt $deadline) {
            $tunnelProcess.Refresh()
            if ($tunnelProcess.HasExited) {
                throw 'The server startup or SSH tunnel process exited. Check the separate window for the exact error.'
            }
            if (Test-HttpEndpoint -Uri $siteUrl) {
                $siteHealthy = $true
                break
            }
            if (((Get-Date) - $lastProgressAt).TotalSeconds -ge 10) {
                Write-Host 'Waiting for SSH authentication or the remote web app...'
                $lastProgressAt = Get-Date
            }
            Start-Sleep -Seconds 2
        }
        if (-not $siteHealthy) {
            throw "The web app did not become healthy within ${timeout} seconds. Check the separate SSH window; the one-click launcher already attempted the remote startup."
        }
    } else {
        Write-Host 'An existing healthy tunnel was found and will be reused.' -ForegroundColor Green
    }

    Write-Host 'Web health check passed.' -ForegroundColor Green
    if (Test-HttpEndpoint -Uri $bodyHealthUrl) {
        Write-Host 'Body API health check passed.' -ForegroundColor Green
    } else {
        Write-Host 'Warning: the web app is reachable, but the Body API health check failed.' -ForegroundColor Yellow
    }
    if (Test-HttpEndpoint -Uri $jobsHealthUrl) {
        Write-Host 'Task API health check passed.' -ForegroundColor Green
    } else {
        Write-Host 'Warning: the web app is reachable, but the Task API health check failed.' -ForegroundColor Yellow
    }

    if (-not $NoBrowser) {
        Start-Process $siteUrl
        Write-Host "Opened ${siteUrl} in the default browser."
    }
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
