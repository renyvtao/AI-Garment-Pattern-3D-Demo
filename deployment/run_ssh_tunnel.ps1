[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9.-]+$')]
    [string]$HostName,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 65535)]
    [int]$Port,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z_][A-Za-z0-9_.-]*$')]
    [string]$UserName,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 65535)]
    [int]$LocalPort,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 65535)]
    [int]$RemotePort,

    [ValidatePattern('^/[A-Za-z0-9._/-]+$')]
    [string]$RemoteProjectRoot = '/opt/ai-garment-pattern-3d-demo'
)

$ErrorActionPreference = 'Stop'
$Host.UI.RawUI.WindowTitle = "AI Garment SSH tunnel - ${HostName}:${Port}"

try {
    $ssh = Get-Command ssh.exe -ErrorAction Stop
} catch {
    Write-Host 'Windows OpenSSH Client was not found. Install it from Windows Optional Features.' -ForegroundColor Red
    Read-Host 'Press Enter to close'
    exit 1
}

Write-Host 'Starting AI Garment services and opening the SSH tunnel.' -ForegroundColor Cyan
Write-Host "Server: ${UserName}@${HostName}:${Port}"
Write-Host "Forward: http://127.0.0.1:${LocalPort} -> server 127.0.0.1:${RemotePort}"
Write-Host 'Enter the SSH password in this window if prompted. The script never stores it.'
Write-Host 'On first connection, verify the SSH host fingerprint before entering yes.'
Write-Host 'Please wait until AI_GARMENT_READY appears.'
Write-Host 'Keep this window open while using the web app.' -ForegroundColor Yellow
Write-Host ''

$remoteCommand = "cd '${RemoteProjectRoot}' && bash deployment/start_services.sh && echo AI_GARMENT_READY && exec sleep infinity"

& $ssh.Source `
    -p $Port `
    -T `
    -o 'ExitOnForwardFailure=yes' `
    -o 'ServerAliveInterval=30' `
    -o 'ServerAliveCountMax=3' `
    -L "${LocalPort}:127.0.0.1:${RemotePort}" `
    "${UserName}@${HostName}" `
    $remoteCommand

$exitCode = $LASTEXITCODE
Write-Host ''
if ($exitCode -eq 0) {
    Write-Host 'The SSH tunnel has closed.' -ForegroundColor Yellow
} else {
    Write-Host "The server startup or SSH tunnel exited with code ${exitCode}." -ForegroundColor Red
    Write-Host 'Check the instance state, host, port, password, project path, and messages above.'
}
Read-Host 'Press Enter to close'
exit $exitCode
