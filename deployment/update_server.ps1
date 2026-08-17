[CmdletBinding()]
param(
    [string]$ConfigPath = (Join-Path $PSScriptRoot 'server_connection.local.json'),
    [Alias('Alias')]
    [string]$ProfileName,
    [string]$HostName,
    [Nullable[int]]$Port,
    [string]$UserName,
    [Nullable[int]]$LocalPort,
    [Nullable[int]]$RemotePort,
    [string]$RemoteProjectRoot
)

$ErrorActionPreference = 'Stop'

function Read-ValueOrDefault {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [AllowEmptyString()][string]$DefaultValue
    )
    $answer = Read-Host "${Label} [${DefaultValue}]"
    if ([string]::IsNullOrWhiteSpace($answer)) {
        return $DefaultValue
    }
    return $answer.Trim()
}

function Convert-ToPort {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)]$Value
    )
    $number = 0
    if (-not [int]::TryParse([string]$Value, [ref]$number) -or $number -lt 1 -or $number -gt 65535) {
        throw "${Name} must be an integer between 1 and 65535."
    }
    return $number
}

if (Test-Path -LiteralPath $ConfigPath -PathType Leaf) {
    $config = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
} else {
    $config = [pscustomobject]@{
        profileName = 'AI Garment GPU Server'
        sshHost = ''
        sshPort = 22
        sshUser = 'root'
        localPort = 3000
        remotePort = 3000
        remoteProjectRoot = '/opt/ai-garment-pattern-3d-demo'
        healthTimeoutSeconds = 180
    }
}

if (-not $config.remoteProjectRoot) {
    $config | Add-Member -NotePropertyName remoteProjectRoot -NotePropertyValue '/opt/ai-garment-pattern-3d-demo'
}

$updateNames = @('ProfileName', 'HostName', 'Port', 'UserName', 'LocalPort', 'RemotePort', 'RemoteProjectRoot')
$hasCommandLineUpdate = $false
foreach ($name in $updateNames) {
    if ($PSBoundParameters.ContainsKey($name)) {
        $hasCommandLineUpdate = $true
        break
    }
}

if (-not $hasCommandLineUpdate) {
    Write-Host 'Update the GPU server connection. Press Enter to keep a value.' -ForegroundColor Cyan
    Write-Host 'This config never accepts or stores an SSH password.' -ForegroundColor Yellow
    $ProfileName = Read-ValueOrDefault -Label 'Profile alias' -DefaultValue ([string]$config.profileName)
    $HostName = Read-ValueOrDefault -Label 'SSH host' -DefaultValue ([string]$config.sshHost)
    $Port = Convert-ToPort -Name 'SSH port' -Value (Read-ValueOrDefault -Label 'SSH port' -DefaultValue ([string]$config.sshPort))
    $UserName = Read-ValueOrDefault -Label 'SSH user' -DefaultValue ([string]$config.sshUser)
    $LocalPort = Convert-ToPort -Name 'Local web port' -Value (Read-ValueOrDefault -Label 'Local web port' -DefaultValue ([string]$config.localPort))
    $RemotePort = Convert-ToPort -Name 'Remote web port' -Value (Read-ValueOrDefault -Label 'Remote web port' -DefaultValue ([string]$config.remotePort))
    $RemoteProjectRoot = Read-ValueOrDefault -Label 'Remote project root' -DefaultValue ([string]$config.remoteProjectRoot)
}

if ($PSBoundParameters.ContainsKey('ProfileName') -or -not $hasCommandLineUpdate) {
    if ([string]::IsNullOrWhiteSpace($ProfileName)) { throw 'Profile alias cannot be empty.' }
    $config.profileName = $ProfileName.Trim()
}
if ($PSBoundParameters.ContainsKey('HostName') -or -not $hasCommandLineUpdate) {
    if ($HostName -notmatch '^[A-Za-z0-9.-]+$') { throw 'Invalid SSH host.' }
    $config.sshHost = $HostName.Trim()
}
if ($PSBoundParameters.ContainsKey('Port') -or -not $hasCommandLineUpdate) {
    $config.sshPort = Convert-ToPort -Name 'SSH port' -Value $Port
}
if ($PSBoundParameters.ContainsKey('UserName') -or -not $hasCommandLineUpdate) {
    if ($UserName -notmatch '^[A-Za-z_][A-Za-z0-9_.-]*$') { throw 'Invalid SSH user.' }
    $config.sshUser = $UserName.Trim()
}
if ($PSBoundParameters.ContainsKey('LocalPort') -or -not $hasCommandLineUpdate) {
    $config.localPort = Convert-ToPort -Name 'Local web port' -Value $LocalPort
}
if ($PSBoundParameters.ContainsKey('RemotePort') -or -not $hasCommandLineUpdate) {
    $config.remotePort = Convert-ToPort -Name 'Remote web port' -Value $RemotePort
}
if ($PSBoundParameters.ContainsKey('RemoteProjectRoot') -or -not $hasCommandLineUpdate) {
    if ($RemoteProjectRoot -notmatch '^/[A-Za-z0-9._/-]+$') { throw 'Invalid remote project root.' }
    $config.remoteProjectRoot = $RemoteProjectRoot.Trim()
}

$parent = Split-Path -Parent $ConfigPath
if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
}
$config | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $ConfigPath -Encoding UTF8

Write-Host 'Connection config updated:' -ForegroundColor Green
Write-Host "  Alias: $($config.profileName)"
Write-Host "  SSH:  $($config.sshUser)@$($config.sshHost):$($config.sshPort)"
Write-Host "  Web:  http://127.0.0.1:$($config.localPort)"
Write-Host "  Root: $($config.remoteProjectRoot)"
Write-Host "  File: ${ConfigPath}"
Write-Host 'No password was saved.'
