param(
    [string]$RemoteHost = "hanzhanghuan@spartan.hpc.unimelb.edu.au"
)

$ErrorActionPreference = "Stop"
$MethodRoot = Split-Path -Parent $PSScriptRoot
$SourceRoot = Join-Path $MethodRoot "src"
$RemoteMethodRoot = "/data/projects/punim1257/Group14/methods/matrix_profile"
$RemoteSourceRoot = "$RemoteMethodRoot/src"
$RemoteJobRoot = "$RemoteMethodRoot/jobs"

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE"
    }
}

Invoke-NativeChecked ssh $RemoteHost "mkdir -p '$RemoteSourceRoot/anomalies' '$RemoteJobRoot'"
Invoke-NativeChecked scp -r "$SourceRoot/anomalies/multiscale" "${RemoteHost}:${RemoteSourceRoot}/anomalies/"
Invoke-NativeChecked scp "$SourceRoot/detect_multiscale_anomalies.py" "${RemoteHost}:${RemoteSourceRoot}/"
Invoke-NativeChecked scp "$SourceRoot/aggregate_multiscale_shards.py" "${RemoteHost}:${RemoteSourceRoot}/"
Invoke-NativeChecked scp "$SourceRoot/plot_multiscale_selected.py" "${RemoteHost}:${RemoteSourceRoot}/"
Invoke-NativeChecked scp "$SourceRoot/meter_repository.json" "${RemoteHost}:${RemoteSourceRoot}/"
Invoke-NativeChecked scp -r "$PSScriptRoot" "${RemoteHost}:${RemoteMethodRoot}/"

Write-Output "Uploaded Matrix Profile application to $RemoteMethodRoot"
