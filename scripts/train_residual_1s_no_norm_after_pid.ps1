param(
    [int]$WaitPid = 0,
    [int]$Epochs = 50,
    [int]$BatchSize = 32
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $projectRoot

$logDir = Join-Path $projectRoot "notebooks\results\advanced_window_experiments"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$logPath = Join-Path $logDir "residual_1s_no_norm_training.log"

"[$(Get-Date -Format o)] Queue started. WaitPid=$WaitPid Epochs=$Epochs BatchSize=$BatchSize" |
    Tee-Object -FilePath $logPath -Append

if ($WaitPid -gt 0) {
    $process = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
    if ($process) {
        "[$(Get-Date -Format o)] Waiting for PID $WaitPid to finish." |
            Tee-Object -FilePath $logPath -Append
        Wait-Process -Id $WaitPid
    }
    else {
        "[$(Get-Date -Format o)] PID $WaitPid is not running; starting immediately." |
            Tee-Object -FilePath $logPath -Append
    }
}

"[$(Get-Date -Format o)] Starting residual_cnn_1s_no_norm." |
    Tee-Object -FilePath $logPath -Append

& python scripts\train_advanced_window_experiments.py `
    --window 1.0 `
    --architecture residual_cnn `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --no-normalize-segments `
    --run-suffix no_norm 2>&1 |
    Tee-Object -FilePath $logPath -Append

if ($LASTEXITCODE -ne 0) {
    "[$(Get-Date -Format o)] residual_cnn_1s_no_norm failed with exit code $LASTEXITCODE." |
        Tee-Object -FilePath $logPath -Append
    exit $LASTEXITCODE
}

"[$(Get-Date -Format o)] Finished residual_cnn_1s_no_norm." |
    Tee-Object -FilePath $logPath -Append
