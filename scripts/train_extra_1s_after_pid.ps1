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

$logPath = Join-Path $logDir "extra_1s_training.log"
$architectures = @("vgg_cnn", "cnn_lstm", "lstm", "mobilenet_cnn")

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

foreach ($architecture in $architectures) {
    "[$(Get-Date -Format o)] Starting $architecture." |
        Tee-Object -FilePath $logPath -Append

    & python scripts\train_advanced_window_experiments.py `
        --window 1.0 `
        --architecture $architecture `
        --epochs $Epochs `
        --batch-size $BatchSize 2>&1 |
        Tee-Object -FilePath $logPath -Append

    if ($LASTEXITCODE -ne 0) {
        "[$(Get-Date -Format o)] $architecture failed with exit code $LASTEXITCODE." |
            Tee-Object -FilePath $logPath -Append
        exit $LASTEXITCODE
    }

    "[$(Get-Date -Format o)] Finished $architecture." |
        Tee-Object -FilePath $logPath -Append
}

"[$(Get-Date -Format o)] All queued 1s extra trainings finished." |
    Tee-Object -FilePath $logPath -Append
