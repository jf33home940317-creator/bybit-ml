# deploy_oracle.ps1
#
# Deploy local source to Oracle VM. Webhook is read from local .env and pushed
# over ssh stdin so this script itself never holds a secret.

$VM_IP   = "140.238.37.45"
$VM_USER = "ubuntu"
# Try primary key, fallback to oracle_dca
$KEY_PRIMARY = "$env:USERPROFILE\Downloads\ssh-key-2026-05-31.key"
$KEY_FALLBACK = "$env:USERPROFILE\.ssh\oracle_dca"
if (Test-Path $KEY_PRIMARY) { $KEY = $KEY_PRIMARY } else { $KEY = $KEY_FALLBACK }
$PROJECT = "E:\93050207\python\BYBIT_ML"
$REMOTE  = "/home/ubuntu/bybit_ml"

# Read webhook from local .env (script must never hardcode it)
$EnvFile = "$PROJECT\.env"
if (-not (Test-Path $EnvFile)) {
    Write-Host "ERROR: local .env missing at $EnvFile" -ForegroundColor Red
    exit 1
}
$WebhookLine = Get-Content $EnvFile | Where-Object { $_ -match '^\s*DISCORD_WEBHOOK_URL\s*=' } | Select-Object -First 1
if (-not $WebhookLine) {
    Write-Host "ERROR: DISCORD_WEBHOOK_URL not found in local .env" -ForegroundColor Red
    exit 1
}
$Webhook = ($WebhookLine -split '=', 2)[1].Trim()
if (-not $Webhook) {
    Write-Host "ERROR: DISCORD_WEBHOOK_URL value is empty" -ForegroundColor Red
    exit 1
}

Write-Host "[1/7] Fixing SSH key permissions..." -ForegroundColor Cyan
icacls $KEY /inheritance:r /grant:r "${env:USERNAME}:R" | Out-Null

Write-Host "[2/7] Packing project files..." -ForegroundColor Cyan
$TMP_ZIP = "$env:TEMP\bybit_ml_deploy.zip"
if (Test-Path $TMP_ZIP) { Remove-Item $TMP_ZIP }

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::Open($TMP_ZIP, 'Create')

foreach ($f in @("config.py","run_live.py","show_results.py","requirements.txt")) {
    $full = Join-Path $PROJECT $f
    if (Test-Path $full) {
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $full, $f) | Out-Null
    }
}

foreach ($dir_name in @("live","data","features","models")) {
    $dir = "$PROJECT\$dir_name"
    if (Test-Path $dir) {
        Get-ChildItem $dir -Recurse -File | ForEach-Object {
            $entry = "$dir_name\" + $_.FullName.Substring(("$dir\").Length)
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $_.FullName, $entry) | Out-Null
        }
    }
}

foreach ($sub in @("models","backtest","features")) {
    $dir = "$PROJECT\storage\$sub"
    if (Test-Path $dir) {
        Get-ChildItem $dir -Recurse -File | Where-Object { $_.Extension -ne ".png" } | ForEach-Object {
            $entry = "storage\$sub\" + $_.FullName.Substring(("$dir\").Length)
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $_.FullName, $entry) | Out-Null
        }
    }
}
$zip.Dispose()
Write-Host "   Done: $TMP_ZIP" -ForegroundColor Green

Write-Host "[3/7] Uploading files to VM (~20MB)..." -ForegroundColor Cyan
scp -i $KEY -o StrictHostKeyChecking=no $TMP_ZIP "${VM_USER}@${VM_IP}:/tmp/bybit_ml.zip"

Write-Host "[4/7] Installing environment on VM..." -ForegroundColor Cyan
$SETUP = "set -e; sudo apt-get update -qq; sudo apt-get install -y -qq python3-pip python3-venv unzip; mkdir -p $REMOTE; cd $REMOTE; unzip -o /tmp/bybit_ml.zip; python3 -m venv venv; source venv/bin/activate; pip install --quiet --upgrade pip; pip install --quiet -r requirements.txt; mkdir -p storage/live; echo DONE"
ssh -i $KEY -o StrictHostKeyChecking=no "${VM_USER}@${VM_IP}" $SETUP

Write-Host "[5/7] Pushing .env (webhook from local) over ssh stdin..." -ForegroundColor Cyan
$EnvContent = "DISCORD_WEBHOOK_URL=$Webhook`nBYBIT_API_KEY=`nBYBIT_API_SECRET=`n"
# PowerShell pipe writes UTF-8 BOM by default. Strip it on the VM side so
# python-dotenv reads the first key correctly (otherwise DISCORD_WEBHOOK_URL
# becomes ﻿DISCORD_WEBHOOK_URL and silently fails).
$ENV_CMD = "cat > $REMOTE/.env && sed -i '1s/^\xEF\xBB\xBF//' $REMOTE/.env && chmod 600 $REMOTE/.env"
$EnvContent | ssh -i $KEY -o StrictHostKeyChecking=no "${VM_USER}@${VM_IP}" $ENV_CMD

Write-Host "[6/7] Setting up systemd auto-start..." -ForegroundColor Cyan
$SVC = "[Unit]`nDescription=BYBIT_ML Live Signal Daemon`nAfter=network-online.target`nWants=network-online.target`n`n[Service]`nType=simple`nUser=ubuntu`nWorkingDirectory=$REMOTE`nExecStart=$REMOTE/venv/bin/python run_live.py`nRestart=always`nRestartSec=60`n`n[Install]`nWantedBy=multi-user.target"
$SVC_CMD = "echo '$SVC' | sudo tee /etc/systemd/system/bybit-ml.service > /dev/null && sudo systemctl daemon-reload && sudo systemctl enable bybit-ml && sudo systemctl restart bybit-ml"
ssh -i $KEY -o StrictHostKeyChecking=no "${VM_USER}@${VM_IP}" $SVC_CMD

Write-Host "[7/7] Checking service status..." -ForegroundColor Cyan
Start-Sleep -Seconds 5
ssh -i $KEY -o StrictHostKeyChecking=no "${VM_USER}@${VM_IP}" "sudo systemctl status bybit-ml --no-pager -l"

Write-Host ""
Write-Host "Deployment complete. Daemon running 24/7 on Oracle VM." -ForegroundColor Green
Write-Host "Live logs: ssh -i `"$KEY`" ubuntu@$VM_IP" -ForegroundColor Yellow
Write-Host "Then run: sudo journalctl -u bybit-ml -f" -ForegroundColor Yellow
