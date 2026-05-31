# upload_final.ps1
#
# One-shot helper: pushes the latest validation reports to VM and refreshes
# the .env from the local .env file. Webhook is NOT in this script.

$VM_IP   = "140.238.37.45"
$VM_USER = "ubuntu"
$KEY     = "$env:USERPROFILE\Downloads\ssh-key-2026-05-31.key"
$PROJECT = "E:\93050207\python\BYBIT_ML"

# Read webhook from local .env
$EnvFile = "$PROJECT\.env"
if (-not (Test-Path $EnvFile)) {
    Write-Host "ERROR: 本機沒有 .env" -ForegroundColor Red
    exit 1
}
$WebhookLine = Get-Content $EnvFile | Where-Object { $_ -match '^\s*DISCORD_WEBHOOK_URL\s*=' } | Select-Object -First 1
if (-not $WebhookLine) {
    Write-Host "ERROR: .env 找不到 DISCORD_WEBHOOK_URL" -ForegroundColor Red
    exit 1
}
$Webhook = ($WebhookLine -split '=', 2)[1].Trim()

Write-Host "[1] Upload validation reports..." -ForegroundColor Cyan
scp -i $KEY -o StrictHostKeyChecking=no `
    "$PROJECT\storage\features\ETHUSDT_validation_report.json" `
    "$PROJECT\storage\features\BTCUSDT_validation_report.json" `
    "${VM_USER}@${VM_IP}:/home/ubuntu/bybit_ml/storage/features/"

Write-Host "[2] Push .env (webhook from local) over ssh stdin..." -ForegroundColor Cyan
$EnvContent = "DISCORD_WEBHOOK_URL=$Webhook`nBYBIT_API_KEY=`nBYBIT_API_SECRET=`n"
$EnvContent | ssh -i $KEY -o StrictHostKeyChecking=no "${VM_USER}@${VM_IP}" "cat > /home/ubuntu/bybit_ml/.env && sed -i '1s/^\xEF\xBB\xBF//' /home/ubuntu/bybit_ml/.env && chmod 600 /home/ubuntu/bybit_ml/.env"

Write-Host "[3] Restart service..." -ForegroundColor Cyan
ssh -i $KEY -o StrictHostKeyChecking=no "${VM_USER}@${VM_IP}" "sudo systemctl restart bybit-ml"
Start-Sleep -Seconds 15

Write-Host "[4] Status and logs:" -ForegroundColor Cyan
ssh -i $KEY -o StrictHostKeyChecking=no "${VM_USER}@${VM_IP}" "sudo systemctl status bybit-ml --no-pager -l && sudo journalctl -u bybit-ml -n 15 --no-pager"
