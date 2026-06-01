# backup_ledger.ps1
#
# SCP the paper trading ledger + prob history from Oracle VM to local backup/.
# Run manually or schedule via Windows Task Scheduler (daily).
#
# Task Scheduler setup:
#   Program: powershell.exe
#   Arguments: -ExecutionPolicy Bypass -File "E:\93050207\python\BYBIT_ML\backup_ledger.ps1"
#   Trigger: Daily 08:00

$VM_IP   = "140.238.37.45"
$VM_USER = "ubuntu"
$KEY     = "$env:USERPROFILE\Downloads\ssh-key-2026-05-31.key"
$REMOTE  = "/home/ubuntu/bybit_ml/storage/live"
$LOCAL   = "E:\93050207\python\BYBIT_ML\backup"

$DATE = Get-Date -Format "yyyyMMdd"

Write-Host "Backing up ledger and prob_history from VM..." -ForegroundColor Cyan

scp -i $KEY -o StrictHostKeyChecking=no "${VM_USER}@${VM_IP}:${REMOTE}/paper_trading_ledger.json" `
    "$LOCAL\ledger_${DATE}.json" 2>$null

scp -i $KEY -o StrictHostKeyChecking=no "${VM_USER}@${VM_IP}:${REMOTE}/prob_history.csv" `
    "$LOCAL\prob_history_${DATE}.csv" 2>$null

scp -i $KEY -o StrictHostKeyChecking=no "${VM_USER}@${VM_IP}:${REMOTE}/active_positions.json" `
    "$LOCAL\positions_${DATE}.json" 2>$null

$files = Get-ChildItem $LOCAL -Filter "*_${DATE}.*"
if ($files) {
    Write-Host "Backed up $($files.Count) files to $LOCAL" -ForegroundColor Green
    $files | ForEach-Object { Write-Host "  $($_.Name) ($([math]::Round($_.Length/1KB, 1)) KB)" }
} else {
    Write-Host "WARNING: No files backed up. Check VM connectivity." -ForegroundColor Yellow
}
