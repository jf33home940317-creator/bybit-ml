# check_results.ps1 — 查看 Oracle VM 上的 paper trading 正確率
$KEY = "$env:USERPROFILE\Downloads\ssh-key-2026-05-31.key"
ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@140.238.37.45 `
    "cd /home/ubuntu/bybit_ml && /home/ubuntu/bybit_ml/venv/bin/python show_results.py"
