#!/bin/bash
# Supervisor entry point. Run once after pod (re)start:
#   setsid bash /workspace/boot.sh > /workspace/boot.log 2>&1 &
if [ ! -f /root/XPolicyLab/policy/Pi_0/openpi/.venv/bin/activate ]; then
  echo "[boot] local runtime missing -> rebuilding from network volume"
  bash /workspace/local_rebuild.sh
fi
echo "[boot] starting supervised server loop"
while true; do
  bash /workspace/run_serve.sh
  echo "[boot] server exited code=$?, restarting in 5s"
  sleep 5
done
