#!/bin/bash
cd /workspace/XPolicyLab/policy/Pi_0
export PATH=/workspace/miniforge3/bin:$PATH
SRC=/workspace/XPolicyLab/policy/Pi_0/checkpoints/RoboDojo-sim-arx_x5-joint-0
LOCAL=/root/pi0ck/RoboDojo-sim-arx_x5-joint-0
if [ ! -d "$LOCAL/60000/params" ]; then
  echo "[runner] copying checkpoint to local disk (MooseFS too slow for orbax)..."
  mkdir -p /root/pi0ck && cp -r "$SRC" /root/pi0ck/
  echo "[runner] copy done"
fi
exec bash setup_eval_policy_server.sh RoboDojo stack_bowls "$LOCAL" arx_x5 joint 0 0 uv 9999 0.0.0.0
