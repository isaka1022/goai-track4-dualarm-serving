#!/bin/bash
# Rebuild the fast LOCAL runtime from the persistent network volume (/workspace).
# Why: FUSE-based network filesystems are pathologically slow for Python's
# many-small-file access patterns (venv imports, Orbax checkpoint shards) —
# venv builds measured at minutes on the network volume vs ~30 s on local NVMe.
set -e

echo "=== [1/4] clone XPolicyLab to LOCAL ==="
rm -rf /root/XPolicyLab
git clone https://github.com/XPolicyLab/XPolicyLab.git /root/XPolicyLab

echo "=== [2/4] env_cfg to local ==="
cp -r /workspace/env_cfg /root/env_cfg

echo "=== [3/4] Pi_0 install (LOCAL venv via uv) ==="
cd /root/XPolicyLab/policy/Pi_0
bash install.sh

echo "=== [4/4] copy checkpoint LOCAL ==="
mkdir -p /root/XPolicyLab/policy/Pi_0/checkpoints
cp -r /workspace/XPolicyLab/policy/Pi_0/checkpoints/RoboDojo-sim-arx_x5-joint-0 \
      /root/XPolicyLab/policy/Pi_0/checkpoints/

echo "LOCAL_REBUILD_DONE"
