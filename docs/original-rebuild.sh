#!/bin/bash
set -e
cd /workspace
echo "=== [1/6] git-lfs ==="
git lfs version >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq git-lfs && git lfs install)
echo "=== [2/6] clone XPolicyLab ==="
rm -rf /workspace/XPolicyLab
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/XPolicyLab/XPolicyLab.git /workspace/XPolicyLab
echo "=== [3/6] Miniforge + pyyaml ==="
curl -sL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /workspace/mf.sh
rm -rf /workspace/miniforge3
bash /workspace/mf.sh -b -p /workspace/miniforge3
/workspace/miniforge3/bin/pip install -q pyyaml
echo "=== [4/6] env_cfg from RoboDojo ==="
cd /workspace && rm -rf rd_tmp env_cfg
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none --sparse https://github.com/RoboDojo-Benchmark/RoboDojo rd_tmp
cd rd_tmp && git sparse-checkout set env_cfg && cp -r env_cfg /workspace/env_cfg
cd /workspace && rm -rf rd_tmp
echo "=== [5/6] Pi_0 install (openpi via uv) ==="
cd /workspace/XPolicyLab/policy/Pi_0 && bash install.sh
echo "=== [6/6] Pi_0 checkpoint (params+assets only) ==="
cd /workspace && rm -rf pi0ckpt
export GIT_LFS_SKIP_SMUDGE=1
git clone --depth 1 --sparse https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo pi0ckpt
cd /workspace/pi0ckpt
B=ckpt/RoboDojo/Pi_0/RoboDojo-sim-arx_x5-joint-0/60000
git sparse-checkout set --no-cone "$B/params" "$B/assets" "$B/_CHECKPOINT_METADATA"
unset GIT_LFS_SKIP_SMUDGE
git lfs pull --include "$B/params/**,$B/assets/**,$B/_CHECKPOINT_METADATA"
DEST=/workspace/XPolicyLab/policy/Pi_0/checkpoints/RoboDojo-sim-arx_x5-joint-0
mkdir -p "$DEST"
mv "/workspace/pi0ckpt/$B" "$DEST/"
cd /workspace && rm -rf pi0ckpt
echo "REBUILD_DONE"
