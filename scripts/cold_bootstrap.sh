#!/bin/bash
# Rebuild the π0 serving stack from nothing, onto a fresh network volume.
#
# When you need this: the original network volume held a 12 GB checkpoint plus
# the toolchain, and local_rebuild.sh assumed it was still there. If the volume
# was deleted (e.g. the account balance hit zero), that assumption is gone and
# there is no persistent source left to copy from. This script recreates it.
#
#   bash scripts/cold_bootstrap.sh
#
# Cost note: this pulls ~12 GB of checkpoint. Run it on the volume you intend
# to keep, not on ephemeral container disk, or you will pay for it twice.
#
# Derived from the bring-up sequence recorded in docs/RUNBOOK.ja.md. The
# checkpoint fetch below is the verified part (it is how the 29 GB upstream
# tree was reduced to 12 GB). The surrounding steps are reconstructed from the
# runbook's summary of the original /workspace/rebuild.sh, which lived on the
# pod and was lost with it — expect to adjust a path on first run.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
CKPT_NAME="RoboDojo-sim-arx_x5-joint-0"
CKPT_DEST="${WORKSPACE}/XPolicyLab/policy/Pi_0/checkpoints/${CKPT_NAME}"

echo "=== [1/6] git-lfs ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git git-lfs curl > /dev/null
git lfs install

echo "=== [2/6] XPolicyLab onto the persistent volume ==="
[ -d "${WORKSPACE}/XPolicyLab" ] || \
  git clone https://github.com/XPolicyLab/XPolicyLab.git "${WORKSPACE}/XPolicyLab"

echo "=== [3/6] Miniforge (conda is required: the reference env pins old deps) ==="
if [ ! -x "${WORKSPACE}/miniforge3/bin/conda" ]; then
  curl -sL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/mf.sh
  bash /tmp/mf.sh -b -p "${WORKSPACE}/miniforge3"
fi
# setup_eval_policy_server.sh calls `conda info --base`, which needs pyyaml.
"${WORKSPACE}/miniforge3/bin/pip" install -q pyyaml || true

echo "=== [4/6] env_cfg (sparse checkout — the full RoboDojo tree is not needed) ==="
if [ ! -d "${WORKSPACE}/env_cfg" ]; then
  rm -rf /tmp/rdcfg
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/RoboDojo-Benchmark/RoboDojo.git /tmp/rdcfg
  cd /tmp/rdcfg && git sparse-checkout set env_cfg
  cp -r /tmp/rdcfg/env_cfg "${WORKSPACE}/env_cfg"
fi

echo "=== [5/6] openpi venv (uv) ==="
cd "${WORKSPACE}/XPolicyLab/policy/Pi_0"
bash install.sh

echo "=== [6/6] π0 checkpoint: params + assets + metadata only ==="
# --no-cone lets us exclude train_state. We never resume training, so the
# optimizer state is dead weight: 29 GB upstream -> 12 GB here.
if [ -d "${CKPT_DEST}" ]; then
  echo "[6] checkpoint already present, skipping"
else
  rm -rf /tmp/rdckpt
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --sparse \
    https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo /tmp/rdckpt
  cd /tmp/rdckpt
  git sparse-checkout set --no-cone \
    "/ckpt/RoboDojo/Pi_0/${CKPT_NAME}/60000/params/**" \
    "/ckpt/RoboDojo/Pi_0/${CKPT_NAME}/60000/assets/**" \
    "/ckpt/RoboDojo/Pi_0/${CKPT_NAME}/60000/metadata/**"
  git lfs pull --include "ckpt/RoboDojo/Pi_0/${CKPT_NAME}/60000/**"
  mkdir -p "$(dirname "${CKPT_DEST}")"
  cp -r "/tmp/rdckpt/ckpt/RoboDojo/Pi_0/${CKPT_NAME}" "${CKPT_DEST}"
fi

du -sh "${CKPT_DEST}" || true
echo "COLD_BOOTSTRAP_DONE"
