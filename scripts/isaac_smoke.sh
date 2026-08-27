#!/bin/bash
# Build a local Isaac Sim box and record evaluation videos of the policy.
#
# Why this exists: the competition entry was deliberately "serve-first" — the
# organizer ran the simulator, we only served π0. That kept the box small
# (no Isaac Sim, ~$20 of GPU) but it also means we never rendered a single
# frame of the robot. This script buys the footage back in one sitting.
#
# Paste into a fresh pod terminal:
#   bash scripts/isaac_smoke.sh
#
# Requirements (RoboDojo official): Ubuntu 22.04, RAM >= 32 GB, VRAM >= 16 GB,
# NVIDIA driver 570/580, CUDA 12.8. Give the container disk >= 100 GB — Isaac
# Sim, Isaac Lab, CuRobo and the assets do not fit in 60 GB alongside the
# checkpoint.
#
# Stages 1-5 are resumable: each one skips itself if its marker already exists,
# and install.sh has its own --from checkpointing (system, conda, base_deps,
# submodules, isaacsim, isaaclab, curobo).
#
# NOTE: stages 2-5 follow the official install guide
# (robodojo-benchmark.com/doc/usage/install-and-download/) but have NOT been
# executed by us yet. Expect to fix at least one thing on first run; that is
# what the resume flags are for. Stage 6 is the genuinely unverified part —
# see the comment there before running it.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
ROBODOJO="${WORKSPACE}/RoboDojo"
MARKERS="${WORKSPACE}/.isaac_smoke"
mkdir -p "${MARKERS}"

stage_done() { [ -f "${MARKERS}/$1" ]; }
mark_done()  { touch "${MARKERS}/$1"; }

echo "=== [0/6] make sure the π0 stack exists on the volume ==="
# Normally the volume still holds the checkpoint and this is a no-op. It is not
# guaranteed though: an empty balance can take the volume with it, and losing it
# silently would waste GPU hours further down. Detect, and rebuild if needed.
# Set SKIP_COLD_BOOTSTRAP=1 if you want to inspect the volume by hand first.
CKPT="${WORKSPACE}/XPolicyLab/policy/Pi_0/checkpoints/RoboDojo-sim-arx_x5-joint-0"
if [ -d "${CKPT}" ]; then
  echo "[0] checkpoint present — skipping the 12 GB fetch"
  du -sh "${CKPT}" || true
elif [ "${SKIP_COLD_BOOTSTRAP:-0}" = "1" ]; then
  echo "[0] checkpoint missing and SKIP_COLD_BOOTSTRAP=1 — stopping here"
  exit 1
else
  echo "[0] checkpoint missing -> cold bootstrap (~12 GB download)"
  bash "$(dirname "$0")/cold_bootstrap.sh"
fi

echo "=== [1/6] Vulkan userspace (Isaac renders through Vulkan, not EGL) ==="
# Unlike the PARC/LIBERO work, which used MUJOCO_GL=egl, Isaac Sim needs the
# Vulkan loader and ICD present even in headless mode.
if ! stage_done vulkan; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq libvulkan1 mesa-vulkan-drivers vulkan-tools git-lfs > /dev/null
  git lfs install
  vulkaninfo --summary > "${WORKSPACE}/vulkaninfo.txt" 2>&1 || \
    echo "[1] vulkaninfo failed — check the log before continuing"
  mark_done vulkan
else
  echo "[1] skip (already done)"
fi

echo "=== [2/6] clone RoboDojo ==="
if ! stage_done clone; then
  [ -d "${ROBODOJO}" ] || git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git "${ROBODOJO}"
  mark_done clone
else
  echo "[2] skip (already done)"
fi

echo "=== [3/6] install Isaac Sim 5.1 + Isaac Lab + CuRobo (the long one) ==="
# This is the step that takes hours. If it dies partway, re-run this script;
# or resume a specific phase manually:
#   cd ${ROBODOJO} && bash scripts/install.sh --from isaaclab
if ! stage_done install; then
  cd "${ROBODOJO}"
  bash scripts/install.sh -i
  mark_done install
else
  echo "[3] skip (already done)"
fi

echo "=== [4/6] fetch simulation assets ==="
if ! stage_done assets; then
  cd "${ROBODOJO}"
  bash scripts/init_assets.sh
  mark_done assets
else
  echo "[4] skip (already done)"
fi

echo "=== [5/6] point embodiment configs at this checkout ==="
if ! stage_done embodiment; then
  cd "${ROBODOJO}"
  python utils/update_embodiment_config_path.py
  mark_done embodiment
else
  echo "[5] skip (already done)"
fi

echo "=== [6/6] run one task headless and keep the video ==="
# UNVERIFIED: the exact evaluation entry point for a Pi_0 policy has not been
# confirmed. Two shapes are plausible and the docs are ambiguous about which
# one drives an already-running server:
#
#   (a) robodojo.sh spawns the policy itself
#       bash scripts/robodojo.sh smoke --policy-dir XPolicyLab/policy/Pi_0 \
#         --ckpt <CKPT> --policy-env uv --env-cfg arx_x5 --action-type joint \
#         --only stack_bowls --headless --fail-fast
#
#   (b) our server runs separately (scripts/run_serve.sh) and the evaluator
#       connects to 127.0.0.1:9999
#
# Check `bash scripts/robodojo.sh --help` on the pod first, then fill this in.
# Do not guess: a wrong invocation burns GPU minutes and produces no video.
echo "[6] STOP — read the comment block above and confirm the eval command."
echo "[6] Expected output once it runs:"
echo "[6]   eval_result/RoboDojo/<task>/<policy>/<env_cfg>/<seed>/<run_id>/"
echo "[6]     episode_*.mp4   <- the footage we came for"
echo "[6]     _result.json    <- success counts, score, eval_time"

echo "ISAAC_SMOKE_SETUP_DONE"
