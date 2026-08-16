#!/bin/bash
# Start the Pi_0 policy server (generalist checkpoint, joint action space).
# Args: bench task ckpt_name env_cfg action_type seed gpu uv_env port host
cd /root/XPolicyLab/policy/Pi_0
export PATH=/workspace/miniforge3/bin:$PATH
exec bash setup_eval_policy_server.sh RoboDojo stack_bowls sim arx_x5 joint 0 0 uv 9999 0.0.0.0
