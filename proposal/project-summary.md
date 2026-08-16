# Project Summary — GOAI 2026 Track 4 (Dual-Arm)

**Team**: isaka1022 (solo, Japan) · **Entry**: Generalist VLA policy serving for dual-arm manipulation

**What we built.** A single generalist vision-language-action policy that handles all 12 GOAI-2026 dual-arm simulation tasks with one set of weights, served on production-grade infrastructure that has been verified end-to-end against the official evaluation protocol.

**Approach.** We serve **π0** (open-source `openpi`, Apache-2.0): a ~3B-parameter VLA whose language conditioning lets one checkpoint execute "stack the bowls" and "fold the clothes" alike, switching tasks purely through the per-step natural-language instruction the evaluator already provides. No per-task checkpoints, no task routing — the architecture a deployable dual-arm system should actually have. The policy outputs 50-step action chunks of 14-dim joint targets (6 joints + gripper per arm), amortizing inference latency over the network control loop.

**Engineering.** The policy server implements the benchmark's msgpack-over-WebSocket RPC (`reset` / `update_obs` / `get_action`) and runs on a single RTX 4090 (7.6 GB VRAM in bf16). Deployment splits storage into a persistent network volume (checkpoint, toolchain) and a fast local runtime, after measuring that FUSE-based network filesystems slow Python imports and checkpoint loads from seconds to minutes. A supervisor script rebuilds the runtime and restarts the server automatically, so recovery from a cold instance is one command.

**Verification (evidence).** Before requesting official evaluation, we validated the full path with the benchmark's *own client class* (`WsModelClient` — the same code the organizer's evaluation loop uses): external WebSocket handshake (HTTP 101), then a live `reset → update_obs → get_action` round-trip returning a 50-step chunk of valid, finite 14-dim actions in the expected per-arm layout. The serving stack is protocol-exact, not merely reachable. An official X-Eval score will follow as soon as our evaluation application is processed (overseas-team registration is being coordinated with the organizers).

**Why this matters.** Server-based evaluation mirrors real robot deployment: policies must be robust services, not notebook artifacts. Our entry treats inference serving as a first-class part of embodied AI — verified protocol compliance, self-healing operations, cost-disciplined GPU usage — while the generalist-VLA choice keeps the intelligence in one instruction-conditioned model.

**Roadmap.** Round 1 establishes a verified π0 baseline score. Next: task-weighted fine-tuning on the GOAI-2026 dataset targeting the weakest tasks in the score breakdown (iterating within the 3-submission rule), a same-harness comparison of open VLA families (GR00T N1.7, OpenVLA-OFT), and sim2real preparation for the on-site finals building on our open-source single-arm sim2real project (`so101-sim2real`, PyPI-published).

**Openness.** Every component on the critical path is open-source (openpi, RoboDojo, XPolicyLab); our deployment scripts, self-test client, and operational runbook will be published on GitHub under MIT.

*(~420 words)*
