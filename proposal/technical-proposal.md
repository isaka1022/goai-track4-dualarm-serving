# Technical Proposal — GOAI 2026 Track 4 "Embodied Future" (Dual-Arm)

**Team**: isaka1022 (solo, Japan) · **Contact**: isaka1022@gmail.com
**Entry**: Generalist VLA policy serving for dual-arm manipulation — π0 on the RoboDojo/X-Eval benchmark

---

## 1. Task Definition

The preliminary round evaluates dual-arm manipulation across the 12 GOAI-2026 simulation tasks (e.g., Stack Bowls, Fold Clothes, Pour Liquid, Hang Mugs) on the ARX X5 dual-arm platform, built on the RoboDojo benchmark (Isaac Sim). The evaluation is **server-based**: the organizer runs the simulation client, which connects to the participant's publicly reachable policy server and exchanges observations for actions over WebSocket. The policy operates in **joint space with 14 DoF** (6 arm joints + 1 gripper, ×2 arms), receiving multi-view RGB observations (`cam_high`, `cam_left_wrist`, `cam_right_wrist`), the 14-dim proprioceptive state, and a per-step natural-language instruction.

Our objective for Round 1: field a **single generalist policy** that handles all 12 tasks through instruction conditioning — no per-task checkpoint switching, no task-ID side channels — served on infrastructure verified against the official evaluation protocol.

## 2. System Architecture

```
[Organizer: Isaac Sim client]
        │  ws:// (msgpack frames)
        │  reset / update_obs / get_action RPC
        ▼
[Public TCP endpoint  host:port]
        ▼
[Policy server (websockets, Python 3.11)]
        │  obs = {images ×3, state(14), instruction}
        ▼
[π0 generalist checkpoint  (openpi, JAX, bf16)]
        │
        └─ action chunk: 50 steps × 14-dim joint targets
```

Design decisions:

- **One generalist checkpoint for all 12 tasks.** π0 consumes the per-step `instruction` string as its language prompt, so a single set of weights covers the full task suite. This eliminates task-routing complexity (no router to mispredict, no N× GPU memory) and mirrors how a deployable dual-arm system should work: one policy, told what to do in natural language.
- **Protocol-exact serving.** The server implements the official RoboDojo evaluation RPC (`reset` → `update_obs` → `get_action`, msgpack over plain WebSocket) and was verified with the *official client class* (`WsModelClient`) — the same code path the organizer's evaluator uses (§5).
- **Action chunking.** Each `get_action` returns a 50-step chunk of 14-dim joint targets, amortizing inference latency (~one forward pass per 50 control steps) and keeping the control loop stable over the public network.

## 3. Core Algorithm

The policy is **π0** (Physical Intelligence, released as open-source `openpi`): a ~3B-parameter vision-language-action model combining a PaliGemma VLM backbone with a **flow-matching action expert**. Key properties for this challenge:

- **Instruction-conditioned multi-task control.** Language conditioning is native, not bolted on — the same weights execute "stack the bowls" and "fold the clothes" by prompt alone. This is the property that makes the 12-task suite tractable with one model.
- **Continuous action generation via flow matching**, well-suited to smooth dual-arm joint control at high frequency (vs. discretized-token action decoding).
- **Bimanual pedigree.** π0's training recipe includes bimanual platforms (ALOHA-style, 14-DoF), matching the ARX X5 action space exactly.
- **Fully open.** Model code and weights are Apache-2.0 (`openpi`); the benchmark stack (RoboDojo, XPolicyLab) is open-source. No closed-source component is on the critical path (compliant with the open-source model requirement).

For Round 1 we serve the official π0 generalist checkpoint for the ARX X5 sim suite (`RoboDojo-sim-arx_x5-joint`, trained on the multi-task sim mixture), establishing a verified baseline score. Our improvement roadmap (§6) builds on this via task-weighted fine-tuning.

## 4. Deployment

- **Hardware**: single RTX 4090 (24 GB) cloud instance; π0 inference in bf16 occupies **7.6 GB VRAM** — comfortable headroom, no quantization needed.
- **Two-tier storage.** A persistent network volume holds the immutable assets (12 GB checkpoint, Python toolchain, repos); the runtime (venv + checkpoint copy) lives on local NVMe. Rationale: distributed FUSE filesystems are pathologically slow for Python's many-small-file access patterns (venv imports, Orbax checkpoint shards) — we measured venv builds going from minutes on the network volume to ~30 s locally. The split gives restart-durability *and* fast, reliable inference.
- **Self-healing boot.** A supervisor script (`boot.sh`) rebuilds the local runtime from the network volume if absent (e.g., after instance restart) and keeps the server under an auto-restart loop. Recovery from a cold instance to a serving endpoint is a single command.
- **Endpoint**: plain `ws://` on a public TCP port (matching the evaluator's `websockets.connect(ws://host:port)` connection form). The endpoint is brought up for evaluation windows and stopped otherwise (cost hygiene).

## 5. Verification Evidence (self-test, 2026-08-16)

Per the requirement for verifiable evidence, we validated the full evaluation path end-to-end before requesting official evaluation:

1. **External reachability**: WebSocket handshake from an external network returns `HTTP 101 Switching Protocols` (`Server: websockets/15.0.1`).
2. **Protocol round-trip with the official client.** Using the benchmark's own `WsModelClient` (msgpack frames; the identical class the organizer's `deploy.py` evaluation loop instantiates), we executed `reset` → `update_obs` → `get_action` against the live server:

```
get_action() returned: list, length (chunk size) = 50
  each step: {left_arm_joint_state (6,), left_ee_joint_state (1,),
              right_arm_joint_state (6,), right_ee_joint_state (1,)}
  concatenated per-step dim = 14, all values finite = True
RESULT: PASS — valid 14-dim action chunk (chunk_size=50)
```

3. **Observation schema conformance**: the server accepts the evaluator's obs layout — three RGB camera images, 14-dim state vector, and the natural-language `instruction` string consumed as the π0 prompt.

An official X-Eval platform score will be attached as soon as the evaluation application is processed (application submitted via the organizer's channel; as an overseas team we are coordinating with the organizers on the registration flow).

## 6. Roadmap (Finals)

1. **Task-weighted fine-tuning** of π0 on the GOAI-2026 dataset, upweighting tasks with the lowest per-task success in the Round-1 score breakdown (the 3-submission/best-score rule enables measure → fine-tune → resubmit iteration).
2. **Model comparison** under the identical serving harness: GR00T N1.7 and OpenVLA-OFT are drop-in candidates (same instruction-conditioned generalist interface), enabling an apples-to-apples benchmark of open VLA families on dual-arm manipulation.
3. **Sim2real transfer** for the on-site finals: the serving architecture is simulator-agnostic (the policy sees only images + state + instruction), and our prior single-arm sim2real work (open-source `so101-sim2real`, PyPI-published) directly informs the domain-adaptation plan.

## 7. Open-Source Statement

- π0 / `openpi`: Apache-2.0 (Physical Intelligence) — model code and weights open.
- RoboDojo benchmark & XPolicyLab baselines: open-source (benchmark organizers).
- **Our contribution** — deployment scripts (`boot.sh`, local-runtime rebuild, supervisor), self-test client, and the operational runbook — will be published on GitHub under MIT upon submission.
- No closed-source model or API is used in the policy's critical path.
