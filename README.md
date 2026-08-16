# GOAI 2026 Track 4 — Dual-Arm Generalist VLA Serving

Deployment scripts, self-test client, and operational runbook for our
**GOAI 2026 Track 4 "Embodied Future" (dual-arm)** entry: a single generalist
**π0** policy served over the official RoboDojo WebSocket evaluation protocol.

Team: `isaka1022` (solo) · Proposal: [`proposal/`](proposal/)

## Architecture

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
[π0 generalist checkpoint  (openpi, JAX, bf16, 7.6 GB VRAM)]
        │
        └─ action chunk: 50 steps × 14-dim joint targets
```

One instruction-conditioned checkpoint (`RoboDojo-sim-arx_x5-joint-0`) covers
all 12 evaluation tasks — no per-task checkpoint switching. See the
[technical proposal](proposal/technical-proposal.md) for design rationale.

## Repository layout

| Path | What it is |
|---|---|
| `scripts/boot.sh` | Supervisor entry point: rebuilds the runtime if missing, then keeps the server under an auto-restart loop. One command from cold instance to serving. |
| `scripts/local_rebuild.sh` | Rebuilds the fast **local** runtime (venv + checkpoint) from the persistent network volume. |
| `scripts/run_serve.sh` | Starts the Pi_0 policy server (XPolicyLab `setup_eval_policy_server.sh`, joint action space, port 9999). |
| `selftest/selftest_pi0.py` | Round-trip self-test using the benchmark's own `WsModelClient` — the identical client class the official evaluation loop uses. |
| `selftest/PASS.log` | Captured output of the self-test against the live server (2026-08-16). |
| `docs/RUNBOOK.ja.md` | Operational runbook (Japanese): decisions, incidents, and verified facts from bring-up. |
| `proposal/` | Technical proposal & project summary (markdown sources of the submitted PDFs). |

## Deployment notes (RunPod, RTX 4090)

- **Two-tier storage.** The persistent network volume (`/workspace`) holds
  immutable assets: the 12 GB checkpoint, miniforge, repos, `env_cfg`. The
  runtime (uv venv + checkpoint copy) lives on **local NVMe** (`/root`).
  FUSE-backed network filesystems are pathologically slow for Python's
  many-small-file access patterns — venv builds went from minutes (network
  volume) to ~30 s (local disk).
- **Restart-safe.** After any pod (re)start:
  `setsid bash /workspace/boot.sh > /workspace/boot.log 2>&1 &`
- **Endpoint**: plain `ws://` on an exposed TCP port (the evaluator connects
  with `websockets.connect(ws://host:port)`).

## Self-test

```bash
cd /root/XPolicyLab && PYTHONPATH=/root/XPolicyLab \
  /root/XPolicyLab/policy/Pi_0/openpi/.venv/bin/python selftest_pi0.py
```

Verified result: `reset → update_obs → get_action` returns a 50-step chunk of
finite 14-dim joint actions (6 + 1 per arm × 2). Full log: `selftest/PASS.log`.

## Upstream projects

- [openpi](https://github.com/Physical-Intelligence/openpi) (π0, Apache-2.0)
- [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) benchmark
- [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) baselines

## License

MIT (this repository's scripts and docs). Upstream projects keep their own licenses.
