"""Round-trip self-test against a running Pi_0 policy server.

Uses the benchmark's own ``WsModelClient`` (msgpack frames over WebSocket) —
the same client class the official evaluation loop (``deploy.py``) instantiates
— to verify that a ``reset -> update_obs -> get_action`` cycle returns a valid
14-dim action chunk.

Run on the serving host:

    cd /root/XPolicyLab && PYTHONPATH=/root/XPolicyLab \
      /root/XPolicyLab/policy/Pi_0/openpi/.venv/bin/python selftest_pi0.py

Expected result (verified 2026-08-16, see selftest/PASS.log):
    chunk of 50 steps, each with left/right arm (6,) + ee (1,) joint states,
    concatenated per-step dim = 14, all values finite.
"""

import numpy as np

from client_server.ws.model_client import WsModelClient

HOST = "127.0.0.1"
PORT = 9999

# Dummy observation matching the evaluator's schema (Pi_0/model.py encode_obs
# fast path): three RGB images, 14-dim state, natural-language instruction.
OBS = {
    "images": {
        "cam_high": np.zeros((224, 224, 3), dtype=np.uint8),
        "cam_left_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
        "cam_right_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
    },
    "state": np.zeros(14, dtype=np.float32),
    "instruction": "stack the bowls",
}


def main() -> None:
    print(f"connecting to ws://{HOST}:{PORT} ...")
    client = WsModelClient(host=HOST, port=PORT)

    print("--- reset ---")
    print("reset() ->", client.call(func_name="reset"))

    print("--- update_obs ---")
    print("update_obs() ->", client.call(func_name="update_obs", obs=OBS))

    print("--- get_action ---")
    actions = client.call(func_name="get_action")
    print(f"get_action() returned type={type(actions)}")
    print(f"  list length (chunk size) = {len(actions)}")
    first = actions[0]
    print(f"  first element keys={list(first.keys())}")
    total_dim = 0
    finite = True
    for key, vec in first.items():
        arr = np.asarray(vec)
        total_dim += arr.size
        finite = finite and bool(np.isfinite(arr).all())
        print(f"    {key}: shape={arr.shape} values={arr}")
    print(f"  concatenated per-step dim = {total_dim}")
    print(f"  all values finite = {finite}")

    ok = total_dim == 14 and finite and len(actions) > 0
    print("\n=== RESULT ===")
    print(
        f"{'PASS' if ok else 'FAIL'}: "
        f"{'got a valid 14-dim action chunk' if ok else 'unexpected action layout'} "
        f"(chunk_size={len(actions)})"
    )


if __name__ == "__main__":
    main()
