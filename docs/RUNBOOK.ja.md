# GOAI Track4 — Stage1 Runbook（Isaac-free serve-first）

一次ソース（RoboDojo論文 arXiv:2607.04434 / GitHub実ファイル / robodojo-benchmark.com/doc）で確定した事実に基づく。

## 核心事実（2026-08-13 確定）

- **採点は運営が Isaac Sim を client として回し、参加者の公開 policy server（推論のみ）に obs→action を要求する**（WebSocket/TCP split）。参加者の箱に Isaac Sim は不要
- **ACT 学習 = PyTorch + HDF5 のみ**（`policy/ACT/conda_env.yaml` は h5py/mujoco/torch のみ、Isaac 非依存）
- **policy server は推論のみ**（`client_server/*/model_server.py` は isaac/omni/curobo を import しない）
- **公式 ACT checkpoint が配布**（`scripts/RoboDojo/download_ckpt.sh huggingface ACT`）→ 学習スキップ可
- Isaac Sim が要るのは **ローカル smoke/eval/client（自己確認）だけ**。xsparkai がこれを必須ゲートにするかは**未確認**（要 xsparkai 申請フロー確認）

## 箱のサイズ（capacity 問題の解消）

- **GPU**: 任意の decent GPU で可。将来ローカル smoke もしたいなら RT-core（RTX A4000/A5000/A4500/3090/4090）を選ぶと潰しが効く。ACT は小モデルなので 24GB で十分
- **ディスク**: **~50GB**（checkpoint 数百MB＋1タスク分 HDF5 数GB）。Isaac Assets 35GB は serve/train には不要
- 100GB・RT必須は「ローカル smoke を回す時だけ」の要件。まずは serve-first で軽く借りる

---

## 実機検証済み手順（2026-08-14 / RunPod RTX3090 Community pod）

**駆動方法（確定）**: RunPod proxy SSH（ssh.runpod.io）は対話専用で自動化不可。
ポッド上で `bore`（アカウント不要TCPトンネル）を立て、本物の sshd を公開して
Mac から直 SSH で駆動する。**このトンネルは serve 公開にも再利用する**。

```bash
# pod 上（対話SSHで1回だけ）
cd /workspace
U=$(curl -sL https://api.github.com/repos/ekzhang/bore/releases/latest | grep -o 'https://[^"]*x86_64-unknown-linux-musl.tar.gz' | head -1)
curl -sL "$U" -o bore.tgz && tar xzf bore.tgz
nohup ./bore local 22 --to bore.pub > /workspace/bore.log 2>&1 &
cat /workspace/bore.log   # → listening at bore.pub:<PORT>
# Mac 側: ssh root@bore.pub -p <PORT> -i ~/.ssh/id_rsa '<cmd>'  で全駆動
```

**環境（確定）**: pod は Python 3.12 / torch 2.8 / driver 580 だが、ACT 参照環境は
**Python 3.9**（`conda_env.yaml`）。3.12 に pip で古い mujoco 等は入らない → **conda 必須**。

```bash
# 1. XPolicyLab は RoboDojo のサブモジュール。これだけ init（Isaac 系 third_party/* はskip）
cd /workspace/RoboDojo && git submodule update --init --depth 1 XPolicyLab

# 2. Miniforge 導入（mamba）
curl -sL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o mf.sh
bash mf.sh -b -p /workspace/miniforge3

# 3. act 環境作成（conda_env.yaml: py3.9/pytorch2.0/cuda11.8/mujoco2.3.3）
/workspace/miniforge3/bin/mamba env create -f /workspace/RoboDojo/XPolicyLab/policy/ACT/conda_env.yaml

# 4. editable install（install.sh の pip torch 行はconda済みなので流用せず、editableのみ）
cd /workspace/RoboDojo/XPolicyLab/policy/ACT/detr && /workspace/miniforge3/envs/act/bin/pip install -e .
cd /workspace/RoboDojo/XPolicyLab && /workspace/miniforge3/envs/act/bin/pip install -e .

# 5. checkpoint 取得（HF RoboDojo-Benchmark/RoboDojo → XPolicyLab/policy/ACT/checkpoints）
bash /workspace/RoboDojo/scripts/RoboDojo/download_ckpt.sh huggingface ACT
```

確定した実パス:
- ACT: `/workspace/RoboDojo/XPolicyLab/policy/ACT/`（install.sh は pip、conda_env.yaml は py3.9）
- ckpt DL: `/workspace/RoboDojo/scripts/RoboDojo/download_ckpt.sh <hf|modelscope> <POLICY>`
- serve: `XPolicyLab/policy/ACT/setup_eval_policy_server.sh` / `RoboDojo/scripts/robodojo.sh server`

### serve までの確定手順と罠（実機で serve 成功・2026-08-14）

```bash
# A. Python は 3.10 で env 作成（XPolicyLab が py>=3.10 必須。conda_env.yaml の 3.9 を書換）
sed 's/python=3.9/python=3.10/' conda_env.yaml > act310.yaml
CONDA_PKGS_DIRS=/root/condapkgs mamba env create -f act310.yaml -p /root/act_env  # env は container disk(/)へ

# B. editable install 後、numpy を必ず <2 に戻す（XPolicyLab が numpy2 を引き h5py が binary不整合で死ぬ）
/root/act_env/bin/pip install -e .../ACT/detr && /root/act_env/bin/pip install -e .../XPolicyLab
/root/act_env/bin/pip install "numpy<2"

# C. checkpoint は全35タスク=16GB。1タスクだけ sparse+lfs で引く（volume 30GB では全部入らない）
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --sparse https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo repo
cd repo && git sparse-checkout set ckpt/RoboDojo/ACT/act-RoboDojo-stack_bowls
git lfs pull --include "ckpt/RoboDojo/ACT/act-RoboDojo-stack_bowls/**"
ln -sfn <cache>/ckpt/RoboDojo/ACT  .../ACT/checkpoints   # download_ckpt.sh 同様の symlink

# D. コードは policy_last.ckpt を要求。実体は policy_epoch_6000_seed_0.ckpt → symlink
ln -sfn policy_epoch_6000_seed_0.ckpt <ckptdir>/arx_x5-100-joint/policy_last.ckpt

# E. serve 起動（conda を PATH に。setsid でデーモン化しないと SSH チャネルがハング）
cd .../ACT && PATH=/workspace/miniforge3/bin:$PATH bash setup_eval_policy_server.sh \
  RoboDojo stack_bowls act-RoboDojo-stack_bowls/arx_x5-100-joint arx_x5 joint 0 0 /root/act_env 9999 0.0.0.0
```

**確定事実**: action_dim=14（双腕2腕×7関節）/ protocol=ws / checkpoint は6000epoch3seed。
**公開**: `bore local 9999 --to bore.pub` → `bore.pub:<PORT>`。TCP到達＋WSハンドシェイク(HTTP101)で検証済み。
**残課題**: この server は stack_bowls 1タスク固定。bore.pub 無料鯖は採点中に切れるリスク（安定運用は Secure public IP 検討）。

### 多タスク化の方針（2026-08-14 確定）: ②-B 汎用ポリシー1本

**プロトコル調査（公開ソース file:line 裏取り）で確定した事実:**
- 評測 client は obs に **`instruction`（自然言語）を毎 `update_obs`/`infer` で送る**（`RoboDojo/env/observation_manager/obs_manager.py:89`）。**汎用VLA（Pi_0/GR00T/OpenVLA）はこれを読みタスクを切替**（`Pi_0/model.py:169`, `GR00T_N17/model.py:224`, `OpenVLA_OFT/model.py:324`）
- `trial_id`/`action_case_id` も全フレームに乗る（`schemas.py:32-49`）が**参照サーバは routing に未使用**。参照サーバは起動時に**1 checkpoint 固定ロード**（`setup_policy_server.py:22-32`）。公式のタスク別ルーティング機構は**無い**（`prepare_case` は harness 未呼び出し）
- → **ACT per-task ルータ自作（②-A）はコード非支持で茨。汎用1本（②-B）が正規の形**でフォーム設計（策略配置1つ・1エンドポイント全タスク）とも一致

**checkpoint 配布の確認（HF `RoboDojo-Benchmark/RoboDojo` tree）:**
- ACT / SmolVLA = `RoboDojo-<taskname>-...` ＝**タスク別**（1本では全タスク不可）
- **Pi_0 / Pi_05 / GR00T_N17 = `RoboDojo-sim-arx_x5-joint-{0,1,2}` ＝汎用1本**（`joint-N` は **seed N**、`joint` は action_type）。π0/GR00T は ~3B で RTX3090 24GB に収まる

**本命 = Pi_0（π0）**: 最も枯れた openpi 実装・事故率最小。policy を ACT→Pi_0、ckpt を `RoboDojo-sim`(seed0)。依存は **openpi(JAX) を uv で導入**（`Pi_0/install.sh`＝`uv sync --group lerobot`）。

## 安定エンドポイント構成（2026-08-15 確定・bore.pub 廃止）

**bore.pub 無料は採点で切れる → 廃止**。RunPod ネイティブの生TCP公開に移行。
- **Network Volume**（EU-RO-1・60GB・/workspace にマウント）＝再起動・再デプロイでも構築物が消えない（container disk `/root` は ephemeral なので必ず /workspace 側へ）
- **Secure Cloud pod（EU-RO-1・RTX4090・$0.74/hr）に exposed TCP 22＋9999** → RunPod が public IP:外部ポートを割当
  - 駆動: `ssh root@<public_ip> -p <22の外部ポート>`（**生TCP・非対話OK＝bore不要**）
  - **policy 公開エンドポイント = `<public_ip>:<9999の外部ポート>`**（Connect パネル「Direct TCP ports」の `…:内部9999`）。xsparkai フォームへ host=public_ip / port=9999外部ポート / action_type=joint で登録
  - 根拠: 運営 client は `websockets.connect(url)`。フォームが host+port 分離で scheme 無し＝`ws://host:port` 平文想定 → 生TCP必須（HTTP プロキシ443/wss では不整合）

**再構築スクリプト**（`/workspace/rebuild.sh`・空 volume から全自動）: ①git-lfs ②XPolicyLab clone ③Miniforge+pyyaml（`conda info --base` が要る）④env_cfg を RoboDojo から sparse 取得→`/workspace/env_cfg` ⑤`Pi_0/install.sh`（openpi uv）⑥π0 checkpoint を **params+assets+metadata のみ** sparse+lfs（`--no-cone` で train_state 除外＝29GB回避）→ `policy/Pi_0/checkpoints/RoboDojo-sim-arx_x5-joint-0/60000/`。

**serve 起動**: `cd policy/Pi_0 && PATH=/workspace/miniforge3/bin:$PATH bash setup_eval_policy_server.sh RoboDojo stack_bowls sim arx_x5 joint 0 0 uv 9999 0.0.0.0`（ckpt_name=`sim`・uv env・supervisor で自動再起動）。

## ✅ 動作確認済み（2026-08-16・π0 汎用1本・外部到達検証済み）

**MooseFS は Python ランタイムに致命的に遅い（FUSE 小ファイル遅延で import が数分ハング）→ ランタイムをローカル disk で動かすのが根治。**

- pod: RTX4090・EU-RO-1・**container disk 60G（ローカル）＋ Network Volume 60G（/workspace・永続）**・exposed TCP 22/9999
- Network Volume は「永続ソース」: checkpoint 12G・env_cfg・miniforge3(+pyyaml)・repo を保持 → 再デプロイでも再DL不要
- **ローカル rebuild**（`/workspace/local_rebuild.sh`）: repo をローカルに fresh clone・env_cfg コピー・**openpi venv をローカルで uv sync（`Prepared 221 packages in 30s`＝MooseFS の数分→秒）**・checkpoint を /workspace から bulk コピー
- **再起動耐性**（`/workspace/boot.sh`）: ランタイムが消えていれば自動再構築 → supervisor ループで serve 自動再起動。pod 停止→再開しても `setsid bash /workspace/boot.sh >log 2>&1 &` 1本で復旧
- **実証**: `LISTEN 0.0.0.0:9999`・GPU 7.6GB（π0 ロード）・Mac から `ws://<public_ip>:<外部ポート>` に **HTTP 101 Switching Protocols**（`Server: websockets/15.0.1`）＝運営 client の接続形と一致
- **xsparkai 登録値**: host=公開IP / port=9999の外部ポート（Connect「Direct TCP ports」の `…:9999`）/ action_type=**joint** / 策略名=任意

**コスト規律**: GPU pod はアイドル放置で溶ける（4090 $0.74/hr で一晩=$10）。作業合間は必ず Stop。採点は「审核通过后开始评测（審査通過後）／提出後すぐ GPU を使わない」明記 → 提出後は Stop し、eval 直前に boot.sh で起こす運用が最安。

---

## Path A（最速・推奨）: checkpoint を serve して即スコア

### A0. box 準備
```bash
ssh <id>@ssh.runpod.io -i ~/.ssh/id_rsa
nvidia-smi; df -h /
git lfs version || (apt-get update && apt-get install -y git-lfs && git lfs install)
```

### A1. リポジトリ取得（Isaac をスキップ）
```bash
cd /workspace
git clone --recursive https://github.com/RoboDojo-Benchmark/RoboDojo.git
git clone https://github.com/XPolicyLab/XPolicyLab.git
cd RoboDojo
# ACT の Isaac-free 環境を作る（policy 側の専用 installer を使う）
bash XPolicyLab/policy/ACT/install.sh   # conda_env.yaml: torch+h5py+mujoco、Isaac無し
# 【要確認】install.sh のパスと conda env 名（`act` 想定）
```

### A2. 公式 ACT checkpoint を取得（学習スキップ）
```bash
cd /workspace/RoboDojo
bash scripts/RoboDojo/download_ckpt.sh huggingface ACT
# → checkpoint 保存先パスを控える（serve の --ckpt に渡す）
```

### A3. policy server を起動（推論のみ・Isaac 不要）
```bash
# robodojo.sh server は sim client を立てず policy server だけ起動する
bash scripts/robodojo.sh server \
  --policy-dir XPolicyLab/policy/ACT --ckpt <DOWNLOADED_CKPT> \
  --policy-env act --env-cfg arx_x5 --action-type joint \
  --policy-port 9999 --bind-host 0.0.0.0
# 代替: XPolicyLab/policy/ACT/setup_eval_policy_server.sh（完全 Isaac-free 経路）
# 【要確認】server が isaac env を source しないか（doctor --skip-isaac で健全性確認可）
```

### A4. エンドポイント公開
- RunPod の TCP Port Mapping で 9999 を公開 → 外部到達 `host:port` を取得
- 【要確認】WebSocket/TCP が RunPod proxy を通るか。通らなければ直 IP:port か SSH リバーストンネル

### A5. 申請（ユーザー操作）— フォーム全項目を実機確認済み（2026-08-14）

`xsparkai.com/goai-2026/apply` の「仿真评测申请」フォーム項目（**これが全部**）:

1. **队伍与联系人**: 队伍名称* / 联系人* / 手机号* / 邮箱*（＝チーム名・担当者・電話・メール。全て個人情報 → **ユーザー本人が入力・送信**）
2. **Policy Server 端点**: **主机（host, placeholder `example.com`）＋ 端口（port, placeholder `9000`）を別欄で入力**。「添加」で複数エンドポイント追加可。→ **URL ではなく host と port を分けて入れる**。我々の場合 host=`bore.pub` / port=公開ポート番号
3. **策略配置**: 策略名（placeholder `my_policy`）＋ 动作类型（ドロップダウン: 末端位姿(ee) / 关节(joint)）→ 我々は **joint**
4. 提交申请（送信ボタン）

**確定した2つの問い:**
- **smoke test は必須ゲートではない**。フォームに self-test 提出欄・Isaac ローカル検証の証跡欄は一切無い。ページ冒頭に「审核通过后开始评测，不会在提交后立刻占用 GPU」＝**審査通過後に運営側が評測を開始**（提出＝即課金ではない）。→ RUNBOOK の核心仮説（参加者は policy serve のみ・Isaac ローカル不要）を**フォームが裏付け**
- **エンドポイントは host＋port 分離**。bore トンネルの `bore.pub` + ポート番号をそのまま入れられる

運営が評価 → スコアがメール → goaihz でコード＋メール提出（締切 **8/20 AOE**）

---

## Path B（任意）: 自前学習 or ローカル smoke

### B1. 1タスク学習（checkpoint に不満なら）
```bash
hf download RoboDojo-Benchmark/GOAI-2026 --repo-type dataset \
  --include "data/hdf5/stack_bowls/**" --local-dir ./data_root
# 罠: DL実体 data_root/data/hdf5/... を bench_name=RoboDojo に橋渡し
mkdir -p data && ln -s "$(pwd)/data_root/data/hdf5" "$(pwd)/data/RoboDojo"
bash XPolicyLab/policy/ACT/process_data.sh RoboDojo stack_bowls arx_x5 joint
bash XPolicyLab/policy/ACT/train.sh RoboDojo stack_bowls arx_x5 joint 0 0
# 6000epoch固定・中間ckpt無し（4090で4〜8h）。まず epochs=500 で経路確認【要確認: epoch指定】
```

### B2. ローカル smoke（Isaac Sim が必要な唯一の作業）
- **これをやる時だけ**フル Isaac Sim 箱（RT-core・100GB・CUDA12.8）が要る
```bash
# フルインストール（Isaac Sim 5.1 + Isaac Lab 2.3 + CuRobo）
bash scripts/install.sh                    # --from <step> で途中再開可
bash scripts/robodojo.sh smoke \
  --policy-dir XPolicyLab/policy/ACT --ckpt <CKPT> \
  --policy-env act --env-cfg arx_x5 --action-type joint \
  --only stack_bowls --fail-fast
```

---

## 未確認（box/申請で最初に潰す）

- [ ] xsparkai 申請に **smoke test が必須ゲート**か（→ 必須なら B2 用に Isaac 箱を別途）
- [ ] `robodojo.sh server` が isaac env を source しないか（`doctor --skip-isaac` で確認）
- [ ] policy server は WS か TCP か、RunPod proxy 越しに到達するか
- [ ] `download_ckpt.sh` の ACT checkpoint 保存先パスと serve への渡し方
- [ ] `arx_x5` が双腕 embodiment の正しい指定か

## コスト管理

- serve/train 用の軽い箱なら $0.2〜0.34/hr。$10 で 30h+
- 使わない時は pod を **Stop**（/workspace 保持）。ローカル smoke 用の重い箱は smoke の時だけ起こす

## 自己テスト PASS（2026-08-16・実測）

送信前ゲートとして、Pod 上で運営と同一の `WsModelClient`（`deploy.py` 実物）で往復を実測：
- プロトコル: msgpack frames over plain WS（`client_server/ws/`）。`reset → update_obs → get_action` が運営 client と 1:1
- 結果: `get_action()` が **50step×14次元**の action チャンク（左腕6+左EE1+右腕6+右EE1）、全値有限。サーバ無傷・9999 LISTEN 継続
- obs 高速パス: `obs["images"]{cam_high,cam_left_wrist,cam_right_wrist}` + `obs["state"](14)` + optional `obs["instruction"]`（`policy/Pi_0/model.py:162-171`）
- テストスクリプト: Pod `/root/selftest_pi0.py`（standalone・serve プロセス kill せず）
→ ハンドシェイクでなく obs→action の実往復まで検証済み。送信して評価に晒す準備完了。

## 現在の状態（2026-08-16 停止時点）

- **Pod 停止済み**: `enormous_lime_dove`（id `0ib2hwmlgs7umi`・RTX 4090・EU-RO-1）。Stop 実行 → **Idle disk cost $0.000/hr**（container disk はローカル、network volume は別枠課金）・エンドポイント UNREACHABLE 確認・残高 $9.41 保全
- **永続ソース健在**: `/workspace`（network volume 60G）に checkpoint 12G・miniforge3・repo・env_cfg。再開は Pod Start → `setsid bash /workspace/boot.sh >log 2>&1 &`。**再開で public IP:port は変わる**
- **再開時の注意**: Stop 後の再開は「空きマシン次第」（4090 availability 依存）。eval ウィンドウで確実に起こせるか要確認
- **登録ブロッカー（未解決）**: xsparkai 申請フォームの 手机号 が**中国本土番号必須**（`请填写有效的中国大陆手机号`）で海外チームは送信不可。EN 切替でもバリデーション不変。→ 運営に海外登録経路を問い合わせる（文面: `overseas-registration-inquiry.md`）。技術（endpoint＋自己テスト）は完了済み、詰まりは登録のみ
- 技術フィールド入力値の控え（再申請時に流用・host/port は再開後の最新値に差し替え）: 协议=ws / 策略名=pi0 / 动作类型=joint(关节) / 邮箱=isaka1022@gmail.com / 队伍名称=isaka1022
