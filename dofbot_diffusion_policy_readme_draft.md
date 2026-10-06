# DOFBOT Diffusion Policy with LeRobot / Isaac Sim / ROS 2

## 概要

Yahboom DOFBOTを対象に、Isaac Simで収集した軌道データからLeRobotのDiffusion Policyを学習し、ROS 2経由でIsaac Simおよび実機DOFBOTを制御する構成です。

現在の構成では画像は使用せず、以下をPolicy入力にしています。

```text
observation.state
  11 joint positions

observation.environment_state
  cube position [x, y]

action
  11 joint target positions
```

今回使用している主なDiffusion Policy設定:

```text
n_obs_steps    = 2
horizon        = 16
n_action_steps = 8
```

---

## 環境

| 項目 | 内容 |
|---|---|
| OS | Ubuntu 24.04 |
| ROS | ROS 2 Jazzy |
| Isaac Sim | 6.0.1 |
| GPU | RTX 4060 Ti |
| LeRobot | v0.5.1 |
| 実機 | Raspberry Pi 4 + Yahboom DOFBOT |
| 制御周期 | 10 Hz |

主要ディレクトリ:

```text
/home/natu/isaacsim
/home/natu/robot/lerobot
/home/natu/dofbot
/home/natu/dofbot_data
/home/natu/ros2_ws
```

Raspberry Pi側:

```text
~/dofbot_ws
~/ros2_jazzy
~/venvs/ros2_jazzy_py311
```

---

# 全体の流れ

```text
Isaac Sim
   ↓
raw episode data
   ↓
LeRobot datasetへ変換
   ↓
Diffusion Policy学習
   ↓
Isaac Simで推論確認
   ↓
ROS 2経由で実機DOFBOTへ適用
```

---

# 1. データ収集

Isaac Sim側のDOFBOT環境:

```text
/home/natu/isaacsim/src/dofbot/dofbot_gripper_imitation_env.py
```

rawデータ保存先:

```text
/home/natu/dofbot_data/diffusion_policy_raw
```

評価時に使用している箱位置:

```text
x = 0.20
y = -0.05
```

HOME姿勢:

```text
[0.8, 0.3, -0.5, 0.3, 0.0, 0, 0, 0, 0, 0, 0]
```

---

# 2. データセット変換

変換スクリプト:

```text
/home/natu/dofbot/scripts/convert_raw_to_lerobot.py
```

rawデータをLeRobot datasetへ変換します。

```bash
cd /home/natu/robot/lerobot

uv run python   /home/natu/dofbot/scripts/convert_raw_to_lerobot.py   --raw-root /home/natu/dofbot_data/diffusion_policy_raw   --output-root /home/natu/dofbot_data/diffusion_policy_train   --repo-id local/dofbot_diffusion_policy   --fps 10   --overwrite
```

変換後:

```text
/home/natu/dofbot_data/diffusion_policy_train
```

dataset repo id:

```text
local/dofbot_diffusion_policy
```

dataset feature:

```text
observation.state
  shape = 11

observation.environment_state
  shape = 2

action
  shape = 11
```

---

# 3. Diffusion Policy学習

学習コマンド:

```bash
cd /home/natu/robot/lerobot

RUN_DIR=/home/natu/dofbot_data/diffusion_policy_outputs/dofbot_dp_v1_$(date +%Y%m%d_%H%M%S)

uv run lerobot-train   --dataset.repo_id=local/dofbot_diffusion_policy   --dataset.root=/home/natu/dofbot_data/diffusion_policy_train   --policy.type=diffusion   --policy.device=cuda   --policy.push_to_hub=false   --output_dir="$RUN_DIR"   --job_name=dofbot_diffusion_policy_v1   --steps=50000   --batch_size=32   --num_workers=4   --log_freq=100   --save_freq=5000   --eval_freq=0   --wandb.enable=false   2>&1 | tee /home/natu/dofbot_data/diffusion_policy_train.log

echo "$RUN_DIR" | tee /home/natu/dofbot_data/latest_diffusion_run.txt
```

学習済みcheckpoint例:

```text
/home/natu/dofbot_data/diffusion_policy_outputs/
  dofbot_dp_v1_20260818_152011/
    checkpoints/050000/pretrained_model
```

---

# 4. Diffusion Policy設定

今回使用している設定:

```text
n_obs_steps    = 2
horizon        = 16
n_action_steps = 8
```

推論時は高速化のため、

```python
policy.diffusion.num_inference_steps = 10
```

を使用。

推論時間:

```text
100 inference steps
  約1.7秒 / chunk

10 inference steps
  約175～180 ms / chunk
```

10 stepsでもIsaac Simでは正常にピックできたため、現在は10を採用。

乱数seedは固定して比較している。

---

# 5. Isaac Simで動作確認

実機とのROS通信を分離するため、Isaac Simではdomain 10を使用。

```bash
export ROS_DOMAIN_ID=10
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

Isaac起動例:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

export ROS_DOMAIN_ID=10
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

cd /home/natu/isaacsim

./python.sh   src/dofbot/dofbot_gripper_imitation_env.py   --no-headless   --data-root /home/natu/dofbot_data/diffusion_policy_eval   --x-min 0.20   --x-max 0.20   --y-min -0.05   --y-max -0.05   --home-jitter-deg 0
```

リセット:

```bash
ros2 service call   /dofbot/reset_episode   dofbot_interfaces/srv/ResetEpisode   "{mode: 1, seed: 1000, x: 0.20, y: -0.05, record: false}"
```

READY確認:

```bash
ros2 topic echo --once /dofbot/episode_state
```

以下なら準備完了:

```text
phase: 2
```

---

# 6. Isaac SimでDiffusion Policy推論

推論スクリプト:

```text
/home/natu/dofbot/scripts/run_diffusion_real_dofbot.py
```

ROS interface:

```text
/joint_states
    ↓
Diffusion Policy
    ↓
/joint_command
```

実行:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

export ROS_DOMAIN_ID=10
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

cd /home/natu/robot/lerobot

uv run python   /home/natu/dofbot/scripts/run_diffusion_real_dofbot.py   --cube-x 0.20   --cube-y -0.05   --steps 200   --joint-state-timeout 5.0
```

Isaac Sim上では複数回ピック成功を確認済み。

---

# 7. Raspberry Pi側のDOFBOT driver

driver:

```text
~/dofbot_ws/src/dofbot_driver/dofbot_driver/dofbot_node.py
```

Pi側起動例:

```bash
source ~/venvs/ros2_jazzy_py311/bin/activate
source ~/ros2_jazzy/install/setup.bash
source ~/dofbot_ws/install/setup.bash

export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
unset ROS_LOCALHOST_ONLY

ros2 run dofbot_driver dofbot_node
```

driverで行っている主な変換:

```text
Isaac / Policy joint angle [rad]
    ↓
DOFBOT servo angle [deg]
```

arm1～arm5:

```text
servo_deg = 90 - rad2deg(q)
```

gripper:

```text
6 joint gripper
    ↓
Servo6 1軸
```

---

# 8. 実機HOME

HOME用script:

```text
~/bin/dofbot_home.sh
```

実行:

```bash
~/bin/dofbot_home.sh
```

---

# 9. 実機で動作確認

Ubuntu PC側:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
unset ROS_LOCALHOST_ONLY
```

HOME:

```bash
~/bin/dofbot_home.sh
```

推論:

```bash
cd /home/natu/robot/lerobot

uv run python   /home/natu/dofbot/scripts/run_diffusion_real_dofbot.py   --cube-x 0.20   --cube-y -0.05   --steps 200   --joint-state-timeout 5.0
```

---

# 10. Sim-to-Real補正

実機closed-loop推論では、そのままの `/joint_states` をPolicyへ入力すると途中で成功軌道から外れた。

A/B/C比較から、arm2/arm3に系統的biasが観測された。

```text
arm2:
Real - Isaac ≈ -0.0527 rad

arm3:
Real - Isaac ≈ -0.0284 rad
```

現在は暫定的に、Policy入力直前だけ以下の補正を入れている。

```python
policy_q = raw_q.copy()

policy_q[1] += 0.0527  # arm2
policy_q[2] += 0.0284  # arm3
```

重要:

- `/joint_states` 自体は変更しない
- `/joint_command` も変更しない
- Diffusion Policyが見る `observation.state` だけ補正

この補正を入れたところ、実機アームの軌道が正常化し、ピック位置まで到達した。

---

# 11. Gripper補正

Pi driver:

```python
GRIPPER_OPEN_DEG = 60.0
GRIPPER_CLOSED_DEG = 145.0
GRIPPER_Q_VALUE = 1.6
```

元々は:

```python
GRIPPER_CLOSED_DEG = 140.0
```

145°へ変更後、実機ピック成功を確認した。

---

# 12. A / B / Cによる動作検証

sim-to-real問題の切り分けとして以下を比較した。

| Case | 実行 | 結果 |
|---|---|---|
| A | Isaac + Diffusion Policy closed-loop | 成功 |
| B | 実機 + Diffusion Policy closed-loop | 失敗 |
| C | 実機 + Aの成功actionをopen-loop replay | 成功 |

bag保存先:

```text
A:
/home/natu/dofbot_data/dp_compare/isaac_seed1234_v2

B:
/home/natu/dofbot_data/dp_compare/real_seed1234

C:
/home/natu/dofbot_data/dp_compare/real_isaac_replay
```

bag記録例:

```bash
ros2 bag record   -o /home/natu/dofbot_data/dp_compare/real_seed1234   /joint_states   /joint_command
```

Isaac成功actionを実機へreplay:

```bash
ros2 bag play   /home/natu/dofbot_data/dp_compare/isaac_seed1234_v2   --topics /joint_command
```

Cでも実機ピックに成功したことから、

```text
成功action軌道そのものは実機で実行可能
```

であることを確認。

---

# 13. デバッグで分かったこと

Bではstep 24付近までは成功軌道に近いが、step 32付近からDiffusion Policyが成功軌道と逆方向へreplanningする。

解析例:

```text
B step 24
nearest C = 25
chunk cosine = +0.927

B step 32
nearest C = 34
chunk cosine = -0.364

B step 48
chunk cosine = -0.929
```

その後は成功軌道のstep 51付近に相当する状態へ張り付き、gripper closeまで進めなかった。

Policy入力のarm2/arm3 offset補正を入れると、この現象が解消した。

---

# 14. 解析用スクリプト

```text
/home/natu/dofbot/scripts/
```

代表的なもの:

```text
compare_dofbot_bags.py
compare_dp_policy_logs.py
analyze_dofbot_state_lag.py
analyze_b_vs_success_path.py
compare_dp_observation_history.py
```

---

# 15. GitHubに置く構成案

```text
dofbot-diffusion-policy/
├── README.md
├── scripts/
│   ├── convert_raw_to_lerobot.py
│   ├── run_diffusion_real_dofbot.py
│   └── dofbot_home.sh
│
├── ros2/
│   └── dofbot_driver/
│       └── dofbot_node.py
│
├── isaac/
│   └── dofbot_gripper_imitation_env.py
│
└── analysis/
    ├── compare_dofbot_bags.py
    ├── compare_dp_policy_logs.py
    ├── analyze_dofbot_state_lag.py
    ├── analyze_b_vs_success_path.py
    └── compare_dp_observation_history.py
```

---

# 現在の状態

現時点:

```text
Isaac Sim:
  Diffusion Policyでピック成功

実機:
  Policy入力にarm2/arm3 offset補正
  + gripper close 145°
  でピック成功
```

今後の主課題:

```text
policy_q[1] += 0.0527
policy_q[2] += 0.0284
```

という暫定補正について、

- 値の妥当性を測定する
- 実機 / Isaac間の正式なjoint calibrationとして整理する
- 推論側で補正するのか
- dataset / trainingへsim-to-real biasとして反映するのか

を決めること。
