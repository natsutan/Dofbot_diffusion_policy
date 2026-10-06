# DOFBOT Diffusion Policy

Yahboom DOFBOTを対象に、**Isaac Simでの教師データ収集 → LeRobotでの模倣学習 → ROS 2を介したシミュレーション・実機推論**を行うための実験用リポジトリです。

Diffusion Policyを中心に、ACTとSmolVLAの収集・推論スクリプト、URDF、Sim-to-Realの比較・解析ツールを含みます。主なタスクは、箱を把持して持ち上げることです。

- [ACTの評価動画](dofbot/act_eval.mp4)
- [既存のDiffusion Policy作業メモ](dofbot_diffusion_policy_readme_draft.md)

## 1. 構成と学習対象

Diffusion PolicyとACTの状態ベースの構成では、画像を使用せず、関節角度と箱の初期位置を入力します。SmolVLAでは、RGB画像・関節角度・言語指示を扱います。

| 手法 | 主な入力 | 出力 | 主な用途 |
| --- | --- | --- | --- |
| Diffusion Policy | 11関節の実測角度、箱の初期XY位置 | 11関節の目標角度 | シミュレーション・実機のピック動作 |
| ACT | 11関節の実測角度、箱の初期XY位置 | 11関節の目標角度 | シミュレーション評価、実機推論、フィードバックの切り分け |
| SmolVLA | RGB画像、11関節の実測角度、言語指示 | 11関節の目標角度 | 赤・青・緑の箱を対象とした言語条件付きピック |

処理の流れは次のとおりです。

1. Isaac Sim上で箱の配置と初期姿勢をリセットする。
2. ROS 2ノードがIKで教師軌道を生成し、ピック動作を実行する。
3. Isaac Simが観測・指令・成否をエピソード単位で保存する。
4. 成功エピソードをLeRobotデータセットに変換して学習する。
5. 学習済みPolicyの目標関節角度をROS 2でIsaac Simまたは実機へ送る。

### 状態・アクションの仕様

| Feature | 形状 | 内容・単位 |
| --- | --- | --- |
| `observation.state` | `(11,)` | 実測関節角度、rad |
| `observation.environment_state` | `(2,)` | 箱の初期位置 `[x, y]`、m。エピソード中は同じ値を使用 |
| `action` | `(11,)` | 各関節の目標角度、rad |

`action`は角度の差分ではなく、**絶対値の目標関節角度**です。箱のXY位置は既知の値として与えます。状態ベースの推論スクリプトには、カメラ画像から箱位置を推定する処理は含まれていません。

関節の順序は以下で統一しています。

```python
JOINT_NAMES = [
    "arm1_Joint",
    "arm2_Joint",
    "arm3_Joint",
    "arm4_Joint",
    "arm5_Joint",
    "Llink1_Joint",
    "Llink2_Joint",
    "Llink3_Joint",
    "Rlink1_Joint",
    "Rlink2_Joint",
    "Rlink3_Joint",
]
```

11関節はアーム5関節とグリッパのモデル上の6関節です。実機ではグリッパを1個のサーボで動かすため、実機ドライバ側で変換します。

## 2. 環境と事前準備

以下は[既存の作業メモ](dofbot_diffusion_policy_readme_draft.md)に記載された実験構成です。対応バージョンの一覧を示すものではありません。

| 項目 | 作業メモの構成 |
| --- | --- |
| 学習・推論PCのOS | Ubuntu 24.04 |
| ROS 2 | Jazzy |
| Isaac Sim | 6.0.1 |
| LeRobot | v0.5.1、Dataset v3 |
| GPU | RTX 4060 Ti |
| 実機 | Raspberry Pi 4 + Yahboom DOFBOT |
| データ記録・指令の基準周期 | 10 Hz |

Isaac Sim、ROS 2、LeRobot、学習データ、学習済みチェックポイントは別途用意してください。このリポジトリには、依存関係を一括導入するトップレベルの`requirements.txt`や`pyproject.toml`はありません。

主に使用する依存関係は以下です。

| 用途 | 依存関係 |
| --- | --- |
| ROS 2ワークスペース | `colcon`、`rosdep`、`rclpy`、`sensor_msgs`、`std_msgs`、NumPy、SciPyなど |
| 学習・Policy推論・データ変換 | LeRobot、PyTorch、NumPy。実行例ではLeRobotの`uv`環境を使用 |
| Isaac Sim環境 | Isaac Sim付属Python、URDF Importer、ROS 2 Bridge、ビルド済み`dofbot_interfaces` |
| rosbag解析 | `rosbag2_py`、`rosidl_runtime_py`、NumPy、Matplotlib |
| SmolVLAの動画保存・変換 | `ffmpeg`、`ffprobe`、LeRobotの画像・動画処理用依存関係 |

### リポジトリの取得とパスの設定

```bash
mkdir -p "$HOME/robot"
git clone https://github.com/natsutan/Dofbot_diffusion_policy.git \
  "$HOME/robot/Dofbot_diffusion_policy"
```

以降の例では次の変数を使用します。インストール先に合わせて変更し、使用する各端末で設定してください。

```bash
export DOFBOT_REPO="$HOME/robot/Dofbot_diffusion_policy"
export DOFBOT_DATA_ROOT="$HOME/dofbot_data"
export ISAAC_SIM_ROOT="$HOME/isaacsim"
export LEROBOT_ROOT="$HOME/robot/lerobot"
```

### 固定パスの変更

Isaac Simスクリプトには、元の環境に合わせた固定パスが残っています。実行するスクリプトの`URDF_PATH`と`USD_OUTPUT_ROOT`を変更してください。例：

```python
REPOSITORY_ROOT = Path("/absolute/path/to/Dofbot_diffusion_policy")
URDF_PATH = REPOSITORY_ROOT / "dofbot/urdf/dofbot.urdf"
USD_OUTPUT_ROOT = Path.home() / "dofbot_generated"
```

`URDF_PATH`は、対応するSTLがそろっている`dofbot/urdf/`のURDFを指定します。`dofbot_gripper_imitation_env.py`は`FORCE_URDF_REIMPORT = True`のため、起動時に既存の生成USDパッケージを削除して再生成します。`USD_OUTPUT_ROOT`には生成物専用の場所を指定してください。

ROS 2のIKノードは、ビルド・インストールされた`dofbot_description`のURDFを読みます。URDFを変更した場合は、Isaac Sim側とROS 2側の両方へ反映してください。

### ROS 2ワークスペースのビルド

```bash
source /opt/ros/jazzy/setup.bash
cd "$DOFBOT_REPO/ros2_ws"
rosdep install --from-paths src --ignore-src -r -y --rosdistro jazzy
colcon build --symlink-install
source "$DOFBOT_REPO/ros2_ws/install/setup.bash"
```

独自メッセージを使用するIsaac Sim・収集・評価用の端末でも、ROS 2とこのワークスペースをsourceしてください。LeRobot環境でROS 2推論を行う場合は、同じPython環境で`rclpy`を読み込めることが必要です。

```bash
cd "$LEROBOT_ROOT"
uv run python -c 'import rclpy, torch, lerobot; from dofbot_interfaces.srv import ResetEpisode'
```

## 3. ディレクトリと主要スクリプト

| パス | 内容 |
| --- | --- |
| [`issac_sim/dofbot/`](issac_sim/dofbot/) | Isaac Sim環境。ディレクトリ名は`issac_sim` |
| [`ros2_ws/src/dofbot_control/`](ros2_ws/src/dofbot_control/) | IKによるピック制御、教師エピソード収集 |
| [`ros2_ws/src/dofbot_interfaces/`](ros2_ws/src/dofbot_interfaces/) | エピソード状態メッセージ、リセット・終了サービス |
| [`ros2_ws/src/dofbot_description/`](ros2_ws/src/dofbot_description/) | ROS 2用のURDF・ロボット記述 |
| [`dofbot/urdf/`](dofbot/urdf/) | Isaac Sim用のURDFとSTL |
| [`dofbot/generated/`](dofbot/generated/) | 同梱された生成USD |
| [`dofbot/scripts/`](dofbot/scripts/) | LeRobot変換、Policy推論、ログ・rosbag解析 |
| [`dofbot/`](dofbot/) | ACTの評価動画・ログなど |

| スクリプト | 役割 |
| --- | --- |
| [`dofbot_gripper_imitation_env.py`](issac_sim/dofbot/dofbot_gripper_imitation_env.py) | 単一の箱を対象とした状態ベースの収集・評価環境 |
| [`dofbot_gripper_ros2.py`](issac_sim/dofbot/dofbot_gripper_ros2.py) | ROS 2関節指令で動かす基本環境 |
| [`dofbot_gripper_imitation_env_realsense.py`](issac_sim/dofbot/dofbot_gripper_imitation_env_realsense.py) | RealSenseカメラを含む環境 |
| [`collect_imitation_episodes.py`](ros2_ws/src/dofbot_control/dofbot_control/collect_imitation_episodes.py) | リセット・IK・ピック・成否確認を繰り返す教師データ収集ノード |
| [`convert_raw_to_lerobot.py`](dofbot/scripts/convert_raw_to_lerobot.py) | 成功エピソードを状態ベースのLeRobotデータセットへ変換 |
| [`run_diffusion_real_dofbot.py`](dofbot/scripts/run_diffusion_real_dofbot.py) | Diffusion PolicyのROS 2推論。Isaac Simと実機で共用可能 |
| [`run_act_inference_ros2.py`](dofbot/scripts/run_act_inference_ros2.py) | Isaac Sim上でのACT評価、リセット・成否集計 |
| [`run_act_real_dofbot.py`](dofbot/scripts/run_act_real_dofbot.py) | 実測関節角度を入力する実機ACT推論 |
| [`run_act_real_open_loop.py`](dofbot/scripts/run_act_real_open_loop.py) | 送信した目標角度を次の入力に使うACT診断モード |
| [`run_smolvla_inference.py`](dofbot/scripts/run_smolvla_inference.py) | RGB画像・関節角度・言語指示によるSmolVLA評価 |

## 4. Diffusion Policy用の教師データ収集

シミュレーション用端末では、ROS 2とワークスペースをsourceした上で、共通の通信設定を使用します。ここでは実機との通信を分けるためにdomain 10を使用します。

```bash
source /opt/ros/jazzy/setup.bash
source "$DOFBOT_REPO/ros2_ws/install/setup.bash"
export ROS_DOMAIN_ID=10
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
unset ROS_LOCALHOST_ONLY
```

### 端末A：Isaac Simの起動

固定パスを変更した後、Isaac Sim付属のPythonで起動します。

```bash
cd "$ISAAC_SIM_ROOT"
./python.sh "$DOFBOT_REPO/issac_sim/dofbot/dofbot_gripper_imitation_env.py" \
  --no-headless \
  --data-root "$DOFBOT_DATA_ROOT/diffusion_policy_raw" \
  --x-min 0.17 --x-max 0.22 \
  --y-min -0.08 --y-max -0.02 \
  --home-jitter-deg 2
```

箱の配置は指定したXY範囲から生成します。`--home-jitter-deg 2`では、アーム5関節の初期角度をHOME姿勢の周囲でそれぞれ±2度変化させます。固定姿勢で比較する場合は`--home-jitter-deg 0`を使用します。

HOME姿勢は次の値です。単位はradです。

```python
[0.8, 0.3, -0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

### 端末B：教師動作の実行

同じROS通信設定の端末から、例えば100エピソード収集します。

```bash
ros2 run dofbot_control collect_imitation_episodes --ros-args \
  -p episode_count:=100 \
  -p placement_mode:=random \
  -p base_seed:=1000 \
  -p record:=true
```

箱の位置を固定する場合：

```bash
ros2 run dofbot_control collect_imitation_episodes --ros-args \
  -p episode_count:=5 \
  -p placement_mode:=specified \
  -p specified_x:=0.20 \
  -p specified_y:=-0.05 \
  -p base_seed:=1000 \
  -p record:=true
```

収集ノードがリセットとIKによる動作を担当し、Isaac Simが成功・失敗の両方を保存します。状態ベースの環境では、初期位置から5 cm以上持ち上がり、箱とグリッパ点の距離が6 cm以下の状態を0.5秒維持すると成功になります。エピソードのタイムアウトはシミュレーション時間で30秒です。

### Rawエピソードの保存内容

各`episode_XXXXXX/`に以下を保存します。`T`は記録フレーム数です。

| ファイル | 内容 |
| --- | --- |
| `metadata.json` | seed、初期状態、関節名、成功判定、終了理由など |
| `timestamps.npy` | シミュレーション時刻、`(T,)` |
| `joint_position.npy` | 実測関節角度、`(T, 11)` |
| `action.npy` | 直近に受け取った目標関節角度、`(T, 11)` |
| `cube_position.npy` | 箱のXYZ位置、`(T, 3)` |
| `gripper_position.npy` | グリッパ点のXYZ位置、`(T, 3)` |
| `phase.npy` | エピソードの状態、`(T,)` |

## 5. LeRobotデータセットへの変換と学習

### データセット変換

```bash
cd "$LEROBOT_ROOT"
uv run python "$DOFBOT_REPO/dofbot/scripts/convert_raw_to_lerobot.py" \
  --raw-root "$DOFBOT_DATA_ROOT/diffusion_policy_raw" \
  --output-root "$DOFBOT_DATA_ROOT/diffusion_policy_train" \
  --repo-id local/dofbot_diffusion_policy \
  --fps 10
```

変換対象は、`metadata.json`が`status: "finished"`かつ`success: true`のエピソードです。配列の形状、関節名、有限値、記録周期なども確認します。変換結果と除外理由は出力先の`conversion_report.json`に保存します。

`observation.environment_state`には、リセット後に落ち着いた箱の初期XY位置を各フレームで繰り返し保存します。現在の箱位置を毎回入力する構成ではありません。

出力先が既に存在する場合、`--overwrite`を追加するとそのディレクトリを削除して再作成します。

### Diffusion Policyの学習

以下は既存作業メモの学習コマンドを、このREADMEのパス設定に合わせて整理した例です。LeRobot v0.5.1の環境を前提とします。

```bash
cd "$LEROBOT_ROOT"
mkdir -p "$DOFBOT_DATA_ROOT/diffusion_policy_outputs"
export DOFBOT_RUN_DIR="$DOFBOT_DATA_ROOT/diffusion_policy_outputs/dofbot_dp_$(date +%Y%m%d_%H%M%S)"

set -o pipefail
uv run lerobot-train \
  --dataset.repo_id=local/dofbot_diffusion_policy \
  --dataset.root="$DOFBOT_DATA_ROOT/diffusion_policy_train" \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$DOFBOT_RUN_DIR" \
  --job_name=dofbot_diffusion_policy \
  --steps=50000 \
  --batch_size=32 \
  --num_workers=4 \
  --log_freq=100 \
  --save_freq=5000 \
  --eval_freq=0 \
  --wandb.enable=false \
  2>&1 | tee "$DOFBOT_DATA_ROOT/diffusion_policy_train.log"
```

学習終了後、チェックポイントの保存先を確認して指定します。

```bash
export DOFBOT_MODEL_PATH="$DOFBOT_RUN_DIR/checkpoints/last/pretrained_model"
```

作業メモに記載されたモデル設定は`n_obs_steps=2`、`horizon=16`、`n_action_steps=8`です。上のコマンドはこれらを明示指定していないため、学習出力の設定を確認してください。推論スクリプトは読み込んだチェックポイントの値を表示します。

## 6. Isaac SimでのDiffusion Policy推論

`run_diffusion_real_dofbot.py`は`/joint_states`を入力とし、`/joint_command`へ目標角度を送ります。**リセットサービスの呼び出し、HOMEへの移動、成功時の自動停止は行いません。** 指定ステップ数の実行またはCtrl+Cで終了します。

### シミュレーション用の観測補正

現在のスクリプトには、実機用の観測補正が常時入っています。

```python
policy_q[1] += 0.0527  # arm2
policy_q[2] += 0.0284  # arm3
```

シミュレーションの基準動作を確認する場合は、この2つの補正値を`0.0`に変更して使用してください。実機に戻すときは、実機用の値を設定します。現状、この切り替え用のCLI引数はありません。

### リセットして推論する

第4節の環境を起動し、同じdomain 10の端末からリセットします。

```bash
ros2 service call /dofbot/reset_episode dofbot_interfaces/srv/ResetEpisode \
  '{mode: 1, seed: 1234, x: 0.20, y: -0.05, record: false}'

ros2 topic echo --once /dofbot/episode_state
```

`phase: 2`がREADYです。RESETTINGなどが表示された場合は、READYになるまで再確認してから推論を開始します。raw評価データを残す場合は`record: true`にします。

```bash
cd "$LEROBOT_ROOT"
uv run python "$DOFBOT_REPO/dofbot/scripts/run_diffusion_real_dofbot.py" \
  --model-path "$DOFBOT_MODEL_PATH" \
  --cube-x 0.20 --cube-y -0.05 \
  --fps 10 \
  --steps 130 \
  --joint-state-timeout 5.0
```

`--cube-x`と`--cube-y`は実際の箱位置に合わせます。繰り返し評価する場合は、各実行前にリセットします。

推論実装の主な設定・挙動：

| 項目 | 現在の実装 |
| --- | --- |
| Diffusionの推論ステップ数 | `policy.diffusion.num_inference_steps = 10`に固定 |
| 乱数seed | `1234`に固定 |
| モデルパス | `--model-path`で明示指定可能 |
| モデルパス省略時 | `/home/natu/dofbot_data/latest_diffusion_run.txt`を参照するため、別環境では明示指定が必要 |
| 関節状態 | 名前で11関節を並べ替える。名前なしの場合は11要素の配列を受け入れる |
| 状態の鮮度 | 推論開始後は、最終受信から`--joint-state-timeout`を超えると終了 |
| 指令の待ち時間 | 各推論・送信後に`1 / fps`秒待つため、実際の送信周期には推論時間も加わる |

## 7. 実機DOFBOTでの推論とSim-to-Real補正

### Raspberry Pi側の準備

実機には、`/joint_command`を受けてサーボを動かし、`/joint_states`を返すROS 2ドライバが必要です。**実機ドライバとHOME用シェルスクリプトは、このリポジトリには含まれていません。**

既存作業メモで使用した外部ファイルは次のとおりです。

| ファイル | 用途 |
| --- | --- |
| `~/dofbot_ws/src/dofbot_driver/dofbot_driver/dofbot_node.py` | 実機の関節指令・状態読み出し、radとサーボ角度の変換 |
| `~/bin/dofbot_home.sh` | 実機を学習時のHOME姿勢へ戻す |

以下は、これらを別途準備済みのRaspberry Piでの起動例です。

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

### 推論PC側

実機側と同じdomain 0へ切り替えます。

```bash
source /opt/ros/jazzy/setup.bash
source "$DOFBOT_REPO/ros2_ws/install/setup.bash"
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
unset ROS_LOCALHOST_ONLY

ros2 topic echo --once /joint_states
```

実機をHOME姿勢に戻し、箱を指定した位置に置いた後、第6節と同じ推論コマンドを実行します。その際、スクリプトの観測補正を実機用の値に設定してください。

### 現在の暫定補正

実測角度をコピーし、Policyへ渡す`observation.state`のみに補正を加えています。

| 関節 | 加算する値 | 度換算 |
| --- | --- | --- |
| arm2 | `+0.0527 rad` | 約`+3.02°` |
| arm3 | `+0.0284 rad` | 約`+1.63°` |

この補正は、受信した`/joint_states`の値や出力する`/joint_command`へ直接加えるものではありません。値は当該実機での暫定値であり、別の個体への適用時には測定が必要です。

既存作業メモでは、実機ドライバに次のグリッパ設定を使用しています。これらは外部ドライバの設定で、推論スクリプトの引数ではありません。

```python
GRIPPER_OPEN_DEG = 60.0
GRIPPER_CLOSED_DEG = 145.0
GRIPPER_Q_VALUE = 1.6
```

### 作業メモに記録された検証結果

| ケース | 条件 | 記録された結果 |
| --- | --- | --- |
| A | Isaac Sim + Diffusion Policyの関節状態フィードバック | ピック成功 |
| B | 実機 + 未補正の関節状態フィードバック | ピック失敗 |
| C | 実機 + Aで成功した関節指令の再生 | ピック成功 |
| 補正後 | 実機 + arm2/arm3の観測補正 + グリッパ閉角145° | ピック成功 |

この比較では、成功した指令軌道を実機で実行できることを確認し、観測値のずれを切り分けています。成功率の統計評価は上の結果には含まれていません。今後の課題は、暫定補正を正式な関節キャリブレーションとして整理し、推論入力または学習データへ反映する方針を決めることです。

## 8. ACT・SmolVLA

### ACT

状態ベースの変換スクリプトはACT用にも使用できます。ACTの学習済みチェックポイントを用意した後、Isaac Sim上で次のように評価します。

```bash
cd "$LEROBOT_ROOT"
uv run python "$DOFBOT_REPO/dofbot/scripts/run_act_inference_ros2.py" \
  --model-path /absolute/path/to/act/checkpoints/last/pretrained_model \
  --dataset-root "$DOFBOT_DATA_ROOT/diffusion_policy_train" \
  --dataset-repo-id local/dofbot_diffusion_policy \
  --episodes 5 \
  --device cuda
```

このスクリプトはエピソードごとのリセットと成否集計を行います。実機用には`run_act_real_dofbot.py`、実測状態の影響を切り分ける診断用には`run_act_real_open_loop.py`を使用します。後者は`/joint_states`を購読せず、送信した目標角度を次の観測として使います。

### SmolVLA

| スクリプト | 役割 |
| --- | --- |
| [`dofbot_smolvla_env.py`](issac_sim/dofbot/dofbot_smolvla_env.py) | RGB・言語指示を保存する収集環境 |
| [`dofbot_smolvla_no_cir_env.py`](issac_sim/dofbot/dofbot_smolvla_no_cir_env.py) | SmolVLA収集環境の別バリアント |
| [`dofbot_smolvla_inference_env.py`](issac_sim/dofbot/dofbot_smolvla_inference_env.py) | オンライン推論用のRGB画像配信を持つ環境 |
| [`collect_smolvla_pick_episodes.py`](ros2_ws/src/dofbot_control/dofbot_control/collect_smolvla_pick_episodes.py) | 色を選び、言語指示と対応するIK教師動作を生成 |
| [`convert_smolvla_raw_to_lerobot.py`](dofbot/scripts/convert_smolvla_raw_to_lerobot.py) | 動画・関節角度・言語指示をLeRobot Dataset v3へ変換 |

SmolVLAの環境では、シミュレーション内のRealSense D455アセットを使用します。Isaac Simのアセットと、動画を扱う場合の`ffmpeg`・`ffprobe`が必要です。

**SmolVLA環境などのOmniGraph側には`ROS_DOMAIN_ID = 0`が固定されている箇所があります。** domain 10を使う場合は、実行する環境スクリプトの定数を次のように変更し、ROS 2ノードと合わせてください。

```python
ROS_DOMAIN_ID = int(os.environ.get("ROS_DOMAIN_ID", "0"))
```

固定パスとdomainを設定した上で、端末AでRGB配信・動画保存を有効にします。

```bash
cd "$ISAAC_SIM_ROOT"
./python.sh "$DOFBOT_REPO/issac_sim/dofbot/dofbot_smolvla_inference_env.py" \
  --no-headless \
  --publish-camera \
  --record-video \
  --data-root "$DOFBOT_DATA_ROOT/smolvla_raw"
```

端末Bで教師エピソードを収集します。

```bash
ros2 run dofbot_control collect_smolvla_pick_episodes --ros-args \
  -p episode_count:=100 \
  -p target_color:=random \
  -p record:=true
```

RGB動画・言語指示を含むrawデータを変換します。

```bash
cd "$LEROBOT_ROOT"
uv run python "$DOFBOT_REPO/dofbot/scripts/convert_smolvla_raw_to_lerobot.py" \
  --raw-root "$DOFBOT_DATA_ROOT/smolvla_raw" \
  --output-root "$DOFBOT_DATA_ROOT/smolvla_lerobot" \
  --repo-id local/dofbot_smolvla_pick
```

動画のFeature名は既定で`observation.images.front`です。変換は既定で成功エピソードのみを対象にし、完了後にデータセットの再読み込みを検証します。状態ベース用の変換スクリプトとは異なるため、SmolVLAにはこちらを使用します。

変換したデータセットでSmolVLAを学習し、チェックポイントを用意した後に評価します。評価時もRGB配信を有効にした環境を使用します。

```bash
cd "$LEROBOT_ROOT"
uv run python "$DOFBOT_REPO/dofbot/scripts/run_smolvla_inference.py" \
  --checkpoint /absolute/path/to/smolvla/checkpoints/last/pretrained_model \
  --dataset-root "$DOFBOT_DATA_ROOT/smolvla_lerobot" \
  --dataset-repo-id local/dofbot_smolvla_pick \
  --episodes 5 \
  --target-color red \
  --device cuda
```

## 9. ROS 2インターフェイス

| 名前 | 型 | 用途 |
| --- | --- | --- |
| `/joint_command` | `sensor_msgs/msg/JointState` | 目標関節角度の送信 |
| `/joint_states` | `sensor_msgs/msg/JointState` | 実測関節角度の受信 |
| `/dofbot/episode_state` | `dofbot_interfaces/msg/EpisodeState` | シミュレーションの進行・成否・箱位置 |
| `/dofbot/reset_episode` | `dofbot_interfaces/srv/ResetEpisode` | 箱位置・seed・記録有無を指定してリセット |
| `/dofbot/end_episode` | `dofbot_interfaces/srv/EndEpisode` | エピソードを終了し、記録結果を取得 |
| `/dofbot/task_instruction` | `std_msgs/msg/String` | SmolVLAの言語指示 |
| `/dofbot/scene_layout` | `std_msgs/msg/String` | SmolVLA環境の箱配置をJSONで配信 |
| `/dofbot/camera/color/image_raw` | `sensor_msgs/msg/Image` | SmolVLA推論用RGB画像。`--publish-camera`で有効化 |

`JointState.position`に角度を格納します。Diffusion Policyの推論スクリプトは`velocity`と`effort`を空配列として送ります。エピソード関連のサービス・トピックはシミュレーション環境が提供します。

| `EpisodeState.phase` | 状態 |
| --- | --- |
| `0` | IDLE |
| `1` | RESETTING |
| `2` | READY |
| `3` | RUNNING |
| `4` | SUCCESS |
| `5` | TERMINATED |

## 10. ログ・rosbagの比較

推論中に、同じdomainの別端末で状態と指令を記録します。

```bash
mkdir -p "$DOFBOT_DATA_ROOT/dp_compare"
ros2 bag record \
  -o "$DOFBOT_DATA_ROOT/dp_compare/real_seed1234" \
  /joint_states /joint_command
```

シミュレーションと実機のbagを用意した後、比較スクリプトを実行します。

```bash
python "$DOFBOT_REPO/dofbot/scripts/compare_dofbot_bags.py" \
  --isaac "$DOFBOT_DATA_ROOT/dp_compare/isaac_seed1234_v2" \
  --real "$DOFBOT_DATA_ROOT/dp_compare/real_seed1234" \
  --out "$DOFBOT_DATA_ROOT/dp_compare/analysis"
```

CSVと、アームの状態差・指令差・グリッパ指令のPNGを出力します。

| スクリプト | 解析内容 |
| --- | --- |
| [`compare_dofbot_bags.py`](dofbot/scripts/compare_dofbot_bags.py) | Isaac Simと実機の状態・指令を比較 |
| [`compare_dp_policy_logs.py`](dofbot/scripts/compare_dp_policy_logs.py) | 推論ログから関節状態・Policy出力を比較 |
| [`analyze_dofbot_state_lag.py`](dofbot/scripts/analyze_dofbot_state_lag.py) | 同じ指令列で動かした場合の状態応答の時間差を推定 |
| [`analyze_b_vs_success_path.py`](dofbot/scripts/analyze_b_vs_success_path.py) | 失敗推論ログと成功再生軌道の進行方向を比較 |
| [`compare_dp_observation_history.py`](dofbot/scripts/compare_dp_observation_history.py) | 失敗実行と成功再生の観測履歴を比較 |

成功bagの指令だけを実機で再生して切り分ける場合は、実機を同じ初期姿勢・箱配置にし、Policyの指令送信を停止してから次を実行します。

```bash
ros2 bag play "$DOFBOT_DATA_ROOT/dp_compare/isaac_seed1234_v2" \
  --topics /joint_command
```

## 11. トラブルシューティング

| 症状 | 確認事項 |
| --- | --- |
| `URDF not found` | 環境スクリプトの`URDF_PATH`をリポジトリ内の実在するファイルへ変更する |
| `dofbot_interfaces`をimportできない | ワークスペースをビルドし、Isaac Sim起動前に`install/setup.bash`をsourceする |
| LeRobot環境で`rclpy`をimportできない | ROS 2をsourceした上で、PythonバージョンとROS 2のPython拡張の互換性を確認する |
| `/joint_states`が届かない | シミュレーションまたは実機ドライバの起動、domain、ネットワーク、関節名を確認する |
| `/joint_states is stale` | 状態配信が止まっていないか確認し、推論負荷と`--joint-state-timeout`を確認する |
| モデルが見つからない | `--model-path`で`pretrained_model`ディレクトリを明示指定する |
| 変換対象の成功エピソードがない | `metadata.json`の`status`・`success`、`record:=true`、保存先、変換時の除外理由を確認する |
| 箱まで到達しない | HOME姿勢、箱の初期XY、関節順序、実機用観測補正を確認する |
| SmolVLAの画像が届かない | `dofbot_smolvla_inference_env.py`で`--publish-camera`を指定し、domainとカメラアセットを確認する |
| SmolVLAの動画変換が失敗する | 収集時の`--record-video`、`record:=true`、`ffmpeg`・`ffprobe`、`camera_rgb.mp4`を確認する |

本READMEの手順は、リポジトリ内のコードと既存作業メモを基に整理しています。固定パス、モデル設定、実機のキャリブレーションは使用する環境に合わせて調整してください。
