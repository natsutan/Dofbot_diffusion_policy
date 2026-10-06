#!/usr/bin/env python3
"""Run a trained LeRobot ACT policy against the DOFBOT Isaac Sim environment.

The script is intentionally standalone. Run it from the LeRobot uv project after
sourcing ROS 2 and the user's workspace. It:

1. Loads a local ACT checkpoint and its saved pre/post-processors.
2. Calls /dofbot/reset_episode with a random cube position.
3. Uses /joint_states as observation.state.
4. Uses the reset response cube x/y as observation.environment_state.
5. Publishes the policy's 11-D action to /joint_command at 10 Hz.
6. Reads /dofbot/episode_state to report success/failure.

Expected packages:
- LeRobot v0.5.1 in the current uv project
- ROS 2 Jazzy
- dofbot_interfaces
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from dofbot_interfaces.msg import EpisodeState
from dofbot_interfaces.srv import EndEpisode, ResetEpisode

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


JOINT_NAMES = (
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
)

JOINT_COMMAND_TOPIC = "/joint_command"
JOINT_STATE_TOPIC = "/joint_states"
EPISODE_STATE_TOPIC = "/dofbot/episode_state"
RESET_SERVICE = "/dofbot/reset_episode"
END_SERVICE = "/dofbot/end_episode"

OBS_STATE = "observation.state"
OBS_ENV_STATE = "observation.environment_state"

MODE_RANDOM = 0
PHASE_READY = 2
PHASE_SUCCESS = 4
PHASE_TERMINATED = 5


@dataclass
class EvaluationResult:
    evaluation_index: int
    episode_id: int
    seed: int
    cube_x: float
    cube_y: float
    success: bool
    truncated: bool
    termination_reason: str
    control_steps: int
    final_lift_height: float
    final_gripper_distance: float


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained LeRobot ACT policy in the DOFBOT Isaac Sim environment."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Path to checkpoints/last/pretrained_model. When omitted, the script "
            "uses ~/dofbot/latest_act_run.txt or the newest model below "
            "~/dofbot/train_outputs."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path.home() / "dofbot/train_data_lerobot/dofbot_pick_state",
        help="Local LeRobot dataset root used to recover metadata/statistics if needed.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default="local/dofbot_pick_state",
        help="Logical repository ID stored with the local dataset.",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=5000)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument("--control-rate-hz", type=float, default=10.0)
    parser.add_argument(
        "--max-control-steps",
        type=int,
        default=300,
        help="Controller-side limit. Isaac Sim also has its own episode timeout.",
    )
    parser.add_argument(
        "--max-command-delta",
        type=float,
        default=0.35,
        help=(
            "Maximum change in each commanded joint per 10 Hz step [rad]. "
            "Set 0 to disable this safety limiter."
        ),
    )
    parser.add_argument(
        "--record-evaluation",
        action="store_true",
        help="Ask Isaac Sim to save raw evaluation episodes as well.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path.home() / "dofbot/eval_results",
    )
    parser.add_argument("--service-timeout-sec", type=float, default=30.0)
    parser.add_argument("--ready-timeout-sec", type=float, default=10.0)
    parser.add_argument("--joint-state-timeout-sec", type=float, default=5.0)
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log one action/status line every N control steps.",
    )
    args = parser.parse_args(argv)

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.control_rate_hz <= 0:
        parser.error("--control-rate-hz must be positive")
    if args.max_control_steps <= 0:
        parser.error("--max-control-steps must be positive")
    if args.max_command_delta < 0:
        parser.error("--max-command-delta must not be negative")
    return args


def resolve_model_path(explicit_path: Optional[Path]) -> Path:
    def normalize(path: Path) -> Optional[Path]:
        path = path.expanduser().resolve()
        candidates = [path]
        if path.name != "pretrained_model":
            candidates.append(path / "checkpoints/last/pretrained_model")
        for candidate in candidates:
            if (candidate / "model.safetensors").is_file() and (candidate / "config.json").is_file():
                return candidate
        return None

    if explicit_path is not None:
        resolved = normalize(explicit_path)
        if resolved is None:
            raise FileNotFoundError(
                f"ACT checkpoint was not found below: {explicit_path.expanduser()}"
            )
        return resolved

    latest_file = Path.home() / "dofbot/latest_act_run.txt"
    if latest_file.is_file():
        text = latest_file.read_text(encoding="utf-8").strip()
        if text:
            resolved = normalize(Path(text))
            if resolved is not None:
                return resolved

    candidates = list(
        (Path.home() / "dofbot/train_outputs").glob(
            "*/checkpoints/last/pretrained_model"
        )
    )
    candidates = [
        path
        for path in candidates
        if (path / "model.safetensors").is_file()
        and (path / "config.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            "No ACT checkpoint was found. Pass --model-path or create "
            "~/dofbot/latest_act_run.txt."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested, but CUDA is unavailable")
    return requested


def as_float_array(value: object) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None


class ActPolicyRuntime:
    def __init__(
        self,
        *,
        model_path: Path,
        dataset_root: Path,
        dataset_repo_id: str,
        device: str,
    ) -> None:
        self.model_path = model_path
        self.dataset_root = dataset_root.expanduser().resolve()
        self.dataset_repo_id = dataset_repo_id
        self.device = device

        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")

        print(f"Loading dataset metadata: {self.dataset_root}", flush=True)
        self.dataset_meta = LeRobotDatasetMetadata(
            repo_id=self.dataset_repo_id,
            root=self.dataset_root,
        )

        print(f"Loading ACT checkpoint: {self.model_path}", flush=True)
        config = PreTrainedConfig.from_pretrained(self.model_path)
        if config.type != "act":
            raise ValueError(
                f"The checkpoint policy type is {config.type!r}, not 'act'"
            )
        config.device = self.device

        self.policy = ACTPolicy.from_pretrained(
            self.model_path,
            config=config,
            local_files_only=True,
        )
        self.policy.to(self.device)
        self.policy.eval()

        # Normalization statistics and processor configuration are normally saved
        # with every LeRobot checkpoint. Fall back to the local dataset metadata
        # for checkpoints that do not contain processor files.
        try:
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=config,
                pretrained_path=str(self.model_path),
                preprocessor_overrides={
                    "device_processor": {"device": self.device},
                },
            )
            print("Loaded pre/post-processors from checkpoint.", flush=True)
        except Exception as exc:
            print(
                "WARNING: could not load processors from checkpoint; "
                f"using dataset statistics instead: {exc}",
                flush=True,
            )
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=config,
                dataset_stats=self.dataset_meta.stats,
            )

        input_features = config.input_features or {}
        output_features = config.output_features or {}
        if OBS_STATE not in input_features:
            raise ValueError(f"Checkpoint does not expect {OBS_STATE}")
        if OBS_ENV_STATE not in input_features:
            raise ValueError(f"Checkpoint does not expect {OBS_ENV_STATE}")
        if "action" not in output_features:
            raise ValueError("Checkpoint does not define an action output")

        state_shape = tuple(input_features[OBS_STATE].shape)
        env_shape = tuple(input_features[OBS_ENV_STATE].shape)
        action_shape = tuple(output_features["action"].shape)
        if state_shape != (len(JOINT_NAMES),):
            raise ValueError(
                f"Expected 11-D state, checkpoint expects {state_shape}"
            )
        if env_shape != (2,):
            raise ValueError(
                f"Expected 2-D environment state [cube_x, cube_y], checkpoint expects {env_shape}"
            )
        if action_shape != (len(JOINT_NAMES),):
            raise ValueError(
                f"Expected 11-D action, checkpoint outputs {action_shape}"
            )

        action_stats = (self.dataset_meta.stats or {}).get("action", {})
        self.action_min = as_float_array(action_stats.get("min"))
        self.action_max = as_float_array(action_stats.get("max"))
        if self.action_min is not None:
            self.action_min = self.action_min.reshape(-1)
        if self.action_max is not None:
            self.action_max = self.action_max.reshape(-1)
        if (
            self.action_min is None
            or self.action_max is None
            or self.action_min.shape != (len(JOINT_NAMES),)
            or self.action_max.shape != (len(JOINT_NAMES),)
        ):
            self.action_min = None
            self.action_max = None
            print("WARNING: action min/max were not available; range clipping is disabled.")

        print(f"Policy device      : {self.device}")
        print(f"Policy chunk_size  : {getattr(config, 'chunk_size', 'unknown')}")
        print(f"Policy action steps: {getattr(config, 'n_action_steps', 'unknown')}")
        print(f"Dataset FPS        : {self.dataset_meta.fps}")

    def reset(self) -> None:
        self.policy.reset()
        if hasattr(self.preprocessor, "reset"):
            self.preprocessor.reset()
        if hasattr(self.postprocessor, "reset"):
            self.postprocessor.reset()

    @torch.inference_mode()
    def infer(self, joint_position: np.ndarray, cube_xy: np.ndarray) -> np.ndarray:
        observation = {
            OBS_STATE: torch.as_tensor(joint_position, dtype=torch.float32),
            OBS_ENV_STATE: torch.as_tensor(cube_xy, dtype=torch.float32),
        }
        processed = self.preprocessor(observation)
        action = self.policy.select_action(processed)
        action = self.postprocessor(action)
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action)
        result = action.detach().cpu().numpy().reshape(-1).astype(np.float64)
        if result.shape != (len(JOINT_NAMES),):
            raise RuntimeError(f"ACT returned an unexpected action shape: {result.shape}")
        if not np.all(np.isfinite(result)):
            raise RuntimeError(f"ACT returned NaN or inf: {result}")
        return result

    def clip_to_training_range(self, action: np.ndarray) -> np.ndarray:
        if self.action_min is None or self.action_max is None:
            return action
        return np.clip(action, self.action_min, self.action_max)


class DofbotActEvaluationNode(Node):
    def __init__(self) -> None:
        super().__init__("dofbot_act_evaluation")

        self.command_publisher = self.create_publisher(
            JointState,
            JOINT_COMMAND_TOPIC,
            10,
        )
        self.joint_subscription = self.create_subscription(
            JointState,
            JOINT_STATE_TOPIC,
            self._on_joint_state,
            10,
        )

        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.episode_subscription = self.create_subscription(
            EpisodeState,
            EPISODE_STATE_TOPIC,
            self._on_episode_state,
            state_qos,
        )
        self.reset_client = self.create_client(ResetEpisode, RESET_SERVICE)
        self.end_client = self.create_client(EndEpisode, END_SERVICE)

        self.latest_joint_position: Optional[np.ndarray] = None
        self.joint_state_counter = 0
        self.latest_episode_state: Optional[EpisodeState] = None

    def _on_joint_state(self, message: JointState) -> None:
        if not message.position:
            return
        if message.name:
            if len(message.name) != len(message.position):
                return
            values = dict(zip(message.name, message.position, strict=True))
            if not all(name in values for name in JOINT_NAMES):
                return
            position = np.asarray([values[name] for name in JOINT_NAMES], dtype=np.float64)
        elif len(message.position) == len(JOINT_NAMES):
            position = np.asarray(message.position, dtype=np.float64)
        else:
            return
        if np.all(np.isfinite(position)):
            self.latest_joint_position = position
            self.joint_state_counter += 1

    def _on_episode_state(self, message: EpisodeState) -> None:
        self.latest_episode_state = message

    def spin_for(self, duration_sec: float) -> None:
        deadline = time.monotonic() + max(0.0, duration_sec)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.02, deadline - time.monotonic()))

    def wait_for_services(self, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            reset_ready = self.reset_client.wait_for_service(timeout_sec=0.2)
            end_ready = self.end_client.wait_for_service(timeout_sec=0.2)
            if reset_ready and end_ready:
                self.get_logger().info("Isaac Sim episode services are ready.")
                return
        raise TimeoutError("Timed out waiting for Isaac Sim episode services")

    def call_service(self, client, request, timeout_sec: float):
        future = client.call_async(request)
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Service call timed out: {client.srv_name}")
            rclpy.spin_once(self, timeout_sec=min(0.05, remaining))
        if not future.done():
            raise RuntimeError(f"Service call did not complete: {client.srv_name}")
        exception = future.exception()
        if exception is not None:
            raise exception
        return future.result()

    def reset_random_episode(
        self,
        *,
        seed: int,
        record: bool,
        service_timeout_sec: float,
        ready_timeout_sec: float,
        joint_state_timeout_sec: float,
    ) -> tuple[int, np.ndarray]:
        previous_joint_counter = self.joint_state_counter
        self.latest_episode_state = None

        request = ResetEpisode.Request()
        request.mode = MODE_RANDOM
        request.x = 0.0
        request.y = 0.0
        request.seed = int(seed)
        request.record = bool(record)

        response = self.call_service(
            self.reset_client,
            request,
            service_timeout_sec,
        )
        if response is None or not response.accepted:
            message = "No response" if response is None else response.message
            raise RuntimeError(f"Reset was rejected: {message}")

        episode_id = int(response.episode_id)
        cube_xy = np.asarray([response.cube_x, response.cube_y], dtype=np.float64)
        self.get_logger().info(
            f"Episode {episode_id} reset: seed={response.seed_used}, "
            f"cube=({cube_xy[0]:.4f}, {cube_xy[1]:.4f})"
        )

        deadline = time.monotonic() + ready_timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            state = self.latest_episode_state
            if (
                state is not None
                and int(state.episode_id) == episode_id
                and int(state.phase) == PHASE_READY
            ):
                break
        else:
            raise TimeoutError(f"Episode {episode_id} did not reach READY")

        joint_deadline = time.monotonic() + joint_state_timeout_sec
        while rclpy.ok() and time.monotonic() < joint_deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if (
                self.latest_joint_position is not None
                and self.joint_state_counter > previous_joint_counter
            ):
                return episode_id, cube_xy
        raise TimeoutError(f"No fresh /joint_states arrived for episode {episode_id}")

    def publish_action(self, action: np.ndarray) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = action.astype(float).tolist()
        message.velocity = []
        message.effort = []
        self.command_publisher.publish(message)

    def current_terminal_state(self, episode_id: int) -> Optional[EpisodeState]:
        state = self.latest_episode_state
        if state is None or int(state.episode_id) != episode_id:
            return None
        if bool(state.terminated) or int(state.phase) in (
            PHASE_SUCCESS,
            PHASE_TERMINATED,
        ):
            return state
        return None

    def end_episode(self, reason: str, timeout_sec: float) -> None:
        request = EndEpisode.Request()
        request.truncated = True
        request.reason = reason
        try:
            response = self.call_service(self.end_client, request, timeout_sec)
            if response is not None:
                self.get_logger().info(
                    f"EndEpisode: accepted={response.accepted}, "
                    f"success={response.success}, reason={response.message}"
                )
        except Exception as exc:
            self.get_logger().error(f"Failed to end episode cleanly: {exc}")


def limit_command_delta(
    action: np.ndarray,
    previous_command: np.ndarray,
    maximum_delta: float,
) -> np.ndarray:
    if maximum_delta <= 0:
        return action
    delta = np.clip(action - previous_command, -maximum_delta, maximum_delta)
    return previous_command + delta


def run_evaluation(args: argparse.Namespace) -> int:
    model_path = resolve_model_path(args.model_path)
    device = select_device(args.device)

    runtime = ActPolicyRuntime(
        model_path=model_path,
        dataset_root=args.dataset_root,
        dataset_repo_id=args.dataset_repo_id,
        device=device,
    )

    rclpy.init()
    node = DofbotActEvaluationNode()
    results: list[EvaluationResult] = []

    try:
        node.wait_for_services(args.service_timeout_sec)
        period_sec = 1.0 / args.control_rate_hz

        for evaluation_index in range(args.episodes):
            seed = args.base_seed + evaluation_index
            node.get_logger().info(
                f"================ evaluation {evaluation_index + 1}/{args.episodes} ================"
            )

            episode_id = -1
            cube_xy = np.zeros(2, dtype=np.float64)
            control_steps = 0
            try:
                episode_id, cube_xy = node.reset_random_episode(
                    seed=seed,
                    record=args.record_evaluation,
                    service_timeout_sec=args.service_timeout_sec,
                    ready_timeout_sec=args.ready_timeout_sec,
                    joint_state_timeout_sec=args.joint_state_timeout_sec,
                )
                runtime.reset()
                assert node.latest_joint_position is not None
                previous_command = node.latest_joint_position.copy()

                next_tick = time.monotonic()
                terminal_state: Optional[EpisodeState] = None

                for control_steps in range(1, args.max_control_steps + 1):
                    # Keep ROS callbacks moving until the next 10 Hz control instant.
                    while rclpy.ok() and time.monotonic() < next_tick:
                        terminal_state = node.current_terminal_state(episode_id)
                        if terminal_state is not None:
                            break
                        rclpy.spin_once(
                            node,
                            timeout_sec=min(0.02, next_tick - time.monotonic()),
                        )
                    if terminal_state is not None:
                        break

                    rclpy.spin_once(node, timeout_sec=0.0)
                    terminal_state = node.current_terminal_state(episode_id)
                    if terminal_state is not None:
                        break
                    if node.latest_joint_position is None:
                        raise RuntimeError("/joint_states became unavailable")

                    action = runtime.infer(node.latest_joint_position, cube_xy)
                    action = runtime.clip_to_training_range(action)
                    action = limit_command_delta(
                        action,
                        previous_command,
                        args.max_command_delta,
                    )
                    node.publish_action(action)
                    previous_command = action

                    if args.log_every > 0 and (
                        control_steps == 1 or control_steps % args.log_every == 0
                    ):
                        state = node.latest_episode_state
                        lift = float(state.lift_height) if state is not None else math.nan
                        distance = (
                            float(state.gripper_distance) if state is not None else math.nan
                        )
                        node.get_logger().info(
                            f"step={control_steps:03d}, "
                            f"lift={lift:.4f} m, gripper_distance={distance:.4f} m, "
                            f"action_norm={np.linalg.norm(action):.3f}"
                        )

                    next_tick += period_sec
                    # If inference took longer than a period, avoid an accumulating delay.
                    if next_tick < time.monotonic() - period_sec:
                        next_tick = time.monotonic()

                # Process any success message emitted immediately after the last action.
                node.spin_for(0.1)
                terminal_state = node.current_terminal_state(episode_id)
                if terminal_state is None:
                    node.end_episode(
                        "act_controller_step_limit",
                        args.service_timeout_sec,
                    )
                    node.spin_for(0.1)
                    terminal_state = node.current_terminal_state(episode_id)

                if terminal_state is None:
                    raise RuntimeError(
                        f"Episode {episode_id} ended without a terminal EpisodeState"
                    )

                result = EvaluationResult(
                    evaluation_index=evaluation_index,
                    episode_id=episode_id,
                    seed=seed,
                    cube_x=float(cube_xy[0]),
                    cube_y=float(cube_xy[1]),
                    success=bool(terminal_state.success),
                    truncated=bool(terminal_state.truncated),
                    termination_reason=str(terminal_state.termination_reason),
                    control_steps=control_steps,
                    final_lift_height=float(terminal_state.lift_height),
                    final_gripper_distance=float(terminal_state.gripper_distance),
                )
                results.append(result)
                node.get_logger().info(
                    f"Episode {episode_id} result: success={result.success}, "
                    f"reason={result.termination_reason}, steps={result.control_steps}"
                )

            except KeyboardInterrupt:
                if episode_id >= 0:
                    node.end_episode("keyboard_interrupt", args.service_timeout_sec)
                raise
            except Exception as exc:
                node.get_logger().error(
                    f"Evaluation {evaluation_index + 1} failed: {type(exc).__name__}: {exc}"
                )
                if episode_id >= 0:
                    node.end_episode("act_controller_exception", args.service_timeout_sec)
                results.append(
                    EvaluationResult(
                        evaluation_index=evaluation_index,
                        episode_id=episode_id,
                        seed=seed,
                        cube_x=float(cube_xy[0]),
                        cube_y=float(cube_xy[1]),
                        success=False,
                        truncated=True,
                        termination_reason=f"controller_exception: {exc}",
                        control_steps=control_steps,
                        final_lift_height=math.nan,
                        final_gripper_distance=math.nan,
                    )
                )

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    successes = sum(result.success for result in results)
    args.result_dir = args.result_dir.expanduser().resolve()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.result_dir / f"act_eval_{timestamp}.json"
    report = {
        "model_path": str(model_path),
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "device": device,
        "control_rate_hz": args.control_rate_hz,
        "episodes_requested": args.episodes,
        "episodes_completed": len(results),
        "successes": successes,
        "failures": len(results) - successes,
        "success_rate": successes / len(results) if results else 0.0,
        "results": [asdict(result) for result in results],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    print("\n========== ACT evaluation summary ==========")
    print(f"Model       : {model_path}")
    print(f"Episodes    : {len(results)}")
    print(f"Successes   : {successes}")
    print(f"Failures    : {len(results) - successes}")
    print(f"Success rate: {report['success_rate'] * 100.0:.1f}%")
    print(f"Report      : {report_path}")

    return 0 if successes > 0 else 2


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    raise SystemExit(run_evaluation(args))


if __name__ == "__main__":
    main()
