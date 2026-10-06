#!/usr/bin/env python3
"""Run a fine-tuned SmolVLA policy against the DOFBOT Isaac Sim environment.

The node consumes live RGB images and joint states, sends a language task,
resets the simulator, and publishes the policy's 11-dimensional joint targets.
Isaac Sim remains responsible for success/failure detection and optional episode
recording.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String

from dofbot_interfaces.msg import EpisodeState
from dofbot_interfaces.srv import EndEpisode, ResetEpisode

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import prepare_observation_for_inference


JOINT_COMMAND_TOPIC = "/joint_command"
JOINT_STATE_TOPIC = "/joint_states"
CAMERA_RGB_TOPIC = "/dofbot/camera/color/image_raw"
TASK_INSTRUCTION_TOPIC = "/dofbot/task_instruction"
EPISODE_STATE_TOPIC = "/dofbot/episode_state"
RESET_SERVICE = "/dofbot/reset_episode"
END_SERVICE = "/dofbot/end_episode"

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
COLOR_NAMES = ("red", "blue", "green")

MODE_RANDOM = 0
PHASE_READY = 2
PHASE_SUCCESS = 4
PHASE_TERMINATED = 5


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SmolVLA policy in Isaac Sim")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home()
        / "dofbot"
        / "train_outputs"
        / "smolvla_pick_v1"
        / "checkpoints"
        / "last"
        / "pretrained_model",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path.home() / "dofbot" / "smolvla_lerobot")
    parser.add_argument("--dataset-repo-id", default="local/dofbot_smolvla_pick")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--target-color", choices=("random", *COLOR_NAMES), default="random")
    parser.add_argument("--target-seed", type=int, default=-1)
    parser.add_argument("--base-seed", type=int, default=-1)
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=5,
        help="Number of queued actions to execute before recomputing an action chunk.",
    )
    parser.add_argument("--episode-timeout", type=float, default=30.0)
    parser.add_argument("--service-timeout", type=float, default=30.0)
    parser.add_argument("--observation-timeout", type=float, default=10.0)
    parser.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask Isaac Sim to retain evaluation episode data.",
    )
    parser.add_argument(
        "--max-action-delta",
        type=float,
        default=0.0,
        help="Optional per-control-step joint target clamp in radians; 0 disables it.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--result-path",
        type=Path,
        default=None,
        help="JSON evaluation summary path. A timestamped file is used when omitted.",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.control_hz <= 0:
        parser.error("--control-hz must be positive")
    if args.replan_steps <= 0:
        parser.error("--replan-steps must be positive")
    if args.episode_timeout <= 0:
        parser.error("--episode-timeout must be positive")
    if args.max_action_delta < 0:
        parser.error("--max-action-delta must be non-negative")
    return args


@dataclass
class EvaluationResult:
    collection_index: int
    episode_id: int
    target_color: str
    task: str
    environment_seed: int
    success: bool
    truncated: bool
    reason: str
    elapsed_sec: float
    control_steps: int
    inference_calls: int


class SmolVLAInferenceNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("dofbot_smolvla_inference")
        self.args = args
        self.latest_image: Optional[np.ndarray] = None
        self.latest_joint_state: Optional[np.ndarray] = None
        self.latest_episode_state: Optional[EpisodeState] = None
        self.active_episode_id: Optional[int] = None

        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.command_publisher = self.create_publisher(JointState, JOINT_COMMAND_TOPIC, 10)
        self.task_publisher = self.create_publisher(String, TASK_INSTRUCTION_TOPIC, state_qos)
        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._on_joint_state, 10)
        self.create_subscription(Image, CAMERA_RGB_TOPIC, self._on_image, qos_profile_sensor_data)
        self.create_subscription(EpisodeState, EPISODE_STATE_TOPIC, self._on_episode_state, state_qos)
        self.reset_client = self.create_client(ResetEpisode, RESET_SERVICE)
        self.end_client = self.create_client(EndEpisode, END_SERVICE)

        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
        self.checkpoint = checkpoint

        self.get_logger().info(f"Loading policy: {checkpoint}")
        self.policy = SmolVLAPolicy.from_pretrained(str(checkpoint)).to(self.device).eval()
        if args.replan_steps > self.policy.config.chunk_size:
            raise ValueError(
                f"replan_steps={args.replan_steps} exceeds chunk_size={self.policy.config.chunk_size}"
            )
        self.policy.config.n_action_steps = int(args.replan_steps)
        self.policy.reset()

        try:
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.policy.config,
                pretrained_path=str(checkpoint),
                preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            )
            self.get_logger().info("Loaded saved policy pre/post processors from checkpoint.")
        except (FileNotFoundError, OSError) as exc:
            self.get_logger().warning(
                f"Checkpoint processors were not found ({exc}); loading dataset statistics instead."
            )
            metadata = LeRobotDatasetMetadata(
                args.dataset_repo_id,
                root=args.dataset_root.expanduser().resolve(),
            )
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.policy.config,
                dataset_stats=metadata.stats,
            )

        self.get_logger().info(
            f"Policy ready: device={self.device}, chunk_size={self.policy.config.chunk_size}, "
            f"executed_before_replan={self.policy.config.n_action_steps}"
        )

    def _on_joint_state(self, message: JointState) -> None:
        if len(message.position) == 0:
            return
        if message.name:
            if len(message.name) != len(message.position):
                return
            positions = dict(zip(message.name, message.position, strict=True))
            if any(name not in positions for name in JOINT_NAMES):
                return
            values = np.asarray([positions[name] for name in JOINT_NAMES], dtype=np.float32)
        elif len(message.position) == len(JOINT_NAMES):
            values = np.asarray(message.position, dtype=np.float32)
        else:
            return
        if np.all(np.isfinite(values)):
            self.latest_joint_state = values

    def _on_image(self, message: Image) -> None:
        if message.height <= 0 or message.width <= 0:
            return
        encoding = message.encoding.lower()
        channels = 4 if encoding in ("rgba8", "bgra8") else 3
        expected_row_bytes = int(message.width) * channels
        if int(message.step) < expected_row_bytes:
            self.get_logger().warning(f"Invalid image step: {message.step}")
            return
        raw = np.frombuffer(message.data, dtype=np.uint8)
        required = int(message.height) * int(message.step)
        if raw.size < required:
            self.get_logger().warning(
                f"Image data is too short: received={raw.size}, required={required}"
            )
            return
        rows = raw[:required].reshape(int(message.height), int(message.step))
        frame = rows[:, :expected_row_bytes].reshape(int(message.height), int(message.width), channels)
        frame = frame[:, :, :3]
        if encoding in ("bgr8", "bgra8"):
            frame = frame[:, :, ::-1]
        elif encoding not in ("rgb8", "rgba8"):
            self.get_logger().warning(f"Unsupported image encoding: {message.encoding}")
            return
        self.latest_image = np.ascontiguousarray(frame)

    def _on_episode_state(self, message: EpisodeState) -> None:
        self.latest_episode_state = message

    def _wait_for_future(self, future, timeout: float, label: str):
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done():
            raise TimeoutError(f"Timed out waiting for {label}")
        if future.exception() is not None:
            raise RuntimeError(f"{label} failed: {future.exception()}")
        return future.result()

    def wait_for_environment(self) -> None:
        deadline = time.monotonic() + self.args.service_timeout
        while rclpy.ok():
            reset_ready = self.reset_client.wait_for_service(timeout_sec=0.1)
            end_ready = self.end_client.wait_for_service(timeout_sec=0.1)
            command_connected = self.command_publisher.get_subscription_count() > 0
            image_ready = self.latest_image is not None
            joints_ready = self.latest_joint_state is not None
            if reset_ready and end_ready and command_connected and image_ready and joints_ready:
                self.get_logger().info("Isaac Sim services, camera, joint state, and command input are ready.")
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Isaac Sim was not ready. Required services/topics: "
                    f"{RESET_SERVICE}, {END_SERVICE}, {CAMERA_RGB_TOPIC}, "
                    f"{JOINT_STATE_TOPIC}, subscriber on {JOINT_COMMAND_TOPIC}."
                )
            rclpy.spin_once(self, timeout_sec=0.05)

    def publish_task(self, task: str) -> None:
        message = String()
        message.data = task
        # Publish more than once so startup discovery cannot lose the command.
        for _ in range(3):
            self.task_publisher.publish(message)
            rclpy.spin_once(self, timeout_sec=0.05)

    def reset_episode(self, collection_index: int):
        request = ResetEpisode.Request()
        request.mode = MODE_RANDOM
        request.x = 0.0
        request.y = 0.0
        request.seed = -1 if self.args.base_seed < 0 else self.args.base_seed + collection_index
        request.record = bool(self.args.record)
        self.latest_episode_state = None
        response = self._wait_for_future(
            self.reset_client.call_async(request),
            self.args.service_timeout,
            "reset_episode",
        )
        if not response.accepted:
            raise RuntimeError(f"Reset rejected: {response.message}")
        self.active_episode_id = int(response.episode_id)
        return response

    def wait_until_ready(self, episode_id: int) -> None:
        deadline = time.monotonic() + self.args.observation_timeout
        while rclpy.ok():
            state = self.latest_episode_state
            if state is not None and int(state.episode_id) == episode_id:
                if int(state.phase) == PHASE_READY:
                    return
                if bool(state.terminated) or int(state.phase) in (PHASE_SUCCESS, PHASE_TERMINATED):
                    raise RuntimeError(
                        f"Episode terminated before inference: {state.termination_reason}"
                    )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Episode {episode_id} did not become READY")
            rclpy.spin_once(self, timeout_sec=0.05)

    def _terminal_state(self) -> Optional[EpisodeState]:
        state = self.latest_episode_state
        if state is None or self.active_episode_id is None:
            return None
        if int(state.episode_id) != self.active_episode_id:
            return None
        if bool(state.terminated) or int(state.phase) in (PHASE_SUCCESS, PHASE_TERMINATED):
            return state
        return None

    def infer_action(self, task: str) -> tuple[np.ndarray, bool]:
        if self.latest_image is None or self.latest_joint_state is None:
            raise RuntimeError("Image or joint state is unavailable")
        queue_was_empty = len(self.policy._queues["action"]) == 0
        observation = {
            "observation.images.front": self.latest_image.copy(),
            "observation.state": self.latest_joint_state.copy(),
        }
        batch = prepare_observation_for_inference(
            observation,
            device=self.device,
            task=task,
            robot_type="dofbot",
        )
        batch = self.preprocessor(batch)
        with torch.inference_mode():
            action = self.policy.select_action(batch)
            action = self.postprocessor(action)
        action_np = action.squeeze(0).detach().to("cpu").float().numpy()
        if action_np.shape != (len(JOINT_NAMES),):
            raise RuntimeError(f"Expected 11 policy actions, received shape {action_np.shape}")
        if not np.all(np.isfinite(action_np)):
            raise RuntimeError(f"Policy returned non-finite action: {action_np}")
        if self.args.max_action_delta > 0 and self.latest_joint_state is not None:
            delta = np.clip(
                action_np - self.latest_joint_state,
                -self.args.max_action_delta,
                self.args.max_action_delta,
            )
            action_np = self.latest_joint_state + delta
        return action_np.astype(np.float64), queue_was_empty

    def publish_action(self, action: np.ndarray) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = action.astype(float).tolist()
        message.velocity = []
        message.effort = []
        self.command_publisher.publish(message)

    def end_episode(self, reason: str, truncated: bool):
        request = EndEpisode.Request()
        request.reason = reason
        request.truncated = bool(truncated)
        response = self._wait_for_future(
            self.end_client.call_async(request),
            self.args.service_timeout,
            "end_episode",
        )
        if not response.accepted:
            raise RuntimeError(f"EndEpisode rejected: {response.message}")
        return response

    def run_one_episode(self, collection_index: int, color: str) -> EvaluationResult:
        task = f"pick up a {color} box."
        self.get_logger().info(
            f"========== inference {collection_index + 1}/{self.args.episodes}: {task} =========="
        )
        self.publish_task(task)
        reset_response = self.reset_episode(collection_index)
        episode_id = int(reset_response.episode_id)
        self.wait_until_ready(episode_id)

        # Clear the action queue whenever the simulator is reset.
        self.policy.reset()
        start = time.monotonic()
        next_control = start
        control_steps = 0
        inference_calls = 0
        terminal: Optional[EpisodeState] = None

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0)
            terminal = self._terminal_state()
            if terminal is not None:
                break
            now = time.monotonic()
            if now - start >= self.args.episode_timeout:
                end_response = self.end_episode("smolvla_inference_timeout", truncated=True)
                return EvaluationResult(
                    collection_index=collection_index,
                    episode_id=int(end_response.episode_id),
                    target_color=color,
                    task=task,
                    environment_seed=int(reset_response.seed_used),
                    success=bool(end_response.success),
                    truncated=True,
                    reason=str(end_response.message),
                    elapsed_sec=time.monotonic() - start,
                    control_steps=control_steps,
                    inference_calls=inference_calls,
                )
            if now >= next_control:
                action, did_infer = self.infer_action(task)
                self.publish_action(action)
                control_steps += 1
                inference_calls += int(did_infer)
                next_control = max(next_control + 1.0 / self.args.control_hz, time.monotonic())
            wait = min(max(next_control - time.monotonic(), 0.0), 0.02)
            rclpy.spin_once(self, timeout_sec=wait)

        assert terminal is not None
        return EvaluationResult(
            collection_index=collection_index,
            episode_id=int(terminal.episode_id),
            target_color=color,
            task=task,
            environment_seed=int(reset_response.seed_used),
            success=bool(terminal.success),
            truncated=bool(terminal.truncated),
            reason=str(terminal.termination_reason),
            elapsed_sec=time.monotonic() - start,
            control_steps=control_steps,
            inference_calls=inference_calls,
        )

    def run(self) -> list[EvaluationResult]:
        self.wait_for_environment()
        rng = random.Random(None if self.args.target_seed < 0 else self.args.target_seed)
        results: list[EvaluationResult] = []
        for index in range(self.args.episodes):
            color = self.args.target_color
            if color == "random":
                color = rng.choice(COLOR_NAMES)
            result = self.run_one_episode(index, color)
            results.append(result)
            self.get_logger().info(
                f"Episode {result.episode_id}: color={result.target_color}, "
                f"success={result.success}, reason={result.reason}, "
                f"steps={result.control_steps}, model_calls={result.inference_calls}"
            )
        return results


def write_results(args: argparse.Namespace, checkpoint: Path, results: list[EvaluationResult]) -> Path:
    result_path = args.result_path
    if result_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = Path.home() / "dofbot" / "eval_results" / f"smolvla_eval_{stamp}.json"
    result_path = result_path.expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    success_count = sum(result.success for result in results)
    payload = {
        "checkpoint": str(checkpoint),
        "created_at": datetime.now().astimezone().isoformat(),
        "episodes": len(results),
        "successes": success_count,
        "failures": len(results) - success_count,
        "success_rate": success_count / len(results) if results else math.nan,
        "settings": {
            "target_color": args.target_color,
            "base_seed": args.base_seed,
            "target_seed": args.target_seed,
            "control_hz": args.control_hz,
            "replan_steps": args.replan_steps,
            "record": args.record,
            "max_action_delta": args.max_action_delta,
        },
        "results": [asdict(result) for result in results],
    }
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_arguments() if argv is None else parse_arguments()
    rclpy.init(args=[])
    node: Optional[SmolVLAInferenceNode] = None
    try:
        node = SmolVLAInferenceNode(args)
        results = node.run()
        path = write_results(args, node.checkpoint, results)
        successes = sum(result.success for result in results)
        print("\n========== SmolVLA evaluation complete ==========")
        print(f"checkpoint   : {node.checkpoint}")
        print(f"successes    : {successes}/{len(results)}")
        print(f"result JSON  : {path}")
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
