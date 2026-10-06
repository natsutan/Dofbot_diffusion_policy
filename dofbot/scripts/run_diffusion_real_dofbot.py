#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
import random
import numpy as np
import torch

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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


class DofbotDiffusionRunner(Node):
    def __init__(self) -> None:
        super().__init__("dofbot_diffusion_runner")
        self.latest_q: np.ndarray | None = None
        self.latest_q_time: float | None = None

        self.publisher = self.create_publisher(
            JointState, "/joint_command", 10
        )
        self.subscription = self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, 10
        )

    def _joint_state_callback(self, msg: JointState) -> None:
        if msg.name:
            if len(msg.name) != len(msg.position):
                self.get_logger().warning(
                    "Ignored /joint_states: name and position lengths differ"
                )
                return
            by_name = dict(zip(msg.name, msg.position))
            missing = [name for name in JOINT_NAMES if name not in by_name]
            if missing:
                self.get_logger().warning(
                    f"Ignored /joint_states: missing joints {missing}"
                )
                return
            q = [by_name[name] for name in JOINT_NAMES]
        elif len(msg.position) == 11:
            q = list(msg.position)
        else:
            self.get_logger().warning(
                f"Ignored /joint_states: expected 11 positions, got {len(msg.position)}"
            )
            return

        q_np = np.asarray(q, dtype=np.float32)
        if not np.isfinite(q_np).all():
            self.get_logger().warning(
                "Ignored /joint_states containing NaN or infinity"
            )
            return

        self.latest_q = q_np
        self.latest_q_time = time.monotonic()

    def publish_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (11,):
            raise ValueError(f"Expected action shape (11,), got {action.shape}")

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = [float(x) for x in action]
        msg.velocity = []
        msg.effort = []
        self.publisher.publish(msg)


def resolve_model_path(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
    else:
        latest_file = Path("/home/natu/dofbot_data/latest_diffusion_run.txt")
        if not latest_file.is_file():
            raise FileNotFoundError(
                f"{latest_file} was not found. Specify --model-path explicitly."
            )

        run_dir = Path(
            latest_file.read_text(encoding="utf-8").strip()
        ).expanduser()

        path = (
            run_dir / "checkpoints" / "last" / "pretrained_model"
        ).resolve()

    if not path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {path}")

    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a trained LeRobot Diffusion Policy on the real DOFBOT through ROS 2."
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--cube-x", type=float, default=0.20)
    parser.add_argument("--cube-y", type=float, default=-0.05)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=130)
    parser.add_argument("--joint-state-timeout", type=float, default=1.0)
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    model_path = resolve_model_path(args.model_path)

    print("========== DOFBOT Diffusion Policy ==========")
    print(f"Model : {model_path}")
    print(f"Cube  : x={args.cube_x:.4f}, y={args.cube_y:.4f}")
    print(f"Loop  : {args.fps:.1f} Hz")
    print(f"Steps : {args.steps}")

    print("\nLoading Diffusion Policy...")
    policy = DiffusionPolicy.from_pretrained(str(model_path))
    policy.diffusion.num_inference_steps = 10
    

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(model_path),
    )

    print(f"Device         : {policy.config.device}")
    print(f"n_obs_steps    : {policy.config.n_obs_steps}")
    print(f"horizon        : {policy.config.horizon}")
    print(f"n_action_steps : {policy.config.n_action_steps}")

    rclpy.init()
    node = DofbotDiffusionRunner()

    try:
        print("\nWaiting for /joint_states...")
        while rclpy.ok() and node.latest_q is None:
            rclpy.spin_once(node, timeout_sec=0.2)

        if node.latest_q is None:
            raise RuntimeError("No /joint_states received")

        print("Initial joint state:")
        print(np.round(node.latest_q, 4))

        policy.reset()

        print("\nStarting inference. Ctrl+C to stop.")

        period = 1.0 / args.fps
        next_tick = time.monotonic()

        for step in range(args.steps):
            rclpy.spin_once(node, timeout_sec=0.0)

            if node.latest_q is None or node.latest_q_time is None:
                raise RuntimeError("Lost /joint_states")

            state_age = time.monotonic() - node.latest_q_time
            if state_age > args.joint_state_timeout:
                raise RuntimeError(
                    f"/joint_states is stale: {state_age:.3f} s"
                )

            raw_q = node.latest_q.copy()

            # Diagnostic sim-to-real state correction.
            # Apply only to the observation seen by the policy.
            # Do NOT modify /joint_states or /joint_command.
            policy_q = raw_q.copy()

            policy_q[1] += 0.0527  # arm2
            policy_q[2] += 0.0284  # arm3

            observation = {
                "observation.state": torch.from_numpy(policy_q),
                "observation.environment_state": torch.tensor(
                    [args.cube_x, args.cube_y],
                    dtype=torch.float32,
                ),
            }

            observation = preprocessor(observation)

            inference_start = time.perf_counter()
            with torch.inference_mode():
                action = policy.select_action(observation)
                action = postprocessor(action)

            inference_ms = (
                time.perf_counter() - inference_start
            ) * 1000.0

            action_np = (
                action.squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            node.publish_action(action_np)

            if step < 10 or step % 10 == 0 or inference_ms > 100.0:
                print(
                    f"step={step:03d} "
                    f"infer={inference_ms:7.1f} ms "
                    f"raw_arm={np.round(raw_q[:5], 3)} "
                    f"policy_arm={np.round(policy_q[:5], 3)} "
                    f"action_arm={np.round(action_np[:5], 3)} "
                    f"gripper={np.round(action_np[5:], 3)}"
                )

            next_tick = time.monotonic() + period

            while rclpy.ok():
                remaining = next_tick - time.monotonic()
                if remaining <= 0:
                    break
                rclpy.spin_once(
                    node,
                    timeout_sec=min(remaining, 0.02),
                )

        print("\nDiffusion Policy run finished.")

    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.")

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
