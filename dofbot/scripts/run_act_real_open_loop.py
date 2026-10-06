#!/usr/bin/env python3
"""
Open-loop ACT inference for the real DOFBOT.

Purpose:
- Do NOT subscribe to /joint_states.
- Start from the same assumed HOME state used for training.
- After publishing each ACT action, assume the robot reached that commanded state.
- Feed that commanded state back as observation.state for the next policy query.

This is a diagnostic mode to separate:
  ACT inference / normalization / ROS command / servo calibration
from:
  real servo readback / joint_states feedback.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


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

# Training/Isaac HOME used in the handover.
TRAINING_HOME = np.array(
    [0.8, 0.3, -0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)


class ActOpenLoopRunner(Node):
    def __init__(self) -> None:
        super().__init__("dofbot_act_open_loop_runner")
        self.publisher = self.create_publisher(JointState, "/joint_command", 10)

    def publish_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (11,):
            raise ValueError(f"Expected action shape (11,), got {action.shape}")

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = [float(v) for v in action]
        msg.velocity = []
        msg.effort = []
        self.publisher.publish(msg)


def resolve_model_path(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
    else:
        latest_file = Path.home() / "dofbot/latest_act_run.txt"
        if not latest_file.is_file():
            raise FileNotFoundError(
                f"{latest_file} was not found. Use --model-path explicitly."
            )

        run_dir = Path(latest_file.read_text().strip()).expanduser()
        path = (run_dir / "checkpoints/last/pretrained_model").resolve()

    if not path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {path}")

    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--cube-x", type=float, default=0.20)
    parser.add_argument("--cube-y", type=float, default=-0.05)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--steps",
        type=int,
        default=130,
        help="Number of action steps to send. 130 is close to the recorded teacher episode length.",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=3.0,
        help="Seconds to wait before sending the first ACT action.",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    model_path = resolve_model_path(args.model_path)

    print("========== Open-loop ACT diagnostic ==========")
    print(f"Model        : {model_path}")
    print(f"Cube         : x={args.cube_x:.3f}, y={args.cube_y:.3f}")
    print(f"Control rate : {args.fps:.1f} Hz")
    print(f"Steps        : {args.steps}")
    print(f"Initial state: {TRAINING_HOME.tolist()}")
    print("NOTE: /joint_states is NOT used.")
    print("NOTE: observation.state is updated from the previous ACT command.")

    policy = ACTPolicy.from_pretrained(str(model_path))
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(model_path),
    )
    policy.reset()

    rclpy.init()
    node = ActOpenLoopRunner()

    # This is deliberately NOT measured from the robot.
    assumed_state = TRAINING_HOME.copy()

    try:
        # Publish HOME briefly first. The Pi-side driver performs the
        # Isaac/ACT -> servo calibration/conversion.
        print("\nPublishing training HOME before ACT inference...")
        home_end = time.monotonic() + 2.0
        while time.monotonic() < home_end:
            node.publish_action(assumed_state)
            rclpy.spin_once(node, timeout_sec=0.01)
            time.sleep(0.09)

        print(f"Starting in {args.start_delay:.1f} s. Ctrl+C to stop.")
        time.sleep(args.start_delay)

        period = 1.0 / args.fps
        next_tick = time.monotonic()

        for step in range(args.steps):
            observation = {
                "observation.state": torch.from_numpy(assumed_state.copy()),
                "observation.environment_state": torch.tensor(
                    [args.cube_x, args.cube_y],
                    dtype=torch.float32,
                ),
            }

            observation = preprocessor(observation)

            with torch.inference_mode():
                action = policy.select_action(observation)
                action = postprocessor(action)

            action_np = (
                action.squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if action_np.shape != (11,):
                raise RuntimeError(
                    f"ACT returned unexpected action shape: {action_np.shape}"
                )

            node.publish_action(action_np)

            # Key diagnostic assumption:
            # treat the command as if the real robot reached it exactly.
            assumed_state = action_np.copy()

            if step < 20 or step % 10 == 0:
                print(
                    f"step={step:03d} "
                    f"action_arm={np.round(action_np[:5], 4)} "
                    f"gripper={np.round(action_np[5:], 4)}"
                )

            next_tick += period
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

        print("\nOpen-loop ACT run finished.")

    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
