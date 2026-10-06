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


class ActRealRunner(Node):
    def __init__(self) -> None:
        super().__init__("dofbot_act_real_runner")
        self.latest_q: np.ndarray | None = None
        self.latest_q_time: float | None = None

        self.pub = self.create_publisher(JointState, "/joint_command", 10)
        self.sub = self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_cb,
            1,
        )

    def _joint_state_cb(self, msg: JointState) -> None:
        if len(msg.name) != len(msg.position):
            self.get_logger().warning("Invalid /joint_states: name/position length mismatch")
            return

        pos_by_name = {name: pos for name, pos in zip(msg.name, msg.position)}
        missing = [name for name in JOINT_NAMES if name not in pos_by_name]
        if missing:
            self.get_logger().warning(f"/joint_states missing joints: {missing}")
            return

        self.latest_q = np.asarray(
            [pos_by_name[name] for name in JOINT_NAMES],
            dtype=np.float32,
        )
        self.latest_q_time = time.monotonic()

    def publish_action(self, action: np.ndarray) -> None:
        if action.shape != (11,):
            raise ValueError(f"Expected action shape (11,), got {action.shape}")

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = [float(v) for v in action]
        msg.velocity = []
        msg.effort = []
        self.pub.publish(msg)


def resolve_model_path(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
    else:
        latest_file = Path.home() / "dofbot/latest_act_run.txt"
        run_dir = Path(latest_file.read_text().strip()).expanduser()
        path = run_dir / "checkpoints/last/pretrained_model"

    if not path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--cube-x", type=float, default=0.20)
    parser.add_argument("--cube-y", type=float, default=-0.05)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=130)
    parser.add_argument("--joint-state-timeout", type=float, default=1.0)
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path)

    print(f"Model : {model_path}")
    print(f"Cube  : x={args.cube_x:.3f}, y={args.cube_y:.3f}")
    print(f"Loop  : {args.fps:.1f} Hz, {args.steps} steps")

    policy = ACTPolicy.from_pretrained(str(model_path))
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(model_path),
    )
    policy.reset()

    rclpy.init()
    node = ActRealRunner()

    try:
        print("Waiting for /joint_states from the real DOFBOT...")
        while rclpy.ok() and node.latest_q is None:
            rclpy.spin_once(node, timeout_sec=0.2)

        if node.latest_q is None:
            raise RuntimeError("No /joint_states received")

        print("Initial real joint state:")
        print(node.latest_q)
        print("Starting ACT inference... Ctrl+C to stop.")

        period = 1.0 / args.fps
        next_tick = time.monotonic()

        for step in range(args.steps):
            rclpy.spin_once(node, timeout_sec=0.0)

            now = time.monotonic()
            if node.latest_q is None or node.latest_q_time is None:
                raise RuntimeError("Lost /joint_states")
            age = now - node.latest_q_time
            if age > args.joint_state_timeout:
                raise RuntimeError(
                    f"/joint_states is stale: age={age:.3f}s "
                    f"(limit={args.joint_state_timeout:.3f}s)"
                )

            observation = {
                "observation.state": torch.from_numpy(node.latest_q.copy()),
                "observation.environment_state": torch.tensor(
                    [args.cube_x, args.cube_y],
                    dtype=torch.float32,
                ),
            }

            observation = preprocessor(observation)

            with torch.inference_mode():
                action = policy.select_action(observation)
                action = postprocessor(action)

            action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
            node.publish_action(action_np)

            if step % 10 == 0:
                print(
                    f"step={step:03d} "
                    f"state[0:5]={np.round(node.latest_q[:5], 3)} "
                    f"action[0:5]={np.round(action_np[:5], 3)}"
                )

            next_tick += period
            while rclpy.ok():
                remaining = next_tick - time.monotonic()
                if remaining <= 0:
                    break
                rclpy.spin_once(node, timeout_sec=min(remaining, 0.02))

        print("ACT run finished.")

    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
