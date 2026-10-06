#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


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

STATE_TOPIC = "/joint_states"
COMMAND_TOPIC = "/joint_command"


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare Isaac Sim and real DOFBOT ROS 2 bags step-by-step."
    )
    p.add_argument("--isaac", required=True, type=Path, help="Isaac Sim bag directory")
    p.add_argument("--real", required=True, type=Path, help="Real robot bag directory")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("dp_bag_compare"),
        help="Output directory",
    )
    p.add_argument(
        "--threshold-rad",
        type=float,
        default=0.05,
        help="Difference threshold used when reporting first divergence",
    )
    return p.parse_args()


def joint_vector(msg):
    if len(msg.position) == 0:
        return None

    if msg.name:
        by_name = dict(zip(msg.name, msg.position))
        if not all(name in by_name for name in JOINT_NAMES):
            return None
        return np.asarray([by_name[name] for name in JOINT_NAMES], dtype=np.float64)

    if len(msg.position) == len(JOINT_NAMES):
        return np.asarray(msg.position, dtype=np.float64)

    return None


def resolve_storage(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    # Prefer opening the MCAP file directly. This bypasses a possibly incomplete
    # metadata.yaml and lets the MCAP plugin recover channel/schema information
    # from the file itself.
    if path.is_dir():
        mcap_files = sorted(path.glob("*.mcap"))
        if len(mcap_files) == 1:
            return mcap_files[0], "mcap"
        if len(mcap_files) > 1:
            raise RuntimeError(
                f"Multiple MCAP files found in {path}; split bags are not supported "
                f"by this comparison script yet: {mcap_files}"
            )

        db3_files = sorted(path.glob("*.db3"))
        if len(db3_files) == 1:
            return db3_files[0], "sqlite3"
        if len(db3_files) > 1:
            raise RuntimeError(
                f"Multiple DB3 files found in {path}; split bags are not supported "
                f"by this comparison script yet: {db3_files}"
            )

        raise RuntimeError(f"No .mcap or .db3 file found in bag directory: {path}")

    suffix = path.suffix.lower()
    if suffix == ".mcap":
        return path, "mcap"
    if suffix == ".db3":
        return path, "sqlite3"

    raise RuntimeError(f"Unsupported bag path: {path}")


def read_bag(path: Path):
    storage_path, storage_id = resolve_storage(path)
    print(f"  opening storage file: {storage_path}")
    print(f"  storage id         : {storage_id}")

    reader = rosbag2_py.SequentialReader()

    storage_options = rosbag2_py.StorageOptions(
        uri=str(storage_path),
        storage_id=storage_id,
    )
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)

    topic_types = {
        x.name: x.type for x in reader.get_all_topics_and_types()
    }

    for required in (STATE_TOPIC, COMMAND_TOPIC):
        if required not in topic_types:
            raise RuntimeError(f"{required} not found in bag: {path}")

    reader.set_filter(
        rosbag2_py.StorageFilter(topics=[STATE_TOPIC, COMMAND_TOPIC])
    )

    timestamps = {STATE_TOPIC: [], COMMAND_TOPIC: []}
    positions = {STATE_TOPIC: [], COMMAND_TOPIC: []}

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        msg_type = get_message(topic_types[topic])
        msg = deserialize_message(data, msg_type)
        q = joint_vector(msg)
        if q is None:
            continue

        timestamps[topic].append(int(timestamp_ns))
        positions[topic].append(q)

    result = {}
    for topic in (STATE_TOPIC, COMMAND_TOPIC):
        result[topic] = {
            "t_ns": np.asarray(timestamps[topic], dtype=np.int64),
            "q": np.asarray(positions[topic], dtype=np.float64),
        }

    return result


def sample_state_before_commands(data):
    state_t = data[STATE_TOPIC]["t_ns"]
    state_q = data[STATE_TOPIC]["q"]
    cmd_t = data[COMMAND_TOPIC]["t_ns"]
    cmd_q = data[COMMAND_TOPIC]["q"]

    if len(state_t) == 0 or len(cmd_t) == 0:
        raise RuntimeError("Bag contains no usable state or command messages")

    paired_state = []
    selected_state_t = []

    for t in cmd_t:
        # State available immediately before this command.
        idx = int(np.searchsorted(state_t, t, side="right") - 1)

        # Normally a prior state exists because recording starts before inference.
        # If not, use the first state so the first command is still represented.
        if idx < 0:
            idx = 0

        paired_state.append(state_q[idx])
        selected_state_t.append(state_t[idx])

    t0 = int(cmd_t[0])

    return {
        "command_t": (cmd_t - t0) / 1e9,
        "state_age": (cmd_t - np.asarray(selected_state_t, dtype=np.int64)) / 1e9,
        "state": np.asarray(paired_state, dtype=np.float64),
        "action": cmd_q,
    }


def gripper_g(q):
    return (-q[:, 5] + q[:, 6] - q[:, 7] + q[:, 8] - q[:, 9] + q[:, 10]) / 6.0


def write_csv(path: Path, isaac, real):
    n = min(len(isaac["action"]), len(real["action"]))

    header = [
        "step",
        "isaac_time_sec",
        "real_time_sec",
        "isaac_state_age_sec",
        "real_state_age_sec",
    ]

    for prefix in ("isaac_state", "real_state", "state_diff",
                   "isaac_action", "real_action", "action_diff"):
        header.extend(f"{prefix}_{name}" for name in JOINT_NAMES)

    header.extend([
        "isaac_state_gripper_g",
        "real_state_gripper_g",
        "isaac_action_gripper_g",
        "real_action_gripper_g",
        "arm_state_max_abs_diff",
        "arm_action_max_abs_diff",
    ])

    isaac_state_g = gripper_g(isaac["state"][:n])
    real_state_g = gripper_g(real["state"][:n])
    isaac_action_g = gripper_g(isaac["action"][:n])
    real_action_g = gripper_g(real["action"][:n])

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i in range(n):
            state_diff = real["state"][i] - isaac["state"][i]
            action_diff = real["action"][i] - isaac["action"][i]

            row = [
                i,
                isaac["command_t"][i],
                real["command_t"][i],
                isaac["state_age"][i],
                real["state_age"][i],
            ]
            row.extend(isaac["state"][i])
            row.extend(real["state"][i])
            row.extend(state_diff)
            row.extend(isaac["action"][i])
            row.extend(real["action"][i])
            row.extend(action_diff)
            row.extend([
                isaac_state_g[i],
                real_state_g[i],
                isaac_action_g[i],
                real_action_g[i],
                np.max(np.abs(state_diff[:5])),
                np.max(np.abs(action_diff[:5])),
            ])
            writer.writerow(row)


def plot_error(path: Path, title: str, values, ylabel: str, threshold: float):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(values)), values, label=title)
    ax.axhline(threshold, linestyle="--", label=f"threshold={threshold:.3f} rad")
    ax.set_xlabel("Policy step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gripper(path: Path, isaac, real):
    n = min(len(isaac["action"]), len(real["action"]))
    isaac_g = gripper_g(isaac["action"][:n])
    real_g = gripper_g(real["action"][:n])

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(n), isaac_g, label="Isaac action gripper g")
    ax.plot(np.arange(n), real_g, linestyle="--", label="Real action gripper g")
    ax.set_xlabel("Policy step")
    ax.set_ylabel("g")
    ax.set_title("Diffusion Policy gripper command")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize(label, data):
    print(
        f"{label}: commands={len(data['action'])}, "
        f"paired_states={len(data['state'])}, "
        f"median_state_age={np.median(data['state_age'])*1000.0:.1f} ms, "
        f"max_state_age={np.max(data['state_age'])*1000.0:.1f} ms"
    )


def first_over_threshold(values, threshold):
    indices = np.flatnonzero(values > threshold)
    return None if len(indices) == 0 else int(indices[0])


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Reading Isaac bag: {args.isaac}")
    isaac = sample_state_before_commands(read_bag(args.isaac))

    print(f"Reading real bag : {args.real}")
    real = sample_state_before_commands(read_bag(args.real))

    summarize("Isaac", isaac)
    summarize("Real ", real)

    n = min(len(isaac["action"]), len(real["action"]))
    if n == 0:
        raise RuntimeError("No common command steps")

    if len(isaac["action"]) != len(real["action"]):
        print(
            f"WARNING: command counts differ; comparing first {n} steps "
            f"(Isaac={len(isaac['action'])}, Real={len(real['action'])})"
        )

    state_diff = real["state"][:n] - isaac["state"][:n]
    action_diff = real["action"][:n] - isaac["action"][:n]

    state_arm_max = np.max(np.abs(state_diff[:, :5]), axis=1)
    action_arm_max = np.max(np.abs(action_diff[:, :5]), axis=1)

    csv_path = args.out / "comparison.csv"
    write_csv(csv_path, isaac, real)

    plot_error(
        args.out / "arm_state_error.png",
        "Isaac vs Real: arm state max absolute difference",
        state_arm_max,
        "max |state difference| [rad]",
        args.threshold_rad,
    )
    plot_error(
        args.out / "arm_action_error.png",
        "Isaac vs Real: arm action max absolute difference",
        action_arm_max,
        "max |action difference| [rad]",
        args.threshold_rad,
    )
    plot_gripper(args.out / "gripper_action.png", isaac, real)

    print("\n========== Comparison summary ==========")
    print(f"Compared steps: {n}")
    print(f"Threshold     : {args.threshold_rad:.4f} rad")

    state_first = first_over_threshold(state_arm_max, args.threshold_rad)
    action_first = first_over_threshold(action_arm_max, args.threshold_rad)

    print(
        "First arm state divergence : "
        + ("none" if state_first is None else f"step {state_first}")
    )
    print(
        "First arm action divergence: "
        + ("none" if action_first is None else f"step {action_first}")
    )

    print("\nPer-joint mean/max absolute difference [rad]:")
    for i, name in enumerate(JOINT_NAMES[:5]):
        s = np.abs(state_diff[:, i])
        a = np.abs(action_diff[:, i])
        print(
            f"{name:12s} "
            f"state mean={np.mean(s):.4f} max={np.max(s):.4f} | "
            f"action mean={np.mean(a):.4f} max={np.max(a):.4f}"
        )

    print("\nOutput:")
    print(f"  {csv_path}")
    print(f"  {args.out / 'arm_state_error.png'}")
    print(f"  {args.out / 'arm_action_error.png'}")
    print(f"  {args.out / 'gripper_action.png'}")


if __name__ == "__main__":
    main()
