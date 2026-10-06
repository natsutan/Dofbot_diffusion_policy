#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

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
        description="Compare 2-observation histories at Diffusion Policy replanning steps."
    )
    p.add_argument("--b-bag", required=True, type=Path,
                   help="B: real closed-loop Diffusion Policy failed run")
    p.add_argument("--c-bag", required=True, type=Path,
                   help="C: real successful replay of Isaac commands")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--chunk-size", type=int, default=8)
    p.add_argument("--max-step", type=int, default=64)
    return p.parse_args()


def resolve_storage(path: Path):
    if path.is_dir():
        mcaps = sorted(path.glob("*.mcap"))
        if len(mcaps) == 1:
            return mcaps[0], "mcap"
        db3s = sorted(path.glob("*.db3"))
        if len(db3s) == 1:
            return db3s[0], "sqlite3"
        raise RuntimeError(f"Expected exactly one .mcap or .db3 in {path}")

    if path.suffix.lower() == ".mcap":
        return path, "mcap"
    if path.suffix.lower() == ".db3":
        return path, "sqlite3"
    raise RuntimeError(f"Unsupported bag: {path}")


def joint_vector(msg):
    if not msg.position:
        return None

    if msg.name:
        d = dict(zip(msg.name, msg.position))
        if not all(n in d for n in JOINT_NAMES):
            return None
        return np.asarray([d[n] for n in JOINT_NAMES], dtype=np.float64)

    if len(msg.position) == len(JOINT_NAMES):
        return np.asarray(msg.position, dtype=np.float64)

    return None


def read_bag(path: Path):
    storage_path, storage_id = resolve_storage(path)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(storage_path), storage_id=storage_id),
        rosbag2_py.ConverterOptions("", ""),
    )

    types = {x.name: x.type for x in reader.get_all_topics_and_types()}
    for topic in (STATE_TOPIC, COMMAND_TOPIC):
        if topic not in types:
            raise RuntimeError(f"{topic} not found in {path}")

    reader.set_filter(
        rosbag2_py.StorageFilter(topics=[STATE_TOPIC, COMMAND_TOPIC])
    )

    state_t, state_q = [], []
    cmd_t, cmd_q = [], []

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        msg = deserialize_message(data, get_message(types[topic]))
        q = joint_vector(msg)
        if q is None:
            continue

        if topic == STATE_TOPIC:
            state_t.append(int(t_ns))
            state_q.append(q)
        else:
            cmd_t.append(int(t_ns))
            cmd_q.append(q)

    state_t = np.asarray(state_t, dtype=np.int64)
    state_q = np.asarray(state_q, dtype=np.float64)
    cmd_t = np.asarray(cmd_t, dtype=np.int64)
    cmd_q = np.asarray(cmd_q, dtype=np.float64)

    if len(cmd_t) < 2:
        raise RuntimeError(f"Too few command messages in {path}")

    # Approximate the observation available to the policy at each control step
    # by taking the latest /joint_states message received before that command.
    obs = []
    age = []

    for t in cmd_t:
        idx = int(np.searchsorted(state_t, t, side="right") - 1)
        if idx < 0:
            idx = 0
        obs.append(state_q[idx])
        age.append((t - state_t[idx]) / 1e9)

    return {
        "obs": np.asarray(obs, dtype=np.float64),
        "action": cmd_q,
        "age": np.asarray(age, dtype=np.float64),
    }


def fmt(v):
    return "[" + " ".join(f"{x:+.3f}" for x in v) + "]"


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    B = read_bag(args.b_bag)
    C = read_bag(args.c_bag)

    n = min(len(B["obs"]), len(C["obs"]))
    max_step = min(args.max_step, n - 1)

    steps = [
        s for s in range(args.chunk_size, max_step + 1, args.chunk_size)
    ]

    rows = []

    print(
        "step | B age(ms) C age(ms) | "
        "pos diff arm2/3 | motion B arm2/3 | motion C arm2/3 | motion-diff"
    )
    print("-" * 122)

    for s in steps:
        # The 2-observation history used when replanning at s is approximately
        # obs[s-1], obs[s].
        b_prev = B["obs"][s - 1, :5]
        b_now = B["obs"][s, :5]
        c_prev = C["obs"][s - 1, :5]
        c_now = C["obs"][s, :5]

        b_motion = b_now - b_prev
        c_motion = c_now - c_prev

        pos_diff = b_now - c_now
        motion_diff = b_motion - c_motion

        pos23 = float(np.linalg.norm(pos_diff[1:3]))
        motion23 = float(np.linalg.norm(motion_diff[1:3]))

        # Similarity of the observed recent motion directions.
        nb = np.linalg.norm(b_motion[1:3])
        nc = np.linalg.norm(c_motion[1:3])
        if nb > 1e-12 and nc > 1e-12:
            motion_cos = float(np.dot(b_motion[1:3], c_motion[1:3]) / (nb * nc))
        else:
            motion_cos = float("nan")

        print(
            f"{s:4d} | "
            f"{B['age'][s]*1000:8.1f} {C['age'][s]*1000:8.1f} | "
            f"{pos23:7.4f} | "
            f"{fmt(b_motion[1:3])} | "
            f"{fmt(c_motion[1:3])} | "
            f"{motion23:7.4f}  cos={motion_cos:+.3f}"
        )

        rows.append({
            "step": s,
            "b_state_age_ms": B["age"][s] * 1000.0,
            "c_state_age_ms": C["age"][s] * 1000.0,

            "b_prev_arm2": b_prev[1],
            "b_prev_arm3": b_prev[2],
            "b_now_arm2": b_now[1],
            "b_now_arm3": b_now[2],

            "c_prev_arm2": c_prev[1],
            "c_prev_arm3": c_prev[2],
            "c_now_arm2": c_now[1],
            "c_now_arm3": c_now[2],

            "b_motion_arm2": b_motion[1],
            "b_motion_arm3": b_motion[2],
            "c_motion_arm2": c_motion[1],
            "c_motion_arm3": c_motion[2],

            "position_distance_arm23_rad": pos23,
            "motion_distance_arm23_rad": motion23,
            "motion_cosine_arm23": motion_cos,

            "b_action_arm2": B["action"][s, 1],
            "b_action_arm3": B["action"][s, 2],
            "c_action_arm2": C["action"][s, 1],
            "c_action_arm3": C["action"][s, 2],
        })

    csv_path = args.out / "observation_history_compare.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nFocus points:")
    print("  step 24: last clearly successful-direction replan")
    print("  step 32: first replan where B action turns away from successful C path")
    print("")
    print("Interpretation:")
    print("  position_distance: current B/C arm2-arm3 state difference.")
    print("  motion B/C:        recent obs[s] - obs[s-1], i.e. what n_obs_steps=2 exposes.")
    print("  motion cosine ~+1: same recent motion direction.")
    print("  motion cosine ~ 0: very different/near-stationary motion.")
    print("  motion cosine < 0: opposite recent motion directions.")
    print("")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
