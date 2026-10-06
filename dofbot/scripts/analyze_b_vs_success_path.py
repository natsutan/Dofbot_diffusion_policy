#!/usr/bin/env python3
import argparse
import csv
import re
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

LINE_RE = re.compile(
    r"step=(?P<step>\d+)\s+"
    r"infer=\s*(?P<infer>[0-9.]+)\s*ms\s+"
    r"state_arm=\[(?P<state>[^\]]+)\]\s+"
    r"action_arm=\[(?P<action>[^\]]+)\]\s+"
    r"gripper=\[(?P<gripper>[^\]]+)\]"
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compare failed real Diffusion-Policy replanning states (B) "
            "against the successful real replay trajectory (C)."
        )
    )
    p.add_argument("--b-log", required=True, type=Path)
    p.add_argument("--c-bag", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--inference-threshold-ms",
        type=float,
        default=20.0,
        help="Rows above this inference time are treated as replanning points.",
    )
    p.add_argument(
        "--ahead",
        type=int,
        default=8,
        help="C steps ahead used to define successful progress direction.",
    )
    p.add_argument(
        "--close-threshold",
        type=float,
        default=0.10,
        help="Gripper g threshold used to detect close start.",
    )
    return p.parse_args()


def vec(text):
    return np.fromstring(text, sep=" ", dtype=np.float64)


def load_b_log(path: Path, threshold_ms: float):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue

        infer_ms = float(m.group("infer"))
        if infer_ms < threshold_ms:
            continue

        state = vec(m.group("state"))
        action = vec(m.group("action"))
        gripper = vec(m.group("gripper"))

        if len(state) != 5 or len(action) != 5:
            continue

        rows.append(
            {
                "step": int(m.group("step")),
                "infer_ms": infer_ms,
                "state": state,
                "action": action,
                "gripper": gripper,
            }
        )

    if not rows:
        raise RuntimeError(f"No replanning rows found in B log: {path}")

    return rows


def resolve_storage(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir():
        mcaps = sorted(path.glob("*.mcap"))
        if len(mcaps) == 1:
            return mcaps[0], "mcap"
        db3s = sorted(path.glob("*.db3"))
        if len(db3s) == 1:
            return db3s[0], "sqlite3"
        if len(mcaps) > 1 or len(db3s) > 1:
            raise RuntimeError("Split bags are not supported.")
        raise RuntimeError(f"No .mcap or .db3 found in {path}")

    if path.suffix.lower() == ".mcap":
        return path, "mcap"
    if path.suffix.lower() == ".db3":
        return path, "sqlite3"

    raise RuntimeError(f"Unsupported bag path: {path}")


def joint_vector(msg):
    if not msg.position:
        return None

    if msg.name:
        by_name = dict(zip(msg.name, msg.position))
        if not all(name in by_name for name in JOINT_NAMES):
            return None
        return np.asarray([by_name[name] for name in JOINT_NAMES], dtype=np.float64)

    if len(msg.position) == len(JOINT_NAMES):
        return np.asarray(msg.position, dtype=np.float64)

    return None


def read_c_bag(path: Path):
    storage_path, storage_id = resolve_storage(path)
    print(f"Reading C bag: {path}")
    print(f"  storage file: {storage_path}")
    print(f"  storage id  : {storage_id}")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(storage_path), storage_id=storage_id),
        rosbag2_py.ConverterOptions("", ""),
    )

    topic_types = {x.name: x.type for x in reader.get_all_topics_and_types()}
    for topic in (STATE_TOPIC, COMMAND_TOPIC):
        if topic not in topic_types:
            raise RuntimeError(f"{topic} not found in C bag")

    reader.set_filter(
        rosbag2_py.StorageFilter(topics=[STATE_TOPIC, COMMAND_TOPIC])
    )

    state_t, state_q, cmd_t, cmd_q = [], [], [], []

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        msg_type = get_message(topic_types[topic])
        msg = deserialize_message(data, msg_type)
        q = joint_vector(msg)
        if q is None:
            continue

        if topic == STATE_TOPIC:
            state_t.append(int(timestamp_ns))
            state_q.append(q)
        else:
            cmd_t.append(int(timestamp_ns))
            cmd_q.append(q)

    state_t = np.asarray(state_t, dtype=np.int64)
    state_q = np.asarray(state_q, dtype=np.float64)
    cmd_t = np.asarray(cmd_t, dtype=np.int64)
    cmd_q = np.asarray(cmd_q, dtype=np.float64)

    if len(cmd_t) == 0 or len(state_t) == 0:
        raise RuntimeError("C bag has no usable command/state samples")

    c_state, c_state_age = [], []
    for t in cmd_t:
        idx = int(np.searchsorted(state_t, t, side="right") - 1)
        if idx < 0:
            idx = 0
        c_state.append(state_q[idx])
        c_state_age.append((t - state_t[idx]) / 1e9)

    return {
        "state": np.asarray(c_state, dtype=np.float64),
        "action": cmd_q,
        "state_age": np.asarray(c_state_age, dtype=np.float64),
    }


def gripper_g(q):
    return (
        -q[:, 5] + q[:, 6] - q[:, 7]
        + q[:, 8] - q[:, 9] + q[:, 10]
    ) / 6.0


def cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def find_first_close_step(c_action, threshold):
    g = gripper_g(c_action)
    idx = np.flatnonzero(g >= threshold)
    return int(idx[0]) if len(idx) else len(c_action) - 1


def analyze(b_rows, c, ahead, close_threshold):
    close_step = find_first_close_step(c["action"], close_threshold)

    # Limit nearest-path matching to the successful pre-close phase.
    c_states = c["state"][: close_step + 1, :5]

    results = []

    for b in b_rows:
        b_state = b["state"]
        b_action = b["action"]

        # Match with arm1..arm4. arm5 is nearly constant and a small zero
        # offset should not dominate the progress estimate.
        diff = c_states[:, :4] - b_state[:4]
        dist = np.linalg.norm(diff, axis=1)
        nearest = int(np.argmin(dist))

        arm23_dist = float(
            np.linalg.norm(c_states[nearest, 1:3] - b_state[1:3])
        )

        # Local successful command at the matched point.
        local_success_delta = (
            c["action"][nearest, :5] - c["state"][nearest, :5]
        )

        # Larger-scale successful direction over one chunk.
        future = min(nearest + ahead, close_step)
        chunk_success_delta = (
            c["state"][future, :5] - c["state"][nearest, :5]
        )

        b_delta = b_action - b_state

        local_cos = cosine(b_delta[:4], local_success_delta[:4])
        chunk_cos = cosine(b_delta[:4], chunk_success_delta[:4])

        chunk_norm = float(np.linalg.norm(chunk_success_delta[:4]))
        if chunk_norm < 1e-12:
            signed_progress = float("nan")
        else:
            signed_progress = float(
                np.dot(b_delta[:4], chunk_success_delta[:4]) / chunk_norm
            )

        results.append(
            {
                "b_step": b["step"],
                "b_infer_ms": b["infer_ms"],
                "nearest_c_step": nearest,
                "progress_error_steps": b["step"] - nearest,
                "distance_arm1_4_rad": float(dist[nearest]),
                "distance_arm2_3_rad": arm23_dist,
                "future_c_step": future,
                "local_direction_cosine": local_cos,
                "chunk_direction_cosine": chunk_cos,
                "signed_success_progress_rad": signed_progress,
                "b_delta_norm_rad": float(np.linalg.norm(b_delta[:4])),
                "c_local_delta_norm_rad": float(
                    np.linalg.norm(local_success_delta[:4])
                ),
                "c_chunk_delta_norm_rad": chunk_norm,
                "b_state_arm1": b_state[0],
                "b_state_arm2": b_state[1],
                "b_state_arm3": b_state[2],
                "b_state_arm4": b_state[3],
                "c_state_arm1": c["state"][nearest, 0],
                "c_state_arm2": c["state"][nearest, 1],
                "c_state_arm3": c["state"][nearest, 2],
                "c_state_arm4": c["state"][nearest, 3],
                "b_action_arm1": b_action[0],
                "b_action_arm2": b_action[1],
                "b_action_arm3": b_action[2],
                "b_action_arm4": b_action[3],
            }
        )

    return results, close_step


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_progress(path, rows, close_step):
    x = np.asarray([r["b_step"] for r in rows])
    y = np.asarray([r["nearest_c_step"] for r in rows])

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker="o", label="B state -> nearest C success step")
    ax.plot(x, x, linestyle="--", label="ideal: C step = B step")
    ax.axhline(
        close_step,
        linestyle=":",
        label=f"C gripper close starts at step {close_step}",
    )
    ax.set_xlabel("B: failed closed-loop policy step")
    ax.set_ylabel("Nearest C successful trajectory step")
    ax.set_title("B progress along the successful real trajectory")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distance(path, rows):
    x = np.asarray([r["b_step"] for r in rows])
    d14 = np.asarray([r["distance_arm1_4_rad"] for r in rows])
    d23 = np.asarray([r["distance_arm2_3_rad"] for r in rows])

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    ax.plot(x, d14, marker="o", label="arm1-4 path distance")
    ax.plot(x, d23, marker="o", label="arm2-3 distance")
    ax.set_xlabel("B policy step")
    ax.set_ylabel("Joint-space distance [rad]")
    ax.set_title("Distance from B state to successful C path")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_direction(path, rows):
    x = np.asarray([r["b_step"] for r in rows])
    local = np.asarray([r["local_direction_cosine"] for r in rows])
    chunk = np.asarray([r["chunk_direction_cosine"] for r in rows])

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    ax.plot(x, local, marker="o", label="B action vs C local command")
    ax.plot(x, chunk, marker="o", label="B action vs C 8-step progress")
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("B policy step")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Does B action point along the successful direction?")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_phase_arm23(path, rows, c, close_step):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)

    ax.plot(
        c["state"][: close_step + 1, 1],
        c["state"][: close_step + 1, 2],
        label="C successful path before gripper close",
    )

    bx = np.asarray([r["b_state_arm2"] for r in rows])
    by = np.asarray([r["b_state_arm3"] for r in rows])
    ax.plot(bx, by, marker="o", linestyle="--", label="B replanning states")

    for r in rows:
        ax.annotate(
            str(r["b_step"]),
            (r["b_state_arm2"], r["b_state_arm3"]),
            fontsize=8,
        )

    ax.set_xlabel("arm2 [rad]")
    ax.set_ylabel("arm3 [rad]")
    ax.set_title("arm2/arm3 phase path: failed B vs successful C")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_summary(rows, close_step):
    print("\n========== B vs successful C path ==========")
    print(f"C gripper close starts around step: {close_step}\n")
    print(
        "Bstep | nearest C | behind | dist14 | dist23 | "
        "local cos | chunk cos | signed progress"
    )
    print("-" * 100)

    for r in rows:
        print(
            f"{r['b_step']:5d} | "
            f"{r['nearest_c_step']:9d} | "
            f"{r['progress_error_steps']:6d} | "
            f"{r['distance_arm1_4_rad']:.3f} | "
            f"{r['distance_arm2_3_rad']:.3f} | "
            f"{r['local_direction_cosine']:9.3f} | "
            f"{r['chunk_direction_cosine']:9.3f} | "
            f"{r['signed_success_progress_rad']:+.4f}"
        )

    matched = np.asarray([r["nearest_c_step"] for r in rows])
    bsteps = np.asarray([r["b_step"] for r in rows])

    print("\nHow to read this:")
    print(
        "  nearest C = where the failed B state lies on the successful "
        "real-robot trajectory."
    )
    print(
        "  behind = B step - nearest C. A growing positive value means "
        "B is progressively falling behind the successful trajectory."
    )
    print(
        "  chunk cos near +1 = B action points forward along the successful path."
    )
    print(
        "  chunk cos near 0 = sideways/hold; below 0 = points backward."
    )
    print(
        "  signed progress <= 0 means the B command is not advancing "
        "along the successful path."
    )

    if len(matched) >= 3:
        recent = matched[-3:]
        if np.max(recent) - np.min(recent) <= 4:
            print(
                "\nNOTE: Final B states map to almost the same C region: "
                "consistent with a closed-loop stall/fixed point."
            )

        growth = (bsteps[-1] - matched[-1]) - (bsteps[0] - matched[0])
        if growth >= 16:
            print(
                "NOTE: B falls progressively behind successful trajectory progress."
            )


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    b_rows = load_b_log(args.b_log, args.inference_threshold_ms)
    c = read_c_bag(args.c_bag)

    print(f"B replanning rows: {len(b_rows)}")
    print(f"C command steps  : {len(c['action'])}")
    print(
        f"C median state age before command: "
        f"{np.median(c['state_age'])*1000.0:.1f} ms"
    )

    rows, close_step = analyze(
        b_rows,
        c,
        args.ahead,
        args.close_threshold,
    )

    csv_path = args.out / "b_vs_success_path.csv"
    save_csv(csv_path, rows)

    plot_progress(args.out / "b_progress_on_success_path.png", rows, close_step)
    plot_distance(args.out / "b_distance_to_success_path.png", rows)
    plot_direction(args.out / "b_action_direction.png", rows)
    plot_phase_arm23(args.out / "arm23_phase_b_vs_c.png", rows, c, close_step)

    print_summary(rows, close_step)

    print("\nOutput:")
    print(f"  {csv_path}")
    print(f"  {args.out / 'b_progress_on_success_path.png'}")
    print(f"  {args.out / 'b_distance_to_success_path.png'}")
    print(f"  {args.out / 'b_action_direction.png'}")
    print(f"  {args.out / 'arm23_phase_b_vs_c.png'}")


if __name__ == "__main__":
    main()
