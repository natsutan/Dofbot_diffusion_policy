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
    "arm1_Joint", "arm2_Joint", "arm3_Joint", "arm4_Joint", "arm5_Joint",
    "Llink1_Joint", "Llink2_Joint", "Llink3_Joint",
    "Rlink1_Joint", "Rlink2_Joint", "Rlink3_Joint",
]
STATE_TOPIC = "/joint_states"
COMMAND_TOPIC = "/joint_command"


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Estimate time lag between Isaac Sim and real DOFBOT joint-state "
            "trajectories recorded with the same /joint_command sequence."
        )
    )
    p.add_argument("--isaac", required=True, type=Path)
    p.add_argument("--real", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--start-step", type=int, default=1,
                   help="First command step used for analysis. Default: 1")
    p.add_argument("--end-step", type=int, default=None,
                   help="Last command step used for analysis. Default: last")
    p.add_argument("--min-lag-ms", type=float, default=-200.0)
    p.add_argument("--max-lag-ms", type=float, default=500.0)
    p.add_argument("--lag-step-ms", type=float, default=5.0)
    p.add_argument("--sample-period-ms", type=float, default=20.0)
    return p.parse_args()


def resolve_storage(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        mcap = sorted(path.glob("*.mcap"))
        if len(mcap) == 1:
            return mcap[0], "mcap"
        db3 = sorted(path.glob("*.db3"))
        if len(db3) == 1:
            return db3[0], "sqlite3"
        raise RuntimeError(f"Expected exactly one .mcap or .db3 file in {path}")
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


def read_bag(path: Path):
    storage_path, storage_id = resolve_storage(path)
    print(f"Reading: {path}")
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
            raise RuntimeError(f"{topic} not found in bag: {path}")
    reader.set_filter(rosbag2_py.StorageFilter(topics=[STATE_TOPIC, COMMAND_TOPIC]))

    ts = {STATE_TOPIC: [], COMMAND_TOPIC: []}
    qs = {STATE_TOPIC: [], COMMAND_TOPIC: []}
    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        msg = deserialize_message(data, get_message(topic_types[topic]))
        q = joint_vector(msg)
        if q is None:
            continue
        ts[topic].append(int(timestamp_ns))
        qs[topic].append(q)

    state_t = np.asarray(ts[STATE_TOPIC], dtype=np.int64)
    state_q = np.asarray(qs[STATE_TOPIC], dtype=np.float64)
    cmd_t = np.asarray(ts[COMMAND_TOPIC], dtype=np.int64)
    cmd_q = np.asarray(qs[COMMAND_TOPIC], dtype=np.float64)
    if len(state_t) < 2 or len(cmd_t) < 2:
        raise RuntimeError(f"Not enough usable data in {path}")

    t0 = cmd_t[0]
    return {
        "state_t": (state_t - t0) / 1e9,
        "state_q": state_q,
        "cmd_t": (cmd_t - t0) / 1e9,
        "cmd_q": cmd_q,
    }


def check_commands(isaac, real):
    n = min(len(isaac["cmd_q"]), len(real["cmd_q"]))
    qdiff = real["cmd_q"][:n] - isaac["cmd_q"][:n]
    tdiff = real["cmd_t"][:n] - isaac["cmd_t"][:n]
    print("\nCommand comparison:")
    print(f"  Isaac commands : {len(isaac['cmd_q'])}")
    print(f"  Real commands  : {len(real['cmd_q'])}")
    print(f"  max |q diff|   : {np.max(np.abs(qdiff)):.9f} rad")
    print(f"  median timing difference : {np.median(tdiff)*1000.0:.1f} ms")
    print(f"  max timing difference    : {np.max(np.abs(tdiff))*1000.0:.1f} ms")


def get_window(isaac, start_step, end_step):
    n = len(isaac["cmd_t"])
    if not 0 <= start_step < n:
        raise ValueError(f"Invalid --start-step {start_step}; command count={n}")
    if end_step is None:
        end_step = n - 1
    if not start_step <= end_step < n:
        raise ValueError(f"Invalid --end-step {end_step}")
    return float(isaac["cmd_t"][start_step]), float(isaac["cmd_t"][end_step]), end_step


def compare_at_lag(isaac, real, joint_index, start_t, end_t, lag_s, dt_s):
    # Positive lag means: Real(t + lag) is compared with Isaac(t),
    # i.e. real reaches the corresponding state later.
    lo = max(start_t, isaac["state_t"][0], real["state_t"][0] - lag_s)
    hi = min(end_t, isaac["state_t"][-1], real["state_t"][-1] - lag_s)
    if hi <= lo:
        return None
    grid = np.arange(lo, hi, dt_s)
    if len(grid) < 5:
        return None

    iq = np.interp(grid, isaac["state_t"], isaac["state_q"][:, joint_index])
    rq = np.interp(grid + lag_s, real["state_t"], real["state_q"][:, joint_index])
    diff = rq - iq
    offset = float(np.mean(diff))
    centered = diff - offset
    rmse = float(np.sqrt(np.mean(diff * diff)))
    centered_rmse = float(np.sqrt(np.mean(centered * centered)))
    corr = float("nan")
    if np.std(iq) > 1e-12 and np.std(rq) > 1e-12:
        corr = float(np.corrcoef(iq, rq)[0, 1])
    return {
        "lag_ms": lag_s * 1000.0,
        "rmse_rad": rmse,
        "static_offset_rad": offset,
        "centered_rmse_rad": centered_rmse,
        "corr": corr,
        "samples": len(grid),
    }


def scan_joint(isaac, real, joint_index, start_t, end_t, lag_values_s, dt_s):
    rows = []
    for lag_s in lag_values_s:
        m = compare_at_lag(isaac, real, joint_index, start_t, end_t, lag_s, dt_s)
        if m is not None:
            rows.append(m)
    if not rows:
        raise RuntimeError(f"No valid lag samples for {JOINT_NAMES[joint_index]}")
    best = min(rows, key=lambda r: r["centered_rmse_rad"])
    zero = min(rows, key=lambda r: abs(r["lag_ms"]))
    return rows, best, zero


def combined_arm23(isaac, real, start_t, end_t, lag_values_s, dt_s):
    rows = []
    for lag_s in lag_values_s:
        centered_parts = []
        raw_parts = []
        for joint_index in (1, 2):
            lo = max(start_t, isaac["state_t"][0], real["state_t"][0] - lag_s)
            hi = min(end_t, isaac["state_t"][-1], real["state_t"][-1] - lag_s)
            if hi <= lo:
                centered_parts = []
                break
            grid = np.arange(lo, hi, dt_s)
            if len(grid) < 5:
                centered_parts = []
                break
            iq = np.interp(grid, isaac["state_t"], isaac["state_q"][:, joint_index])
            rq = np.interp(grid + lag_s, real["state_t"], real["state_q"][:, joint_index])
            diff = rq - iq
            raw_parts.append(diff)
            centered_parts.append(diff - np.mean(diff))
        if not centered_parts:
            continue
        raw = np.concatenate(raw_parts)
        centered = np.concatenate(centered_parts)
        rows.append({
            "lag_ms": lag_s * 1000.0,
            "rmse_rad": float(np.sqrt(np.mean(raw * raw))),
            "centered_rmse_rad": float(np.sqrt(np.mean(centered * centered))),
        })
    best = min(rows, key=lambda r: r["centered_rmse_rad"])
    zero = min(rows, key=lambda r: abs(r["lag_ms"]))
    return rows, best, zero


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_scan(path, rows, title):
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    ax.plot([r["lag_ms"] for r in rows], [r["centered_rmse_rad"] for r in rows])
    ax.set_xlabel("Assumed real-robot lag [ms]")
    ax.set_ylabel("Offset-removed RMSE [rad]")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_aligned(path, isaac, real, joint_index, start_t, end_t, best_lag_ms, dt_s):
    lag_s = best_lag_ms / 1000.0
    lo = max(start_t, isaac["state_t"][0], real["state_t"][0] - lag_s)
    hi = min(end_t, isaac["state_t"][-1], real["state_t"][-1] - lag_s)
    grid = np.arange(lo, hi, dt_s)
    iq = np.interp(grid, isaac["state_t"], isaac["state_q"][:, joint_index])
    rq = np.interp(grid + lag_s, real["state_t"], real["state_q"][:, joint_index])
    offset = float(np.mean(rq - iq))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(grid, iq, label="Isaac")
    ax.plot(grid, rq - offset, label=f"Real shifted {best_lag_ms:.0f} ms, offset removed")
    ax.set_xlabel("Time from first command [s]")
    ax.set_ylabel("Joint position [rad]")
    ax.set_title(f"{JOINT_NAMES[joint_index]}: lag-aligned trajectory")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    isaac = read_bag(args.isaac)
    real = read_bag(args.real)
    check_commands(isaac, real)

    start_t, end_t, end_step = get_window(isaac, args.start_step, args.end_step)
    lag_ms = np.arange(args.min_lag_ms, args.max_lag_ms + 0.25 * args.lag_step_ms,
                       args.lag_step_ms)
    lag_values_s = lag_ms / 1000.0
    dt_s = args.sample_period_ms / 1000.0

    print("\nAnalysis window:")
    print(f"  steps: {args.start_step} .. {end_step}")
    print(f"  time : {start_t:.3f} .. {end_t:.3f} s")

    summary = []
    for joint_index in (1, 2):
        rows, best, zero = scan_joint(
            isaac, real, joint_index, start_t, end_t, lag_values_s, dt_s
        )
        improve = 100.0 * (zero["centered_rmse_rad"] - best["centered_rmse_rad"]) / max(
            zero["centered_rmse_rad"], 1e-12
        )
        name = JOINT_NAMES[joint_index]
        summary.append({
            "joint": name,
            "best_lag_ms": best["lag_ms"],
            "approx_100ms_steps": best["lag_ms"] / 100.0,
            "zero_lag_centered_rmse_rad": zero["centered_rmse_rad"],
            "best_centered_rmse_rad": best["centered_rmse_rad"],
            "improvement_percent": improve,
            "static_offset_rad": best["static_offset_rad"],
            "corr": best["corr"],
        })
        save_csv(args.out / f"{name}_lag_scan.csv", rows)
        plot_scan(args.out / f"{name}_lag_scan.png", rows, f"{name}: lag scan")
        plot_aligned(args.out / f"{name}_aligned.png", isaac, real, joint_index,
                     start_t, end_t, best["lag_ms"], dt_s)

    combined_rows, combined_best, combined_zero = combined_arm23(
        isaac, real, start_t, end_t, lag_values_s, dt_s
    )
    combined_improve = 100.0 * (
        combined_zero["centered_rmse_rad"] - combined_best["centered_rmse_rad"]
    ) / max(combined_zero["centered_rmse_rad"], 1e-12)
    save_csv(args.out / "arm23_combined_lag_scan.csv", combined_rows)
    plot_scan(args.out / "arm23_combined_lag_scan.png", combined_rows,
              "arm2 + arm3: combined lag scan")

    summary_path = args.out / "lag_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print("\n========== Lag analysis ==========")
    print("Positive lag means the real robot reaches the corresponding Isaac state later.\n")
    for s in summary:
        print(
            f"{s['joint']:12s}: best lag={s['best_lag_ms']:7.1f} ms "
            f"(~{s['approx_100ms_steps']:.2f} x 100-ms steps), "
            f"RMSE {s['zero_lag_centered_rmse_rad']:.4f} -> "
            f"{s['best_centered_rmse_rad']:.4f} rad "
            f"({s['improvement_percent']:.1f}% better), "
            f"offset={s['static_offset_rad']:+.4f} rad, corr={s['corr']:.4f}"
        )
    print(
        f"\narm2+arm3 combined: best lag={combined_best['lag_ms']:.1f} ms, "
        f"RMSE {combined_zero['centered_rmse_rad']:.4f} -> "
        f"{combined_best['centered_rmse_rad']:.4f} rad "
        f"({combined_improve:.1f}% better)"
    )
    print("\nInterpretation:")
    print("  - Best lag clearly > 0 and RMSE improves strongly: mainly dynamic delay.")
    print("  - Best lag near 0 but offset large: mainly calibration/position offset.")
    print("  - arm2 and arm3 have very different lags: joint-specific dynamics.")
    print("  - Lag correction barely helps: not explainable by simple time delay.")
    print("\nOutput:")
    for p in [
        summary_path,
        args.out / "arm2_Joint_lag_scan.png",
        args.out / "arm3_Joint_lag_scan.png",
        args.out / "arm23_combined_lag_scan.png",
        args.out / "arm2_Joint_aligned.png",
        args.out / "arm3_Joint_aligned.png",
    ]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
