#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import numpy as np


LINE_RE = re.compile(
    r"step=(?P<step>\d+)\s+"
    r"infer=\s*(?P<infer>[0-9.]+)\s*ms\s+"
    r"state_arm=\[(?P<state>[^\]]+)\]\s+"
    r"action_arm=\[(?P<action>[^\]]+)\]\s+"
    r"gripper=\[(?P<gripper>[^\]]+)\]"
)


def vec(s):
    return np.fromstring(s, sep=" ", dtype=np.float64)


def load_log(path):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        step = int(m.group("step"))
        rows[step] = {
            "step": step,
            "infer_ms": float(m.group("infer")),
            "state": vec(m.group("state")),
            "action": vec(m.group("action")),
            "gripper": vec(m.group("gripper")),
        }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isaac-log", required=True)
    ap.add_argument("--real-log", required=True)
    ap.add_argument("--out", default="policy_log_compare.csv")
    ap.add_argument(
        "--inference-threshold-ms",
        type=float,
        default=100.0,
        help="Only compare steps where at least one run performed a heavy policy inference.",
    )
    args = ap.parse_args()

    isaac = load_log(args.isaac_log)
    real = load_log(args.real_log)

    common = sorted(set(isaac) & set(real))
    heavy = [
        s for s in common
        if max(isaac[s]["infer_ms"], real[s]["infer_ms"])
        >= args.inference_threshold_ms
    ]

    if not heavy:
        raise RuntimeError("No common heavy-inference steps were found in the two logs.")

    header = [
        "step", "isaac_infer_ms", "real_infer_ms",
        "state_max_abs_diff", "action_max_abs_diff",
    ]
    for i in range(5):
        header += [
            f"isaac_state_arm{i+1}",
            f"real_state_arm{i+1}",
            f"state_diff_arm{i+1}",
        ]
    for i in range(5):
        header += [
            f"isaac_action_arm{i+1}",
            f"real_action_arm{i+1}",
            f"action_diff_arm{i+1}",
        ]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for s in heavy:
            a = isaac[s]
            r = real[s]
            sd = r["state"] - a["state"]
            ad = r["action"] - a["action"]

            row = [
                s, a["infer_ms"], r["infer_ms"],
                float(np.max(np.abs(sd))),
                float(np.max(np.abs(ad))),
            ]
            for i in range(5):
                row += [a["state"][i], r["state"][i], sd[i]]
            for i in range(5):
                row += [a["action"][i], r["action"][i], ad[i]]
            w.writerow(row)

    print("step | infer I/R(ms) | state max | action max | arm2 state I/R | arm3 state I/R | arm2 action I/R | arm3 action I/R")
    print("-" * 128)
    for s in heavy:
        a = isaac[s]
        r = real[s]
        sd = r["state"] - a["state"]
        ad = r["action"] - a["action"]
        print(
            f"{s:3d} | "
            f"{a['infer_ms']:7.1f}/{r['infer_ms']:7.1f} | "
            f"{np.max(np.abs(sd)):.3f} | "
            f"{np.max(np.abs(ad)):.3f} | "
            f"{a['state'][1]: .3f}/{r['state'][1]: .3f} | "
            f"{a['state'][2]: .3f}/{r['state'][2]: .3f} | "
            f"{a['action'][1]: .3f}/{r['action'][1]: .3f} | "
            f"{a['action'][2]: .3f}/{r['action'][2]: .3f}"
        )

    print()
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
