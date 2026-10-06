#!/usr/bin/env python3
"""Convert successful DOFBOT Isaac Sim episodes to a local LeRobot v3 dataset.

Default paths:
  Raw input : ~/dofbot/train_data_raw
  Output    : ~/dofbot/train_data_lerobot/dofbot_pick_state

The converted policy inputs are:
  observation.state              actual DOFBOT joint positions, 11 values
  observation.environment_state  known initial cube XY, 2 values

The training target is:
  action                         commanded joint targets, 11 values

Only episodes whose metadata.json contains ``success: true`` are converted.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_JOINT_NAMES = [
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

REQUIRED_ARRAY_FILES = (
    "timestamps.npy",
    "joint_position.npy",
    "cube_position.npy",
    "action.npy",
)


@dataclass(frozen=True)
class ValidatedEpisode:
    raw_dir: Path
    raw_episode_index: int
    joint_names: tuple[str, ...]
    initial_cube_xy: np.ndarray
    joint_position: np.ndarray
    action: np.ndarray
    timestamps: np.ndarray

    @property
    def steps(self) -> int:
        return int(self.joint_position.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert successful Isaac Sim DOFBOT episodes into a local "
            "LeRobot dataset."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("~/dofbot/train_data_raw"),
        help="Directory containing episode_XXXXXX raw directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("~/dofbot/train_data_lerobot/dofbot_pick_state"),
        help="Destination directory for the local LeRobot dataset.",
    )
    parser.add_argument(
        "--repo-id",
        default="local/dofbot_pick_state",
        help="Logical LeRobot repository ID stored with the local dataset.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="LeRobot dataset sampling rate. The raw recorder currently uses 10 Hz.",
    )
    parser.add_argument(
        "--task",
        default="Pick up the cube and lift it.",
        help="Natural-language task label saved for every frame.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output-root first when it already exists.",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=2,
        help="Skip successful episodes shorter than this many frames.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("metadata root must be a JSON object")
    return data


def raw_episode_index_from_dir(path: Path) -> int:
    try:
        return int(path.name.removeprefix("episode_"))
    except ValueError as exc:
        raise ValueError(f"invalid episode directory name: {path.name}") from exc


def get_initial_cube_xy(metadata: dict[str, Any]) -> np.ndarray:
    # Use the pose measured after the reset settle period when available.
    position = metadata.get(
        "settled_initial_cube_position",
        metadata.get("initial_cube_position"),
    )
    if not isinstance(position, list) or len(position) < 2:
        raise ValueError(
            "metadata has neither a valid settled_initial_cube_position nor "
            "initial_cube_position"
        )

    cube_xy = np.asarray(position[:2], dtype=np.float32)
    if cube_xy.shape != (2,) or not np.isfinite(cube_xy).all():
        raise ValueError(f"invalid initial cube XY: {position!r}")
    return cube_xy


def get_joint_names(metadata: dict[str, Any]) -> tuple[str, ...]:
    names = metadata.get("joint_names", DEFAULT_JOINT_NAMES)
    if not isinstance(names, list) or len(names) != 11:
        raise ValueError(f"joint_names must contain 11 names, got {names!r}")
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError("joint_names contains an invalid name")
    if len(set(names)) != len(names):
        raise ValueError("joint_names contains duplicates")
    return tuple(names)


def validate_success_episode(
    episode_dir: Path,
    *,
    expected_fps: int,
    min_steps: int,
) -> ValidatedEpisode:
    metadata_path = episode_dir / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError("metadata.json is missing")

    metadata = load_json(metadata_path)
    if metadata.get("status") != "finished":
        raise ValueError(f"status is not finished: {metadata.get('status')!r}")
    if metadata.get("success") is not True:
        raise ValueError("episode did not succeed")

    raw_fps = float(metadata.get("sample_rate_hz", expected_fps))
    if abs(raw_fps - expected_fps) > 1e-6:
        raise ValueError(
            f"sample rate mismatch: raw={raw_fps:g} Hz, requested={expected_fps} Hz"
        )

    missing = [name for name in REQUIRED_ARRAY_FILES if not (episode_dir / name).is_file()]
    if missing:
        raise ValueError(f"missing array files: {', '.join(missing)}")

    timestamps = np.load(episode_dir / "timestamps.npy", allow_pickle=False)
    joint_position = np.load(episode_dir / "joint_position.npy", allow_pickle=False)
    cube_position = np.load(episode_dir / "cube_position.npy", allow_pickle=False)
    action = np.load(episode_dir / "action.npy", allow_pickle=False)

    if timestamps.ndim != 1:
        raise ValueError(f"timestamps shape must be (N,), got {timestamps.shape}")
    steps = int(timestamps.shape[0])
    if steps < min_steps:
        raise ValueError(f"too few frames: {steps} < {min_steps}")
    if joint_position.shape != (steps, 11):
        raise ValueError(
            f"joint_position shape must be ({steps}, 11), got {joint_position.shape}"
        )
    if action.shape != (steps, 11):
        raise ValueError(f"action shape must be ({steps}, 11), got {action.shape}")
    if cube_position.shape != (steps, 3):
        raise ValueError(
            f"cube_position shape must be ({steps}, 3), got {cube_position.shape}"
        )

    for name, array in (
        ("timestamps", timestamps),
        ("joint_position", joint_position),
        ("cube_position", cube_position),
        ("action", action),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or infinity")

    recorded_steps = metadata.get("recorded_steps")
    if recorded_steps is not None and int(recorded_steps) != steps:
        raise ValueError(
            f"recorded_steps mismatch: metadata={recorded_steps}, arrays={steps}"
        )

    return ValidatedEpisode(
        raw_dir=episode_dir,
        raw_episode_index=int(
            metadata.get("episode_index", raw_episode_index_from_dir(episode_dir))
        ),
        joint_names=get_joint_names(metadata),
        initial_cube_xy=get_initial_cube_xy(metadata),
        joint_position=np.asarray(joint_position, dtype=np.float32),
        action=np.asarray(action, dtype=np.float32),
        timestamps=np.asarray(timestamps, dtype=np.float64),
    )


def discover_episodes(
    raw_root: Path,
    *,
    expected_fps: int,
    min_steps: int,
) -> tuple[list[ValidatedEpisode], list[dict[str, Any]]]:
    valid: list[ValidatedEpisode] = []
    skipped: list[dict[str, Any]] = []

    for episode_dir in sorted(raw_root.glob("episode_*")):
        if not episode_dir.is_dir():
            continue
        try:
            episode = validate_success_episode(
                episode_dir,
                expected_fps=expected_fps,
                min_steps=min_steps,
            )
        except Exception as exc:
            skipped.append(
                {
                    "raw_directory": str(episode_dir),
                    "reason": str(exc),
                }
            )
            continue
        valid.append(episode)

    return valid, skipped


def create_features(joint_names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (11,),
            "names": list(joint_names),
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["cube_initial_x_m", "cube_initial_y_m"],
        },
        "action": {
            "dtype": "float32",
            "shape": (11,),
            "names": [f"{name}_target" for name in joint_names],
        },
    }


def convert(args: argparse.Namespace) -> None:
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.min_steps <= 0:
        raise ValueError("--min-steps must be positive")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw data directory not found: {raw_root}")

    valid_episodes, skipped = discover_episodes(
        raw_root,
        expected_fps=args.fps,
        min_steps=args.min_steps,
    )
    if not valid_episodes:
        raise RuntimeError(
            f"No valid successful episodes were found under {raw_root}. "
            f"Skipped {len(skipped)} directories."
        )

    reference_joint_names = valid_episodes[0].joint_names
    consistent_episodes: list[ValidatedEpisode] = []
    for episode in valid_episodes:
        if episode.joint_names != reference_joint_names:
            skipped.append(
                {
                    "raw_directory": str(episode.raw_dir),
                    "reason": "joint_names differ from the first valid episode",
                }
            )
            continue
        consistent_episodes.append(episode)

    if not consistent_episodes:
        raise RuntimeError("No episodes remain after checking joint-name consistency")

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"output directory already exists: {output_root}\n"
                "Re-run with --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    output_root.parent.mkdir(parents=True, exist_ok=True)

    print(f"Raw input       : {raw_root}")
    print(f"LeRobot output  : {output_root}")
    print(f"Repo ID         : {args.repo_id}")
    print(f"FPS             : {args.fps}")
    print(f"Successful input: {len(consistent_episodes)}")
    print(f"Skipped input   : {len(skipped)}")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_root,
        fps=args.fps,
        robot_type="dofbot_sim",
        features=create_features(reference_joint_names),
        use_videos=False,
    )

    converted: list[dict[str, Any]] = []
    total_frames = 0

    try:
        for lerobot_episode_index, episode in enumerate(consistent_episodes):
            try:
                for frame_index in range(episode.steps):
                    dataset.add_frame(
                        {
                            "observation.state": episode.joint_position[frame_index],
                            "observation.environment_state": episode.initial_cube_xy.copy(),
                            "action": episode.action[frame_index],
                            "task": args.task,
                        }
                    )
                dataset.save_episode()
            except Exception:
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer(delete_images=False)
                raise

            converted.append(
                {
                    "lerobot_episode_index": lerobot_episode_index,
                    "raw_episode_index": episode.raw_episode_index,
                    "raw_directory": str(episode.raw_dir),
                    "frames": episode.steps,
                    "initial_cube_xy_m": episode.initial_cube_xy.astype(float).tolist(),
                    "raw_start_timestamp": float(episode.timestamps[0]),
                    "raw_end_timestamp": float(episode.timestamps[-1]),
                }
            )
            total_frames += episode.steps
            print(
                f"Converted {lerobot_episode_index + 1:4d}/{len(consistent_episodes)}: "
                f"raw episode {episode.raw_episode_index}, "
                f"frames={episode.steps}, "
                f"cube_xy=({episode.initial_cube_xy[0]:.4f}, "
                f"{episode.initial_cube_xy[1]:.4f})"
            )
    finally:
        # LeRobot v0.5.1 Dataset v3 requires finalization to close parquet writers.
        dataset.finalize()

    report = {
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "repo_id": args.repo_id,
        "fps": args.fps,
        "task": args.task,
        "observation": {
            "observation.state": "actual joint positions, 11 dimensions",
            "observation.environment_state": (
                "known initial cube x/y, repeated for every frame in the episode"
            ),
        },
        "action": "commanded joint targets, 11 dimensions",
        "converted_episode_count": len(converted),
        "converted_frame_count": total_frames,
        "skipped_episode_count": len(skipped),
        "converted_episodes": converted,
        "skipped_episodes": skipped,
    }
    report_path = output_root / "conversion_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Reopen the finalized local dataset as a final integrity check.
    check_dataset = LeRobotDataset(repo_id=args.repo_id, root=output_root)
    first = check_dataset[0]

    print("\nConversion complete")
    print(f"Episodes : {check_dataset.num_episodes}")
    print(f"Frames   : {check_dataset.num_frames}")
    print(f"Features : {list(check_dataset.features)}")
    print(f"First observation.state shape             : {tuple(first['observation.state'].shape)}")
    print(
        "First observation.environment_state shape : "
        f"{tuple(first['observation.environment_state'].shape)}"
    )
    print(f"First action shape                        : {tuple(first['action'].shape)}")
    print(f"Report   : {report_path}")


def main() -> None:
    args = parse_args()
    try:
        convert(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
