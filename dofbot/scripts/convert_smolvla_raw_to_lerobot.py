#!/usr/bin/env python3
"""Convert DOFBOT SmolVLA raw episodes into a local LeRobotDataset v3 dataset.

Expected raw episode layout (created by dofbot_smolvla_pick_env.py):

    episode_000000/
      metadata.json
      task_instruction.txt
      timestamps.npy
      joint_position.npy      # [T, 11] actual joint positions
      action.npy              # [T, 11] commanded joint positions
      camera_rgb.mp4          # T RGB frames, synchronized by index
      camera_rgb_timestamps.npy
      camera_rgb_frame_valid.npy
      ...

By default, only episodes whose metadata contains ``success: true`` are
converted. Failed episodes remain in the raw directory for diagnosis but are
not treated as demonstrations for behavior-cloning / SmolVLA training.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    # Some LeRobot installations expose the class from the package root.
    from lerobot.datasets import LeRobotDataset


DEFAULT_RAW_ROOT = Path.home() / "dofbot" / "smolvla_raw"
DEFAULT_OUTPUT_ROOT = Path.home() / "dofbot" / "smolvla_lerobot"
DEFAULT_REPO_ID = "local/dofbot_smolvla_pick"
DEFAULT_VIDEO_KEY = "observation.images.front"
VIDEO_FILENAME = "camera_rgb.mp4"


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frame_count: int
    average_fps: float


@dataclass(frozen=True)
class EpisodeSpec:
    source_dir: Path
    source_episode_index: int
    task: str
    joint_names: tuple[str, ...]
    frame_count: int
    fps: int
    video: VideoInfo
    invalid_video_frames: int
    success: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert synchronized DOFBOT SmolVLA raw episodes to a local "
            "LeRobotDataset v3 dataset."
        )
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=(
            "Logical LeRobot dataset id stored in metadata. This need not exist "
            "on the Hugging Face Hub when --output-root is used."
        ),
    )
    parser.add_argument(
        "--video-key",
        default=DEFAULT_VIDEO_KEY,
        help="Dataset feature name for the fixed front RGB camera.",
    )
    parser.add_argument(
        "--include-failures",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also convert failed trajectories. Disabled by default because failed "
            "actions should normally not be behavior-cloning targets."
        ),
    )
    parser.add_argument(
        "--require-all-video-frames-valid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reject an episode when Isaac Sim marked any RGB frame invalid. "
            "Normally invalid captures were replaced with the preceding frame."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete an existing output directory before conversion.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="LeRobot temporary PNG writer threads used before MP4 encoding.",
    )
    parser.add_argument(
        "--vcodec",
        default="h264",
        help="LeRobot output video codec. h264 keeps the resulting dataset as MP4/H.264.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of accepted episodes, useful for a smoke test.",
    )
    parser.add_argument(
        "--verify-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reload the completed local dataset and verify episode/frame counts.",
    )
    return parser.parse_args()


def require_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required executable was not found in PATH: {name}")
    return path


def parse_ratio(value: str | None) -> float:
    if not value or value == "0/0" or value == "N/A":
        return math.nan
    if "/" in value:
        numerator, denominator = value.split("/", maxsplit=1)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else math.nan
    return float(value)


def probe_video(path: Path, ffprobe: str) -> VideoInfo:
    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {path}:\n{completed.stderr.strip()}"
        )

    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected one RGB video stream in {path}, found {len(streams)}")

    stream = streams[0]
    frame_count_text = stream.get("nb_read_frames")
    if frame_count_text in (None, "N/A"):
        raise RuntimeError(f"ffprobe could not count frames in {path}")

    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        frame_count=int(frame_count_text),
        average_fps=parse_ratio(stream.get("avg_frame_rate")),
    )


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def load_task(episode_dir: Path, metadata: dict) -> str:
    task_path = episode_dir / "task_instruction.txt"
    if task_path.is_file():
        task = task_path.read_text(encoding="utf-8").strip()
    else:
        task = str(metadata.get("task_instruction", "")).strip()
    if not task:
        raise ValueError(f"Task instruction is empty: {episode_dir}")
    return task


def first_dimension(path: Path) -> int:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim == 0:
        raise ValueError(f"Expected an array with a time dimension: {path}")
    return int(array.shape[0])


def validate_vector_array(path: Path, frame_count: int, width: int) -> None:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    expected = (frame_count, width)
    if array.shape != expected:
        raise ValueError(f"{path}: expected shape {expected}, found {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{path}: expected floating-point values, found {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{path}: contains NaN or infinite values")


def preflight_episode(
    episode_dir: Path,
    *,
    ffprobe: str,
    include_failures: bool,
    require_all_video_frames_valid: bool,
) -> tuple[EpisodeSpec | None, str | None]:
    metadata_path = episode_dir / "metadata.json"
    if not metadata_path.is_file():
        return None, "metadata.json missing"

    metadata = load_json(metadata_path)
    if metadata.get("status") != "finished":
        return None, f"status={metadata.get('status')!r}, not finished"

    success = bool(metadata.get("success", False))
    if not success and not include_failures:
        return None, "failed episode (kept in raw data, excluded from training dataset)"

    task = load_task(episode_dir, metadata)
    joint_names_raw = metadata.get("joint_names")
    if not isinstance(joint_names_raw, list) or not joint_names_raw:
        raise ValueError(f"joint_names is missing or invalid in {metadata_path}")
    joint_names = tuple(str(name) for name in joint_names_raw)

    timestamps_path = episode_dir / "timestamps.npy"
    joint_path = episode_dir / "joint_position.npy"
    action_path = episode_dir / "action.npy"
    video_path = episode_dir / VIDEO_FILENAME
    for required in (timestamps_path, joint_path, action_path, video_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required file is missing: {required}")

    timestamps = np.load(timestamps_path, mmap_mode="r", allow_pickle=False)
    if timestamps.ndim != 1 or timestamps.size == 0:
        raise ValueError(f"{timestamps_path}: expected non-empty shape [T], found {timestamps.shape}")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError(f"{timestamps_path}: contains NaN or infinite values")
    if timestamps.size > 1 and np.any(np.diff(timestamps) < 0.0):
        raise ValueError(f"{timestamps_path}: timestamps are not monotonic")
    frame_count = int(timestamps.shape[0])

    recorded_steps = metadata.get("recorded_steps")
    if recorded_steps is not None and int(recorded_steps) != frame_count:
        raise ValueError(
            f"{episode_dir}: metadata recorded_steps={recorded_steps}, "
            f"but timestamps contain {frame_count} frames"
        )

    validate_vector_array(joint_path, frame_count, len(joint_names))
    validate_vector_array(action_path, frame_count, len(joint_names))

    # Validate other synchronized arrays when present, even though they are not
    # included as SmolVLA inputs.
    for optional_name in (
        "cube_position.npy",
        "gripper_position.npy",
        "phase.npy",
        "camera_rgb_timestamps.npy",
        "camera_rgb_frame_valid.npy",
        "camera_rgb_sample_indices.npy",
    ):
        optional_path = episode_dir / optional_name
        if optional_path.is_file() and first_dimension(optional_path) != frame_count:
            raise ValueError(
                f"{optional_path}: first dimension does not match timestamps ({frame_count})"
            )

    camera_timestamps_path = episode_dir / "camera_rgb_timestamps.npy"
    if camera_timestamps_path.is_file():
        camera_timestamps = np.load(camera_timestamps_path, mmap_mode="r", allow_pickle=False)
        if not np.allclose(camera_timestamps, timestamps, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"{camera_timestamps_path}: camera and state timestamps are not identical"
            )

    sample_indices_path = episode_dir / "camera_rgb_sample_indices.npy"
    if sample_indices_path.is_file():
        sample_indices = np.load(sample_indices_path, mmap_mode="r", allow_pickle=False)
        expected_indices = np.arange(frame_count, dtype=sample_indices.dtype)
        if not np.array_equal(sample_indices, expected_indices):
            raise ValueError(f"{sample_indices_path}: unexpected frame-to-sample mapping")

    invalid_video_frames = 0
    valid_path = episode_dir / "camera_rgb_frame_valid.npy"
    if valid_path.is_file():
        validity = np.load(valid_path, mmap_mode="r", allow_pickle=False)
        invalid_video_frames = int(validity.size - np.count_nonzero(validity))
        if require_all_video_frames_valid and invalid_video_frames:
            raise ValueError(
                f"{valid_path}: contains {invalid_video_frames} invalid/repeated RGB frames"
            )

    sample_rate = float(metadata.get("sample_rate_hz", 0.0))
    fps = int(round(sample_rate))
    if fps <= 0 or not math.isclose(sample_rate, fps, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"{metadata_path}: sample_rate_hz must be a positive integer, found {sample_rate}"
        )

    video = probe_video(video_path, ffprobe)
    if video.frame_count != frame_count:
        raise ValueError(
            f"{video_path}: video has {video.frame_count} frames, synchronized arrays have {frame_count}"
        )
    if math.isfinite(video.average_fps) and not math.isclose(
        video.average_fps, fps, rel_tol=0.0, abs_tol=0.01
    ):
        raise ValueError(
            f"{video_path}: average FPS {video.average_fps:.6f} does not match dataset FPS {fps}"
        )

    return (
        EpisodeSpec(
            source_dir=episode_dir,
            source_episode_index=int(metadata.get("episode_index", -1)),
            task=task,
            joint_names=joint_names,
            frame_count=frame_count,
            fps=fps,
            video=video,
            invalid_video_frames=invalid_video_frames,
            success=success,
        ),
        None,
    )


def read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode_rgb_frames(path: Path, *, width: int, height: int, ffmpeg: str) -> Iterator[np.ndarray]:
    frame_bytes = width * height * 3
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    completed_normally = False
    try:
        while True:
            payload = read_exact(process.stdout, frame_bytes)
            if not payload:
                break
            if len(payload) != frame_bytes:
                raise RuntimeError(
                    f"Truncated RGB frame while decoding {path}: "
                    f"expected {frame_bytes} bytes, received {len(payload)}"
                )
            yield np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy()
        completed_normally = True
    finally:
        if not completed_normally and process.poll() is None:
            process.terminate()
        with contextlib.suppress(Exception):
            process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if completed_normally and return_code != 0:
            raise RuntimeError(f"ffmpeg failed while decoding {path}:\n{stderr.strip()}")


def scan_episodes(args: argparse.Namespace, ffprobe: str) -> tuple[list[EpisodeSpec], list[dict]]:
    raw_root = args.raw_root.expanduser().resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_root}")

    episode_dirs = sorted(path for path in raw_root.glob("episode_*") if path.is_dir())
    if not episode_dirs:
        raise RuntimeError(f"No episode_* directories were found in {raw_root}")

    accepted: list[EpisodeSpec] = []
    skipped: list[dict] = []
    for episode_dir in episode_dirs:
        spec, skip_reason = preflight_episode(
            episode_dir,
            ffprobe=ffprobe,
            include_failures=args.include_failures,
            require_all_video_frames_valid=args.require_all_video_frames_valid,
        )
        if spec is None:
            skipped.append({"source": str(episode_dir), "reason": str(skip_reason)})
            continue
        accepted.append(spec)
        if args.limit is not None and len(accepted) >= args.limit:
            break

    if not accepted:
        raise RuntimeError(
            "No episodes were accepted. Check metadata success flags, videos, and synchronized arrays."
        )

    reference = accepted[0]
    for spec in accepted[1:]:
        if spec.joint_names != reference.joint_names:
            raise ValueError(
                f"Joint names differ between {reference.source_dir} and {spec.source_dir}"
            )
        if spec.fps != reference.fps:
            raise ValueError(
                f"FPS differs between {reference.source_dir} ({reference.fps}) "
                f"and {spec.source_dir} ({spec.fps})"
            )
        if (spec.video.width, spec.video.height) != (
            reference.video.width,
            reference.video.height,
        ):
            raise ValueError(
                f"Video resolution differs between {reference.source_dir} and {spec.source_dir}"
            )

    return accepted, skipped


def prepare_output(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_root}\n"
                "Use --overwrite only when you intentionally want to replace it."
            )
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)


def make_features(spec: EpisodeSpec, video_key: str) -> dict[str, dict]:
    names = list(spec.joint_names)
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(names),),
            "names": names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(names),),
            "names": names,
        },
        video_key: {
            "dtype": "video",
            "shape": (spec.video.height, spec.video.width, 3),
            "names": ["height", "width", "channels"],
        },
    }


def convert(args: argparse.Namespace) -> None:
    ffmpeg = require_executable("ffmpeg")
    ffprobe = require_executable("ffprobe")

    if args.image_writer_threads < 0:
        raise ValueError("--image-writer-threads must be zero or positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if not args.video_key.startswith("observation.images."):
        raise ValueError("--video-key should normally start with 'observation.images.'")
    if args.include_failures:
        print(
            "WARNING: --include-failures is enabled. Failed trajectories will be "
            "treated as demonstrations by standard behavior-cloning training.",
            file=sys.stderr,
        )

    accepted, skipped = scan_episodes(args, ffprobe)
    reference = accepted[0]
    output_root = args.output_root.expanduser().resolve()
    prepare_output(output_root, args.overwrite)

    print("\n========== Conversion plan ==========")
    print(f"raw root       : {args.raw_root.expanduser().resolve()}")
    print(f"output root    : {output_root}")
    print(f"repo id        : {args.repo_id}")
    print(f"accepted       : {len(accepted)} episodes")
    print(f"skipped        : {len(skipped)} episodes")
    print(f"fps            : {reference.fps}")
    print(f"video          : {reference.video.width}x{reference.video.height} H.264/MP4")
    print(f"state/action   : {len(reference.joint_names)} joints")
    print(f"video key      : {args.video_key}")

    features = make_features(reference, args.video_key)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_root,
        fps=reference.fps,
        robot_type="dofbot",
        features=features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=args.image_writer_threads,
        vcodec=args.vcodec,
    )

    converted: list[dict] = []
    total_frames = 0
    try:
        for output_episode_index, spec in enumerate(accepted):
            print(
                f"[{output_episode_index + 1:03d}/{len(accepted):03d}] "
                f"{spec.source_dir.name}: {spec.task!r}, {spec.frame_count} frames"
            )
            states = np.load(
                spec.source_dir / "joint_position.npy", mmap_mode="r", allow_pickle=False
            )
            actions = np.load(
                spec.source_dir / "action.npy", mmap_mode="r", allow_pickle=False
            )

            decoded_count = 0
            for decoded_count, rgb_frame in enumerate(
                decode_rgb_frames(
                    spec.source_dir / VIDEO_FILENAME,
                    width=spec.video.width,
                    height=spec.video.height,
                    ffmpeg=ffmpeg,
                ),
                start=1,
            ):
                frame_index = decoded_count - 1
                if frame_index >= spec.frame_count:
                    raise RuntimeError(
                        f"Decoded more video frames than expected in {spec.source_dir}"
                    )
                dataset.add_frame(
                    {
                        args.video_key: rgb_frame,
                        "observation.state": np.asarray(states[frame_index], dtype=np.float32),
                        "action": np.asarray(actions[frame_index], dtype=np.float32),
                        "task": spec.task,
                    }
                )

            if decoded_count != spec.frame_count:
                raise RuntimeError(
                    f"Decoded {decoded_count} frames from {spec.source_dir / VIDEO_FILENAME}, "
                    f"expected {spec.frame_count}"
                )

            dataset.save_episode(parallel_encoding=False)
            total_frames += spec.frame_count
            converted.append(
                {
                    "lerobot_episode_index": output_episode_index,
                    "source_episode_index": spec.source_episode_index,
                    "source_directory": str(spec.source_dir),
                    "task": spec.task,
                    "success": spec.success,
                    "frame_count": spec.frame_count,
                    "invalid_or_repeated_rgb_frames": spec.invalid_video_frames,
                }
            )

        dataset.finalize()
    except Exception:
        with contextlib.suppress(Exception):
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer(delete_images=True)
        with contextlib.suppress(Exception):
            dataset.finalize()
        raise

    report = {
        "format": "LeRobotDataset v3",
        "repo_id": args.repo_id,
        "raw_root": str(args.raw_root.expanduser().resolve()),
        "output_root": str(output_root),
        "success_only": not args.include_failures,
        "fps": reference.fps,
        "video_key": args.video_key,
        "video_resolution_hwc": [reference.video.height, reference.video.width, 3],
        "joint_names": list(reference.joint_names),
        "converted_episode_count": len(converted),
        "converted_frame_count": total_frames,
        "converted_episodes": converted,
        "skipped_episodes": skipped,
    }
    (output_root / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.verify_load:
        loaded = LeRobotDataset(repo_id=args.repo_id, root=output_root)
        if loaded.num_episodes != len(converted):
            raise RuntimeError(
                f"Verification failed: dataset reports {loaded.num_episodes} episodes, "
                f"expected {len(converted)}"
            )
        if loaded.num_frames != total_frames:
            raise RuntimeError(
                f"Verification failed: dataset reports {loaded.num_frames} frames, "
                f"expected {total_frames}"
            )

    print("\n========== Conversion complete ==========")
    print(f"dataset        : {output_root}")
    print(f"episodes       : {len(converted)}")
    print(f"frames         : {total_frames}")
    print(f"skipped        : {len(skipped)}")
    print(f"report         : {output_root / 'conversion_report.json'}")
    print("features:")
    for key, value in features.items():
        print(f"  {key}: dtype={value['dtype']} shape={value['shape']}")


def main() -> None:
    args = parse_args()
    try:
        convert(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
