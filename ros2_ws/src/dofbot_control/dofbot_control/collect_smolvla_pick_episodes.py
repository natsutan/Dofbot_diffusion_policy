#!/usr/bin/env python3
"""Collect color-conditioned DOFBOT SmolVLA pick-and-lift episodes.

For every episode this ROS 2 node chooses red, blue, or green, publishes a
language instruction such as ``pick up a red box.``, resets the Isaac Sim
SmolVLA environment, reads the matching cube pose from /dofbot/scene_layout,
solves IK, and executes the verified pick-and-lift teacher trajectory. Isaac Sim
owns synchronized recording and labels both successful and failed episodes.

Expected ROS 2 packages:
- dofbot_description
- dofbot_control
- dofbot_interfaces
- std_msgs
"""

from __future__ import annotations

import json
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from dofbot_interfaces.msg import EpisodeState
from dofbot_interfaces.srv import EndEpisode, ResetEpisode


# =============================================================================
# Robot / ROS settings
# =============================================================================

DESCRIPTION_PACKAGE = "dofbot_description"
URDF_RELATIVE_PATH = Path("urdf") / "dofbot.urdf"

ARM_JOINT_NAMES = (
    "arm1_Joint",
    "arm2_Joint",
    "arm3_Joint",
    "arm4_Joint",
    "arm5_Joint",
)

GRIPPER_JOINT_NAMES = (
    "Llink1_Joint",
    "Llink2_Joint",
    "Llink3_Joint",
    "Rlink1_Joint",
    "Rlink2_Joint",
    "Rlink3_Joint",
)

JOINT_NAMES = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES

BASE_LINK = "base_link"
TIP_LINK = "Gripping_point_Link"
TOPIC_NAME = "/joint_command"

# Stable pose shared with the Isaac Sim episode reset.
HOME_Q = np.array([0.8, 0.3, -0.5, 0.3, 0.0], dtype=float)

# The verified fixed-position controller targeted Gripping_point_Link 20 mm in
# +X from the cube centre. Keep the same geometric relationship while x and y
# are supplied by /dofbot/reset_episode for every episode.
TOOL_OFFSET = np.array([0.02, 0.0, 0.0], dtype=float)

# In the successful fixed-position run, cube centre z was 0.015 m and the
# pre-grasp target z was 0.120 m. Express that as a clearance from the actual
# cube centre returned by Isaac Sim so the code remains valid if floor/cube
# dimensions are changed later.
PREGRASP_CLEARANCE_ABOVE_CUBE_M = 0.105

# Move vertically downward from the working pre-grasp pose.
DESCEND_DISTANCE_M = 0.040

# Lift vertically after the gripper has closed.
LIFT_DISTANCE_M = 0.080

# Previously verified DOFBOT gripper commands.
GRIPPER_OPEN_Q = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=float,
)
GRIPPER_Q_VALUE = 1.6
GRIPPER_CLOSED_Q = np.array(
    [-GRIPPER_Q_VALUE, GRIPPER_Q_VALUE, -GRIPPER_Q_VALUE, GRIPPER_Q_VALUE, -GRIPPER_Q_VALUE, GRIPPER_Q_VALUE],
    dtype=float,
)

PUBLISH_RATE_HZ = 10.0
MOVE_TO_PREGRASP_SEC = 5.0
DESCEND_DURATION_SEC = 2.5
GRIPPER_CLOSE_DURATION_SEC = 1.5
GRIPPER_SETTLE_SEC = 1.0
LIFT_DURATION_SEC = 3.0
FINAL_HOLD_SEC = 5.0
# Gripper orientation constraint.
# This is the axis of Gripping_point_Link that should point toward the object.
# Start with local +Z. If the gripper points the wrong way, try [0, 0, -1],
# [1, 0, 0], [-1, 0, 0], [0, 1, 0], or [0, -1, 0].
TOOL_APPROACH_AXIS_LOCAL = np.array([1.0, 0.0, 0.0], dtype=float)

# World -Z means vertically downward in the Z-up Isaac Sim stage.
TARGET_APPROACH_AXIS_WORLD = np.array([0.0, 0.0, -1.0], dtype=float)

# Position errors are measured in metres, while axis errors are dimensionless.
# This weight balances downward orientation against position accuracy.
IK_ORIENTATION_WEIGHT = 0.10

# Regularization keeps the solution near HOME_Q and selects a less-twisted
# solution when several configurations satisfy the task.
IK_REGULARIZATION_WEIGHT = 0.02

# Keep the wrist-roll joint at its initial HOME_Q angle during IK.
# For this DOFBOT model, arm5_Joint is treated as the wrist rotation.
LOCK_WRIST_ROTATION = True
WRIST_JOINT_NAME = "arm5_Joint"

IK_WARNING_ERROR_M = 0.02
IK_WARNING_ORIENTATION_DEG = 10.0


# =============================================================================
# URDF model
# =============================================================================

@dataclass(frozen=True)
class JointInfo:
    name: str
    parent: str
    child: str
    joint_type: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


def _parse_vector(
    text: Optional[str],
    default: Sequence[float],
    *,
    field_name: str,
) -> np.ndarray:
    """Parse a three-element URDF vector."""
    values = np.asarray(
        default if text is None else [float(value) for value in text.split()],
        dtype=float,
    )
    if values.shape != (3,):
        raise ValueError(f"{field_name} must contain exactly 3 values: {text!r}")
    return values


def rpy_to_rot(rpy: np.ndarray) -> np.ndarray:
    """Convert URDF fixed-axis roll-pitch-yaw to a rotation matrix."""
    roll, pitch, yaw = rpy

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ]
    )
    ry = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ]
    )
    rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return rz @ ry @ rx


def axis_angle_to_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """Convert an axis-angle rotation to a rotation matrix."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1.0e-12:
        return np.eye(3)

    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_minus_c = 1.0 - c

    return np.array(
        [
            [
                c + x * x * one_minus_c,
                x * y * one_minus_c - z * s,
                x * z * one_minus_c + y * s,
            ],
            [
                y * x * one_minus_c + z * s,
                c + y * y * one_minus_c,
                y * z * one_minus_c - x * s,
            ],
            [
                z * x * one_minus_c - y * s,
                z * y * one_minus_c + x * s,
                c + z * z * one_minus_c,
            ],
        ]
    )


def make_origin_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rpy_to_rot(rpy)
    transform[:3, 3] = xyz
    return transform


def make_joint_motion(joint: JointInfo, position: float) -> np.ndarray:
    transform = np.eye(4)

    if joint.joint_type in ("revolute", "continuous"):
        transform[:3, :3] = axis_angle_to_rot(joint.axis, position)
    elif joint.joint_type == "prismatic":
        transform[:3, 3] = joint.axis * position

    return transform


def resolve_urdf_path() -> Path:
    """Resolve the installed URDF through the ROS 2 ament index."""
    try:
        package_share = get_package_share_path(DESCRIPTION_PACKAGE)
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"ROS 2 package '{DESCRIPTION_PACKAGE}' was not found. "
            "Build the workspace and run: "
            "source ~/ros2_ws/install/setup.bash"
        ) from exc

    urdf_path = package_share / URDF_RELATIVE_PATH
    if not urdf_path.is_file():
        raise FileNotFoundError(
            f"URDF was not installed: {urdf_path}\n"
            "Check dofbot_description/CMakeLists.txt and rebuild the workspace."
        )

    return urdf_path


def parse_urdf_joints(urdf_path: Path) -> Dict[str, JointInfo]:
    """Read joint geometry and limits required by the small IK solver."""
    root = ET.parse(urdf_path).getroot()
    joints: Dict[str, JointInfo] = {}

    for joint_element in root.findall("joint"):
        name = joint_element.attrib["name"]
        joint_type = joint_element.attrib.get("type", "fixed")

        parent_element = joint_element.find("parent")
        child_element = joint_element.find("child")
        if parent_element is None or child_element is None:
            raise ValueError(f"Joint '{name}' has no parent or child element")

        parent = parent_element.attrib["link"]
        child = child_element.attrib["link"]

        origin_element = joint_element.find("origin")
        if origin_element is None:
            xyz = np.zeros(3, dtype=float)
            rpy = np.zeros(3, dtype=float)
        else:
            xyz = _parse_vector(
                origin_element.attrib.get("xyz"),
                [0.0, 0.0, 0.0],
                field_name=f"joint '{name}' origin xyz",
            )
            rpy = _parse_vector(
                origin_element.attrib.get("rpy"),
                [0.0, 0.0, 0.0],
                field_name=f"joint '{name}' origin rpy",
            )

        axis_element = joint_element.find("axis")
        axis = _parse_vector(
            None if axis_element is None else axis_element.attrib.get("xyz"),
            [1.0, 0.0, 0.0],
            field_name=f"joint '{name}' axis",
        )

        # Continuous joints do not normally have lower/upper position limits.
        if joint_type == "continuous":
            lower = -math.pi
            upper = math.pi
        else:
            limit_element = joint_element.find("limit")
            if limit_element is not None:
                lower = float(limit_element.attrib.get("lower", -math.pi))
                upper = float(limit_element.attrib.get("upper", math.pi))
            else:
                lower = -math.pi
                upper = math.pi

        if lower > upper:
            raise ValueError(
                f"Invalid limits for joint '{name}': lower={lower}, upper={upper}"
            )

        joints[name] = JointInfo(
            name=name,
            parent=parent,
            child=child,
            joint_type=joint_type,
            xyz=xyz,
            rpy=rpy,
            axis=axis,
            lower=lower,
            upper=upper,
        )

    return joints


def build_chain(
    joints: Dict[str, JointInfo],
    base_link: str,
    tip_link: str,
) -> List[JointInfo]:
    """Build the unique URDF chain from base_link to tip_link."""
    child_to_joint = {joint.child: joint for joint in joints.values()}
    chain_reversed: List[JointInfo] = []
    visited_links = set()
    current_link = tip_link

    while current_link != base_link:
        if current_link in visited_links:
            raise RuntimeError(f"Loop detected while tracing link '{current_link}'")
        visited_links.add(current_link)

        joint = child_to_joint.get(current_link)
        if joint is None:
            raise RuntimeError(
                f"Cannot find a parent joint while tracing "
                f"'{base_link}' -> '{tip_link}'. Missing link: '{current_link}'"
            )

        chain_reversed.append(joint)
        current_link = joint.parent

    return list(reversed(chain_reversed))


def forward_kinematics(
    chain: Sequence[JointInfo],
    joint_positions: Dict[str, float],
) -> np.ndarray:
    transform = np.eye(4)

    for joint in chain:
        transform = transform @ make_origin_transform(joint.xyz, joint.rpy)
        transform = transform @ make_joint_motion(
            joint,
            joint_positions.get(joint.name, 0.0),
        )

    return transform


def _normalized(vector: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        raise ValueError(f"{name} must not be a zero vector")
    return vector / norm


def solve_position_axis_ik(
    chain: Sequence[JointInfo],
    target_xyz: np.ndarray,
    target_approach_axis_world: np.ndarray,
    tool_approach_axis_local: np.ndarray,
    initial_q: np.ndarray,
) -> np.ndarray:
    """Solve position IK while aligning one tool axis with a world axis.

    The DOFBOT arm has five arm joints. A full Cartesian pose has six degrees
    of freedom, so constraining all three position and all three orientation
    components is generally over-constrained. Position plus one direction axis
    is effectively a five-DOF task and is appropriate for pointing the gripper
    downward while leaving rotation around that axis free.
    """
    target_xyz = np.asarray(target_xyz, dtype=float)
    initial_q = np.asarray(initial_q, dtype=float)
    target_axis = _normalized(
        target_approach_axis_world, name="target_approach_axis_world"
    )
    tool_axis_local = _normalized(
        tool_approach_axis_local, name="tool_approach_axis_local"
    )

    if target_xyz.shape != (3,):
        raise ValueError(f"target_xyz must have shape (3,), got {target_xyz.shape}")
    if initial_q.shape != (len(ARM_JOINT_NAMES),):
        raise ValueError(
            f"initial_q must have {len(ARM_JOINT_NAMES)} values, "
            f"got {initial_q.shape}"
        )

    chain_joint_map = {joint.name: joint for joint in chain}
    missing_joints = [
        joint_name
        for joint_name in ARM_JOINT_NAMES
        if joint_name not in chain_joint_map
    ]
    if missing_joints:
        raise RuntimeError(
            "The base-to-tip chain does not contain all arm joints: "
            + ", ".join(missing_joints)
        )

    active_joints = [chain_joint_map[name] for name in ARM_JOINT_NAMES]
    lower = np.array([joint.lower for joint in active_joints], dtype=float)
    upper = np.array([joint.upper for joint in active_joints], dtype=float)
    initial_q = np.clip(initial_q, lower, upper)

    if WRIST_JOINT_NAME not in ARM_JOINT_NAMES:
        raise RuntimeError(
            f"WRIST_JOINT_NAME is not an arm joint: {WRIST_JOINT_NAME}"
        )

    wrist_index = ARM_JOINT_NAMES.index(WRIST_JOINT_NAME)
    if LOCK_WRIST_ROTATION:
        free_indices = np.array(
            [index for index in range(len(ARM_JOINT_NAMES)) if index != wrist_index],
            dtype=int,
        )
    else:
        free_indices = np.arange(len(ARM_JOINT_NAMES), dtype=int)

    initial_free_q = initial_q[free_indices]
    lower_free = lower[free_indices]
    upper_free = upper[free_indices]

    def expand_free_q(free_q: np.ndarray) -> np.ndarray:
        """Restore the full arm vector, keeping the wrist at HOME_Q."""
        full_q = initial_q.copy()
        full_q[free_indices] = free_q
        return full_q

    def evaluate(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_map = {
            name: float(value)
            for name, value in zip(ARM_JOINT_NAMES, q, strict=True)
        }
        transform = forward_kinematics(chain, q_map)
        current_xyz = transform[:3, 3]
        current_axis = _normalized(
            transform[:3, :3] @ tool_axis_local,
            name="current_tool_axis_world",
        )
        return current_xyz, current_axis

    def residual(free_q: np.ndarray) -> np.ndarray:
        q = expand_free_q(free_q)
        current_xyz, current_axis = evaluate(q)
        position_error = current_xyz - target_xyz

        # Difference of unit vectors is zero only when both axes have the same
        # direction. Rotation around that axis is intentionally left free,
        # while arm5_Joint is held exactly at its initial angle.
        axis_error = IK_ORIENTATION_WEIGHT * (current_axis - target_axis)
        regularization = IK_REGULARIZATION_WEIGHT * (
            free_q - initial_free_q
        )
        return np.concatenate((position_error, axis_error, regularization))

    result = least_squares(
        residual,
        initial_free_q,
        bounds=(lower_free, upper_free),
        xtol=1.0e-8,
        ftol=1.0e-8,
        gtol=1.0e-8,
        max_nfev=1000,
    )

    solution = expand_free_q(np.asarray(result.x, dtype=float))
    final_xyz, final_axis = evaluate(solution)
    position_error_m = float(np.linalg.norm(final_xyz - target_xyz))
    axis_dot = float(np.clip(np.dot(final_axis, target_axis), -1.0, 1.0))
    orientation_error_deg = math.degrees(math.acos(axis_dot))

    print("\n========== IK result ==========")
    print(f"success              = {result.success}")
    print(f"status               = {result.status}")
    print(f"message              = {result.message}")
    print(f"target_xyz           = {target_xyz}")
    print(f"final_xyz            = {final_xyz}")
    print(f"target_tool_axis     = {target_axis}")
    print(f"final_tool_axis      = {final_axis}")
    print(f"q                    = {solution}")
    print(f"wrist locked         = {LOCK_WRIST_ROTATION}")
    print(
        f"{WRIST_JOINT_NAME}       = {solution[wrist_index]:.6f} rad "
        f"(initial {initial_q[wrist_index]:.6f} rad)"
    )
    print(f"position error       = {position_error_m:.6f} m")
    print(f"orientation error    = {orientation_error_deg:.3f} deg")

    if not result.success:
        print("WARNING: scipy did not report successful convergence.")
    if position_error_m > IK_WARNING_ERROR_M:
        print(
            f"WARNING: IK position error exceeds "
            f"{IK_WARNING_ERROR_M:.3f} m. Check the target and weights."
        )
    if orientation_error_deg > IK_WARNING_ORIENTATION_DEG:
        print(
            f"WARNING: tool-axis error exceeds "
            f"{IK_WARNING_ORIENTATION_DEG:.1f} degrees. "
            "Check TOOL_APPROACH_AXIS_LOCAL or target reachability."
        )

    return solution


def gripper_open() -> np.ndarray:
    return GRIPPER_OPEN_Q.copy()


def gripper_closed() -> np.ndarray:
    return GRIPPER_CLOSED_Q.copy()


def lerp(start: np.ndarray, goal: np.ndarray, ratio: float) -> np.ndarray:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    return (1.0 - ratio) * start + ratio * goal


def smooth_ratio(elapsed: float, duration: float) -> float:
    """Return a cubic smooth-step ratio in the range [0, 1]."""
    if duration <= 0.0:
        return 1.0
    ratio = float(np.clip(elapsed / duration, 0.0, 1.0))
    return ratio * ratio * (3.0 - 2.0 * ratio)


# =============================================================================
# ROS 2 imitation-learning episode runner
# =============================================================================

RESET_SERVICE_NAME = "/dofbot/reset_episode"
END_SERVICE_NAME = "/dofbot/end_episode"
EPISODE_STATE_TOPIC = "/dofbot/episode_state"
SCENE_LAYOUT_TOPIC = "/dofbot/scene_layout"
TASK_INSTRUCTION_TOPIC = "/dofbot/task_instruction"
SUPPORTED_TARGET_COLORS = ("red", "blue", "green")

MODE_RANDOM = 0
MODE_SPECIFIED = 1

PHASE_IDLE = 0
PHASE_RESETTING = 1
PHASE_READY = 2
PHASE_RUNNING = 3
PHASE_SUCCESS = 4
PHASE_TERMINATED = 5

SERVICE_WAIT_TIMEOUT_SEC = 30.0
RESET_RESPONSE_TIMEOUT_SEC = 10.0
READY_TIMEOUT_SEC = 10.0
SCENE_LAYOUT_TIMEOUT_SEC = 10.0
TASK_ACK_TIMEOUT_SEC = 3.0
END_RESPONSE_TIMEOUT_SEC = 10.0


@dataclass(frozen=True)
class EpisodeTargets:
    cube_xyz: np.ndarray
    pregrasp_xyz: np.ndarray
    grasp_xyz: np.ndarray
    lift_xyz: np.ndarray
    pregrasp_q: np.ndarray
    grasp_q: np.ndarray
    lift_q: np.ndarray


@dataclass(frozen=True)
class SceneTarget:
    episode_id: int
    target_color: str
    task_instruction: str
    cube_xyz: np.ndarray


class DofbotSmolVlaEpisodeRunner(Node):
    """Collect one or more IK-teacher episodes against the Isaac Sim API.

    For every episode this node:
    1. chooses a target color and publishes its language instruction,
    2. calls /dofbot/reset_episode,
    3. reads the target cube pose from the READY /dofbot/scene_layout JSON,
    4. solves IK and publishes the proven trajectory to /joint_command,
    5. watches /dofbot/episode_state for target-specific success/failure, and
    6. explicitly ends the episode if the sequence finishes without success.

    Isaac Sim owns synchronized data recording and the final success label.
    """

    def __init__(self) -> None:
        super().__init__("dofbot_smolvla_episode_runner")

        self._declare_parameters()
        self._read_parameters()
        self._validate_settings()

        self.publisher = self.create_publisher(JointState, TOPIC_NAME, 10)

        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.task_instruction_publisher = self.create_publisher(
            String,
            TASK_INSTRUCTION_TOPIC,
            state_qos,
        )
        self.reset_client = self.create_client(ResetEpisode, RESET_SERVICE_NAME)
        self.end_client = self.create_client(EndEpisode, END_SERVICE_NAME)
        self.state_subscription = self.create_subscription(
            EpisodeState,
            EPISODE_STATE_TOPIC,
            self._on_episode_state,
            state_qos,
        )
        self.scene_layout_subscription = self.create_subscription(
            String,
            SCENE_LAYOUT_TOPIC,
            self._on_scene_layout,
            state_qos,
        )

        urdf_path = resolve_urdf_path()
        self.get_logger().info(f"URDF: {urdf_path}")
        joints = parse_urdf_joints(urdf_path)
        self.chain = build_chain(joints, BASE_LINK, TIP_LINK)

        self.get_logger().info("IK chain:")
        for joint in self.chain:
            self.get_logger().info(
                f"  {joint.name}: {joint.parent} -> {joint.child}, "
                f"type={joint.joint_type}, axis={joint.axis.tolist()}"
            )

        self.home_q = HOME_Q.copy()
        self.latest_episode_state: Optional[EpisodeState] = None
        self.latest_scene_layout: Optional[dict] = None
        self.active_episode_id: Optional[int] = None
        self.active_target_color: Optional[str] = None
        self.active_task_instruction = ""
        self.target_rng = np.random.default_rng(
            None if self.target_seed < 0 else self.target_seed
        )

        self.get_logger().info(f"Publishing commands to: {TOPIC_NAME}")
        self.get_logger().info(f"Reset service: {RESET_SERVICE_NAME}")
        self.get_logger().info(f"End service: {END_SERVICE_NAME}")
        self.get_logger().info(f"Episode state: {EPISODE_STATE_TOPIC}")
        self.get_logger().info(f"Scene layout: {SCENE_LAYOUT_TOPIC}")
        self.get_logger().info(f"Task instruction: {TASK_INSTRUCTION_TOPIC}")
        self.get_logger().info(f"Target color mode: {self.target_color}")
        self.get_logger().info(
            f"Strict IK validation: {self.strict_ik_validation}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("episode_count", 1)
        self.declare_parameter("placement_mode", "random")
        self.declare_parameter("specified_x", 0.20)
        self.declare_parameter("specified_y", -0.05)
        self.declare_parameter("base_seed", -1)
        self.declare_parameter("target_color", "random")
        self.declare_parameter("target_seed", -1)
        self.declare_parameter("record", True)
        self.declare_parameter("stop_on_failure", False)
        # The proven fixed-position program only warned about IK residuals.
        # Keep that behavior by default; strict rejection can be enabled later.
        self.declare_parameter("strict_ik_validation", False)

    def _read_parameters(self) -> None:
        self.episode_count = int(self.get_parameter("episode_count").value)
        self.placement_mode = str(
            self.get_parameter("placement_mode").value
        ).strip().lower()
        self.specified_x = float(self.get_parameter("specified_x").value)
        self.specified_y = float(self.get_parameter("specified_y").value)
        self.base_seed = int(self.get_parameter("base_seed").value)
        self.target_color = str(
            self.get_parameter("target_color").value
        ).strip().lower()
        self.target_seed = int(self.get_parameter("target_seed").value)
        self.record_enabled = bool(self.get_parameter("record").value)
        self.stop_on_failure = bool(self.get_parameter("stop_on_failure").value)
        self.strict_ik_validation = bool(
            self.get_parameter("strict_ik_validation").value
        )

    def _validate_settings(self) -> None:
        if self.episode_count <= 0:
            raise ValueError("episode_count must be positive")
        if self.placement_mode not in ("random", "specified"):
            raise ValueError(
                "placement_mode must be either 'random' or 'specified'"
            )
        if self.target_color not in ("random",) + SUPPORTED_TARGET_COLORS:
            raise ValueError(
                "target_color must be random, red, blue, or green"
            )
        if PUBLISH_RATE_HZ <= 0.0:
            raise ValueError("PUBLISH_RATE_HZ must be positive")
        for name, value in (
            ("MOVE_TO_PREGRASP_SEC", MOVE_TO_PREGRASP_SEC),
            ("DESCEND_DURATION_SEC", DESCEND_DURATION_SEC),
            ("GRIPPER_CLOSE_DURATION_SEC", GRIPPER_CLOSE_DURATION_SEC),
            ("GRIPPER_SETTLE_SEC", GRIPPER_SETTLE_SEC),
            ("LIFT_DURATION_SEC", LIFT_DURATION_SEC),
            ("FINAL_HOLD_SEC", FINAL_HOLD_SEC),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must not be negative")
        if MOVE_TO_PREGRASP_SEC <= 0.0:
            raise ValueError("MOVE_TO_PREGRASP_SEC must be positive")
        if DESCEND_DISTANCE_M <= 0.0:
            raise ValueError("DESCEND_DISTANCE_M must be positive")
        if LIFT_DISTANCE_M <= 0.0:
            raise ValueError("LIFT_DISTANCE_M must be positive")

    def _on_episode_state(self, message: EpisodeState) -> None:
        self.latest_episode_state = message

    def _on_scene_layout(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            if not isinstance(payload, dict):
                raise ValueError("scene layout JSON root must be an object")
            self.latest_scene_layout = payload
        except Exception as exc:
            self.get_logger().warning(f"Ignored invalid scene layout JSON: {exc}")

    def publish_command(
        self,
        arm_q: np.ndarray,
        gripper_q: np.ndarray,
    ) -> None:
        arm_q = np.asarray(arm_q, dtype=float)
        gripper_q = np.asarray(gripper_q, dtype=float)

        if arm_q.shape != (len(ARM_JOINT_NAMES),):
            raise ValueError(
                f"arm_q must have {len(ARM_JOINT_NAMES)} values, "
                f"got {arm_q.shape}"
            )
        if gripper_q.shape != (len(GRIPPER_JOINT_NAMES),):
            raise ValueError(
                f"gripper_q must have {len(GRIPPER_JOINT_NAMES)} values, "
                f"got {gripper_q.shape}"
            )

        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = (
            [float(value) for value in arm_q]
            + [float(value) for value in gripper_q]
        )
        message.velocity = []
        message.effort = []
        self.publisher.publish(message)

    def wait_for_environment(self) -> None:
        deadline = time.monotonic() + SERVICE_WAIT_TIMEOUT_SEC
        while rclpy.ok():
            reset_ready = self.reset_client.wait_for_service(timeout_sec=0.2)
            end_ready = self.end_client.wait_for_service(timeout_sec=0.2)
            joint_subscriber_ready = self.publisher.get_subscription_count() > 0
            task_subscriber_ready = (
                self.task_instruction_publisher.get_subscription_count() > 0
            )

            if (
                reset_ready
                and end_ready
                and joint_subscriber_ready
                and task_subscriber_ready
            ):
                self.get_logger().info("Isaac Sim SmolVLA episode API is ready.")
                return

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for Isaac Sim. Required: "
                    f"{RESET_SERVICE_NAME}, {END_SERVICE_NAME}, subscribers on "
                    f"{TOPIC_NAME} and {TASK_INSTRUCTION_TOPIC}."
                )

            rclpy.spin_once(self, timeout_sec=0.05)

    def _wait_for_future(self, future, timeout_sec: float, operation: str):
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done():
            raise TimeoutError(f"Timed out while waiting for {operation}")
        exception = future.exception()
        if exception is not None:
            raise RuntimeError(f"{operation} failed: {exception}") from exception
        return future.result()

    def choose_target_color(self) -> str:
        if self.target_color != "random":
            return self.target_color
        index = int(self.target_rng.integers(0, len(SUPPORTED_TARGET_COLORS)))
        return SUPPORTED_TARGET_COLORS[index]

    @staticmethod
    def make_task_instruction(target_color: str) -> str:
        if target_color not in SUPPORTED_TARGET_COLORS:
            raise ValueError(f"Unsupported target color: {target_color}")
        return f"pick up a {target_color} box."

    def publish_task_instruction(self, target_color: str) -> str:
        instruction = self.make_task_instruction(target_color)
        message = String()
        message.data = instruction
        self.task_instruction_publisher.publish(message)
        self.active_target_color = target_color
        self.active_task_instruction = instruction
        self.get_logger().info(
            f"Task: {instruction!r} (target_color={target_color})"
        )

        deadline = time.monotonic() + TASK_ACK_TIMEOUT_SEC
        while rclpy.ok() and time.monotonic() < deadline:
            layout = self.latest_scene_layout
            if (
                isinstance(layout, dict)
                and str(layout.get("task_instruction", "")) == instruction
                and str(layout.get("target_cube_color", "")) == target_color
            ):
                return instruction
            rclpy.spin_once(self, timeout_sec=0.05)
        raise TimeoutError(
            f"Isaac Sim did not acknowledge task instruction {instruction!r}"
        )

    def wait_for_scene_target(
        self,
        *,
        episode_id: int,
        target_color: str,
    ) -> SceneTarget:
        deadline = time.monotonic() + SCENE_LAYOUT_TIMEOUT_SEC
        while rclpy.ok():
            layout = self.latest_scene_layout
            if isinstance(layout, dict):
                layout_episode_id = int(layout.get("episode_id", -1))
                layout_phase = int(layout.get("phase", -1))
                layout_target = str(layout.get("target_cube_color", ""))
                if (
                    layout_episode_id == episode_id
                    and layout_phase == PHASE_READY
                    and layout_target == target_color
                ):
                    cubes = layout.get("cubes", [])
                    if not isinstance(cubes, list):
                        raise ValueError("scene layout cubes must be a list")
                    for cube in cubes:
                        if not isinstance(cube, dict):
                            continue
                        if str(cube.get("color", "")) != target_color:
                            continue
                        position = np.asarray(
                            cube.get("initial_position", []), dtype=float
                        )
                        if position.shape != (3,) or not np.all(np.isfinite(position)):
                            raise ValueError(
                                f"Invalid {target_color} cube position: {position}"
                            )
                        return SceneTarget(
                            episode_id=episode_id,
                            target_color=target_color,
                            task_instruction=str(
                                layout.get("task_instruction", "")
                            ),
                            cube_xyz=position,
                        )
                    raise RuntimeError(
                        f"Scene layout contains no {target_color} cube"
                    )

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for READY scene layout for episode "
                    f"{episode_id}, target={target_color}"
                )
            rclpy.spin_once(self, timeout_sec=0.05)

        raise RuntimeError("ROS 2 shutdown while waiting for scene layout")

    def reset_episode(self, collection_index: int):
        request = ResetEpisode.Request()
        if self.placement_mode == "random":
            request.mode = MODE_RANDOM
            request.x = 0.0
            request.y = 0.0
        else:
            request.mode = MODE_SPECIFIED
            request.x = self.specified_x
            request.y = self.specified_y

        request.seed = (
            -1 if self.base_seed < 0 else self.base_seed + collection_index
        )
        request.record = self.record_enabled

        self.latest_episode_state = None
        self.active_episode_id = None

        self.get_logger().info(
            f"Requesting episode reset: index={collection_index}, "
            f"mode={self.placement_mode}, seed={request.seed}, "
            f"record={request.record}"
        )
        response = self._wait_for_future(
            self.reset_client.call_async(request),
            RESET_RESPONSE_TIMEOUT_SEC,
            "reset_episode",
        )
        if not response.accepted:
            raise RuntimeError(f"Reset rejected: {response.message}")

        self.active_episode_id = int(response.episode_id)
        self.get_logger().info(
            f"Reset accepted: episode_id={response.episode_id}, "
            f"seed={response.seed_used}. Waiting for READY scene layout."
        )
        return response

    def wait_until_ready(self, episode_id: int) -> None:
        deadline = time.monotonic() + READY_TIMEOUT_SEC
        while rclpy.ok():
            state = self.latest_episode_state
            if state is not None and int(state.episode_id) == episode_id:
                if int(state.phase) == PHASE_READY:
                    self.get_logger().info(f"Episode {episode_id} is READY.")
                    return
                if int(state.phase) in (PHASE_SUCCESS, PHASE_TERMINATED):
                    raise RuntimeError(
                        f"Episode {episode_id} terminated during reset: "
                        f"{state.termination_reason}"
                    )

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for episode {episode_id} to become READY"
                )
            rclpy.spin_once(self, timeout_sec=0.05)

    def build_episode_targets(self, scene_target: SceneTarget) -> EpisodeTargets:
        cube_xyz = np.asarray(scene_target.cube_xyz, dtype=float).copy()
        self.get_logger().info(
            f"Target cube: color={scene_target.target_color}, "
            f"position={cube_xyz.tolist()} m, "
            f"instruction={scene_target.task_instruction!r}"
        )
        pregrasp_xyz = (
            cube_xyz
            + TOOL_OFFSET
            + np.array(
                [0.0, 0.0, PREGRASP_CLEARANCE_ABOVE_CUBE_M],
                dtype=float,
            )
        )
        grasp_xyz = pregrasp_xyz + np.array(
            [0.0, 0.0, -DESCEND_DISTANCE_M],
            dtype=float,
        )
        lift_xyz = grasp_xyz + np.array(
            [0.0, 0.0, LIFT_DISTANCE_M],
            dtype=float,
        )

        self.get_logger().info(
            f"Pre-grasp target: {pregrasp_xyz.tolist()} m"
        )
        pregrasp_q = solve_position_axis_ik(
            chain=self.chain,
            target_xyz=pregrasp_xyz,
            target_approach_axis_world=TARGET_APPROACH_AXIS_WORLD,
            tool_approach_axis_local=TOOL_APPROACH_AXIS_LOCAL,
            initial_q=self.home_q,
        )

        self.get_logger().info(
            f"Grasp target: {grasp_xyz.tolist()} m "
            f"(descend {DESCEND_DISTANCE_M * 1000.0:.1f} mm)"
        )
        grasp_q = solve_position_axis_ik(
            chain=self.chain,
            target_xyz=grasp_xyz,
            target_approach_axis_world=TARGET_APPROACH_AXIS_WORLD,
            tool_approach_axis_local=TOOL_APPROACH_AXIS_LOCAL,
            initial_q=pregrasp_q,
        )

        self.get_logger().info(
            f"Lift target: {lift_xyz.tolist()} m "
            f"(lift {LIFT_DISTANCE_M * 1000.0:.1f} mm)"
        )
        lift_q = solve_position_axis_ik(
            chain=self.chain,
            target_xyz=lift_xyz,
            target_approach_axis_world=TARGET_APPROACH_AXIS_WORLD,
            tool_approach_axis_local=TOOL_APPROACH_AXIS_LOCAL,
            initial_q=grasp_q,
        )

        self._require_acceptable_ik(pregrasp_q, pregrasp_xyz, "pre-grasp")
        self._require_acceptable_ik(grasp_q, grasp_xyz, "grasp")
        self._require_acceptable_ik(lift_q, lift_xyz, "lift")

        return EpisodeTargets(
            cube_xyz=cube_xyz,
            pregrasp_xyz=pregrasp_xyz,
            grasp_xyz=grasp_xyz,
            lift_xyz=lift_xyz,
            pregrasp_q=pregrasp_q,
            grasp_q=grasp_q,
            lift_q=lift_q,
        )

    def _require_acceptable_ik(
        self,
        q: np.ndarray,
        target_xyz: np.ndarray,
        label: str,
    ) -> None:
        q_map = {
            name: float(value)
            for name, value in zip(ARM_JOINT_NAMES, q, strict=True)
        }
        transform = forward_kinematics(self.chain, q_map)
        actual_xyz = transform[:3, 3]
        actual_axis = _normalized(
            transform[:3, :3] @ TOOL_APPROACH_AXIS_LOCAL,
            name=f"{label}_tool_axis",
        )
        target_axis = _normalized(
            TARGET_APPROACH_AXIS_WORLD,
            name="target_approach_axis_world",
        )
        position_error = float(np.linalg.norm(actual_xyz - target_xyz))
        axis_dot = float(np.clip(np.dot(actual_axis, target_axis), -1.0, 1.0))
        orientation_error_deg = math.degrees(math.acos(axis_dot))

        problems: list[str] = []
        if position_error > IK_WARNING_ERROR_M:
            problems.append(
                f"position error {position_error:.6f} m "
                f"> {IK_WARNING_ERROR_M:.6f} m"
            )
        if orientation_error_deg > IK_WARNING_ORIENTATION_DEG:
            problems.append(
                f"orientation error {orientation_error_deg:.3f} deg "
                f"> {IK_WARNING_ORIENTATION_DEG:.3f} deg"
            )

        if not problems:
            return

        message = f"{label} IK quality warning: " + "; ".join(problems)
        if self.strict_ik_validation:
            raise RuntimeError("Rejected " + message)

        # The original fixed-position controller used the same IK solver and
        # proceeded after these warnings.  For now, preserve that proven
        # behavior and let Isaac Sim's physical success detector label the
        # episode.
        self.get_logger().warning(message + "; continuing because strict_ik_validation=false")

    def _current_terminal_state(self) -> Optional[EpisodeState]:
        state = self.latest_episode_state
        if state is None or self.active_episode_id is None:
            return None
        if int(state.episode_id) != self.active_episode_id:
            return None
        if bool(state.terminated) or int(state.phase) in (
            PHASE_SUCCESS,
            PHASE_TERMINATED,
        ):
            return state
        return None

    def _run_interpolation(
        self,
        *,
        label: str,
        start_arm_q: np.ndarray,
        goal_arm_q: np.ndarray,
        start_gripper_q: np.ndarray,
        goal_gripper_q: np.ndarray,
        duration: float,
    ) -> Optional[EpisodeState]:
        self.get_logger().info(label)
        if duration <= 0.0:
            self.publish_command(goal_arm_q, goal_gripper_q)
            rclpy.spin_once(self, timeout_sec=0.0)
            return self._current_terminal_state()

        period = 1.0 / PUBLISH_RATE_HZ
        start_time = time.monotonic()
        next_publish_time = start_time

        while rclpy.ok():
            terminal = self._current_terminal_state()
            if terminal is not None:
                return terminal

            now = time.monotonic()
            elapsed = now - start_time
            if now >= next_publish_time:
                ratio = smooth_ratio(elapsed, duration)
                self.publish_command(
                    lerp(start_arm_q, goal_arm_q, ratio),
                    lerp(start_gripper_q, goal_gripper_q, ratio),
                )
                next_publish_time += period

            if elapsed >= duration:
                self.publish_command(goal_arm_q, goal_gripper_q)
                return self._current_terminal_state()

            wait_time = min(max(next_publish_time - time.monotonic(), 0.0), 0.05)
            rclpy.spin_once(self, timeout_sec=wait_time)

        return self._current_terminal_state()

    def _hold_command(
        self,
        *,
        label: str,
        arm_q: np.ndarray,
        gripper_q: np.ndarray,
        duration: float,
    ) -> Optional[EpisodeState]:
        return self._run_interpolation(
            label=label,
            start_arm_q=arm_q,
            goal_arm_q=arm_q,
            start_gripper_q=gripper_q,
            goal_gripper_q=gripper_q,
            duration=duration,
        )

    def execute_pick_sequence(
        self,
        targets: EpisodeTargets,
    ) -> Optional[EpisodeState]:
        open_q = gripper_open()
        closed_q = gripper_closed()

        phases = (
            dict(
                label="Moving from home to pre-grasp with gripper open.",
                start_arm_q=self.home_q,
                goal_arm_q=targets.pregrasp_q,
                start_gripper_q=open_q,
                goal_gripper_q=open_q,
                duration=MOVE_TO_PREGRASP_SEC,
            ),
            dict(
                label=(
                    f"Descending {DESCEND_DISTANCE_M * 1000.0:.1f} mm."
                ),
                start_arm_q=targets.pregrasp_q,
                goal_arm_q=targets.grasp_q,
                start_gripper_q=open_q,
                goal_gripper_q=open_q,
                duration=DESCEND_DURATION_SEC,
            ),
            dict(
                label="Closing the gripper around the cube.",
                start_arm_q=targets.grasp_q,
                goal_arm_q=targets.grasp_q,
                start_gripper_q=open_q,
                goal_gripper_q=closed_q,
                duration=GRIPPER_CLOSE_DURATION_SEC,
            ),
            dict(
                label="Holding the grasp before lifting.",
                start_arm_q=targets.grasp_q,
                goal_arm_q=targets.grasp_q,
                start_gripper_q=closed_q,
                goal_gripper_q=closed_q,
                duration=GRIPPER_SETTLE_SEC,
            ),
            dict(
                label=(
                    f"Lifting {LIFT_DISTANCE_M * 1000.0:.1f} mm with "
                    "gripper closed."
                ),
                start_arm_q=targets.grasp_q,
                goal_arm_q=targets.lift_q,
                start_gripper_q=closed_q,
                goal_gripper_q=closed_q,
                duration=LIFT_DURATION_SEC,
            ),
            dict(
                label="Holding the lifted pose.",
                start_arm_q=targets.lift_q,
                goal_arm_q=targets.lift_q,
                start_gripper_q=closed_q,
                goal_gripper_q=closed_q,
                duration=FINAL_HOLD_SEC,
            ),
        )

        for phase in phases:
            terminal = self._run_interpolation(**phase)
            if terminal is not None:
                return terminal
        return self._current_terminal_state()

    def end_episode(self, *, truncated: bool, reason: str):
        request = EndEpisode.Request()
        request.truncated = bool(truncated)
        request.reason = str(reason)
        response = self._wait_for_future(
            self.end_client.call_async(request),
            END_RESPONSE_TIMEOUT_SEC,
            "end_episode",
        )
        if not response.accepted:
            raise RuntimeError(f"EndEpisode rejected: {response.message}")
        return response

    def _summarize_terminal_state(self, state: EpisodeState) -> bool:
        success = bool(state.success)
        self.get_logger().info(
            f"Episode {state.episode_id} terminal state: success={success}, "
            f"truncated={state.truncated}, lift={state.lift_height:.4f} m, "
            f"gripper_distance={state.gripper_distance:.4f} m, "
            f"reason={state.termination_reason}"
        )
        return success

    def run_collection(self) -> None:
        self.wait_for_environment()

        successes = 0
        failures = 0
        for collection_index in range(self.episode_count):
            if not rclpy.ok():
                break

            self.get_logger().info(
                "=" * 20
                + f" collection {collection_index + 1}/{self.episode_count} "
                + "=" * 20
            )

            target_color = self.choose_target_color()
            self.publish_task_instruction(target_color)

            reset_response = self.reset_episode(collection_index)
            episode_id = int(reset_response.episode_id)
            self.wait_until_ready(episode_id)
            scene_target = self.wait_for_scene_target(
                episode_id=episode_id,
                target_color=target_color,
            )

            try:
                targets = self.build_episode_targets(scene_target)
            except Exception as exc:
                self.get_logger().error(
                    f"Episode {episode_id} IK preparation failed: {exc}"
                )
                end_response = self.end_episode(
                    truncated=True,
                    reason=f"ik_preparation_failed: {exc}",
                )
                failures += 1
                self.get_logger().info(
                    f"Episode {end_response.episode_id} saved with "
                    f"{end_response.recorded_steps} steps at "
                    f"{end_response.dataset_path}"
                )
                if self.stop_on_failure:
                    break
                continue

            terminal_state = self.execute_pick_sequence(targets)
            if terminal_state is not None:
                episode_success = self._summarize_terminal_state(terminal_state)
            else:
                end_response = self.end_episode(
                    truncated=False,
                    reason="pick_sequence_completed_without_success",
                )
                episode_success = bool(end_response.success)
                self.get_logger().info(
                    f"Episode {end_response.episode_id} ended: "
                    f"success={end_response.success}, "
                    f"steps={end_response.recorded_steps}, "
                    f"dataset={end_response.dataset_path}"
                )

            if episode_success:
                successes += 1
            else:
                failures += 1
                if self.stop_on_failure:
                    break

        self.get_logger().info(
            f"Collection finished: successes={successes}, failures={failures}, "
            f"attempted={successes + failures}"
        )


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[DofbotSmolVlaEpisodeRunner] = None
    try:
        node = DofbotSmolVlaEpisodeRunner()
        node.run_collection()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f"Fatal error: {exc}")
        else:
            print(f"Fatal error: {exc}")
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()