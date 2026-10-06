#!/usr/bin/env python3
"""Pick and lift a cube with the DOFBOT while pointing the gripper downward.

This node:
- loads dofbot.urdf from the installed dofbot_description package,
- solves IK for pre-grasp, grasp, and lift positions,
- publishes arm and gripper commands as sensor_msgs/msg/JointState,
- waits for the Isaac Sim subscriber before starting,
- moves above the cube, descends, closes the gripper, and lifts the cube.

Expected ROS 2 packages:
- dofbot_description
- dofbot_control
"""

from __future__ import annotations

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
from sensor_msgs.msg import JointState


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

# Stable pose used in the previous environment.
HOME_Q = np.array([0.8, 0.3, -0.5, 0.3, 0], dtype=float)

# The current pose that successfully places Gripping_point_Link above the cube.
TOOL_OFFSET = np.array([0.02, 0.0, 0.0], dtype=float)
PREGRASP_XYZ = np.array([0.20, -0.05, 0.12], dtype=float) + TOOL_OFFSET

# Move vertically downward from the working pre-grasp pose.
# Start with 40 mm. Tune this value in small increments if the fingers are still
# above the cube or descend too far.
DESCEND_DISTANCE_M = 0.040
GRASP_XYZ = PREGRASP_XYZ + np.array([0.0, 0.0, -DESCEND_DISTANCE_M])

# Lift vertically after the gripper has closed.
LIFT_DISTANCE_M = 0.080
LIFT_XYZ = GRASP_XYZ + np.array([0.0, 0.0, LIFT_DISTANCE_M])

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
START_DELAY_SEC = 1.0
WAIT_FOR_SUBSCRIBER = True

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
# ROS 2 node
# =============================================================================

class IkMoveAboveBox(Node):
    def __init__(self) -> None:
        super().__init__("ik_pick_and_lift_box")

        durations = {
            "MOVE_TO_PREGRASP_SEC": MOVE_TO_PREGRASP_SEC,
            "DESCEND_DURATION_SEC": DESCEND_DURATION_SEC,
            "GRIPPER_CLOSE_DURATION_SEC": GRIPPER_CLOSE_DURATION_SEC,
            "GRIPPER_SETTLE_SEC": GRIPPER_SETTLE_SEC,
            "LIFT_DURATION_SEC": LIFT_DURATION_SEC,
            "FINAL_HOLD_SEC": FINAL_HOLD_SEC,
        }
        if PUBLISH_RATE_HZ <= 0.0:
            raise ValueError("PUBLISH_RATE_HZ must be positive")
        for name, value in durations.items():
            if value < 0.0:
                raise ValueError(f"{name} must not be negative")
        if MOVE_TO_PREGRASP_SEC == 0.0:
            raise ValueError("MOVE_TO_PREGRASP_SEC must be positive")
        if DESCEND_DISTANCE_M <= 0.0:
            raise ValueError("DESCEND_DISTANCE_M must be positive")
        if LIFT_DISTANCE_M <= 0.0:
            raise ValueError("LIFT_DISTANCE_M must be positive")

        self.publisher = self.create_publisher(JointState, TOPIC_NAME, 10)

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

        self.get_logger().info(
            f"Pre-grasp target: {PREGRASP_XYZ.tolist()} m"
        )
        self.pregrasp_q = solve_position_axis_ik(
            chain=self.chain,
            target_xyz=PREGRASP_XYZ,
            target_approach_axis_world=TARGET_APPROACH_AXIS_WORLD,
            tool_approach_axis_local=TOOL_APPROACH_AXIS_LOCAL,
            initial_q=self.home_q,
        )

        self.get_logger().info(
            f"Grasp target: {GRASP_XYZ.tolist()} m "
            f"(descend {DESCEND_DISTANCE_M * 1000.0:.1f} mm)"
        )
        self.grasp_q = solve_position_axis_ik(
            chain=self.chain,
            target_xyz=GRASP_XYZ,
            target_approach_axis_world=TARGET_APPROACH_AXIS_WORLD,
            tool_approach_axis_local=TOOL_APPROACH_AXIS_LOCAL,
            initial_q=self.pregrasp_q,
        )

        self.get_logger().info(
            f"Lift target: {LIFT_XYZ.tolist()} m "
            f"(lift {LIFT_DISTANCE_M * 1000.0:.1f} mm)"
        )
        self.lift_q = solve_position_axis_ik(
            chain=self.chain,
            target_xyz=LIFT_XYZ,
            target_approach_axis_world=TARGET_APPROACH_AXIS_WORLD,
            tool_approach_axis_local=TOOL_APPROACH_AXIS_LOCAL,
            initial_q=self.grasp_q,
        )

        self.done = False
        self._motion_start_time: Optional[float] = None
        self._last_wait_log_time = 0.0
        self._phase = "waiting"

        self.timer = self.create_timer(
            1.0 / PUBLISH_RATE_HZ,
            self._on_timer,
        )

        self.get_logger().info(f"Publishing to: {TOPIC_NAME}")
        self.get_logger().info(
            "Sequence: home -> pre-grasp -> descend -> close -> settle -> lift"
        )
        if WAIT_FOR_SUBSCRIBER:
            self.get_logger().info("Waiting for an Isaac Sim subscriber...")

    def publish_command(
        self,
        arm_q: np.ndarray,
        gripper_q: np.ndarray,
    ) -> None:
        arm_q = np.asarray(arm_q, dtype=float)
        gripper_q = np.asarray(gripper_q, dtype=float)

        if arm_q.shape != (len(ARM_JOINT_NAMES),):
            raise ValueError(
                f"arm_q must have {len(ARM_JOINT_NAMES)} values, got {arm_q.shape}"
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

        # Only positionCommand is connected in the Isaac Sim Action Graph.
        # Unused JointState arrays are therefore left empty.
        message.velocity = []
        message.effort = []

        self.publisher.publish(message)

    def _set_phase(self, phase: str, message: str) -> None:
        if self._phase != phase:
            self._phase = phase
            self.get_logger().info(message)

    def _on_timer(self) -> None:
        now = time.monotonic()

        if WAIT_FOR_SUBSCRIBER and self.publisher.get_subscription_count() == 0:
            if now - self._last_wait_log_time >= 2.0:
                self.get_logger().info(
                    f"No subscriber on {TOPIC_NAME}; start Isaac Sim and press Play."
                )
                self._last_wait_log_time = now
            return

        if self._motion_start_time is None:
            self._motion_start_time = now + START_DELAY_SEC
            self._set_phase(
                "delay",
                f"Subscriber detected. Starting in {START_DELAY_SEC:.1f} s.",
            )
            self.publish_command(self.home_q, gripper_open())
            return

        elapsed = now - self._motion_start_time
        if elapsed < 0.0:
            self.publish_command(self.home_q, gripper_open())
            return

        phase_end = MOVE_TO_PREGRASP_SEC
        if elapsed <= phase_end:
            self._set_phase(
                "move_to_pregrasp",
                "Moving to the pre-grasp pose above the cube with gripper open.",
            )
            ratio = smooth_ratio(elapsed, MOVE_TO_PREGRASP_SEC)
            self.publish_command(
                lerp(self.home_q, self.pregrasp_q, ratio),
                gripper_open(),
            )
            return

        phase_start = phase_end
        phase_end += DESCEND_DURATION_SEC
        if elapsed <= phase_end:
            self._set_phase(
                "descending",
                f"Descending {DESCEND_DISTANCE_M * 1000.0:.1f} mm.",
            )
            ratio = smooth_ratio(elapsed - phase_start, DESCEND_DURATION_SEC)
            self.publish_command(
                lerp(self.pregrasp_q, self.grasp_q, ratio),
                gripper_open(),
            )
            return

        phase_start = phase_end
        phase_end += GRIPPER_CLOSE_DURATION_SEC
        if elapsed <= phase_end:
            self._set_phase(
                "closing",
                "Closing the gripper around the cube.",
            )
            ratio = smooth_ratio(
                elapsed - phase_start,
                GRIPPER_CLOSE_DURATION_SEC,
            )
            self.publish_command(
                self.grasp_q,
                lerp(gripper_open(), gripper_closed(), ratio),
            )
            return

        phase_end += GRIPPER_SETTLE_SEC
        if elapsed <= phase_end:
            self._set_phase(
                "grip_settle",
                "Holding the closed gripper before lifting.",
            )
            self.publish_command(self.grasp_q, gripper_closed())
            return

        phase_start = phase_end
        phase_end += LIFT_DURATION_SEC
        if elapsed <= phase_end:
            self._set_phase(
                "lifting",
                f"Lifting {LIFT_DISTANCE_M * 1000.0:.1f} mm with gripper closed.",
            )
            ratio = smooth_ratio(elapsed - phase_start, LIFT_DURATION_SEC)
            self.publish_command(
                lerp(self.grasp_q, self.lift_q, ratio),
                gripper_closed(),
            )
            return

        phase_end += FINAL_HOLD_SEC
        if elapsed <= phase_end:
            self._set_phase(
                "final_hold",
                "Holding the lifted pose with gripper closed.",
            )
            self.publish_command(self.lift_q, gripper_closed())
            return

        # Send one final command and stop this node cleanly.
        self.publish_command(self.lift_q, gripper_closed())
        self.timer.cancel()
        self.done = True
        self._set_phase("done", "Pick-and-lift sequence completed.")


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[IkMoveAboveBox] = None

    try:
        node = IkMoveAboveBox()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
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