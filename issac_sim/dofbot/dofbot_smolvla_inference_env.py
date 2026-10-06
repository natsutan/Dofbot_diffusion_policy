from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# =============================================================================
# Command-line settings
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DOFBOT SmolVLA data-collection environment for Isaac Sim",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run without the Isaac Sim GUI. Default: GUI enabled.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "dofbot_dataset_raw",
        help="Directory in which raw episode folders are stored.",
    )
    parser.add_argument(
        "--cube-x",
        type=float,
        default=0.20,
        help="Fixed X coordinate used for all three cubes.",
    )
    parser.add_argument("--cube-y-min", type=float, default=-0.12)
    parser.add_argument("--cube-y-max", type=float, default=0.12)
    parser.add_argument(
        "--cube-clearance",
        type=float,
        default=0.010,
        help="Minimum free space between cube faces in meters.",
    )
    parser.add_argument(
        "--record-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Encode one RealSense RGB frame for every recorded state sample. "
            "Video is written only for reset requests whose record field is true."
        ),
    )
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument(
        "--publish-camera",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Publish the RealSense RGB stream as sensor_msgs/Image for online "
            "SmolVLA inference."
        ),
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=None,
        help=(
            "Optional master seed used when reset requests specify seed=-1. "
            "If omitted, NumPy initializes the generator from OS entropy."
        ),
    )
    args, _unknown = parser.parse_known_args()
    return args


ARGS = parse_arguments()

# =============================================================================
# ROS 2 / Isaac Sim startup settings
# =============================================================================
# These values must be set before the ROS 2 bridge extension is loaded.
os.environ.setdefault("ROS_DOMAIN_ID", "0")
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_DISTRO", "jazzy")

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": bool(ARGS.headless)})

# Isaac Sim and ROS imports must follow SimulationApp construction.
try:
    import omni.graph.core as og
    import omni.kit.app
    import omni.timeline
    import omni.usd

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

    _extension_manager = omni.kit.app.get_app().get_extension_manager()
    for _early_extension in (
        "isaacsim.asset.importer.urdf",
        "isaacsim.core.prims",
        "isaacsim.ros2.bridge",
        "isaacsim.storage.native",
        "omni.kit.viewport.utility",
        "isaacsim.sensors.experimental.rtx",
    ):
        _extension_manager.set_extension_enabled_immediate(
            _early_extension,
            True,
        )
    for _ in range(10):
        simulation_app.update()

    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    from isaacsim.core.prims import Articulation, RigidPrim, XFormPrim
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.storage.native import get_assets_root_path
    from isaacsim.sensors.experimental.rtx import CameraSensor
    from omni.kit.viewport.utility import create_viewport_window

    import rclpy
    from builtin_interfaces.msg import Time
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import String

    from dofbot_interfaces.msg import EpisodeState
    from dofbot_interfaces.srv import EndEpisode, ResetEpisode
except BaseException:
    print("\nFATAL: Isaac Sim import/startup failed.", file=sys.stderr, flush=True)
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    simulation_app.close(exit_code=1)
    raise


# =============================================================================
# User and environment settings
# =============================================================================

URDF_PATH = Path("/home/natu/dofbot/urdf/dofbot.urdf")
USD_OUTPUT_ROOT = Path("/home/natu/dofbot/generated")
FORCE_URDF_REIMPORT = False

ROBOT_REFERENCE_PATH = "/World/DOFBOT"
ROBOT_REFERENCE_POSITION = (0.0, 0.0, 0.0)
GRIPPER_POINT_LINK_NAME = "Gripping_point_Link"

JOINT_COMMAND_TOPIC = "/joint_command"
JOINT_STATE_TOPIC = "/joint_states"
EPISODE_STATE_TOPIC = "/dofbot/episode_state"
RESET_SERVICE = "/dofbot/reset_episode"
END_SERVICE = "/dofbot/end_episode"

ROS_DOMAIN_ID = 0
GRAPH_PATH = "/ActionGraph"

STAGE_UNITS_IN_METERS = 1.0
STAGE_UP_AXIS = "Z"

FLOOR_CENTER_Z = -0.005
FLOOR_HALF_THICKNESS = 0.005
FLOOR_TOP_Z = FLOOR_CENTER_Z + FLOOR_HALF_THICKNESS

CUBE_SIZE = 0.030
CUBE_MASS = 0.0027
CUBE_STATIC_FRICTION = 1.0
CUBE_DYNAMIC_FRICTION = 0.8
CUBE_RESTITUTION = 0.0
CUBE_ROOT_PATH = "/World/PickCubes"
CUBE_COLOR_NAMES = ("red", "blue", "green")
CUBE_PATHS = {
    "red": CUBE_ROOT_PATH + "/RedCube",
    "blue": CUBE_ROOT_PATH + "/BlueCube",
    "green": CUBE_ROOT_PATH + "/GreenCube",
}
COLOR_RGB = {
    "red": Gf.Vec3f(0.90, 0.05, 0.05),
    "blue": Gf.Vec3f(0.05, 0.20, 0.95),
    "green": Gf.Vec3f(0.05, 0.75, 0.15),
}


INITIAL_CUBE_Y = {
    "red": -0.10,
    "blue": 0.00,
    "green": 0.10,
}

TARGET_MARKER_ROOT_PATH = "/World/TargetMarkers"
TARGET_MARKER_POSITIONS_XY = (
    (0.20, 0.08),
    (0.20, 0.00),
    (0.20, -0.08),
)
TARGET_MARKER_PATHS = tuple(
    f"{TARGET_MARKER_ROOT_PATH}/Marker_{index}" for index in range(3)
)
TARGET_MARKER_COLOR_NAMES = ("white", "black", "yellow")
# TARGET_MARKER_COLOR_RGB = {
#     "white": Gf.Vec3f(0.95, 0.95, 0.95),
#     "black": Gf.Vec3f(0.01, 0.01, 0.01),
#     "yellow": Gf.Vec3f(0.95, 0.80, 0.02),
# }
TARGET_MARKER_COLOR_RGB = {
    "white": Gf.Vec3f(0.95, 0.95, 0.95),
    "black": Gf.Vec3f(0.95, 0.95, 0.95),
    "yellow": Gf.Vec3f(0.95, 0.95, 0.95),
}


# Flat visual-only circles. Adjacent centers are 80 mm apart, while each
# marker is 50 mm in diameter, leaving a 30 mm gap between marker regions.
TARGET_MARKER_DIAMETER = 0.050
TARGET_MARKER_RADIUS = TARGET_MARKER_DIAMETER / 2.0
TARGET_MARKER_SEGMENTS = 64
TARGET_MARKER_Z_OFFSET = 0.0002

SCENE_LAYOUT_TOPIC = "/dofbot/scene_layout"
TASK_INSTRUCTION_TOPIC = "/dofbot/task_instruction"
CAMERA_RGB_TOPIC = "/dofbot/camera/color/image_raw"
CAMERA_FRAME_ID = "realsense_color_optical_frame"

# Fixed RealSense D455 placement selected in the GUI.
REALSENSE_RIG_PATH = "/World/RealSenseRig"
REALSENSE_SENSOR_PATH = REALSENSE_RIG_PATH + "/D455"
REALSENSE_COLOR_CAMERA_PATH = (
    "/World/RealSenseRig/D455/RSD455/Camera_OmniVision_OV9782_Color"
)
REALSENSE_ASSET_RELATIVE_PATH = "/Isaac/Sensors/RealSense/D455/rsd455.usd"
REALSENSE_POSITION = (0.75, 0.00, 0.30)
REALSENSE_LOOK_AT = (0.0, 0.0, 0.20)
REALSENSE_ROLL_DEGREES = 180.0

# A second floating Viewport is opened at startup. The ordinary main Viewport
# remains available as the normal Perspective view.
REALSENSE_VIEWPORT_NAME = "RealSense RGB"
REALSENSE_VIEWPORT_WIDTH = 640
REALSENSE_VIEWPORT_HEIGHT = 480
REALSENSE_VIEWPORT_POSITION_X = 950
REALSENSE_VIEWPORT_POSITION_Y = 80

VIDEO_FILENAME = "camera_rgb.mp4"
VIDEO_FRAME_TIMESTAMPS_FILENAME = "camera_rgb_timestamps.npy"
VIDEO_FRAME_VALID_FILENAME = "camera_rgb_frame_valid.npy"
VIDEO_SAMPLE_INDICES_FILENAME = "camera_rgb_sample_indices.npy"

ARM_JOINT_NAMES = [
    "arm1_Joint",
    "arm2_Joint",
    "arm3_Joint",
    "arm4_Joint",
    "arm5_Joint",
]

GRIPPER_JOINT_NAMES = [
    "Llink1_Joint",
    "Llink2_Joint",
    "Llink3_Joint",
    "Rlink1_Joint",
    "Rlink2_Joint",
    "Rlink3_Joint",
]

DRIVEN_JOINT_NAMES = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES

# Same home pose as the current ROS 2 IK controller, with the gripper open.
HOME_JOINT_POSITION = np.asarray(
    [0.8, 0.3, -0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)

JOINT_DRIVE_STIFFNESS = 10000.0
JOINT_DRIVE_DAMPING = 1000.0
JOINT_DRIVE_MAX_FORCE = 100000.0

RECORD_RATE_HZ = 10.0
RECORD_PERIOD_SEC = 1.0 / RECORD_RATE_HZ
RESET_SETTLE_SEC = 0.50
EPISODE_TIMEOUT_SEC = 30.0

SUCCESS_LIFT_HEIGHT_M = 0.050
SUCCESS_GRIPPER_DISTANCE_M = 0.060
SUCCESS_HOLD_SEC = 0.50
CUBE_FALL_Z_M = -0.020

MODE_RANDOM = 0
MODE_SPECIFIED = 1

PHASE_IDLE = 0
PHASE_RESETTING = 1
PHASE_READY = 2
PHASE_RUNNING = 3
PHASE_SUCCESS = 4
PHASE_TERMINATED = 5


# =============================================================================
# Generic helpers
# =============================================================================


def extract_target_cube_color(task_instruction: str) -> str | None:
    """Extract exactly one supported cube color from a task instruction."""
    text = str(task_instruction).strip().lower()
    matches = {
        color
        for color in CUBE_COLOR_NAMES
        if re.search(rf"\b{re.escape(color)}\b", text) is not None
    }
    if len(matches) != 1:
        return None
    return next(iter(matches))


def enable_extension(extension_name: str) -> None:
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate(extension_name, True)
    if not ext_manager.is_extension_enabled(extension_name):
        raise RuntimeError(f"Failed to enable extension: {extension_name}")
    print(f"{extension_name} = enabled", flush=True)


def enable_required_extensions() -> None:
    for extension_name in (
        "isaacsim.asset.importer.urdf",
        "isaacsim.core.nodes",
        "isaacsim.core.prims",
        "isaacsim.ros2.bridge",
        "isaacsim.storage.native",
        "omni.kit.viewport.utility",
        "isaacsim.sensors.experimental.rtx",
        # Exposes the standard ROS 2 simulation_interfaces services/actions.
        "isaacsim.ros2.sim_control",
    ):
        enable_extension(extension_name)


def update_app(frame_count: int) -> None:
    for _ in range(frame_count):
        simulation_app.update()


def as_numpy(value) -> np.ndarray:
    """Convert numpy/torch/warp-like output into a detached numpy array."""
    if value is None:
        raise RuntimeError("Isaac Sim returned None instead of a physics value")
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def simulation_time_message(sim_time: float) -> Time:
    sec = math.floor(sim_time)
    nanosec = int(round((sim_time - sec) * 1_000_000_000.0))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    msg = Time()
    msg.sec = int(sec)
    msg.nanosec = int(nanosec)
    return msg


def next_episode_index(data_root: Path) -> int:
    maximum = -1
    if data_root.is_dir():
        for path in data_root.glob("episode_*"):
            if not path.is_dir():
                continue
            try:
                maximum = max(maximum, int(path.name.removeprefix("episode_")))
            except ValueError:
                continue
    return maximum + 1


# =============================================================================
# Stage setup
# =============================================================================


def create_new_stage():
    context = omni.usd.get_context()
    context.new_stage()
    update_app(5)

    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("Failed to create a new USD stage")

    UsdGeom.SetStageMetersPerUnit(stage, STAGE_UNITS_IN_METERS)
    UsdGeom.SetStageUpAxis(stage, STAGE_UP_AXIS)
    UsdGeom.Xform.Define(stage, "/World")
    return stage


def add_physics_scene(stage) -> None:
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)


def add_floor(stage) -> None:
    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, FLOOR_CENTER_Z))
    floor.AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, FLOOR_HALF_THICKNESS))

    collision = UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    collision.CreateCollisionEnabledAttr(True)

    material_path = "/World/Materials/FloorMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path + "/PBRShader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.2, 0.2, 0.2)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(floor.GetPrim()).Bind(material)


def add_pick_cubes(stage) -> None:
    UsdGeom.Xform.Define(stage, CUBE_ROOT_PATH)
    cube_center_z = FLOOR_TOP_Z + CUBE_SIZE / 2.0

    physics_material_path = "/World/Materials/PickCubePhysicsMaterial"
    physics_material = UsdShade.Material.Define(stage, physics_material_path)
    material_api = UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
    material_api.CreateStaticFrictionAttr(CUBE_STATIC_FRICTION)
    material_api.CreateDynamicFrictionAttr(CUBE_DYNAMIC_FRICTION)
    material_api.CreateRestitutionAttr(CUBE_RESTITUTION)

    print("\n========== Pick cubes created ==========")
    for color_name in CUBE_COLOR_NAMES:
        cube_path = CUBE_PATHS[color_name]
        cube = UsdGeom.Cube.Define(stage, cube_path)
        cube.CreateSizeAttr(CUBE_SIZE)
        cube.CreateExtentAttr(
            [
                Gf.Vec3f(-CUBE_SIZE / 2.0),
                Gf.Vec3f(CUBE_SIZE / 2.0),
            ]
        )
        cube.AddTranslateOp().Set(
            Gf.Vec3d(
                float(ARGS.cube_x),
                INITIAL_CUBE_Y[color_name],
                cube_center_z,
            )
        )
        cube.CreateDisplayColorAttr().Set([COLOR_RGB[color_name]])

        cube_prim = cube.GetPrim()
        collision = UsdPhysics.CollisionAPI.Apply(cube_prim)
        collision.CreateCollisionEnabledAttr(True)

        rigid_body = UsdPhysics.RigidBodyAPI.Apply(cube_prim)
        rigid_body.CreateRigidBodyEnabledAttr(True)
        rigid_body.CreateKinematicEnabledAttr(False)
        rigid_body.CreateStartsAsleepAttr(False)

        mass = UsdPhysics.MassAPI.Apply(cube_prim)
        mass.CreateMassAttr(CUBE_MASS)

        UsdShade.MaterialBindingAPI.Apply(cube_prim).Bind(
            physics_material,
            UsdShade.Tokens.weakerThanDescendants,
            "physics",
        )
        print(
            f"{color_name:5s}: path={cube_path}, "
            f"position=({ARGS.cube_x:.3f}, {INITIAL_CUBE_Y[color_name]:.3f}, "
            f"{cube_center_z:.3f})"
        )


def set_marker_color(stage, marker_path: str, color_name: str) -> None:
    marker_prim = stage.GetPrimAtPath(marker_path)
    if not marker_prim.IsValid():
        raise RuntimeError(f"Invalid target marker prim: {marker_path}")
    if color_name not in TARGET_MARKER_COLOR_RGB:
        raise ValueError(f"Unsupported target marker color: {color_name}")
    gprim = UsdGeom.Gprim(marker_prim)
    gprim.GetDisplayColorAttr().Set([TARGET_MARKER_COLOR_RGB[color_name]])


def create_flat_circle_mesh(
    stage,
    marker_path: str,
    *,
    radius: float,
    segments: int,
) -> UsdGeom.Mesh:
    """Create a zero-thickness, visual-only circular mesh in the local XY plane."""
    if radius <= 0.0:
        raise ValueError("Circle radius must be positive")
    if segments < 3:
        raise ValueError("Circle mesh requires at least three segments")

    mesh = UsdGeom.Mesh.Define(stage, marker_path)
    points = [Gf.Vec3f(0.0, 0.0, 0.0)]
    points.extend(
        Gf.Vec3f(
            radius * math.cos(2.0 * math.pi * index / segments),
            radius * math.sin(2.0 * math.pi * index / segments),
            0.0,
        )
        for index in range(segments)
    )

    face_vertex_counts = [3] * segments
    face_vertex_indices: list[int] = []
    for index in range(segments):
        current = index + 1
        following = ((index + 1) % segments) + 1
        face_vertex_indices.extend((0, current, following))

    mesh.CreatePointsAttr().Set(points)
    mesh.CreateFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr().Set(face_vertex_indices)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr().Set(
        [
            Gf.Vec3f(-radius, -radius, 0.0),
            Gf.Vec3f(radius, radius, 0.0),
        ]
    )
    # No CollisionAPI, RigidBodyAPI, or MassAPI is applied. The marker is only
    # rendered and has no effect on physics.
    return mesh


def add_target_markers(stage) -> None:
    UsdGeom.Xform.Define(stage, TARGET_MARKER_ROOT_PATH)
    marker_z = FLOOR_TOP_Z + TARGET_MARKER_Z_OFFSET
    print("\n========== Target markers created ==========")
    for index, (marker_path, position_xy, color_name) in enumerate(
        zip(
            TARGET_MARKER_PATHS,
            TARGET_MARKER_POSITIONS_XY,
            TARGET_MARKER_COLOR_NAMES,
            strict=True,
        )
    ):
        marker = create_flat_circle_mesh(
            stage,
            marker_path,
            radius=TARGET_MARKER_RADIUS,
            segments=TARGET_MARKER_SEGMENTS,
        )
        marker.AddTranslateOp().Set(
            Gf.Vec3d(position_xy[0], position_xy[1], marker_z)
        )
        marker.CreateDisplayColorAttr().Set([TARGET_MARKER_COLOR_RGB[color_name]])
        print(
            f"slot={index}: path={marker_path}, color={color_name}, "
            f"position=({position_xy[0]:.3f}, {position_xy[1]:.3f}), "
            f"diameter={TARGET_MARKER_DIAMETER:.3f} m, visual_only=True"
        )


def add_light(stage) -> None:
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(3000.0)
    dome.CreateExposureAttr(0.0)


def add_realsense_camera(stage) -> str:
    """Add a fixed RealSense D455 asset for interactive GUI placement."""
    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError("Isaac Sim assets root path could not be resolved")

    sensor_asset_url = assets_root + REALSENSE_ASSET_RELATIVE_PATH

    # Keep the sensor USD under a neutral parent Xform. Editing this parent in
    # the GUI moves the complete D455 rig without modifying its internal camera
    # calibration or child transforms.
    rig = UsdGeom.Xform.Define(stage, REALSENSE_RIG_PATH)
    position = Gf.Vec3d(*REALSENSE_POSITION)
    target = Gf.Vec3d(*REALSENSE_LOOK_AT)
    forward = target - position
    if forward.GetLength() < 1.0e-9:
        raise ValueError("RealSense position and look-at point must differ")
    forward.Normalize()

    # The D455 rig uses +X as its forward direction. First roll the rig around
    # its local +X axis, then rotate +X toward the workspace target. A 180-degree
    # local roll preserves the viewing direction while correcting an upside-down
    # color image.
    look_rotation = Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), forward)
    upright_roll = Gf.Rotation(
        Gf.Vec3d(1.0, 0.0, 0.0),
        REALSENSE_ROLL_DEGREES,
    )
    rotation = upright_roll * look_rotation
    transform = Gf.Matrix4d(1.0)
    transform.SetRotate(rotation)
    transform.SetTranslateOnly(position)
    rig.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(transform)

    stage_utils.add_reference_to_stage(
        usd_path=sensor_asset_url,
        path=REALSENSE_SENSOR_PATH,
    )
    update_app(30)

    sensor_prim = stage.GetPrimAtPath(REALSENSE_SENSOR_PATH)
    if not sensor_prim.IsValid():
        raise RuntimeError(
            f"RealSense D455 reference was not created at {REALSENSE_SENSOR_PATH}"
        )

    # The supplied D455 asset contains a rigid body. This camera is an external,
    # fixed camera, so disable every rigid body below the referenced sensor to
    # prevent it from falling when the timeline starts.
    disabled_rigid_bodies: list[str] = []
    camera_paths: list[str] = []
    for prim in Usd.PrimRange(sensor_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body = UsdPhysics.RigidBodyAPI(prim)
            rigid_body.CreateRigidBodyEnabledAttr(False)
            disabled_rigid_bodies.append(str(prim.GetPath()))
        if prim.IsA(UsdGeom.Camera):
            camera_paths.append(str(prim.GetPath()))

    color_camera_prim = stage.GetPrimAtPath(REALSENSE_COLOR_CAMERA_PATH)
    if not color_camera_prim.IsValid() or not color_camera_prim.IsA(UsdGeom.Camera):
        available = "\n".join(f"  {path}" for path in camera_paths)
        raise RuntimeError(
            "The configured RealSense color Camera prim was not found: "
            f"{REALSENSE_COLOR_CAMERA_PATH}\n"
            f"Available camera prims:\n{available}"
        )

    color_camera_path = REALSENSE_COLOR_CAMERA_PATH

    print("\n========== RealSense D455 added ==========")
    print(f"rig path       = {REALSENSE_RIG_PATH}")
    print(f"sensor path    = {REALSENSE_SENSOR_PATH}")
    print(f"asset URL      = {sensor_asset_url}")
    print(
        "position       = "
        f"({REALSENSE_POSITION[0]:.3f}, "
        f"{REALSENSE_POSITION[1]:.3f}, {REALSENSE_POSITION[2]:.3f}) m"
    )
    print(
        "look-at point  = "
        f"({REALSENSE_LOOK_AT[0]:.3f}, "
        f"{REALSENSE_LOOK_AT[1]:.3f}, {REALSENSE_LOOK_AT[2]:.3f}) m"
    )
    print(f"rig roll       = {REALSENSE_ROLL_DEGREES:.1f} degrees")
    print(f"rigid bodies disabled = {len(disabled_rigid_bodies)}")
    print(f"RGB camera path = {color_camera_path}")

    return color_camera_path


def create_realsense_viewport(camera_path: str):
    """Create a second GUI Viewport while leaving the main Perspective view intact."""
    if ARGS.headless:
        print("Headless mode: the RealSense GUI Viewport was not created.")
        return None

    camera_prim = omni.usd.get_context().get_stage().GetPrimAtPath(camera_path)
    if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"Invalid RealSense Camera prim for Viewport: {camera_path}")

    viewport_window = create_viewport_window(
        name=REALSENSE_VIEWPORT_NAME,
        width=REALSENSE_VIEWPORT_WIDTH,
        height=REALSENSE_VIEWPORT_HEIGHT,
        position_x=REALSENSE_VIEWPORT_POSITION_X,
        position_y=REALSENSE_VIEWPORT_POSITION_Y,
        camera_path=Sdf.Path(camera_path),
    )
    if viewport_window is None:
        raise RuntimeError("Failed to create the RealSense RGB Viewport window")

    # Give Kit several frames to finish creating the floating window and render
    # the first camera image before physics playback starts.
    update_app(20)
    print("\n========== RealSense Viewport created ==========")
    print(f"window name = {REALSENSE_VIEWPORT_NAME}")
    print(f"camera path = {camera_path}")
    print(
        f"window size = {REALSENSE_VIEWPORT_WIDTH} x "
        f"{REALSENSE_VIEWPORT_HEIGHT}"
    )
    print("The original main Viewport remains the normal Perspective view.")
    return viewport_window


# =============================================================================
# URDF import and robot setup
# =============================================================================


def expected_generated_usd_path(urdf_path: Path) -> Path:
    robot_name = urdf_path.stem
    return USD_OUTPUT_ROOT / robot_name / f"{robot_name}.usda"


def create_urdf_import_config(urdf_path: Path) -> URDFImporterConfig:
    return URDFImporterConfig(
        urdf_path=str(urdf_path),
        usd_path=str(USD_OUTPUT_ROOT),
        merge_fixed_joints=False,
        merge_mesh=False,
        debug_mode=False,
        collision_from_visuals=False,
        allow_self_collision=False,
        fix_base=True,
        link_density=None,
        run_asset_transformer=True,
        run_multi_physics_conversion=True,
    )


def generate_or_reuse_robot_usd(urdf_path: Path) -> Path:
    urdf_path = urdf_path.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    USD_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cached_usd = expected_generated_usd_path(urdf_path)
    package_dir = cached_usd.parent

    if FORCE_URDF_REIMPORT and package_dir.exists():
        print(f"Removing old generated USD package: {package_dir}")
        shutil.rmtree(package_dir)

    if cached_usd.is_file():
        print(f"Reusing generated robot USD: {cached_usd}")
        return cached_usd

    if package_dir.exists():
        print(f"Removing incomplete generated USD package: {package_dir}")
        shutil.rmtree(package_dir)

    print(f"Importing URDF: {urdf_path}")
    importer = URDFImporter(create_urdf_import_config(urdf_path))
    generated_path = Path(importer.import_urdf()).expanduser().resolve()
    if not generated_path.is_file():
        raise RuntimeError(
            "URDF importer returned a path that does not exist: "
            f"{generated_path}"
        )
    print(f"Generated robot USD: {generated_path}")
    return generated_path


def get_joint_external_name(prim) -> str:
    override_attr = prim.GetAttribute("isaac:nameOverride")
    if override_attr and override_attr.HasAuthoredValueOpinion():
        value = override_attr.Get()
        if value:
            return str(value)
    return prim.GetName()


def find_descendant_by_name(stage, root_path: str, target_name: str):
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return None

    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == target_name or get_joint_external_name(prim) == target_name:
            return prim
    return None


def print_prim_world_position(stage, prim, label: str) -> Gf.Vec3d:
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    position = xform_cache.GetLocalToWorldTransform(prim).ExtractTranslation()
    print(
        f"{label} world position = "
        f"({position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}) m"
    )
    return position


def add_robot_reference(robot_usd_path: Path) -> str:
    stage_utils.add_reference_to_stage(
        usd_path=str(robot_usd_path),
        path=ROBOT_REFERENCE_PATH,
    )
    update_app(20)

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(ROBOT_REFERENCE_PATH)
    if not prim.IsValid():
        raise RuntimeError(f"Robot reference was not created at {ROBOT_REFERENCE_PATH}")

    xform_api = UsdGeom.XformCommonAPI(prim)
    if not xform_api.SetTranslate(Gf.Vec3d(*ROBOT_REFERENCE_POSITION)):
        raise RuntimeError(
            f"Failed to set robot reference position: {ROBOT_REFERENCE_POSITION}"
        )

    update_app(5)
    print(f"Robot USD referenced at: {ROBOT_REFERENCE_PATH}")
    print_prim_world_position(stage, prim, "robot reference")

    base_link_prim = find_descendant_by_name(stage, ROBOT_REFERENCE_PATH, "base_link")
    if base_link_prim is None:
        print("WARNING: base_link prim was not found below the robot reference.")
    else:
        print(f"base_link prim path     = {base_link_prim.GetPath()}")
        print_prim_world_position(stage, base_link_prim, "base_link")

    return ROBOT_REFERENCE_PATH


def find_articulation_root_path(stage, search_root_path: str) -> str:
    search_root = stage.GetPrimAtPath(search_root_path)
    if not search_root.IsValid():
        raise RuntimeError(f"Invalid robot reference path: {search_root_path}")

    candidates: list[str] = []
    for prim in Usd.PrimRange(search_root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            candidates.append(str(prim.GetPath()))

    if not candidates:
        raise RuntimeError(
            "No UsdPhysics.ArticulationRootAPI prim was found below "
            f"{search_root_path}"
        )

    candidates.sort(key=lambda path: (path.count("/"), path))
    articulation_root_path = candidates[0]
    print(f"Articulation root path: {articulation_root_path}")
    return articulation_root_path


def iter_physics_joints(root_prim) -> Iterable:
    supported = {
        "PhysicsRevoluteJoint",
        "PhysicsPrismaticJoint",
        "PhysicsFixedJoint",
        "PhysicsSphericalJoint",
    }
    for prim in Usd.PrimRange(root_prim):
        if prim.GetTypeName() in supported:
            yield prim


def set_usd_joint_drives(
    stage,
    robot_reference_path: str,
    joint_names: Iterable[str],
    stiffness: float = JOINT_DRIVE_STIFFNESS,
    damping: float = JOINT_DRIVE_DAMPING,
    max_force: float = JOINT_DRIVE_MAX_FORCE,
) -> None:
    robot_prim = stage.GetPrimAtPath(robot_reference_path)
    if not robot_prim.IsValid():
        raise RuntimeError(f"Invalid robot path: {robot_reference_path}")

    requested_names = set(joint_names)
    found_names: set[str] = set()
    print("\n========== Set USD Joint Drives ==========")

    for prim in iter_physics_joints(robot_prim):
        if prim.GetTypeName() != "PhysicsRevoluteJoint":
            continue

        external_name = get_joint_external_name(prim)
        prim_name = prim.GetName()
        if external_name not in requested_names and prim_name not in requested_names:
            continue

        matched_name = external_name if external_name in requested_names else prim_name
        found_names.add(matched_name)

        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(float(stiffness))
        drive.CreateDampingAttr(float(damping))
        drive.CreateMaxForceAttr(float(max_force))
        print(
            f"{matched_name:20s} path={prim.GetPath()} "
            f"stiffness={stiffness} damping={damping} maxForce={max_force}"
        )

    missing_names = sorted(requested_names - found_names)
    if missing_names:
        print("WARNING: The following driven joints were not found:")
        for name in missing_names:
            print(f"  {name}")
    if not found_names:
        raise RuntimeError(
            "None of the requested driven joints were found in the imported USD"
        )


# =============================================================================
# ROS 2 Action Graph: continuous joint command and /clock
# =============================================================================


def create_ros2_action_graph(articulation_root_path: str) -> None:
    og.Controller.edit(
        {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                (
                    "SubscribeJointState",
                    "isaacsim.ros2.bridge.ROS2SubscribeJointState",
                ),
                (
                    "JointNameResolver",
                    "isaacsim.core.nodes.IsaacJointNameResolver",
                ),
                (
                    "ArticulationController",
                    "isaacsim.core.nodes.IsaacArticulationController",
                ),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            og.Controller.Keys.CONNECT: [
                (
                    "OnPlaybackTick.outputs:tick",
                    "SubscribeJointState.inputs:execIn",
                ),
                (
                    "SubscribeJointState.outputs:execOut",
                    "JointNameResolver.inputs:execIn",
                ),
                (
                    "JointNameResolver.outputs:execOut",
                    "ArticulationController.inputs:execIn",
                ),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                (
                    "SubscribeJointState.outputs:jointNames",
                    "JointNameResolver.inputs:jointNames",
                ),
                (
                    "JointNameResolver.outputs:jointNames",
                    "ArticulationController.inputs:jointNames",
                ),
                (
                    "JointNameResolver.outputs:robotPath",
                    "ArticulationController.inputs:robotPath",
                ),
                (
                    "SubscribeJointState.outputs:positionCommand",
                    "ArticulationController.inputs:positionCommand",
                ),
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                (
                    "ReadSimTime.outputs:simulationTime",
                    "PublishClock.inputs:timeStamp",
                ),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("Context.inputs:domain_id", ROS_DOMAIN_ID),
                ("Context.inputs:useDomainIDEnvVar", False),
                ("SubscribeJointState.inputs:topicName", JOINT_COMMAND_TOPIC),
                (
                    "JointNameResolver.inputs:robotPath",
                    articulation_root_path,
                ),
                ("PublishClock.inputs:topicName", "/clock"),
            ],
        },
    )

    print("\n========== ROS 2 Action Graph created ==========")
    print(f"graph_path             = {GRAPH_PATH}")
    print(f"articulation_root_path = {articulation_root_path}")
    print(f"topic_name             = {JOINT_COMMAND_TOPIC}")
    print(f"domain_id              = {ROS_DOMAIN_ID}")


# =============================================================================
# Episode recording
# =============================================================================


@dataclass
class EpisodeResult:
    episode_id: int
    success: bool
    truncated: bool
    reason: str
    recorded_steps: int
    dataset_path: str


class FfmpegVideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "--record-video was specified, but ffmpeg was not found in PATH"
            )
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.9g}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.frame_count = 0

    def write(self, frame_rgb: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        if self.process.poll() is not None:
            stderr = b""
            if self.process.stderr is not None:
                stderr = self.process.stderr.read()
            raise RuntimeError(
                "ffmpeg exited before video recording completed: "
                + stderr.decode("utf-8", errors="replace")
            )
        frame = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        expected_shape = (self.height, self.width, 3)
        if frame.shape != expected_shape:
            raise ValueError(
                f"Expected RGB frame shape {expected_shape}, received {frame.shape}"
            )
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            stderr = b""
            if self.process.stderr is not None:
                stderr = self.process.stderr.read()
            raise RuntimeError(
                "ffmpeg pipe closed while writing video: "
                + stderr.decode("utf-8", errors="replace")
            ) from exc
        self.frame_count += 1

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        stderr = b""
        if self.process.stderr is not None:
            stderr = self.process.stderr.read()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {return_code}: "
                + stderr.decode("utf-8", errors="replace")
            )


class EpisodeRecorder:
    def __init__(
        self,
        data_root: Path,
        *,
        video_requested: bool,
        video_width: int,
        video_height: int,
    ) -> None:
        self.data_root = data_root.expanduser().resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.video_requested = bool(video_requested)
        self.video_width = int(video_width)
        self.video_height = int(video_height)
        self.active = False
        self.record_enabled = False
        self.video_enabled = False
        self.video_writer: FfmpegVideoWriter | None = None
        self.last_video_frame: np.ndarray | None = None
        self.episode_id = -1
        self.episode_dir: Path | None = None
        self.metadata: dict = {}
        self.timestamps: list[float] = []
        self.joint_positions: list[np.ndarray] = []
        self.cube_positions: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.gripper_positions: list[np.ndarray] = []
        self.phases: list[int] = []
        self.video_frame_valid: list[bool] = []
        self.instruction_history: list[dict] = []

    def start(
        self,
        *,
        episode_id: int,
        seed: int,
        cube_positions: dict[str, np.ndarray],
        target_layout: list[dict],
        task_instruction: str,
        target_cube_color: str | None,
        sim_time: float,
        record_enabled: bool,
        randomization_bounds: dict,
    ) -> None:
        self.active = True
        self.record_enabled = bool(record_enabled)
        self.video_enabled = self.record_enabled and self.video_requested
        self.episode_id = int(episode_id)
        self.episode_dir = self.data_root / f"episode_{episode_id:06d}"
        self.timestamps.clear()
        self.joint_positions.clear()
        self.cube_positions.clear()
        self.actions.clear()
        self.gripper_positions.clear()
        self.phases.clear()
        self.video_frame_valid.clear()
        self.instruction_history.clear()
        self.last_video_frame = None
        self.video_writer = None

        initial_cubes = {
            color: position.astype(float).tolist()
            for color, position in cube_positions.items()
        }
        self.metadata = {
            "episode_index": int(episode_id),
            "seed": int(seed),
            "record_enabled": self.record_enabled,
            "status": "in_progress",
            "start_simulation_time": float(sim_time),
            "cube_colors": list(CUBE_COLOR_NAMES),
            "initial_cube_positions": initial_cubes,
            "target_markers": target_layout,
            "task_instruction": str(task_instruction),
            "target_cube_color": target_cube_color or "",
            "joint_names": list(DRIVEN_JOINT_NAMES),
            "sample_rate_hz": RECORD_RATE_HZ,
            "randomization_bounds": randomization_bounds,
            "video": {
                "requested_at_startup": self.video_requested,
                "enabled_for_episode": self.video_enabled,
                "filename": VIDEO_FILENAME if self.video_enabled else "",
                "width": self.video_width,
                "height": self.video_height,
                "nominal_fps": RECORD_RATE_HZ,
                "synchronization": (
                    "Video frame i corresponds exactly to timestamps[i], "
                    "joint_position[i], cube_position[i], action[i], and phase[i]."
                ),
            },
            "success_definition": {
                "minimum_lift_m": SUCCESS_LIFT_HEIGHT_M,
                "maximum_gripper_distance_m": SUCCESS_GRIPPER_DISTANCE_M,
                "hold_time_sec": SUCCESS_HOLD_SEC,
                "applies_to": target_cube_color or "any cube (manual unrecorded mode)",
            },
        }

        if task_instruction:
            self.instruction_history.append(
                {"simulation_time": float(sim_time), "text": str(task_instruction)}
            )

        if self.record_enabled:
            self.episode_dir.mkdir(parents=True, exist_ok=False)
            if self.video_enabled:
                self.video_writer = FfmpegVideoWriter(
                    self.episode_dir / VIDEO_FILENAME,
                    self.video_width,
                    self.video_height,
                    RECORD_RATE_HZ,
                )
            self._write_metadata()
            self._write_task_instruction()

    def set_task_instruction(
        self,
        *,
        timestamp: float,
        text: str,
        target_cube_color: str | None,
    ) -> None:
        """Record a mid-episode instruction event without changing the task label."""
        text = str(text)
        self.metadata["latest_received_task_instruction"] = text
        self.metadata["latest_received_target_cube_color"] = (
            target_cube_color or ""
        )
        self.instruction_history.append(
            {
                "simulation_time": float(timestamp),
                "text": text,
                "target_cube_color": target_cube_color or "",
            }
        )
        if self.record_enabled:
            self._write_metadata()

    def append(
        self,
        *,
        timestamp: float,
        joint_position: np.ndarray,
        cube_position: np.ndarray,
        action: np.ndarray,
        gripper_position: np.ndarray,
        phase: int,
        rgb_frame: np.ndarray | None,
    ) -> None:
        if not self.active or not self.record_enabled:
            return
        self.timestamps.append(float(timestamp))
        self.joint_positions.append(joint_position.astype(np.float64, copy=True))
        self.cube_positions.append(cube_position.astype(np.float64, copy=True))
        self.actions.append(action.astype(np.float64, copy=True))
        self.gripper_positions.append(gripper_position.astype(np.float64, copy=True))
        self.phases.append(int(phase))

        if self.video_enabled:
            valid = rgb_frame is not None
            if valid:
                frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
                self.last_video_frame = frame.copy()
            elif self.last_video_frame is not None:
                frame = self.last_video_frame
            else:
                frame = np.zeros(
                    (self.video_height, self.video_width, 3), dtype=np.uint8
                )
            assert self.video_writer is not None
            self.video_writer.write(frame)
            self.video_frame_valid.append(bool(valid))

    def finalize(
        self,
        *,
        success: bool,
        truncated: bool,
        reason: str,
        sim_time: float,
        final_cube_positions: dict[str, np.ndarray],
        maximum_cube_heights: dict[str, float],
    ) -> EpisodeResult:
        if not self.active:
            raise RuntimeError("No episode is currently active")

        dataset_path = ""
        recorded_steps = len(self.timestamps)
        video_close_error = ""
        if self.video_writer is not None:
            try:
                self.video_writer.close()
            except Exception as exc:
                video_close_error = str(exc)
            finally:
                self.video_writer = None

        if self.record_enabled:
            assert self.episode_dir is not None
            np.save(
                self.episode_dir / "timestamps.npy",
                np.asarray(self.timestamps, dtype=np.float64),
            )
            np.save(
                self.episode_dir / "joint_position.npy",
                np.asarray(self.joint_positions, dtype=np.float64).reshape(
                    recorded_steps, len(DRIVEN_JOINT_NAMES)
                ),
            )
            np.save(
                self.episode_dir / "cube_position.npy",
                np.asarray(self.cube_positions, dtype=np.float64).reshape(
                    recorded_steps, len(CUBE_COLOR_NAMES), 3
                ),
            )
            np.save(
                self.episode_dir / "action.npy",
                np.asarray(self.actions, dtype=np.float64).reshape(
                    recorded_steps, len(DRIVEN_JOINT_NAMES)
                ),
            )
            np.save(
                self.episode_dir / "gripper_position.npy",
                np.asarray(self.gripper_positions, dtype=np.float64).reshape(
                    recorded_steps, 3
                ),
            )
            np.save(
                self.episode_dir / "phase.npy",
                np.asarray(self.phases, dtype=np.uint8),
            )
            if self.video_enabled:
                np.save(
                    self.episode_dir / VIDEO_FRAME_TIMESTAMPS_FILENAME,
                    np.asarray(self.timestamps, dtype=np.float64),
                )
                np.save(
                    self.episode_dir / VIDEO_FRAME_VALID_FILENAME,
                    np.asarray(self.video_frame_valid, dtype=np.bool_),
                )
                np.save(
                    self.episode_dir / VIDEO_SAMPLE_INDICES_FILENAME,
                    np.arange(recorded_steps, dtype=np.int64),
                )
            (self.episode_dir / "instruction_history.json").write_text(
                json.dumps(self.instruction_history, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            dataset_path = str(self.episode_dir)

        self.metadata.update(
            {
                "status": "finished",
                "success": bool(success),
                "failure": not bool(success),
                "result_label": "success" if success else "failure",
                "successful_cube_color": (
                    str(self.metadata.get("target_cube_color", "")) if success else ""
                ),
                "terminated": True,
                "truncated": bool(truncated),
                "termination_reason": str(reason),
                "end_simulation_time": float(sim_time),
                "recorded_steps": int(recorded_steps),
                "final_cube_positions": {
                    color: position.astype(float).tolist()
                    for color, position in final_cube_positions.items()
                },
                "maximum_cube_heights": {
                    color: float(height)
                    for color, height in maximum_cube_heights.items()
                },
                "instruction_history": list(self.instruction_history),
            }
        )
        self.metadata["video"]["frame_count"] = (
            recorded_steps if self.video_enabled else 0
        )
        self.metadata["video"]["valid_frame_count"] = int(
            np.count_nonzero(self.video_frame_valid)
        )
        if video_close_error:
            self.metadata["video"]["encoder_error"] = video_close_error
        if self.record_enabled:
            self._write_metadata()

        result = EpisodeResult(
            episode_id=self.episode_id,
            success=bool(success),
            truncated=bool(truncated),
            reason=str(reason),
            recorded_steps=recorded_steps,
            dataset_path=dataset_path,
        )
        self.active = False
        if video_close_error:
            raise RuntimeError(video_close_error)
        return result

    def _write_task_instruction(self) -> None:
        assert self.episode_dir is not None
        (self.episode_dir / "task_instruction.txt").write_text(
            str(self.metadata.get("task_instruction", "")) + "\n",
            encoding="utf-8",
        )

    def _write_metadata(self) -> None:
        assert self.episode_dir is not None
        temporary = self.episode_dir / "metadata.json.tmp"
        final = self.episode_dir / "metadata.json"
        temporary.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(final)


# =============================================================================
# Imitation-learning environment and ROS 2 API
# =============================================================================


class DofbotImitationEnvironment:
    def __init__(
        self,
        *,
        stage,
        articulation_root_path: str,
        gripper_point_prim_path: str,
        timeline,
        data_root: Path,
        cube_x: float,
        cube_y_bounds: tuple[float, float],
        cube_clearance: float,
        master_seed: int | None,
        camera_sensor: CameraSensor | None,
    ) -> None:
        self.stage = stage
        self.timeline = timeline
        self.cube_x = float(cube_x)
        self.cube_y_bounds = cube_y_bounds
        self.cube_clearance = float(cube_clearance)
        self.master_rng = np.random.default_rng(master_seed)
        self.camera_sensor = camera_sensor
        self.recorder = EpisodeRecorder(
            data_root,
            video_requested=ARGS.record_video,
            video_width=ARGS.video_width,
            video_height=ARGS.video_height,
        )
        self.next_episode_id = next_episode_index(self.recorder.data_root)

        self.robot = Articulation(
            prim_paths_expr=articulation_root_path,
            name="dofbot_imitation_articulation",
            reset_xform_properties=False,
        )
        self.cubes: dict[str, RigidPrim] = {
            color: RigidPrim(
                prim_paths_expr=CUBE_PATHS[color],
                name=f"dofbot_{color}_cube",
                reset_xform_properties=False,
            )
            for color in CUBE_COLOR_NAMES
        }
        self.robot.initialize()
        for cube in self.cubes.values():
            cube.initialize()

        self.joint_indices = np.asarray(
            [self.robot.get_dof_index(name) for name in DRIVEN_JOINT_NAMES],
            dtype=np.int64,
        )
        self.gripper_point_prim = stage.GetPrimAtPath(gripper_point_prim_path)
        if not self.gripper_point_prim.IsValid():
            raise RuntimeError(
                f"Invalid gripper point prim path: {gripper_point_prim_path}"
            )
        self.gripper_point = XFormPrim(
            prim_paths_expr=gripper_point_prim_path,
            name="dofbot_gripper_point",
            reset_xform_properties=False,
        )

        self.last_action = HOME_JOINT_POSITION.copy()
        self.phase = PHASE_IDLE
        self.episode_id = 0
        self.seed_used = -1
        self.initial_cube_positions = self.get_cube_positions_dict()
        self.maximum_cube_heights = {
            color: float(position[2])
            for color, position in self.initial_cube_positions.items()
        }
        self.target_layout = self.current_target_layout()
        self.current_task_instruction = ""
        self.current_target_cube_color: str | None = None
        self.episode_task_instruction = ""
        self.episode_target_cube_color: str | None = None
        self.success = False
        self.terminated = False
        self.truncated = False
        self.termination_reason = ""
        self.success_condition_since: float | None = None
        self.ready_after_sim_time = 0.0
        self.episode_start_sim_time = 0.0
        self.last_record_sim_time = -math.inf
        self.last_camera_publish_sim_time = -math.inf
        self.last_result: EpisodeResult | None = None

        self.node = Node("dofbot_smolvla_environment")
        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.joint_state_publisher = self.node.create_publisher(
            JointState,
            JOINT_STATE_TOPIC,
            10,
        )
        self.episode_state_publisher = self.node.create_publisher(
            EpisodeState,
            EPISODE_STATE_TOPIC,
            state_qos,
        )
        self.scene_layout_publisher = self.node.create_publisher(
            String,
            SCENE_LAYOUT_TOPIC,
            state_qos,
        )
        camera_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.camera_publisher = (
            self.node.create_publisher(Image, CAMERA_RGB_TOPIC, camera_qos)
            if ARGS.publish_camera
            else None
        )
        self.command_subscription = self.node.create_subscription(
            JointState,
            JOINT_COMMAND_TOPIC,
            self.on_joint_command,
            10,
        )
        self.task_instruction_subscription = self.node.create_subscription(
            String,
            TASK_INSTRUCTION_TOPIC,
            self.on_task_instruction,
            state_qos,
        )
        self.reset_service = self.node.create_service(
            ResetEpisode,
            RESET_SERVICE,
            self.on_reset_episode,
        )
        self.end_service = self.node.create_service(
            EndEpisode,
            END_SERVICE,
            self.on_end_episode,
        )
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

        print("\n========== VLA environment API ==========")
        print(f"reset service      = {RESET_SERVICE}")
        print(f"end service        = {END_SERVICE}")
        print(f"state topic        = {EPISODE_STATE_TOPIC}")
        print(f"scene layout topic = {SCENE_LAYOUT_TOPIC} (std_msgs/String JSON)")
        print(f"task text topic    = {TASK_INSTRUCTION_TOPIC} (std_msgs/String)")
        print(f"joint state        = {JOINT_STATE_TOPIC}")
        print(
            f"camera RGB topic   = {CAMERA_RGB_TOPIC} "
            f"({'enabled' if self.camera_publisher is not None else 'disabled'})"
        )
        print(f"data root          = {self.recorder.data_root}")
        print(f"cube x             = {self.cube_x:.3f} m")
        print(
            f"cube y range       = [{cube_y_bounds[0]:.3f}, "
            f"{cube_y_bounds[1]:.3f}] m"
        )
        print(f"cube face clearance= {self.cube_clearance:.3f} m")
        print(f"video requested    = {ARGS.record_video}")

    def shutdown(self) -> None:
        if self.recorder.active:
            try:
                self.finalize_episode(
                    success=False,
                    truncated=True,
                    reason="simulator_shutdown",
                )
            except Exception as exc:
                print(f"WARNING: failed to finalize episode during shutdown: {exc}")
        self.executor.remove_node(self.node)
        self.node.destroy_node()
        self.executor.shutdown()

    def spin_ros_once(self) -> None:
        self.executor.spin_once(timeout_sec=0.0)

    def on_joint_command(self, message: JointState) -> None:
        if not message.position:
            return

        updated = self.last_action.copy()
        if message.name:
            if len(message.name) != len(message.position):
                self.node.get_logger().warning(
                    "Ignored /joint_command: name and position lengths differ"
                )
                return
            value_by_name = dict(zip(message.name, message.position, strict=True))
            for index, name in enumerate(DRIVEN_JOINT_NAMES):
                if name in value_by_name:
                    updated[index] = float(value_by_name[name])
        elif len(message.position) == len(DRIVEN_JOINT_NAMES):
            updated[:] = np.asarray(message.position, dtype=np.float64)
        else:
            self.node.get_logger().warning(
                "Ignored unnamed /joint_command whose position length is not 11"
            )
            return

        self.last_action = updated
        if self.recorder.active and self.phase == PHASE_READY:
            self.phase = PHASE_RUNNING

    def on_task_instruction(self, message: String) -> None:
        text = str(message.data).strip()
        target_color = extract_target_cube_color(text)
        self.current_task_instruction = text
        self.current_target_cube_color = target_color
        sim_time = float(self.timeline.get_current_time())
        if self.recorder.active:
            self.recorder.set_task_instruction(
                timestamp=sim_time,
                text=text,
                target_cube_color=target_color,
            )
            if target_color != self.episode_target_cube_color:
                self.node.get_logger().warning(
                    "Task text changed during an active episode. The success "
                    f"target remains {self.episode_target_cube_color!r}."
                )
        self.publish_scene_layout()
        self.node.get_logger().info(
            f"Task instruction updated: {text!r}, target_color={target_color!r}"
        )

    def on_reset_episode(self, request, response):
        try:
            mode = int(request.mode)
            if mode not in (MODE_RANDOM, MODE_SPECIFIED):
                raise ValueError(f"Unsupported reset mode: {mode}")

            if self.recorder.active:
                self.finalize_episode(
                    success=False,
                    truncated=True,
                    reason="reset_before_previous_episode_finished",
                )

            seed = int(request.seed)
            if seed < 0:
                seed = int(self.master_rng.integers(0, 2**31 - 1))
            rng = np.random.default_rng(seed)

            if bool(request.record) and self.current_target_cube_color is None:
                raise ValueError(
                    "Recorded episodes require a task instruction containing exactly "
                    "one supported color: red, blue, or green."
                )

            forced_red_y: float | None = None
            if mode == MODE_SPECIFIED:
                if not math.isclose(float(request.x), self.cube_x, abs_tol=1.0e-6):
                    raise ValueError(
                        f"All cubes use fixed x={self.cube_x:.4f}; received x={request.x:.4f}"
                    )
                forced_red_y = float(request.y)
                self.validate_cube_y(forced_red_y)

            cube_positions = self.reset_environment(
                rng=rng,
                seed=seed,
                record=bool(request.record),
                forced_red_y=forced_red_y,
            )
            red_position = cube_positions["red"]
            response.accepted = True
            response.episode_id = int(self.episode_id)
            response.seed_used = int(seed)
            # Backward compatibility: the legacy scalar fields report the red cube.
            response.cube_x = float(red_position[0])
            response.cube_y = float(red_position[1])
            response.cube_z = float(red_position[2])
            response.message = (
                "Reset accepted. Full three-cube and marker layout is published on "
                f"{SCENE_LAYOUT_TOPIC}. Wait until phase becomes READY."
            )
        except Exception as exc:
            response.accepted = False
            response.episode_id = int(self.episode_id)
            response.seed_used = -1
            response.cube_x = 0.0
            response.cube_y = 0.0
            response.cube_z = 0.0
            response.message = str(exc)
            self.node.get_logger().error(f"Reset failed: {exc}")
        return response

    def on_end_episode(self, request, response):
        try:
            if not self.recorder.active:
                if self.last_result is None:
                    raise RuntimeError("No active or completed episode exists")
                result = self.last_result
            else:
                reason = str(request.reason).strip() or "controller_finished"
                result = self.finalize_episode(
                    success=self.success,
                    truncated=bool(request.truncated),
                    reason=reason,
                )

            response.accepted = True
            response.success = bool(result.success)
            response.episode_id = int(result.episode_id)
            response.recorded_steps = int(result.recorded_steps)
            response.dataset_path = result.dataset_path
            response.message = result.reason
        except Exception as exc:
            response.accepted = False
            response.success = False
            response.episode_id = int(self.episode_id)
            response.recorded_steps = 0
            response.dataset_path = ""
            response.message = str(exc)
            self.node.get_logger().error(f"End episode failed: {exc}")
        return response

    def validate_cube_y(self, y: float) -> None:
        if not (self.cube_y_bounds[0] <= y <= self.cube_y_bounds[1]):
            raise ValueError(
                f"y={y:.4f} is outside "
                f"[{self.cube_y_bounds[0]}, {self.cube_y_bounds[1]}]"
            )

    def sample_cube_y_positions(
        self,
        rng: np.random.Generator,
        *,
        forced_red_y: float | None,
    ) -> dict[str, float]:
        minimum_center_distance = CUBE_SIZE + self.cube_clearance
        y_min, y_max = self.cube_y_bounds
        if y_max - y_min < 2.0 * minimum_center_distance:
            raise ValueError(
                "Cube Y range is too narrow for three non-overlapping cubes: "
                f"range={y_max - y_min:.4f}, required at least "
                f"{2.0 * minimum_center_distance:.4f}"
            )

        for _attempt in range(5000):
            values: dict[str, float] = {}
            if forced_red_y is not None:
                values["red"] = forced_red_y
            for color in CUBE_COLOR_NAMES:
                if color in values:
                    continue
                values[color] = float(rng.uniform(y_min, y_max))
            ys = [values[color] for color in CUBE_COLOR_NAMES]
            if all(
                abs(ys[i] - ys[j]) >= minimum_center_distance
                for i in range(len(ys))
                for j in range(i + 1, len(ys))
            ):
                return values
        raise RuntimeError(
            "Failed to sample three non-overlapping cube positions after 5000 attempts"
        )

    def randomize_target_markers(self, rng: np.random.Generator) -> list[dict]:
        colors = list(TARGET_MARKER_COLOR_NAMES)
        rng.shuffle(colors)
        layout: list[dict] = []
        marker_z = FLOOR_TOP_Z + TARGET_MARKER_Z_OFFSET
        for index, (marker_path, position_xy, color_name) in enumerate(
            zip(TARGET_MARKER_PATHS, TARGET_MARKER_POSITIONS_XY, colors, strict=True)
        ):
            set_marker_color(self.stage, marker_path, color_name)
            layout.append(
                {
                    "slot_index": index,
                    "prim_path": marker_path,
                    "color": color_name,
                    "position": [float(position_xy[0]), float(position_xy[1]), float(marker_z)],
                    "radius_m": TARGET_MARKER_RADIUS,
                    "diameter_m": TARGET_MARKER_DIAMETER,
                    "visual_only": True,
                }
            )
        return layout

    def current_target_layout(self) -> list[dict]:
        marker_z = FLOOR_TOP_Z + TARGET_MARKER_Z_OFFSET
        return [
            {
                "slot_index": index,
                "prim_path": marker_path,
                "color": color_name,
                "position": [float(x), float(y), float(marker_z)],
                "radius_m": TARGET_MARKER_RADIUS,
                "diameter_m": TARGET_MARKER_DIAMETER,
                "visual_only": True,
            }
            for index, (marker_path, (x, y), color_name) in enumerate(
                zip(
                    TARGET_MARKER_PATHS,
                    TARGET_MARKER_POSITIONS_XY,
                    TARGET_MARKER_COLOR_NAMES,
                    strict=True,
                )
            )
        ]

    def reset_environment(
        self,
        *,
        rng: np.random.Generator,
        seed: int,
        record: bool,
        forced_red_y: float | None,
    ) -> dict[str, np.ndarray]:
        was_playing = self.timeline.is_playing()
        self.timeline.pause()

        home = HOME_JOINT_POSITION.reshape(1, -1)
        zeros = np.zeros_like(home)
        self.robot.set_joint_positions(home, joint_indices=self.joint_indices)
        self.robot.set_joint_velocities(zeros, joint_indices=self.joint_indices)
        self.robot.set_joint_position_targets(home, joint_indices=self.joint_indices)

        sampled_y = self.sample_cube_y_positions(
            rng,
            forced_red_y=forced_red_y,
        )
        cube_z = FLOOR_TOP_Z + CUBE_SIZE / 2.0
        for color in CUBE_COLOR_NAMES:
            position = np.asarray([[self.cube_x, sampled_y[color], cube_z]], dtype=np.float64)
            orientation = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
            self.cubes[color].set_world_poses(
                positions=position,
                orientations=orientation,
            )
            self.cubes[color].set_velocities(np.zeros((1, 6), dtype=np.float64))

        self.target_layout = self.randomize_target_markers(rng)
        self.last_action = HOME_JOINT_POSITION.copy()
        if was_playing or not self.timeline.is_playing():
            self.timeline.play()

        sim_time = float(self.timeline.get_current_time())
        actual_cube_positions = self.get_cube_positions_dict()
        self.episode_id = self.next_episode_id
        self.next_episode_id += 1
        self.seed_used = int(seed)
        self.initial_cube_positions = {
            color: position.copy()
            for color, position in actual_cube_positions.items()
        }
        self.maximum_cube_heights = {
            color: float(position[2])
            for color, position in actual_cube_positions.items()
        }
        self.success = False
        self.terminated = False
        self.truncated = False
        self.termination_reason = ""
        self.success_condition_since = None
        self.phase = PHASE_RESETTING
        self.ready_after_sim_time = sim_time + RESET_SETTLE_SEC
        self.episode_start_sim_time = sim_time
        self.last_record_sim_time = -math.inf
        self.last_result = None
        self.episode_task_instruction = self.current_task_instruction
        self.episode_target_cube_color = self.current_target_cube_color

        self.recorder.start(
            episode_id=self.episode_id,
            seed=seed,
            cube_positions=actual_cube_positions,
            target_layout=self.target_layout,
            task_instruction=self.episode_task_instruction,
            target_cube_color=self.episode_target_cube_color,
            sim_time=sim_time,
            record_enabled=record,
            randomization_bounds={
                "cube_x": self.cube_x,
                "cube_y_min": self.cube_y_bounds[0],
                "cube_y_max": self.cube_y_bounds[1],
                "cube_size_m": CUBE_SIZE,
                "cube_face_clearance_m": self.cube_clearance,
            },
        )
        self.publish_scene_layout()

        cube_summary = ", ".join(
            f"{color}=({position[0]:.3f},{position[1]:.3f})"
            for color, position in actual_cube_positions.items()
        )
        marker_summary = ", ".join(
            f"{item['color']}@y={item['position'][1]:.3f}"
            for item in self.target_layout
        )
        self.node.get_logger().info(
            f"Episode {self.episode_id} reset: cubes[{cube_summary}], "
            f"markers[{marker_summary}], target={self.episode_target_cube_color}, "
            f"seed={seed}, record={record}, video={self.recorder.video_enabled}"
        )
        return actual_cube_positions

    def tick(self) -> None:
        sim_time = float(self.timeline.get_current_time())

        if self.recorder.active and self.phase == PHASE_RESETTING:
            if sim_time >= self.ready_after_sim_time:
                self.phase = PHASE_READY
                self.episode_start_sim_time = sim_time
                self.last_record_sim_time = -math.inf
                self.initial_cube_positions = self.get_cube_positions_dict()
                self.maximum_cube_heights = {
                    color: float(position[2])
                    for color, position in self.initial_cube_positions.items()
                }
                self.recorder.metadata["settled_initial_cube_positions"] = {
                    color: position.astype(float).tolist()
                    for color, position in self.initial_cube_positions.items()
                }
                if self.recorder.record_enabled:
                    self.recorder._write_metadata()
                self.publish_scene_layout()
                self.node.get_logger().info(
                    f"Episode {self.episode_id} is READY"
                )

        actual_joint_position = self.get_joint_position()
        cube_positions = self.get_cube_positions_array()
        gripper_position = self.get_gripper_position()
        for index, color in enumerate(CUBE_COLOR_NAMES):
            self.maximum_cube_heights[color] = max(
                self.maximum_cube_heights[color],
                float(cube_positions[index, 2]),
            )

        if self.recorder.active and self.phase in (PHASE_READY, PHASE_RUNNING):
            self.update_success_and_termination(
                sim_time=sim_time,
                cube_positions=cube_positions,
                gripper_position=gripper_position,
            )

        should_record_sample = (
            self.recorder.active
            and self.phase != PHASE_RESETTING
            and sim_time - self.last_record_sim_time >= RECORD_PERIOD_SEC - 1e-9
        )
        should_publish_camera = (
            self.camera_publisher is not None
            and sim_time - self.last_camera_publish_sim_time
            >= RECORD_PERIOD_SEC - 1e-9
        )

        rgb_frame = None
        if (should_record_sample and self.recorder.video_enabled) or should_publish_camera:
            rgb_frame = self.capture_rgb_frame()

        if should_record_sample:
            self.recorder.append(
                timestamp=sim_time,
                joint_position=actual_joint_position,
                cube_position=cube_positions,
                action=self.last_action,
                gripper_position=gripper_position,
                phase=self.phase,
                rgb_frame=rgb_frame if self.recorder.video_enabled else None,
            )
            self.last_record_sim_time = sim_time

        if should_publish_camera:
            if rgb_frame is not None:
                self.publish_camera_image(sim_time, rgb_frame)
            self.last_camera_publish_sim_time = sim_time

        self.publish_joint_state(sim_time, actual_joint_position)
        self.publish_episode_state(sim_time, cube_positions, gripper_position)

    def update_success_and_termination(
        self,
        *,
        sim_time: float,
        cube_positions: np.ndarray,
        gripper_position: np.ndarray,
    ) -> None:
        monitored_colors = (
            (self.episode_target_cube_color,)
            if self.episode_target_cube_color in CUBE_COLOR_NAMES
            else CUBE_COLOR_NAMES
        )
        target_satisfied = False
        successful_color: str | None = None
        for color in monitored_colors:
            index = CUBE_COLOR_NAMES.index(color)
            lift_height = float(
                cube_positions[index, 2] - self.initial_cube_positions[color][2]
            )
            gripper_distance = float(
                np.linalg.norm(cube_positions[index] - gripper_position)
            )
            if (
                lift_height >= SUCCESS_LIFT_HEIGHT_M
                and gripper_distance <= SUCCESS_GRIPPER_DISTANCE_M
            ):
                target_satisfied = True
                successful_color = color
                break

        if target_satisfied:
            if self.success_condition_since is None:
                self.success_condition_since = sim_time
            elif sim_time - self.success_condition_since >= SUCCESS_HOLD_SEC:
                self.success = True
                self.phase = PHASE_SUCCESS
                self.finalize_episode(
                    success=True,
                    truncated=False,
                    reason=f"target_{successful_color}_cube_lifted_and_held",
                )
                return
        else:
            self.success_condition_since = None

        if np.any(cube_positions[:, 2] < CUBE_FALL_Z_M):
            self.finalize_episode(
                success=False,
                truncated=False,
                reason="a_cube_fell_below_floor",
            )
            return

        if sim_time - self.episode_start_sim_time >= EPISODE_TIMEOUT_SEC:
            self.finalize_episode(
                success=False,
                truncated=True,
                reason="episode_timeout",
            )

    def finalize_episode(
        self,
        *,
        success: bool,
        truncated: bool,
        reason: str,
    ) -> EpisodeResult:
        sim_time = float(self.timeline.get_current_time())
        cube_positions = self.get_cube_positions_array()
        final_phase = PHASE_SUCCESS if success else PHASE_TERMINATED
        rgb_frame = self.capture_rgb_frame() if self.recorder.video_enabled else None
        self.recorder.append(
            timestamp=sim_time,
            joint_position=self.get_joint_position(),
            cube_position=cube_positions,
            action=self.last_action,
            gripper_position=self.get_gripper_position(),
            phase=final_phase,
            rgb_frame=rgb_frame,
        )
        final_cube_positions = {
            color: cube_positions[index].copy()
            for index, color in enumerate(CUBE_COLOR_NAMES)
        }
        result = self.recorder.finalize(
            success=success,
            truncated=truncated,
            reason=reason,
            sim_time=sim_time,
            final_cube_positions=final_cube_positions,
            maximum_cube_heights=self.maximum_cube_heights,
        )
        self.success = bool(success)
        self.terminated = True
        self.truncated = bool(truncated)
        self.termination_reason = str(reason)
        self.phase = PHASE_SUCCESS if success else PHASE_TERMINATED
        self.last_result = result
        self.node.get_logger().info(
            f"Episode {result.episode_id} finished: success={result.success}, "
            f"truncated={result.truncated}, steps={result.recorded_steps}, "
            f"reason={result.reason}"
        )
        return result

    def capture_rgb_frame(self) -> np.ndarray | None:
        if self.camera_sensor is None:
            return None
        data, _info = self.camera_sensor.get_data("rgb")
        if data is None:
            return None
        array = as_numpy(data)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            self.node.get_logger().warning(
                f"Unexpected RGB camera shape: {array.shape}"
            )
            return None
        array = array[:, :, :3]
        if np.issubdtype(array.dtype, np.floating):
            maximum = float(np.nanmax(array)) if array.size else 0.0
            if maximum <= 1.0 + 1.0e-6:
                array = array * 255.0
            array = np.clip(array, 0.0, 255.0)
        frame = np.ascontiguousarray(array, dtype=np.uint8)
        expected_shape = (ARGS.video_height, ARGS.video_width, 3)
        if frame.shape != expected_shape:
            self.node.get_logger().warning(
                f"Camera frame shape {frame.shape} does not match {expected_shape}"
            )
            return None
        return frame

    def publish_camera_image(self, sim_time: float, frame: np.ndarray) -> None:
        if self.camera_publisher is None:
            return
        message = Image()
        message.header.stamp = simulation_time_message(sim_time)
        message.header.frame_id = CAMERA_FRAME_ID
        message.height = int(frame.shape[0])
        message.width = int(frame.shape[1])
        message.encoding = "rgb8"
        message.is_bigendian = 0
        message.step = int(frame.shape[1] * 3)
        message.data = frame.tobytes()
        self.camera_publisher.publish(message)

    def get_joint_position(self) -> np.ndarray:
        value = self.robot.get_joint_positions(joint_indices=self.joint_indices)
        array = as_numpy(value).reshape(-1)
        if array.size != len(DRIVEN_JOINT_NAMES):
            raise RuntimeError(
                f"Expected 11 joint positions, received shape {as_numpy(value).shape}"
            )
        return array.astype(np.float64, copy=False)

    def get_cube_positions_dict(self) -> dict[str, np.ndarray]:
        return {
            color: self.get_single_cube_position(color)
            for color in CUBE_COLOR_NAMES
        }

    def get_cube_positions_array(self) -> np.ndarray:
        return np.stack(
            [self.get_single_cube_position(color) for color in CUBE_COLOR_NAMES],
            axis=0,
        )

    def get_single_cube_position(self, color: str) -> np.ndarray:
        positions, _orientations = self.cubes[color].get_world_poses(usd=False)
        array = as_numpy(positions).reshape(-1, 3)
        return array[0].astype(np.float64, copy=True)

    def get_gripper_position(self) -> np.ndarray:
        try:
            positions, _orientations = self.gripper_point.get_world_poses(usd=False)
            array = as_numpy(positions).reshape(-1, 3)
            return array[0].astype(np.float64, copy=True)
        except Exception:
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            position = cache.GetLocalToWorldTransform(
                self.gripper_point_prim
            ).ExtractTranslation()
            return np.asarray(
                [position[0], position[1], position[2]],
                dtype=np.float64,
            )

    def publish_joint_state(
        self,
        sim_time: float,
        joint_position: np.ndarray,
    ) -> None:
        message = JointState()
        message.header.stamp = simulation_time_message(sim_time)
        message.name = list(DRIVEN_JOINT_NAMES)
        message.position = joint_position.astype(float).tolist()
        message.velocity = []
        message.effort = []
        self.joint_state_publisher.publish(message)

    def publish_episode_state(
        self,
        sim_time: float,
        cube_positions: np.ndarray,
        gripper_position: np.ndarray,
    ) -> None:
        # Preserve the existing message definition. The scalar cube fields report
        # the active target cube; if no target is set, they fall back to red.
        state_color = (
            self.episode_target_cube_color
            if self.episode_target_cube_color in CUBE_COLOR_NAMES
            else "red"
        )
        state_index = CUBE_COLOR_NAMES.index(state_color)
        state_position = cube_positions[state_index]
        message = EpisodeState()
        message.header.stamp = simulation_time_message(sim_time)
        message.episode_id = int(self.episode_id)
        message.phase = int(self.phase)
        message.seed = int(self.seed_used)
        message.cube_x = float(state_position[0])
        message.cube_y = float(state_position[1])
        message.cube_z = float(state_position[2])
        message.lift_height = float(
            state_position[2] - self.initial_cube_positions[state_color][2]
        )
        message.gripper_distance = float(
            np.linalg.norm(state_position - gripper_position)
        )
        message.success = bool(self.success)
        message.terminated = bool(self.terminated)
        message.truncated = bool(self.truncated)
        message.termination_reason = self.termination_reason
        self.episode_state_publisher.publish(message)

    def publish_scene_layout(self) -> None:
        cube_positions = self.get_cube_positions_dict()
        payload = {
            "episode_id": int(self.episode_id),
            "phase": int(self.phase),
            "seed": int(self.seed_used),
            "frame_id": "world",
            "cube_size_m": CUBE_SIZE,
            "cubes": [
                {
                    "color": color,
                    "prim_path": CUBE_PATHS[color],
                    "initial_position": self.initial_cube_positions.get(
                        color, cube_positions[color]
                    ).astype(float).tolist(),
                    "current_position": cube_positions[color].astype(float).tolist(),
                }
                for color in CUBE_COLOR_NAMES
            ],
            "target_markers": self.target_layout,
            "task_instruction": (
                self.episode_task_instruction
                if self.recorder.active
                else self.current_task_instruction
            ),
            "target_cube_color": (
                self.episode_target_cube_color
                if self.recorder.active
                else self.current_target_cube_color
            ) or "",
        }
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.scene_layout_publisher.publish(message)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    if ARGS.cube_y_min >= ARGS.cube_y_max:
        raise ValueError("--cube-y-min must be smaller than --cube-y-max")
    if ARGS.cube_clearance < 0.0:
        raise ValueError("--cube-clearance must be non-negative")
    if ARGS.video_width <= 0 or ARGS.video_height <= 0:
        raise ValueError("Video dimensions must be positive")

    enable_required_extensions()
    update_app(20)

    stage = create_new_stage()
    add_physics_scene(stage)
    add_floor(stage)
    add_pick_cubes(stage)
    add_target_markers(stage)
    add_light(stage)
    realsense_camera_path = add_realsense_camera(stage)

    robot_usd_path = generate_or_reuse_robot_usd(URDF_PATH)
    robot_reference_path = add_robot_reference(robot_usd_path)

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Current USD stage is unavailable")

    articulation_root_path = find_articulation_root_path(
        stage,
        robot_reference_path,
    )
    set_usd_joint_drives(
        stage=stage,
        robot_reference_path=robot_reference_path,
        joint_names=DRIVEN_JOINT_NAMES,
    )

    gripper_prim = find_descendant_by_name(
        stage,
        robot_reference_path,
        GRIPPER_POINT_LINK_NAME,
    )
    if gripper_prim is None:
        raise RuntimeError(
            f"Could not find {GRIPPER_POINT_LINK_NAME} below {robot_reference_path}"
        )
    gripper_point_prim_path = str(gripper_prim.GetPath())
    print(f"Gripper point prim path: {gripper_point_prim_path}")

    create_ros2_action_graph(articulation_root_path)
    update_app(30)

    realsense_viewport_window = create_realsense_viewport(realsense_camera_path)

    camera_sensor: CameraSensor | None = None
    if ARGS.record_video or ARGS.publish_camera:
        camera_sensor = CameraSensor(
            realsense_camera_path,
            resolution=(ARGS.video_height, ARGS.video_width),
            annotators=["rgb"],
        )
        # A few rendered frames are required before RGB data becomes valid.
        update_app(30)
        print("\n========== RealSense RGB sensor ready ==========")
        print(f"resolution = {ARGS.video_width} x {ARGS.video_height}")
        print(f"fps        = {RECORD_RATE_HZ:.1f}")
        print(f"recording  = {ARGS.record_video}")
        print(f"ROS publish= {ARGS.publish_camera}")
        if ARGS.record_video:
            print(f"codec      = H.264 / {VIDEO_FILENAME}")
        if ARGS.publish_camera:
            print(f"topic      = {CAMERA_RGB_TOPIC}")

    if not rclpy.ok():
        rclpy.init(args=[])

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    update_app(10)

    _realsense_viewport_window = realsense_viewport_window
    _camera_sensor = camera_sensor

    environment: DofbotImitationEnvironment | None = None
    try:
        environment = DofbotImitationEnvironment(
            stage=stage,
            articulation_root_path=articulation_root_path,
            gripper_point_prim_path=gripper_point_prim_path,
            timeline=timeline,
            data_root=ARGS.data_root,
            cube_x=ARGS.cube_x,
            cube_y_bounds=(ARGS.cube_y_min, ARGS.cube_y_max),
            cube_clearance=ARGS.cube_clearance,
            master_seed=ARGS.master_seed,
            camera_sensor=camera_sensor,
        )

        print("\nStarting simulation automatically...")
        print(f"Waiting for JointState position commands on {JOINT_COMMAND_TOPIC}.")
        print(f"Send task text on {TASK_INSTRUCTION_TOPIC}.")
        if ARGS.publish_camera:
            print(f"Publishing RGB images on {CAMERA_RGB_TOPIC}.")
        print("Call /dofbot/reset_episode before starting each episode.")

        while simulation_app.is_running():
            environment.spin_ros_once()
            simulation_app.update()
            environment.tick()
    finally:
        if environment is not None:
            environment.shutdown()
        timeline.stop()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        print("\nFATAL: DOFBOT environment terminated because of an exception.", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close(exit_code=1)
        raise
    else:
        simulation_app.close()
