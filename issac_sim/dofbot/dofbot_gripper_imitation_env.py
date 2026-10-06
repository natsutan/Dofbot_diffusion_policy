from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# =============================================================================
# Command-line settings
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DOFBOT imitation-learning environment server for Isaac Sim",
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
    parser.add_argument("--x-min", type=float, default=0.17)
    parser.add_argument("--x-max", type=float, default=0.22)
    parser.add_argument("--y-min", type=float, default=-0.08)
    parser.add_argument("--y-max", type=float, default=-0.02)
    parser.add_argument(
        "--home-jitter-deg",
        type=float,
        default=2.0,
        help=(
            "Uniform per-joint randomization applied to arm1..arm5 around "
            "HOME_JOINT_POSITION at every episode reset, in degrees. "
            "Default: +/-2 deg."
        ),
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=20260727,
        help="Seed used when a reset request specifies seed=-1.",
    )
    args, _unknown = parser.parse_known_args()
    return args


ARGS = parse_arguments()

# =============================================================================
# ROS 2 / Isaac Sim startup settings
# =============================================================================
# These values must be set before the ROS 2 bridge extension is loaded.
# os.environ.setdefault("ROS_DOMAIN_ID", "0")
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

    import rclpy
    from builtin_interfaces.msg import Time
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState

    from dofbot_interfaces.msg import EpisodeState
    from dofbot_interfaces.srv import EndEpisode, ResetEpisode
except Exception:
    simulation_app.close()
    raise


# =============================================================================
# User and environment settings
# =============================================================================

URDF_PATH = Path("/home/natu/dofbot/urdf/dofbot.urdf")
USD_OUTPUT_ROOT = Path("/home/natu/dofbot/generated")
FORCE_URDF_REIMPORT = True

ROBOT_REFERENCE_PATH = "/World/DOFBOT"
ROBOT_REFERENCE_POSITION = (0.0, 0.0, 0.0)
GRIPPER_POINT_LINK_NAME = "Gripping_point_Link"

JOINT_COMMAND_TOPIC = "/joint_command"
JOINT_STATE_TOPIC = "/joint_states"
EPISODE_STATE_TOPIC = "/dofbot/episode_state"
RESET_SERVICE = "/dofbot/reset_episode"
END_SERVICE = "/dofbot/end_episode"

ROS_DOMAIN_ID = int(os.environ.get("ROS_DOMAIN_ID", "0"))
GRAPH_PATH = "/ActionGraph"

STAGE_UNITS_IN_METERS = 1.0
STAGE_UP_AXIS = "Z"

FLOOR_CENTER_Z = -0.005
FLOOR_HALF_THICKNESS = 0.005
FLOOR_TOP_Z = FLOOR_CENTER_Z + FLOOR_HALF_THICKNESS

PICK_CUBE_PATH = "/World/PickCube"
PICK_CUBE_SIZE = 0.030
PICK_CUBE_INITIAL_POSITION_XY = (0.20, -0.05)
PICK_CUBE_MASS = 0.0027
PICK_CUBE_STATIC_FRICTION = 1.0
PICK_CUBE_DYNAMIC_FRICTION = 0.8
PICK_CUBE_RESTITUTION = 0.0

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
ARM_HOME_JITTER_RAD = math.radians(float(ARGS.home_jitter_deg))

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


def enable_extension(extension_name: str) -> None:
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate(extension_name, True)
    if not ext_manager.is_extension_enabled(extension_name):
        raise RuntimeError(f"Failed to enable extension: {extension_name}")
    print(f"{extension_name} = enabled")


def enable_required_extensions() -> None:
    for extension_name in (
        "isaacsim.asset.importer.urdf",
        "isaacsim.core.nodes",
        "isaacsim.core.prims",
        "isaacsim.ros2.bridge",
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


def add_pick_cube(stage) -> None:
    cube_center_z = FLOOR_TOP_Z + PICK_CUBE_SIZE / 2.0
    cube = UsdGeom.Cube.Define(stage, PICK_CUBE_PATH)
    cube.CreateSizeAttr(PICK_CUBE_SIZE)
    cube.CreateExtentAttr(
        [
            Gf.Vec3f(-PICK_CUBE_SIZE / 2.0),
            Gf.Vec3f(PICK_CUBE_SIZE / 2.0),
        ]
    )
    cube.AddTranslateOp().Set(
        Gf.Vec3d(
            PICK_CUBE_INITIAL_POSITION_XY[0],
            PICK_CUBE_INITIAL_POSITION_XY[1],
            cube_center_z,
        )
    )
    cube.CreateDisplayColorAttr().Set([Gf.Vec3f(0.85, 0.25, 0.10)])

    cube_prim = cube.GetPrim()
    collision = UsdPhysics.CollisionAPI.Apply(cube_prim)
    collision.CreateCollisionEnabledAttr(True)

    rigid_body = UsdPhysics.RigidBodyAPI.Apply(cube_prim)
    rigid_body.CreateRigidBodyEnabledAttr(True)
    rigid_body.CreateKinematicEnabledAttr(False)
    rigid_body.CreateStartsAsleepAttr(False)

    mass = UsdPhysics.MassAPI.Apply(cube_prim)
    mass.CreateMassAttr(PICK_CUBE_MASS)

    physics_material_path = "/World/Materials/PickCubePhysicsMaterial"
    physics_material = UsdShade.Material.Define(stage, physics_material_path)
    material_api = UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
    material_api.CreateStaticFrictionAttr(PICK_CUBE_STATIC_FRICTION)
    material_api.CreateDynamicFrictionAttr(PICK_CUBE_DYNAMIC_FRICTION)
    material_api.CreateRestitutionAttr(PICK_CUBE_RESTITUTION)
    UsdShade.MaterialBindingAPI.Apply(cube_prim).Bind(
        physics_material,
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )

    print("\n========== Pick cube created ==========")
    print(f"prim_path = {PICK_CUBE_PATH}")
    print(
        "position  = "
        f"({PICK_CUBE_INITIAL_POSITION_XY[0]:.3f}, "
        f"{PICK_CUBE_INITIAL_POSITION_XY[1]:.3f}, {cube_center_z:.3f}) m"
    )
    print(f"size      = {PICK_CUBE_SIZE:.3f} m")
    print(f"mass      = {PICK_CUBE_MASS:.4f} kg")


def add_light(stage) -> None:
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(3000.0)
    dome.CreateExposureAttr(0.0)


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


class EpisodeRecorder:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root.expanduser().resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.active = False
        self.record_enabled = False
        self.episode_id = -1
        self.episode_dir: Path | None = None
        self.metadata: dict = {}
        self.timestamps: list[float] = []
        self.joint_positions: list[np.ndarray] = []
        self.cube_positions: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.gripper_positions: list[np.ndarray] = []
        self.phases: list[int] = []

    def start(
        self,
        *,
        episode_id: int,
        seed: int,
        cube_position: np.ndarray,
        sim_time: float,
        record_enabled: bool,
        randomization_bounds: dict,
        reset_joint_position: np.ndarray,
    ) -> None:
        self.active = True
        self.record_enabled = bool(record_enabled)
        self.episode_id = int(episode_id)
        self.episode_dir = self.data_root / f"episode_{episode_id:06d}"
        self.timestamps.clear()
        self.joint_positions.clear()
        self.cube_positions.clear()
        self.actions.clear()
        self.gripper_positions.clear()
        self.phases.clear()
        self.metadata = {
            "episode_index": int(episode_id),
            "seed": int(seed),
            "record_enabled": self.record_enabled,
            "status": "in_progress",
            "start_simulation_time": float(sim_time),
            "initial_cube_position": cube_position.astype(float).tolist(),
            "reset_joint_position_command": reset_joint_position.astype(float).tolist(),
            "joint_names": list(DRIVEN_JOINT_NAMES),
            "sample_rate_hz": RECORD_RATE_HZ,
            "randomization_bounds": randomization_bounds,
            "success_definition": {
                "minimum_lift_m": SUCCESS_LIFT_HEIGHT_M,
                "maximum_gripper_distance_m": SUCCESS_GRIPPER_DISTANCE_M,
                "hold_time_sec": SUCCESS_HOLD_SEC,
            },
        }

        if self.record_enabled:
            self.episode_dir.mkdir(parents=True, exist_ok=False)
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
    ) -> None:
        if not self.active or not self.record_enabled:
            return
        self.timestamps.append(float(timestamp))
        self.joint_positions.append(joint_position.astype(np.float64, copy=True))
        self.cube_positions.append(cube_position.astype(np.float64, copy=True))
        self.actions.append(action.astype(np.float64, copy=True))
        self.gripper_positions.append(gripper_position.astype(np.float64, copy=True))
        self.phases.append(int(phase))

    def finalize(
        self,
        *,
        success: bool,
        truncated: bool,
        reason: str,
        sim_time: float,
        final_cube_position: np.ndarray,
        maximum_cube_height: float,
    ) -> EpisodeResult:
        if not self.active:
            raise RuntimeError("No episode is currently active")

        dataset_path = ""
        recorded_steps = len(self.timestamps)
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
                    recorded_steps, 3
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
            dataset_path = str(self.episode_dir)

        self.metadata.update(
            {
                "status": "finished",
                "success": bool(success),
                "terminated": True,
                "truncated": bool(truncated),
                "termination_reason": str(reason),
                "end_simulation_time": float(sim_time),
                "recorded_steps": int(recorded_steps),
                "final_cube_position": final_cube_position.astype(float).tolist(),
                "maximum_cube_height": float(maximum_cube_height),
            }
        )
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
        return result

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
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        master_seed: int,
    ) -> None:
        self.stage = stage
        self.timeline = timeline
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.master_rng = np.random.default_rng(master_seed)
        self.recorder = EpisodeRecorder(data_root)
        self.next_episode_id = next_episode_index(self.recorder.data_root)

        self.robot = Articulation(
            prim_paths_expr=articulation_root_path,
            name="dofbot_imitation_articulation",
            reset_xform_properties=False,
        )
        self.cube = RigidPrim(
            prim_paths_expr=PICK_CUBE_PATH,
            name="dofbot_pick_cube",
            reset_xform_properties=False,
        )
        self.robot.initialize()
        self.cube.initialize()

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
        self.initial_cube_position = self.get_cube_position()
        self.maximum_cube_height = float(self.initial_cube_position[2])
        self.success = False
        self.terminated = False
        self.truncated = False
        self.termination_reason = ""
        self.success_condition_since: float | None = None
        self.ready_after_sim_time = 0.0
        self.episode_start_sim_time = 0.0
        self.last_record_sim_time = -math.inf
        self.last_result: EpisodeResult | None = None

        self.node = Node("dofbot_imitation_environment")
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
        self.command_subscription = self.node.create_subscription(
            JointState,
            JOINT_COMMAND_TOPIC,
            self.on_joint_command,
            10,
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

        print("\n========== Imitation environment API ==========")
        print(f"reset service   = {RESET_SERVICE}")
        print(f"end service     = {END_SERVICE}")
        print(f"state topic     = {EPISODE_STATE_TOPIC}")
        print(f"joint state     = {JOINT_STATE_TOPIC}")
        print(f"data root       = {self.recorder.data_root}")
        print(f"random x range  = [{x_bounds[0]:.3f}, {x_bounds[1]:.3f}] m")
        print(f"random y range  = [{y_bounds[0]:.3f}, {y_bounds[1]:.3f}] m")

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

            if mode == MODE_RANDOM:
                x = float(rng.uniform(*self.x_bounds))
                y = float(rng.uniform(*self.y_bounds))
            else:
                x = float(request.x)
                y = float(request.y)
                self.validate_specified_position(x, y)

            cube_position = self.reset_environment(
                x=x,
                y=y,
                seed=seed,
                rng=rng,
                record=bool(request.record),
            )
            response.accepted = True
            response.episode_id = int(self.episode_id)
            response.seed_used = int(seed)
            response.cube_x = float(cube_position[0])
            response.cube_y = float(cube_position[1])
            response.cube_z = float(cube_position[2])
            response.message = (
                "Reset accepted. Wait until /dofbot/episode_state phase becomes READY."
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

    def validate_specified_position(self, x: float, y: float) -> None:
        if not (self.x_bounds[0] <= x <= self.x_bounds[1]):
            raise ValueError(
                f"x={x:.4f} is outside [{self.x_bounds[0]}, {self.x_bounds[1]}]"
            )
        if not (self.y_bounds[0] <= y <= self.y_bounds[1]):
            raise ValueError(
                f"y={y:.4f} is outside [{self.y_bounds[0]}, {self.y_bounds[1]}]"
            )

    def reset_environment(
        self,
        *,
        x: float,
        y: float,
        seed: int,
        rng: np.random.Generator,
        record: bool,
    ) -> np.ndarray:
        was_playing = self.timeline.is_playing()
        self.timeline.pause()

        # Teleport all controlled joints and clear velocities. This is only used
        # at episode boundaries; ordinary motion still uses /joint_command.
        # Each arm joint is independently randomized around the nominal HOME pose
        # to model small real-robot reset/sensor variations. Gripper joints remain
        # open and are not randomized.
        randomized_home = HOME_JOINT_POSITION.copy()
        randomized_home[: len(ARM_JOINT_NAMES)] += rng.uniform(
            -ARM_HOME_JITTER_RAD,
            ARM_HOME_JITTER_RAD,
            size=len(ARM_JOINT_NAMES),
        )
        home = randomized_home.reshape(1, -1)
        zeros = np.zeros_like(home)
        self.robot.set_joint_positions(home, joint_indices=self.joint_indices)
        self.robot.set_joint_velocities(zeros, joint_indices=self.joint_indices)
        self.robot.set_joint_position_targets(home, joint_indices=self.joint_indices)

        cube_z = FLOOR_TOP_Z + PICK_CUBE_SIZE / 2.0
        cube_position = np.asarray([[x, y, cube_z]], dtype=np.float64)
        cube_orientation = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        self.cube.set_world_poses(
            positions=cube_position,
            orientations=cube_orientation,
        )
        self.cube.set_velocities(np.zeros((1, 6), dtype=np.float64))

        self.last_action = randomized_home.copy()
        if was_playing or not self.timeline.is_playing():
            self.timeline.play()

        sim_time = float(self.timeline.get_current_time())
        actual_cube_position = self.get_cube_position()
        self.episode_id = self.next_episode_id
        self.next_episode_id += 1
        self.seed_used = int(seed)
        self.initial_cube_position = actual_cube_position.copy()
        self.maximum_cube_height = float(actual_cube_position[2])
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

        self.recorder.start(
            episode_id=self.episode_id,
            seed=seed,
            cube_position=actual_cube_position,
            sim_time=sim_time,
            record_enabled=record,
            randomization_bounds={
                "x_min": self.x_bounds[0],
                "x_max": self.x_bounds[1],
                "y_min": self.y_bounds[0],
                "y_max": self.y_bounds[1],
                "home_jitter_deg": float(ARGS.home_jitter_deg),
            },
            reset_joint_position=randomized_home,
        )

        self.node.get_logger().info(
            f"Episode {self.episode_id} reset: "
            f"cube=({x:.4f}, {y:.4f}, {cube_z:.4f}), seed={seed}, record={record}, "
            f"arm_home={np.round(randomized_home[:len(ARM_JOINT_NAMES)], 5).tolist()}"
        )
        return actual_cube_position

    def tick(self) -> None:
        sim_time = float(self.timeline.get_current_time())

        if self.recorder.active and self.phase == PHASE_RESETTING:
            if sim_time >= self.ready_after_sim_time:
                self.phase = PHASE_READY
                self.episode_start_sim_time = sim_time
                self.last_record_sim_time = -math.inf
                # Re-read the settled cube pose as the episode's reference.
                self.initial_cube_position = self.get_cube_position()
                self.maximum_cube_height = float(self.initial_cube_position[2])
                self.recorder.metadata["settled_initial_cube_position"] = (
                    self.initial_cube_position.astype(float).tolist()
                )
                self.recorder.metadata["settled_initial_joint_position"] = (
                    self.get_joint_position().astype(float).tolist()
                )
                if self.recorder.record_enabled:
                    self.recorder._write_metadata()
                self.node.get_logger().info(
                    f"Episode {self.episode_id} is READY at "
                    f"({self.initial_cube_position[0]:.4f}, "
                    f"{self.initial_cube_position[1]:.4f})"
                )

        actual_joint_position = self.get_joint_position()
        cube_position = self.get_cube_position()
        gripper_position = self.get_gripper_position()
        self.maximum_cube_height = max(
            self.maximum_cube_height,
            float(cube_position[2]),
        )

        if self.recorder.active and self.phase in (PHASE_READY, PHASE_RUNNING):
            self.update_success_and_termination(
                sim_time=sim_time,
                cube_position=cube_position,
                gripper_position=gripper_position,
            )

        if (
            self.recorder.active
            and self.phase != PHASE_RESETTING
            and sim_time - self.last_record_sim_time >= RECORD_PERIOD_SEC - 1e-9
        ):
            self.recorder.append(
                timestamp=sim_time,
                joint_position=actual_joint_position,
                cube_position=cube_position,
                action=self.last_action,
                gripper_position=gripper_position,
                phase=self.phase,
            )
            self.last_record_sim_time = sim_time

        self.publish_joint_state(sim_time, actual_joint_position)
        self.publish_episode_state(sim_time, cube_position, gripper_position)

    def update_success_and_termination(
        self,
        *,
        sim_time: float,
        cube_position: np.ndarray,
        gripper_position: np.ndarray,
    ) -> None:
        lift_height = float(cube_position[2] - self.initial_cube_position[2])
        gripper_distance = float(np.linalg.norm(cube_position - gripper_position))
        success_condition = (
            lift_height >= SUCCESS_LIFT_HEIGHT_M
            and gripper_distance <= SUCCESS_GRIPPER_DISTANCE_M
        )

        if success_condition:
            if self.success_condition_since is None:
                self.success_condition_since = sim_time
            elif sim_time - self.success_condition_since >= SUCCESS_HOLD_SEC:
                self.success = True
                self.phase = PHASE_SUCCESS
                self.finalize_episode(
                    success=True,
                    truncated=False,
                    reason="cube_lifted_near_gripper_and_held",
                )
                return
        else:
            self.success_condition_since = None

        if float(cube_position[2]) < CUBE_FALL_Z_M:
            self.finalize_episode(
                success=False,
                truncated=False,
                reason="cube_fell_below_floor",
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
        cube_position = self.get_cube_position()
        final_phase = PHASE_SUCCESS if success else PHASE_TERMINATED
        # Always retain the terminal state, even when success is detected between
        # two regular 10 Hz samples.
        self.recorder.append(
            timestamp=sim_time,
            joint_position=self.get_joint_position(),
            cube_position=cube_position,
            action=self.last_action,
            gripper_position=self.get_gripper_position(),
            phase=final_phase,
        )
        result = self.recorder.finalize(
            success=success,
            truncated=truncated,
            reason=reason,
            sim_time=sim_time,
            final_cube_position=cube_position,
            maximum_cube_height=self.maximum_cube_height,
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

    def get_joint_position(self) -> np.ndarray:
        value = self.robot.get_joint_positions(joint_indices=self.joint_indices)
        array = as_numpy(value).reshape(-1)
        if array.size != len(DRIVEN_JOINT_NAMES):
            raise RuntimeError(
                f"Expected 11 joint positions, received shape {as_numpy(value).shape}"
            )
        return array.astype(np.float64, copy=False)

    def get_cube_position(self) -> np.ndarray:
        # usd=False reads the live Fabric/physics transform rather than an
        # authored USD transform that may lag behind simulation.
        positions, _orientations = self.cube.get_world_poses(usd=False)
        array = as_numpy(positions).reshape(-1, 3)
        return array[0].astype(np.float64, copy=True)

    def get_gripper_position(self) -> np.ndarray:
        try:
            positions, _orientations = self.gripper_point.get_world_poses(usd=False)
            array = as_numpy(positions).reshape(-1, 3)
            return array[0].astype(np.float64, copy=True)
        except Exception:
            # Fallback for configurations where Fabric access is unavailable.
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
        cube_position: np.ndarray,
        gripper_position: np.ndarray,
    ) -> None:
        message = EpisodeState()
        message.header.stamp = simulation_time_message(sim_time)
        message.episode_id = int(self.episode_id)
        message.phase = int(self.phase)
        message.seed = int(self.seed_used)
        message.cube_x = float(cube_position[0])
        message.cube_y = float(cube_position[1])
        message.cube_z = float(cube_position[2])
        message.lift_height = float(
            cube_position[2] - self.initial_cube_position[2]
        )
        message.gripper_distance = float(
            np.linalg.norm(cube_position - gripper_position)
        )
        message.success = bool(self.success)
        message.terminated = bool(self.terminated)
        message.truncated = bool(self.truncated)
        message.termination_reason = self.termination_reason
        self.episode_state_publisher.publish(message)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    if ARGS.x_min > ARGS.x_max:
        raise ValueError("--x-min must be smaller than --x-max")
    if ARGS.y_min > ARGS.y_max:
        raise ValueError("--y-min must be smaller than --y-max")
    if ARGS.home_jitter_deg < 0.0:
        raise ValueError("--home-jitter-deg must be non-negative")

    enable_required_extensions()
    update_app(20)

    stage = create_new_stage()
    add_physics_scene(stage)
    add_floor(stage)
    add_pick_cube(stage)
    add_light(stage)

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

    if not rclpy.ok():
        # Empty args prevent Isaac Sim command-line arguments from being parsed by rclpy.
        rclpy.init(args=[])

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    update_app(10)

    environment: DofbotImitationEnvironment | None = None
    try:
        environment = DofbotImitationEnvironment(
            stage=stage,
            articulation_root_path=articulation_root_path,
            gripper_point_prim_path=gripper_point_prim_path,
            timeline=timeline,
            data_root=ARGS.data_root,
            x_bounds=(ARGS.x_min, ARGS.x_max),
            y_bounds=(ARGS.y_min, ARGS.y_max),
            master_seed=ARGS.master_seed,
        )

        print("\nStarting simulation automatically...")
        print(f"Waiting for JointState position commands on {JOINT_COMMAND_TOPIC}.")
        print("Call /dofbot/reset_episode before starting each pick.")

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
    finally:
        simulation_app.close()
